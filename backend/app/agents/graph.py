"""
LangGraph definition of the GenAI Assistant agent.

    START
      |
      v
  condense_question   (LLM: rewrite follow-ups into a standalone query using memory)
      |
      v
   retrieve           (vector search over all of the user's documents)
      |
      v
    agent  <------+   (Claude with tools: search_documents + MCP weather/FX tools)
      |           |
  tool calls?     |
   yes |  no      |
      v   \       |
    tools  \------+-- (execute MCP / built-in tools, feed results back)
            \
             v
          finalize    (citations = retrieved chunks, flagged when cited)
             |
            END
"""
import logging
import re
import time
from datetime import date
from typing import Any, Callable, Dict, List

from langgraph.graph import END, START, StateGraph

from app.agents.state import AgentState
from app.config.app_config import AIConfigConstants
from app.exceptions.custom_exceptions import ExternalToolException
from app.prompts.templates import (
    AGENT_TURN_TEMPLATE,
    CONDENSE_QUESTION_TEMPLATE,
    CONTEXT_CHUNK_FORMAT,
    FALLBACK_ANSWER,
    HISTORY_USER_TURN_TEMPLATE,
    NO_CONTEXT_PLACEHOLDER,
    SEARCH_TOOL_DESCRIPTION,
    TOOL_BUDGET_EXHAUSTED_NOTE,
    escape_delimiters,
)
from app.services.interfaces.external_tool_client_interface import IExternalToolClient
from app.services.interfaces.llm_service_interface import ILLMService
from app.services.interfaces.retrieval_service_interface import IRetrievalService

logger = logging.getLogger(__name__)

CITATION_RE = re.compile(r"\[(\d+):(\d+)\]")

SEARCH_TOOL_SPEC = {
    "name": AIConfigConstants.SEARCH_TOOL_NAME,
    "description": SEARCH_TOOL_DESCRIPTION,
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for."},
            "top_k": {"type": "integer", "minimum": 1, "maximum": 8,
                      "description": "Number of excerpts to return (default 4)."},
        },
        "required": ["query"],
    },
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def format_chunks(chunks: List[Dict[str, Any]]) -> str:
    if not chunks:
        return NO_CONTEXT_PLACEHOLDER
    return "\n\n".join(
        CONTEXT_CHUNK_FORMAT.format(
            document_id=c["document_id"], chunk_index=c["chunk_index"],
            title=escape_delimiters(c["document_title"]).replace('"', "'"),
            score=c["score"], chunk_text=escape_delimiters(c["chunk_text"]),
        )
        for c in chunks
    )


def merge_chunks(existing: List[Dict[str, Any]], new: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_id = {c["chunk_id"]: dict(c) for c in existing}
    for c in new:
        prev = by_id.get(c["chunk_id"])
        if prev is None or c["score"] > prev["score"]:
            by_id[c["chunk_id"]] = dict(c)
    return list(by_id.values())


def history_to_messages(history: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """Replay memory as alternating user/assistant messages (user turns wrapped in a template)."""
    messages: List[Dict[str, Any]] = []
    for turn in history:
        role = turn.get("role")
        if role not in ("user", "assistant") or not turn.get("content"):
            continue
        content = (HISTORY_USER_TURN_TEMPLATE.format(content=turn["content"])
                   if role == "user" else turn["content"])
        if messages and messages[-1]["role"] == role:  # enforce alternation
            messages[-1]["content"] += "\n\n" + content
        else:
            messages.append({"role": role, "content": content})
    while messages and messages[0]["role"] != "user":
        messages.pop(0)
    if messages and messages[-1]["role"] == "user":
        # Unanswered trailing user turn: drop it, the new question follows.
        messages.pop()
    return messages


def _trace(state: AgentState, node: str, started: float, **details) -> List[Dict[str, Any]]:
    entry = {"node": node, "ms": int((time.perf_counter() - started) * 1000)}
    entry.update(details)
    return list(state.get("trace", [])) + [entry]


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + f"... [truncated {len(text) - limit} chars]"


# ---------------------------------------------------------------------------
# graph factory
# ---------------------------------------------------------------------------
def build_agent_graph(llm: ILLMService, retrieval: IRetrievalService, tool_client: IExternalToolClient,
                      today: Callable[[], str] = lambda: date.today().isoformat()):
    max_iterations = AIConfigConstants.AGENT_MAX_TOOL_ITERATIONS

    def _add_usage(state: AgentState, result) -> Dict[str, Any]:
        return {
            "input_tokens": state.get("input_tokens", 0) + (result.input_tokens or 0),
            "output_tokens": state.get("output_tokens", 0) + (result.output_tokens or 0),
            "model": result.model or llm.model_name,
        }

    # ---- node: condense_question ------------------------------------------
    def condense_question(state: AgentState) -> Dict[str, Any]:
        started = time.perf_counter()
        question = state["question"]
        history = state.get("history") or []
        if not history:
            return {"standalone_question": question,
                    "trace": _trace(state, "condense_question", started, skipped=True)}

        transcript = "\n".join(f"{t['role']}: {t['content'][:1500]}" for t in history[-6:])
        result = llm.generate(
            system=CONDENSE_QUESTION_TEMPLATE.system,
            messages=[{"role": "user", "content": CONDENSE_QUESTION_TEMPLATE.format(
                history=transcript, question=question)}],
        )
        rewritten = (result.text or "").strip().splitlines()[0].strip() if result.text else ""
        standalone = rewritten[:AIConfigConstants.MAX_MESSAGE_CHARS] or question
        update = {"standalone_question": standalone,
                  "trace": _trace(state, "condense_question", started, standalone_question=standalone)}
        update.update(_add_usage(state, result))
        return update

    # ---- node: retrieve -----------------------------------------------------
    def retrieve(state: AgentState) -> Dict[str, Any]:
        started = time.perf_counter()
        chunks = retrieval.search(
            user_id=state["user_id"], query=state["standalone_question"],
            top_k=AIConfigConstants.RAG_TOP_K, document_ids=state.get("document_ids") or None,
        )
        found = [c.to_dict() for c in chunks]
        return {"retrieved": merge_chunks(state.get("retrieved", []), found),
                "trace": _trace(state, "retrieve", started, query=state["standalone_question"],
                                hits=len(found))}

    # ---- node: agent --------------------------------------------------------
    def agent(state: AgentState) -> Dict[str, Any]:
        started = time.perf_counter()
        messages = list(state.get("llm_messages") or [])
        if not messages:
            scope = state.get("document_ids") or []
            turn = AGENT_TURN_TEMPLATE.format(
                today=today(),
                scope=(f"only documents {scope}" if scope else "all of the user's documents"),
                context=format_chunks(state.get("retrieved", [])),
                question=state["question"],
            )
            messages = history_to_messages(state.get("history") or []) + [{"role": "user", "content": turn}]

        result = llm.generate(system=AGENT_TURN_TEMPLATE.system, messages=messages,
                              tools=state.get("tool_specs") or None)
        iterations = state.get("iterations", 0) + 1
        messages.append({"role": "assistant",
                         "content": result.content_blocks or [{"type": "text", "text": result.text or ""}]})
        pending = [{"id": c.id, "name": c.name, "input": c.input} for c in result.tool_calls]
        update = {
            "llm_messages": messages,
            "pending_tool_calls": pending,
            "iterations": iterations,
            "answer": result.text or "",
            "trace": _trace(state, "agent", started, iteration=iterations,
                            requested_tools=[c["name"] for c in pending], stop_reason=result.stop_reason),
        }
        update.update(_add_usage(state, result))
        return update

    def route_after_agent(state: AgentState) -> str:
        if state.get("pending_tool_calls") and state.get("iterations", 0) <= max_iterations:
            return "tools"
        return "finalize"

    # ---- node: tools --------------------------------------------------------
    def _run_search_tool(state: AgentState, args: Dict[str, Any]):
        query = str(args.get("query") or "").strip()
        if not query:
            return "Error: 'query' is required.", True, []
        try:
            top_k = max(1, min(int(args.get("top_k") or AIConfigConstants.RAG_TOP_K), 8))
        except (TypeError, ValueError):
            top_k = AIConfigConstants.RAG_TOP_K
        chunks = [c.to_dict() for c in retrieval.search(
            user_id=state["user_id"], query=query, top_k=top_k,
            document_ids=state.get("document_ids") or None)]
        return format_chunks(chunks), False, chunks

    def tools(state: AgentState) -> Dict[str, Any]:
        started = time.perf_counter()
        mcp_names = set(state.get("mcp_tool_names") or [])
        retrieved = list(state.get("retrieved", []))
        tool_log = list(state.get("tool_calls", []))
        results_blocks = []

        for call in state.get("pending_tool_calls") or []:
            t0 = time.perf_counter()
            name, args = call["name"], call.get("input") or {}
            source = "builtin" if name == AIConfigConstants.SEARCH_TOOL_NAME else "mcp"
            try:
                if name == AIConfigConstants.SEARCH_TOOL_NAME:
                    output, is_error, chunks = _run_search_tool(state, args)
                    retrieved = merge_chunks(retrieved, chunks)
                elif name in mcp_names:
                    res = tool_client.call_tool(name, args)
                    output, is_error = res.content, res.is_error
                else:
                    output, is_error = f"Error: unknown tool '{name}'.", True
            except ExternalToolException as exc:
                output, is_error = f"Error: {exc.detail}", True
            except Exception as exc:  # never let one tool crash the whole turn
                logger.exception("Tool %s failed", name)
                output, is_error = f"Error: tool '{name}' failed ({type(exc).__name__}).", True

            output = _truncate(str(output), AIConfigConstants.MAX_TOOL_RESULT_CHARS)
            results_blocks.append({"type": "tool_result", "tool_use_id": call["id"],
                                   "content": output, "is_error": is_error})
            tool_log.append({"name": name, "source": source, "input": args,
                             "output": _truncate(output, 2000), "is_error": is_error,
                             "latency_ms": int((time.perf_counter() - t0) * 1000)})
            logger.info("tool=%s source=%s is_error=%s", name, source, is_error)

        if state.get("iterations", 0) >= max_iterations:
            results_blocks.append({"type": "text", "text": TOOL_BUDGET_EXHAUSTED_NOTE})

        messages = list(state.get("llm_messages") or []) + [{"role": "user", "content": results_blocks}]
        return {
            "llm_messages": messages,
            "pending_tool_calls": [],
            "retrieved": retrieved,
            "tool_calls": tool_log,
            "trace": _trace(state, "tools", started, executed=[c["name"] for c in state.get("pending_tool_calls") or []]),
        }

    # ---- node: finalize -----------------------------------------------------
    def finalize(state: AgentState) -> Dict[str, Any]:
        started = time.perf_counter()
        answer = (state.get("answer") or "").strip() or FALLBACK_ANSWER
        cited = {(int(d), int(i)) for d, i in CITATION_RE.findall(answer)}
        citations = []
        for c in state.get("retrieved", []):
            item = dict(c)
            item["cited"] = (c["document_id"], c["chunk_index"]) in cited
            citations.append(item)
        citations.sort(key=lambda c: (not c["cited"], -c["score"]))
        citations = citations[:AIConfigConstants.MAX_CITATIONS]
        return {"answer": answer, "citations": citations,
                "trace": _trace(state, "finalize", started, citations=len(citations),
                                cited=sum(1 for c in citations if c["cited"]))}

    graph = StateGraph(AgentState)
    graph.add_node("condense_question", condense_question)
    graph.add_node("retrieve", retrieve)
    graph.add_node("agent", agent)
    graph.add_node("tools", tools)
    graph.add_node("finalize", finalize)
    graph.add_edge(START, "condense_question")
    graph.add_edge("condense_question", "retrieve")
    graph.add_edge("retrieve", "agent")
    graph.add_conditional_edges("agent", route_after_agent, {"tools": "tools", "finalize": "finalize"})
    graph.add_edge("tools", "agent")
    graph.add_edge("finalize", END)
    return graph.compile()


def tool_specs_for(mcp_tools) -> List[Dict[str, Any]]:
    """Built-in search tool + every MCP tool, in Anthropic tool format."""
    return [SEARCH_TOOL_SPEC] + [t.to_llm_tool() for t in mcp_tools
                                 if t.name != AIConfigConstants.SEARCH_TOOL_NAME]


__all__ = ["build_agent_graph", "tool_specs_for", "SEARCH_TOOL_SPEC", "format_chunks",
           "history_to_messages", "merge_chunks"]
