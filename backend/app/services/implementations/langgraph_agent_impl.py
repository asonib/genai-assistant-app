import logging
import time

from app.config.app_config import AIConfigConstants
from app.exceptions.custom_exceptions import AgentExecutionException
from app.prompts.templates import AGENT_TURN_TEMPLATE
from app.services.interfaces.agent_interface import AgentInput, AgentOutput, IAgentRunner
from app.services.interfaces.external_tool_client_interface import IExternalToolClient
from app.services.interfaces.llm_service_interface import ILLMService
from app.services.interfaces.retrieval_service_interface import IRetrievalService
from rest_framework.exceptions import APIException

logger = logging.getLogger(__name__)


class LangGraphAgentRunner(IAgentRunner):
    """
    IAgentRunner backed by the LangGraph StateGraph in app/agents/graph.py.
    The graph is compiled once per runner; tools are discovered from the MCP
    server on every run (the client caches `tools/list`).
    """

    def __init__(self, llm_service: ILLMService, retrieval_service: IRetrievalService,
                 tool_client: IExternalToolClient):
        from app.agents.graph import build_agent_graph  # imports langgraph lazily

        self.llm_service = llm_service
        self.retrieval_service = retrieval_service
        self.tool_client = tool_client
        self._graph = build_agent_graph(llm_service, retrieval_service, tool_client)

    @property
    def name(self) -> str:
        return "langgraph:condense->retrieve->agent<->tools->finalize"

    def run(self, agent_input: AgentInput) -> AgentOutput:
        from app.agents.graph import tool_specs_for

        started = time.perf_counter()
        mcp_tools = self.tool_client.list_tools()
        initial_state = {
            "user_id": agent_input.user_id,
            "question": agent_input.question,
            "history": list(agent_input.history),
            "document_ids": list(agent_input.document_ids),
            "tool_specs": tool_specs_for(mcp_tools),
            "mcp_tool_names": [t.name for t in mcp_tools],
            "retrieved": [],
            "llm_messages": [],
            "tool_calls": [],
            "trace": [],
            "iterations": 0,
            "input_tokens": 0,
            "output_tokens": 0,
        }
        try:
            final = self._graph.invoke(initial_state,
                                       config={"recursion_limit": AIConfigConstants.AGENT_RECURSION_LIMIT})
        except APIException:
            raise  # LLM / embedding / vector-store errors keep their HTTP mapping
        except Exception as exc:
            logger.exception("Agent graph failed")
            raise AgentExecutionException(f"Agent failed: {type(exc).__name__}: {exc}")

        logger.info("agent_run user=%s iterations=%s tools=%s citations=%s ms=%s",
                    agent_input.user_id, final.get("iterations"),
                    [t["name"] for t in final.get("tool_calls", [])], len(final.get("citations", [])),
                    int((time.perf_counter() - started) * 1000))
        return AgentOutput(
            answer=final["answer"],
            citations=final.get("citations", []),
            tool_calls=final.get("tool_calls", []),
            trace=final.get("trace", []),
            model_name=final.get("model") or self.llm_service.model_name,
            prompt_template=AGENT_TURN_TEMPLATE.name,
            standalone_question=final.get("standalone_question", agent_input.question),
            input_tokens=final.get("input_tokens") or None,
            output_tokens=final.get("output_tokens") or None,
        )
