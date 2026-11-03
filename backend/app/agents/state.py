from typing import Any, Dict, List, TypedDict


class AgentState(TypedDict, total=False):
    """
    LangGraph state for one assistant turn.

    Nodes return partial updates; no reducers are used, so every key a node
    returns replaces the previous value (nodes always return fresh lists).
    NOTE: LangGraph forbids node names that collide with state keys.
    """
    # ---- input ----
    user_id: int
    question: str
    history: List[Dict[str, str]]          # [{"role", "content"}], oldest first
    document_ids: List[int]                # user-selected scope ([] = all docs)
    tool_specs: List[Dict[str, Any]]       # Anthropic tool definitions offered to the model
    mcp_tool_names: List[str]

    # ---- working memory ----
    standalone_question: str
    retrieved: List[Dict[str, Any]]        # RetrievedChunk dicts, deduped by chunk_id
    llm_messages: List[Dict[str, Any]]     # Anthropic-format transcript of this turn
    pending_tool_calls: List[Dict[str, Any]]
    iterations: int

    # ---- output ----
    answer: str
    citations: List[Dict[str, Any]]
    tool_calls: List[Dict[str, Any]]
    trace: List[Dict[str, Any]]
    input_tokens: int
    output_tokens: int
    model: str
