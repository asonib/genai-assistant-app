"""Deterministic, offline test doubles for the AI infrastructure interfaces."""
import hashlib
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from app.exceptions.custom_exceptions import ExternalToolException
from app.services.interfaces.embedding_service_interface import IEmbeddingService
from app.services.interfaces.external_tool_client_interface import IExternalToolClient, ToolResult, ToolSpec
from app.services.interfaces.llm_service_interface import ILLMService, LLMResult, ToolCall
from app.services.interfaces.vector_store_interface import IVectorStore

DIM = 384


class FakeEmbeddingService(IEmbeddingService):
    """Hashed bag-of-words embeddings: texts sharing words get high cosine similarity."""

    @property
    def model_name(self) -> str:
        return "fake-hash-embedding"

    @property
    def dimension(self) -> int:
        return DIM

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(DIM, dtype=np.float32)
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            if len(tok) < 3:
                continue
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % DIM] += 1.0
        norm = np.linalg.norm(v)
        return v / norm if norm else v

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        return np.stack([self._vec(t) for t in texts]) if texts else np.zeros((0, DIM), np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self._vec(text)


class InMemoryVectorStore(IVectorStore):
    def __init__(self):
        self.data: Dict[str, Dict[int, np.ndarray]] = {}

    @property
    def backend_name(self) -> str:
        return "faiss"

    def add(self, namespace: str, ids: Sequence[int], vectors: np.ndarray) -> None:
        bucket = self.data.setdefault(namespace, {})
        for i, v in zip(ids, vectors):
            bucket[int(i)] = np.asarray(v, dtype=np.float32)

    def remove(self, namespace: str, ids: Sequence[int]) -> int:
        bucket = self.data.get(namespace, {})
        return sum(1 for i in ids if bucket.pop(int(i), None) is not None)

    def search(self, namespace: str, query_vector: np.ndarray, top_k: int) -> List[Tuple[int, float]]:
        bucket = self.data.get(namespace, {})
        scored = sorted(((i, float(np.dot(v, query_vector))) for i, v in bucket.items()),
                        key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def count(self, namespace: str) -> int:
        return len(self.data.get(namespace, {}))


def text_reply(text: str) -> LLMResult:
    return LLMResult(text=text, model="fake-claude", input_tokens=10, output_tokens=5, stop_reason="end_turn",
                     content_blocks=[{"type": "text", "text": text}])


def tool_reply(*calls: Tuple[str, Dict[str, Any]], text: str = "") -> LLMResult:
    tool_calls = [ToolCall(id=f"toolu_{i}_{name}", name=name, input=args) for i, (name, args) in enumerate(calls)]
    blocks = ([{"type": "text", "text": text}] if text else []) + [
        {"type": "tool_use", "id": c.id, "name": c.name, "input": c.input} for c in tool_calls]
    return LLMResult(text=text, model="fake-claude", input_tokens=20, output_tokens=8, stop_reason="tool_use",
                     tool_calls=tool_calls, content_blocks=blocks)


Responder = Callable[[str, List[Dict[str, Any]], Optional[List[Dict[str, Any]]]], LLMResult]


class ScriptedLLMService(ILLMService):
    """
    Returns scripted replies in order. Calls made *without* tools (the
    condense_question node) are answered by `condense` so scripts only
    describe the agent loop.
    """

    def __init__(self, script: Optional[List[Union[LLMResult, Responder]]] = None,
                 condense: Optional[Callable[[str], str]] = None):
        self.script = list(script or [])
        self.condense = condense or (lambda question: question)
        self.calls: List[Dict[str, Any]] = []

    @property
    def model_name(self) -> str:
        return "fake-claude"

    def queue(self, *replies):
        self.script.extend(replies)

    def generate(self, system, messages, tools=None) -> LLMResult:
        import copy
        self.calls.append({"system": system, "messages": copy.deepcopy(messages), "tools": tools})
        if not tools:
            follow_up = re.search(r"<follow_up>\n(.*?)\n</follow_up>", messages[-1]["content"], re.S)
            return text_reply(self.condense(follow_up.group(1) if follow_up else ""))
        if not self.script:
            return text_reply("default fake answer")
        item = self.script.pop(0)
        return item(system, messages, tools) if callable(item) else item

    @property
    def agent_calls(self):
        return [c for c in self.calls if c["tools"]]


WEATHER_TOOL = ToolSpec(
    name="get_weather",
    description="Current weather and forecast for a city.",
    input_schema={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
)
FX_TOOL = ToolSpec(
    name="convert_currency",
    description="Convert an amount between currencies.",
    input_schema={"type": "object", "properties": {"amount": {"type": "number"},
                                                   "from_currency": {"type": "string"},
                                                   "to_currency": {"type": "string"}}},
)


class FakeToolClient(IExternalToolClient):
    def __init__(self, tools=None, responses: Optional[Dict[str, Any]] = None, fail: bool = False):
        self.tools = list(tools if tools is not None else [WEATHER_TOOL, FX_TOOL])
        self.responses = responses or {
            "get_weather": '{"location": {"name": "Pune"}, "current": {"temperature": 29.5, "conditions": "Clear sky"}}',
            "convert_currency": '{"amount": 100, "from": "USD", "to": "INR", "rate": 83.2, "converted": 8320.0}',
        }
        self.fail = fail
        self.calls: List[Tuple[str, Dict[str, Any]]] = []

    @property
    def transport(self) -> str:
        return "fake"

    def list_tools(self):
        return list(self.tools)

    def call_tool(self, name, arguments):
        self.calls.append((name, dict(arguments)))
        if self.fail:
            raise ExternalToolException("MCP server crashed")
        if name not in self.responses:
            return ToolResult(content=f"unknown tool {name}", is_error=True)
        return ToolResult(content=self.responses[name])
