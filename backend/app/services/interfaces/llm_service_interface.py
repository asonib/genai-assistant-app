from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ToolCall:
    """A tool invocation requested by the model (Anthropic `tool_use` block)."""
    id: str
    name: str
    input: Dict[str, Any]


@dataclass(frozen=True)
class LLMResult:
    text: str
    model: str
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    stop_reason: Optional[str] = None
    tool_calls: List[ToolCall] = field(default_factory=list)
    # Provider-native content blocks of the assistant turn, as plain dicts.
    # They are echoed back verbatim on the next call of a tool-use loop.
    content_blocks: List[Dict[str, Any]] = field(default_factory=list)


class ILLMService(ABC):
    """Contract for a chat-completion LLM provider with tool calling."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        pass

    @abstractmethod
    def generate(self, system: str, messages: List[Dict[str, Any]],
                 tools: Optional[List[Dict[str, Any]]] = None) -> LLMResult:
        """
        Run one completion.
        :param system: system prompt (role + rules)
        :param messages: [{"role": "user"|"assistant", "content": str | [blocks]}]
        :param tools: [{"name", "description", "input_schema"}] the model may call
        """
        pass
