from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class AgentInput:
    user_id: int
    question: str
    # Previous turns, oldest first: [{"role": "user"|"assistant", "content": str}]
    history: List[Dict[str, str]] = field(default_factory=list)
    document_ids: List[int] = field(default_factory=list)


@dataclass
class AgentOutput:
    answer: str
    citations: List[Dict[str, Any]]
    tool_calls: List[Dict[str, Any]]
    trace: List[Dict[str, Any]]
    model_name: str
    prompt_template: str
    standalone_question: str
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None


class IAgentRunner(ABC):
    """Runs one user turn through the agent and returns the final answer."""

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def run(self, agent_input: AgentInput) -> AgentOutput:
        pass
