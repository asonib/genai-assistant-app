from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass(frozen=True)
class ToolSpec:
    """Tool metadata as advertised by the MCP server (`tools/list`)."""
    name: str
    description: str
    input_schema: Dict[str, Any] = field(default_factory=lambda: {"type": "object", "properties": {}})

    def to_llm_tool(self) -> Dict[str, Any]:
        return {"name": self.name, "description": self.description, "input_schema": self.input_schema}


@dataclass(frozen=True)
class ToolResult:
    content: str
    is_error: bool = False


class IExternalToolClient(ABC):
    """Client for external tools served over the Model Context Protocol."""

    @property
    @abstractmethod
    def transport(self) -> str:
        pass

    @abstractmethod
    def list_tools(self) -> List[ToolSpec]:
        """Return available tools; must return [] (not raise) if the server is down."""
        pass

    @abstractmethod
    def call_tool(self, name: str, arguments: Dict[str, Any]) -> ToolResult:
        """Invoke a tool. Raises ExternalToolException on transport failure."""
        pass
