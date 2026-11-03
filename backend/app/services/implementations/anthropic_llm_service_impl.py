import logging
from typing import Any, Dict, List, Optional

from app.config.app_config import AIConfigConstants
from app.exceptions.custom_exceptions import LLMNotConfiguredException, LLMServiceException
from app.services.interfaces.llm_service_interface import ILLMService, LLMResult, ToolCall

logger = logging.getLogger(__name__)


def _block_to_dict(block) -> Optional[Dict[str, Any]]:
    """Convert an SDK content block to the plain-dict form accepted by messages.create()."""
    btype = getattr(block, "type", None)
    if btype == "text":
        return {"type": "text", "text": block.text}
    if btype == "tool_use":
        return {"type": "tool_use", "id": block.id, "name": block.name, "input": dict(block.input or {})}
    return None  # thinking / other block types are not replayed


class AnthropicLLMService(ILLMService):
    """Claude via the official `anthropic` SDK (Messages API + native tool use)."""

    def __init__(self, api_key: str, model: str,
                 max_tokens: int = AIConfigConstants.LLM_MAX_TOKENS,
                 timeout: float = AIConfigConstants.LLM_TIMEOUT_SECONDS):
        # No temperature/top_p: recent SDK/model versions reject sampling params.
        # Behaviour is controlled by the prompt templates instead.
        self._api_key = api_key
        self._model = model
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._client = None

    @property
    def model_name(self) -> str:
        return self._model

    def _get_client(self):
        if not self._api_key:
            raise LLMNotConfiguredException()
        if self._client is None:
            try:
                import anthropic
            except ImportError as exc:
                raise LLMNotConfiguredException(f"anthropic SDK is not installed: {exc}")
            self._client = anthropic.Anthropic(api_key=self._api_key, timeout=self._timeout, max_retries=2)
        return self._client

    def generate(self, system: str, messages: List[Dict[str, Any]],
                 tools: Optional[List[Dict[str, Any]]] = None) -> LLMResult:
        client = self._get_client()
        import anthropic

        kwargs = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "system": system,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools

        try:
            response = client.messages.create(**kwargs)
        except anthropic.AuthenticationError:
            logger.exception("Anthropic authentication failed")
            raise LLMNotConfiguredException("ANTHROPIC_API_KEY was rejected by the provider.")
        except anthropic.RateLimitError:
            logger.warning("Anthropic rate limit hit")
            raise LLMServiceException("LLM provider rate limit reached. Please retry shortly.")
        except anthropic.APIError as exc:
            logger.exception("Anthropic API error")
            raise LLMServiceException(f"LLM provider error: {getattr(exc, 'message', str(exc))}")

        blocks = [b for b in (_block_to_dict(x) for x in response.content) if b is not None]
        text = "".join(b["text"] for b in blocks if b["type"] == "text").strip()
        tool_calls = [ToolCall(id=b["id"], name=b["name"], input=b["input"]) for b in blocks if b["type"] == "tool_use"]
        usage = getattr(response, "usage", None)
        return LLMResult(
            text=text,
            model=getattr(response, "model", self._model),
            input_tokens=getattr(usage, "input_tokens", None),
            output_tokens=getattr(usage, "output_tokens", None),
            stop_reason=getattr(response, "stop_reason", None),
            tool_calls=tool_calls,
            content_blocks=blocks,
        )
