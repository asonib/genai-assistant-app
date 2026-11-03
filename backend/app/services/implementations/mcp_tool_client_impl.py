"""
MCP client for the custom tools server in <repo>/mcp_server.

Two transports are supported:
  * stdio (default) - the server is spawned as a subprocess per operation.
    Zero setup; ~0.5-1s overhead per call, mitigated by caching `tools/list`.
  * http            - connect to a long-running server started with
    `python mcp_server/server.py --transport http` (Streamable HTTP).

Django views are synchronous, so each async MCP operation is executed with
asyncio on a private event loop (or on a worker thread if a loop is already
running, e.g. under ASGI).
"""
import asyncio
import concurrent.futures
import json
import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional

from app.config.app_config import AIConfigConstants
from app.exceptions.custom_exceptions import ExternalToolException
from app.services.interfaces.external_tool_client_interface import IExternalToolClient, ToolResult, ToolSpec

logger = logging.getLogger(__name__)


def run_coroutine_sync(coro_factory, timeout: float):
    """Run `coro_factory()` to completion from synchronous code, with a timeout."""

    async def _with_timeout():
        return await asyncio.wait_for(coro_factory(), timeout=timeout)

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_with_timeout())
    # A loop is already running in this thread (ASGI): use a worker thread.
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, _with_timeout()).result(timeout=timeout + 5)


def _result_to_text(result) -> str:
    parts = []
    for item in getattr(result, "content", None) or []:
        if getattr(item, "type", None) == "text":
            parts.append(item.text)
        else:
            parts.append(f"[{getattr(item, 'type', 'content')} omitted]")
    if not parts and getattr(result, "structuredContent", None):
        parts.append(json.dumps(result.structuredContent, ensure_ascii=False))
    return "\n".join(parts).strip()


class McpToolClient(IExternalToolClient):
    def __init__(self, transport: str = "stdio", command: Optional[str] = None, args: Optional[List[str]] = None,
                 url: Optional[str] = None,
                 call_timeout: float = AIConfigConstants.MCP_CALL_TIMEOUT_SECONDS,
                 cache_seconds: float = AIConfigConstants.MCP_TOOLS_CACHE_SECONDS):
        if transport not in ("stdio", "http"):
            raise ValueError("MCP transport must be 'stdio' or 'http'")
        self._transport = transport
        self._command = command
        self._args = list(args or [])
        self._url = url
        self._call_timeout = call_timeout
        self._cache_seconds = cache_seconds
        self._tools_cache: Optional[List[ToolSpec]] = None
        self._tools_cached_at = 0.0
        self._lock = threading.Lock()

    @property
    def transport(self) -> str:
        return self._transport

    # ---- session plumbing --------------------------------------------------
    async def _with_session(self, fn):
        from mcp import ClientSession  # lazy: the backend runs without `mcp` installed

        if self._transport == "stdio":
            from mcp import StdioServerParameters
            from mcp.client.stdio import stdio_client

            params = StdioServerParameters(
                command=self._command,
                args=self._args,
                env=dict(os.environ),  # keep proxy / PATH settings for outbound HTTP
                cwd=os.path.dirname(self._args[0]) if self._args else None,
            )
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    return await fn(session)

        from mcp.client import streamable_http
        connect = getattr(streamable_http, "streamablehttp_client", None) or streamable_http.streamable_http_client
        async with connect(self._url) as streams:
            read, write = streams[0], streams[1]
            async with ClientSession(read, write) as session:
                await session.initialize()
                return await fn(session)

    def _run(self, fn):
        try:
            return run_coroutine_sync(lambda: self._with_session(fn), timeout=self._call_timeout)
        except ImportError as exc:
            raise ExternalToolException(f"The 'mcp' package is not installed: {exc}")
        except (asyncio.TimeoutError, concurrent.futures.TimeoutError):
            raise ExternalToolException(f"MCP server did not respond within {self._call_timeout:.0f}s.")
        except ExternalToolException:
            raise
        except BaseException as exc:  # ExceptionGroup from anyio task groups, OSError, ...
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            logger.warning("MCP %s operation failed: %r", self._transport, exc)
            raise ExternalToolException(f"MCP server error: {exc}")

    # ---- IExternalToolClient ---------------------------------------------
    def list_tools(self) -> List[ToolSpec]:
        with self._lock:
            if self._tools_cache is not None and time.monotonic() - self._tools_cached_at < self._cache_seconds:
                return list(self._tools_cache)

        async def _list(session):
            result = await session.list_tools()
            return [
                ToolSpec(
                    name=t.name,
                    description=(t.description or "").strip(),
                    input_schema=dict(t.inputSchema or {"type": "object", "properties": {}}),
                )
                for t in result.tools
            ]

        try:
            tools = self._run(_list)
        except ExternalToolException as exc:
            logger.warning("MCP tools unavailable, continuing without them: %s", exc.detail)
            return []
        with self._lock:
            self._tools_cache, self._tools_cached_at = tools, time.monotonic()
        logger.info("MCP tools discovered: %s", [t.name for t in tools])
        return list(tools)

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> ToolResult:
        async def _call(session):
            return await session.call_tool(name, arguments or {})

        result = self._run(_call)
        return ToolResult(content=_result_to_text(result) or "(empty result)",
                          is_error=bool(getattr(result, "isError", False)))


class DisabledToolClient(IExternalToolClient):
    """Used when MCP_ENABLED=False: the agent only has document search."""

    @property
    def transport(self) -> str:
        return "disabled"

    def list_tools(self) -> List[ToolSpec]:
        return []

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> ToolResult:
        raise ExternalToolException("External tools are disabled.")
