"""
Custom MCP server exposing live external tools to the GenAI Assistant agent.

Tools: get_weather, convert_currency, get_exchange_rates

Run modes:
  python mcp_server/server.py                      # stdio (spawned by the Django backend)
  python mcp_server/server.py --transport http     # Streamable HTTP on 127.0.0.1:8765/mcp

Inspect with the MCP Inspector:  npx @modelcontextprotocol/inspector python mcp_server/server.py
"""
import argparse
import json
import os
import sys
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcp.server.fastmcp import FastMCP  # noqa: E402

import tools  # noqa: E402

mcp = FastMCP(
    "genai-assistant-tools",
    host=os.getenv("MCP_HOST", "127.0.0.1"),
    port=int(os.getenv("MCP_PORT", "8765")),
)


def _ok(payload) -> str:
    return json.dumps(payload, ensure_ascii=False)


@mcp.tool()
def get_weather(city: str, days: int = 3, units: str = "metric") -> str:
    """Current weather and a daily forecast (1-7 days) for a city.

    Args:
        city: City name, optionally with country, e.g. "Pune" or "Paris, France".
        days: Number of forecast days, 1-7 (default 3).
        units: "metric" (°C, km/h) or "imperial" (°F, mph).
    """
    try:
        return _ok(tools.get_weather(city.split(",")[0], days=days, units=units))
    except tools.ToolError as exc:
        raise ValueError(str(exc))


@mcp.tool()
def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """Convert an amount between currencies using the latest ECB reference rates.

    Args:
        amount: Amount to convert, e.g. 250.
        from_currency: ISO 4217 code of the source currency, e.g. "USD".
        to_currency: ISO 4217 code of the target currency, e.g. "INR".
    """
    try:
        return _ok(tools.convert_currency(amount, from_currency, to_currency))
    except tools.ToolError as exc:
        raise ValueError(str(exc))


@mcp.tool()
def get_exchange_rates(base: str = "USD", symbols: Optional[List[str]] = None) -> str:
    """Latest exchange rates for a base currency (ECB reference rates, updated daily).

    Args:
        base: ISO 4217 base currency code, e.g. "EUR".
        symbols: Optional list of target currency codes, e.g. ["USD", "INR"]. Omit for all.
    """
    try:
        return _ok(tools.get_exchange_rates(base, symbols))
    except tools.ToolError as exc:
        raise ValueError(str(exc))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    args = parser.parse_args(argv)
    mcp.run(transport="streamable-http" if args.transport == "http" else "stdio")


if __name__ == "__main__":
    main()
