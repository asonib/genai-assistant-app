"""
Pure tool logic for the custom MCP server (no MCP dependency, unit-testable).

Data sources (free, no API key):
  * Weather  - Open-Meteo geocoding + forecast API   https://open-meteo.com
  * Currency - Frankfurter (European Central Bank reference rates)  https://frankfurter.dev
"""
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable, Dict, List, Optional

GEOCODE_URL = os.getenv("OPEN_METEO_GEOCODE_URL", "https://geocoding-api.open-meteo.com/v1/search")
FORECAST_URL = os.getenv("OPEN_METEO_FORECAST_URL", "https://api.open-meteo.com/v1/forecast")
FX_URL = os.getenv("FRANKFURTER_URL", "https://api.frankfurter.dev/v1")
HTTP_TIMEOUT = float(os.getenv("TOOLS_HTTP_TIMEOUT", "10"))
USER_AGENT = "genai-assistant-mcp/1.0"

# WMO weather interpretation codes -> text (Open-Meteo docs).
WEATHER_CODES = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Depositing rime fog",
    51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
    56: "Light freezing drizzle", 57: "Dense freezing drizzle",
    61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
    66: "Light freezing rain", 67: "Heavy freezing rain",
    71: "Slight snow fall", 73: "Moderate snow fall", 75: "Heavy snow fall", 77: "Snow grains",
    80: "Slight rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
    85: "Slight snow showers", 86: "Heavy snow showers",
    95: "Thunderstorm", 96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail",
}


class ToolError(Exception):
    """User-facing tool failure (bad input, unknown city, upstream error)."""


HttpGetJson = Callable[[str, Dict[str, Any]], Any]


def http_get_json(url: str, params: Dict[str, Any]) -> Any:
    query = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
    request = urllib.request.Request(f"{url}?{query}", headers={"User-Agent": USER_AGENT,
                                                                 "Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT) as response:  # noqa: S310 (fixed https hosts)
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")[:300]
        raise ToolError(f"Upstream API returned HTTP {exc.code}: {body}")
    except (urllib.error.URLError, TimeoutError) as exc:
        raise ToolError(f"Could not reach upstream API: {exc}")
    except json.JSONDecodeError:
        raise ToolError("Upstream API returned invalid JSON.")


# ---------------------------------------------------------------------------
# Weather
# ---------------------------------------------------------------------------
def get_weather(city: str, days: int = 3, units: str = "metric",
                fetch: Optional[HttpGetJson] = None) -> Dict[str, Any]:
    fetch = fetch or http_get_json
    city = (city or "").strip()
    if not city:
        raise ToolError("'city' is required.")
    if units not in ("metric", "imperial"):
        raise ToolError("'units' must be 'metric' or 'imperial'.")
    days = max(1, min(int(days), 7))

    geo = fetch(GEOCODE_URL, {"name": city, "count": 1, "language": "en", "format": "json"})
    results = (geo or {}).get("results") or []
    if not results:
        raise ToolError(f"Could not find a location named '{city}'.")
    place = results[0]

    imperial = units == "imperial"
    forecast = fetch(FORECAST_URL, {
        "latitude": place["latitude"],
        "longitude": place["longitude"],
        "current": "temperature_2m,apparent_temperature,relative_humidity_2m,weather_code,wind_speed_10m",
        "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
        "forecast_days": days,
        "timezone": "auto",
        "temperature_unit": "fahrenheit" if imperial else "celsius",
        "wind_speed_unit": "mph" if imperial else "kmh",
    })
    current = forecast.get("current") or {}
    daily = forecast.get("daily") or {}
    temp_unit = "°F" if imperial else "°C"
    wind_unit = "mph" if imperial else "km/h"

    daily_rows: List[Dict[str, Any]] = []
    for i, day in enumerate(daily.get("time") or []):
        def at(key):
            values = daily.get(key) or []
            return values[i] if i < len(values) else None
        daily_rows.append({
            "date": day,
            "conditions": WEATHER_CODES.get(at("weather_code"), "Unknown"),
            "temp_max": at("temperature_2m_max"),
            "temp_min": at("temperature_2m_min"),
            "precipitation_probability_pct": at("precipitation_probability_max"),
        })

    return {
        "location": {
            "name": place.get("name"),
            "region": place.get("admin1"),
            "country": place.get("country"),
            "latitude": place.get("latitude"),
            "longitude": place.get("longitude"),
            "timezone": forecast.get("timezone"),
        },
        "units": {"temperature": temp_unit, "wind_speed": wind_unit},
        "current": {
            "time": current.get("time"),
            "temperature": current.get("temperature_2m"),
            "feels_like": current.get("apparent_temperature"),
            "humidity_pct": current.get("relative_humidity_2m"),
            "wind_speed": current.get("wind_speed_10m"),
            "conditions": WEATHER_CODES.get(current.get("weather_code"), "Unknown"),
        },
        "daily": daily_rows,
        "source": "Open-Meteo",
    }


# ---------------------------------------------------------------------------
# Currency (FX)
# ---------------------------------------------------------------------------
def _currency(code: str, field: str) -> str:
    code = (code or "").strip().upper()
    if len(code) != 3 or not code.isalpha():
        raise ToolError(f"'{field}' must be a 3-letter ISO 4217 currency code (e.g. USD, EUR, INR).")
    return code


def get_exchange_rates(base: str = "USD", symbols: Optional[List[str]] = None,
                       fetch: Optional[HttpGetJson] = None) -> Dict[str, Any]:
    fetch = fetch or http_get_json
    base = _currency(base, "base")
    wanted = [_currency(s, "symbols") for s in (symbols or [])]
    wanted = [s for s in wanted if s != base]
    data = fetch(f"{FX_URL}/latest", {"base": base, "symbols": ",".join(wanted) if wanted else None})
    rates = (data or {}).get("rates")
    if not rates:
        raise ToolError(f"No exchange rates available for base currency {base}.")
    return {"base": base, "date": data.get("date"), "rates": rates,
            "source": "Frankfurter (ECB reference rates)"}


def convert_currency(amount: float, from_currency: str, to_currency: str,
                     fetch: Optional[HttpGetJson] = None) -> Dict[str, Any]:
    try:
        amount = float(amount)
    except (TypeError, ValueError):
        raise ToolError("'amount' must be a number.")
    if amount < 0:
        raise ToolError("'amount' must be non-negative.")
    src = _currency(from_currency, "from_currency")
    dst = _currency(to_currency, "to_currency")
    if src == dst:
        return {"amount": amount, "from": src, "to": dst, "rate": 1.0, "converted": round(amount, 4),
                "date": None, "source": "identity"}
    rates = get_exchange_rates(src, [dst], fetch=fetch)
    rate = rates["rates"].get(dst)
    if rate is None:
        raise ToolError(f"Currency {dst} is not supported.")
    return {"amount": amount, "from": src, "to": dst, "rate": rate, "converted": round(amount * rate, 4),
            "date": rates["date"], "source": rates["source"]}
