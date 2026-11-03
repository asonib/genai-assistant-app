"""Offline tests for the MCP tool logic:  python -m unittest discover -s mcp_server"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tools  # noqa: E402

GEO = {"results": [{"name": "Pune", "admin1": "Maharashtra", "country": "India",
                    "latitude": 18.52, "longitude": 73.86}]}
FORECAST = {
    "timezone": "Asia/Kolkata",
    "current": {"time": "2026-11-02T18:00", "temperature_2m": 27.4, "apparent_temperature": 28.1,
                "relative_humidity_2m": 61, "weather_code": 2, "wind_speed_10m": 9.3},
    "daily": {"time": ["2026-11-02", "2026-11-03"], "weather_code": [2, 61],
              "temperature_2m_max": [31.0, 29.5], "temperature_2m_min": [19.2, 20.1],
              "precipitation_probability_max": [5, 70]},
}


class FakeHttp:
    def __init__(self, routes):
        self.routes = routes
        self.calls = []

    def __call__(self, url, params):
        self.calls.append((url, params))
        for prefix, payload in self.routes.items():
            if url.startswith(prefix):
                if isinstance(payload, Exception):
                    raise payload
                return payload
        raise AssertionError(f"unexpected URL {url}")


class WeatherTests(unittest.TestCase):
    def test_weather_happy_path(self):
        http = FakeHttp({tools.GEOCODE_URL: GEO, tools.FORECAST_URL: FORECAST})
        out = tools.get_weather("Pune", days=2, fetch=http)
        self.assertEqual(out["location"]["country"], "India")
        self.assertEqual(out["current"]["conditions"], "Partly cloudy")
        self.assertEqual(out["daily"][1]["conditions"], "Slight rain")
        self.assertEqual(out["daily"][1]["precipitation_probability_pct"], 70)
        self.assertEqual(out["units"]["temperature"], "°C")
        forecast_params = http.calls[1][1]
        self.assertEqual((forecast_params["latitude"], forecast_params["forecast_days"]), (18.52, 2))

    def test_imperial_units_and_day_clamp(self):
        http = FakeHttp({tools.GEOCODE_URL: GEO, tools.FORECAST_URL: FORECAST})
        out = tools.get_weather("Pune", days=99, units="imperial", fetch=http)
        self.assertEqual(out["units"], {"temperature": "°F", "wind_speed": "mph"})
        self.assertEqual(http.calls[1][1]["forecast_days"], 7)
        self.assertEqual(http.calls[1][1]["temperature_unit"], "fahrenheit")

    def test_unknown_city_and_bad_input(self):
        http = FakeHttp({tools.GEOCODE_URL: {"results": []}})
        with self.assertRaisesRegex(tools.ToolError, "Could not find"):
            tools.get_weather("Atlantis", fetch=http)
        with self.assertRaises(tools.ToolError):
            tools.get_weather("  ", fetch=http)
        with self.assertRaises(tools.ToolError):
            tools.get_weather("Pune", units="kelvin", fetch=http)

    def test_upstream_error_propagates_as_tool_error(self):
        http = FakeHttp({tools.GEOCODE_URL: tools.ToolError("Could not reach upstream API")})
        with self.assertRaises(tools.ToolError):
            tools.get_weather("Pune", fetch=http)


class CurrencyTests(unittest.TestCase):
    def test_convert(self):
        http = FakeHttp({tools.FX_URL: {"base": "USD", "date": "2026-10-30", "rates": {"INR": 83.25}}})
        out = tools.convert_currency(250, "usd", "inr", fetch=http)
        self.assertEqual(out["converted"], 20812.5)
        self.assertEqual(out["rate"], 83.25)
        self.assertEqual(http.calls[0][1], {"base": "USD", "symbols": "INR"})

    def test_same_currency_needs_no_network(self):
        out = tools.convert_currency(5, "EUR", "eur", fetch=FakeHttp({}))
        self.assertEqual(out["converted"], 5.0)

    def test_validation(self):
        for args in [(1, "US", "INR"), (1, "USD", "123"), (-1, "USD", "INR"), ("ten", "USD", "INR")]:
            with self.assertRaises(tools.ToolError, msg=args):
                tools.convert_currency(*args, fetch=FakeHttp({}))

    def test_rates(self):
        http = FakeHttp({tools.FX_URL: {"base": "EUR", "date": "2026-10-30", "rates": {"USD": 1.08, "INR": 90.1}}})
        out = tools.get_exchange_rates("eur", ["usd", "inr", "eur"], fetch=http)
        self.assertEqual(out["rates"]["USD"], 1.08)
        self.assertEqual(http.calls[0][1]["symbols"], "USD,INR")


if __name__ == "__main__":
    unittest.main()
