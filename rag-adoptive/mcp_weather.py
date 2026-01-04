"""
MCP weather server (STDIO) using public, keyless APIs (Open-Meteo + NWS).
Exposes two tools:
- get_alerts(state): US weather alerts by 2-letter state code
- get_forecast(latitude, longitude): 5-period forecast for given coords

Run: uv run mcp_weather.py
"""

import asyncio
import logging
from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP

logging.basicConfig(level=logging.INFO)

mcp = FastMCP("weather")

NWS_API_BASE = "https://api.weather.gov"
USER_AGENT = "weather-mcp/1.0"


async def make_nws_request(url: str) -> dict[str, Any] | None:
    """Make a request to the NWS API with proper error handling."""
    headers = {"User-Agent": USER_AGENT, "Accept": "application/geo+json"}
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers, timeout=30.0)
            response.raise_for_status()
            return response.json()
        except Exception as exc:  # noqa: BLE001
            logging.warning("NWS request failed: %s", exc)
            return None


def format_alert(feature: dict) -> str:
    """Format an alert feature into a readable string."""
    props = feature.get("properties", {})
    return (
        f"Event: {props.get('event', 'Unknown')}\n"
        f"Area: {props.get('areaDesc', 'Unknown')}\n"
        f"Severity: {props.get('severity', 'Unknown')}\n"
        f"Description: {props.get('description', 'No description available')}\n"
        f"Instructions: {props.get('instruction', 'No specific instructions provided')}"
    )


@mcp.tool()
async def get_alerts(state: str) -> str:
    """Get weather alerts for a US state (2-letter code, e.g., TX, CA)."""
    url = f"{NWS_API_BASE}/alerts/active/area/{state}"
    data = await make_nws_request(url)

    if not data or "features" not in data:
        return "Unable to fetch alerts or no alerts found."

    if not data["features"]:
        return "No active alerts for this state."

    alerts = [format_alert(feature) for feature in data["features"]]
    return "\n---\n".join(alerts)


async def make_open_meteo_request(latitude: float, longitude: float) -> dict[str, Any] | None:
    """Call Open-Meteo for a short forecast."""
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current_weather": True,
        "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_probability_max"],
        "timezone": "auto",
    }
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=30.0)
            resp.raise_for_status()
            return resp.json()
    except Exception as exc:  # noqa: BLE001
        logging.warning("Open-Meteo request failed: %s", exc)
        return None


@mcp.tool()
async def get_forecast(latitude: float, longitude: float) -> str:
    """Get weather forecast for a location (uses Open-Meteo, no API key)."""
    data = await make_open_meteo_request(latitude, longitude)
    if not data:
        return "Unable to fetch forecast data for this location."

    cw = data.get("current_weather", {})
    daily = data.get("daily", {})
    max_t = _first(daily.get("temperature_2m_max"))
    min_t = _first(daily.get("temperature_2m_min"))
    precip = _first(daily.get("precipitation_probability_max"))
    temp = cw.get("temperature")
    wind = cw.get("windspeed")
    code = cw.get("weathercode")

    return (
        f"Now: {temp}°C, wind {wind} km/h, code {code}. "
        f"Today: high {max_t}°C / low {min_t}°C, precip chance {precip}%."
    )


def _first(seq):
    if not seq:
        return None
    if isinstance(seq, list):
        return seq[0]
    return seq


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    asyncio.run(asyncio.to_thread(main))
