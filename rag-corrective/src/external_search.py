"""
External search for CRAG fallback (public, no API keys).

- Weather/forecast: Open-Meteo geocoding + forecast (free, no key).
- If not a weather query, returns a simple placeholder noting no live data.
"""

import os
import re
from typing import Callable, Dict, List, Optional, Tuple

import requests
from openai import OpenAI

from src.config import Settings


ToolHandler = Callable[[str, Settings], Optional[str]]

TOOLS: Dict[str, Dict[str, object]] = {
    "weather_forecast": {
        "description": "Get current weather and today forecast for a city or multiple cities",
        "keywords": ["weather", "forecast", "temperature", "rain", "snow", "wind", "trip", "travel", "plan"],
        "handler": None,  # filled after function definition
    }
}


def external_search(query: str, settings: Optional[Settings] = None) -> List[str]:
    settings = settings or Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        database_url=os.getenv("DATABASE_URL", ""),
    )
    tool_name = select_external_tool(query, settings)
    results: List[str] = []
    locations: List[str] = []

    if tool_name == "weather_forecast":
        locations = llm_extract_locations(query, settings) or [query]

    # If no tool selected but multiple locations found (trip-style queries), fallback to weather
    if not tool_name:
        locs = llm_extract_locations(query, settings)
        if locs:
            tool_name = "weather_forecast"
            locations = locs

    if tool_name == "weather_forecast":
        if not locations:
            locations = [query]
        for loc in locations:
            fixed_query = llm_correct_location(loc, settings)
            weather = fetch_weather_and_forecast(fixed_query)
            if weather:
                results.append(weather)

    if not results:
        results.append(f"(External API) No live data available for: {query} (tool selected: {tool_name})")
    return results


def select_external_tool(query: str, settings: Settings) -> Optional[str]:
    """Pick the best external tool using keywords first, then LLM routing."""
    normalized = query.lower()
    for name, meta in TOOLS.items():
        for kw in meta.get("keywords", []):
            if kw in normalized:
                return name
    # LLM routing as a fallback for ambiguous queries
    return llm_route_tool(query, settings)


def fetch_weather_and_forecast(query: str) -> Optional[str]:
    location = geocode_location(query)
    if not location:
        return None
    lat, lon, name = location
    try:
        resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "current_weather": True,
                "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_probability_max"],
                "timezone": "auto",
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        cw = data.get("current_weather", {})
        daily = data.get("daily", {})
        temp = cw.get("temperature")
        wind = cw.get("windspeed")
        conditions = cw.get("weathercode")
        max_t = _first(daily.get("temperature_2m_max"))
        min_t = _first(daily.get("temperature_2m_min"))
        precip = _first(daily.get("precipitation_probability_max"))
        return (
            f"(External API) Weather for {name}: now {temp}°C, wind {wind} km/h, code {conditions}. "
            f"Today: high {max_t}°C / low {min_t}°C, precip chance {precip}%."
        )
    except Exception:
        return None


def geocode_location(query: str) -> Optional[Tuple[float, float, str]]:
    candidates = []
    tokens = _tokenize(query)
    corrected_tokens = _apply_corrections(tokens)

    # Original and simplified queries
    candidates.append(query)
    simplified = _simplify_location_query(query)
    if simplified and simplified != query:
        candidates.append(simplified)

    # Corrected token joins
    if corrected_tokens:
        candidates.append(" ".join(corrected_tokens))
        if len(corrected_tokens) >= 2:
            candidates.append(" ".join(corrected_tokens[:2]))
            candidates.append(corrected_tokens[0])

    # Add city-only candidate from tokens
    if tokens:
        candidates.append(tokens[0])

    # Deduplicate while preserving order
    seen = set()
    unique_candidates = []
    for cand in candidates:
        key = cand.lower().strip()
        if key and key not in seen:
            seen.add(key)
            unique_candidates.append(cand)

    for cand in unique_candidates:
        try:
            resp = requests.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={"name": cand, "count": 1, "language": "en", "format": "json"},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results") or []
            if not results:
                continue
            hit = results[0]
            return hit.get("latitude"), hit.get("longitude"), hit.get("name")
        except Exception:
            continue
    return None


def _simplify_location_query(query: str) -> str:
    # Drop obvious weather words and filler to improve geocoding.
    drop_words = {
        "weather",
        "forecast",
        "today",
        "tomorrow",
        "how",
        "is",
        "the",
        "in",
        "for",
        "current",
        "conditions",
        "like",
        "now",
    }
    parts = [w for w in re.split(r"[^a-zA-Z0-9]+", query) if w]
    cleaned = [w for w in parts if w.lower() not in drop_words]
    if not cleaned:
        return query
    return " ".join(cleaned)


def _tokenize(text: str) -> List[str]:
    return [w for w in re.split(r"[^a-zA-Z0-9]+", text) if w]


def _apply_corrections(tokens: List[str]) -> List[str]:
    corrections = {
        "tecas": "texas",
        "texes": "texas",
        "txas": "texas",
        "dalas": "dallas",
    }
    corrected = []
    for tok in tokens:
        lower = tok.lower()
        corrected.append(corrections.get(lower, tok))
    return corrected


def llm_correct_location(query: str, settings: Settings) -> str:
    """
    Use the chat model to clean and correct a location string into 'City, State' (US) format.
    Falls back to the original query on failure.
    """
    prompt = (
        "Normalize the following location to a concise 'City, State' or 'City, Country' string. "
        "Fix misspellings. If unsure, return the best guess without extra text.\n"
        f"Location: {query}"
    )
    try:
        client = OpenAI(api_key=settings.openai_api_key)
        resp = client.chat.completions.create(
            model=settings.chat_model,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        content = resp.choices[0].message.content.strip()
        return content or query
    except Exception:
        return query


def llm_route_tool(query: str, settings: Settings) -> Optional[str]:
    """Use the chat model to pick a tool name from TOOLS or return None."""
    tool_list = ", ".join(TOOLS.keys())
    prompt = (
        "Choose the best external tool for the user request from this list: "
        f"{tool_list}. Reply with only the tool name, or 'none' if no match.\n"
        f"Request: {query}"
    )
    try:
        client = OpenAI(api_key=settings.openai_api_key)
        resp = client.chat.completions.create(
            model=settings.chat_model,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        content = (resp.choices[0].message.content or "").strip().lower()
        if content in TOOLS:
            return content
        return None
    except Exception:
        return None


def llm_extract_locations(query: str, settings: Settings, max_locations: int = 3) -> List[str]:
    """
    Use LLM to extract up to max_locations location strings from the query.
    Returns list of city/state/country strings.
    """
    prompt = (
        "Extract up to {max_locations} location names (city, state, or country) from the request. "
        "Return them as a comma-separated list, no extra text.\n"
        f"Request: {query}"
    ).format(max_locations=max_locations)
    try:
        client = OpenAI(api_key=settings.openai_api_key)
        resp = client.chat.completions.create(
            model=settings.chat_model,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        content = (resp.choices[0].message.content or "").strip()
        if not content:
            return []
        parts = [p.strip() for p in content.split(",") if p.strip()]
        return parts[:max_locations]
    except Exception:
        return []


def _first(seq):
    if not seq:
        return None
    if isinstance(seq, list):
        return seq[0]
    return seq
