"""
Legacy tool runner (not used by the LangGraph agent). Retained for optional extension.
"""

from dataclasses import dataclass
from typing import List, Optional

from src.config import Settings
from src.external_search import external_search


@dataclass
class ToolResult:
    name: Optional[str]
    content: str
    source: Optional[str] = None


def run_tools(action: str, query: str, settings: Settings) -> List[ToolResult]:
    results: List[ToolResult] = []
    if "weather" in action:
        for ctx in external_search(query, settings):
            results.append(ToolResult(name="weather", content=ctx, source="external_search"))
    return results
