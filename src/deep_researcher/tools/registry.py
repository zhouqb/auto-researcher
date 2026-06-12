"""Search-tool registry: which search backends agents actually get.

Six backends exist, but handing every searcher all of them inflates prompts
and invites dead calls (Semantic Scholar 429s without a key, Tavily errors
when unconfigured). ``SEARCH_TOOLS`` in Settings selects the enabled subset;
agents' toolsets and their instruction "tool guide" are both built from it.
"""

from __future__ import annotations

from typing import Callable, NamedTuple

from .discovery import search_github, search_openreview, search_web
from .literature import search_arxiv, search_openalex, search_semantic_scholar


class SearchTool(NamedTuple):
    fn: Callable
    guide: str  # one clause for the agent instruction's tool guide


SEARCH_TOOL_REGISTRY: dict[str, SearchTool] = {
    "openalex": SearchTool(
        search_openalex,
        "search_openalex is your primary paper index (citations, venues)",
    ),
    "arxiv": SearchTool(
        search_arxiv,
        "search_arxiv covers the latest preprints",
    ),
    "semantic_scholar": SearchTool(
        search_semantic_scholar,
        "search_semantic_scholar is a fallback paper index (rate-limited "
        "without a key — use it only if the other indexes return errors)",
    ),
    "openreview": SearchTool(
        search_openreview,
        "search_openreview adds peer-review signal (venue decisions, review "
        "links) for ML venues",
    ),
    "github": SearchTool(
        search_github,
        "search_github finds implementations and adoption signal (stars) "
        "when the facet concerns a method or tool",
    ),
    "web": SearchTool(
        search_web,
        "search_web covers engineering blogs/docs the indexes miss (skip it "
        "if it reports it is not configured)",
    ),
}


def parse_search_tools(spec: str) -> list[str]:
    """Validated, deduplicated tool names from a comma-separated spec."""
    names: list[str] = []
    for raw in spec.split(","):
        name = raw.strip().lower()
        if name and name not in names:
            names.append(name)
    unknown = [n for n in names if n not in SEARCH_TOOL_REGISTRY]
    if unknown:
        raise ValueError(
            f"unknown search tool(s) {unknown} in SEARCH_TOOLS; "
            f"valid names: {', '.join(SEARCH_TOOL_REGISTRY)}"
        )
    if not names:
        raise ValueError(
            f"SEARCH_TOOLS is empty; valid names: {', '.join(SEARCH_TOOL_REGISTRY)}"
        )
    return names


def search_tool_fns(names: list[str]) -> list[Callable]:
    return [SEARCH_TOOL_REGISTRY[n].fn for n in names]


def search_tool_guide(names: list[str]) -> str:
    return "Tool guide: " + "; ".join(SEARCH_TOOL_REGISTRY[n].guide for n in names) + "."
