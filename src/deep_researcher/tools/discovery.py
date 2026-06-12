"""Discovery tools beyond paper indexes: OpenReview, GitHub, web search.

- search_openreview: peer reviews / venue decisions — a quality signal no
  paper index has.
- search_github: implementations and adoption signal (stars) for a method.
- search_web: engineering blogs and docs via Tavily (needs TAVILY_API_KEY;
  returns a clear error when unconfigured so agents can move on).
"""

from __future__ import annotations

import subprocess
from functools import lru_cache
from typing import Any, Optional

import httpx

from ..config import get_settings

_OPENREVIEW_BASE = "https://api2.openreview.net/notes/search"
_GITHUB_BASE = "https://api.github.com/search/repositories"
_TAVILY_BASE = "https://api.tavily.com/search"
_TIMEOUT = 30.0


def _value(content: dict[str, Any], key: str) -> Optional[Any]:
    """OpenReview API v2 wraps fields as {'value': ...}; v1 notes don't."""
    field = content.get(key)
    if isinstance(field, dict):
        return field.get("value")
    return field


def _shorten(text: Optional[str], limit: int = 500) -> Optional[str]:
    if text and len(text) > limit:
        return text[:limit].rsplit(" ", 1)[0] + "…"
    return text


def search_openreview(query: str, limit: int = 8) -> dict[str, Any]:
    """Search OpenReview for peer-reviewed submissions (ICLR/NeurIPS/etc.).

    Useful for the review signal other indexes lack: venue decisions and
    links to the full reviews on the forum page.

    Args:
        query: Keyword query, e.g. "mixture of experts routing".
        limit: Max papers to return (1-15).

    Returns:
        dict with "papers": list of {title, venue, year, abstract, forum_url}.
    """
    limit = max(1, min(limit, 15))
    try:
        resp = httpx.get(
            _OPENREVIEW_BASE,
            # Search results mix submissions with their reviews (~1/3 are
            # titled submissions); over-fetch a fixed 100 and keep only
            # titled, unique-forum notes.
            params={"term": query, "limit": 100, "content": "all"},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        notes = resp.json().get("notes") or []
    except httpx.HTTPError as e:
        return {"error": f"OpenReview request failed: {e}"}

    papers, seen_forums = [], set()
    for note in notes:
        content = note.get("content") or {}
        title = _value(content, "title")
        forum = note.get("forum")
        if not title or forum in seen_forums:
            continue
        seen_forums.add(forum)
        cdate = note.get("cdate")
        papers.append({
            "title": title,
            "venue": _value(content, "venue") or _value(content, "venueid"),
            "year": (
                int(cdate / 1000 // 31556952) + 1970 if isinstance(cdate, (int, float)) else None
            ),
            "abstract": _shorten(_value(content, "abstract")),
            "forum_url": f"https://openreview.net/forum?id={forum}" if forum else None,
        })
        if len(papers) >= limit:
            break
    return {"papers": papers}


@lru_cache(maxsize=1)
def _github_token() -> Optional[str]:
    settings = get_settings()
    if settings.github_token:
        return settings.github_token
    try:  # fall back to the gh CLI's stored credentials
        out = subprocess.run(
            ["gh", "auth", "token"], capture_output=True, text=True, timeout=5
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.TimeoutExpired):
        return None


def search_github(
    query: str, limit: int = 8, sort: str = "best-match", language: Optional[str] = None
) -> dict[str, Any]:
    """Search GitHub repositories — find implementations of a method or paper.

    Args:
        query: Keyword query, e.g. "switch transformer implementation".
        limit: Max repositories to return (1-15).
        sort: "best-match" (relevance) or "stars".
        language: Optional language filter, e.g. "Python".

    Returns:
        dict with "repos": list of {full_name, description, stars, language,
        url, last_push, topics}.
    """
    q = query + (f" language:{language}" if language else "")
    params: dict[str, Any] = {"q": q, "per_page": max(1, min(limit, 15))}
    if sort == "stars":
        params["sort"] = "stars"
    headers = {"Accept": "application/vnd.github+json"}
    if token := _github_token():
        headers["Authorization"] = f"Bearer {token}"
    try:
        resp = httpx.get(_GITHUB_BASE, params=params, headers=headers, timeout=_TIMEOUT)
        resp.raise_for_status()
        items = resp.json().get("items") or []
    except httpx.HTTPError as e:
        return {"error": f"GitHub search failed: {e}"}
    return {
        "repos": [
            {
                "full_name": r.get("full_name"),
                "description": _shorten(r.get("description"), 300),
                "stars": r.get("stargazers_count"),
                "language": r.get("language"),
                "url": r.get("html_url"),
                "last_push": (r.get("pushed_at") or "")[:10] or None,
                "topics": (r.get("topics") or [])[:6],
            }
            for r in items
        ]
    }


def search_web(query: str, max_results: int = 6) -> dict[str, Any]:
    """Search the web (engineering blogs, docs, news) via Tavily.

    Use for material paper indexes miss: vendor engineering blogs, framework
    docs, postmortems. Requires TAVILY_API_KEY; if unconfigured, returns an
    error — just continue with the other search tools.

    Args:
        query: Natural-language query.
        max_results: Max results to return (1-10).

    Returns:
        dict with "results": list of {title, url, snippet}.
    """
    key = get_settings().tavily_api_key
    if not key:
        return {"error": "web search not configured (set TAVILY_API_KEY); use other tools"}
    try:
        resp = httpx.post(
            _TAVILY_BASE,
            headers={"Authorization": f"Bearer {key}"},
            json={"query": query, "max_results": max(1, min(max_results, 10))},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        results = resp.json().get("results") or []
    except httpx.HTTPError as e:
        return {"error": f"Tavily request failed: {e}"}
    return {
        "results": [
            {
                "title": r.get("title"),
                "url": r.get("url"),
                "snippet": _shorten(r.get("content")),
            }
            for r in results
        ]
    }
