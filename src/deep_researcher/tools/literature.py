"""Literature search tools: Semantic Scholar Graph API + arXiv Atom API.

Both are plain ADK function tools returning compact JSON-able dicts — full
papers never enter context, only metadata + abstracts/TLDRs. Semantic Scholar
allows ~100 requests / 5 min unauthenticated; an optional API key
(`SEMANTIC_SCHOLAR_API_KEY`) raises that. Retries once on 429.
"""

from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from typing import Any, Optional

import httpx

from ..config import get_settings

_S2_BASE = "https://api.semanticscholar.org/graph/v1"
_S2_FIELDS = "title,authors,year,venue,citationCount,abstract,tldr,externalIds,url,openAccessPdf"
_ARXIV_BASE = "https://export.arxiv.org/api/query"
_TIMEOUT = 30.0


def _s2_headers() -> dict[str, str]:
    key = get_settings().semantic_scholar_api_key
    return {"x-api-key": key} if key else {}


def _s2_get(url: str, params: dict[str, Any]) -> httpx.Response:
    # The unauthenticated shared pool 429s often; back off a few times.
    for delay in (0, 5, 15, 30):
        if delay:
            time.sleep(delay)
        resp = httpx.get(url, params=params, headers=_s2_headers(), timeout=_TIMEOUT)
        if resp.status_code != 429:
            break
    resp.raise_for_status()
    return resp


def _shorten(text: Optional[str], limit: int = 600) -> Optional[str]:
    if text and len(text) > limit:
        return text[:limit].rsplit(" ", 1)[0] + "…"
    return text


def _format_s2_paper(p: dict[str, Any]) -> dict[str, Any]:
    tldr = (p.get("tldr") or {}).get("text")
    external = p.get("externalIds") or {}
    return {
        "title": p.get("title"),
        "authors": [a.get("name") for a in (p.get("authors") or [])][:6],
        "year": p.get("year"),
        "venue": p.get("venue") or None,
        "citation_count": p.get("citationCount"),
        "arxiv_id": external.get("ArXiv"),
        "doi": external.get("DOI"),
        "url": p.get("url"),
        "pdf_url": (p.get("openAccessPdf") or {}).get("url"),
        "tldr_or_abstract": tldr or _shorten(p.get("abstract")),
    }


def search_semantic_scholar(
    query: str,
    limit: int = 8,
    year_from: Optional[int] = None,
    fields_of_study: Optional[str] = None,
) -> dict[str, Any]:
    """Search Semantic Scholar for papers by relevance.

    Args:
        query: Keyword query, e.g. "mixture of experts routing".
        limit: Max papers to return (1-20).
        year_from: Only include papers published in or after this year.
        fields_of_study: Optional comma-separated filter, e.g. "Computer Science".

    Returns:
        dict with "papers": list of {title, authors, year, venue, citation_count,
        arxiv_id, doi, url, pdf_url, tldr_or_abstract}.
    """
    params: dict[str, Any] = {
        "query": query,
        "limit": max(1, min(limit, 20)),
        "fields": _S2_FIELDS,
    }
    if year_from:
        params["year"] = f"{year_from}-"
    if fields_of_study:
        params["fieldsOfStudy"] = fields_of_study
    try:
        data = _s2_get(f"{_S2_BASE}/paper/search", params).json()
    except httpx.HTTPError as e:
        return {"error": f"Semantic Scholar request failed: {e}"}
    papers = [_format_s2_paper(p) for p in data.get("data") or []]
    return {"total": data.get("total", len(papers)), "papers": papers}


_OPENALEX_BASE = "https://api.openalex.org/works"


def _reconstruct_abstract(inverted: Optional[dict[str, list[int]]]) -> Optional[str]:
    if not inverted:
        return None
    positions = [(pos, word) for word, posns in inverted.items() for pos in posns]
    return _shorten(" ".join(word for _, word in sorted(positions)))


def search_openalex(query: str, limit: int = 8, year_from: Optional[int] = None) -> dict[str, Any]:
    """Search OpenAlex for papers by relevance (no API key, reliable rate limits).

    Args:
        query: Keyword query, e.g. "mixture of experts routing".
        limit: Max papers to return (1-20).
        year_from: Only include papers published in or after this year.

    Returns:
        dict with "papers": list of {title, authors, year, venue, citation_count,
        doi, url, pdf_url, abstract}.
    """
    params: dict[str, Any] = {
        "search": query,
        "per-page": max(1, min(limit, 20)),
        "select": "display_name,authorships,publication_year,primary_location,"
                  "cited_by_count,doi,ids,abstract_inverted_index,open_access",
    }
    if year_from:
        params["filter"] = f"from_publication_date:{year_from}-01-01"
    mailto = get_settings().openalex_mailto
    if mailto:
        params["mailto"] = mailto
    try:
        resp = httpx.get(_OPENALEX_BASE, params=params, timeout=_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPError as e:
        return {"error": f"OpenAlex request failed: {e}"}
    papers = []
    for w in data.get("results") or []:
        source = ((w.get("primary_location") or {}).get("source") or {})
        papers.append({
            "title": w.get("display_name"),
            "authors": [
                (a.get("author") or {}).get("display_name")
                for a in (w.get("authorships") or [])
            ][:6],
            "year": w.get("publication_year"),
            "venue": source.get("display_name"),
            "citation_count": w.get("cited_by_count"),
            "doi": w.get("doi"),
            "url": (w.get("primary_location") or {}).get("landing_page_url")
                   or (w.get("ids") or {}).get("openalex"),
            "pdf_url": (w.get("open_access") or {}).get("oa_url"),
            "abstract": _reconstruct_abstract(w.get("abstract_inverted_index")),
        })
    return {"papers": papers}


_ATOM = "{http://www.w3.org/2005/Atom}"
_ARXIV_NS = "{http://arxiv.org/schemas/atom}"


def _parse_arxiv_feed(xml_text: str) -> list[dict[str, Any]]:
    root = ET.fromstring(xml_text)
    papers = []
    for entry in root.findall(f"{_ATOM}entry"):
        arxiv_url = (entry.findtext(f"{_ATOM}id") or "").strip()
        arxiv_id = arxiv_url.rsplit("/abs/", 1)[-1] if "/abs/" in arxiv_url else None
        papers.append({
            "title": " ".join((entry.findtext(f"{_ATOM}title") or "").split()),
            "authors": [
                a.findtext(f"{_ATOM}name")
                for a in entry.findall(f"{_ATOM}author")
            ][:6],
            "published": (entry.findtext(f"{_ATOM}published") or "")[:10],
            "arxiv_id": arxiv_id,
            "url": arxiv_url,
            "primary_category": (
                entry.find(f"{_ARXIV_NS}primary_category").get("term")
                if entry.find(f"{_ARXIV_NS}primary_category") is not None
                else None
            ),
            "abstract": _shorten(" ".join((entry.findtext(f"{_ATOM}summary") or "").split())),
        })
    return papers


def search_arxiv(query: str, max_results: int = 8, sort_by: str = "relevance") -> dict[str, Any]:
    """Search arXiv for papers (best for the newest preprints).

    Args:
        query: Keyword query, e.g. "state space models long context".
        max_results: Max papers to return (1-20).
        sort_by: "relevance" or "submittedDate" (newest first).

    Returns:
        dict with "papers": list of {title, authors, published, arxiv_id, url,
        primary_category, abstract}.
    """
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max(1, min(max_results, 20)),
        "sortBy": sort_by if sort_by in ("relevance", "submittedDate") else "relevance",
        "sortOrder": "descending",
    }
    try:
        resp = httpx.get(_ARXIV_BASE, params=params, timeout=_TIMEOUT, follow_redirects=True)
        resp.raise_for_status()
        papers = _parse_arxiv_feed(resp.text)
    except (httpx.HTTPError, ET.ParseError) as e:
        return {"error": f"arXiv request failed: {e}"}
    return {"papers": papers}
