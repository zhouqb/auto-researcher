"""Shared FTS5 query handling for the catalog and experience stores."""

from __future__ import annotations

from typing import Optional

_MAX_TERMS = 24


def sanitize_fts_query(query: str) -> Optional[str]:
    """Reduce a natural-language query to a safe OR-joined FTS5 keyword query.

    FTS5 treats quotes, colons, parens, hyphens, etc. as syntax — raw user
    text raises sqlite3.OperationalError. Returns None when nothing usable
    remains.
    """
    terms = [
        t for t in "".join(c if c.isalnum() else " " for c in query).split()
        if len(t) > 1
    ]
    if not terms:
        return None
    return " OR ".join(terms[:_MAX_TERMS])
