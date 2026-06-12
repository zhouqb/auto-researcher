"""Experience-memory tools (design §5): retrieve at planning, record after runs."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Optional

from google.adk.tools import ToolContext

from ..config import get_settings
from ..storage.experiences import ExperienceStore


@lru_cache(maxsize=1)
def _store() -> ExperienceStore:
    return ExperienceStore(get_settings().db_path)


async def search_experiences(query: str, k: int = 5) -> dict[str, Any]:
    """Search past experiment experiences (successes AND failures) across all projects.

    Call this BEFORE planning an experiment: prior failure modes tell you what
    to avoid; prior successes tell you what worked. Treat hits as evidence,
    not law — a single old failure does not blacklist an approach.

    Args:
        query: Keywords describing the hypothesis/method, e.g.
            "quasi-random sampling integration RMSE stdlib".
        k: Max experiences to return.

    Returns:
        {"experiences": [{outcome, failure_mode, hypothesis, method, result,
                          lessons, confidence, created_at, experience_id}]}
    """
    hits = _store().search(query, k=k)
    return {"experiences": [h.brief() for h in hits]}


async def record_experience(
    hypothesis: str,
    outcome: str,
    lessons: str,
    tool_context: ToolContext,
    method: Optional[dict[str, Any]] = None,
    result: Optional[dict[str, Any]] = None,
    failure_mode: Optional[str] = None,
    codex_thread_id: Optional[str] = None,
    confidence: float = 0.7,
    supersedes: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Record one experiment experience to permanent cross-project memory.

    Call this after analyzing EVERY experiment run — failures and
    inconclusive runs are as valuable as successes. Be concrete in lessons
    ("X caused Y; do Z instead"), they are what future planners read.

    Args:
        hypothesis: What the experiment tested, one sentence.
        outcome: One of: success, failure, inconclusive, aborted.
        lessons: 1-3 concrete, transferable lessons.
        method: {"dataset": ..., "approach": ..., "key_params": {...}}.
        result: {"metric": ..., "value": ..., "baseline": ...}.
        failure_mode: One of: OOM, data_leakage, no_improvement, bug,
            divergence — or null for successes.
        codex_thread_id: thread_id of the run, for traceability.
        confidence: 0-1, how much to trust this conclusion.
        supersedes: experience_ids this record overturns (e.g. a re-test
            disproving an earlier failure).
    """
    exp = _store().record(
        project_id=tool_context.session.id,
        hypothesis=hypothesis,
        outcome=outcome,
        lessons=lessons,
        method=method,
        result=result,
        failure_mode=failure_mode,
        iteration=1,
        branch="main",
        codex_thread_id=codex_thread_id,
        confidence=confidence,
        supersedes=supersedes,
    )
    return {"experience_id": exp.experience_id, "outcome": exp.outcome}
