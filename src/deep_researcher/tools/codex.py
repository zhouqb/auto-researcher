"""`codex_exec` tool: runs one Codex-implemented experiment (design §13).

Phase 1 scope: a single experiment branch ("main", iteration 1), executed
synchronously within the agent turn. Idempotent — the run_id derives from the
prompt, so a repeated call returns the cached result instead of a duplicate
paid run. Every run updates the `budget/budget.json` ledger artifact.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Optional

from google.adk.tools import ToolContext
from google.genai import types

from ..codex import prepare_workspace, run_codex
from ..config import get_settings

BUDGET_FILE = "budget/budget.json"


def _project_id(tool_context: ToolContext) -> str:
    return tool_context.session.id


async def _update_budget(
    tool_context: ToolContext, entry: dict[str, Any]
) -> dict[str, int]:
    existing = await tool_context.load_artifact(BUDGET_FILE)
    ledger = (
        json.loads(existing.text)
        if existing and existing.text
        else {"entries": [], "totals": {}}
    )
    ledger["entries"].append(entry)
    totals: dict[str, int] = {}
    for e in ledger["entries"]:
        for k, v in (e.get("usage") or {}).items():
            totals[k] = totals.get(k, 0) + v
        totals["wallclock_s"] = totals.get("wallclock_s", 0) + int(e.get("wallclock_s", 0))
    ledger["totals"] = totals
    await tool_context.save_artifact(
        BUDGET_FILE,
        types.Part(text=json.dumps(ledger, indent=2)),
        custom_metadata={
            "kind": "budget",
            "title": "Budget ledger",
            "summary": f"{len(ledger['entries'])} runs; totals: {json.dumps(totals)}",
            "created_by": tool_context.agent_name,
        },
    )
    return totals


async def codex_exec(
    task_prompt: str,
    tool_context: ToolContext,
    resume_thread_id: Optional[str] = None,
) -> dict[str, Any]:
    """Execute a code experiment with Codex in a sandboxed workspace.

    Call this AFTER the user approved the experiment budget (Gate 2). The
    workspace persists across calls, so a follow-up call can fix or extend
    earlier work; pass resume_thread_id (from a previous result) to continue
    the same Codex conversation with full context — preferred when debugging
    a failed run.

    Args:
        task_prompt: Complete, self-contained experiment instructions for
            Codex: hypothesis, method, dataset, baseline, metric, and the
            requirement to write metrics.json per the workspace AGENTS.md.
        resume_thread_id: thread_id of a prior run to continue (fix loop).

    Returns:
        {status, run_id, thread_id, final_message, metrics, usage,
         wallclock_s, budget_totals, cached}
    """
    settings = get_settings()
    project_id = _project_id(tool_context)
    exp_dir = settings.projects_dir / project_id / "iter_1" / "exp_main"
    workspace = exp_dir / "repo"
    prepare_workspace(workspace)

    seed = f"{task_prompt}|{resume_thread_id or ''}"
    run_id = hashlib.sha1(seed.encode()).hexdigest()[:10]
    run_dir = exp_dir / "runs" / run_id

    result = await run_codex(
        workspace=workspace,
        prompt=task_prompt,
        run_dir=run_dir,
        run_id=run_id,
        model=settings.codex_model,
        resume_thread_id=resume_thread_id or None,
        timeout_s=settings.codex_timeout_s,
    )

    budget_totals: dict[str, int] = {}
    if not result.cached:
        budget_totals = await _update_budget(
            tool_context,
            {
                "run_id": run_id,
                "thread_id": result.thread_id,
                "branch": "main",
                "iteration": 1,
                "status": result.status,
                "usage": result.usage,
                "wallclock_s": result.wallclock_s,
                "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            },
        )
        # Small, durable run summary in the catalog (raw JSONL stays on disk).
        await tool_context.save_artifact(
            f"iter_1/exp_main/runs/{run_id}/result.json",
            types.Part(text=json.dumps(
                {k: v for k, v in result.__dict__.items() if k != "cached"},
                indent=2,
            )),
            custom_metadata={
                "kind": "run_log",
                "title": f"Codex run {run_id}",
                "summary": f"{result.status}; metrics: {json.dumps(result.metrics)[:200]}",
                "run_id": run_id,
                "branch": "main",
                "iteration": 1,
                "created_by": tool_context.agent_name,
            },
        )

    return {
        "status": result.status,
        "run_id": run_id,
        "thread_id": result.thread_id,
        "final_message": (result.final_message or "")[:2000],
        "metrics": result.metrics,
        "usage": result.usage,
        "wallclock_s": result.wallclock_s,
        "error": result.error,
        "budget_totals": budget_totals,
        "cached": result.cached,
    }
