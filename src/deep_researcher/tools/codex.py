"""Codex experiment tools (design §13 + §4 parallel branches).

`codex_exec` runs ONE branch (also the fix-loop entry via resume_thread_id);
`run_experiments` fans out several branches concurrently, capped by
`max_codex_concurrency`. Both are idempotent (run_id = prompt hash → cached
result markers) and update the `budget/budget.json` ledger. Every launch is
registered in the jobs table so the UI can watch and kill branches.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Optional

from google.adk.tools import ToolContext
from google.genai import types

from ..codex import CodexRunResult, prepare_workspace, run_codex
from ..codex.runner import RESULT_MARKER
from ..config import get_settings
from ..notify import notify
from ..storage.jobs import JobsStore

BUDGET_FILE = "budget/budget.json"


@lru_cache(maxsize=1)
def _jobs() -> JobsStore:
    return JobsStore(get_settings().db_path)


def _safe_branch(branch_id: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", branch_id).strip("-") or "main"
    return slug[:24]


def _project_id(tool_context: ToolContext) -> str:
    return tool_context.session.id


async def _update_budget(
    tool_context: ToolContext, entries: list[dict[str, Any]]
) -> dict[str, int]:
    existing = await tool_context.load_artifact(BUDGET_FILE)
    ledger = (
        json.loads(existing.text)
        if existing and existing.text
        else {"entries": [], "totals": {}}
    )
    ledger["entries"].extend(entries)
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


async def _run_branch(
    tool_context: ToolContext,
    task_prompt: str,
    branch_id: str,
    resume_thread_id: Optional[str] = None,
) -> tuple[CodexRunResult, str]:
    """Run one branch's Codex turn; returns (result, run_id)."""
    settings = get_settings()
    project_id = _project_id(tool_context)
    branch = _safe_branch(branch_id)
    exp_dir = settings.projects_dir / project_id / "iter_1" / f"exp_{branch}"
    workspace = exp_dir / "repo"
    await asyncio.to_thread(prepare_workspace, workspace)  # git ops off the loop

    seed = f"{branch}|{task_prompt}|{resume_thread_id or ''}"
    run_id = hashlib.sha1(seed.encode()).hexdigest()[:10]
    run_dir = exp_dir / "runs" / run_id
    if branch == "main" and not (run_dir / RESULT_MARKER).exists():
        # Pre-3a markers were keyed without the branch; honor them so an ADK
        # resume never re-launches a paid run that already completed (§16).
        legacy_id = hashlib.sha1(
            f"{task_prompt}|{resume_thread_id or ''}".encode()
        ).hexdigest()[:10]
        legacy_dir = exp_dir / "runs" / legacy_id
        if (legacy_dir / RESULT_MARKER).exists():
            run_id, run_dir = legacy_id, legacy_dir
    job_id = f"{project_id}:{run_id}"

    def on_spawn(pid: int, pgid: int) -> None:
        _jobs().start(
            project_id=project_id, branch=branch, run_id=run_id, pid=pid, pgid=pgid
        )

    result = await run_codex(
        workspace=workspace,
        prompt=task_prompt,
        run_dir=run_dir,
        run_id=run_id,
        model=settings.codex_model,
        resume_thread_id=resume_thread_id or None,
        timeout_s=settings.codex_timeout_s,
        on_spawn=on_spawn,
    )
    if not result.cached:
        # Precedence rule: an explicit user kill always wins over whatever
        # the process managed to write before dying (single rule, also
        # applied by monitor.list_runs).
        job = _jobs().get(job_id)
        if job is not None and job.status == "killed":
            result.status = "killed"
        else:
            _jobs().start(project_id=project_id, branch=branch, run_id=run_id)
            _jobs().finish(job_id, result.status)
        await asyncio.to_thread(  # osascript/webhook off the loop
            notify,
            f"Experiment {branch}: {result.status}",
            f"{project_id} · run {run_id} · {result.wallclock_s:.0f}s",
        )
    return result, run_id


def _budget_entry(branch: str, run_id: str, result: CodexRunResult) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "thread_id": result.thread_id,
        "branch": branch,
        "iteration": 1,
        "status": result.status,
        "usage": result.usage,
        "wallclock_s": result.wallclock_s,
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


async def _register_run_artifact(
    tool_context: ToolContext, branch: str, run_id: str, result: CodexRunResult
) -> None:
    await tool_context.save_artifact(
        f"iter_1/exp_{branch}/runs/{run_id}/result.json",
        types.Part(text=json.dumps(
            {k: v for k, v in result.__dict__.items() if k != "cached"}, indent=2
        )),
        custom_metadata={
            "kind": "run_log",
            "title": f"Codex run {run_id} ({branch})",
            "summary": f"{result.status}; metrics: {json.dumps(result.metrics)[:200]}",
            "run_id": run_id,
            "branch": branch,
            "iteration": 1,
            "created_by": tool_context.agent_name,
        },
    )


def _result_payload(branch: str, run_id: str, result: CodexRunResult) -> dict[str, Any]:
    return {
        "branch": branch,
        "status": result.status,
        "run_id": run_id,
        "thread_id": result.thread_id,
        "final_message": (result.final_message or "")[:1500],
        "metrics": result.metrics,
        "usage": result.usage,
        "wallclock_s": result.wallclock_s,
        "error": result.error,
        "cached": result.cached,
    }


async def codex_exec(
    task_prompt: str,
    tool_context: ToolContext,
    branch_id: str = "main",
    resume_thread_id: Optional[str] = None,
) -> dict[str, Any]:
    """Execute ONE code-experiment branch with Codex in a sandboxed workspace.

    Call this AFTER the user approved the experiment budget (Gate 2). Each
    branch has a persistent workspace at iter_1/exp_<branch_id>/repo; pass
    resume_thread_id (from a previous result) to continue that branch's Codex
    conversation with full context — preferred when debugging a failed run.
    For several independent branches, use run_experiments instead.

    Args:
        task_prompt: Complete, self-contained experiment instructions for
            Codex: hypothesis, method, dataset, baseline, metric, and the
            requirement to write metrics.json per the workspace AGENTS.md.
        branch_id: Short branch slug, e.g. "main", "B1", "halton".
        resume_thread_id: thread_id of a prior run to continue (fix loop).

    Returns:
        {branch, status, run_id, thread_id, final_message, metrics, usage,
         wallclock_s, error, budget_totals, cached}
    """
    result, run_id = await _run_branch(
        tool_context, task_prompt, branch_id, resume_thread_id
    )
    branch = _safe_branch(branch_id)
    budget_totals: dict[str, int] = {}
    if not result.cached:
        budget_totals = await _update_budget(
            tool_context, [_budget_entry(branch, run_id, result)]
        )
        await _register_run_artifact(tool_context, branch, run_id, result)
    payload = _result_payload(branch, run_id, result)
    payload["budget_totals"] = budget_totals
    return payload


async def run_experiments(
    branches: list[dict], tool_context: ToolContext
) -> dict[str, Any]:
    """Run SEVERAL independent experiment branches concurrently with Codex.

    Call this AFTER the user approved the total budget (Gate 2). Branches run
    in parallel, capped at max_codex_concurrency. Each branch gets its own
    isolated workspace; results come back ranked-ready (metrics per branch).

    Args:
        branches: 2-4 items, each {"branch_id": "B1", "task_prompt": "..."}.
            Every task_prompt must be complete and self-contained (the
            implementing agents cannot see this conversation or each other).

    Returns:
        {"results": [{branch, status, metrics, ...} per branch],
         "budget_totals": {...}}
    """
    settings = get_settings()
    if not branches:
        return {"error": "branches must be a non-empty list"}
    seen: set[str] = set()
    for b in branches:
        slug = _safe_branch(b.get("branch_id", ""))
        if not b.get("task_prompt"):
            return {"error": f"branch {slug!r} is missing task_prompt"}
        if slug in seen:
            return {"error": f"duplicate branch_id {slug!r}"}
        seen.add(slug)

    semaphore = asyncio.Semaphore(settings.max_codex_concurrency)

    async def _one(spec: dict) -> tuple[CodexRunResult, str, str]:
        async with semaphore:
            result, run_id = await _run_branch(
                tool_context, spec["task_prompt"], spec["branch_id"]
            )
            return result, run_id, _safe_branch(spec["branch_id"])

    outcomes = await asyncio.gather(*(_one(b) for b in branches))

    entries = [
        _budget_entry(branch, run_id, result)
        for result, run_id, branch in outcomes
        if not result.cached
    ]
    budget_totals = await _update_budget(tool_context, entries) if entries else {}
    for result, run_id, branch in outcomes:
        if not result.cached:
            await _register_run_artifact(tool_context, branch, run_id, result)

    return {
        "results": [
            _result_payload(branch, run_id, result)
            for result, run_id, branch in outcomes
        ],
        "budget_totals": budget_totals,
    }
