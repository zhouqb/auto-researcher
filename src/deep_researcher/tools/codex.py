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
import subprocess
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from google.adk.tools import ToolContext
from google.genai import types

from ..codex import CodexRunResult, prepare_workspace, run_codex, seed_sha
from ..codex.runner import RESULT_MARKER
from ..codex.workspace import DIAGNOSIS_FILE
from ..config import get_settings
from ..notify import notify
from ..storage.jobs import JobsStore

BUDGET_FILE = "budget/budget.json"

# A diagnosis turn resumes the branch's own Codex session (it already has the
# full implementation context: every command, test output, and edit). It must
# investigate WITHOUT changing code — only diagnosis.json (git-excluded) is
# written, so the branch's change.diff stays clean.
_ANALYSIS_PROMPT = """\
You just implemented an experiment in this workspace. Now act as a skeptical \
reviewer of your own work and diagnose WHY it turned out the way it did. Do NOT \
modify any source file and do NOT commit.

1. Establish the outcome from the evidence you already have: re-read your result \
   file (outcome.json for a repo change, metrics.json for a research \
   experiment); re-run the test/experiment command ONCE if it helps confirm. \
   Decide the verdict: success | failure | underperformed | inconclusive.
2. Find the DECISIVE root cause — the single factor that best explains the \
   outcome. Cite concrete evidence: the specific failing test + error, the \
   metric vs baseline, the key line(s) you changed, or the assumption that \
   broke. Separate the root cause from incidental symptoms. If a \
   `.dr_langfuse.py` helper and an `eval/out/*/cases.jsonl` dump exist, use \
   them to inspect WHY specific cases failed (run `python .dr_langfuse.py \
   --failed` then `--case <id>` for the agent's real trajectory).
3. Name the single most promising next step (a fix to try, a stronger baseline, \
   a confound to control).
4. Write your diagnosis to `diagnosis.json` at the workspace root — it is \
   git-excluded, so do NOT commit it, and do NOT edit any other file:
   {"branch": "<id>", "verdict": "...", "root_cause": "<1-2 sentences>",
    "evidence": ["..."], "key_factors": ["..."], "next_step": "..."}

Finish with a 4-6 sentence plain-language summary. If the outcome is genuinely \
ambiguous, say so (verdict "inconclusive") — never fabricate a clean story."""


def _repo_context(tool_context: ToolContext) -> tuple[Optional[str], Optional[str]]:
    """(source_repo_path, test_command) from session state; (None, None) in research mode."""
    state = getattr(tool_context, "state", None) or {}
    return state.get("target_repo_path"), state.get("repo_test_command")


def _compute_change_diff(workspace: Path) -> Optional[str]:
    """Full change vs the seed commit (committed + uncommitted, scaffolding excluded)."""
    seed = seed_sha(workspace)
    if not seed:
        return None
    # stage everything (git-excluded scaffolding stays out) so the diff captures
    # the change whether or not the coding agent committed it.
    subprocess.run(["git", "add", "-A"], cwd=workspace, check=False, capture_output=True)
    out = subprocess.run(
        ["git", "diff", "--cached", seed],
        cwd=workspace, check=False, capture_output=True, text=True,
    )
    return out.stdout if out.stdout.strip() else None


@lru_cache(maxsize=1)
def _jobs() -> JobsStore:
    return JobsStore(get_settings().db_path)


def _safe_branch(branch_id: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", branch_id).strip("-") or "main"
    return slug[:24]


def _project_id(tool_context: ToolContext) -> str:
    return tool_context.session.id


def _iteration(tool_context: ToolContext) -> int:
    """Current optimization iteration from session state (1-based); the
    orchestrator bumps it via advance_iteration between rounds."""
    state = getattr(tool_context, "state", None) or {}
    try:
        return max(1, int(state.get("iteration", 1)))
    except (TypeError, ValueError):
        return 1


def _branch_workspace(project_id: str, branch: str, iteration: int = 1) -> Path:
    """Deterministic path to a branch's persistent Codex workspace."""
    return (get_settings().projects_dir / project_id / f"iter_{iteration}"
            / f"exp_{branch}" / "repo")


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
    is_analysis: bool = False,
) -> tuple[CodexRunResult, str]:
    """Run one branch's Codex turn; returns (result, run_id).

    ``is_analysis`` marks a post-run diagnosis turn: it resumes the branch's
    session but must not alter the recorded change, so the change.diff save is
    skipped (the diagnosis writes only git-excluded diagnosis.json).
    """
    settings = get_settings()
    project_id = _project_id(tool_context)
    iteration = _iteration(tool_context)
    branch = _safe_branch(branch_id)
    source_repo, test_command = _repo_context(tool_context)
    exp_dir = settings.projects_dir / project_id / f"iter_{iteration}" / f"exp_{branch}"
    workspace = exp_dir / "repo"
    link_paths = [p.strip() for p in (settings.repo_data_links or "").split(",")
                  if p.strip()]
    await asyncio.to_thread(  # git ops off the loop
        prepare_workspace,
        workspace,
        Path(source_repo) if source_repo else None,
        test_command,
        eval_command=settings.repo_eval_command or None,
        objective_metric=settings.objective_metric,
        link_paths=link_paths or None,
    )

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
        if source_repo and result.status == "completed" and not is_analysis:
            await _save_change_diff(tool_context, workspace, branch, result, iteration)
    return result, run_id


async def _save_change_diff(
    tool_context: ToolContext, workspace: Path, branch: str,
    result: CodexRunResult, iteration: int = 1
) -> None:
    """Save the branch's change vs the seed as iter_<N>/exp_<branch>/change.diff."""
    diff = await asyncio.to_thread(_compute_change_diff, workspace)
    if not diff:
        return
    path = f"iter_{iteration}/exp_{branch}/change.diff"
    await tool_context.save_artifact(
        path,
        types.Part(text=diff),
        custom_metadata={
            "kind": "diff",
            "title": f"Change diff ({branch})",
            "summary": f"{diff.count(chr(10))} lines; {len(diff)} chars",
            "branch": branch,
            "created_by": tool_context.agent_name,
        },
    )
    result.change_diff_path = path


def _budget_entry(
    branch: str, run_id: str, result: CodexRunResult, kind: str = "experiment",
    iteration: int = 1,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "thread_id": result.thread_id,
        "branch": branch,
        "iteration": iteration,
        "kind": kind,  # "experiment" | "analysis"
        "status": result.status,
        "usage": result.usage,
        "wallclock_s": result.wallclock_s,
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


async def _register_run_artifact(
    tool_context: ToolContext, branch: str, run_id: str, result: CodexRunResult
) -> None:
    iteration = _iteration(tool_context)
    await tool_context.save_artifact(
        f"iter_{iteration}/exp_{branch}/runs/{run_id}/result.json",
        types.Part(text=json.dumps(
            {k: v for k, v in result.__dict__.items() if k != "cached"}, indent=2
        )),
        custom_metadata={
            "kind": "run_log",
            "title": f"Codex run {run_id} ({branch})",
            "summary": f"{result.status}; metrics: {json.dumps(result.metrics)[:200]}",
            "run_id": run_id,
            "branch": branch,
            "iteration": iteration,
            "created_by": tool_context.agent_name,
        },
    )


def _result_payload(branch: str, run_id: str, result: CodexRunResult) -> dict[str, Any]:
    payload = {
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
    if result.change_diff_path:  # repo-improvement mode
        payload["change_diff_path"] = result.change_diff_path
    return payload


async def codex_exec(
    task_prompt: str,
    tool_context: ToolContext,
    branch_id: str = "main",
    resume_thread_id: Optional[str] = None,
) -> dict[str, Any]:
    """Execute ONE code-experiment branch with Codex in a sandboxed workspace.

    Call this AFTER the user approved the experiment budget (Gate 2). Each
    branch has a persistent workspace at iter_<N>/exp_<branch_id>/repo; pass
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
            tool_context,
            [_budget_entry(branch, run_id, result, iteration=_iteration(tool_context))],
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

    iteration = _iteration(tool_context)
    entries = [
        _budget_entry(branch, run_id, result, iteration=iteration)
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


async def _save_diagnosis(
    tool_context: ToolContext, branch: str, result: CodexRunResult
) -> dict[str, Any]:
    """Read the analysis turn's diagnosis.json (fallback: final message) and
    save it as iter_<N>/exp_<branch>/diagnosis.md. Returns {verdict, root_cause}."""
    iteration = _iteration(tool_context)
    workspace = _branch_workspace(_project_id(tool_context), branch, iteration)
    diag_path = workspace / DIAGNOSIS_FILE
    verdict, root_cause = None, None
    if diag_path.exists():
        body = await asyncio.to_thread(diag_path.read_text)
        try:
            parsed = json.loads(body)
            verdict = parsed.get("verdict")
            root_cause = parsed.get("root_cause")
        except json.JSONDecodeError:
            pass
        content = f"```json\n{body.strip()}\n```\n"
        if result.final_message:
            content += f"\n## Summary\n\n{result.final_message}\n"
    elif result.final_message:
        content = result.final_message  # no structured file → keep the prose
    else:
        return {"verdict": None, "root_cause": None, "diagnosis_path": None}

    path = f"iter_{iteration}/exp_{branch}/diagnosis.md"
    await tool_context.save_artifact(
        path,
        types.Part(text=content),
        custom_metadata={
            "kind": "analysis",
            "title": f"Diagnosis ({branch})",
            "summary": (root_cause or result.final_message or "diagnosis")[:200],
            "branch": branch,
            "created_by": tool_context.agent_name,
        },
    )
    return {"verdict": verdict, "root_cause": root_cause, "diagnosis_path": path}


async def analyze_experiment(
    branch_id: str,
    resume_thread_id: str,
    tool_context: ToolContext,
) -> dict[str, Any]:
    """Diagnose WHY one experiment branch worked or not, by resuming its Codex session.

    Call this AFTER a branch's run completes (before result_analyst), once per
    branch. It resumes the branch's own Codex thread — inheriting the full
    implementation context (commands, test output, edits) — and asks it to
    diagnose the outcome WITHOUT changing code, writing a root-cause analysis to
    iter_<N>/exp_<branch_id>/diagnosis.md. Idempotent and budgeted like any run.

    Args:
        branch_id: The branch to diagnose, e.g. "main", "B1".
        resume_thread_id: That branch's thread_id (from its run result) — the
            session to resume. Required; without it there is no context to reuse.

    Returns:
        {branch, status, run_id, thread_id, verdict, root_cause,
         diagnosis_path, budget_totals, cached}
    """
    if not resume_thread_id:
        return {"error": "resume_thread_id is required to reuse the branch's session"}
    result, run_id = await _run_branch(
        tool_context, _ANALYSIS_PROMPT, branch_id, resume_thread_id, is_analysis=True
    )
    branch = _safe_branch(branch_id)
    budget_totals: dict[str, int] = {}
    if not result.cached:
        budget_totals = await _update_budget(
            tool_context,
            [_budget_entry(branch, run_id, result, kind="analysis",
                           iteration=_iteration(tool_context))],
        )
        await _register_run_artifact(tool_context, branch, run_id, result)
    diagnosis = await _save_diagnosis(tool_context, branch, result)
    return {
        "branch": branch,
        "status": result.status,
        "run_id": run_id,
        "thread_id": result.thread_id,
        "budget_totals": budget_totals,
        "cached": result.cached,
        **diagnosis,
    }


def _commit_all(repo: Path, message: str) -> None:
    """Commit any pending change (git-excluded scaffolding stays out). Harmless
    no-op when the tree is already clean."""
    subprocess.run(["git", "add", "-A"], cwd=repo, check=False, capture_output=True)
    subprocess.run(
        ["git", "-c", "user.email=auto-researcher@local",
         "-c", "user.name=auto-researcher", "commit", "-m", message],
        cwd=repo, check=False, capture_output=True,
    )


async def advance_iteration(
    winner_branch_id: str, tool_context: ToolContext
) -> dict[str, Any]:
    """Carry the winning branch forward as the baseline for the NEXT iteration.

    Call this once per iteration, AFTER the user approves the winner. It commits
    the winning branch's change, repoints the project at that workspace (so the
    next iteration's branches clone the winner and build ON it), and bumps the
    iteration counter. Subsequent run_experiments/analysis then write under
    iter_<N+1>/ automatically.

    Args:
        winner_branch_id: The branch chosen by result_analyst / the user.

    Returns:
        {previous_iteration, new_iteration, baseline_repo, winner} or {error}.
    """
    state = getattr(tool_context, "state", None)
    if state is None:
        return {"error": "no session state available"}
    iteration = _iteration(tool_context)
    winner = _safe_branch(winner_branch_id)
    winner_repo = _branch_workspace(_project_id(tool_context), winner, iteration)
    if not (winner_repo / ".git").is_dir():
        return {"error": f"no workspace for winner '{winner}' at iter_{iteration}"}

    # Commit the winning change so a clone of this repo carries it as the seed.
    await asyncio.to_thread(_commit_all, winner_repo, f"iter {iteration} winner: {winner}")

    new_iteration = iteration + 1
    state["iteration"] = new_iteration
    state["target_repo_path"] = str(winner_repo)

    record = {
        "previous_iteration": iteration,
        "new_iteration": new_iteration,
        "winner": winner,
        "baseline_repo": str(winner_repo),
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    await tool_context.save_artifact(
        f"decisions/iter_{iteration}_winner.json",
        types.Part(text=json.dumps(record, indent=2)),
        custom_metadata={
            "kind": "decision",
            "title": f"Iteration {iteration} winner: {winner}",
            "summary": f"baseline for iter {new_iteration} → {winner_repo}",
            "iteration": iteration,
            "created_by": tool_context.agent_name,
        },
    )
    return record
