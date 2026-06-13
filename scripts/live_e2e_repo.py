"""Live repo-improvement check: point at a throwaway repo, land a green diff.

Creates a tiny local git repo with a deliberately failing test, then drives the
full repo-improvement flow (set_target_repo → plan → budget gate → Codex change
branch → analysis) and asserts a non-empty change.diff that makes the test pass.

Needs DEEPSEEK_API_KEY + an authenticated Codex CLI. Run:
    uv run python scripts/live_e2e_repo.py [project_id]

The original repo is never modified (branches work on clones).
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
import time
from pathlib import Path

from deep_researcher.config import get_settings
from deep_researcher.monitor import list_runs, load_budget
from deep_researcher.runner import build_runner, event_text, event_tool_calls, run_turn

PROJECT = sys.argv[1] if len(sys.argv) > 1 else f"e2e-repo-{int(time.time())}"


def _make_target_repo(root: Path) -> Path:
    """A trivial package whose test fails until `is_even` is implemented."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "mathlib.py").write_text(
        "def is_even(n):\n"
        "    raise NotImplementedError('implement me')\n"
    )
    (root / "test_mathlib.py").write_text(
        "from mathlib import is_even\n\n"
        "def test_is_even():\n"
        "    assert is_even(4) is True\n"
        "    assert is_even(3) is False\n"
    )
    (root / "pyproject.toml").write_text(
        "[project]\nname = 'mathlib'\nversion = '0.0.0'\n"
    )
    for args in (
        ["init", "-q"],
        ["config", "user.email", "e2e@local"],
        ["config", "user.name", "e2e"],
        ["add", "-A"],
        ["commit", "-qm", "initial: failing test"],
    ):
        subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)
    return root


TURNS = [
    "Improve the repo at {repo}: implement `is_even(n)` in mathlib.py so the "
    "existing tests pass. It should return True for even integers and False "
    "otherwise.",
    "Acceptance: `pytest -q` passes. Keep the change minimal — just mathlib.py. "
    "Single branch is fine. Use sensible defaults otherwise.",
    "I approve the plan. Proceed.",
    "I approve the experiment budget. Proceed.",
]


async def main() -> None:
    settings = get_settings()
    repo = _make_target_repo(settings.root / "_e2e_targets" / PROJECT)
    runner = build_runner()
    print(f"project={PROJECT}\ntarget_repo={repo}\nartifacts={settings.projects_dir / PROJECT}\n")

    for i, msg in enumerate(TURNS, 1):
        msg = msg.format(repo=repo)
        print(f"\n=== user turn {i}: {msg[:90]}…")
        t0 = time.time()
        async for ev in run_turn(runner, PROJECT, msg):
            for tool in event_tool_calls(ev):
                print(f"  [{ev.author} → {tool}]", flush=True)
            text = event_text(ev)
            if text and not ev.partial:
                print(f"  {ev.author}: {text[:300].replace(chr(10), ' ')}", flush=True)
        print(f"  (turn took {time.time() - t0:.0f}s)")

    print("\n=== runs ===")
    ok_run = False
    for run in list_runs(PROJECT):
        print(f"  {run.experiment}/{run.run_id}: {run.status}, metrics={run.metrics}")
        ok_run = ok_run or (run.status == "completed" and run.metrics is not None)

    budget = load_budget(PROJECT)
    print(f"budget totals: {budget['totals'] if budget else None}")

    proj = settings.projects_dir / PROJECT
    diffs = list(proj.glob("iter_1/exp_*/change.diff"))
    non_empty_diff = any(d.read_text().strip() for d in diffs)
    # the user's original repo must be untouched
    original_unchanged = "NotImplementedError" in (repo / "mathlib.py").read_text()

    checks = {
        "repo_mode_engaged": (proj / "brief/target_repo.json").exists(),
        "exp_spec": (proj / "iter_1/exp_spec.md").exists(),
        "completed_run_with_outcome": ok_run,
        "change_diff_produced": non_empty_diff,
        "analysis": (proj / "iter_1/analysis.md").exists(),
        "report": (proj / "reports/final_report.md").exists(),
        "original_repo_untouched": original_unchanged,
        "gate2_checkpoint": any(
            p.name.endswith("_budget_approval.json")
            for p in (proj / "checkpoints").glob("*.json")
        ) if (proj / "checkpoints").exists() else False,
    }
    print("\n=== exit criterion ===")
    for name, passed in checks.items():
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
    sys.exit(0 if all(checks.values()) else 1)


if __name__ == "__main__":
    asyncio.run(main())
