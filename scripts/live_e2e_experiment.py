"""Live Phase 1 exit-criterion check: real Codex experiment with budget gate.

Needs DEEPSEEK_API_KEY + an authenticated Codex CLI. Run:
    uv run python scripts/live_e2e_experiment.py [project_id]
"""

from __future__ import annotations

import asyncio
import sys
import time

from deep_researcher.config import get_settings
from deep_researcher.monitor import list_runs, load_budget
from deep_researcher.runner import build_runner, event_text, event_tool_calls, run_turn

PROJECT = sys.argv[1] if len(sys.argv) > 1 else f"e2e-exp-{int(time.time())}"
TURNS = [
    "How do quasi-random (low-discrepancy) sequences compare to pseudo-random "
    "sampling for Monte Carlo integration accuracy? Please include a small "
    "code experiment.",

    "Scope: a concise cited report plus one small experiment. Experiment "
    "constraints: Python STDLIB ONLY (the sandbox has no network, so no pip "
    "installs), wallclock under 3 minutes, e.g. compare Halton-sequence vs "
    "uniform pseudo-random Monte Carlo error when integrating 1-2 smooth test "
    "functions in 2D, across sample sizes and 3 seeds; metric: RMSE vs known "
    "analytic integral. Plots: vega.json only, skip PNG (no matplotlib). "
    "Otherwise use sensible defaults.",

    "I approve the plan. Proceed.",

    "I approve the experiment budget. Proceed.",
]


async def main() -> None:
    settings = get_settings()
    runner = build_runner()
    print(f"project={PROJECT}\nartifacts={settings.projects_dir / PROJECT}\n")

    for i, msg in enumerate(TURNS, 1):
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
        print(f"  {run.experiment}/{run.run_id}: {run.status}, "
              f"usage={run.usage}, metrics={run.metrics}")
        ok_run = ok_run or (run.status == "completed" and run.metrics is not None)

    budget = load_budget(PROJECT)
    print(f"budget totals: {budget['totals'] if budget else None}")

    proj = settings.projects_dir / PROJECT
    checks = {
        "exp_spec": (proj / "iter_1/exp_spec.md").exists(),
        "analysis": (proj / "iter_1/analysis.md").exists(),
        "report": (proj / "reports/final_report.md").exists(),
        "budget": budget is not None and bool(budget.get("entries")),
        "gate2_checkpoint": any(
            p.name.endswith("_budget_approval.json")
            for p in (proj / "checkpoints").glob("*.json")
        ) if (proj / "checkpoints").exists() else False,
        "completed_run_with_metrics": ok_run,
        "workspace_metrics_json": bool(list(proj.glob("iter_1/exp_main/repo/metrics.json"))),
    }
    print("\n=== exit criterion ===")
    for name, passed in checks.items():
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
    sys.exit(0 if all(checks.values()) else 1)


if __name__ == "__main__":
    asyncio.run(main())
