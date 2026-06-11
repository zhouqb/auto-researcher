"""Live Phase 3a check: idea tournament + parallel Codex branches + ranking.

Needs DEEPSEEK_API_KEY + authenticated Codex CLI. Run:
    uv run python scripts/live_e2e_parallel.py [project_id]
"""

from __future__ import annotations

import asyncio
import sys
import time

from deep_researcher.config import get_settings
from deep_researcher.monitor import list_runs, load_budget
from deep_researcher.runner import build_runner, event_text, event_tool_calls, run_turn
from deep_researcher.storage import ExperienceStore

PROJECT = sys.argv[1] if len(sys.argv) > 1 else f"e2e-par-{int(time.time())}"
TURNS = [
    "Under a fixed evaluation budget, which simple gradient-free optimizer "
    "minimizes classic 2D test functions best: random search, simulated "
    "annealing, or a (1+1) evolution strategy? Please compare them with "
    "parallel branch experiments.",

    "Scope: a concise cited report plus branch experiments. Run the idea "
    "tournament, then design up to 3 branches — one per optimizer is the "
    "obvious split. Constraints: Python STDLIB ONLY (sandbox has no network, "
    "no pip installs); shared metric: median best f-value on Rosenbrock and "
    "Rastrigin after 2000 evaluations across 5 seeded restarts (lower is "
    "better); baseline: random search; under 3 minutes wallclock per branch; "
    "plots as vega.json only (no matplotlib). Otherwise sensible defaults.",

    "I approve the plan. Proceed.",

    "I approve the experiment budget. Proceed with all branches.",
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
                print(f"  {ev.author}: {text[:280].replace(chr(10), ' ')}", flush=True)
        print(f"  (turn took {time.time() - t0:.0f}s)")

    print("\n=== runs ===")
    runs = list_runs(PROJECT)
    for run in runs:
        print(f"  {run.experiment}/{run.run_id}: {run.status}, metrics="
              f"{str(run.metrics)[:120]}")
    completed = [r for r in runs if r.status == "completed" and r.metrics]

    budget = load_budget(PROJECT)
    print(f"budget totals: {budget['totals'] if budget else None}")

    experiences = ExperienceStore(settings.db_path).recent(project_id=PROJECT, k=10)
    print(f"experiences recorded: {len(experiences)} "
          f"({[e.outcome for e in experiences]})")

    proj = settings.projects_dir / PROJECT
    checks = {
        "hypotheses_json": (proj / "iter_1/hypotheses.json").exists(),
        "exp_spec": (proj / "iter_1/exp_spec.md").exists(),
        ">=2 parallel branches completed with metrics": len(completed) >= 2,
        "distinct branch workspaces": len({r.experiment for r in completed}) >= 2,
        "analysis": (proj / "iter_1/analysis.md").exists(),
        "report": (proj / "reports/final_report.md").exists(),
        "budget multiple entries": bool(budget) and len(budget["entries"]) >= 2,
        "gate2_checkpoint": any(
            p.name.endswith("_budget_approval.json")
            for p in (proj / "checkpoints").glob("*.json")
        ) if (proj / "checkpoints").exists() else False,
        "experiences >=2": len(experiences) >= 2,
    }
    print("\n=== exit criterion ===")
    for name, passed in checks.items():
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
    sys.exit(0 if all(checks.values()) else 1)


if __name__ == "__main__":
    asyncio.run(main())
