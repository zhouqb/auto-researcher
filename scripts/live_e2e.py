"""Live Phase 0 exit-criterion check (needs DEEPSEEK_API_KEY).

Drives a real project through: question → clarifying answers → plan approval
→ literature pipeline → cited report. Run: uv run python scripts/live_e2e.py
"""

from __future__ import annotations

import asyncio
import sys
import time

from deep_researcher.config import get_settings
from deep_researcher.runner import (
    build_runner, event_text, event_tool_calls, run_turn,
)
from deep_researcher.storage import ArtifactCatalog

PROJECT = sys.argv[1] if len(sys.argv) > 1 else f"e2e-{int(time.time())}"
QUESTION = (
    "What routing strategies make Mixture-of-Experts models efficient at "
    "inference time, and what are the main open problems?"
)
TURNS = [
    QUESTION,
    "Scope: inference-time efficiency only (not training). Success criterion: "
    "a well-cited survey-style report. Depth: concise but thorough. "
    "Otherwise use sensible defaults.",
    "I approve the plan. Proceed.",
]


async def main() -> None:
    settings = get_settings()
    runner = build_runner()
    print(f"project={PROJECT}\nartifacts={settings.projects_dir / PROJECT}\n")

    for i, msg in enumerate(TURNS, 1):
        print(f"\n=== user turn {i}: {msg[:80]}…")
        t0 = time.time()
        async for ev in run_turn(runner, PROJECT, msg):
            for tool in event_tool_calls(ev):
                print(f"  [{ev.author} → {tool}]")
            text = event_text(ev)
            if text and not ev.partial:
                print(f"  {ev.author}: {text[:400].replace(chr(10), ' ')}")
        print(f"  (turn took {time.time() - t0:.0f}s)")

    catalog = ArtifactCatalog(settings.db_path)
    paths = catalog.list_paths(project_id=PROJECT)
    print("\n=== artifacts ===")
    for p in paths:
        print(" ", p)
    report = settings.projects_dir / PROJECT / "reports/final_report.md"
    ok = report.exists()
    print(f"\nreport exists: {ok}")
    if ok:
        body = report.read_text()
        print(f"report length: {len(body)} chars; 'References' section: {'References' in body}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    asyncio.run(main())
