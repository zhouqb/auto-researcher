"""Experiment workspace setup: AGENTS.md contract + git init (design §13)."""

from __future__ import annotations

import subprocess
from pathlib import Path

# Kept deliberately small — AGENTS.md consumes Codex context on every turn.
AGENTS_MD = """\
# Experiment conventions

You are implementing one research experiment inside this workspace.

## Contract (mandatory)
- Write final results to `metrics.json` at the workspace root:
  `{"metric": "<name>", "value": <number>, "baseline": <number or null>,
    "n_seeds": <int>, "details": {...}}`
- Every plot is emitted twice: `plots/<name>.vega.json` (Vega-Lite spec) and
  `plots/<name>.png`.
- Commit after each logical step with a clear message (`git add -A && git commit`).
- Pin random seeds; record them in metrics.json `details.seeds`.
- Write dependencies to `requirements.txt` if you add any.

## Forbidden
- Destructive operations outside this workspace; network access unless the
  task says otherwise; fabricating numbers — if a run fails, report the
  failure honestly in metrics.json (`{"error": "..."}`).
"""


def prepare_workspace(workspace: Path) -> None:
    """Idempotent: creates the directory, AGENTS.md, plots/, and a git repo."""
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "plots").mkdir(exist_ok=True)
    agents = workspace / "AGENTS.md"
    if not agents.exists():
        agents.write_text(AGENTS_MD)
    if not (workspace / ".git").exists():
        subprocess.run(
            ["git", "init", "-q"], cwd=workspace, check=False, capture_output=True
        )
        subprocess.run(
            ["git", "add", "-A"], cwd=workspace, check=False, capture_output=True
        )
        subprocess.run(
            ["git", "commit", "-qm", "init workspace"],
            cwd=workspace, check=False, capture_output=True,
        )
