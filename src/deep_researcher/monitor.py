"""Run-monitor helpers: read Codex run state for a project from disk.

The UI tails ``codex_events.jsonl`` files directly (design §12: run monitor),
so live visibility never depends on the agent event stream or the DB.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .codex import ParsedEvents, parse_event_line
from .codex.runner import EVENTS_FILE, RESULT_MARKER
from .config import get_settings
from .storage.jobs import JobsStore


@dataclass
class RunInfo:
    run_id: str
    experiment: str  # e.g. "iter_1/exp_main"
    status: str  # running | completed | failed | timeout
    thread_id: Optional[str] = None
    usage: dict[str, int] = field(default_factory=dict)
    wallclock_s: float = 0.0
    commands: list[str] = field(default_factory=list)
    files_changed: list[str] = field(default_factory=list)
    last_message: Optional[str] = None
    metrics: Optional[dict[str, Any]] = None
    events_path: Optional[Path] = None


# (mtime_ns, size) → ParsedEvents per file: the UIs poll every few seconds,
# and event files for long runs reach MBs — only re-parse when the file grew.
_events_cache: dict[Path, tuple[tuple[int, int], ParsedEvents]] = {}


def _parse_events_file(path: Path) -> ParsedEvents:
    if not path.exists():
        return ParsedEvents()
    stat = path.stat()
    stamp = (stat.st_mtime_ns, stat.st_size)
    cached = _events_cache.get(path)
    if cached is not None and cached[0] == stamp:
        return cached[1]
    acc = ParsedEvents()
    for line in path.read_text(errors="replace").splitlines():
        parse_event_line(line, acc)
    _events_cache[path] = (stamp, acc)
    return acc


def kill_run(project_id: str, run_id: str) -> bool:
    """Kill one branch's running Codex process group (design §11.2)."""
    return JobsStore(get_settings().db_path).kill(f"{project_id}:{run_id}")


def list_runs(project_id: str) -> list[RunInfo]:
    project_dir = get_settings().projects_dir / project_id
    job_status = {
        j.run_id: j.status
        for j in JobsStore(get_settings().db_path).for_project(project_id)
    }
    runs: list[RunInfo] = []
    for run_dir in sorted(project_dir.glob("iter_*/exp_*/runs/*")):
        if not run_dir.is_dir():
            continue
        experiment = run_dir.parent.parent.relative_to(project_dir).as_posix()
        marker = run_dir / RESULT_MARKER
        events_path = run_dir / EVENTS_FILE
        acc = _parse_events_file(events_path)
        info = RunInfo(
            run_id=run_dir.name,
            experiment=experiment,
            status=job_status.get(run_dir.name, "running"),
            thread_id=acc.thread_id,
            usage=acc.usage,
            commands=acc.commands,
            files_changed=acc.files_changed,
            last_message=acc.agent_messages[-1] if acc.agent_messages else None,
            events_path=events_path,
        )
        if marker.exists():
            try:
                result = json.loads(marker.read_text())
                # Precedence: an explicit user kill (jobs table) always wins
                # over the marker the process wrote before dying — same rule
                # as tools/codex._run_branch.
                if job_status.get(run_dir.name) != "killed":
                    info.status = result.get("status", "completed")
                info.wallclock_s = result.get("wallclock_s", 0.0)
                info.metrics = result.get("metrics")
                info.usage = result.get("usage") or info.usage
            except json.JSONDecodeError:
                pass
        runs.append(info)
    return runs


def load_budget(project_id: str) -> Optional[dict[str, Any]]:
    path = get_settings().projects_dir / project_id / "budget/budget.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
