"""Codex CLI driver (design §13): `codex exec --json` subprocess + JSONL parsing.

Each run streams raw JSONL events to ``<runs_dir>/<run_id>/codex_events.jsonl``
(high-volume output stays on disk, never in the DB) and finishes by writing a
``result.json`` marker. The marker doubles as the idempotency check: calling
``run_codex`` again with the same run_id returns the cached result instead of
launching a duplicate paid run (design §16, non-negotiable for resumability).

Event schema (verified against codex-cli 0.136):
    {"type":"thread.started","thread_id":"..."}
    {"type":"item.completed","item":{"type":"agent_message"|"file_change"|"command_execution",...}}
    {"type":"turn.completed","usage":{"input_tokens":...,"cached_input_tokens":...,"output_tokens":...}}
    {"type":"turn.failed", ...}
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

RESULT_MARKER = "result.json"
EVENTS_FILE = "codex_events.jsonl"
METRICS_FILE = "metrics.json"


@dataclass
class CodexRunResult:
    status: str  # 'completed' | 'failed' | 'timeout'
    run_id: str
    thread_id: Optional[str] = None
    final_message: Optional[str] = None
    metrics: Optional[dict[str, Any]] = None
    usage: dict[str, int] = field(default_factory=dict)
    wallclock_s: float = 0.0
    exit_code: Optional[int] = None
    error: Optional[str] = None
    events_path: Optional[str] = None
    cached: bool = False


@dataclass
class ParsedEvents:
    thread_id: Optional[str] = None
    agent_messages: list[str] = field(default_factory=list)
    commands: list[str] = field(default_factory=list)
    files_changed: list[str] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    turn_failed: bool = False
    error: Optional[str] = None


def parse_event_line(line: str, acc: ParsedEvents) -> None:
    """Fold one JSONL event into the accumulator (tolerant of non-JSON noise)."""
    line = line.strip()
    if not line.startswith("{"):
        return
    try:
        ev = json.loads(line)
    except json.JSONDecodeError:
        return
    etype = ev.get("type")
    if etype == "thread.started":
        acc.thread_id = ev.get("thread_id")
    elif etype == "turn.completed":
        for k, v in (ev.get("usage") or {}).items():
            if isinstance(v, int):
                acc.usage[k] = acc.usage.get(k, 0) + v
    elif etype == "turn.failed":
        acc.turn_failed = True
        acc.error = json.dumps(ev.get("error") or ev)[:500]
    elif etype == "item.completed":
        item = ev.get("item") or {}
        itype = item.get("type")
        if itype == "agent_message" and item.get("text"):
            acc.agent_messages.append(item["text"])
        elif itype == "command_execution":
            acc.commands.append(item.get("command", ""))
        elif itype == "file_change":
            acc.files_changed.extend(
                c.get("path", "") for c in item.get("changes") or []
            )


def read_cached_result(run_dir: Path) -> Optional[CodexRunResult]:
    marker = run_dir / RESULT_MARKER
    if marker.exists():
        data = json.loads(marker.read_text())
        data["cached"] = True
        return CodexRunResult(**data)
    return None


def _codex_binary() -> str:
    return os.environ.get("CODEX_BINARY", "codex")


async def run_codex(
    *,
    workspace: Path,
    prompt: str,
    run_dir: Path,
    run_id: str,
    model: Optional[str] = None,
    resume_thread_id: Optional[str] = None,
    timeout_s: float = 3600,
) -> CodexRunResult:
    """Run one non-interactive Codex turn, sandboxed to the workspace."""
    cached = read_cached_result(run_dir)
    if cached is not None:
        return cached

    workspace.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    events_path = run_dir / EVENTS_FILE
    last_msg_path = run_dir / "last_message.txt"

    cmd = [_codex_binary(), "exec"]
    if resume_thread_id:
        cmd += ["resume", resume_thread_id]
    cmd += [
        "--json",
        "--sandbox", "workspace-write",
        "--skip-git-repo-check",
        "-C", str(workspace),
        "-o", str(last_msg_path),
    ]
    if model:
        cmd += ["--model", model]
    cmd += [prompt]

    acc = ParsedEvents()
    t0 = time.time()
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,  # own process group → clean kill on timeout
    )

    async def _consume() -> None:
        with events_path.open("a") as sink:
            assert proc.stdout is not None
            async for raw in proc.stdout:
                line = raw.decode("utf-8", errors="replace")
                sink.write(line)
                sink.flush()
                parse_event_line(line, acc)

    status = "completed"
    error: Optional[str] = None
    try:
        await asyncio.wait_for(_consume(), timeout=timeout_s)
        await proc.wait()
    except asyncio.TimeoutError:
        status = "timeout"
        error = f"Codex run exceeded {timeout_s}s; process group killed."
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
        await proc.wait()

    stderr_tail = ""
    if proc.stderr is not None:
        stderr_tail = (await proc.stderr.read()).decode("utf-8", errors="replace")[-2000:]

    if status == "completed" and (proc.returncode != 0 or acc.turn_failed):
        status = "failed"
        error = acc.error or stderr_tail.strip()[-500:] or f"exit code {proc.returncode}"

    final_message = None
    if last_msg_path.exists():
        final_message = last_msg_path.read_text().strip() or None
    if final_message is None and acc.agent_messages:
        final_message = acc.agent_messages[-1]

    metrics = None
    metrics_path = workspace / METRICS_FILE
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text())
        except json.JSONDecodeError:
            metrics = {"error": "metrics.json is not valid JSON"}

    result = CodexRunResult(
        status=status,
        run_id=run_id,
        thread_id=acc.thread_id,
        final_message=final_message,
        metrics=metrics,
        usage=acc.usage,
        wallclock_s=round(time.time() - t0, 1),
        exit_code=proc.returncode,
        error=error,
        events_path=str(events_path),
    )
    payload = asdict(result)
    payload.pop("cached", None)
    (run_dir / RESULT_MARKER).write_text(json.dumps(payload, indent=2))
    return result
