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
import logging
import os
import signal
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

RESULT_MARKER = "result.json"
EVENTS_FILE = "codex_events.jsonl"
METRICS_FILE = "metrics.json"
OUTCOME_FILE = "outcome.json"  # repo-improvement mode's result file

logger = logging.getLogger(__name__)


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
    # repo-improvement mode: project-relative path to the saved change diff
    change_diff_path: Optional[str] = None
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
    on_spawn=None,  # Callable[[pid, pgid], None] — lets the jobs table track the process
) -> CodexRunResult:
    """Run one non-interactive Codex turn, sandboxed to the workspace."""
    cached = read_cached_result(run_dir)
    if cached is not None:
        logger.info("codex run %s: returning cached result (%s)", run_id, cached.status)
        return cached
    logger.info("codex run %s starting (workspace=%s, resume=%s)",
                run_id, workspace, resume_thread_id or "-")

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
    if on_spawn is not None:
        try:
            on_spawn(proc.pid, os.getpgid(proc.pid))
        except ProcessLookupError:
            pass

    async def _consume_stdout() -> None:
        with events_path.open("a") as sink:
            assert proc.stdout is not None
            async for raw in proc.stdout:
                line = raw.decode("utf-8", errors="replace")
                sink.write(line)
                sink.flush()
                parse_event_line(line, acc)

    # stderr must be drained CONCURRENTLY: a child filling the ~64KB pipe
    # buffer would otherwise block and hang the run until timeout.
    stderr_chunks: deque[bytes] = deque(maxlen=64)  # keep only the tail

    async def _consume_stderr() -> None:
        assert proc.stderr is not None
        while chunk := await proc.stderr.read(4096):
            stderr_chunks.append(chunk)

    status = "completed"
    error: Optional[str] = None
    try:
        await asyncio.wait_for(
            asyncio.gather(_consume_stdout(), _consume_stderr()), timeout=timeout_s
        )
        await proc.wait()
    except asyncio.TimeoutError:
        status = "timeout"
        error = f"Codex run exceeded {timeout_s}s; process group killed."
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
        await proc.wait()

    stderr_tail = b"".join(stderr_chunks).decode("utf-8", errors="replace")[-2000:]

    if status == "completed" and (proc.returncode != 0 or acc.turn_failed):
        if proc.returncode in (-signal.SIGTERM, -signal.SIGKILL):
            status = "killed"
            error = "Run was killed (kill-branch or shutdown)."
        else:
            status = "failed"
            error = acc.error or stderr_tail.strip()[-500:] or f"exit code {proc.returncode}"

    final_message = None
    if last_msg_path.exists():
        final_message = last_msg_path.read_text().strip() or None
    if final_message is None and acc.agent_messages:
        final_message = acc.agent_messages[-1]

    # The run's comparable result: outcome.json in repo mode, else metrics.json.
    metrics = None
    for fname in (OUTCOME_FILE, METRICS_FILE):
        result_path = workspace / fname
        if result_path.exists():
            try:
                metrics = json.loads(result_path.read_text())
            except json.JSONDecodeError:
                metrics = {"error": f"{fname} is not valid JSON"}
            break

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
    logger.info("codex run %s finished: %s in %.0fs (usage=%s)%s",
                run_id, result.status, result.wallclock_s, result.usage,
                f" error={result.error}" if result.error else "")
    return result
