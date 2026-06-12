"""Codex runner tests with a fake `codex` binary (no quota, no network)."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from deep_researcher.codex import (
    ParsedEvents,
    parse_event_line,
    prepare_workspace,
    run_codex,
)

pytestmark = pytest.mark.asyncio

# Verbatim line shapes from codex-cli 0.136 --json output.
SAMPLE_EVENTS = [
    '{"type":"thread.started","thread_id":"019eb8b2-d716-7642-8315-4ca7806b27ba"}',
    '{"type":"turn.started"}',
    '{"type":"item.completed","item":{"id":"item_0","type":"agent_message","text":"Working on it."}}',
    '{"type":"item.completed","item":{"id":"item_1","type":"file_change","changes":[{"path":"/ws/hello.txt","kind":"add"}],"status":"completed"}}',
    '{"type":"item.completed","item":{"id":"item_2","type":"command_execution","command":"python run.py","exit_code":0}}',
    '{"type":"turn.completed","usage":{"input_tokens":29079,"cached_input_tokens":16640,"output_tokens":79}}',
]


def test_parse_event_lines():
    acc = ParsedEvents()
    for line in SAMPLE_EVENTS + ["not json", ""]:
        parse_event_line(line, acc)
    assert acc.thread_id == "019eb8b2-d716-7642-8315-4ca7806b27ba"
    assert acc.agent_messages == ["Working on it."]
    assert acc.commands == ["python run.py"]
    assert acc.files_changed == ["/ws/hello.txt"]
    assert acc.usage["input_tokens"] == 29079 and acc.usage["output_tokens"] == 79
    assert not acc.turn_failed


def _fake_codex(tmp_path: Path, body: str) -> Path:
    """Install a fake `codex` shell script and point CODEX_BINARY at it."""
    script = tmp_path / "fake_codex"
    script.write_text("#!/bin/sh\n" + body)
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    os.environ["CODEX_BINARY"] = str(script)
    return script


@pytest.fixture(autouse=True)
def _restore_codex_binary():
    yield
    os.environ.pop("CODEX_BINARY", None)


async def test_run_codex_success_and_idempotency(tmp_path):
    ws = tmp_path / "ws"
    run_dir = tmp_path / "runs" / "r1"
    counter = tmp_path / "invocations"
    _fake_codex(tmp_path, f"""
echo run >> {counter}
# find the -C and -o argument values
while [ $# -gt 0 ]; do
  case "$1" in
    -C) WS="$2"; shift 2;;
    -o) OUT="$2"; shift 2;;
    *) shift;;
  esac
done
cat <<'EOF'
{SAMPLE_EVENTS[0]}
{SAMPLE_EVENTS[2]}
{SAMPLE_EVENTS[5]}
EOF
echo '{{"metric": "acc", "value": 0.9, "baseline": 0.8}}' > "$WS/metrics.json"
echo "All done." > "$OUT"
""")
    result = await run_codex(
        workspace=ws, prompt="do the thing", run_dir=run_dir, run_id="r1"
    )
    assert result.status == "completed"
    assert result.thread_id == "019eb8b2-d716-7642-8315-4ca7806b27ba"
    assert result.final_message == "All done."
    assert result.metrics == {"metric": "acc", "value": 0.9, "baseline": 0.8}
    assert result.usage["input_tokens"] == 29079
    assert not result.cached
    assert (run_dir / "codex_events.jsonl").exists()
    assert json.loads((run_dir / "result.json").read_text())["status"] == "completed"

    # Second call: cached, no second invocation of the binary (idempotent).
    again = await run_codex(
        workspace=ws, prompt="do the thing", run_dir=run_dir, run_id="r1"
    )
    assert again.cached and again.status == "completed"
    assert counter.read_text().count("run") == 1


async def test_run_codex_failure(tmp_path):
    _fake_codex(tmp_path, """
echo '{"type":"thread.started","thread_id":"t-fail"}'
echo '{"type":"turn.failed","error":{"message":"model refused"}}'
exit 1
""")
    result = await run_codex(
        workspace=tmp_path / "ws", prompt="x",
        run_dir=tmp_path / "runs" / "rf", run_id="rf",
    )
    assert result.status == "failed"
    assert "model refused" in (result.error or "")
    assert result.thread_id == "t-fail"


async def test_run_codex_timeout(tmp_path):
    _fake_codex(tmp_path, "sleep 30\n")
    result = await run_codex(
        workspace=tmp_path / "ws", prompt="x",
        run_dir=tmp_path / "runs" / "rt", run_id="rt", timeout_s=1,
    )
    assert result.status == "timeout"


def test_prepare_workspace_idempotent(tmp_path):
    ws = tmp_path / "repo"
    prepare_workspace(ws)
    assert (ws / "AGENTS.md").exists() and (ws / ".git").exists()
    marker = (ws / "AGENTS.md").read_text()
    prepare_workspace(ws)  # no clobber
    assert (ws / "AGENTS.md").read_text() == marker


async def test_stderr_flood_does_not_deadlock(tmp_path):
    """Child writing > pipe-buffer to stderr must not hang the run (regression)."""
    _fake_codex(tmp_path, """
# 256KB of stderr before any stdout: deadlocks unless stderr is drained
i=0
while [ $i -lt 64 ]; do
  head -c 4096 /dev/zero | tr '\\0' 'e' >&2
  i=$((i+1))
done
echo '{"type":"thread.started","thread_id":"t-stderr"}'
echo '{"type":"turn.completed","usage":{"input_tokens":5,"output_tokens":1}}'
""")
    import time
    t0 = time.time()
    result = await run_codex(
        workspace=tmp_path / "ws", prompt="x",
        run_dir=tmp_path / "runs" / "rs", run_id="rs", timeout_s=20,
    )
    assert result.status == "completed"
    assert result.thread_id == "t-stderr"
    assert time.time() - t0 < 15, "run should finish promptly, not hit timeout"
