"""Phase 3a tests: parallel branches, concurrency cap, jobs table, kill-branch."""

from __future__ import annotations

import asyncio
import json
import os
import stat

import pytest
from google.adk.apps import App
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from scripted_llm import SCRIPTS, ScriptedLlm, _call, _text, patch_models

import deep_researcher.config as config_mod
import deep_researcher.tools.codex as codex_tools
from deep_researcher.agents import build_root_agent
from deep_researcher.storage import ArtifactCatalog, LocalArtifactService
from deep_researcher.storage.jobs import JobsStore

pytestmark = pytest.mark.asyncio


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-dummy")
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("MAX_CODEX_CONCURRENCY", "2")
    config_mod.get_settings.cache_clear()
    codex_tools._jobs.cache_clear()
    settings = config_mod.get_settings()
    yield settings, tmp_path
    config_mod.get_settings.cache_clear()
    codex_tools._jobs.cache_clear()


def _install_fake_codex(tmp_path, monkeypatch, *, sleep_s: float = 0.4):
    """Fake codex that records concurrent invocations and emits metrics."""
    gauge = tmp_path / "gauge"
    gauge.mkdir(exist_ok=True)
    fake = tmp_path / "fake_codex"
    fake.write_text(f"""#!/bin/sh
while [ $# -gt 0 ]; do
  case "$1" in
    -C) WS="$2"; shift 2;;
    -o) OUT="$2"; shift 2;;
    *) LAST="$1"; shift;;
  esac
done
# track concurrency via marker files
M="{gauge}/$$"
touch "$M"
N=$(ls {gauge} | wc -l | tr -d ' ')
echo "$N" >> "{gauge}.max"
sleep {sleep_s}
rm -f "$M"
B=$(basename "$WS" | sed 's/repo//')
echo '{{"type":"thread.started","thread_id":"t-'$$'"}}'
echo '{{"type":"turn.completed","usage":{{"input_tokens":100,"output_tokens":10}}}}'
echo '{{"metric": "score", "value": 0.5, "baseline": 0.4, "ws": "'$B'"}}' > "$WS/metrics.json"
echo "branch done" > "$OUT"
""")
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("CODEX_BINARY", str(fake))
    return gauge


def _make_runner(settings):
    root = build_root_agent()
    patch_models(root, ScriptedLlm(model="scripted"))
    catalog = ArtifactCatalog(settings.db_path)
    return Runner(
        app=App(name=settings.app_name, root_agent=root),
        session_service=InMemorySessionService(),
        artifact_service=LocalArtifactService(settings.root, catalog),
    ), catalog


async def _send(runner, msg, session_id):
    events = []
    async for ev in runner.run_async(
        user_id="local", session_id=session_id,
        new_message=types.Content(role="user", parts=[types.Part(text=msg)]),
    ):
        events.append(ev)
    return events


async def test_run_experiments_parallel_with_concurrency_cap(env, monkeypatch):
    settings, tmp_path = env
    gauge = _install_fake_codex(tmp_path, monkeypatch)

    SCRIPTS.clear()
    SCRIPTS.update({
        "orchestrator": [
            [_call("run_experiments", branches=[
                {"branch_id": "B1", "task_prompt": "run experiment one"},
                {"branch_id": "B2", "task_prompt": "run experiment two"},
                {"branch_id": "B3", "task_prompt": "run experiment three"},
            ])],
            [_text("All branches finished.")],
        ],
    })
    runner, catalog = _make_runner(settings)
    await runner.session_service.create_session(
        app_name=settings.app_name, user_id="local", session_id="proj-par"
    )
    events = await _send(runner, "run them", "proj-par")

    responses = [
        p.function_response.response
        for ev in events if ev.content and ev.content.parts
        for p in ev.content.parts
        if p.function_response and p.function_response.name == "run_experiments"
    ]
    assert responses
    results = responses[0]["results"]
    assert sorted(r["branch"] for r in results) == ["B1", "B2", "B3"]
    assert all(r["status"] == "completed" for r in results)
    # each branch ran in its own workspace
    for b in ("B1", "B2", "B3"):
        ws = settings.projects_dir / "proj-par" / "iter_1" / f"exp_{b}" / "repo"
        assert json.loads((ws / "metrics.json").read_text())["value"] == 0.5

    # concurrency never exceeded the cap of 2
    peaks = [int(x) for x in (str(gauge) + ".max") and open(str(gauge) + ".max").read().split()]
    assert max(peaks) <= 2

    # budget aggregated across the three branches
    budget = json.loads(
        (settings.projects_dir / "proj-par" / "budget/budget.json").read_text()
    )
    assert budget["totals"]["input_tokens"] == 300
    assert {e["branch"] for e in budget["entries"]} == {"B1", "B2", "B3"}

    # jobs table reached terminal state for all three
    jobs = JobsStore(settings.db_path).for_project("proj-par")
    assert len(jobs) == 3 and all(j.status == "completed" for j in jobs)


async def test_run_experiments_input_validation(env, monkeypatch):
    settings, tmp_path = env
    _install_fake_codex(tmp_path, monkeypatch)
    from deep_researcher.tools.codex import run_experiments

    class FakeCtx:
        agent_name = "test"
        class session:
            id = "proj-x"

    assert "error" in await run_experiments([], FakeCtx())
    assert "error" in await run_experiments(
        [{"branch_id": "A", "task_prompt": "x"}, {"branch_id": "A", "task_prompt": "y"}],
        FakeCtx(),
    )
    assert "error" in await run_experiments([{"branch_id": "A"}], FakeCtx())


def test_jobs_kill_marks_killed(env):
    settings, _ = env
    store = JobsStore(settings.db_path)
    # no real process: pid absent → kill() still flips status
    store.start(project_id="p", branch="B1", run_id="r1")
    assert store.get("p:r1").status == "running"
    store.kill("p:r1")
    assert store.get("p:r1").status == "killed"
    # killing a terminal job is a no-op
    assert store.kill("p:r1") is False


async def test_kill_branch_terminates_real_process(env, monkeypatch, tmp_path):
    settings, _ = env
    fake = tmp_path / "slow_codex"
    fake.write_text("#!/bin/sh\nsleep 30\n")
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("CODEX_BINARY", str(fake))

    from deep_researcher.tools.codex import codex_exec

    class FakeCtx:
        agent_name = "test"
        class session:
            id = "proj-kill"

        @staticmethod
        async def load_artifact(*a, **k):
            return None

        @staticmethod
        async def save_artifact(*a, **k):
            return 0

    task = asyncio.create_task(
        codex_exec("long task", FakeCtx(), branch_id="B1")
    )
    store = JobsStore(settings.db_path)
    for _ in range(100):
        await asyncio.sleep(0.05)
        jobs = store.for_project("proj-kill")
        if jobs and jobs[0].pid:
            break
    assert jobs and jobs[0].pid, "job should register its pid"
    assert store.kill(jobs[0].job_id) is True

    result = await asyncio.wait_for(task, timeout=10)
    assert result["status"] == "killed"
    assert store.get(jobs[0].job_id).status == "killed"
