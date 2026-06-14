"""Repo-improvement codex flow: seed from a repo, edit, surface diff + outcome."""

from __future__ import annotations

import json
import stat
import subprocess
import types as pytypes
from pathlib import Path

import pytest

import deep_researcher.config as config_mod
import deep_researcher.tools.codex as codex_tools

pytestmark = pytest.mark.asyncio


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-dummy")
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data"))
    config_mod.get_settings.cache_clear()
    codex_tools._jobs.cache_clear()
    yield config_mod.get_settings(), tmp_path
    config_mod.get_settings.cache_clear()
    codex_tools._jobs.cache_clear()


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def _source_repo(path: Path) -> Path:
    path.mkdir(parents=True)
    (path / "app.py").write_text("def add(a, b):\n    return a + b\n")
    _git(path, "init", "-q")
    _git(path, "config", "user.email", "t@t.io")
    _git(path, "config", "user.name", "t")
    _git(path, "add", "-A")
    _git(path, "commit", "-qm", "init")
    return path


def _fake_codex(tmp_path: Path, monkeypatch) -> None:
    fake = tmp_path / "fake_codex"
    fake.write_text(
        "#!/bin/sh\n"
        'while [ $# -gt 0 ]; do case "$1" in -C) WS="$2"; shift 2;; '
        '-o) OUT="$2"; shift 2;; *) shift;; esac; done\n'
        'echo \'{"type":"thread.started","thread_id":"t-repo"}\'\n'
        'echo \'{"type":"turn.completed","usage":{"input_tokens":5,"output_tokens":2}}\'\n'
        'printf "\\n# improved\\n" >> "$WS/app.py"\n'
        'echo \'{"approach":"B1","changed_files":["app.py"],'
        '"tests":{"command":"pytest","passed":1,"total":1,"failures":[],"green":true},'
        '"acceptance_met":true,"summary":"tweak"}\' > "$WS/outcome.json"\n'
        'echo done > "$OUT"\n'
    )
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("CODEX_BINARY", str(fake))


class _Ctx:
    agent_name = "tester"

    def __init__(self, repo: Path):
        self.state = {"mode": "repo_improvement",
                      "target_repo_path": str(repo),
                      "repo_test_command": "pytest -q"}
        self.session = pytypes.SimpleNamespace(id="proj-repo")
        self.saved: dict = {}

    async def load_artifact(self, filename, *a, **k):
        text = self.saved.get(filename)
        return pytypes.SimpleNamespace(text=text) if text is not None else None

    async def save_artifact(self, filename, part, custom_metadata=None):
        self.saved[filename] = part.text
        return 0


async def test_repo_branch_seeds_edits_and_produces_diff(env, monkeypatch, tmp_path):
    source = _source_repo(tmp_path / "source")
    _fake_codex(tmp_path, monkeypatch)
    ctx = _Ctx(source)

    out = await codex_tools.codex_exec("improve add()", ctx, branch_id="B1")

    assert out["status"] == "completed"
    # outcome.json surfaced as the run's comparable result
    assert out["metrics"]["acceptance_met"] is True
    assert out["metrics"]["tests"]["green"] is True
    # a change diff was produced and saved
    assert out["change_diff_path"] == "iter_1/exp_B1/change.diff"
    diff = ctx.saved["iter_1/exp_B1/change.diff"]
    assert "app.py" in diff and "# improved" in diff
    # the user's ORIGINAL repo is untouched (experiment ran on a clone)
    assert "# improved" not in (source / "app.py").read_text()


def _fake_codex_with_diagnosis(tmp_path: Path, monkeypatch) -> None:
    """Fake that edits + writes outcome.json normally, but on a `resume` turn
    (the analyzer) writes diagnosis.json and leaves the code untouched."""
    fake = tmp_path / "fake_codex"
    fake.write_text(
        "#!/bin/sh\n"
        "MODE=exp\n"
        'while [ $# -gt 0 ]; do case "$1" in -C) WS="$2"; shift 2;; '
        '-o) OUT="$2"; shift 2;; resume) MODE=analysis; shift;; *) shift;; esac; done\n'
        'WS="${WS:-$PWD}"\n'  # resume passes no -C → codex runs with cwd=workspace
        'echo \'{"type":"thread.started","thread_id":"t-repo"}\'\n'
        'echo \'{"type":"turn.completed","usage":{"input_tokens":5,"output_tokens":2}}\'\n'
        'if [ "$MODE" = "analysis" ]; then\n'
        '  echo \'{"branch":"B1","verdict":"success","root_cause":"add was correct",'
        '"evidence":["pytest green"],"key_factors":["minimal"],"next_step":"ship"}\''
        ' > "$WS/diagnosis.json"\n'
        '  echo "Diagnosis: it passed." > "$OUT"\n'
        "else\n"
        '  printf "\\n# improved\\n" >> "$WS/app.py"\n'
        '  echo \'{"approach":"B1","changed_files":["app.py"],'
        '"tests":{"command":"pytest","passed":1,"total":1,"failures":[],"green":true},'
        '"acceptance_met":true,"summary":"tweak"}\' > "$WS/outcome.json"\n'
        '  echo done > "$OUT"\n'
        "fi\n"
    )
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("CODEX_BINARY", str(fake))


async def test_analyze_experiment_diagnoses_without_touching_diff(
    env, monkeypatch, tmp_path
):
    source = _source_repo(tmp_path / "source")
    _fake_codex_with_diagnosis(tmp_path, monkeypatch)
    ctx = _Ctx(source)

    run = await codex_tools.codex_exec("improve add()", ctx, branch_id="B1")
    diff_before = ctx.saved["iter_1/exp_B1/change.diff"]

    out = await codex_tools.analyze_experiment(
        branch_id="B1", resume_thread_id=run["thread_id"], tool_context=ctx
    )

    # structured diagnosis surfaced + saved
    assert out["status"] == "completed"
    assert out["verdict"] == "success"
    assert out["diagnosis_path"] == "iter_1/exp_B1/diagnosis.md"
    assert "add was correct" in ctx.saved["iter_1/exp_B1/diagnosis.md"]
    # the analysis turn must NOT have rewritten the branch's change diff
    assert ctx.saved["iter_1/exp_B1/change.diff"] == diff_before
    # diagnosis.json never reaches the clone's commit/diff (git-excluded);
    # the original repo is still untouched
    assert "# improved" not in (source / "app.py").read_text()
    # the diagnosis turn is billed as its own 'analysis' budget entry
    ledger = json.loads(ctx.saved["budget/budget.json"])
    kinds = [e.get("kind") for e in ledger["entries"]]
    assert "analysis" in kinds and "experiment" in kinds


async def test_analyze_experiment_requires_thread_id(env, monkeypatch, tmp_path):
    source = _source_repo(tmp_path / "source")
    _fake_codex(tmp_path, monkeypatch)
    ctx = _Ctx(source)
    out = await codex_tools.analyze_experiment(
        branch_id="B1", resume_thread_id="", tool_context=ctx
    )
    assert "error" in out


async def test_research_mode_unaffected(env, monkeypatch, tmp_path):
    """No target repo in state → greenfield workspace, no diff artifact."""
    fake = tmp_path / "fake_codex"
    fake.write_text(
        "#!/bin/sh\n"
        'while [ $# -gt 0 ]; do case "$1" in -C) WS="$2"; shift 2;; '
        '-o) OUT="$2"; shift 2;; *) shift;; esac; done\n'
        'echo \'{"type":"thread.started","thread_id":"t1"}\'\n'
        'echo \'{"type":"turn.completed","usage":{"input_tokens":1,"output_tokens":1}}\'\n'
        'echo \'{"metric":"acc","value":0.9}\' > "$WS/metrics.json"\n'
        'echo done > "$OUT"\n'
    )
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("CODEX_BINARY", str(fake))

    class ResearchCtx(_Ctx):
        def __init__(self):
            self.state = {}  # no repo → research mode
            self.session = pytypes.SimpleNamespace(id="proj-research")
            self.agent_name = "tester"
            self.saved = {}

    ctx = ResearchCtx()
    out = await codex_tools.codex_exec("run experiment", ctx, branch_id="main")
    assert out["status"] == "completed"
    assert out["metrics"]["metric"] == "acc"
    assert "change_diff_path" not in out
    assert not any(k.endswith("change.diff") for k in ctx.saved)
