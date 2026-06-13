"""Workspace setup: greenfield AGENTS.md and repo-improvement seeding."""

from __future__ import annotations

import subprocess
from pathlib import Path

from deep_researcher.codex import prepare_workspace, seed_sha
from deep_researcher.codex.workspace import (
    CONTRACT_FILE,
    OUTCOME_FILE,
    SEED_MARKER,
)


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def _make_source_repo(path: Path) -> Path:
    path.mkdir(parents=True)
    (path / "app.py").write_text("def add(a, b):\n    return a + b\n")
    (path / "AGENTS.md").write_text("# Repo's own conventions\nUse black.\n")
    _git(path, "init", "-q")
    _git(path, "config", "user.email", "t@t.io")
    _git(path, "config", "user.name", "t")
    _git(path, "add", "-A")
    _git(path, "commit", "-qm", "initial")
    return path


def test_greenfield_workspace_unchanged(tmp_path):
    ws = tmp_path / "ws"
    prepare_workspace(ws)
    assert (ws / "AGENTS.md").read_text().startswith("# Experiment conventions")
    assert (ws / "plots").is_dir()
    assert (ws / ".git").is_dir()
    # no repo-mode scaffolding leaks into the greenfield path
    assert not (ws / SEED_MARKER).exists()
    assert seed_sha(ws) is None


def test_repo_workspace_seeds_from_source_and_preserves_history(tmp_path):
    source = _make_source_repo(tmp_path / "source")
    source_head = _git(source, "rev-parse", "HEAD")

    ws = tmp_path / "exp" / "repo"
    prepare_workspace(ws, source_repo=source, test_command="pytest -q")

    # seeded content + the repo's OWN AGENTS.md is preserved (not clobbered)
    assert (ws / "app.py").exists()
    assert (ws / "AGENTS.md").read_text().startswith("# Repo's own conventions")
    # history carried over → clean base for diffing
    assert _git(ws, "rev-parse", "HEAD") == source_head
    assert seed_sha(ws) == source_head
    # our contract sidecar exists and names the test command
    assert "pytest -q" in (ws / CONTRACT_FILE).read_text()


def test_scaffolding_is_git_excluded(tmp_path):
    source = _make_source_repo(tmp_path / "source")
    ws = tmp_path / "exp" / "repo"
    prepare_workspace(ws, source_repo=source, test_command="pytest")
    _git(ws, "config", "user.email", "t@t.io")
    _git(ws, "config", "user.name", "t")

    # simulate the coding agent writing outcome.json then committing everything
    (ws / OUTCOME_FILE).write_text('{"green": true}')
    (ws / "app.py").write_text("def add(a, b):\n    return a + b + 0\n")
    _git(ws, "add", "-A")
    _git(ws, "commit", "-qm", "change")

    committed = _git(ws, "show", "--name-only", "--format=", "HEAD").split()
    # the real change is committed; none of our scaffolding is
    assert "app.py" in committed
    for scaffold in (OUTCOME_FILE, CONTRACT_FILE, SEED_MARKER):
        assert scaffold not in committed
    # status is clean (excludes hide the untracked sidecars)
    assert _git(ws, "status", "--porcelain") == ""


def test_repo_seeding_is_idempotent(tmp_path):
    source = _make_source_repo(tmp_path / "source")
    ws = tmp_path / "exp" / "repo"
    prepare_workspace(ws, source_repo=source, test_command="pytest")

    # coding agent edits a file; a second prepare (resume/rerun) must not clobber it
    (ws / "app.py").write_text("EDITED")
    prepare_workspace(ws, source_repo=source, test_command="pytest")
    assert (ws / "app.py").read_text() == "EDITED"


def test_repo_workspace_non_git_source(tmp_path):
    source = tmp_path / "plain"
    source.mkdir()
    (source / "main.py").write_text("print('hi')\n")

    ws = tmp_path / "exp" / "repo"
    prepare_workspace(ws, source_repo=source, test_command="pytest")
    assert (ws / "main.py").exists()
    assert (ws / ".git").is_dir()  # git-init'd so we can still diff
    assert seed_sha(ws) is not None


def test_guess_kind_diff():
    from deep_researcher.storage.catalog import guess_kind

    assert guess_kind("iter_1/exp_main/change.diff") == "diff"
    assert guess_kind("x.patch") == "diff"
    assert guess_kind("reports/final_report.md") == "report"
    assert guess_kind("iter_1/whatever.md") == "other"
