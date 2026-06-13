"""tools/repo.py: target-repo resolution, test detection, scoped reads."""

from __future__ import annotations

import subprocess
import types as pytypes
from pathlib import Path

import pytest

import deep_researcher.config as config_mod
from deep_researcher.tools.repo import (
    list_repo_tree,
    read_repo_file,
    set_target_repo,
)

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def fresh_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data"))
    config_mod.get_settings.cache_clear()
    yield
    config_mod.get_settings.cache_clear()


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def _make_repo(path: Path, files: dict[str, str]) -> Path:
    path.mkdir(parents=True)
    for rel, content in files.items():
        f = path / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(content)
    _git(path, "init", "-q")
    _git(path, "config", "user.email", "t@t.io")
    _git(path, "config", "user.name", "t")
    _git(path, "add", "-A")
    _git(path, "commit", "-qm", "init")
    return path


class _Ctx:
    agent_name = "tester"

    def __init__(self):
        self.state: dict = {}
        self.session = pytypes.SimpleNamespace(id="proj-x")
        self.saved: dict = {}

    async def save_artifact(self, filename, part, custom_metadata=None):
        self.saved[filename] = part.text
        return 0


async def test_local_path_sets_repo_mode_and_detects_pytest(tmp_path):
    repo = _make_repo(tmp_path / "src_repo", {
        "pyproject.toml": "[project]\nname='x'\n",
        "tests/test_x.py": "def test_x():\n    assert True\n",
    })
    ctx = _Ctx()
    out = await set_target_repo(str(repo), ctx)
    assert out["mode"] == "repo_improvement"
    assert out["repo_test_command"] == "pytest -q"
    assert ctx.state["mode"] == "repo_improvement"
    assert ctx.state["target_repo_path"] == str(repo.resolve())
    assert "brief/target_repo.json" in ctx.saved


async def test_url_like_path_is_cloned(tmp_path):
    # a path ending in .git is treated as a URL → cloned into source_repo/
    source = _make_repo(tmp_path / "origin.git", {"app.py": "x = 1\n"})
    ctx = _Ctx()
    out = await set_target_repo(str(source), ctx)
    assert out["mode"] == "repo_improvement"
    cloned = Path(out["target_repo_path"])
    assert cloned.name == "source_repo"
    assert (cloned / "app.py").exists()
    assert "proj-x" in cloned.as_posix()  # cloned under this project's dir


async def test_explicit_test_command_wins(tmp_path):
    repo = _make_repo(tmp_path / "r", {"pyproject.toml": "[project]\nname='x'\n"})
    ctx = _Ctx()
    out = await set_target_repo(str(repo), ctx, test_command="make check")
    assert out["repo_test_command"] == "make check"


async def test_npm_detection(tmp_path):
    repo = _make_repo(tmp_path / "r", {
        "package.json": '{"scripts": {"test": "vitest"}}',
    })
    ctx = _Ctx()
    out = await set_target_repo(str(repo), ctx)
    assert out["repo_test_command"] == "npm test"


async def test_missing_local_path_errors(tmp_path):
    ctx = _Ctx()
    out = await set_target_repo(str(tmp_path / "nope"), ctx)
    assert "error" in out
    assert "mode" not in ctx.state


async def test_read_and_list_are_scoped(tmp_path):
    repo = _make_repo(tmp_path / "r", {
        "src/main.py": "print('hi')\n",
        "node_modules/junk.js": "noise\n",
    })
    ctx = _Ctx()
    await set_target_repo(str(repo), ctx)

    got = await read_repo_file("src/main.py", ctx)
    assert "print('hi')" in got["content"]

    tree = await list_repo_tree(ctx, ".", depth=3)
    assert "src/" in tree["entries"] and "src/main.py" in tree["entries"]
    assert not any("node_modules" in e for e in tree["entries"])  # noise skipped


async def test_path_escape_is_rejected(tmp_path):
    repo = _make_repo(tmp_path / "r", {"a.py": "1\n"})
    (tmp_path / "secret.txt").write_text("top secret")
    ctx = _Ctx()
    await set_target_repo(str(repo), ctx)
    out = await read_repo_file("../secret.txt", ctx)
    assert "error" in out
    assert "secret" not in out.get("content", "")


async def test_read_tools_need_a_target(tmp_path):
    ctx = _Ctx()
    assert "error" in await read_repo_file("x.py", ctx)
    assert "error" in await list_repo_tree(ctx)
