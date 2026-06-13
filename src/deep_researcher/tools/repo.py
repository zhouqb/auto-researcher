"""Target-repo tools for repo-improvement mode.

``set_target_repo`` turns a local path or git URL into a local working copy,
auto-detects how to run its tests, and records it into session state — which
flips the project into 'repo improvement' mode (the orchestrator and the
experiment branches read this state). ``list_repo_tree``/``read_repo_file`` let
the designer agents ground their plans in the real code without re-cloning.

Safety: each experiment branch works on its OWN clone of the resolved path
(see codex/workspace.py), so the user's repo is never mutated by experiments.
The read tools refuse any path that escapes the target root.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Optional

from google.adk.tools import ToolContext
from google.genai import types

from ..config import get_settings

# Directories never worth walking/serving when grounding a plan.
_NOISE_DIRS = frozenset(
    {".git", "node_modules", ".venv", "venv", "__pycache__", ".next",
     "dist", "build", ".mypy_cache", ".pytest_cache", ".ruff_cache"}
)
_TREE_CAP = 400
_FILE_CAP = 24_000


def _looks_like_url(s: str) -> bool:
    return s.startswith(("http://", "https://", "ssh://", "git://", "git@")) or (
        s.endswith(".git")
    )


def _detect_test_command(path: Path) -> Optional[str]:
    """Best-effort guess at the repo's test command; None if unsure."""
    if (path / "package.json").exists():
        try:
            pkg = json.loads((path / "package.json").read_text())
            if (pkg.get("scripts") or {}).get("test"):
                return "npm test"
        except (json.JSONDecodeError, OSError):
            pass
    if (path / "pyproject.toml").exists() or (path / "setup.py").exists() or (
        path / "tests"
    ).is_dir() or (path / "test").is_dir():
        return "pytest -q"
    makefile = path / "Makefile"
    if makefile.exists():
        try:
            if any(
                line.startswith("test:") for line in makefile.read_text().splitlines()
            ):
                return "make test"
        except OSError:
            pass
    return get_settings().repo_default_test_command


def _resolved_root(tool_context: ToolContext) -> Optional[Path]:
    raw = tool_context.state.get("target_repo_path")
    return Path(raw) if raw else None


def _safe_member(root: Path, relative: str) -> Optional[Path]:
    """Resolve ``relative`` under ``root``; None if it escapes (symlinks too)."""
    candidate = (root / relative).resolve()
    root = root.resolve()
    if candidate == root or root in candidate.parents:
        return candidate
    return None


async def set_target_repo(
    repo: str, tool_context: ToolContext, test_command: str = ""
) -> dict[str, Any]:
    """Point the project at an existing repository to improve (repo mode).

    Call this during BRIEF when the user wants to change an existing codebase.
    It resolves a local path or a git URL (URLs are cloned locally), detects a
    default test command, and switches the project into repo-improvement mode.

    Args:
        repo: A local filesystem path OR a git URL (https/ssh/git@). URLs are
            cloned into the project's source_repo/ directory.
        test_command: Optional explicit command to run the repo's tests, e.g.
            "pytest -q" or "npm test". Auto-detected when omitted.

    Returns:
        {mode, target_repo_path, repo_test_command, is_git, source} or {error}.
    """
    settings = get_settings()
    proj_dir = settings.projects_dir / tool_context.session.id

    if _looks_like_url(repo):
        clone_dir = proj_dir / "source_repo"
        if not (clone_dir / ".git").is_dir():
            clone_dir.parent.mkdir(parents=True, exist_ok=True)
            proc = subprocess.run(
                ["git", "clone", repo, str(clone_dir)],
                capture_output=True, text=True,
            )
            if proc.returncode != 0:
                return {"error": f"git clone failed: {proc.stderr.strip()[-400:]}"}
        resolved = clone_dir.resolve()
    else:
        resolved = Path(repo).expanduser().resolve()
        if not resolved.is_dir():
            return {"error": f"local path is not a directory: {resolved}"}

    is_git = (resolved / ".git").is_dir()
    cmd = test_command.strip() or _detect_test_command(resolved) or ""

    tool_context.state["mode"] = "repo_improvement"
    tool_context.state["target_repo_path"] = str(resolved)
    tool_context.state["repo_test_command"] = cmd

    record = {
        "source": repo,
        "resolved_path": str(resolved),
        "is_git": is_git,
        "test_command": cmd,
    }
    await tool_context.save_artifact(
        "brief/target_repo.json",
        types.Part(text=json.dumps(record, indent=2)),
        custom_metadata={
            "kind": "brief",
            "title": "Target repository",
            "summary": f"{repo} → {resolved} (tests: {cmd or 'unknown'})",
            "created_by": tool_context.agent_name,
        },
    )
    return {"mode": "repo_improvement", "target_repo_path": str(resolved),
            "repo_test_command": cmd, "is_git": is_git, "source": repo}


async def list_repo_tree(
    tool_context: ToolContext, subpath: str = ".", depth: int = 2
) -> dict[str, Any]:
    """List files/dirs in the target repo (to ground a change plan in real code).

    Args:
        subpath: Repo-relative directory to list (default the repo root).
        depth: How many directory levels to descend (1-4).

    Returns:
        {root, entries: [relative paths]} or {error}.
    """
    root = _resolved_root(tool_context)
    if root is None:
        return {"error": "no target repo set; call set_target_repo first"}
    base = _safe_member(root, subpath)
    if base is None or not base.is_dir():
        return {"error": f"not a directory inside the repo: {subpath!r}"}
    depth = max(1, min(depth, 4))

    entries: list[str] = []
    def walk(d: Path, level: int) -> None:
        if level > depth or len(entries) >= _TREE_CAP:
            return
        for child in sorted(d.iterdir()):
            if child.name in _NOISE_DIRS or child.name.startswith("."):
                continue
            if len(entries) >= _TREE_CAP:
                return
            rel = child.relative_to(root).as_posix()
            entries.append(rel + ("/" if child.is_dir() else ""))
            if child.is_dir():
                walk(child, level + 1)

    walk(base, 1)
    return {"root": str(root), "entries": entries,
            "truncated": len(entries) >= _TREE_CAP}


async def read_repo_file(filename: str, tool_context: ToolContext) -> dict[str, Any]:
    """Read one file from the target repo (read-only; for grounding plans).

    Args:
        filename: Repo-relative path, e.g. "src/app/main.py".

    Returns:
        {filename, content} or {error}.
    """
    root = _resolved_root(tool_context)
    if root is None:
        return {"error": "no target repo set; call set_target_repo first"}
    member = _safe_member(root, filename)
    if member is None or not member.is_file():
        return {"error": f"no such file in the repo: {filename!r}"}
    try:
        text = member.read_text(errors="replace")
    except OSError as e:
        return {"error": f"could not read {filename!r}: {e}"}
    if len(text) > _FILE_CAP:
        text = text[:_FILE_CAP] + f"\n…[truncated, {len(text)} chars total]"
    return {"filename": filename, "content": text}
