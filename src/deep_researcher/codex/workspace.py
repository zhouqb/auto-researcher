"""Experiment workspace setup (design §13).

Two shapes, chosen by ``prepare_workspace(source_repo=...)``:

- **Research (greenfield)** — an empty git repo + a research ``AGENTS.md``
  contract (write ``metrics.json`` + plots). The original behavior.
- **Repo improvement** — seed the workspace from an existing repository (git
  clone, history preserved so we can diff ``seed..HEAD`` later) and drop a
  code-change contract. The repo's own ``AGENTS.md``/conventions are left
  untouched; our scaffolding lives in git-excluded sidecar files so it never
  pollutes the change diff.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Optional

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

# Sidecar files written into a repo-improvement workspace. All are added to
# .git/info/exclude so Codex's `git add -A` never commits them and they never
# show up in the `seed..HEAD` change diff.
SEED_MARKER = ".dr_seeded"  # holds the seed commit SHA; also the idempotency guard
CONTRACT_FILE = ".dr_contract.md"
OUTCOME_FILE = "outcome.json"
DIAGNOSIS_FILE = "diagnosis.json"  # written by the post-run analyzer turn
LANGFUSE_SKILL = ".dr_langfuse.py"  # trace-lookup helper (installed when keys set)
_SKILL_SRC = Path(__file__).parent / "skills" / "langfuse_traces.py"
_SCAFFOLD = (SEED_MARKER, CONTRACT_FILE, OUTCOME_FILE, DIAGNOSIS_FILE, LANGFUSE_SKILL)

_LANGFUSE_NOTE = """
## Root-cause tracing (optional)
A `.dr_langfuse.py` helper is available. To inspect WHY specific eval cases
failed (the agent's real LLM prompts/outputs and tool calls), run:
  python .dr_langfuse.py --failed            # recent failed cases + links
  python .dr_langfuse.py --case <case_id>    # one case's full trajectory
Pair it with the eval's `eval/out/*/cases.jsonl` dump for diagnosis.
"""

_CODE_CONTRACT = """\
# Code-change experiment contract

You are implementing ONE approach to a requested change inside this working
copy of an existing repository. The repo's own conventions (its README,
AGENTS.md, lint/test config) still apply — follow them.

## Contract (mandatory)
- Make the change described in your task. Keep it focused: touch only what the
  change requires.
- Run the repo's tests and iterate until they pass AND the task's acceptance
  criteria are met. Test command:
  {test_command}
- Write `outcome.json` at the repo root (it is git-excluded — do NOT commit it):
  {{"approach": "<short id>",
    "changed_files": ["..."],
    "tests": {{"command": "<cmd>", "passed": <int>, "total": <int>,
              "failures": ["..."], "green": <bool>}},
    "acceptance_met": <bool>,
    "summary": "<2-4 sentences: what you changed and why>"}}
- Commit your change with git (`git add -A && git commit`). The scaffolding
  files above are excluded, so they will not be committed.
- Add or update tests when the change warrants it.

## Forbidden
- Touching files unrelated to the change; network access unless the task says
  so; fabricating test results — if tests fail, say so honestly in
  outcome.json (`"green": false`).
"""


def prepare_workspace(
    workspace: Path,
    source_repo: Optional[Path] = None,
    test_command: Optional[str] = None,
) -> None:
    """Idempotent workspace setup; greenfield unless ``source_repo`` is given."""
    if source_repo is not None:
        _prepare_repo_workspace(workspace, Path(source_repo), test_command)
    else:
        _prepare_research_workspace(workspace)


def _prepare_research_workspace(workspace: Path) -> None:
    """Empty dir, AGENTS.md, plots/, fresh git repo (the greenfield default)."""
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "plots").mkdir(exist_ok=True)
    agents = workspace / "AGENTS.md"
    if not agents.exists():
        agents.write_text(AGENTS_MD)
    if not (workspace / ".git").exists():
        _git(workspace, "init", "-q")
        _git(workspace, "add", "-A")
        _git(workspace, "commit", "-qm", "init workspace")


def _prepare_repo_workspace(
    workspace: Path, source_repo: Path, test_command: Optional[str]
) -> None:
    """Seed from an existing repo; idempotent (a seeded workspace is left alone)."""
    if (workspace / SEED_MARKER).exists():
        return  # already seeded — never clobber the coding agent's in-progress work
    source = source_repo.expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"source repo not found: {source}")

    workspace.parent.mkdir(parents=True, exist_ok=True)
    if workspace.exists() and not any(workspace.iterdir()):
        workspace.rmdir()  # git clone needs a non-existent / empty target

    if (source / ".git").is_dir():
        # --local hardlinks the source's immutable objects (git's default for
        # a local path): branches share history on disk but keep independent
        # refs, index, and working tree — full isolation, safe to mutate (git
        # objects are content-addressed and never modified in place; even gc
        # in one clone only decrements a hardlink count). NOT a worktree: the
        # whole repo, .git included, stays inside the sandbox's -C workspace.
        subprocess.run(
            ["git", "clone", "--local", str(source), str(workspace)],
            check=True,
            capture_output=True,
        )
    else:
        shutil.copytree(source, workspace)
        _git(workspace, "init", "-q")
        _git(workspace, "add", "-A")
        _git(workspace, "commit", "-qm", "seed workspace from source")

    seed_sha = _git_out(workspace, "rev-parse", "HEAD")
    _exclude_scaffold(workspace)
    contract = _CODE_CONTRACT.format(
        test_command=test_command or "(detect and run the repo's own test suite)"
    )
    if _install_langfuse_skill(workspace):
        contract += _LANGFUSE_NOTE
    (workspace / CONTRACT_FILE).write_text(contract)
    (workspace / SEED_MARKER).write_text((seed_sha or "") + "\n")


def _install_langfuse_skill(workspace: Path) -> bool:
    """Copy the Langfuse trace-lookup helper into the workspace when Langfuse
    keys are configured (git-excluded). Returns whether it was installed."""
    from ..config import get_settings

    settings = get_settings()
    if not (settings.langfuse_public_key and settings.langfuse_secret_key):
        return False
    if _SKILL_SRC.exists():
        shutil.copyfile(_SKILL_SRC, workspace / LANGFUSE_SKILL)
        return True
    return False


def _exclude_scaffold(workspace: Path) -> None:
    """Hide our sidecar files from git so they stay out of commits and diffs."""
    info = workspace / ".git" / "info"
    info.mkdir(parents=True, exist_ok=True)
    exclude = info / "exclude"
    existing = exclude.read_text() if exclude.exists() else ""
    additions = [name for name in _SCAFFOLD if name not in existing]
    if additions:
        body = (existing.rstrip() + "\n" if existing.strip() else "") + "\n".join(
            additions
        )
        exclude.write_text(body + "\n")


def seed_sha(workspace: Path) -> Optional[str]:
    """The commit the workspace was seeded at (for diffing), or None."""
    marker = workspace / SEED_MARKER
    if marker.exists():
        return marker.read_text().strip() or None
    return None


def _git(workspace: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=workspace, check=False, capture_output=True)


def _git_out(workspace: Path, *args: str) -> Optional[str]:
    out = subprocess.run(
        ["git", *args], cwd=workspace, check=False, capture_output=True, text=True
    )
    return out.stdout.strip() or None
