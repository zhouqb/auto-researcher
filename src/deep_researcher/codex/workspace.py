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

# Used instead of _CODE_CONTRACT when the branch optimizes a METRIC: the eval
# command writes metrics.json (the objective), so the branch is ranked by it
# rather than by tests. We deliberately do NOT ask for outcome.json — the run
# result is metrics.json (see codex/runner.py result precedence).
_EVAL_CONTRACT = """\
# Code-change experiment contract (metric objective)

You are implementing ONE approach to IMPROVE a metric in this working copy of an
existing repository. The repo's own conventions (README, AGENTS.md, lint/test
config) still apply — follow them.

## Contract (mandatory)
- Make the change. Keep it focused: touch only what improving the metric needs.
- Score your change by running the evaluation command — it writes `metrics.json`
  at the repo root (your objective; do NOT hand-write it):
  {eval_command}
- You are judged by `metrics.json` "value" — HIGHER IS BETTER ({objective_metric}).
  Read `eval/out/*/cases.jsonl` to see which cases failed and why, then iterate
  to raise the metric. Re-run the eval after each change.
- Commit your code change with git (`git add -A && git commit`). metrics.json,
  the eval output, and any linked data dirs are git-excluded — do not commit them.

## Forbidden
- Fabricating numbers — `metrics.json` must come from running the eval command.
- Touching files unrelated to the change; network access unless the task says so.
"""


def prepare_workspace(
    workspace: Path,
    source_repo: Optional[Path] = None,
    test_command: Optional[str] = None,
    *,
    eval_command: Optional[str] = None,
    objective_metric: str = "execution_accuracy",
    link_paths: Optional[list[str]] = None,
) -> None:
    """Idempotent workspace setup; greenfield unless ``source_repo`` is given.

    ``eval_command`` switches the repo contract to metric-objective mode (the
    branch is ranked by the metrics.json the command writes). ``link_paths`` are
    dirs symlinked from the source working tree into the clone (e.g. gitignored
    datasets the eval needs) — linked, not copied, and git-excluded.
    """
    if source_repo is not None:
        _prepare_repo_workspace(
            workspace, Path(source_repo), test_command,
            eval_command=eval_command, objective_metric=objective_metric,
            link_paths=link_paths,
        )
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
    workspace: Path, source_repo: Path, test_command: Optional[str],
    *, eval_command: Optional[str] = None,
    objective_metric: str = "execution_accuracy",
    link_paths: Optional[list[str]] = None,
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

    # Symlink datasets etc. from the source working tree (linked, not copied);
    # exclude them — and metrics.json in eval mode — so they never enter the diff.
    linked = _link_paths(workspace, source_repo, link_paths or [])
    extra_excludes = list(linked) + (["metrics.json"] if eval_command else [])
    _exclude_scaffold(workspace, extra=extra_excludes)
    if eval_command:
        # .git/info/exclude only hides UNtracked paths; if the repo already
        # tracks metrics.json, skip-worktree keeps the eval's overwrite out of
        # `git add -A` (so it never lands in the seed..HEAD change diff).
        _ignore_tracked_file(workspace, "metrics.json")

    if eval_command:
        contract = _EVAL_CONTRACT.format(
            eval_command=eval_command, objective_metric=objective_metric
        )
    else:
        contract = _CODE_CONTRACT.format(
            test_command=test_command or "(detect and run the repo's own test suite)"
        )
    if _install_langfuse_skill(workspace):
        contract += _LANGFUSE_NOTE
    (workspace / CONTRACT_FILE).write_text(contract)
    (workspace / SEED_MARKER).write_text((seed_sha or "") + "\n")


def _link_paths(workspace: Path, source_repo: Path, names: list[str]) -> list[str]:
    """Symlink each existing source/<name> into workspace/<name>; return the
    names actually linked (so they can be git-excluded)."""
    linked: list[str] = []
    source = source_repo.expanduser().resolve()
    for name in names:
        src = source / name
        dst = workspace / name
        if not src.exists() or dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.symlink_to(src, target_is_directory=src.is_dir())
        linked.append(name)
    return linked


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


def _exclude_scaffold(workspace: Path, extra: Optional[list[str]] = None) -> None:
    """Hide our sidecar files from git so they stay out of commits and diffs."""
    info = workspace / ".git" / "info"
    info.mkdir(parents=True, exist_ok=True)
    exclude = info / "exclude"
    existing = exclude.read_text() if exclude.exists() else ""
    names = list(_SCAFFOLD) + list(extra or [])
    additions = [name for name in names if name not in existing]
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


def _ignore_tracked_file(workspace: Path, name: str) -> None:
    """If ``name`` is tracked, mark it skip-worktree so local rewrites (e.g. the
    eval's metrics.json) are not staged by `git add -A`. No-op if untracked."""
    if _git_out(workspace, "ls-files", "--", name):
        _git(workspace, "update-index", "--skip-worktree", name)


def _git(workspace: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=workspace, check=False, capture_output=True)


def _git_out(workspace: Path, *args: str) -> Optional[str]:
    out = subprocess.run(
        ["git", *args], cwd=workspace, check=False, capture_output=True, text=True
    )
    return out.stdout.strip() or None
