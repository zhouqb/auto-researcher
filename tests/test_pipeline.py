"""Offline integration test: full Phase 0 flow with a scripted mock LLM.

Validates the plumbing end-to-end without an API key: clarify → brief →
save_plan (facets into state) → plan-approval gate → transfer →
parallel literature fan-out (state templating) → synthesis → cited report,
with every artifact landing in the store and catalog.
"""

from __future__ import annotations

import pytest
from google.adk.apps import App
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from scripted_llm import SCRIPTS, ScriptedLlm, _call, _text, patch_models

import deep_researcher.config as config_mod
from deep_researcher.agents import build_root_agent
from deep_researcher.storage import ArtifactCatalog, LocalArtifactService

pytestmark = pytest.mark.asyncio


def _notes_call(i: int) -> types.Part:
    return _call(
        "write_artifact",
        filename=f"lit/facet_{i}/notes.md",
        content=f"# Notes facet {i}\n- Paper X{i} (2024) arXiv:240{i}.0001",
        kind="lit_notes",
        title=f"Facet {i} notes",
        summary=f"Notes for facet {i}",
    )


@pytest.fixture
def harness(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-dummy")
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data"))
    config_mod.get_settings.cache_clear()
    settings = config_mod.get_settings()

    SCRIPTS.clear()
    SCRIPTS.update({
        "orchestrator": [
            # turn 1: clarifying questions
            [_text("Q1: scope? Q2: success criteria?")],
            # turn 2: brief → plan → gate
            [_call(
                "write_artifact",
                filename="brief/research_brief.md",
                content="# Brief\nQuestion, scope, criteria.",
                kind="brief", title="Research brief", summary="The contract.",
            )],
            [_call(
                "save_plan",
                plan_markdown="# Plan\nObjectives... Outline...",
                lit_facets=["facet one", "facet two"],
            )],
            [_text("Plan saved. Facets: one, two. Approve?")],
            # turn 3: approval → checkpoint → decision → staged AgentTools
            [_call("record_checkpoint", gate="plan_approval",
                   decision="approved", comments="looks good")],
            [_call("append_decision", context="Gate 1", decision="Plan approved",
                   evidence="plan/plan.md v0")],
            [_call("literature_review", request="run the literature review")],
            [_call("report_writer", request="write the final report")],
            [_text("Done. Report at reports/final_report.md")],
        ],
        "lit_searcher_1": [[_notes_call(1)], [_text("Facet one summary; lit/facet_1/notes.md")]],
        "lit_searcher_2": [[_notes_call(2)], [_text("Facet two summary; lit/facet_2/notes.md")]],
        "lit_searcher_3": [[_text("No facet assigned.")]],
        "lit_synthesizer": [
            [_call(
                "write_artifact", filename="lit/synthesis.md",
                content="# Synthesis\nThemes... [1] Paper X1 [2] Paper X2",
                kind="lit_notes", title="Synthesis", summary="Cross-facet synthesis",
            )],
            [_text("Synthesis done; lit/synthesis.md")],
        ],
        "report_writer": [
            [_call("read_artifact", filename="lit/synthesis.md")],
            [_call(
                "write_artifact", filename="reports/final_report.md",
                content="# Report\nFindings [1][2].\n## References\n[1] X1 [2] X2",
                kind="report", title="Final report", summary="Cited final report",
            )],
            [_call("append_decision", context="Reporting", decision="Report completed",
                   evidence="reports/final_report.md")],
            [_text("Report complete: reports/final_report.md (2 refs)")],
        ],
    })

    root = build_root_agent()
    patch_models(root, ScriptedLlm(model="scripted"))

    catalog = ArtifactCatalog(settings.db_path)
    runner = Runner(
        app=App(name=settings.app_name, root_agent=root),
        session_service=InMemorySessionService(),
        artifact_service=LocalArtifactService(settings.root, catalog),
    )
    return runner, catalog, settings


async def _send(runner, msg: str, session_id="proj-test"):
    events = []
    async for ev in runner.run_async(
        user_id="local",
        session_id=session_id,
        new_message=types.Content(role="user", parts=[types.Part(text=msg)]),
    ):
        events.append(ev)
    return events


async def test_full_phase0_flow(harness):
    runner, catalog, settings = harness
    await runner.session_service.create_session(
        app_name=settings.app_name, user_id="local", session_id="proj-test"
    )

    # Turn 1: research question → clarifying questions
    events = await _send(runner, "How do MoE routers affect throughput?")
    assert any(ev.content and ev.content.parts and ev.content.parts[0].text
               and "Q1" in ev.content.parts[0].text for ev in events)

    # Turn 2: answers → brief + plan + gate question
    events = await _send(runner, "Scope: inference only. Success: cited survey.")
    texts = " ".join(
        p.text for ev in events if ev.content and ev.content.parts
        for p in ev.content.parts if p.text
    )
    assert "Approve" in texts
    brief = catalog.get(project_id="proj-test", path="brief/research_brief.md")
    plan = catalog.get(project_id="proj-test", path="plan/plan.md")
    assert brief is not None and brief.kind == "brief"
    assert plan is not None and plan.kind == "plan"

    # State got the facets (templating input for searchers)
    session = await runner.session_service.get_session(
        app_name=settings.app_name, user_id="local", session_id="proj-test"
    )
    assert session.state["lit_facets"] == ["facet one", "facet two"]
    assert session.state["facet_3"] == ""

    # Turn 3: approval → checkpoint, decision log, staged tools through report
    events = await _send(runner, "approve")
    called = [
        p.function_call.name
        for ev in events if ev.content and ev.content.parts
        for p in ev.content.parts if p.function_call
    ]
    assert "literature_review" in called and "report_writer" in called

    paths = catalog.list_paths(project_id="proj-test")
    assert "lit/facet_1/notes.md" in paths
    assert "lit/facet_2/notes.md" in paths
    assert "lit/synthesis.md" in paths
    assert "reports/final_report.md" in paths
    assert "decisions/decisions.md" in paths
    assert any(p.startswith("checkpoints/") and p.endswith("_plan_approval.json")
               for p in paths)

    # Decision log accumulated both entries (gate + report) as versions
    decisions = catalog.versions(project_id="proj-test", path="decisions/decisions.md")
    assert len(decisions) == 2

    # Files are on disk in the design §7.2 layout
    proj_dir = settings.projects_dir / "proj-test"
    assert (proj_dir / "reports/final_report.md").exists()
    assert (proj_dir / "brief/research_brief.md").exists()

    # All scripted responses were consumed (every agent ran exactly as planned)
    assert all(not q for q in SCRIPTS.values())


async def test_experiment_flow_with_budget_gate(harness, tmp_path, monkeypatch):
    """Gate 2 → codex_exec (fake binary) → result_analyst, budget ledger updated."""
    import os, stat

    runner, catalog, settings = harness
    await runner.session_service.create_session(
        app_name=settings.app_name, user_id="local", session_id="proj-exp"
    )

    fake = tmp_path / "fake_codex"
    fake.write_text("""#!/bin/sh
while [ $# -gt 0 ]; do
  case "$1" in
    -C) WS="$2"; shift 2;;
    -o) OUT="$2"; shift 2;;
    *) shift;;
  esac
done
echo '{"type":"thread.started","thread_id":"t-exp-1"}'
echo '{"type":"item.completed","item":{"id":"i0","type":"agent_message","text":"ran"}}'
echo '{"type":"turn.completed","usage":{"input_tokens":1000,"output_tokens":50}}'
echo '{"metric": "rmse", "value": 0.42, "baseline": 0.5}' > "$WS/metrics.json"
echo "Experiment finished." > "$OUT"
""")
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("CODEX_BINARY", str(fake))

    SCRIPTS.clear()
    SCRIPTS.update({
        "orchestrator": [
            # turn 1: design + gate 2
            [_call("experiment_designer", request="design the experiment")],
            [_text("Spec ready. Estimated 2 min / ~50k tokens. Approve the budget?")],
            # turn 2: approval → checkpoint → run → analysis → done
            [_call("record_checkpoint", gate="budget_approval",
                   decision="approved", comments="go")],
            [_call("codex_exec", task_prompt="Implement and run the experiment per spec.")],
            [_call("result_analyst", request="analyze run (see latest run_id)")],
            [_text("Experiment supported the hypothesis. See iter_1/analysis.md")],
        ],
        "experiment_designer": [
            [_call(
                "write_artifact", filename="iter_1/exp_spec.md",
                content="# Spec\nHypothesis...\n## Codex task prompt\nDo X.",
                kind="exp_spec", title="Experiment spec", summary="Spec v0",
            )],
            [_text("Spec saved; est. 2 min, ~50k tokens; iter_1/exp_spec.md")],
        ],
        "result_analyst": [
            [_call("read_artifact", filename="iter_1/exp_spec.md")],
            [_call(
                "write_artifact", filename="iter_1/analysis.md",
                content="# Analysis\nrmse 0.42 beats baseline 0.5. Supported.",
                kind="analysis", title="Analysis", summary="Hypothesis supported",
            )],
            [_text("Analysis saved; iter_1/analysis.md")],
        ],
    })

    await _send(runner, "design and run the experiment", session_id="proj-exp")
    events = await _send(runner, "approve the budget", session_id="proj-exp")

    # codex_exec ran and returned parsed metrics
    responses = [
        p.function_response.response
        for ev in events if ev.content and ev.content.parts
        for p in ev.content.parts
        if p.function_response and p.function_response.name == "codex_exec"
    ]
    assert responses and responses[0]["status"] == "completed"
    assert responses[0]["metrics"]["value"] == 0.42
    assert responses[0]["thread_id"] == "t-exp-1"

    paths = catalog.list_paths(project_id="proj-exp")
    assert "iter_1/exp_spec.md" in paths and "iter_1/analysis.md" in paths
    assert "budget/budget.json" in paths
    assert any(p.startswith("iter_1/exp_main/runs/") for p in paths)
    assert any(p.endswith("_budget_approval.json") for p in paths)

    import json as _json
    budget = _json.loads(
        (settings.projects_dir / "proj-exp" / "budget/budget.json").read_text()
    )
    assert budget["totals"]["input_tokens"] == 1000
    assert len(budget["entries"]) == 1

    # workspace prepared with the AGENTS.md contract + git repo
    ws = settings.projects_dir / "proj-exp" / "iter_1/exp_main/repo"
    assert (ws / "AGENTS.md").exists() and (ws / ".git").exists()
    assert all(not q for q in SCRIPTS.values())


def _git(cwd, *args):
    import subprocess
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


async def test_repo_improvement_flow(harness, tmp_path, monkeypatch):
    """Repo mode end-to-end: set_target_repo → design → gate → codex → diff."""
    import stat

    runner, catalog, settings = harness
    await runner.session_service.create_session(
        app_name=settings.app_name, user_id="local", session_id="proj-repo"
    )

    # a tiny source repo to improve
    source = tmp_path / "source"
    source.mkdir()
    (source / "app.py").write_text("def add(a, b):\n    return a + b\n")
    _git(source, "init", "-q")
    _git(source, "config", "user.email", "t@t.io")
    _git(source, "config", "user.name", "t")
    _git(source, "add", "-A")
    _git(source, "commit", "-qm", "init")

    # fake codex: edit the seeded clone + write outcome.json (no metrics/plots)
    fake = tmp_path / "fake_codex"
    fake.write_text(
        "#!/bin/sh\n"
        "MODE=exp\n"
        'while [ $# -gt 0 ]; do case "$1" in -C) WS="$2"; shift 2;; '
        '-o) OUT="$2"; shift 2;; resume) MODE=analysis; shift;; *) shift;; esac; done\n'
        'WS="${WS:-$PWD}"\n'  # resume passes no -C → codex runs with cwd=workspace
        'echo \'{"type":"thread.started","thread_id":"t-repo-1"}\'\n'
        'echo \'{"type":"turn.completed","usage":{"input_tokens":900,"output_tokens":40}}\'\n'
        'if [ "$MODE" = "analysis" ]; then\n'
        '  echo \'{"branch":"main","verdict":"success","root_cause":"change was minimal and correct",'
        '"evidence":["tests green"],"key_factors":["one-line edit"],"next_step":"ship"}\''
        ' > "$WS/diagnosis.json"\n'
        '  echo "Diagnosis: tests pass for the right reason." > "$OUT"\n'
        "else\n"
        '  printf "\\n# improved\\n" >> "$WS/app.py"\n'
        '  echo \'{"approach":"main","changed_files":["app.py"],'
        '"tests":{"command":"pytest","passed":1,"total":1,"failures":[],"green":true},'
        '"acceptance_met":true,"summary":"tweak"}\' > "$WS/outcome.json"\n'
        '  echo "changed app.py" > "$OUT"\n'
        "fi\n"
    )
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("CODEX_BINARY", str(fake))

    SCRIPTS.clear()
    SCRIPTS.update({
        "orchestrator": [
            # turn 1: set repo → brief → plan (no facets) → gate 1
            [_call("set_target_repo", repo=str(source))],
            [_call("write_artifact", filename="brief/research_brief.md",
                   content="# Brief\nAdd a comment to app.py.", kind="brief",
                   title="Brief", summary="change goal")],
            [_call("save_plan", plan_markdown="# Plan\nSingle-branch change.",
                   lit_facets=[])],
            [_text("Plan ready (repo mode, no literature). Approve?")],
            # turn 2: approve plan → design → gate 2
            [_call("record_checkpoint", gate="plan_approval",
                   decision="approved", comments="ok")],
            [_call("experiment_designer", request="design single branch")],
            [_text("Spec ready. ~1 min. Approve budget?")],
            # turn 3: approve budget → run → analyze → done
            [_call("record_checkpoint", gate="budget_approval",
                   decision="approved", comments="go")],
            [_call("codex_exec", task_prompt="Make the change per spec.",
                   branch_id="main")],
            [_call("analyze_experiment", branch_id="main",
                   resume_thread_id="t-repo-1")],
            [_call("result_analyst", request="analyze the change")],
            [_text("Change ready; iter_1/exp_main/change.diff")],
        ],
        "experiment_designer": [
            [_call("write_artifact", filename="iter_1/exp_spec.md",
                   content="# Spec\n## Branch main\n### Codex task prompt\nAdd a comment.",
                   kind="exp_spec", title="Spec", summary="v0")],
            [_text("Spec saved; iter_1/exp_spec.md")],
        ],
        "result_analyst": [
            [_call("read_artifact", filename="iter_1/exp_spec.md")],
            [_call("write_artifact", filename="iter_1/analysis.md",
                   content="# Analysis\nTests green; change minimal. Ready.",
                   kind="analysis", title="Analysis", summary="ready")],
            [_text("Analysis saved; winner: main")],
        ],
    })

    # turn 1: repo mode engages
    await _send(runner, f"improve the repo at {source}", session_id="proj-repo")
    session = await runner.session_service.get_session(
        app_name=settings.app_name, user_id="local", session_id="proj-repo"
    )
    assert session.state["mode"] == "repo_improvement"
    assert session.state["target_repo_path"] == str(source.resolve())
    paths = catalog.list_paths(project_id="proj-repo")
    assert "brief/target_repo.json" in paths

    await _send(runner, "approve", session_id="proj-repo")
    events = await _send(runner, "approve budget", session_id="proj-repo")

    # codex_exec ran in repo mode: outcome surfaced + change diff produced
    responses = [
        p.function_response.response
        for ev in events if ev.content and ev.content.parts
        for p in ev.content.parts
        if p.function_response and p.function_response.name == "codex_exec"
    ]
    assert responses and responses[0]["status"] == "completed"
    assert responses[0]["metrics"]["acceptance_met"] is True
    assert responses[0]["change_diff_path"] == "iter_1/exp_main/change.diff"

    paths = catalog.list_paths(project_id="proj-repo")
    assert "iter_1/exp_main/change.diff" in paths
    diff = catalog.get(project_id="proj-repo", path="iter_1/exp_main/change.diff")
    assert diff.kind == "diff"

    # analyze_experiment resumed the branch's session and wrote a diagnosis
    analyze = [
        p.function_response.response
        for ev in events if ev.content and ev.content.parts
        for p in ev.content.parts
        if p.function_response and p.function_response.name == "analyze_experiment"
    ]
    assert analyze and analyze[0]["verdict"] == "success"
    assert "iter_1/exp_main/diagnosis.md" in paths
    diag = catalog.get(project_id="proj-repo", path="iter_1/exp_main/diagnosis.md")
    assert diag.kind == "analysis"

    # the user's ORIGINAL repo is untouched — the branch worked on a clone
    assert "# improved" not in (source / "app.py").read_text()
    assert all(not q for q in SCRIPTS.values())
