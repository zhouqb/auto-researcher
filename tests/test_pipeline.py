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
