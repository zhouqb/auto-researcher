"""Offline integration test: full Phase 0 flow with a scripted mock LLM.

Validates the plumbing end-to-end without an API key: clarify → brief →
save_plan (facets into state) → plan-approval gate → transfer →
parallel literature fan-out (state templating) → synthesis → cited report,
with every artifact landing in the store and catalog.
"""

from __future__ import annotations

from typing import AsyncGenerator

import pytest
from google.adk.apps import App
from google.adk.models import BaseLlm, LlmRequest, LlmResponse
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

import deep_researcher.config as config_mod
from deep_researcher.agents import build_root_agent
from deep_researcher.storage import ArtifactCatalog, LocalArtifactService

pytestmark = pytest.mark.asyncio


def _call(name: str, **args) -> types.Part:
    return types.Part(function_call=types.FunctionCall(name=name, args=args))


def _text(t: str) -> types.Part:
    return types.Part(text=t)


def _notes_call(i: int) -> types.Part:
    return _call(
        "write_artifact",
        filename=f"lit/facet_{i}/notes.md",
        content=f"# Notes facet {i}\n- Paper X{i} (2024) arXiv:240{i}.0001",
        kind="lit_notes",
        title=f"Facet {i} notes",
        summary=f"Notes for facet {i}",
    )


# Module-level script store: agent name → queue of scripted model responses.
SCRIPTS: dict[str, list[list[types.Part]]] = {}


def _system_instruction_text(llm_request: LlmRequest) -> str:
    si = llm_request.config.system_instruction if llm_request.config else None
    if si is None:
        return ""
    if isinstance(si, str):
        return si
    parts = getattr(si, "parts", None) or []
    return " ".join(p.text or "" for p in parts)


def _agent_from_request(llm_request: LlmRequest) -> str:
    # Identify the calling agent by markers planted in our instructions.
    si = _system_instruction_text(llm_request)
    for i in (1, 2, 3):
        if f"literature searcher #{i}" in si:
            return f"lit_searcher_{i}"
    if "synthesize a research team" in si:
        return "lit_synthesizer"
    if "final research report" in si:
        return "report_writer"
    return "orchestrator"


class ScriptedLlm(BaseLlm):
    """Pops one scripted response per model call, keyed by calling agent."""

    @classmethod
    def supported_models(cls) -> list[str]:
        return [".*"]

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        agent = _agent_from_request(llm_request)
        queue = SCRIPTS.get(agent)
        assert queue, f"No scripted response left for agent {agent!r}"
        parts = queue.pop(0)
        yield LlmResponse(
            content=types.Content(role="model", parts=parts), turn_complete=True
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
            # turn 3: approval → checkpoint → decision → transfer
            [_call("record_checkpoint", gate="plan_approval",
                   decision="approved", comments="looks good")],
            [_call("append_decision", context="Gate 1", decision="Plan approved",
                   evidence="plan/plan.md v0")],
            [_call("transfer_to_agent", agent_name="research_pipeline")],
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
    mock = ScriptedLlm(model="scripted")

    def patch_models(agent):
        if hasattr(agent, "model"):
            agent.model = mock
        for sub in agent.sub_agents:
            patch_models(sub)

    patch_models(root)

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

    # Turn 3: approval → checkpoint, decision log, pipeline through report
    events = await _send(runner, "approve")
    authors = {ev.author for ev in events}
    assert {"lit_searcher_1", "lit_searcher_2", "lit_searcher_3",
            "lit_synthesizer", "report_writer"} <= authors

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
