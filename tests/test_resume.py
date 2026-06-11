"""Crash-recovery test (design §10): kill an invocation mid-flight, resume it.

Uses the real DatabaseSessionService on a temp SQLite file and a resumable
App. The crash happens after the first tool call persisted; on resume, ADK
replays the completed step (no duplicate side effects) and finishes the turn.
"""

from __future__ import annotations

import pytest
from google.adk.apps import App, ResumabilityConfig
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService
from google.genai import types

import deep_researcher.config as config_mod
from deep_researcher.agents import build_root_agent
from deep_researcher.runner import find_resumable_invocation
from deep_researcher.storage import ArtifactCatalog, LocalArtifactService

from scripted_llm import SCRIPTS, ScriptedLlm, _call, _text, patch_models

pytestmark = pytest.mark.asyncio

PROJECT = "proj-resume"


def _make_runner(settings) -> Runner:
    root = build_root_agent()
    patch_models(root, ScriptedLlm(model="scripted"))
    catalog = ArtifactCatalog(settings.db_path)
    return Runner(
        app=App(
            name=settings.app_name,
            root_agent=root,
            resumability_config=ResumabilityConfig(is_resumable=True),
        ),
        session_service=DatabaseSessionService(
            db_url=f"sqlite+aiosqlite:///{settings.db_path}"
        ),
        artifact_service=LocalArtifactService(settings.root, catalog),
    )


async def test_crash_and_resume_without_duplicate_side_effects(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-dummy")
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data"))
    config_mod.get_settings.cache_clear()
    settings = config_mod.get_settings()

    SCRIPTS.clear()
    SCRIPTS.update({
        "orchestrator": [
            [_call(
                "write_artifact", filename="brief/research_brief.md",
                content="# Brief", kind="brief", title="Brief", summary="contract",
            )],
            [_call(
                "save_plan", plan_markdown="# Plan", lit_facets=["facet one"],
            )],
            [_text("Plan ready. Approve?")],
        ],
    })

    runner = _make_runner(settings)
    await runner.session_service.create_session(
        app_name=settings.app_name, user_id="local", session_id=PROJECT
    )

    # Run the turn but "crash" right after the first tool result is persisted.
    saw_tool_response = False
    agen = runner.run_async(
        user_id="local", session_id=PROJECT,
        new_message=types.Content(role="user", parts=[types.Part(text="go")]),
    )
    async for ev in agen:
        if ev.content and ev.content.parts and any(
            p.function_response is not None for p in ev.content.parts
        ):
            saw_tool_response = True
            break  # simulated crash
    await agen.aclose()
    assert saw_tool_response

    catalog = ArtifactCatalog(settings.db_path)
    assert len(catalog.versions(project_id=PROJECT, path="brief/research_brief.md")) == 1
    assert catalog.get(project_id=PROJECT, path="plan/plan.md") is None

    # Fresh runner (new process semantics), same DB: detect + resume.
    runner2 = _make_runner(settings)
    session = await runner2.session_service.get_session(
        app_name=settings.app_name, user_id="local", session_id=PROJECT
    )
    invocation_id = find_resumable_invocation(session)
    assert invocation_id, "interrupted invocation should be detected as resumable"

    final_texts = []
    async for ev in runner2.run_async(
        user_id="local", session_id=PROJECT, invocation_id=invocation_id
    ):
        if ev.content and ev.content.parts:
            final_texts += [p.text for p in ev.content.parts if p.text]

    assert any("Approve" in t for t in final_texts)
    # Completed step was replayed, not re-executed: still exactly one brief.
    assert len(catalog.versions(project_id=PROJECT, path="brief/research_brief.md")) == 1
    # The interrupted step did run on resume.
    assert catalog.get(project_id=PROJECT, path="plan/plan.md") is not None

    # And the session now ends in a final response (nothing left to resume).
    session = await runner2.session_service.get_session(
        app_name=settings.app_name, user_id="local", session_id=PROJECT
    )
    assert find_resumable_invocation(session) is None
