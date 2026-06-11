"""Runner assembly: ADK Runner on SQLite sessions + the local artifact store.

One project maps to one ADK session (project_id == session_id), so a
project's artifacts live at ``<data_root>/projects/<session_id>/``.

Resumability (design §10): the App is marked resumable, so completed steps of
a crashed invocation replay from the event log and only the incomplete step
re-runs. `codex_exec` is idempotent (result markers), making replay safe.
Compaction (design §6): older events are LLM-summarized in a sliding window
to keep long projects inside the context budget.
"""

from __future__ import annotations

import re
import uuid
from typing import AsyncIterator, Optional

from google.adk.apps import App, ResumabilityConfig
from google.adk.apps.app import EventsCompactionConfig
from google.adk.apps.llm_event_summarizer import LlmEventSummarizer
from google.adk.events import Event
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService, Session
from google.genai import types

from .agents import build_root_agent
from .config import get_settings
from .storage import ArtifactCatalog, LocalArtifactService

DEFAULT_USER_ID = "local"


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug[:48] or f"project-{uuid.uuid4().hex[:8]}"


def build_app() -> App:
    settings = get_settings()
    return App(
        name=settings.app_name,
        root_agent=build_root_agent(),
        resumability_config=ResumabilityConfig(is_resumable=True),
        events_compaction_config=EventsCompactionConfig(
            summarizer=LlmEventSummarizer(
                llm=LiteLlm(model=settings.worker_model)
            ),
            compaction_interval=settings.compaction_interval,
            overlap_size=settings.compaction_overlap,
        ),
    )


def build_runner() -> Runner:
    settings = get_settings()
    catalog = ArtifactCatalog(settings.db_path)
    return Runner(
        app=build_app(),
        session_service=DatabaseSessionService(db_url=settings.session_db_url),
        artifact_service=LocalArtifactService(settings.root, catalog),
    )


def find_resumable_invocation(session: Session) -> Optional[str]:
    """Invocation id of a crashed/incomplete final invocation, if any.

    Heuristic per ADK semantics: a finished invocation ends with a final
    response event from the root agent (text, not partial, no pending
    function calls). If the session's last event isn't one, the invocation
    was cut short and can be resumed.
    """
    if not session.events:
        return None
    last = session.events[-1]
    if last.author == "user":
        return None
    if last.is_final_response() and not last.partial:
        return None
    return last.invocation_id


async def resume_invocation(
    runner: Runner,
    project_id: str,
    invocation_id: str,
    user_id: str = DEFAULT_USER_ID,
) -> AsyncIterator[Event]:
    """Resume a crashed invocation; completed steps replay from the event log."""
    async for event in runner.run_async(
        user_id=user_id,
        session_id=project_id,
        invocation_id=invocation_id,
    ):
        yield event


async def get_or_create_session(
    runner: Runner, project_id: str, user_id: str = DEFAULT_USER_ID
) -> Session:
    settings = get_settings()
    session = await runner.session_service.get_session(
        app_name=settings.app_name, user_id=user_id, session_id=project_id
    )
    if session is None:
        session = await runner.session_service.create_session(
            app_name=settings.app_name, user_id=user_id, session_id=project_id
        )
    return session


async def list_projects(runner: Runner, user_id: str = DEFAULT_USER_ID) -> list[Session]:
    settings = get_settings()
    response = await runner.session_service.list_sessions(
        app_name=settings.app_name, user_id=user_id
    )
    return list(response.sessions)


async def run_turn(
    runner: Runner,
    project_id: str,
    message: str,
    user_id: str = DEFAULT_USER_ID,
) -> AsyncIterator[Event]:
    """Send one user message to a project's session and yield response events."""
    await get_or_create_session(runner, project_id, user_id)
    async for event in runner.run_async(
        user_id=user_id,
        session_id=project_id,
        new_message=types.Content(role="user", parts=[types.Part(text=message)]),
    ):
        yield event


def event_text(event: Event) -> Optional[str]:
    """Concatenated text parts of an event, if any."""
    if event.content and event.content.parts:
        texts = [p.text for p in event.content.parts if p.text]
        if texts:
            return "\n".join(texts)
    return None


def event_tool_calls(event: Event) -> list[str]:
    if not (event.content and event.content.parts):
        return []
    return [
        p.function_call.name
        for p in event.content.parts
        if p.function_call is not None and p.function_call.name
    ]
