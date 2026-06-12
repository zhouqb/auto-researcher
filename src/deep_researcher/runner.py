"""Runner assembly: ADK Runner on SQLite sessions + the local artifact store.

Phase 0 maps one project to one ADK session (project_id == session_id), so a
project's artifacts live at ``<data_root>/projects/<session_id>/``.
"""

from __future__ import annotations

import re
import uuid
from typing import AsyncIterator, Optional

from google.adk.apps import App
from google.adk.events import Event
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


def build_runner() -> Runner:
    settings = get_settings()
    catalog = ArtifactCatalog(settings.db_path)
    return Runner(
        app=App(name=settings.app_name, root_agent=build_root_agent()),
        session_service=DatabaseSessionService(db_url=settings.session_db_url),
        artifact_service=LocalArtifactService(settings.root, catalog),
    )


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
