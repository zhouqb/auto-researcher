"""FastAPI gateway (design §12.2): AG-UI chat plane + project REST API.

    Next.js / CopilotKit UI ◄── AG-UI SSE ──► /agui (ag_ui_adk middleware)
                            ◄── REST ──────► /api/* (projects, artifacts,
                                             runs, board, budget, kill, resume)

AG-UI thread id == ADK session id == project id, so the frontend selects a
project simply by setting the chat thread. Run with:

    uv run uvicorn deep_researcher.gateway:app --port 8042
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google.adk.sessions import DatabaseSessionService

from .config import get_settings
from .monitor import kill_run, list_runs, load_budget
from .runner import (
    DEFAULT_USER_ID,
    build_app,
    build_runner,
    event_text,
    find_resumable_invocation,
    slugify,
)
from .storage import ArtifactCatalog, LocalArtifactService
from .storage.jobs import TERMINAL, JobsStore

logger = logging.getLogger(__name__)


def _purge_tree(path: Path) -> list[str]:
    """Remove a directory tree, returning the paths that could NOT be removed.

    Unlike ``shutil.rmtree(..., ignore_errors=True)``, which silently swallows
    failures and lets a partial delete report success, this logs every failure
    and returns the residue so the caller can surface it (e.g. a Codex run still
    holding a file mid-teardown leaves a half-deleted project we want to know
    about, not hide).
    """
    if not path.exists():
        return []
    residue: list[str] = []

    def _onexc(_func: Any, p: str, exc: BaseException) -> None:
        residue.append(str(p))
        logger.warning("delete_project: could not remove %s: %s", p, exc)

    shutil.rmtree(path, onexc=_onexc)
    return residue


def create_gateway() -> FastAPI:
    from ag_ui_adk import ADKAgent, add_adk_fastapi_endpoint

    from .observability import setup_observability

    setup_observability()
    settings = get_settings()
    catalog = ArtifactCatalog(settings.db_path)
    session_service = DatabaseSessionService(db_url=settings.session_db_url)
    artifact_service = LocalArtifactService(settings.root, catalog)

    api = FastAPI(title="Deep Researcher Gateway")
    api.add_middleware(
        CORSMiddleware,
        # local origins only, in all their shapes: any port (Next.js auto-bumps
        # when Langfuse holds 3000) and *.localhost names served by the local
        # Caddy proxy (e.g. http://researcher.localhost, no port)
        allow_origin_regex=r"http://([a-z0-9-]+\.)?(localhost|127\.0\.0\.1)(:\d+)?",
        allow_methods=["*"],
        allow_headers=["*"],
    )

    adk_agent = ADKAgent.from_app(
        build_app(),
        user_id=DEFAULT_USER_ID,
        session_service=session_service,
        artifact_service=artifact_service,
        use_in_memory_services=False,
        use_thread_id_as_session_id=True,
        emit_messages_snapshot=True,  # rehydrate chat history on each run
        session_timeout_seconds=None,        # projects are long-lived
        delete_session_on_cleanup=False,     # never garbage-collect a project
        execution_timeout_seconds=int(get_settings().codex_timeout_s) + 600,
    )
    add_adk_fastapi_endpoint(api, adk_agent, path="/agui")

    # -- projects ----------------------------------------------------------

    @api.get("/api/dashboard")
    async def dashboard() -> list[dict[str, Any]]:
        """Multi-project overview: status, budget, runs, report presence."""
        response = await session_service.list_sessions(
            app_name=settings.app_name, user_id=DEFAULT_USER_ID
        )
        cards = []
        for s in sorted(
            response.sessions, key=lambda x: x.last_update_time or 0, reverse=True
        ):
            runs_info = list_runs(s.id)
            budget_info = load_budget(s.id) or {"totals": {}}
            proj_dir = settings.projects_dir / s.id
            cards.append({
                "id": s.id,
                "last_update_time": s.last_update_time,
                "has_report": (proj_dir / "reports/final_report.md").exists(),
                "running_runs": sum(1 for r in runs_info if r.status == "running"),
                "total_runs": len(runs_info),
                "budget_totals": budget_info.get("totals", {}),
                "artifact_count": len(catalog.list_paths(project_id=s.id)),
            })
        return cards

    @api.get("/api/projects")
    async def projects() -> list[dict[str, Any]]:
        response = await session_service.list_sessions(
            app_name=settings.app_name, user_id=DEFAULT_USER_ID
        )
        return [
            {"id": s.id, "last_update_time": s.last_update_time}
            for s in response.sessions
        ]

    @api.post("/api/projects")
    async def create_project(body: dict) -> dict[str, Any]:
        question = (body.get("question") or "").strip()
        if not question:
            raise HTTPException(400, "question is required")
        project_id = body.get("id") or slugify(question)
        existing = await session_service.get_session(
            app_name=settings.app_name, user_id=DEFAULT_USER_ID,
            session_id=project_id,
        )
        if existing is None:
            await session_service.create_session(
                app_name=settings.app_name, user_id=DEFAULT_USER_ID,
                session_id=project_id,
            )
        return {"id": project_id, "question": question}

    @api.delete("/api/projects/{project_id}")
    async def delete_project(project_id: str) -> dict[str, Any]:
        """Remove a project everywhere: session, catalog rows, jobs, files."""
        session = await session_service.get_session(
            app_name=settings.app_name, user_id=DEFAULT_USER_ID,
            session_id=project_id,
        )
        if session is None:
            raise HTTPException(404, "no such project")
        jobs = JobsStore(settings.db_path)
        for job in jobs.for_project(project_id):
            if job.status not in TERMINAL:
                jobs.kill(job.job_id)
        await session_service.delete_session(
            app_name=settings.app_name, user_id=DEFAULT_USER_ID,
            session_id=project_id,
        )
        artifacts_deleted = catalog.delete_project(project_id)
        jobs.delete_project(project_id)
        api.state.resume_locks.pop(project_id, None)
        residue = await asyncio.to_thread(
            _purge_tree, settings.projects_dir / project_id
        )
        result: dict[str, Any] = {
            "deleted": project_id, "artifacts_deleted": artifacts_deleted
        }
        if residue:
            logger.error(
                "delete_project %s: %d path(s) left on disk after delete",
                project_id, len(residue),
            )
            result["incomplete"] = True
            result["residual_paths"] = residue
        return result

    @api.get("/api/projects/{project_id}/history")
    async def history(project_id: str) -> list[dict[str, Any]]:
        session = await session_service.get_session(
            app_name=settings.app_name, user_id=DEFAULT_USER_ID,
            session_id=project_id,
        )
        if session is None:
            raise HTTPException(404, "no such project")
        return [
            {"author": ev.author, "text": event_text(ev)}
            for ev in session.events
            if event_text(ev) and not ev.partial
        ]

    @api.get("/api/projects/{project_id}/status")
    async def status(project_id: str) -> dict[str, Any]:
        session = await session_service.get_session(
            app_name=settings.app_name, user_id=DEFAULT_USER_ID,
            session_id=project_id,
        )
        if session is None:
            raise HTTPException(404, "no such project")
        return {
            "id": project_id,
            "resumable_invocation": find_resumable_invocation(session),
        }

    # One resume at a time per project: a double-click (or a resume racing a
    # live /agui run) must not replay the same invocation twice.
    api.state.resume_locks = defaultdict(asyncio.Lock)

    # One runner for all resumes: each build_runner() opens a fresh session-db
    # engine whose pool is never disposed (fd leak per /resume request).
    @lru_cache(maxsize=1)
    def resume_runner():
        return build_runner()

    @api.post("/api/projects/{project_id}/resume")
    async def resume(project_id: str) -> dict[str, Any]:
        lock: asyncio.Lock = api.state.resume_locks[project_id]
        if lock.locked():
            raise HTTPException(409, "a resume is already in progress")
        async with lock:
            runner = resume_runner()
            session = await runner.session_service.get_session(
                app_name=settings.app_name, user_id=DEFAULT_USER_ID,
                session_id=project_id,
            )
            if session is None:
                raise HTTPException(404, "no such project")
            invocation_id = find_resumable_invocation(session)
            if invocation_id is None:
                return {"resumed": False, "reason": "nothing to resume"}
            texts: list[str] = []
            async for ev in runner.run_async(
                user_id=DEFAULT_USER_ID, session_id=project_id,
                invocation_id=invocation_id,
            ):
                if (t := event_text(ev)) and not ev.partial:
                    texts.append(t)
            return {"resumed": True, "final_messages": texts[-3:]}

    # -- artifacts ----------------------------------------------------------

    @api.get("/api/projects/{project_id}/artifacts")
    async def artifacts(project_id: str) -> list[dict[str, Any]]:
        return [
            {
                "id": r.id, "path": r.path, "kind": r.kind, "version": r.version,
                "title": r.title, "summary": r.summary, "created_by": r.created_by,
                "created_at": r.created_at,
            }
            for r in catalog.list_latest(project_id=project_id)
        ]

    @api.get("/api/projects/{project_id}/artifacts/content")
    async def artifact_content(
        project_id: str, path: str, version: Optional[int] = None
    ) -> dict[str, Any]:
        record = catalog.get(project_id=project_id, path=path, version=version)
        if record is None:
            raise HTTPException(404, "no such artifact")
        part = await artifact_service.load_artifact(
            app_name=settings.app_name, user_id=DEFAULT_USER_ID,
            session_id=project_id, filename=path, version=record.version,
        )
        if part is None:
            raise HTTPException(404, "artifact content missing on disk")
        body: dict[str, Any] = {
            "path": path, "version": record.version, "kind": record.kind,
            "title": record.title,
            "versions": [r.version for r in catalog.versions(
                project_id=project_id, path=path)],
        }
        if part.text is not None:
            body["text"] = part.text
        elif part.inline_data is not None:
            import base64
            body["base64"] = base64.b64encode(part.inline_data.data).decode()
            body["mime_type"] = part.inline_data.mime_type
        return body

    @api.get("/api/projects/{project_id}/artifacts/{artifact_id}/lineage")
    async def lineage(project_id: str, artifact_id: str) -> dict[str, Any]:
        edges = catalog.lineage(artifact_id)
        nodes = {}
        for child, parent, _rel in edges:
            for art_id in (child, parent):
                if art_id not in nodes and (rec := catalog.get_by_id(art_id)):
                    nodes[art_id] = {
                        "id": rec.id, "path": rec.path, "kind": rec.kind,
                        "version": rec.version, "title": rec.title,
                    }
        return {
            "edges": [
                {"child": c, "parent": p, "relation": r} for c, p, r in edges
            ],
            "nodes": nodes,
        }

    # -- runs / board / budget ----------------------------------------------

    @api.get("/api/projects/{project_id}/runs")
    async def runs(project_id: str) -> list[dict[str, Any]]:
        return [
            {
                "run_id": r.run_id, "experiment": r.experiment, "status": r.status,
                "thread_id": r.thread_id, "usage": r.usage,
                "wallclock_s": r.wallclock_s, "commands": r.commands[-10:],
                "files_changed": r.files_changed[-15:],
                "last_message": r.last_message, "metrics": r.metrics,
            }
            for r in list_runs(project_id)
        ]

    @api.post("/api/projects/{project_id}/runs/{run_id}/kill")
    async def kill(project_id: str, run_id: str) -> dict[str, Any]:
        return {"killed": kill_run(project_id, run_id)}

    @api.get("/api/projects/{project_id}/board")
    async def board(project_id: str) -> dict[str, Any]:
        path = settings.projects_dir / project_id / "plan/board.json"
        if not path.exists():
            return {"items": []}
        return json.loads(path.read_text())

    @api.get("/api/projects/{project_id}/budget")
    async def budget(project_id: str) -> dict[str, Any]:
        return load_budget(project_id) or {"entries": [], "totals": {}}

    return api


app = create_gateway()
