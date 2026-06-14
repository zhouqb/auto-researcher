"""Gateway REST API tests (TestClient; AG-UI route registration smoke)."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

import deep_researcher.config as config_mod


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-dummy")
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data"))
    config_mod.get_settings.cache_clear()
    import deep_researcher.tools.codex as codex_tools
    codex_tools._jobs.cache_clear()

    from deep_researcher.gateway import create_gateway
    with TestClient(create_gateway()) as c:
        yield c, config_mod.get_settings()
    config_mod.get_settings.cache_clear()


def test_routes_registered(client):
    c, _ = client
    paths = {r.path for r in c.app.routes}
    assert "/agui" in paths
    assert "/api/projects" in paths


def test_project_lifecycle_and_panes(client):
    c, settings = client

    # create
    r = c.post("/api/projects", json={"question": "How do MoE routers work?"})
    assert r.status_code == 200
    pid = r.json()["id"]
    assert pid == "how-do-moe-routers-work"

    # list
    assert pid in [p["id"] for p in c.get("/api/projects").json()]

    # empty panes are well-formed
    assert c.get(f"/api/projects/{pid}/history").json() == []
    assert c.get(f"/api/projects/{pid}/runs").json() == []
    assert c.get(f"/api/projects/{pid}/board").json() == {"items": []}
    assert c.get(f"/api/projects/{pid}/budget").json()["totals"] == {}
    assert c.get(f"/api/projects/{pid}/status").json()["resumable_invocation"] is None

    # artifacts: seed one via the artifact layout + catalog, then read through API
    proj_dir = settings.projects_dir / pid
    (proj_dir / "plan").mkdir(parents=True)
    (proj_dir / "plan/board.json").write_text(json.dumps(
        {"items": [{"id": "lit-1", "title": "Facet 1", "type": "lit_task",
                    "status": "in_progress"}]}
    ))
    board = c.get(f"/api/projects/{pid}/board").json()
    assert board["items"][0]["status"] == "in_progress"

    # 404s
    assert c.get("/api/projects/nope/history").status_code == 404
    assert c.get(
        f"/api/projects/{pid}/artifacts/content", params={"path": "nope.md"}
    ).status_code == 404


def test_artifact_content_roundtrip(client):
    c, settings = client
    import asyncio
    from deep_researcher.storage import ArtifactCatalog, LocalArtifactService
    from google.genai import types

    pid = "proj-gw"
    catalog = ArtifactCatalog(settings.db_path)
    svc = LocalArtifactService(settings.root, catalog)
    asyncio.run(svc.save_artifact(
        app_name=settings.app_name, user_id="local", session_id=pid,
        filename="reports/final_report.md",
        artifact=types.Part(text="# Report\nwith $e=mc^2$"),
        custom_metadata={"kind": "report", "title": "Final report",
                         "summary": "demo"},
    ))

    arts = c.get(f"/api/projects/{pid}/artifacts").json()
    assert arts[0]["path"] == "reports/final_report.md"
    art_id = arts[0]["id"]

    content = c.get(
        f"/api/projects/{pid}/artifacts/content",
        params={"path": "reports/final_report.md"},
    ).json()
    assert "$e=mc^2$" in content["text"]
    assert content["versions"] == [0]

    lineage = c.get(f"/api/projects/{pid}/artifacts/{art_id}/lineage").json()
    assert lineage == {"edges": [], "nodes": {}}


def test_kill_endpoint(client):
    c, settings = client
    from deep_researcher.storage.jobs import JobsStore

    store = JobsStore(settings.db_path)
    store.start(project_id="p-kill", branch="B1", run_id="r9")
    r = c.post("/api/projects/p-kill/runs/r9/kill")
    assert r.status_code == 200
    assert store.get("p-kill:r9").status == "killed"


def test_dashboard(client):
    c, settings = client
    c.post("/api/projects", json={"question": "Dashboard smoke?"})
    cards = c.get("/api/dashboard").json()
    assert len(cards) == 1
    card = cards[0]
    assert card["id"] == "dashboard-smoke"
    assert card["has_report"] is False
    assert card["running_runs"] == 0 and card["total_runs"] == 0


@pytest.mark.asyncio
async def test_resume_endpoint_guards(tmp_path, monkeypatch):
    """Resume: 409 while another resume holds the lock; clean no-op otherwise."""
    import asyncio
    import httpx

    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-dummy")
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data"))
    config_mod.get_settings.cache_clear()
    import deep_researcher.tools.codex as codex_tools
    codex_tools._jobs.cache_clear()

    from deep_researcher.gateway import create_gateway
    app = create_gateway()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        r = await c.post("/api/projects", json={"question": "Resume guard?"})
        pid = r.json()["id"]

        # concurrent resume in flight → 409
        lock: asyncio.Lock = app.state.resume_locks[pid]
        await lock.acquire()
        try:
            r = await c.post(f"/api/projects/{pid}/resume")
            assert r.status_code == 409
        finally:
            lock.release()

        # nothing to resume → explicit no-op, not an error
        r = await c.post(f"/api/projects/{pid}/resume")
        assert r.status_code == 200
        assert r.json() == {"resumed": False, "reason": "nothing to resume"}

        # unknown project → 404
        r = await c.post("/api/projects/nope/resume")
        assert r.status_code == 404
    config_mod.get_settings.cache_clear()


def test_cors_allows_any_localhost_port(client):
    """Next.js auto-bumps to 3001+ when Langfuse holds 3000 (regression);
    *.localhost names come from the local Caddy proxy."""
    c, _ = client
    for origin in (
        "http://localhost:3001",
        "http://127.0.0.1:4567",
        "http://researcher.localhost",
        "http://langfuse.localhost",
    ):
        r = c.options(
            "/api/projects",
            headers={
                "Origin": origin,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )
        assert r.status_code == 200
        assert r.headers["access-control-allow-origin"] == origin
    # non-local origins stay rejected (regex is full-match)
    for origin in (
        "http://evil.example:3000",
        "http://researcher.localhost.evil.example",
        "https://localhost.evil.example",
    ):
        r = c.options(
            "/api/projects",
            headers={
                "Origin": origin,
                "Access-Control-Request-Method": "POST",
            },
        )
        assert "access-control-allow-origin" not in r.headers, origin


def test_delete_project_removes_everything(client):
    c, settings = client
    from deep_researcher.storage import ArtifactCatalog
    from deep_researcher.storage.jobs import JobsStore

    pid = c.post("/api/projects", json={"question": "Doomed project?"}).json()["id"]

    # seed: one artifact (catalog + file), one finished job
    catalog = ArtifactCatalog(settings.db_path)
    catalog.register(project_id=pid, path="lit/notes.md", version=0,
                     title="notes", body_text="findings")
    proj_dir = settings.projects_dir / pid
    proj_dir.mkdir(parents=True, exist_ok=True)
    (proj_dir / "stale.txt").write_text("x")
    jobs = JobsStore(settings.db_path)
    jobs.start(project_id=pid, branch="main", run_id="r1")
    jobs.finish(f"{pid}:r1", "completed")

    r = c.delete(f"/api/projects/{pid}")
    assert r.status_code == 200
    assert r.json()["deleted"] == pid
    assert r.json()["artifacts_deleted"] == 1

    assert pid not in [p["id"] for p in c.get("/api/projects").json()]
    assert c.get(f"/api/projects/{pid}/history").status_code == 404
    assert catalog.list_paths(project_id=pid) == []
    assert jobs.for_project(pid) == []
    assert not proj_dir.exists()

    # second delete: project no longer exists
    assert c.delete(f"/api/projects/{pid}").status_code == 404


def test_delete_project_surfaces_undeletable_residue(client, monkeypatch):
    """A file that can't be removed must be reported, not silently swallowed."""
    import deep_researcher.gateway as gw

    c, settings = client
    pid = c.post("/api/projects", json={"question": "Sticky?"}).json()["id"]
    proj_dir = settings.projects_dir / pid
    proj_dir.mkdir(parents=True, exist_ok=True)
    (proj_dir / "busy.txt").write_text("held open by a live run")

    stuck = str(proj_dir / "busy.txt")

    def _fake_rmtree(path, *, onexc):  # simulate an unremovable file
        onexc(None, stuck, OSError("Device or resource busy"))

    monkeypatch.setattr(gw.shutil, "rmtree", _fake_rmtree)

    body = c.delete(f"/api/projects/{pid}").json()
    # the metadata delete still succeeds, but the residue is surfaced
    assert body["deleted"] == pid
    assert body["incomplete"] is True
    assert stuck in body["residual_paths"]
