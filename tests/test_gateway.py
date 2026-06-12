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
