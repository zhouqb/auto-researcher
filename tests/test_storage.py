import pytest
from google.genai import types

from deep_researcher.storage import ArtifactCatalog, LocalArtifactService

pytestmark = pytest.mark.asyncio


@pytest.fixture
def service(tmp_path):
    catalog = ArtifactCatalog(tmp_path / "test.db")
    return LocalArtifactService(tmp_path, catalog), catalog, tmp_path


async def test_save_load_roundtrip(service):
    svc, catalog, root = service
    v = await svc.save_artifact(
        app_name="app", user_id="u1", session_id="proj1",
        filename="brief/research_brief.md",
        artifact=types.Part(text="# Brief\nhello"),
        custom_metadata={"kind": "brief", "title": "Research brief",
                         "summary": "The contract.", "created_by": "scope_writer"},
    )
    assert v == 0
    part = await svc.load_artifact(
        app_name="app", user_id="u1", session_id="proj1",
        filename="brief/research_brief.md",
    )
    assert part.text == "# Brief\nhello"
    # latest file is human-browsable at the semantic path
    assert (root / "projects/proj1/brief/research_brief.md").read_text() == "# Brief\nhello"
    rec = catalog.get(project_id="proj1", path="brief/research_brief.md")
    assert rec.kind == "brief" and rec.created_by == "scope_writer"
    assert rec.content_hash.startswith("sha256:")


async def test_versioning_and_supersedes(service):
    svc, catalog, root = service
    common = dict(app_name="app", user_id="u1", session_id="proj1", filename="plan/plan.md")
    v0 = await svc.save_artifact(**common, artifact=types.Part(text="v0"))
    v1 = await svc.save_artifact(**common, artifact=types.Part(text="v1"))
    assert (v0, v1) == (0, 1)
    assert await svc.list_versions(**common) == [0, 1]
    old = await svc.load_artifact(**common, version=0)
    new = await svc.load_artifact(**common)
    assert (old.text, new.text) == ("v0", "v1")
    # supersedes lineage edge between consecutive versions
    recs = catalog.versions(project_id="proj1", path="plan/plan.md")
    edges = catalog.lineage(recs[1].id)
    assert (recs[1].id, recs[0].id, "supersedes") in edges


async def test_user_namespace_and_keys(service):
    svc, _, root = service
    await svc.save_artifact(
        app_name="app", user_id="u1", session_id="proj1",
        filename="user:NOTES.md", artifact=types.Part(text="cross-project note"),
    )
    await svc.save_artifact(
        app_name="app", user_id="u1", session_id="proj1",
        filename="lit/facet_1/notes.md", artifact=types.Part(text="notes"),
    )
    keys = await svc.list_artifact_keys(app_name="app", user_id="u1", session_id="proj1")
    assert keys == ["lit/facet_1/notes.md", "user:NOTES.md"]
    assert (root / "users/u1/NOTES.md").exists()


async def test_binary_artifact(service):
    svc, _, _ = service
    data = b"\x89PNG fake"
    await svc.save_artifact(
        app_name="app", user_id="u1", session_id="proj1", filename="plots/x.png",
        artifact=types.Part.from_bytes(data=data, mime_type="image/png"),
    )
    part = await svc.load_artifact(
        app_name="app", user_id="u1", session_id="proj1", filename="plots/x.png"
    )
    assert part.inline_data.data == data and part.inline_data.mime_type == "image/png"


async def test_path_traversal_rejected(service):
    svc, _, _ = service
    with pytest.raises(ValueError):
        await svc.save_artifact(
            app_name="app", user_id="u1", session_id="proj1",
            filename="../evil.md", artifact=types.Part(text="x"),
        )


async def test_fts_search(service):
    svc, catalog, _ = service
    await svc.save_artifact(
        app_name="app", user_id="u1", session_id="proj1",
        filename="lit/synthesis.md",
        artifact=types.Part(text="Mixture-of-experts routing improves throughput."),
        custom_metadata={"summary": "Survey of MoE routing strategies"},
    )
    hits = catalog.search("routing")
    assert hits and hits[0].path == "lit/synthesis.md"


async def test_search_operator_chars_do_not_crash(service):
    svc, catalog, _ = service
    await svc.save_artifact(
        app_name="app", user_id="u1", session_id="proj1",
        filename="lit/notes.md",
        artifact=types.Part(text="Non-IID data distribution shift"),
        custom_metadata={"summary": "non-iid robustness notes"},
    )
    # FTS5 operator characters must be sanitized, not raise OperationalError
    hits = catalog.search('non-iid: "data" (shift)')
    assert hits and hits[0].path == "lit/notes.md"
    assert catalog.search("!!! ()") == []


async def test_search_project_filter_applies_in_sql(service):
    svc, catalog, _ = service
    # flood project "other" with matches that would fill a small LIMIT
    for i in range(3):
        await svc.save_artifact(
            app_name="app", user_id="u1", session_id="other",
            filename=f"lit/f{i}.md",
            artifact=types.Part(text="gradient descent convergence analysis"),
        )
    await svc.save_artifact(
        app_name="app", user_id="u1", session_id="target",
        filename="lit/mine.md",
        artifact=types.Part(text="gradient descent convergence analysis"),
    )
    hits = catalog.search("gradient convergence", project_id="target", limit=2)
    assert [h.project_id for h in hits] == ["target"]
