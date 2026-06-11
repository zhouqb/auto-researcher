import pytest

from deep_researcher.storage.experiences import ExperienceStore


@pytest.fixture
def store(tmp_path):
    return ExperienceStore(tmp_path / "exp.db")


def test_record_and_get(store):
    exp = store.record(
        project_id="p1",
        hypothesis="Halton beats pseudo-random for smooth 2D integrands",
        outcome="success",
        lessons="QMC error ~O(1/N); use scrambling for variance estimates.",
        method={"approach": "Halton bases 2,3", "key_params": {"dims": 2}},
        result={"metric": "rmse_ratio", "value": 0.18},
        codex_thread_id="t-1",
    )
    got = store.get(exp.experience_id)
    assert got.outcome == "success"
    assert got.method["approach"] == "Halton bases 2,3"
    assert got.confidence == 0.7


def test_invalid_outcome_rejected(store):
    with pytest.raises(ValueError):
        store.record(project_id="p", hypothesis="h", outcome="great", lessons="l")


def test_search_keywords_and_failure_modes(store):
    store.record(
        project_id="p1", hypothesis="Large batch speeds up MoE training",
        outcome="failure", failure_mode="OOM",
        lessons="Batch 512 OOMs on 24GB GPU; cap at 128.",
        method={"approach": "MoE finetune"},
    )
    store.record(
        project_id="p2", hypothesis="Quasi-random sampling improves integration",
        outcome="success", lessons="Halton good in low dims.",
    )
    hits = store.search("MoE batch size OOM GPU")
    assert hits and hits[0].failure_mode == "OOM"
    # operator characters must not break FTS5 syntax
    assert store.search('batch-size: "OOM" (gpu)') != []


def test_supersedes_hides_stale_records(store):
    old = store.record(
        project_id="p1", hypothesis="Approach X cannot work for task Y",
        outcome="failure", failure_mode="no_improvement",
        lessons="X showed no improvement.",
    )
    store.record(
        project_id="p1", hypothesis="Approach X works for task Y with fix Z",
        outcome="success", lessons="Earlier failure was a bug; X works with Z.",
        supersedes=[old.experience_id],
    )
    hits = store.search("approach X task Y")
    ids = [h.experience_id for h in hits]
    assert old.experience_id not in ids
    assert any("works with Z" in h.lessons for h in hits)


def test_recent_scoped_by_project(store):
    store.record(project_id="a", hypothesis="h1", outcome="success", lessons="l1")
    store.record(project_id="b", hypothesis="h2", outcome="failure", lessons="l2")
    assert [e.project_id for e in store.recent(project_id="a")] == ["a"]
    assert len(store.recent()) == 2
