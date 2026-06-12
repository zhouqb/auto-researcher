"""Monitor tests: events-file parse cache."""

from __future__ import annotations

from deep_researcher.monitor import _events_cache, _parse_events_file


def test_events_parse_cache(tmp_path):
    _events_cache.clear()
    path = tmp_path / "codex_events.jsonl"
    path.write_text('{"type":"thread.started","thread_id":"t1"}\n')

    first = _parse_events_file(path)
    assert first.thread_id == "t1"
    # unchanged file → cached object returned, no re-parse
    assert _parse_events_file(path) is first

    # file grew → cache invalidated, new parse picks up the new event
    with path.open("a") as f:
        f.write('{"type":"turn.completed","usage":{"input_tokens":7}}\n')
    second = _parse_events_file(path)
    assert second is not first
    assert second.usage["input_tokens"] == 7

    # missing file → fresh empty accumulator, no cache entry
    missing = _parse_events_file(tmp_path / "nope.jsonl")
    assert missing.thread_id is None
