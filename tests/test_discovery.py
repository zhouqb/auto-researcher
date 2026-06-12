"""Discovery tool tests: OpenReview/GitHub/Tavily parsing, offline."""

from __future__ import annotations

import httpx
import pytest

import deep_researcher.config as config_mod
import deep_researcher.tools.discovery as disc


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("err", request=None, response=None)


@pytest.fixture(autouse=True)
def fresh_settings(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data"))
    config_mod.get_settings.cache_clear()
    disc._github_token.cache_clear()
    yield
    config_mod.get_settings.cache_clear()
    disc._github_token.cache_clear()


def test_openreview_filters_reviews_and_dedupes(monkeypatch):
    notes = [
        # a review note (no title) — must be skipped
        {"forum": "f1", "content": {"rating": {"value": 8}, "summary": {"value": "ok"}}},
        # the submission itself (API v2 value-wrapped fields)
        {"forum": "f1", "cdate": 1700000000000, "content": {
            "title": {"value": "Expert Choice Routing"},
            "venue": {"value": "ICLR 2024 poster"},
            "abstract": {"value": "We invert the assignment. " * 40},
        }},
        # duplicate forum — must be deduped
        {"forum": "f1", "content": {"title": {"value": "Expert Choice Routing"}}},
        # a v1-style note (plain string fields) — must still parse
        {"forum": "f2", "content": {"title": "Hash Layers", "venue": "NeurIPS 2021"}},
    ]
    monkeypatch.setattr(disc.httpx, "get", lambda *a, **k: FakeResponse({"notes": notes}))
    out = disc.search_openreview("routing", limit=5)
    assert [p["title"] for p in out["papers"]] == ["Expert Choice Routing", "Hash Layers"]
    p = out["papers"][0]
    assert p["venue"] == "ICLR 2024 poster"
    assert p["forum_url"] == "https://openreview.net/forum?id=f1"
    assert p["year"] == 2023  # cdate 2023-11
    assert p["abstract"].endswith("…")


def test_github_parses_and_uses_token(monkeypatch):
    captured = {}

    def fake_get(url, params=None, headers=None, timeout=None):
        captured.update(params=params, headers=headers)
        return FakeResponse({"items": [{
            "full_name": "a/b", "description": "impl", "stargazers_count": 42,
            "language": "Python", "html_url": "https://github.com/a/b",
            "pushed_at": "2026-01-02T03:04:05Z", "topics": ["moe", "pytorch"],
        }]})

    monkeypatch.setenv("GITHUB_TOKEN", "gh-test-token")
    config_mod.get_settings.cache_clear()
    monkeypatch.setattr(disc.httpx, "get", fake_get)
    out = disc.search_github("moe impl", limit=3, language="Python")
    repo = out["repos"][0]
    assert repo["stars"] == 42 and repo["last_push"] == "2026-01-02"
    assert "language:Python" in captured["params"]["q"]
    assert captured["headers"]["Authorization"] == "Bearer gh-test-token"


def test_web_search_unconfigured_returns_clear_error():
    out = disc.search_web("anything")
    assert "not configured" in out["error"]


def test_web_search_parses_results(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    config_mod.get_settings.cache_clear()
    monkeypatch.setattr(disc.httpx, "post", lambda *a, **k: FakeResponse(
        {"results": [{"title": "Blog", "url": "https://x.dev/p", "content": "long " * 200}]}
    ))
    out = disc.search_web("moe inference blog")
    assert out["results"][0]["url"] == "https://x.dev/p"
    assert out["results"][0]["snippet"].endswith("…")


def test_error_paths_return_error_dict(monkeypatch):
    def boom(*a, **k):
        raise httpx.ConnectError("down")
    monkeypatch.setattr(disc.httpx, "get", boom)
    assert "error" in disc.search_openreview("x")
    assert "error" in disc.search_github("x")
