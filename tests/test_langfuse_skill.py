"""Codex Langfuse trace-lookup skill (offline; network boundary stubbed)."""

from __future__ import annotations

import base64

from deep_researcher.codex.skills import langfuse_traces as lt


def test_auth_header():
    assert lt.auth_header("pk", "sk") == "Basic " + base64.b64encode(b"pk:sk").decode()


def test_find_case_trace_filters_by_name(monkeypatch):
    seen = {}

    def fake_get(host, path, pk, sk, params=None, **kw):
        seen["path"], seen["params"] = path, params
        return {"data": [{"id": "t1", "name": "eval_case:bird_1_shop"}]}

    monkeypatch.setattr(lt, "api_get", fake_get)
    trace = lt.find_case_trace("http://h", "pk", "sk", "bird_1_shop")
    assert trace["id"] == "t1"
    assert seen["path"] == "/api/public/traces"
    assert seen["params"]["name"] == "eval_case:bird_1_shop"


def test_list_failed_keeps_only_failed_eval_cases(monkeypatch):
    page = {"data": [
        {"name": "eval_case:a", "id": "1",
         "metadata": {"ada.correct": False, "ada.category": "wrong_result"}},
        {"name": "eval_case:b", "id": "2", "metadata": {"ada.correct": True}},
        {"name": "other_trace", "id": "3", "metadata": {"ada.correct": False}},
    ]}
    monkeypatch.setattr(lt, "api_get", lambda *a, **k: page)
    failed = lt.list_failed("http://h", "pk", "sk", limit=10)
    assert [t["name"] for t in failed] == ["eval_case:a"]


def test_format_trajectory_includes_link_and_observations():
    trace = {"id": "t1", "name": "eval_case:x",
             "metadata": {"ada.correct": False, "ada.category": "wrong_result"}}
    obs = [{"type": "GENERATION", "name": "call_llm", "input": "q?", "output": "SELECT 1"}]
    out = lt.format_trajectory("http://h:3000", trace, obs)
    assert "eval_case:x" in out
    assert "http://h:3000/trace/t1" in out
    assert "call_llm" in out and "SELECT 1" in out
    assert "ada.category" in out


def test_main_without_keys_falls_back(monkeypatch, capsys):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    assert lt.main(["--failed"]) == 2
    assert "cases.jsonl" in capsys.readouterr().err


def test_data_list_tolerates_malformed_payloads():
    assert lt._data_list({"data": {}}) == []          # object, not array
    assert lt._data_list({"data": "oops"}) == []       # string
    assert lt._data_list({"error": "x"}) == []         # error payload, no data
    assert lt._data_list("not json at all") == []      # not even a dict
    # non-dict entries are dropped, dict entries kept
    assert lt._data_list({"data": [{"id": 1}, "bad", 7]}) == [{"id": 1}]


def test_main_falls_back_on_non_json_body(monkeypatch, capsys):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")

    def bad_json(*a, **k):
        raise ValueError("Expecting value: line 1 column 1 (char 0)")

    monkeypatch.setattr(lt, "api_get", bad_json)
    assert lt.main(["--case", "bird_1_x"]) == 2
    assert "Fall back" in capsys.readouterr().err
