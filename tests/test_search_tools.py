"""SEARCH_TOOLS setting: registry parsing and agent toolset wiring."""

from __future__ import annotations

import pytest

import deep_researcher.config as config_mod
from deep_researcher.agents import build_root_agent
from deep_researcher.tools.registry import (
    SEARCH_TOOL_REGISTRY,
    parse_search_tools,
    search_tool_guide,
)


@pytest.fixture(autouse=True)
def fresh_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data"))
    config_mod.get_settings.cache_clear()
    yield
    config_mod.get_settings.cache_clear()


def _agent_tool(root, name):
    for tool in root.tools:
        if getattr(getattr(tool, "agent", None), "name", None) == name:
            return tool.agent
    raise AssertionError(f"agent tool {name!r} not found")


def _tool_names(agent):
    return [t.__name__ for t in agent.tools]


def test_parse_normalizes_and_dedups():
    assert parse_search_tools(" OpenAlex, arxiv ,arxiv, github,") == [
        "openalex", "arxiv", "github",
    ]


def test_parse_rejects_unknown_and_empty():
    with pytest.raises(ValueError, match="unknown search tool.*tavily"):
        parse_search_tools("openalex,tavily")
    with pytest.raises(ValueError, match="empty"):
        parse_search_tools(" , ")


def test_guide_mentions_only_enabled_tools():
    guide = search_tool_guide(["openalex", "arxiv"])
    assert "search_openalex" in guide and "search_arxiv" in guide
    for name in ("semantic_scholar", "openreview", "github", "web"):
        assert f"search_{name}" not in guide


def test_default_set_is_s2_primary_quartet():
    assert parse_search_tools(config_mod.Settings().search_tools) == [
        "semantic_scholar", "arxiv", "openalex", "github",
    ]
    root = build_root_agent()
    lit = _agent_tool(root, "literature_review")
    searcher = lit.sub_agents[0].sub_agents[0]  # lit_fanout → lit_searcher_1
    names = _tool_names(searcher)
    assert names == [
        "search_semantic_scholar", "search_arxiv", "search_openalex",
        "search_github", "write_artifact",
    ]
    assert "search_semantic_scholar is your primary paper index" in searcher.instruction
    for off in ("search_openreview", "search_web"):
        assert off not in searcher.instruction
    designer = _agent_tool(root, "experiment_designer")
    assert "search_github" in _tool_names(designer)


def test_env_override_changes_toolset(monkeypatch):
    monkeypatch.setenv("SEARCH_TOOLS", "openalex,web")
    config_mod.get_settings.cache_clear()
    root = build_root_agent()
    searcher = _agent_tool(root, "literature_review").sub_agents[0].sub_agents[0]
    assert _tool_names(searcher) == ["search_openalex", "search_web", "write_artifact"]
    assert "search_web" in searcher.instruction
    # github off → designer loses the tool and the instruction stops mentioning it
    designer = _agent_tool(root, "experiment_designer")
    assert "search_github" not in _tool_names(designer)
    assert "search_github" not in designer.instruction


def test_registry_covers_documented_names():
    assert set(SEARCH_TOOL_REGISTRY) == {
        "openalex", "arxiv", "semantic_scholar", "openreview", "github", "web",
    }
