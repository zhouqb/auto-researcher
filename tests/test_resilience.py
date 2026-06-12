"""Truncated-tool-call resilience: max_tokens cap, tolerant parsing, append.

Regression for the live failure where DeepSeek hit its default 4k output cap
mid `write_artifact(content=...)`, ADK raised JSONDecodeError ("Unterminated
string"), and the whole run died.
"""

from __future__ import annotations

import pytest

import deep_researcher.config as config_mod
import deep_researcher.patches as patches_mod
from deep_researcher.tools.artifacts import write_artifact


@pytest.fixture(autouse=True)
def fresh_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data"))
    config_mod.get_settings.cache_clear()
    yield
    config_mod.get_settings.cache_clear()


def test_models_get_max_tokens():
    from deep_researcher.agents import build_root_agent

    root = build_root_agent()
    assert root.model._additional_args["max_tokens"] == 8192
    lit = next(
        t.agent for t in root.tools
        if getattr(getattr(t, "agent", None), "name", None) == "literature_review"
    )
    searcher = lit.sub_agents[0].sub_agents[0]
    assert searcher.model._additional_args["max_tokens"] == 8192


def test_tolerant_tool_call_parsing():
    patches_mod._applied = False
    patches_mod.apply_adk_patches()
    try:
        from google.adk.models import lite_llm

        parse = lite_llm._parse_tool_call_arguments
        # valid arguments still parse normally
        assert parse('{"filename": "a.md", "content": "ok"}') == {
            "filename": "a.md", "content": "ok",
        }
        assert parse("") == {}
        # truncated mid-string (the live failure shape) degrades to {}
        # instead of raising — ADK then sends the model a retryable error
        assert parse('{"filename": "reports/final_report.md", "content": "x') == {}
    finally:
        # unpatch so other tests see ADK's stock behavior
        from google.adk.models import lite_llm

        lite_llm._parse_tool_call_arguments = (
            lite_llm._parse_tool_call_arguments._original
        )
        patches_mod._applied = False


class _StubToolContext:
    """Just enough ToolContext for write_artifact."""

    agent_name = "tester"

    def __init__(self):
        self.saved: dict[str, str] = {}

    async def save_artifact(self, filename, part, custom_metadata=None):
        self.saved[filename] = part.text
        return self.saved.setdefault("_versions", 0)

    async def load_artifact(self, filename, version=None):
        from google.genai import types

        if filename not in self.saved:
            return None
        return types.Part(text=self.saved[filename])


@pytest.mark.asyncio
async def test_write_artifact_append_builds_long_docs():
    ctx = _StubToolContext()
    await write_artifact(
        "reports/final_report.md", "# Report\n\nPart one.", "report",
        "Report", "demo", ctx,
    )
    await write_artifact(
        "reports/final_report.md", "## Section 2\n\nPart two.", "report",
        "Report", "demo", ctx, append=True,
    )
    text = ctx.saved["reports/final_report.md"]
    assert text.startswith("# Report") and text.endswith("Part two.")
    # append on a missing file is just a plain write
    await write_artifact(
        "lit/new.md", "fresh", "lit_notes", "t", "s", ctx, append=True,
    )
    assert ctx.saved["lit/new.md"] == "fresh"
