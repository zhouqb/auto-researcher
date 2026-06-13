"""Shared scripted mock LLM for offline integration tests."""

from __future__ import annotations

from typing import AsyncGenerator

from google.adk.models import BaseLlm, LlmRequest, LlmResponse
from google.genai import types

# Module-level script store: agent name → queue of scripted model responses.
SCRIPTS: dict[str, list[list[types.Part]]] = {}


def _call(name: str, **args) -> types.Part:
    return types.Part(function_call=types.FunctionCall(name=name, args=args))


def _text(t: str) -> types.Part:
    return types.Part(text=t)


def _system_instruction_text(llm_request: LlmRequest) -> str:
    si = llm_request.config.system_instruction if llm_request.config else None
    if si is None:
        return ""
    if isinstance(si, str):
        return si
    parts = getattr(si, "parts", None) or []
    return " ".join(p.text or "" for p in parts)


def _agent_from_request(llm_request: LlmRequest) -> str:
    # Identify the calling agent by markers planted in our instructions.
    si = _system_instruction_text(llm_request)
    for i in (1, 2, 3):
        if f"literature searcher #{i}" in si:
            return f"lit_searcher_{i}"
    if "synthesize a research team" in si:
        return "lit_synthesizer"
    if "final deliverable" in si:
        return "report_writer"
    if "code-expressible experiment" in si:
        return "experiment_designer"
    if "analyze experiment results" in si:
        return "result_analyst"
    if "scientific critic" in si:
        return "critic"
    return "orchestrator"


class ScriptedLlm(BaseLlm):
    """Pops one scripted response per model call, keyed by calling agent."""

    @classmethod
    def supported_models(cls) -> list[str]:
        return [".*"]

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        agent = _agent_from_request(llm_request)
        queue = SCRIPTS.get(agent)
        assert queue, f"No scripted response left for agent {agent!r}"
        parts = queue.pop(0)
        yield LlmResponse(
            content=types.Content(role="model", parts=parts), turn_complete=True
        )


def patch_models(agent, mock: ScriptedLlm) -> None:
    """Swap every model in the tree (sub_agents AND AgentTool-wrapped agents)."""
    from google.adk.tools.agent_tool import AgentTool

    if hasattr(agent, "model"):
        agent.model = mock
    for sub in agent.sub_agents:
        patch_models(sub, mock)
    for tool in getattr(agent, "tools", []):
        if isinstance(tool, AgentTool):
            patch_models(tool.agent, mock)
