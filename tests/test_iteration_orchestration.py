"""The orchestrator exposes the multi-iteration loop (structural, no LLM call)."""

from __future__ import annotations

from deep_researcher.agents.root import build_root_agent
from deep_researcher.tools.codex import advance_iteration


def test_advance_iteration_is_a_root_tool():
    agent = build_root_agent()
    assert advance_iteration in agent.tools


def test_orchestrator_instruction_describes_the_loop():
    instr = build_root_agent().instruction
    # the gated iteration loop + carry-forward + stop conditions are documented
    assert "ITERATE" in instr
    assert "advance_iteration" in instr
    assert "GATE 3" in instr
    # stop condition referenced (max_iterations is interpolated to its number)
    assert "STOP" in instr and "iteration" in instr.lower()


def test_stage_agents_carry_the_iteration_note():
    # the experiment-stage sub-agents are told to remap iter_1 -> iter_<N>
    agent = build_root_agent()
    noted = 0
    for tool in agent.tools:
        sub = getattr(tool, "agent", None)
        if sub is not None and "iter_<N>" in (getattr(sub, "instruction", "") or ""):
            noted += 1
    assert noted >= 3  # result_analyst, critic, report_writer at minimum
