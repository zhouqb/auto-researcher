"""Agent tree (design §3), Phase 1: orchestrator drives stages via AgentTools.

    orchestrator (LlmAgent, root — owns the decision thread and ALL gates)
    ├── tool: literature_review (SequentialAgent: parallel fanout → synthesis)
    ├── tool: experiment_designer (LlmAgent → iter_1/exp_spec.md)
    ├── tool: codex_exec (sandboxed Codex experiment, after Gate 2)
    ├── tool: result_analyst (LlmAgent → iter_1/analysis.md)
    └── tool: report_writer (LlmAgent → reports/final_report.md)

Stage agents are wrapped as AgentTools rather than transfer targets so that
every human gate (plan approval, experiment budget) happens in the root chat
thread — single decision owner (Cognition's continuity principle), parallelism
only inside genuinely independent literature facets.

DeepSeek note: multi-tool-call responses through LiteLLM parse unreliably
(google/adk-python#5024), so every instruction demands one tool call per
response.
"""

from __future__ import annotations

from google.adk.agents import LlmAgent, ParallelAgent, SequentialAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.agent_tool import AgentTool

from ..config import get_settings
from ..tools import (
    append_decision,
    list_artifacts,
    read_artifact,
    record_checkpoint,
    save_plan,
    search_arxiv,
    search_openalex,
    search_semantic_scholar,
    write_artifact,
)
from ..tools.codex import codex_exec

_TOOL_DISCIPLINE = (
    "Call at most ONE tool per response; wait for its result before the next call."
)


def _make_lit_searcher(index: int, model: LiteLlm) -> LlmAgent:
    return LlmAgent(
        name=f"lit_searcher_{index}",
        model=model,
        description=f"Literature searcher for facet #{index} of the research plan.",
        disallow_transfer_to_parent=True,
        disallow_transfer_to_peers=True,
        tools=[search_openalex, search_arxiv, search_semantic_scholar, write_artifact],
        output_key=f"lit_notes_{index}",
        instruction=f"""You are literature searcher #{index} on a research team.

Your assigned facet: {{facet_{index}?}}

If the facet above is empty or missing, respond exactly "No facet assigned." and make no tool calls.

Otherwise:
1. Run 2-4 searches with search_openalex and search_arxiv, varying keywords.
   {_TOOL_DISCIPLINE} Use search_semantic_scholar only if the others return errors.
2. Pick the 5-10 most relevant papers (favor recency and citation count; include
   seminal works where relevant).
3. Save notes with write_artifact to filename 'lit/facet_{index}/notes.md'
   (kind 'lit_notes'): the facet, one entry per paper (title, authors, year,
   venue, citation count, arXiv id or DOI, url, 2-4 sentence relevance note),
   then a "Key takeaways" section.
4. End with a structured summary as your final response: the facet, 3-6 key
   findings, the top 5 references (title, year, url), and the artifact filename.
Never invent papers; only cite what the search tools returned.""",
    )


def _build_literature_review(worker_model: LiteLlm, n: int) -> SequentialAgent:
    lit_fanout = ParallelAgent(
        name="lit_fanout",
        description="Runs literature searchers over distinct facets in parallel.",
        sub_agents=[_make_lit_searcher(i + 1, worker_model) for i in range(n)],
    )
    summaries_block = "\n".join(
        f"[facet {i + 1}] {{lit_notes_{i + 1}?}}" for i in range(n)
    )
    lit_synthesizer = LlmAgent(
        name="lit_synthesizer",
        model=worker_model,
        description="Merges per-facet literature notes into one synthesis.",
        disallow_transfer_to_parent=True,
        disallow_transfer_to_peers=True,
        tools=[read_artifact, write_artifact],
        output_key="lit_synthesis",
        instruction=f"""You synthesize a research team's literature findings.

Searcher summaries:
{summaries_block}

1. Where a summary is too thin, load the full notes with read_artifact
   ('lit/facet_<i>/notes.md'). {_TOOL_DISCIPLINE}
2. Save the synthesis with write_artifact to 'lit/synthesis.md' (kind
   'lit_notes'): cross-facet themes, agreements and contradictions, gaps and
   opportunities, and a consolidated numbered reference list (title, authors,
   year, venue, url).
3. End with a concise synthesis summary and the artifact filename.
Keep every citation traceable to the searchers' notes; never invent papers.""",
    )
    return SequentialAgent(
        name="literature_review",
        description=(
            "Runs the full literature stage for the approved plan: parallel "
            "searchers over the plan's facets, then a synthesis saved to "
            "'lit/synthesis.md'. Input: a one-line request, e.g. 'run the "
            "literature review for the approved plan'."
        ),
        sub_agents=[lit_fanout, lit_synthesizer],
    )


def _build_experiment_designer(model: LiteLlm) -> LlmAgent:
    return LlmAgent(
        name="experiment_designer",
        model=model,
        description=(
            "Designs the experiment for the approved plan: writes "
            "'iter_1/exp_spec.md' (hypothesis, method, dataset, baseline, "
            "metric, budget estimate, stop conditions, and a ready-to-run "
            "Codex task prompt). Input: a one-line request, optionally with "
            "constraints from the user."
        ),
        disallow_transfer_to_parent=True,
        disallow_transfer_to_peers=True,
        tools=[read_artifact, write_artifact],
        output_key="exp_spec_summary",
        instruction=f"""You design ONE focused, code-expressible experiment.

Steps ({_TOOL_DISCIPLINE}):
1. read_artifact 'brief/research_brief.md', then 'plan/plan.md', then
   'lit/synthesis.md' (one per response).
2. Design the smallest experiment that meaningfully tests the plan's main
   hypothesis on the user's local machine (CPU-friendly unless told
   otherwise; minutes not hours; standard pip-installable deps; small or
   synthetic datasets).
3. Save with write_artifact to 'iter_1/exp_spec.md' (kind 'exp_spec'):
   hypothesis; method; dataset; baseline; metric + success threshold; seeds;
   stop conditions; estimated wallclock and rough Codex token cost; and a
   section titled "## Codex task prompt" containing COMPLETE, self-contained
   instructions for the implementing agent (it cannot see this conversation),
   including: implement in Python, run it, and write metrics.json + dual-format
   plots per AGENTS.md.
4. End with: a 5-10 line spec summary, the budget estimate (wallclock +
   tokens), and the artifact filename.""",
    )


def _build_result_analyst(model: LiteLlm) -> LlmAgent:
    return LlmAgent(
        name="result_analyst",
        model=model,
        description=(
            "Analyzes a completed Codex experiment run: reads the run result "
            "and metrics, writes 'iter_1/analysis.md'. Input: the run_id and "
            "a one-line request."
        ),
        disallow_transfer_to_parent=True,
        disallow_transfer_to_peers=True,
        tools=[read_artifact, list_artifacts, write_artifact],
        output_key="analysis_summary",
        instruction=f"""You analyze experiment results with scientific skepticism.

Steps ({_TOOL_DISCIPLINE}):
1. read_artifact 'iter_1/exp_spec.md'.
2. list_artifacts, then read_artifact the run result
   ('iter_1/exp_main/runs/<run_id>/result.json' — run_id is in your request).
3. Save with write_artifact to 'iter_1/analysis.md' (kind 'analysis'):
   what ran; metric vs. baseline vs. the spec's success threshold; variance
   across seeds if present; validity concerns (confounds, tiny data, single
   seed, metric mismatch); verdict — supported / not supported / inconclusive;
   suggested follow-ups.
4. End with a concise analysis summary and the artifact filename.
Report failures honestly; never embellish numbers.""",
    )


def _build_report_writer(model: LiteLlm) -> LlmAgent:
    return LlmAgent(
        name="report_writer",
        model=model,
        description=(
            "Composes the final cited research report from the brief, plan, "
            "literature synthesis, and (if present) experiment analysis. "
            "Input: a one-line request."
        ),
        disallow_transfer_to_parent=True,
        disallow_transfer_to_peers=True,
        tools=[read_artifact, list_artifacts, write_artifact, append_decision],
        instruction=f"""You write the final research report.

Steps ({_TOOL_DISCIPLINE}):
1. read_artifact 'brief/research_brief.md' (the contract — answer exactly this).
2. read_artifact 'plan/plan.md' (follow its report outline).
3. read_artifact 'lit/synthesis.md'; read facet notes if you need more depth.
4. If the project ran an experiment, read 'iter_1/exp_spec.md' and
   'iter_1/analysis.md' too (check list_artifacts when unsure).
5. Save with write_artifact to 'reports/final_report.md' (kind 'report'):
   executive summary; background; findings organized by the plan's key
   questions, with inline numeric citations like [3]; an Experiment section
   (method, results table, verdict) when one exists; discussion (open
   problems, promising directions); a References section listing every cited
   work (title, authors, year, venue, url). Cite only papers present in the
   literature notes — never invent references. Use $...$ KaTeX for math.
6. append_decision recording that the report was completed (evidence:
   'reports/final_report.md').
End with: a 5-10 sentence summary of the report's conclusions, the artifact
filename, and the number of references.""",
    )


def build_root_agent() -> LlmAgent:
    settings = get_settings()
    orchestrator_model = LiteLlm(model=settings.orchestrator_model)
    worker_model = LiteLlm(model=settings.worker_model)

    return LlmAgent(
        name="orchestrator",
        model=orchestrator_model,
        description="Principal orchestrator of the Deep Researcher project.",
        tools=[
            write_artifact,
            read_artifact,
            list_artifacts,
            save_plan,
            record_checkpoint,
            append_decision,
            AgentTool(_build_literature_review(worker_model, settings.max_lit_facets)),
            AgentTool(_build_experiment_designer(worker_model)),
            codex_exec,
            AgentTool(_build_result_analyst(worker_model)),
            AgentTool(_build_report_writer(worker_model)),
        ],
        instruction=f"""You are the Principal Orchestrator of Deep Researcher, a
human-steered research system. The user is the PI; you propose and execute.
The conversation is ephemeral; the artifacts are the project — every document
you produce is saved via tools, never only pasted into chat.

Workflow — follow strictly, one step at a time ({_TOOL_DISCIPLINE}):

1. CLARIFY. On receiving the research question, ask at most
   {settings.max_clarifying_questions} high-value clarifying questions in ONE
   message (scope, success criteria, constraints, whether a code experiment
   is wanted) — and only the ones that genuinely change the work. If nothing
   needs clarifying, or the user declines to answer, state the defaults you
   are assuming.

2. BRIEF. Save the research brief with write_artifact to
   'brief/research_brief.md' (kind 'brief'): clarified question, scope and
   non-goals, success criteria, whether an experiment is in scope, assumed
   defaults. This is the project contract. Tell the user it is saved, in one
   line.

3. PLAN. Call save_plan with (a) the full plan markdown — objectives, key
   questions, literature facets, experiment outline (if in scope), success
   criteria, report outline — and (b) lit_facets: 2-{settings.max_lit_facets}
   distinct, independently searchable literature sub-questions.

4. GATE 1 — plan approval. Present a concise plan summary in chat (facets as
   a bullet list; experiment yes/no) and ask the user explicitly to approve
   or request changes. NEVER proceed without explicit approval. On change
   requests, revise and call save_plan again, then re-ask. Once approved:
   record_checkpoint(gate='plan_approval', decision='approved',
   comments=<user's words>), then append_decision (evidence: 'plan/plan.md').

5. LITERATURE. Call the literature_review tool. Then give the user a short
   synthesis summary (reference 'lit/synthesis.md').

6. EXPERIMENT (only if in scope per the brief):
   a. Call experiment_designer, then present its spec summary AND budget
      estimate (wallclock + tokens) to the user.
   b. GATE 2 — budget approval. Ask the user explicitly to approve the
      experiment budget. NEVER call codex_exec without it. Once approved:
      record_checkpoint(gate='budget_approval', decision='approved',
      comments=<user's words>).
   c. Call codex_exec with the COMPLETE Codex task prompt from
      'iter_1/exp_spec.md' (read it first if you don't have it verbatim).
   d. If the run failed: inspect the error; you may call codex_exec once more
      with resume_thread_id=<thread_id> and concise fix instructions. If it
      fails again, append_decision recording the failure and move on.
   e. Call result_analyst (include the run_id in your request), then present
      its analysis summary.

7. REPORT. Call report_writer. Then give the user the report's conclusions
   and filename, and append_decision is already handled by the writer.

Steering: at any point the user may redirect, kill the experiment, or change
the plan — record significant redirections with append_decision.

Style: be concise; reference artifacts by filename instead of pasting them;
never fabricate results or citations.""",
    )
