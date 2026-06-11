"""Phase 0 agent tree (design §3 subset, mirroring the ADK deep-search sample).

    orchestrator (LlmAgent, root — owns the decision thread)
    │   clarify (≤3 questions) → research_brief.md → plan.md → Gate 1 approval
    └── research_pipeline (SequentialAgent)
        ├── lit_fanout (ParallelAgent)
        │   ├── lit_searcher_1..N   (isolated contexts, unique state keys)
        ├── lit_synthesizer         → lit/synthesis.md
        └── report_writer           → reports/final_report.md

A single orchestrator owns decisions (Cognition's continuity principle);
parallelism only where branches are independent (literature facets).

DeepSeek note: multi-tool-call responses through LiteLLM parse unreliably
(google/adk-python#5024), so every instruction demands one tool call per
response.
"""

from __future__ import annotations

from google.adk.agents import LlmAgent, ParallelAgent, SequentialAgent
from google.adk.models.lite_llm import LiteLlm

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


def build_root_agent() -> LlmAgent:
    settings = get_settings()
    orchestrator_model = LiteLlm(model=settings.orchestrator_model)
    worker_model = LiteLlm(model=settings.worker_model)
    n = settings.max_lit_facets

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

    report_writer = LlmAgent(
        name="report_writer",
        model=worker_model,
        description="Composes the final cited research report.",
        disallow_transfer_to_parent=True,
        disallow_transfer_to_peers=True,
        tools=[read_artifact, write_artifact, append_decision],
        instruction=f"""You write the final research report.

Steps ({_TOOL_DISCIPLINE}):
1. read_artifact 'brief/research_brief.md' (the contract — answer exactly this).
2. read_artifact 'plan/plan.md' (follow its report outline).
3. read_artifact 'lit/synthesis.md'; read facet notes if you need more depth.
4. Save with write_artifact to 'reports/final_report.md' (kind 'report'):
   executive summary; background; findings organized by the plan's key
   questions, with inline numeric citations like [3]; discussion (open
   problems, promising directions); a References section listing every cited
   work (title, authors, year, venue, url). Cite only papers present in the
   literature notes — never invent references. Use $...$ KaTeX for math.
5. append_decision recording that the report was completed (evidence:
   'reports/final_report.md').
End with: a 5-10 sentence summary of the report's conclusions, the artifact
filename, and the number of references.""",
    )

    research_pipeline = SequentialAgent(
        name="research_pipeline",
        description=(
            "Executes the approved research plan: parallel literature search "
            "over the plan's facets, synthesis, then a cited final report."
        ),
        sub_agents=[lit_fanout, lit_synthesizer, report_writer],
    )

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
        ],
        sub_agents=[research_pipeline],
        instruction=f"""You are the Principal Orchestrator of Deep Researcher, a
human-steered research system. The user is the PI; you propose and execute.
The conversation is ephemeral; the artifacts are the project — every document
you produce is saved via tools, never only pasted into chat.

Workflow — follow strictly, one step at a time ({_TOOL_DISCIPLINE}):

1. CLARIFY. On receiving the research question, ask at most
   {settings.max_clarifying_questions} high-value clarifying questions in ONE
   message (scope, success criteria, constraints, intended depth) — and only
   the ones that genuinely change the work. If nothing needs clarifying, or
   the user declines to answer, state the defaults you are assuming.

2. BRIEF. Save the research brief with write_artifact to
   'brief/research_brief.md' (kind 'brief'): clarified question, scope and
   non-goals, success criteria, assumed defaults. This is the project
   contract. Tell the user it is saved, in one line.

3. PLAN. Call save_plan with (a) the full plan markdown — objectives, key
   questions, literature facets, success criteria, report outline — and
   (b) lit_facets: 2-{settings.max_lit_facets} distinct, independently
   searchable literature sub-questions.

4. GATE 1 — plan approval. Present a concise plan summary in chat (facets as
   a bullet list) and ask the user explicitly to approve or request changes.
   NEVER proceed without explicit approval. On change requests, revise and
   call save_plan again, then re-ask.

5. EXECUTE. Once the user approves: first record_checkpoint(gate=
   'plan_approval', decision='approved', comments=<user's words>); then
   append_decision summarizing what was approved (evidence: 'plan/plan.md');
   then transfer_to_agent to 'research_pipeline'.

Style: be concise; reference artifacts by filename instead of pasting them;
never fabricate results or citations.""",
    )
