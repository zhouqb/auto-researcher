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
    record_experience,
    save_plan,
    search_arxiv,
    search_experiences,
    search_github,
    search_openalex,
    search_openreview,
    search_semantic_scholar,
    search_web,
    update_board,
    write_artifact,
)
from ..tools.codex import codex_exec, run_experiments

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
        tools=[
            search_openalex, search_arxiv, search_openreview, search_github,
            search_web, search_semantic_scholar, write_artifact,
        ],
        output_key=f"lit_notes_{index}",
        instruction=f"""You are literature searcher #{index} on a research team.

Your assigned facet: {{facet_{index}?}}

If the facet above is empty or missing, respond exactly "No facet assigned." and make no tool calls.

Otherwise:
1. Run 2-5 searches, varying keywords. {_TOOL_DISCIPLINE}
   Tool guide: search_openalex + search_arxiv are your primary paper indexes;
   search_openreview adds peer-review signal (venue decisions, review links)
   for ML venues; search_github finds implementations and adoption signal
   (stars) when the facet concerns a method or tool; search_web covers
   engineering blogs/docs the indexes miss (skip it if it reports it is not
   configured). Use search_semantic_scholar only if the others return errors.
2. Pick the 5-10 most relevant sources (favor recency and citation count;
   include seminal works where relevant).
3. Save notes with write_artifact to filename 'lit/facet_{index}/notes.md'
   (kind 'lit_notes'): the facet, one entry per source (title, authors, year,
   venue, citation count or stars, arXiv id or DOI, url, 2-4 sentence
   relevance note), then a "Key takeaways" section.
4. End with a structured summary as your final response: the facet, 3-6 key
   findings, the top 5 references (title, year, url), and the artifact filename.
Never invent sources; only cite what the search tools returned.""",
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


_IDEA_PERSONAS = {
    1: ("conservative", "literature-grounded, incremental: the safest approach "
        "most directly supported by the synthesis's strongest findings"),
    2: ("novel", "creative: an unconventional angle, cross-domain transfer, or "
        "a surprising combination the literature hints at but hasn't tested"),
    3: ("efficient", "minimalist: the cheapest, fastest experiment that would "
        "still produce a decisive signal on the core question"),
}


def _make_idea_generator(index: int, model: LiteLlm) -> LlmAgent:
    persona, style = _IDEA_PERSONAS[index]
    return LlmAgent(
        name=f"idea_generator_{index}",
        model=model,
        description=f"Generates {persona} experiment ideas.",
        disallow_transfer_to_parent=True,
        disallow_transfer_to_peers=True,
        tools=[read_artifact],
        output_key=f"idea_batch_{index}",
        instruction=f"""You are idea generator #{index} ({persona}) on a research team.
Your style: {style}.

Steps ({_TOOL_DISCIPLINE}):
1. read_artifact 'plan/plan.md', then 'lit/synthesis.md' (one per response).
2. Propose exactly 2 candidate experiment approaches for the plan's main
   question, true to your style. Constraints: code-expressible, runnable on a
   local machine in minutes, measurable against a baseline.
3. Final response — for each candidate: a short id (e.g. "{persona[:4]}-1"),
   one-sentence hypothesis, method sketch (3-5 lines), expected signal, and
   the main risk.""",
    )


def _build_idea_tournament(worker_model: LiteLlm) -> SequentialAgent:
    fanout = ParallelAgent(
        name="idea_fanout",
        description="Generates diverse experiment ideas in parallel.",
        sub_agents=[_make_idea_generator(i, worker_model) for i in (1, 2, 3)],
    )
    judge = LlmAgent(
        name="idea_judge",
        model=worker_model,
        description="Ranks candidate experiment ideas via pairwise comparison.",
        disallow_transfer_to_parent=True,
        disallow_transfer_to_peers=True,
        tools=[write_artifact],
        output_key="idea_ranking",
        instruction=f"""You judge candidate experiment ideas in a research tournament.

Candidates:
[conservative] {{idea_batch_1?}}
[novel] {{idea_batch_2?}}
[efficient] {{idea_batch_3?}}

1. Merge near-duplicate candidates first (note merges).
2. Rank by pairwise comparison on: decisiveness of the expected signal,
   validity (confounds, baseline quality), feasibility/cost, and novelty.
   For each pair you compare, note one-line reasoning — like a short
   scientific debate.
3. Save with write_artifact to 'iter_1/hypotheses.json' (kind 'hypothesis'):
   JSON with all candidates ({{"id", "hypothesis", "method", "persona",
   "rank", "rationale"}}) ordered by rank. {_TOOL_DISCIPLINE}
4. Final response: the ranked list (id + hypothesis + one-line rationale
   each) and which top candidates you recommend running as branches.""",
    )
    return SequentialAgent(
        name="idea_tournament",
        description=(
            "Generates diverse candidate experiment ideas in parallel "
            "(conservative/novel/efficient personas) and ranks them by "
            "pairwise LLM judgment. Output: ranked candidates saved to "
            "'iter_1/hypotheses.json'. Input: a one-line request."
        ),
        sub_agents=[fanout, judge],
    )


def _build_experiment_designer(model: LiteLlm) -> LlmAgent:
    return LlmAgent(
        name="experiment_designer",
        model=model,
        description=(
            "Designs the experiment branch(es) for the approved plan: writes "
            "'iter_1/exp_spec.md' (per-branch hypothesis, method, dataset, "
            "baseline, shared metric, budget estimate, stop conditions, and a "
            "ready-to-run Codex task prompt per branch). Input: a one-line "
            "request naming which ranked candidates to design (or 'single "
            "branch' for one), plus any user constraints."
        ),
        disallow_transfer_to_parent=True,
        disallow_transfer_to_peers=True,
        tools=[read_artifact, write_artifact, search_github],
        output_key="exp_spec_summary",
        instruction=f"""You design focused, code-expressible experiments.

Steps ({_TOOL_DISCIPLINE}):
1. read_artifact 'brief/research_brief.md', then 'plan/plan.md', then
   'lit/synthesis.md'; if your request names ranked candidates, also
   'iter_1/hypotheses.json' (one per response). You may use search_github
   once to locate a reference implementation worth citing in the spec
   (never as a dependency — experiments stay self-contained).
2. Design the smallest experiment per requested branch that meaningfully
   tests its hypothesis on the user's local machine (CPU-friendly unless
   told otherwise; minutes not hours; standard pip-installable deps or
   stdlib; small or synthetic datasets). All branches MUST share the same
   metric and baseline so results are comparable.
3. Save with write_artifact to 'iter_1/exp_spec.md' (kind 'exp_spec'):
   the shared metric, baseline, seeds, and success threshold; then one
   section per branch — "## Branch <id>" with hypothesis, method, stop
   conditions, estimated wallclock + rough Codex token cost, and a
   subsection "### Codex task prompt" containing COMPLETE, self-contained
   instructions for the implementing agent (it cannot see this conversation
   or other branches), including: implement in Python, run it, and write
   metrics.json + dual-format plots per AGENTS.md.
4. End with: a per-branch one-line summary, the TOTAL budget estimate
   (wallclock + tokens), and the artifact filename.""",
    )


def _build_result_analyst(model: LiteLlm) -> LlmAgent:
    return LlmAgent(
        name="result_analyst",
        model=model,
        description=(
            "Analyzes completed Codex experiment run(s): reads run results "
            "and metrics across branches, ranks them, writes "
            "'iter_1/analysis.md'. Input: the branch run_ids and a one-line "
            "request."
        ),
        disallow_transfer_to_parent=True,
        disallow_transfer_to_peers=True,
        tools=[read_artifact, list_artifacts, write_artifact],
        output_key="analysis_summary",
        instruction=f"""You analyze experiment results with scientific skepticism.

Steps ({_TOOL_DISCIPLINE}):
1. read_artifact 'iter_1/exp_spec.md'.
2. list_artifacts, then read_artifact each run result named in your request
   ('iter_1/exp_<branch>/runs/<run_id>/result.json'), one per response.
3. Save with write_artifact to 'iter_1/analysis.md' (kind 'analysis'):
   what ran per branch; a comparison table — branch × (metric, baseline,
   delta, status); ranking on the shared metric (AIDE-style: prune buggy/
   failed branches, identify the best); variance across seeds if present;
   validity concerns (confounds, tiny data, single seed, metric mismatch);
   per-branch verdict — supported / not supported / inconclusive; whether
   the best branch deserves a refinement round; suggested follow-ups.
4. End with a concise analysis summary (including the branch ranking) and
   the artifact filename.
Report failures honestly; never embellish numbers. A failed branch is a
finding, not an embarrassment.""",
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


def _build_critic(model: LiteLlm) -> LlmAgent:
    return LlmAgent(
        name="critic",
        model=model,
        description=(
            "Reviews the project's analysis and final report for validity "
            "before delivery: unsupported claims, data leakage/usage problems, "
            "metric mismatch, overgeneralization. Writes 'iter_1/critique.md'. "
            "Input: a one-line request."
        ),
        disallow_transfer_to_parent=True,
        disallow_transfer_to_peers=True,
        tools=[read_artifact, list_artifacts, write_artifact],
        output_key="critique_summary",
        instruction=f"""You are the project's scientific critic (design §14 guardrails).
Your job is to find problems, not to praise.

Steps ({_TOOL_DISCIPLINE}):
1. read_artifact 'reports/final_report.md'.
2. read_artifact 'iter_1/analysis.md' and 'iter_1/exp_spec.md' if they exist
   (check list_artifacts when unsure).
3. Check for, with severity (blocking | warning | note):
   - claims in the report not supported by the literature notes or metrics
   - DATA LEAKAGE signs: test data influencing training/tuning, metric
     computed on data the method saw, target leakage in features
   - DATA USAGE problems: dataset too small/synthetic for the claim's
     breadth, baseline mis-specified or missing
   - statistics: single-seed conclusions stated as robust, missing variance,
     metric/objective mismatch, p-hacking patterns
   - overgeneralization beyond the tested conditions; uncited numbers
4. Save with write_artifact to 'iter_1/critique.md' (kind 'critique'):
   one section per finding — severity, what, where (artifact + section),
   suggested fix. If genuinely clean, say so and note residual limitations.
5. End with: counts by severity and a one-line verdict
   (deliverable / needs-revision), plus the artifact filename.""",
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
            update_board,
            record_checkpoint,
            append_decision,
            search_experiences,
            record_experience,
            AgentTool(_build_literature_review(worker_model, settings.max_lit_facets)),
            AgentTool(_build_idea_tournament(worker_model)),
            AgentTool(_build_experiment_designer(worker_model)),
            codex_exec,
            run_experiments,
            AgentTool(_build_result_analyst(worker_model)),
            AgentTool(_build_report_writer(worker_model)),
            AgentTool(_build_critic(worker_model)),
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

3. PLAN. First call search_experiences with keywords from the question and
   intended method — past failures tell you what to avoid, past successes
   what to reuse; mention relevant hits in the plan ("avoiding X, which
   caused Y in a prior project"). Then call save_plan with (a) the full plan
   markdown — objectives, key questions, literature facets, experiment
   outline (if in scope), success criteria, report outline — and
   (b) lit_facets: 2-{settings.max_lit_facets} distinct, independently
   searchable literature sub-questions. Then call update_board once with the
   plan's work items (lit facets, experiment, report), all status 'backlog'.

4. GATE 1 — plan approval. Present a concise plan summary in chat (facets as
   a bullet list; experiment yes/no) and ask the user explicitly to approve
   or request changes. NEVER proceed without explicit approval. On change
   requests, revise and call save_plan again, then re-ask. Once approved:
   record_checkpoint(gate='plan_approval', decision='approved',
   comments=<user's words>), then append_decision (evidence: 'plan/plan.md').

5. LITERATURE. Call the literature_review tool. Then give the user a short
   synthesis summary (reference 'lit/synthesis.md').

6. EXPERIMENT (only if in scope per the brief):
   a. Decide breadth: for a simple, well-defined question one branch is
      right; for open questions with several plausible approaches, run the
      idea_tournament tool and pick the top 2-{settings.max_experiment_branches}
      ranked candidates as branches. State your choice and why. If the user
      explicitly asks for the tournament (or for specific branches), their
      request overrides your judgment — run it.
   b. Call experiment_designer (name the chosen candidates, or 'single
      branch'), then present its per-branch spec summary AND TOTAL budget
      estimate (wallclock + tokens) to the user.
   c. GATE 2 — budget approval. Ask the user explicitly to approve the
      experiment budget. NEVER call codex_exec or run_experiments without
      it. Once approved: record_checkpoint(gate='budget_approval',
      decision='approved', comments=<user's words>).
   d. Read 'iter_1/exp_spec.md' for the verbatim Codex task prompts, then:
      one branch → codex_exec; several branches → run_experiments with
      [{{branch_id, task_prompt}} ...]. Branches run in parallel, each in an
      isolated workspace.
   e. If a branch failed: inspect the error; you may call codex_exec once
      with branch_id=<branch> and resume_thread_id=<thread_id> and concise
      fix instructions. If it fails again, append_decision recording the
      failure and treat the branch as pruned (a failed branch is a finding).
   f. Call result_analyst (list every branch and run_id in your request),
      then present its comparison/ranking summary. If it recommends refining
      the winning branch and budget allows, you may run one refinement via
      codex_exec (branch_id=<winner>, resume_thread_id=<its thread_id>)
      — but ask the user first if it needs meaningfully more budget.
   g. Call record_experience ONCE PER BRANCH with that branch's hypothesis,
      outcome (success / failure / inconclusive / aborted), concrete
      lessons, method, result, and codex thread_id. Failures and
      inconclusive runs are the most valuable memory.

7. REPORT. Call report_writer.

8. CRITIQUE (guardrail). Call critic. If it reports BLOCKING findings, call
   report_writer once more with the specific fixes required, then re-run
   critic; surface anything still blocking to the user honestly instead of
   shipping it. Then give the user the report's conclusions, the critique
   verdict, and both filenames.

Board upkeep: after each stage completes (literature, experiment, analysis,
report), call update_board moving the corresponding items to 'done' (or
'killed'/'blocked' with a status_reason).

Steering: at any point the user may redirect, kill the experiment, or change
the plan — record significant redirections with append_decision.

Style: be concise; reference artifacts by filename instead of pasting them;
never fabricate results or citations.""",
    )
