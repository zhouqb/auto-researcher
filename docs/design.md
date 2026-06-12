# Deep Researcher: Technical Design

**An ADK-orchestrated, Codex-backed, human-steered autonomous research agent**
*Consolidated design (v3) — merges the original architecture doc and the v2 addendum (artifact subsystem, fully local stack, steering-first UI).*

---

## 1. Executive Summary

Deep Researcher is a long-running, locally hosted multi-agent system that takes an ML/AI/SWE research question and executes the full research lifecycle — clarification, planning, literature review, idea generation, experiment design, code implementation and execution, result analysis, critique, plan revision, and report writing — **under continuous human steering**. It goes beyond literature synthesis: literature search supplies *inspiring ideas*, and OpenAI Codex *runs real code experiments* on the user's compute to validate them.

Three principles anchor the design:

1. **Parallel exploration.** Recent evidence (Google AI co-scientist, Anthropic's multi-agent research system, AI Scientist-v2 / AIDE tree search) shows parallel research branches with principled selection materially outperform single-threaded agents.
2. **Experience memory retaining both successes and failures.** Reflexion, ExpeL, and Manus's production lessons all show that contrasting failed and successful trajectories — and keeping failure traces visible — is load-bearing for compounding capability.
3. **The conversation is ephemeral; the artifacts are the project.** Every meaningful output — research, engineering, *and logistics* (briefs, plans, decision logs, budgets) — is a registered, versioned, lineage-tracked artifact. Agents communicate by reference, never by pasting large content into context.

The stack is deliberately boring: Google's Agent Development Kit (ADK) for orchestration (driven by a Gemini Developer API key — **no GCP/Vertex/AWS anywhere**), SQLite + the local filesystem for all state, OpenAI Codex in non-interactive `exec` mode (through the user's ChatGPT subscription) as the implementation subagent, and a rich web UI (chat + board + artifact browser + run monitor) as the control plane.

### Goals

- End-to-end support for an ML/AI/SWE research project with rigorous human checkpoints (plan approval, experiment-budget approval, mid-run review) **and** continuous steering (interrupt, redirect, kill-branch, edit-plan) between checkpoints.
- Parallel researcher branches with principled selection (metric-driven pruning; LLM-judge/Elo tournament where no metric exists).
- Durable artifacts (code, data, results, reports, **and** logistic documents: design doc, plan, progress board, decision log, budget ledger) with full provenance lineage.
- Durable experience memory (success + failure trajectories) retrieved at planning time.
- Crash recovery and pause/resume for tasks that run hours to days.
- Codex as the sole code-writing/execution backend via `codex exec`/SDK.
- A UI that renders rich content: markdown, math (KaTeX), plots (Vega-Lite), diagrams (Mermaid), code diffs.

### Non-Goals

- Full autonomy. This is a PI-and-lab model: the user steers heavily; the system proposes and executes.
- Cloud infrastructure. Everything runs locally; backup is copying a folder.
- Physical-science lab automation (code-expressible experiments only).
- Foundation-model pretraining; the system orchestrates experiments, it does not train LLMs from scratch.

---

## 2. Survey of Related Work (with design takeaways)

### 2.1 AI-scientist / autonomous research systems

- **Sakana AI Scientist v1 → v2** (arXiv:2408.06292; arXiv:2504.08066). v2 removes human-authored templates and uses *progressive agentic tree search* managed by an experiment-manager agent, with a VLM feedback loop on figures; it produced the first fully AI-generated paper to pass peer review at an ICLR 2025 workshop. v2 explores multiple independent search trees in parallel; nodes are classified buggy/non-buggy, with debugging on buggy nodes and refinement on good ones. **Takeaway:** tree search + experiment-manager + explicit buggy/non-buggy bookkeeping.
- **AIDE** (arXiv:2502.13138, Weco AI). Frames ML engineering as code optimization via tree search; each node is a script version, edges are improvement steps; valid nodes are refined, broken nodes fixed or pruned. With o1-preview it earned medals in 16.9% of MLE-bench competitions vs. 4.4% for OpenHands, rising to 34.1% at pass@8. **Takeaway:** solution-tree search with independent node evaluation is the canonical experiment-search structure (AI Scientist-v2's search is built on it).
- **Google MLE-STAR** (arXiv:2506.15692; NeurIPS 2025). Uses web search to retrieve effective models for an initial solution, then *targeted code-block refinement guided by ablation studies*, plus ensembling and robustness modules (debugger, data-leakage checker, data-usage checker). Reported medals in ~64% of MLE-Bench-Lite competitions (36% gold); same-backbone (Gemini-2.0-Flash) it lifts AIDE's any-medal rate from 25.8% to 43.9%. **Crucially, MLE-STAR is open-sourced as the ADK `machine-learning-engineering` sample** — the closest existing reference for this design's experiment engine. **Takeaway:** search-for-ideas + ablation-driven targeted refinement maps directly onto "literature search for inspiration + iterative experiment refinement."
- **Agent Laboratory + AgentRxiv** (arXiv:2501.04227; arXiv:2503.18102). Role-specialized agents (PhD/Postdoc/ML Engineer/Professor) run review → experiment → report; *human feedback at each stage significantly improved quality*; co-pilot mode supports checkpoints; AgentRxiv lets agents build on prior agent research. **Takeaway:** role specialization + HITL checkpoints + a shared cumulative knowledge base.
- **Google AI co-scientist** (arXiv:2502.18864; Nature 2026). A Supervisor dispatches specialized agents (Generation, Reflection, Ranking, Evolution, Proximity, Meta-review) *in parallel*; the Ranking agent runs an **Elo tournament with pairwise simulated scientific debates**; Elo quality scales with test-time compute. **Takeaway:** the canonical pattern for parallel idea generation + tournament selection + evolution.
- **AlphaEvolve** (arXiv:2506.13131, DeepMind). Evolutionary coding agent: Gemini Flash (breadth) + Pro (depth) propose code edits, automated evaluators score, evolution keeps the best. **Takeaway:** evaluator-in-the-loop search over code; model-ensemble diversity.
- **Anthropic's multi-agent research system.** Orchestrator-worker: a lead agent plans, persists the plan to memory, spawns 3–5 parallel subagents with isolated contexts, then a citation pass. Outperformed single-agent Claude Opus 4 by 90.2% on their internal eval, at ~15× tokens — and token usage alone explained ~80% of performance variance. **Takeaway:** orchestrator-worker + context isolation + plan persistence; budget the agent count.
- **Open-source references:** LangChain `open_deep_research`; `gemini-fullstack-langgraph-quickstart` (query-gen → research → reflection/gap-check → iterate); ADK `deep-search` / `gemini-fullstack` samples (interactive planner → plan approval → `SequentialAgent` pipeline → `LoopAgent` critic refinement → cited report). **Takeaway:** the ADK deep-search sample is a near-drop-in skeleton for the non-experiment portion of the pipeline.

### 2.2 Evidence for parallel exploration

Anthropic's 90.2% lift; co-scientist's compute-scaling Elo gains; AIDE's ~4× medal advantage from tree search; AI Scientist-v2's parallel trees. **Counterpoint — Cognition's "Don't Build Multi-Agents":** multi-agent setups fragment context and produce conflicting decisions; share as much context as possible and avoid splitting decisions that can conflict. **Design resolution:** parallelize only where branches are *genuinely independent* (literature facets, separate experiment branches) while a single orchestrator owns the decision thread, with rich structured summaries — not lossy one-liners — passed across boundaries.

### 2.3 Evidence for retaining success AND failure experiences

- **Reflexion** (arXiv:2303.11366): verbal self-reflection distills failed trajectories into self-hints for later trials.
- **ExpeL** (arXiv:2308.10144): gathers success *and* failure trajectories and contrasts pairs to extract reusable insights; both example retrieval and insight extraction improve success.
- **Voyager / AWM / Buffer-of-Thoughts / A-MEM:** external skill libraries and reusable workflow/thought templates.
- **Manus context-engineering lessons:** *keep* failed actions and stack traces in context so the model's prior shifts away from repeating mistakes; recite goals via `todo.md`; treat the file system as unlimited external context.
- **Memory surveys** (arXiv:2404.13501 and successors) flag the central risk of reflective memory: *self-reinforcing error* — requiring confidence scores, contradiction checks, supersede links, and expiry.

### 2.4 Context engineering for long-horizon agents

Anthropic's prescription: **compaction** (summarize near the limit; start with tool-result clearing), **structured note-taking** (NOTES.md-style external memory), and **sub-agent context isolation**. Manus adds KV-cache stability (stable prefixes, append-only context) and recitation. Cognition adds the continuity caution and a dedicated compaction model for hand-offs. **Takeaway:** offload large outputs to artifacts and pass references; compact at stage boundaries; recite the plan; isolate sub-agent contexts; keep recent turns raw.

---

## 3. High-Level Architecture

Top-level orchestration is a `LoopAgent` ("Research Loop") wrapping a `SequentialAgent` ("Iteration Pipeline"), seeded by an intake phase and terminated by a reporting phase. A single **Principal Orchestrator** (`LlmAgent`, root) owns the decision thread and delegates; this honors Cognition's continuity principle while fanning out genuinely independent work to `ParallelAgent`s.

```
RootApp (App, ResumabilityConfig(is_resumable=True))
└── PrincipalOrchestrator (LlmAgent, root_agent)
    ├── IntakePhase (SequentialAgent)
    │   ├── ClarifierAgent (LlmAgent)                 # asks ≤N clarifying questions (default 3)
    │   │     └── ask_user (LongRunningFunctionTool)  # HITL pause
    │   └── ScopeWriterAgent (LlmAgent → artifact: research_brief.md)
    │
    ├── ResearchLoop (LoopAgent, max_iterations=K, exit via escalate=True)
    │   └── IterationPipeline (SequentialAgent)
    │       ├── PlannerAgent (LlmAgent → artifacts: plan.md, board.json)
    │       │     └── request_plan_approval (LongRunningFunctionTool)   # Gate 1
    │       ├── LiteratureFanout (ParallelAgent)
    │       │     ├── LitSearcher_A … _N (LlmAgent + search tools)      # parallel, isolated
    │       ├── LitSynthesizer (LlmAgent → artifact: lit notes)
    │       ├── HypothesisGenerator (ParallelAgent → N idea branches)
    │       ├── IdeaTournament (CustomAgent: LLM-judge / Elo ranking)
    │       ├── ExperimentDesigner (LlmAgent → artifact: exp_spec per branch)
    │       │     └── request_budget_approval (LongRunningFunctionTool) # Gate 2
    │       ├── ExperimentFanout (ParallelAgent over selected branches)
    │       │     └── CodexExperimentAgent (per branch)
    │       │           └── codex_exec (LongRunningFunctionTool)        # Codex subagent
    │       ├── ResultAnalyst (LlmAgent → artifact: analysis.md)
    │       │     └── midrun_review (LongRunningFunctionTool, optional) # Gate 3
    │       ├── CritiqueAgent (LlmAgent → artifact: critique)
    │       └── PlanReviser (LlmAgent: appends to decisions.md; sets escalate=True if converged)
    │             └── writes experience records (success+failure) to memory
    │
    └── ReportingPhase (SequentialAgent)
        ├── SectionPlanner → SectionWriters → CitationPass
        └── FinalReportComposer (LlmAgent → artifact: final_report.md)
```

Component notes:

- **Question intake + clarifier.** `ClarifierAgent` emits at most *N* (default 3) high-value clarifying questions in a single batch, mirroring the ADK deep-search sample's interactive planner. Questions surface via an `ask_user` `LongRunningFunctionTool` that pauses the invocation. If declined, defaults are assumed and recorded in `research_brief.md` — the project's contract, re-versioned only with explicit user approval.
- **Planner.** Produces a structured plan (objectives, hypotheses, experiment list, success metrics, budget estimate) as `plan.md` + machine-readable `board.json`. Gated by approval.
- **Literature review (parallel).** A `ParallelAgent` fans out N `LitSearcher` agents over distinct facets. Each writes to a *unique* state key (ADK `ParallelAgent` branches share session state — unique keys avoid the documented race condition). `LitSynthesizer` merges. This is Anthropic's orchestrator-worker pattern with isolated contexts.
- **Idea generation + tournament.** Parallel candidate ideas; tournament ranking (§4).
- **Experiment design + Codex execution.** `ExperimentDesigner` emits an `exp_spec` per selected branch (dataset, baseline, metric, compute budget, stop conditions); `CodexExperimentAgent` drives Codex (§13). Gated by budget approval.
- **Analysis, critique, revision.** `ResultAnalyst` interprets metrics/plots; `CritiqueAgent` (co-scientist Reflection-agent analog) checks validity; `PlanReviser` either iterates or signals convergence (`escalate=True` exits the loop). Every iteration appends to `decisions.md` and writes structured success/failure experiences to memory.
- **Report generation.** Mirrors the ADK deep-search sample: outline → section research/critique loop → composed, cited report emitted as a versioned artifact.

---

## 4. Parallel Exploration Design

- **Mechanism.** ADK `ParallelAgent` for in-process fan-out (literature facets, hypothesis branches, independent experiment branches). Each sub-agent runs in its own execution branch with an isolated context and writes to unique state keys. For experiments, parallelism also means multiple concurrent `codex exec` processes, bounded by `max_codex_concurrency` (subscription rate limits, §13).
- **Branch selection.** Two regimes:
  - *Metric-driven* (experiments): rank by validation metric, prune buggy/under-performing branches, refine the best — AIDE/MLE-STAR-style best-first tree search.
  - *LLM-as-judge* (hypotheses, where no metric exists): co-scientist-style pairwise "debates" judged by an LLM with Elo updates; cluster near-duplicates first (Proximity-agent analog) to avoid wasting budget on redundant candidates.
- **Resource budgeting.** Given Anthropic's ~15× token multiplier and Codex quotas, expose explicit knobs: `max_parallel_lit`, `max_idea_branches`, `max_experiment_branches`, `max_codex_concurrency`, plus per-iteration token/credit ceilings. Default small (3–5 parallel units, matching Anthropic's subagent count); escalate parallelism only after early branches show promise.
- **Topology stance.** Workers never talk to each other; only the orchestrator integrates results. This bounds coordination complexity and addresses Cognition's conflicting-decision concern.

---

## 5. Memory Design

Three tiers, all local:

| Tier | Mechanism | Scope / lifetime | Contents |
|---|---|---|---|
| Short-term working | ADK Session `State` (`output_key`; `temp:`/`user:`/`app:` prefixes) on SQLite | Current invocation/iteration | plan ref, exp_spec refs, analysis refs, branch keys |
| Working notes | File artifacts (`todo.md`, `NOTES.md` per project) | Project lifetime | recitation scratchpad, orchestrator notes |
| **Experience store** | **SQLite tables + FTS5** (optional `sqlite-vec`/local Chroma later), surfaced via a `search_experiences` tool | Permanent, cross-project | structured success+failure experiment trajectories |

**Experience record schema (retains success AND failure):**

```json
{
  "experience_id": "uuid",
  "project_id": "…", "iteration": 3, "branch": "B2",
  "hypothesis": "…",
  "method": {"dataset": "…", "model": "…", "key_hparams": {"…": "…"},
             "code_artifact_ref": "art_…"},
  "result": {"metric": "val_acc", "value": 0.873, "baseline": 0.851,
             "plots_ref": "art_…"},
  "outcome": "success | failure | inconclusive | aborted",
  "failure_mode": "OOM | data_leakage | no_improvement | bug | divergence | null",
  "lessons": "Targeted refinement of the augmentation block helped; larger batch caused OOM on this GPU.",
  "codex_thread_id": "0199…", "tokens_used": 84210, "wallclock_s": 1820,
  "confidence": 0.7, "supersedes": ["experience_id…"], "created_at": "…"
}
```

- **Why both arms:** ExpeL's gains come from *contrasting* success/failure pairs; Reflexion's from distilled failure hints; Manus keeps failure traces in context. At planning time, `PlannerAgent` retrieves top-k relevant experiences (by hypothesis/method similarity) and is explicitly prompted with prior failure modes ("avoid X, which caused OOM here") and successful strategies.
- **Retrieval without a vector DB:** build an FTS5 query from the current hypothesis/method keywords → top-20 → LLM rerank to top-k → inject with outcome labels and lessons. FTS5 + rerank is strong up to thousands of experiences; add `sqlite-vec` embeddings only if retrieval quality demonstrably lags. This keeps the memory tier dependency-free.
- **Guardrails against self-reinforcing error:** `confidence` scores; `supersedes` links to overturn stale conclusions; contradiction checks before injection; expiry/down-weighting of old records. A single failure never permanently blacklists an approach without re-test evidence.

---

## 6. Context Management

- **Artifact offloading.** Large outputs (full logs, datasets, plots, generated code) go to the artifact store; only `{artifact_id, summary}` references enter session state. Agents load full content on demand via a `read_artifact(id, range?)` tool — Anthropic's "let agents retrieve autonomously" plus Manus's "file system as context." Never paste a training log into an LLM context.
- **Compaction at stage boundaries.** Between iterations and at agent hand-offs, summarize the iteration into a high-fidelity structured record (plan, what ran, metrics, decisions, open issues) and clear tool-result bloat; keep the last few turns raw.
- **Sub-agent context isolation.** Each parallel searcher/experiment agent receives only its narrow brief; the orchestrator synthesizes.
- **Recitation.** Maintain `todo.md`/plan artifacts re-injected at the end of context each iteration to fight goal drift on long horizons.
- **Structured inter-stage summaries** use fixed schemas (plan, exp_spec, analysis) so compaction is predictable and low-variance.
- **KV-cache hygiene:** stable system-prompt prefixes, append-only context, deterministic JSON serialization.

---

## 7. Artifact Subsystem

The guiding principle: **the conversation is ephemeral; the artifacts are the project.** Every meaningful output — research, engineering, and logistics — is a registered, versioned, lineage-tracked artifact.

### 7.1 Artifact taxonomy

| Class | Artifacts | Typical format | Producer | Lifecycle |
|---|---|---|---|---|
| **Logistic / management** | research brief, design doc, plan, board, todo, decision log (ADR-style), budget ledger, checkpoint records, risk register | md / json | orchestrator + user | living documents, versioned on every edit |
| **Research** | literature notes, annotated bibliography (refs.bib), hypothesis records, experiment specs, analysis memos, critique reports | md / json | lit/planner/analyst agents | append + version |
| **Engineering** | code repos, AGENTS.md, env lockfiles, datasets, model checkpoints, run logs, metrics.json | git / binary / jsonl | Codex | git-versioned (code), content-addressed (data) |
| **Results** | metrics tables, comparison reports, figures (Vega-Lite spec + PNG) | json / vega / md | Codex + ResultAnalyst | versioned, immutable per run |
| **Deliverables** | interim reports, final report, slide outline | md (→ pdf on demand) | report agents | versioned |

The logistic class is what makes a long-running, heavily steered project legible:

- **`research_brief.md`** — clarified question, scope, success criteria, assumed defaults. Produced by intake, signed off at Gate 1. The *contract*; every plan revision cites it.
- **`design.md`** — the evolving technical design of the research itself (baselines, datasets, evaluation protocol, ablation plan). The ExperimentDesigner reads/writes it; the user can edit it directly in the UI, producing a new version the orchestrator must acknowledge at the next checkpoint.
- **`plan.md` + `board.json`** — plan narrative plus a machine-readable kanban: columns (`backlog | in_progress | blocked | awaiting_review | done | killed`), items typed `lit_task | hypothesis | experiment | analysis | report_section`, each carrying `branch`, `iteration`, `budget_est`, `status_reason`, artifact refs. The orchestrator updates it every loop (the structured cousin of Manus-style recitation; free-form `todo.md` remains as scratchpad). The UI renders it; Jira can mirror it (§8).
- **`decisions.md`** — append-only ADR-style log: *context → decision → evidence (artifact links) → consequences*. Example: "Pruned branch B2: no improvement over baseline across 2 seeds (evidence: `art_8f31`, `art_8f4a`); reallocating budget to B1 refinement." Skimming this file is how the user catches the system going sideways without reading transcripts. Every PlanReviser iteration and gate outcome appends here.
- **`budget.json`** — ledger of tokens/credits/wallclock per iteration and branch, fed by Codex `turn.completed` usage events and model-call accounting; powers the UI budget meter and auto-pause thresholds (§14).
- **`checkpoints/<ts>_<gate>.json`** — every HITL gate: what was asked/shown, the user's decision, free-text comments. Doubles as steering history.

### 7.2 Storage: filesystem layout + SQLite catalog

Everything under one root; backup = copying a folder.

```
~/deep-researcher/
  deep_researcher.db                  # SQLite: ADK sessions + artifact catalog + experiences + jobs
  projects/<pid>/
    brief/research_brief.md
    design/design.md
    plan/plan.md   plan/board.json   plan/todo.md
    decisions/decisions.md
    budget/budget.json
    checkpoints/<ts>_<gate>.json
    lit/<facet>/notes.md  lit/refs.bib
    iter_<n>/
      hypotheses.json
      exp_<branch>/
        repo/                         # git repo; AGENTS.md, src/, metrics contract
        data/ -> ../../../../data_store/<sha256>    # symlinked, content-addressed
        runs/<run_id>/codex_events.jsonl  train.log
        plots/<name>.vega.json  <name>.png
        analysis.md
    reports/interim_<n>.md  final_report.md
  data_store/<sha256>/                # datasets & checkpoints, deduped across branches
```

**Catalog schema** (same SQLite DB as ADK sessions):

```sql
CREATE TABLE artifacts (
  id            TEXT PRIMARY KEY,      -- 'art_' + ulid
  project_id    TEXT NOT NULL,
  kind          TEXT NOT NULL,         -- 'brief','design','plan','board','decision','budget',
                                       -- 'lit_notes','hypothesis','exp_spec','code','dataset',
                                       -- 'checkpoint_model','run_log','metrics','plot','analysis','report'
  path          TEXT NOT NULL,         -- relative to project root (or data_store hash)
  version       INTEGER NOT NULL,
  content_hash  TEXT,
  iteration     INTEGER, branch TEXT, run_id TEXT,
  title         TEXT,
  summary       TEXT,                  -- short LLM-written summary, injected into agent context
  meta          JSON,                  -- e.g. {"metric":"val_acc","value":0.873} for metrics
  created_by    TEXT,                  -- agent name | 'user' | 'codex'
  created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE artifact_lineage (
  child_id  TEXT, parent_id TEXT,
  relation  TEXT,                      -- 'derived_from' | 'evidence_for' | 'supersedes' | 'implements'
  PRIMARY KEY (child_id, parent_id, relation)
);

CREATE VIRTUAL TABLE artifact_fts USING fts5(artifact_id UNINDEXED, title, summary, body);
```

**Rules that make this work:**

1. **Reference-passing only.** Agents register artifacts and place `{artifact_id, summary}` into state; full content is loaded on demand.
2. **Lineage is mandatory for claims.** Every numeric claim in a report must chain: report → analysis → metrics → run → code commit → exp_spec → plan → brief. "Where did this number come from?" is one recursive CTE over `artifact_lineage`, exposed in the UI as click-through.
3. **Versioning policy.** Code: git, commit per Codex step (`file_change` JSONL events give the diff trail for free). Documents: `version+1` row plus a `supersedes` lineage edge. Datasets/checkpoints: content-addressed in `data_store/` (sha256 names) — identical data across branches stored once.
4. **ADK integration.** Implement `LocalArtifactService(BaseArtifactService)` (~100 LoC): `save_artifact` writes to the layout and inserts into the catalog; `load_artifact`/`list_artifact_keys` read it. Register on the `Runner`. (ADK ships only `InMemoryArtifactService` and `GcsArtifactService`; the base interface is small and made for custom backends.)
5. **Summaries are part of registration.** Whoever registers an artifact supplies a 1–3 sentence summary (mechanically generated for `metrics.json`). This keeps context windows small and the FTS index useful.

---

## 8. Progress Management and the Optional Jira Mirror

**Local-first stance:** `board.json` is the single source of truth. The UI renders it as a kanban; the orchestrator transitions item statuses every iteration; the user can drag items in the UI (which writes a new board version and queues a steering note for the orchestrator).

**Jira mirror (optional, deferred to Phase 3+):**

- **Mapping:** project → Epic; iteration → label or Sprint; experiment branch → Task; analysis/report sections → Tasks; HITL gates → comments on the Epic; `decisions.md` entries → comments with artifact links.
- **Sync direction:** one-way push (local → Jira) by default, via a `jira_sync` ADK tool against the Jira Cloud REST API — or the Atlassian MCP server (`mcp-atlassian`) for tool-native integration. Two-way sync limited to *status transitions* made in Jira, polled and merged only at checkpoint boundaries to avoid mid-run conflicts.
- **Honest recommendation:** for a single user, the local board + UI is strictly better (no auth plumbing, no sync conflicts, richer rendering). Jira earns its keep only when progress must be visible to other humans. Bidirectional sync is a known complexity trap; keep Jira a mirror, never a master.

---

## 9. Local Infrastructure Stack (no GCP / Vertex / AWS)

**ADK does not require GCP.** It runs against the Gemini Developer API with just an API key (`GOOGLE_GENAI_USE_VERTEXAI=FALSE`, `GOOGLE_API_KEY=…`), and via its LiteLLM wrapper it can drive Anthropic/OpenAI/local models. Every cloud convenience has a local substitute:

| Concern | Cloud option (not used) | Local choice | Notes |
|---|---|---|---|
| Sessions / state / resumability | VertexAiSessionService, Cloud SQL | `DatabaseSessionService("sqlite:///~/deep-researcher/deep_researcher.db")` | built into ADK; `ResumabilityConfig` works unchanged on top |
| Artifacts | GcsArtifactService | custom `LocalArtifactService` + catalog (§7) | ~100 LoC |
| Long-term memory | Vertex Memory Bank / RAG corpus | SQLite experience store + FTS5; optional `sqlite-vec`/local Chroma later | §5 |
| Model access | Vertex AI | Gemini Developer API key; LiteLLM for non-Google models | Codex unchanged (subscription auth) |
| Serving | Cloud Run / Agent Engine | one local FastAPI process (wrapping `Runner.run_async`) under systemd/launchd; `adk api_server` for dev | |
| Job queue | cloud queue | `jobs` table in SQLite + asyncio worker pool (`max_codex_concurrency`) | survives restarts; workers re-adopt `running` jobs via idempotency markers |
| Notifications | pub/sub | SSE to the UI; optional desktop notification / Slack webhook | |

**SQLite-specific engineering notes:**

- **WAL mode + single-writer discipline.** Enable WAL; route all DB writes through one writer coroutine (ADK's `append_event` already serializes session writes; do the same for catalog/jobs DAOs). Parallel branches are the only contention source, and their high-volume output (Codex JSONL, training logs) goes to **per-run files on disk, not the DB** — the DB stores only status rows and registered summaries.
- **One DB, several schemas.** ADK session tables, artifact catalog, experience store, and jobs share one file; backup/restore and machine migration are `rsync` of one folder.
- **Escape hatch for contention:** if ≥ ~8 concurrent branches ever saturate the writer, split the `jobs` table into a separate DB file (SQLite locks are per-file) — not a server DB.

---

## 10. Long-Running Task & Async Design

- **Async runs.** Everything drives through `runner.run_async(user_id, session_id, new_message)`; ADK's event loop yields/pauses/resumes around long-running tools.
- **Resumability (crash recovery).** `App(resumability_config=ResumabilityConfig(is_resumable=True))` (Python ADK ≥ 1.16). ADK logs completed agent/tool steps as Events; on restart it reinstates completed tool results and re-runs only the incomplete step. `SequentialAgent` resumes from `current_sub_agent`; `LoopAgent` from `current_sub_agent` + `times_looped`; `ParallelAgent` re-runs only unfinished sub-agents.
- **Idempotency caveat (critical).** ADK may re-run a long-running tool more than once on resume. The `codex_exec` tool must check for an existing completed result (marker file / recorded `thread_id` in the workspace) before launching — otherwise a resume duplicates an expensive paid run.
- **Resume invocation.** `runner.run_async(..., invocation_id='…')` resumes a specific invocation (ID from event history). Resume is not currently supported from `adk web`/CLI — drive it programmatically from the FastAPI gateway.
- **Checkpointing.** Sessions persist via the SQLite `DatabaseSessionService`; artifacts via `LocalArtifactService`; experiences in their own tables. All state changes go through `append_event` (never mutate `session.state` directly) so persistence and resume bookkeeping stay correct.
- **Job queue.** Codex experiments are the long pole and rate-limited: experiment launches land on the `jobs` table; an asyncio worker pool (capped at `max_codex_concurrency`) executes them; the `LongRunningFunctionTool` returns an operation id immediately and the orchestrator polls/awaits, freeing the loop. After a crash: restart → scan `jobs` for `running` rows → check workspace markers → re-attach or re-launch.

---

## 11. Human-in-the-Loop and Steering

This is a PI-and-lab model: hard gates at major transitions, plus continuous steering between them.

### 11.1 Checkpoint gates

Each gate is a `LongRunningFunctionTool` (`is_long_running=True`) — the canonical ADK HITL pattern: the tool returns a "pending" ticket, the run pauses, the user decides in the UI, the response resumes the invocation (with the matching `invocation_id` when resumability is on). Every gate decision is persisted as a checkpoint artifact.

- **Gate 1 — Plan approval.** After `PlannerAgent` (the deep-search sample's "explicit approval → delegate" flow). UI shows the plan with an inline editor; an edited plan becomes a new artifact version.
- **Gate 2 — Experiment-budget approval.** Before launching Codex experiments; surfaces estimated tokens/credits/wallclock and compute footprint.
- **Gate 3 — Mid-experiment review (configurable).** After first results in long branches; the user can redirect, kill a branch, or continue.

### 11.2 Interrupts (three levels)

1. **Cooperative pause** — flag in `jobs`; the orchestrator yields at the next stage boundary; resumability persists the invocation for later resume. In-flight Codex runs finish their current turn.
2. **Hard cancel** — cancel the asyncio task driving `run_async`; SIGTERM the `codex exec` process group; mark the run `aborted`; write an experience record with `outcome: "aborted"`. Resume re-plans from the last completed event.
3. **Kill-branch** — per-branch job cancellation without touching siblings (each ParallelAgent branch maps to a job row + process group).

### 11.3 Steering inbox

The user can type at any time. Mid-run free-text messages queue and are injected into the orchestrator's context at the next checkpoint (UI acknowledges "queued for next checkpoint"); slash-command messages (`/pause`, `/resume`, `/kill B2`, `/budget +20`, `/status`) act immediately. Hot-modifying a *running* Codex thread is deliberately unsupported (consistency nightmare); the unit of redirection is the checkpoint or the branch.

### 11.4 Notifications

ADK Events stream to the UI via SSE on every gate, branch completion, and budget threshold; optional desktop/Slack webhook for unattended periods.

---

## 12. User Interface

Chat is the control plane, but it must render rich content and sit beside persistent panes for the project's living artifacts.

### 12.1 Requirements

1. **Chat as control plane.** Every gate, decision, redirect, and question flows through chat — structured *cards* with buttons (Approve / Edit / Reject-with-comment) plus free text and slash commands.
2. **Rich rendering** in chat and artifact views: GFM markdown + tables; **KaTeX** math; syntax-highlighted code with **diff view** (Codex `file_change` events render as diffs); **Mermaid** diagrams; **interactive plots** via Vega-Lite; images; artifact links that open the browser pane.
3. **Four persistent panes** beside chat:
   - **Plan/Board** — kanban from `board.json`; drag = steering input.
   - **Artifact browser** — tree of the project folder; clicking renders by kind (md → rich, vega.json → interactive chart, repo → file tree + diffs, metrics → table); lineage click-through for evidence chains.
   - **Run monitor** — live per-branch timeline tailing Codex `--json` JSONL: commands executed, files changed, tokens burned. This is how a wayward experiment is caught early.
   - **Checkpoint inbox + budget meter** — pending gates; spend vs. caps per branch/iteration.
4. **Interrupt controls** wired to §11.2.

### 12.2 Architecture

```
Next.js / React UI  ◄── SSE/WebSocket ──►  FastAPI gateway  ──►  ADK Runner (run_async)
  ├─ Chat (AG-UI / CopilotKit, or assistant-ui)                    ├─ SQLite (sessions, catalog,
  ├─ Board pane (board.json)                                       │          experiences, jobs)
  ├─ Artifact browser (catalog + files)                            ├─ LocalArtifactService
  ├─ Run monitor (JSONL tails → SSE)                               └─ codex exec subprocess pool
  └─ Checkpoint cards (LongRunningFunctionTool round-trips)
```

- **Protocol.** Lowest-friction: the **AG-UI protocol with `ag_ui_adk` middleware + a CopilotKit React frontend** — it maps ADK events and `LongRunningFunctionTool` pauses into a typed event stream with built-in HITL approval round-trips, and works with `ResumabilityConfig`. Alternative: plain `adk api_server` SSE with custom handlers (more control, more code). Decide with a one-day spike; the backend is identical either way.
- **Rendering stack:** `react-markdown` + `remark-gfm` + `remark-math` + `rehype-katex`; a Mermaid component for ```mermaid fences; `react-vega` for `*.vega.json`; Monaco (or `react-diff-view`) for code and diffs; `shiki` for highlighting.
- **The rendering contract (enforced at the producer side):**
  - Each experiment's `AGENTS.md` instructs Codex: *every plot is emitted twice — `plots/<name>.vega.json` (Vega-Lite spec) and `plots/<name>.png` (fallback); every result lands in `metrics.json` per the contract.* Interactive charts in the UI come free.
  - Orchestrator/analyst agents write markdown using KaTeX `$...$` for math, ```mermaid for diagrams, and `[artifact:art_xxx]` link syntax the UI resolves into pane-opening links.
  - The FastAPI layer rewrites artifact links and injects plot specs so the chat stream renders without the client knowing the filesystem.
- **Run monitor plumbing:** one tailer task per active run streams `codex_events.jsonl` → multiplexed SSE channel `runs/<run_id>`; the same parsed events update `budget.json` (token usage from `turn.completed`) and the branch's job row.

### 12.3 UI phasing

- **Phase 0:** `adk web` for plumbing/debug (the event graph is genuinely useful) **plus a one-page Streamlit app** — chat transcript, markdown + Plotly rendering, Approve/Reject buttons, pause flag. Covers ~80% of steering needs in about a day and validates gate round-trips.
- **Phases 1–2:** the Next.js + AG-UI app with chat + checkpoint cards + run monitor (the three steering-critical surfaces). Artifact browser starts as "render markdown + show images."
- **Phase 3:** full artifact browser with lineage click-through, board pane with drag-steering, budget meter, multi-project dashboard.

---

## 13. Codex Integration

**Invocation patterns.** Two interchangeable backends, both authenticated via the user's ChatGPT subscription (`codex login`; OAuth tokens cached in `~/.codex/auth.json` — treat as a secret):

- **CLI:** `codex exec --json --sandbox workspace-write -C <workspace> -o <last_msg_file> "<task prompt>"`. Parse the JSONL event stream: `thread.started` (capture `thread_id`), `item.completed` for `command_execution` / `file_change` / `agent_message` items, `turn.completed` for token `usage`. `--output-last-message` captures the final summary. Prefer explicit `--sandbox workspace-write` over the deprecated `--full-auto`.
- **SDK:** TypeScript `@openai/codex-sdk` (`codex.startThread()` → `thread.run(prompt)` → `codex.resumeThread(id)`) or the Python equivalent (`thread_start(model=…, sandbox=…)`, `thread.run(...).final_response`). Cleaner programmatic surface; recommended for the production wrapper.

**ADK wrapper sketch (`codex_exec` as `LongRunningFunctionTool`):**

```python
def codex_exec(task_prompt: str, workspace: str, branch_id: str,
               model: str = "<current codex model>", tool_context=None) -> dict:
    # Idempotency: if this branch already completed, return cached result (resume-safe)
    if (done := read_marker(workspace)):     # avoids duplicate paid runs on ADK resume
        return done
    thread = codex.thread_start(model=model, sandbox=Sandbox.workspace_write)
    write_state(tool_context, f"codex_thread:{branch_id}", thread.id)  # for fix loops
    result = thread.run(task_prompt)         # streams JSONL; long-running
    metrics = parse_metrics(workspace)       # read metrics.json the code wrote
    write_marker(workspace, {"thread_id": thread.id, "metrics": metrics})
    return {"status": "completed", "thread_id": thread.id,
            "final": result.final_response, "metrics": metrics}
# registered with is_long_running=True → ADK pauses; jobs queue executes; idempotent on resume
```

- **`AGENTS.md` per workspace.** Each experiment workspace ships an `AGENTS.md` encoding conventions: framework, how to run tests/training, the `metrics.json` contract, the dual plot convention (`.vega.json` + `.png`), "commit each step," forbidden destructive ops. Keep it small — it consumes context/quota on every Codex turn.
- **Iterative fix loop (session resume).** On a buggy run, re-prompt the *same* Codex thread (`exec resume <thread_id>` / `codex.resumeThread(id)`) with the error context, preserving the prior transcript — mirroring AIDE's debug-the-buggy-node behavior.
- **Sandboxing.** Default `workspace-write` (writes confined to workspace + tmp; network off by default). For experiments needing package installs/network, run Codex inside a Docker dev-container and only there consider `danger-full-access` — never unsandboxed on the host (the AI Scientist sandbox warning applies: LLM-written code can do dangerous things). Use `--add-dir` to grant specific extra paths rather than broad access.
- **Error handling.** Classify outcomes from the JSONL stream and exit codes: `turn.failed`, non-zero `command_execution`, OOM/timeout. Retry transient failures with backoff; on persistent failure, record a *failure* experience (with stack trace, à la Manus) and either spawn a fix-thread or prune the branch.
- **Cost / rate-limit handling under subscription.** Codex usage counts toward ChatGPT agentic usage limits (5-hour windows + weekly quotas); on exhaustion, add credits or wait for reset. Mitigations: cap `max_codex_concurrency`; minimize `AGENTS.md`/MCP context; track usage from `turn.completed` events into `budget.json`; queue and serialize near limits. For unattended bulk runs, API-key auth is the sanctioned alternative (OpenAI recommends API keys for automation, subscription for personal tooling).

---

## 14. Evaluation & Guardrails

- **Experiment validity.** Each Codex experiment must (a) compare against a declared baseline, (b) write machine-readable `metrics.json`, (c) pass MLE-STAR-style **data-leakage** and **data-usage** checks (dedicated checker tools/agents), and (d) report variance across ≥2 seeds where feasible. The `CritiqueAgent` flags unsupported claims, p-hacking, or metric/objective mismatch.
- **Budget caps.** Hard ceilings on tokens/credits, wallclock, and branch count per iteration and per project; the budget gate exposes estimates before spend; auto-pause on threshold breach (thresholds live in `budget.json`).
- **Code-execution safety.** Codex confined to `workspace-write` (or Docker for network); no host-level full access; `AGENTS.md` forbids destructive ops; the run monitor surfaces every `command_execution` for human eyes; secrets never co-resident with untrusted code.
- **Reproducibility.** Pin seeds, log environment/deps (lockfiles as artifacts), commit code per step, content-address datasets.

---

## 15. Implementation Roadmap

- **Phase 0 — Skeleton (1–2 wks).** Fork the ADK `deep-search`/`gemini-fullstack` sample. IntakePhase (`ClarifierAgent`, ≤3 questions), `PlannerAgent`, plan-approval gate, parallel literature searchers, report composer. SQLite `DatabaseSessionService` + `LocalArtifactService` + catalog schema from day one. `research_brief.md`/`plan.md`/`decisions.md` conventions. `adk web` for debugging + one-page Streamlit steering app. *Exit criterion: a clean run that asks ≤3 clarifying questions, gets plan approval, and produces a cited report.*
- **Phase 1 — Codex execution (2–3 wks).** `codex_exec` `LongRunningFunctionTool` (single experiment, no parallelism), `AGENTS.md` convention with the metrics/plot rendering contract, JSONL parsing, fix loop via thread resume, budget-approval gate + `budget.json` ledger, run monitor (JSONL tail → SSE), checkpoint cards. *Exit criterion: one real experiment runs end-to-end with artifacts, live monitoring, and a budget gate.*
- **Phase 2 — Memory + resumability (2–3 wks).** Experience store (success+failure schema) on SQLite FTS5, retrieval at planning time, `ResumabilityConfig`, idempotent Codex tool, crash recovery, pause/hard-cancel, steering inbox, `board.json`. Compaction at boundaries + `todo.md` recitation.
- **Phase 3 — Parallel exploration + full UI (2–4 wks).** `ParallelAgent` fan-out for literature/ideas/experiments, Elo/LLM-judge tournament, metric-driven pruning (AIDE/MLE-STAR; reference the MLE-STAR ADK sample directly), concurrency + budget controls, mid-run review gate. Next.js + AG-UI app: lineage click-through, board drag-steering, budget meter. Optional Jira mirror. `sqlite-vec` embeddings only if FTS5 retrieval proves insufficient.
- **Phase 4 — Hardening (ongoing).** Leakage/usage/critique guardrail agents, notification webhooks, multi-project dashboard, folder-backup automation.

---

## 16. Open Questions & Risks

- **Multi-agent vs. single-agent tension (Cognition vs. Anthropic).** Parallelism helps for independent breadth but risks fragmented context. Mitigation: single orchestrator owns decisions; parallel only for independent branches; rich structured hand-offs. Open: where the crossover lies for experiment branches sharing a codebase.
- **Codex subscription rate limits** may throttle parallel experiments or stall multi-day runs. Mitigation: concurrency caps, credits, optional API-key path for bulk.
- **Idempotency on resume** — ADK may re-run long-running tools; a non-idempotent Codex launch wastes money. Mitigation: marker/thread-id checks built into the wrapper (non-negotiable before enabling resumability).
- **Self-reinforcing memory error** — failure records could wrongly blacklist good approaches. Mitigation: confidence scores, supersede links, contradiction checks, expiry.
- **AI-generated science validity** — passing review ≠ correctness; hallucinated results are a known failure mode. Mitigation: validity checks, human gates, reproducibility requirements, mandatory lineage for claims.
- **Cost** — ~15× token multiplier for multi-agent plus Codex credits. Mitigation: budget gates, small defaults, escalate only on promise; auto-pause when a branch shows no improvement over baseline across 2 seeds.
- **SQLite under parallel load** — per-run logs on disk + single-writer DAO should suffice for ≤ ~8 concurrent branches; escape hatch is a separate DB file for `jobs`.
- **ADK maturity** — resumability is recent (Python ≥1.16); behaviors differ across SDK languages; resume not supported from `adk web`. Mitigation: pin versions; drive resume from the gateway.
- **Frontend framework choice** — CopilotKit/AG-UI vs `assistant-ui` vs hand-rolled SSE; decide via a one-day spike of the gate round-trip.
- **Mid-run steering granularity** — current design queues free text to the next checkpoint; whether some steering deserves injection into *running* branches via a Codex thread nudge is deferred until real usage data exists.
- **Plot data volume** — inline Vega-Lite data bloats artifacts for large sweeps; need a spec-references-CSV convention beyond N rows.
- **Security** — LLM-written code execution is inherently risky. Mitigation: sandbox/Docker, transcript review in the run monitor, no host full-access.

---

## 17. References

**Papers**
- Sakana AI Scientist v1 — arXiv:2408.06292 · github.com/SakanaAI/AI-Scientist
- Sakana AI Scientist v2 — arXiv:2504.08066 · github.com/SakanaAI/AI-Scientist-v2 · sakana.ai/ai-scientist-nature
- AIDE — arXiv:2502.13138 · aide.ml · github.com/WecoAI/aideml
- MLE-STAR — arXiv:2506.15692 · research.google/blog (MLE-STAR) · github.com/google/adk-samples (python/agents/machine-learning-engineering)
- MLE-bench — arXiv:2410.07095
- Agent Laboratory — arXiv:2501.04227 · AgentRxiv — arXiv:2503.18102 · github.com/SamuelSchmidgall/AgentLaboratory
- AI-Researcher — arXiv:2505.18705
- Google AI co-scientist — arXiv:2502.18864 · research.google/blog/accelerating-scientific-breakthroughs-with-an-ai-co-scientist
- AlphaEvolve — arXiv:2506.13131 · deepmind.google/blog (AlphaEvolve)
- Reflexion — arXiv:2303.11366 · ExpeL — arXiv:2308.10144
- Agent memory surveys — arXiv:2404.13501 and successors

**Engineering blogs**
- Anthropic, "How we built our multi-agent research system" — anthropic.com/engineering/multi-agent-research-system
- Anthropic, "Effective context engineering for AI agents" — anthropic.com/engineering/effective-context-engineering-for-ai-agents
- Manus, "Context engineering for AI agents: lessons from building Manus" (Yichao Ji)
- Cognition, "Don't Build Multi-Agents" — cognition.ai/blog/dont-build-multi-agents
- LangChain, "Context engineering for agents" — langchain.com/blog/context-engineering-for-agents

**Frameworks & tools**
- Google ADK docs — google.github.io/adk-docs (workflow agents · sessions/state · runtime/resume · tools/function-tools · artifacts)
- ADK samples — github.com/google/adk-samples (deep-search · gemini-fullstack · machine-learning-engineering)
- AG-UI protocol + `ag_ui_adk` middleware; CopilotKit — docs.copilotkit.ai
- OpenAI Codex — developers.openai.com/codex (noninteractive · cli/reference · sdk · agent-approvals-security) · help.openai.com ("Using Codex with your ChatGPT plan")
- LangChain open_deep_research — github.com/langchain-ai/open_deep_research
- gemini-fullstack-langgraph-quickstart — github.com/google-gemini/gemini-fullstack-langgraph-quickstart
- sqlite-vec — github.com/asg017/sqlite-vec · SQLite FTS5 — sqlite.org/fts5.html
- Atlassian Jira Cloud REST API · mcp-atlassian — github.com/sooperset/mcp-atlassian

*Caveats: several cited performance figures are vendor-reported on internal evals (Anthropic's 90.2%, MLE-STAR's medal rates, AIDE's MLE-bench numbers) — treat as directional. Codex flags/models/quotas and ADK APIs evolve quickly; verify against live docs at implementation time.*
