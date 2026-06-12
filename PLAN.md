# Implementation Plan & Status

Implements [docs/design.md](docs/design.md) (Deep Researcher v3) phase by phase.
Deviations from the design doc are listed per phase under **Decisions**.

## Phase 0 — Skeleton ✅ (complete; live E2E validated 2026-06-11)

Pipeline: clarify (≤3 questions) → `brief/research_brief.md` → plan + facets →
**Gate 1 plan approval (chat)** → parallel literature searchers → synthesis →
cited `reports/final_report.md`. SQLite sessions + local artifact store +
catalog from day one. Streamlit steering UI + terminal REPL.

- [x] uv project, Python 3.13, ADK 2.2 (`google-adk[db]`), LiteLLM, pinned via `uv.lock`
- [x] `config.py` — `~/data/deep-researcher` root, DeepSeek models, knobs
- [x] `storage/catalog.py` — artifacts + lineage + FTS5 (design §7.2 schema)
- [x] `storage/artifact_service.py` — `LocalArtifactService(BaseArtifactService)`,
      versioned files (latest at semantic path, history under `.versions/`),
      auto `supersedes` lineage, catalog registration via `custom_metadata`
- [x] tools — `search_openalex` (primary), `search_arxiv`, `search_semantic_scholar`
      (needs key; unauth pool is saturated), `write_artifact`, `read_artifact`,
      `list_artifacts`, `save_plan` (facets → state), `append_decision` (ADR log),
      `record_checkpoint` (gate records)
- [x] agents — `orchestrator` (root, owns decisions) → `research_pipeline`
      (SequentialAgent) → `lit_fanout` (ParallelAgent ×3, `{facet_i?}` state
      templating, unique `lit_notes_i` output keys) → `lit_synthesizer` → `report_writer`
- [x] runner + CLI REPL (`uv run deep-researcher "<question>"`)
- [x] Streamlit app (`uv run streamlit run app/streamlit_app.py`): chat,
      approve/revise gate buttons, artifact browser pane
- [x] tests: storage round-trip/versioning/lineage/FTS, literature parsers,
      full-pipeline offline integration test with a scripted mock LLM
- [x] live end-to-end run (`scripts/live_e2e.py`, project `demo-moe-routing`):
      clarify → plan approval → 3 parallel searchers (~20 live searches) →
      synthesis of 25 papers → 26KB cited report with KaTeX math, ~4 min,
      zero DeepSeek tool-call parse failures

**Decisions (vs. design doc)**
- DeepSeek via LiteLLM instead of Gemini (user decision). Mitigation for
  google/adk-python#5024 (multi-tool-call parse flakes): all instructions
  demand one tool call per response; model swap is a config change.
- OpenAlex added as primary lit search; Semantic Scholar kept but secondary
  until an API key exists (unauthenticated pool 429s persistently).
- Gate 1 runs through chat (deep-search-sample style) rather than a
  `LongRunningFunctionTool`; LRFT gates arrive with Phase 1's budget gate.
- Phase 0 maps project_id == session_id (one ADK session per project).
- ADK 2.2 deprecates `SequentialAgent`/`ParallelAgent` in favor of `Workflow`;
  still functional — migrate when touching Phase 3 parallelism.

## Phase 1 — Codex execution ✅ (complete; live E2E validated 2026-06-11)

- [x] `codex/runner.py` — `codex exec --json` subprocess driver; JSONL schema
      verified against codex-cli 0.136; events streamed to
      `runs/<run_id>/codex_events.jsonl`; **idempotent** via `result.json`
      marker (run_id = prompt hash → duplicate calls return cached, design §16)
- [x] `codex/workspace.py` — `AGENTS.md` contract (metrics.json, dual plots,
      commit per step, forbidden ops) + git init, idempotent
- [x] `tools/codex.py` — `codex_exec` tool: workspace at
      `iter_1/exp_main/repo`, `--sandbox workspace-write`, fix loop via
      `resume_thread_id`, `budget/budget.json` ledger updated per run
- [x] agents restructured: stages are **AgentTools** on the orchestrator
      (literature_review, experiment_designer, result_analyst, report_writer)
      so all gates live in the root chat thread; Gate 2 budget approval before
      any `codex_exec` call
- [x] Streamlit run monitor: per-run expander tailing `codex_events.jsonl`
      (status/commands/files/tokens/metrics, 3s auto-refresh) + budget meter
- [x] tests: JSONL parser, fake-codex success/idempotency/failure/timeout,
      workspace prep, experiment flow with budget gate (offline)
- [x] live E2E (`scripts/live_e2e_experiment.py`, project
      `demo-qmc-integration`): plan gate → 28-ref literature stage → spec +
      budget estimate → Gate 2 → real Codex run (150s, ~250k in / 7k out
      tokens, metrics.json + dual plots + git commits per step) → honest
      analysis (caught a real flaw in the Halton leaping scheme) → cited
      report. All 7 exit checks passed.

**Decisions (vs. design doc)**
- Stages via AgentTool instead of transfer-to-pipeline: gates can't pause a
  `SequentialAgent` mid-flight without LRFT plumbing; AgentTool keeps the
  single decision thread and chat-level gates. Cost: sub-agent activity is
  opaque in the chat stream (AgentTool consumes inner events) — Codex runs
  stay visible via the file-based run monitor; richer streaming is a Phase 3
  UI concern.
- `codex_exec` runs synchronously inside the turn (Phase 1 = single
  experiment); the jobs queue + LRFT pause/resume arrive with Phase 2
  resumability.
- CLI (`codex exec --json`) over the SDK: stable, already authenticated, no
  extra dependency.

## Phase 2 — Memory + resumability ✅ (complete 2026-06-11)

- [x] `storage/experiences.py` — experience store (success+failure schema §5)
      on SQLite FTS5, cross-project; supersede links hide overturned records;
      FTS query sanitization; confidence scores
- [x] tools: `search_experiences` (orchestrator calls it BEFORE planning),
      `record_experience` (after every experiment analysis, failures included)
- [x] `ResumabilityConfig(is_resumable=True)` on the App; crash recovery:
      `find_resumable_invocation` + `resume_invocation`; CLI `--resume`;
      Streamlit "Resume interrupted run" banner. Verified by an offline
      crash/resume test: interrupted after a persisted tool call, resumed on a
      fresh runner, completed step replayed (no duplicate artifact writes)
- [x] context compaction: `EventsCompactionConfig` + `LlmEventSummarizer`
      (DeepSeek), interval/overlap in config
- [x] `board.json`: `update_board` tool (orchestrator updates at stage
      transitions) + Streamlit Board tab

**Decisions (vs. design doc)**
- Pause/hard-cancel + steering inbox deferred to Phase 3: turns are currently
  synchronous (the user steers between turns), so mid-run interrupts only
  become meaningful once experiments move onto the async jobs queue.
- `todo.md` recitation: superseded by ADK's native event compaction + the
  orchestrator re-reading `plan.md`; revisit if goal drift appears in long
  projects.
- ADK marks ResumabilityConfig/EventsCompactionConfig experimental — pinned
  google-adk 2.2.0 in uv.lock; re-verify on upgrades.

## Phase 3a — Parallel experiment branches ✅ (complete; live E2E 2026-06-11)

- [x] `storage/jobs.py` — jobs table shared across processes (agent registers
      pid/pgid; UI reads + kills); kill-branch SIGTERMs one branch's process
      group without touching siblings (design §11.2)
- [x] `run_experiments` tool — N branches concurrently, capped by
      `max_codex_concurrency` (default 2); per-branch isolated workspaces
      `iter_1/exp_<branch>/`; aggregated budget entries; `codex_exec` keeps
      single-branch + fix-loop duty (`branch_id`, `resume_thread_id`)
- [x] idea tournament — ParallelAgent of 3 personas (conservative / novel /
      efficient) → LLM-judge pairwise ranking → `iter_1/hypotheses.json`
      (co-scientist Generation+Ranking analog, sized for a local stack)
- [x] result_analyst upgraded to cross-branch comparison: shared-metric
      ranking, AIDE-style pruning of failed branches, refinement
      recommendation for the winner
- [x] orchestrator workflow: breadth decision (single vs. tournament+branches),
      Gate 2 over the TOTAL budget, parallel launch, per-branch fix loop,
      one experience record PER BRANCH
- [x] Streamlit: kill-branch button on running runs; killed status surfaced
- [x] tests: parallel fan-out with concurrency-cap gauge, input validation,
      jobs kill semantics, real-process kill-branch (25 tests total)
- [x] live E2E (`scripts/live_e2e_parallel.py`, project `demo-gradfree-opt`):
      3 real Codex branches in parallel (RS/SA/ES on Rosenbrock+Rastrigin,
      shared metric, ~8 min, ~980k in / 22k out tokens), cross-branch ranking
      (SA won Rosenbrock by ~3 orders of magnitude), 3 experience records
      incl. one failure, 30-ref report. Tournament validated in a follow-up
      turn (orchestrator initially skipped it because the user message also
      prescribed the branch split — instruction now makes explicit user
      requests override that judgment).

**Deferred**
- Gate 3 mid-run review (needs async jobs decoupled from the chat turn)
- Elo numbers on the tournament (pairwise judge ranking suffices at N≤6
  candidates; Elo earns its keep with evolution rounds)

## Phase 3b — Gateway + Next.js/AG-UI app ✅ (complete 2026-06-11)

- [x] `gateway.py` — FastAPI: AG-UI chat plane at `/agui` via `ag_ui_adk`
      middleware (`ADKAgent.from_app`, thread id == project id, SQLite
      services, no session GC, messages-snapshot rehydration) + REST API
      (projects, history, status/resume, artifacts + content + lineage,
      runs + kill, board, budget) with CORS for the UI dev server.
      Run: `uv run uvicorn deep_researcher.gateway:app --port 8042`
- [x] `ui/` — Next.js 16 + CopilotKit v2 (`CopilotChat` bound to the
      project's thread) + Tailwind: project sidebar, chat, Runs pane
      (3s polling, budget meter, kill-branch button), Board pane, Artifacts
      pane (markdown + KaTeX via remark-math/rehype-katex, Vega-Lite via
      react-vega `VegaEmbed`, images, version selector, lineage
      click-through), resume banner. Run: `cd ui && npm run dev`
- [x] validation: gateway REST tests (4); `npm run build` clean; live AG-UI
      SSE round-trip through `/agui` (RUN_STARTED → streamed deltas →
      RUN_FINISHED with correct project context); Playwright screenshots of
      Runs/Board/Artifacts panes against the real demo projects

**Decisions (vs. design doc)**
- AG-UI + CopilotKit chosen over hand-rolled SSE (the design's recommended
  default): `ag_ui_adk` is officially documented for ADK and handles
  streaming + session mapping; the REST panes stay plain FastAPI.
- Chat history renders after the first run on a thread (AG-UI snapshot
  semantics); the gateway's `/history` endpoint exists if pre-run hydration
  is wanted later.
- CopilotKit's dev-mode inspector overlay (`cpk-web-inspector`) intercepts
  clicks in `next dev`; absent in production builds.
- Streamlit app retained as the lightweight fallback UI.

## Phase 4 — Hardening

- leakage/usage/critique guardrails, notifications, multi-project dashboard

## Environment

`~/.env` keys: `DEEPSEEK_API_KEY` (required),
`SEMANTIC_SCHOLAR_API_KEY` (optional), `OPENALEX_MAILTO` (optional, polite pool),
`DATA_ROOT` (optional, default `~/data/deep-researcher`).
