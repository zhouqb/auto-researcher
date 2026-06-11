# Implementation Plan & Status

Implements [docs/design.md](docs/design.md) (Deep Researcher v3) phase by phase.
Deviations from the design doc are listed per phase under **Decisions**.

## Phase 0 — Skeleton ✅ (code complete; live E2E pending API key)

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
- [ ] **live end-to-end run** — blocked on `DEEPSEEK_API_KEY` in `~/.env`

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

## Phase 1 — Codex execution (next)

- `codex_exec` `LongRunningFunctionTool`, **idempotent** (marker/thread-id check
  before launch — non-negotiable, design §16), `--sandbox workspace-write`
- `AGENTS.md` workspace contract: `metrics.json`, dual plots (`.vega.json` + `.png`),
  commit per step, forbidden ops
- JSONL event parsing (`thread.started`, `item.completed`, `turn.completed` usage)
- fix loop via `codex exec resume <thread_id>`
- Gate 2 budget approval (LRFT) + `budget.json` ledger
- run monitor: tail `codex_events.jsonl` into the UI

## Phase 2 — Memory + resumability

- experience store (success+failure schema §5) on FTS5; retrieval at planning time
- `ResumabilityConfig(is_resumable=True)`; crash recovery; pause/hard-cancel
- steering inbox; `board.json`; compaction at stage boundaries; `todo.md` recitation

## Phase 3 — Parallel exploration + full UI

- experiment branch fan-out, Elo/LLM-judge tournament, metric pruning (AIDE/MLE-STAR)
- concurrency + budget knobs; mid-run review gate (Gate 3)
- Next.js + AG-UI app (lineage click-through, board drag, budget meter)

## Phase 4 — Hardening

- leakage/usage/critique guardrails, notifications, multi-project dashboard

## Environment

`~/.env` keys: `DEEPSEEK_API_KEY` (required),
`SEMANTIC_SCHOLAR_API_KEY` (optional), `OPENALEX_MAILTO` (optional, polite pool),
`DATA_ROOT` (optional, default `~/data/deep-researcher`).
