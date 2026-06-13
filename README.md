# Deep Researcher

A locally hosted, human-steered multi-agent system. It handles two kinds of
request:

- **Research** — give it an ML/AI/SWE question and it runs the full research
  lifecycle: clarification, planning, parallel literature search, idea
  tournament, **real code experiments** (implemented and executed by OpenAI
  Codex in a sandbox), analysis, critique, and a cited final report.
- **Repo improvement** — point it at an existing repo (a local path or a git
  URL) and describe a change. It tries competing implementation approaches in
  parallel isolated clones, runs the repo's own tests, picks the one that goes
  green, and hands you a `change.diff` (plus a PR description) — never touching
  your original repo or pushing anything unprompted.

In both cases you approve the plan and the budget before anything expensive
runs. You are the PI; the system proposes and executes. Everything durable is a
versioned, lineage-tracked artifact on your disk. The conversation is
ephemeral; the artifacts are the project.

- **Design:** [docs/design.md](docs/design.md)
- **Implementation status & decisions:** [PLAN.md](PLAN.md) (all phases complete)

## How a project runs

```
your question
   │
   ▼
clarify (≤3 questions) ──► research_brief.md          ← the project contract
   │
   ▼
plan + literature facets ──► plan.md, board.json
   │
   ▼ ❶ GATE: you approve the plan
   │
parallel literature searchers (Semantic Scholar / arXiv / OpenAlex / GitHub)
   ──► lit/facet_*/notes.md ──► lit/synthesis.md
   │
   ▼ (if an experiment is in scope)
idea tournament (3 personas → LLM-judge ranking) ──► hypotheses.json
experiment designer ──► exp_spec.md (+ cost estimate)
   │
   ▼ ❷ GATE: you approve the budget
   │
parallel Codex experiment branches (sandboxed, killable, budget-metered)
   ──► repo/ · metrics.json · plots/ · run logs
   │
   ▼
result analyst (cross-branch ranking) ──► analysis.md
experience memory updated (successes AND failures, cross-project)
   │
   ▼
report writer ──► final_report.md ──► critic reviews; blocking findings
                                       force a revision before delivery
```

You can steer at any point in chat: redirect, request plan changes, kill a
branch from the run monitor, or stop. Every gate decision, kill, and pivot is
recorded (`checkpoints/`, `decisions.md`).

**Repo improvement** reuses the same lifecycle with the deliverable swapped:
when you give a repo (path or URL), the system enters repo mode, usually skips
literature, and the experiment branches become competing implementation
approaches. Each branch runs in its own **clone** of the repo (your original is
never modified), makes the change, runs the repo's tests (auto-detected:
pytest / `npm test` / `make test`, or set `REPO_DEFAULT_TEST_COMMAND`), and
iterates to green. The analyst ranks by *tests-pass + acceptance criteria +
diff minimality* and the winner's change lands at
`iter_1/exp_<branch>/change.diff`. The report includes a ready-to-use PR
description; opening a `gh` PR happens only if you ask.

```
"improve the repo at ~/code/myproj: make the cache LRU-bounded"
   │  set_target_repo (clone if URL, detect test command) → repo mode
   ▼  brief + plan (no literature) ──► ❶ GATE: approve plan
approach tournament (conservative / novel / minimal) ──► hypotheses.json
change designer ──► exp_spec.md (per-approach Codex prompts) ──► ❷ GATE: budget
   ▼
parallel branches, each a CLONE: make change · run repo tests · outcome.json
   ──► iter_1/exp_<branch>/change.diff
   ▼
analyst ranks (green? acceptance? minimal?) ──► winner
report (+ PR description) ──► critic reviews the diff ──► change.diff delivered
```

## Stack

| Concern | Choice |
|---|---|
| Orchestration | Google ADK (LoopAgent/ParallelAgent, sessions, resumability) |
| Models | DeepSeek via LiteLLM (`deepseek/deepseek-chat`; swappable in config) |
| Code experiments | OpenAI Codex CLI, `--sandbox workspace-write`, via your ChatGPT login |
| Literature search | Semantic Scholar (primary; key recommended), arXiv, OpenAlex (keyless secondary), GitHub (implementations) on by default; OpenReview (peer-review signal) and Tavily web search (needs key) opt-in via `SEARCH_TOOLS` |
| State | One SQLite file + plain files under `DATA_ROOT` — backup = copy a folder |
| UI | Next.js + AG-UI/CopilotKit (full) · Streamlit (lightweight) · CLI |

No cloud infrastructure: no GCP, no Vertex, no AWS. Everything runs on your
machine.

## Setup

Prerequisites: [uv](https://docs.astral.sh/uv/), Node ≥ 20 (for the web UI),
and the [Codex CLI](https://developers.openai.com/codex) authenticated via
`codex login` (only needed for experiments).

```sh
uv sync                       # Python deps
(cd ui && npm install)        # web UI deps (optional)
```

Add keys to `~/.env`, a project-local `.env`, or your shell profile:

```sh
DEEPSEEK_API_KEY=sk-...          # required — drives all agents
TAVILY_API_KEY=tvly-...          # optional: web search key (also add "web" to SEARCH_TOOLS)
GITHUB_TOKEN=ghp_...             # optional: raises GitHub search limits (else falls back to `gh auth token`)
SEMANTIC_SCHOLAR_API_KEY=...     # recommended: S2 is the primary paper index (unauth pool 429s heavily)
OPENALEX_MAILTO=you@example.com  # optional: OpenAlex "polite pool" (faster, more reliable)
NOTIFY_WEBHOOK_URL=https://...   # optional: Slack-style webhook on run completion
```

## Run

**Web UI** (chat, multi-project dashboard, live run monitor with kill-branch,
kanban board, artifact browser with KaTeX/Vega rendering and lineage
click-through):

```sh
./scripts/serve.sh   # stable service: backend ignores source edits (use in screen/tmux)
./scripts/dev.sh     # development: backend auto-reloads on src/ changes
```

Both start backend + frontend together; one Ctrl+C stops both. Use serve.sh
for the instance you actually research with — dev.sh's auto-reload kills
in-flight runs on every backend edit.

The UI runs on http://localhost:3001 (pinned; Langfuse holds 3000).
Or run the two halves in separate terminals:

```sh
uv run uvicorn deep_researcher.gateway:app --port 8042 --reload --reload-dir src
cd ui && npm run dev
```

**Friendly names** (optional): a local Caddy proxy serves
http://researcher.localhost and http://langfuse.localhost — no port numbers.
One-time setup (macOS resolves `*.localhost` to loopback natively):

```sh
brew install caddy
cp scripts/Caddyfile "$(brew --prefix)/etc/Caddyfile"
brew services start caddy   # always-on, survives reboots
```

For clean Langfuse logins under the new name, set
`NEXTAUTH_URL=http://langfuse.localhost` in the Langfuse deployment's `.env`
and recreate its `langfuse-web` container.

**Streamlit** (single-page fallback — chat, gate buttons, runs, board,
artifacts):

```sh
uv run streamlit run app/streamlit_app.py
```

**Terminal REPL**:

```sh
uv run deep-researcher "How do MoE routing strategies affect inference throughput?"
uv run deep-researcher --project my-project            # reopen a project
uv run deep-researcher --project my-project --resume   # recover a crashed run
```

A typical session: ask the question → answer its clarifying questions → read
the plan summary and say "approve" → wait out the literature stage → review
the experiment spec + cost estimate and approve the budget → watch branches in
the run monitor → read the report.

## What you get on disk

Each project lives at `DATA_ROOT/projects/<project-id>/` (default
`~/data/deep-researcher`):

```
brief/research_brief.md      # the contract: question, scope, success criteria
plan/plan.md  plan/board.json
decisions/decisions.md       # append-only ADR log — skim this to audit the project
checkpoints/<ts>_<gate>.json # every approval you gave, with your words
lit/facet_*/notes.md  lit/synthesis.md
iter_1/
  hypotheses.json            # ranked candidate ideas (tournament output)
  exp_spec.md  analysis.md  critique.md
  exp_<branch>/
    repo/                    # git repo Codex worked in: code, metrics.json, plots/
    runs/<run_id>/codex_events.jsonl  result.json
budget/budget.json           # tokens + wallclock per run, with totals
reports/final_report.md      # cited; every claim traceable via the lineage graph
```

The SQLite database (`deep_researcher.db`) holds chat sessions, the artifact
catalog (versions, lineage, full-text search), cross-project experience
memory, and the jobs table. `scripts/backup.sh` archives the whole data root
(keeps the last 10).

## Safety & cost controls

- **Two hard gates** — nothing is planned-around without your approval, and no
  Codex run launches before you approve its budget estimate.
- **Sandboxed execution** — experiments run under Codex `workspace-write`
  confinement (writes limited to the branch workspace, network off).
- **Kill-branch** — terminate any running branch from the UI without touching
  its siblings.
- **Idempotent runs** — a crash/resume never re-launches a Codex run that
  already completed (result markers keyed by prompt).
- **Budget ledger** — every run's tokens and wallclock land in `budget.json`,
  surfaced as a meter in both UIs.
- **Critic guardrail** — a dedicated agent reviews the report for unsupported
  claims, leakage signs, and single-seed overreach; blocking findings force a
  revision before delivery.

## Observability

Logs go to the console and `DATA_ROOT/logs/deep_researcher.log` (rotating,
level via `LOG_LEVEL`).

Tracing is built in but off by default. ADK instruments every agent step,
LLM call, and tool call with OpenTelemetry spans; to see them, self-host
[Langfuse](https://langfuse.com/self-hosting) (free, MIT-licensed core):

```sh
git clone https://github.com/langfuse/langfuse.git && cd langfuse
docker compose up -d        # UI at http://localhost:3000
```

Create a project in the Langfuse UI, grab its API keys, and add:

```sh
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=http://localhost:3000   # default
```

Restart the gateway and every research turn appears as a trace — the full
orchestrator → sub-agent → model/tool span tree with latencies. Without the
keys, tracing stays a silent no-op.

## Configuration

All settings are env vars (or `.env` entries); defaults in
`src/deep_researcher/config.py`:

| Variable | Default | Meaning |
|---|---|---|
| `DATA_ROOT` | `~/data/deep-researcher` | where all state lives |
| `ORCHESTRATOR_MODEL` / `WORKER_MODEL` | `deepseek/deepseek-chat` | LiteLLM model ids |
| `CODEX_MODEL` | Codex CLI default | model for experiment runs |
| `CODEX_TIMEOUT_S` | `3600` | per-run wallclock cap |
| `REPO_DEFAULT_TEST_COMMAND` | (auto-detect) | fallback test command for repo improvement when none is detected |
| `SEARCH_TOOLS` | `semantic_scholar,arxiv,openalex,github` | enabled search backends (also available: `openreview`, `web`) |
| `MAX_LIT_FACETS` | `3` | parallel literature searchers |
| `MAX_EXPERIMENT_BRANCHES` | `3` | parallel experiment branches |
| `MAX_CODEX_CONCURRENCY` | `2` | concurrent Codex processes |
| `MAX_CLARIFYING_QUESTIONS` | `3` | intake question budget |
| `DESKTOP_NOTIFICATIONS` | `true` | notify on run completion (macOS/Linux) |

## Development

```sh
uv run pytest          # 77 tests, no API key needed (scripted mock LLM + fake codex binary)
cd ui && npm run build # type-checks the web UI
```

Live end-to-end checks (cost real tokens/Codex quota):

```sh
uv run python scripts/live_e2e.py             # literature-only project
uv run python scripts/live_e2e_experiment.py  # single Codex experiment, both gates
uv run python scripts/live_e2e_parallel.py    # tournament + parallel branches
```

Repo layout: `src/deep_researcher/` (agents, tools, storage, codex driver,
gateway, monitor) · `app/` (Streamlit) · `ui/` (Next.js) · `tests/` ·
`scripts/` · `docs/design.md` · `PLAN.md`.
