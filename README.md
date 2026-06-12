# auto-researcher

Deep Researcher: a locally hosted, human-steered multi-agent research system.
It takes an ML/AI/SWE research question and runs the research lifecycle —
clarification, planning (with human plan approval), parallel literature
search, synthesis, and a cited report — with code experiments via OpenAI
Codex arriving in Phase 1. Design: [docs/design.md](docs/design.md); status:
[PLAN.md](PLAN.md).

Built on Google ADK (orchestration), DeepSeek via LiteLLM (models), SQLite +
filesystem (all state), OpenAlex/arXiv/Semantic Scholar (literature search),
Streamlit (Phase 0 steering UI).

## Setup

```sh
uv sync
```

Add to `~/.env` (or a project-local `.env`):

```sh
DEEPSEEK_API_KEY=sk-...          # required
SEMANTIC_SCHOLAR_API_KEY=...     # optional, unlocks S2 search
OPENALEX_MAILTO=you@example.com  # optional, OpenAlex polite pool
DATA_ROOT=~/data/deep-researcher # optional, this is the default
```

## Run

Steering UI (chat + plan-approval gate + artifact browser):

```sh
uv run streamlit run app/streamlit_app.py
```

Terminal REPL:

```sh
uv run deep-researcher "How do MoE routing strategies affect inference throughput?"
uv run deep-researcher --project my-project        # resume
```

Everything durable lands under `DATA_ROOT`: one SQLite database (sessions +
artifact catalog) and `projects/<id>/` with the research brief, plan,
decision log, checkpoint records, literature notes, and reports. Backup is
copying that folder.

## Tests

```sh
uv run pytest
```

Includes an offline integration test that drives the whole agent pipeline
with a scripted mock LLM — no API key needed.
