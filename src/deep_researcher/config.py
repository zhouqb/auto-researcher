"""Runtime configuration for Deep Researcher.

Settings load from the environment, ``~/.env``, and a project-local ``.env``
(later sources win). Everything lives under ``data_root`` (design §7.2):
one SQLite database plus a ``projects/`` tree of artifacts.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(str(Path.home() / ".env"), ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Model access. LiteLLM model ids; DeepSeek by default per project decision.
    deepseek_api_key: Optional[str] = None
    orchestrator_model: str = "deepseek/deepseek-chat"
    worker_model: str = "deepseek/deepseek-chat"

    # Search backends handed to agents (comma-separated subset of: openalex,
    # arxiv, semantic_scholar, openreview, github, web). Default is the lean
    # keyless trio; semantic_scholar (429s without a key), openreview, and
    # web (needs TAVILY_API_KEY) are opt-in.
    search_tools: str = "openalex,arxiv,github"

    # Optional: raises Semantic Scholar rate limits when present.
    semantic_scholar_api_key: Optional[str] = None
    # Optional: joins OpenAlex's "polite pool" (faster, more consistent).
    openalex_mailto: Optional[str] = None
    # Key for search_web (Tavily); add "web" to SEARCH_TOOLS to enable it.
    tavily_api_key: Optional[str] = None
    # Optional: raises GitHub search rate limits (falls back to `gh auth token`).
    github_token: Optional[str] = None

    app_name: str = "deep_researcher"
    data_root: Path = Path("~/data/deep-researcher")

    # Codex experiment execution (design §13).
    codex_model: Optional[str] = None  # None → Codex CLI default
    codex_timeout_s: float = 3600

    # Steering knobs (design §4). Default small; escalate only on promise.
    max_clarifying_questions: int = 3
    max_lit_facets: int = 3
    max_experiment_branches: int = 3
    max_codex_concurrency: int = 2

    # Context compaction (design §6): summarize every N events, keep overlap.
    compaction_interval: int = 40
    compaction_overlap: int = 8

    # Notifications (design §11.4).
    desktop_notifications: bool = True
    notify_webhook_url: Optional[str] = None

    # Observability: file/console logging level; optional Langfuse tracing
    # (self-hosted, both keys required to enable OTLP span export).
    log_level: str = "INFO"
    langfuse_host: str = "http://localhost:3000"
    langfuse_public_key: Optional[str] = None
    langfuse_secret_key: Optional[str] = None

    @property
    def root(self) -> Path:
        return self.data_root.expanduser()

    @property
    def db_path(self) -> Path:
        return self.root / "deep_researcher.db"

    @property
    def session_db_url(self) -> str:
        return f"sqlite+aiosqlite:///{self.db_path}"

    @property
    def projects_dir(self) -> Path:
        return self.root / "projects"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.root.mkdir(parents=True, exist_ok=True)
    settings.projects_dir.mkdir(parents=True, exist_ok=True)
    # LiteLLM reads provider keys from the environment.
    if settings.deepseek_api_key and not os.environ.get("DEEPSEEK_API_KEY"):
        os.environ["DEEPSEEK_API_KEY"] = settings.deepseek_api_key
    return settings
