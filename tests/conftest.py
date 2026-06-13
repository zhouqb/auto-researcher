"""Test-suite hermeticity.

Settings read the developer's ``~/.env`` as a config source, so anything a
test does not explicitly override falls through to real machine config.
Every Settings field that gates an external side effect (keys, exporters,
webhooks, binaries, data paths) is neutralized here; environment variables
take precedence over the env file, and per-test ``monkeypatch.setenv`` still
overrides these defaults.

Maintained manually: when adding a Settings field with an external effect,
add its neutralization here.
"""

import os
import tempfile

# Default data root: a throwaway dir, so a test that forgets to set
# DATA_ROOT can never write into ~/data/deep-researcher.
os.environ.setdefault("DATA_ROOT", tempfile.mkdtemp(prefix="dr-test-"))

# Model/provider keys: dummies so nothing authenticates anywhere real.
os.environ.setdefault("DEEPSEEK_API_KEY", "test-dummy")
os.environ.setdefault("SEMANTIC_SCHOLAR_API_KEY", "")
os.environ.setdefault("OPENALEX_MAILTO", "")
os.environ.setdefault("TAVILY_API_KEY", "")  # search_web must report unconfigured
os.environ.setdefault("GITHUB_TOKEN", "test-dummy")  # skip `gh auth token` subprocess

# Notifications: never fire desktop popups or webhooks from tests.
os.environ.setdefault("DESKTOP_NOTIFICATIONS", "false")
os.environ.setdefault("NOTIFY_WEBHOOK_URL", "")

# Tracing: never export spans to a real Langfuse from tests.
os.environ.setdefault("LANGFUSE_PUBLIC_KEY", "")
os.environ.setdefault("LANGFUSE_SECRET_KEY", "")

# Codex: a test that forgets to install a fake binary fails fast instead of
# launching a real (paid) Codex run.
os.environ.setdefault("CODEX_BINARY", "false")

# Repo-improvement mode: no machine-specific fallback test command bleeds in.
os.environ.setdefault("REPO_DEFAULT_TEST_COMMAND", "")
