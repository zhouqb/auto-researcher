#!/usr/bin/env bash
# Dev mode: gateway + web UI with auto-reload on code changes.
#
# - Backend restarts on any change under src/ (uvicorn --reload). An in-flight
#   chat turn dies with the restart, but detached Codex runs survive (own
#   process groups, idempotent run markers) and sessions are resumable.
# - Frontend hot-reloads via next dev. Package upgrades (node_modules) still
#   need a rerun of this script — code changes don't.
#
# Ctrl+C stops both.
set -euo pipefail
cd "$(dirname "$0")/.."

uv run uvicorn deep_researcher.gateway:app --port 8042 --reload --reload-dir src &
GATEWAY_PID=$!
trap 'kill "$GATEWAY_PID" 2>/dev/null' EXIT

cd ui && npm run dev
