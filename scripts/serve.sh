#!/usr/bin/env bash
# Service mode: gateway + web UI, NO backend auto-reload.
#
# Use this for the long-running instance (e.g. in a screen session). The
# gateway ignores source edits, so in-flight research runs survive while
# development happens in the same working tree — restart deliberately to
# pick up backend changes:
#
#   screen -S researcher -X quit && screen -dmS researcher ./scripts/serve.sh
#
# The UI still runs `next dev` (hot reload): UI reloads never kill a backend
# run, and production builds would need a rebuild on every change instead.
# For active backend development use scripts/dev.sh (auto-reload) instead.
set -euo pipefail
cd "$(dirname "$0")/.."

uv run uvicorn deep_researcher.gateway:app --port 8042 &
GATEWAY_PID=$!
trap 'kill "$GATEWAY_PID" 2>/dev/null' EXIT

cd ui && npm run dev
