#!/usr/bin/env bash
# Restart the long-running `researcher` service cleanly.
#
# `screen -X quit` SIGKILLs serve.sh (no trap can run), so a bare quit+start
# leaves an orphaned uvicorn holding :8042 and the new gateway fails to bind.
# This sequences it correctly: quit the session → stop.sh frees the ports →
# start a fresh detached session.
set -uo pipefail
cd "$(dirname "$0")/.."

screen -S researcher -X quit 2>/dev/null || true
./scripts/stop.sh
screen -dmS researcher ./scripts/serve.sh
echo "researcher restarted — gateway :8042, UI :3001 (give next dev a few seconds)"
