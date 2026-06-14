#!/usr/bin/env bash
# Service mode: gateway + web UI, NO backend auto-reload.
#
# Use this for the long-running instance (e.g. in a screen session). The
# gateway ignores source edits, so in-flight research runs survive while
# development happens in the same working tree — restart deliberately to
# pick up backend changes. To restart, use:
#
#   ./scripts/restart.sh
#
# NOT `screen -X quit` on its own: screen quit SIGKILLs this shell (uncatchable
# — no trap runs), which orphans the backgrounded uvicorn/next (reparented to
# init) and leaves :8042 / :3001 held. restart.sh runs scripts/stop.sh to free
# those ports by process group before relaunching.
#
# The trap below DOES tear both children down cleanly for the *catchable*
# cases — Ctrl-C and `kill <pid>` (SIGINT/SIGTERM). `set -m` (job control) puts
# each background job in its own process group, so `kill -- -<pid>` reaches the
# wrapper AND its grandchildren (uv→python, npm→next); portable to macOS, which
# has no `setsid`.
#
# The UI runs `next dev` (hot reload): UI reloads never kill a backend run,
# and production builds would need a rebuild on every change instead. For
# active backend development use scripts/dev.sh (auto-reload) instead.
set -euo pipefail
set -m  # job control: each `&` job becomes its own process group leader
cd "$(dirname "$0")/.."

pids=()
cleanup() {
  trap - INT TERM HUP EXIT  # disarm so cleanup runs once
  for pid in "${pids[@]}"; do
    kill -TERM "-${pid}" 2>/dev/null || true  # negative pid → whole group
  done
  wait 2>/dev/null || true
}
trap cleanup INT TERM HUP EXIT

uv run uvicorn deep_researcher.gateway:app --port 8042 &
pids+=("$!")

( cd ui && exec npm run dev ) &
pids+=("$!")

wait  # block until a child exits or a signal fires cleanup
