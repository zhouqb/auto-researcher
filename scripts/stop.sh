#!/usr/bin/env bash
# Free the service's ports: kill whatever holds :8042 (gateway) and :3001
# (this UI). Needed because `screen -X quit` SIGKILLs serve.sh's shell without
# running any trap, orphaning the backgrounded uvicorn/next (reparented to
# init). We kill each listener's whole PROCESS GROUP, so the wrapper and its
# grandchildren go together (uv→python, npm→next). Port-scoped — it only ever
# touches this service, never the langfuse UI on :3000.
set -uo pipefail

# Per-port selectors only (a bare -iTCP would OR in ALL tcp and defeat the
# scoping). 8042 = gateway, 3001 = this UI; :3000 (langfuse) is left alone.
LSOF_PORTS=(-iTCP:8042 -iTCP:3001)

# pgids of every process currently LISTENing on our ports
listener_pgids() {
  local pid
  for pid in $(lsof -nP -sTCP:LISTEN -t "${LSOF_PORTS[@]}" 2>/dev/null); do
    ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' '
  done | sort -u
}

signal_groups() {  # $1 = signal
  local pgid
  for pgid in $(listener_pgids); do
    [ -n "$pgid" ] && kill "-$1" "-${pgid}" 2>/dev/null || true
  done
}

signal_groups TERM           # ask politely
for _ in 1 2 3 4 5; do       # wait up to ~5s for graceful exit
  [ -z "$(listener_pgids)" ] && break
  sleep 1
done
signal_groups KILL           # force any survivor

if [ -n "$(listener_pgids)" ]; then
  echo "stop.sh: WARNING — something still holds :8042/:3001" >&2
  exit 1
fi
echo "stop.sh: :8042 and :3001 free"
