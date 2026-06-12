#!/bin/sh
# Backup the Deep Researcher data root (design: "backup is copying a folder").
# Usage: scripts/backup.sh [dest_dir]   (default ~/backups/deep-researcher)
# Keeps the 10 most recent archives.

set -eu

DATA_ROOT="${DATA_ROOT:-$HOME/data/deep-researcher}"
DEST="${1:-$HOME/backups/deep-researcher}"
KEEP=10

[ -d "$DATA_ROOT" ] || { echo "data root not found: $DATA_ROOT" >&2; exit 1; }
mkdir -p "$DEST"

STAMP=$(date +%Y%m%dT%H%M%S)
ARCHIVE="$DEST/deep-researcher-$STAMP.tar.gz"

# Exclude reproducible bulk: per-run logs stay, model checkpoints/datasets out.
tar -czf "$ARCHIVE" \
  --exclude='data_store' \
  -C "$(dirname "$DATA_ROOT")" "$(basename "$DATA_ROOT")"

echo "wrote $ARCHIVE ($(du -h "$ARCHIVE" | cut -f1))"

# prune old archives
ls -1t "$DEST"/deep-researcher-*.tar.gz 2>/dev/null | tail -n +$((KEEP + 1)) |
  while read -r old; do rm -f "$old" && echo "pruned $old"; done
