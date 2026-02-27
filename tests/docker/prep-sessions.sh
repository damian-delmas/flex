#!/usr/bin/env bash
# prep-sessions.sh — copy last N days of Claude Code sessions to a staging dir.
#
# Usage:
#   bash tests/docker/prep-sessions.sh          # default: 30 days
#   bash tests/docker/prep-sessions.sh 7        # last 7 days
#   bash tests/docker/prep-sessions.sh 60       # last 60 days
#
# Output: /tmp/flex-staging/  (ready to mount into container)
# Then:
#   docker run -it --rm \
#     -v flex-dev-cell:/root/.flex \
#     -v ~/.flex/models:/root/.flex/models:ro \
#     -v /tmp/flex-staging:/root/.claude/projects:ro \
#     -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
#     flex-dev

set -euo pipefail

DAYS="${1:-30}"
SRC="$HOME/.claude/projects"
DEST="/tmp/flex-staging"

echo "Copying last ${DAYS} days of sessions from $SRC → $DEST"

rm -rf "$DEST"
mkdir -p "$DEST"

# Find JONLs modified in last N days, preserve directory structure
count=0
while IFS= read -r f; do
    rel="${f#$SRC/}"
    dir="$(dirname -- "$rel")"
    mkdir -p "$DEST/$dir"
    cp "$f" "$DEST/$dir/" 2>/dev/null && count=$((count + 1)) || true
done < <(find "$SRC" -name "*.jsonl" -mtime "-${DAYS}")

echo "  Copied $count sessions (${DAYS}d window)"
echo "  Staging dir: $DEST"
echo ""
echo "Run:"
echo "  docker run -it --rm \\"
echo "    -v flex-dev-cell:/root/.flex \\"
echo "    -v ~/.flex/models:/root/.flex/models:ro \\"
echo "    -v $DEST:/root/.claude/projects:ro \\"
echo "    -e ANTHROPIC_API_KEY=\$ANTHROPIC_API_KEY \\"
echo "    flex-dev"
