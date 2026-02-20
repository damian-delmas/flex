#!/bin/bash
#
# PostToolUse Hook - Notify worker of session activity
#
# FAST PATH: Just notify session_id (~1ms)
# Worker daemon syncs from JSONL in background
#
set -uo pipefail

SESSION_ID=$(jq -r '.session_id // empty')
[[ -z "$SESSION_ID" ]] && exit 0

QUEUE_DB="${FLEX_HOME:-$HOME/.flex}/queue.db"
sqlite3 "$QUEUE_DB" \
  "CREATE TABLE IF NOT EXISTS claude_code_pending (session_id TEXT PRIMARY KEY, ts INTEGER);
   INSERT OR REPLACE INTO claude_code_pending VALUES ('$SESSION_ID', $(date +%s))" \
  2>/dev/null || true
