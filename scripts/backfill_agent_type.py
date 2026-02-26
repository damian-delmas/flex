#!/usr/bin/env python3
"""One-time backfill: populate agent_type on existing _edges_delegations rows.

Re-parses JONLs for sessions with delegation edges, extracts subagent_type
from Task tool_use blocks, UPDATEs where agent_type IS NULL.

Safe to run multiple times — only touches NULL rows, skips already-populated.
"""

import json
import re
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from flex.registry import resolve_cell
from flex.modules.claude_code.compile.worker import find_jsonl


def main():
    db_path = resolve_cell('claude_code')
    conn = sqlite3.connect(str(db_path), timeout=30.0)
    conn.row_factory = sqlite3.Row

    # Get sessions with NULL agent_type delegations
    rows = conn.execute("""
        SELECT DISTINCT d.parent_source_id as session_id
        FROM _edges_delegations d
        WHERE d.agent_type IS NULL
          AND d.parent_source_id IS NOT NULL
    """).fetchall()

    if not rows:
        print("No NULL agent_type rows — nothing to backfill")
        return

    print(f"Backfilling agent_type for {len(rows)} sessions...")
    updated = 0
    skipped = 0

    for row in rows:
        session_id = row['session_id']
        jsonl_path = find_jsonl(session_id)
        if not jsonl_path or not jsonl_path.exists():
            skipped += 1
            continue

        # Build tool_use_id -> subagent_type map from JSONL
        tuid_to_agent_type = {}
        try:
            with open(jsonl_path, 'r') as f:
                for line in f:
                    if 'Task' not in line:
                        continue
                    try:
                        e = json.loads(line)
                    except (json.JSONDecodeError, ValueError):
                        continue
                    if e.get('type') != 'assistant':
                        continue
                    msg = e.get('message', {})
                    for block in msg.get('content', []):
                        if isinstance(block, dict) and block.get('type') == 'tool_use' \
                                and block.get('name') == 'Task':
                            tuid = block.get('id', '')
                            at = block.get('input', {}).get('subagent_type')
                            if tuid and at:
                                tuid_to_agent_type[tuid] = at
        except Exception as e:
            print(f"  [{session_id[:8]}] read error: {e}", file=sys.stderr)
            skipped += 1
            continue

        if not tuid_to_agent_type:
            skipped += 1
            continue

        # Also build agentId -> parentToolUseID from progress entries
        agent_to_parent = {}
        try:
            with open(jsonl_path, 'r') as f:
                for line in f:
                    if 'agentId' not in line or 'progress' not in line:
                        continue
                    try:
                        e = json.loads(line)
                    except (json.JSONDecodeError, ValueError):
                        continue
                    if e.get('type') != 'progress':
                        continue
                    data = e.get('data', {})
                    if isinstance(data, str):
                        try:
                            data = json.loads(data)
                        except (json.JSONDecodeError, ValueError):
                            continue
                    aid = data.get('agentId', '') if isinstance(data, dict) else ''
                    ptuid = e.get('parentToolUseID', '')
                    if aid and ptuid and aid not in agent_to_parent:
                        agent_to_parent[aid] = ptuid
        except Exception:
            pass

        # Get NULL agent_type delegations for this session
        deleg_rows = conn.execute("""
            SELECT chunk_id, child_session_id
            FROM _edges_delegations
            WHERE parent_source_id = ? AND agent_type IS NULL
        """, (session_id,)).fetchall()

        for dr in deleg_rows:
            child_sid = dr['child_session_id']
            # Extract agent hash from child_session_id (agent-{hash})
            agent_hash = child_sid.replace('agent-', '') if child_sid.startswith('agent-') else ''

            # Try to resolve via agentId -> parentToolUseID -> subagent_type
            at = None
            ptuid = agent_to_parent.get(agent_hash)
            if ptuid:
                at = tuid_to_agent_type.get(ptuid)

            # Fallback: if only one Task tool_use type in the session, use it
            if not at and len(set(tuid_to_agent_type.values())) == 1:
                at = next(iter(tuid_to_agent_type.values()))

            if at:
                conn.execute("""
                    UPDATE _edges_delegations
                    SET agent_type = ?
                    WHERE chunk_id = ? AND child_session_id = ?
                """, (at, dr['chunk_id'], child_sid))
                updated += 1

    conn.commit()
    conn.close()

    total_null = sum(1 for r in rows) * 1  # approximate
    print(f"Done. Updated {updated} rows, skipped {skipped} sessions.")


if __name__ == '__main__':
    main()
