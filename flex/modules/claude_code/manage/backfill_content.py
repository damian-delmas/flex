#!/usr/bin/env python3
"""
JSONL Content Backfill — parent_uuid + tool content extraction.

Two passes over all JSONL files:
  Pass 1: parent_uuid — UPDATE _types_message rows from JSONL parentUuid
  Pass 2: tool_content — extract tool_use inputs + tool_result outputs to _raw_content

Usage:
  python -m flex.modules.claude_code.manage.backfill_content              # run both
  python -m flex.modules.claude_code.manage.backfill_content --dry-run     # report only
  python -m flex.modules.claude_code.manage.backfill_content --limit 50    # 50 sessions
  python -m flex.modules.claude_code.manage.backfill_content --pass parent_uuid
  python -m flex.modules.claude_code.manage.backfill_content --pass tool_content
"""

import hashlib
import json
import sys
import time
import argparse
import sqlite3
from pathlib import Path

from flex.registry import resolve_cell


CLAUDE_PROJECTS = Path.home() / ".claude/projects"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def find_all_jsonl() -> list[Path]:
    """Discover all JSONL session files."""
    return sorted(CLAUDE_PROJECTS.rglob("*.jsonl"))


def parse_jsonl_entries(jsonl_path: Path) -> list[tuple[int, dict]]:
    """Parse JSONL file, yielding (line_num, entry) pairs.

    Skips snapshot lines, progress entries, and malformed JSON.
    """
    entries = []
    try:
        with open(jsonl_path, 'r', errors='replace') as f:
            lines = f.readlines()
    except Exception:
        return entries

    for line_num, line in enumerate(lines, 1):
        try:
            entry = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue

        entry_type = entry.get('type')
        if entry_type in ('progress', 'file-history-snapshot'):
            continue
        if entry_type not in ('user', 'assistant'):
            continue

        entries.append((line_num, entry))

    return entries


def _normalize_tool_result_content(content) -> str | None:
    """Normalize tool_result content to a string.

    - str -> return as-is
    - list[dict] -> join text values with newline
    - None/empty -> return None
    """
    if isinstance(content, str):
        return content if content else None
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get('type') == 'text':
                parts.append(item.get('text', ''))
            elif isinstance(item, str):
                parts.append(item)
        return '\n'.join(parts) if parts else None
    return None


def _store_content(conn: sqlite3.Connection, chunk_id: str, raw: str,
                   tool_name: str, ts: int):
    """Store raw content — no size cap. SHA-256 dedup."""
    if not raw or len(raw) < 10:
        return
    # Sanitize surrogates before hashing and storing
    raw = raw.encode('utf-8', errors='surrogatepass').decode('utf-8', errors='replace')
    h = hashlib.sha256(raw.encode('utf-8')).hexdigest()
    conn.execute(
        "INSERT OR IGNORE INTO _raw_content VALUES (?,?,?,?,?)",
        (h, raw, tool_name, len(raw), ts)
    )
    conn.execute(
        "INSERT OR IGNORE INTO _edges_raw_content VALUES (?,?)",
        (chunk_id, h)
    )


def _ensure_tables(conn: sqlite3.Connection):
    """Ensure content store tables exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS _raw_content (
            hash TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            tool_name TEXT,
            byte_length INTEGER,
            first_seen INTEGER
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS _edges_raw_content (
            chunk_id TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            PRIMARY KEY (chunk_id, content_hash)
        )
    """)


# ─────────────────────────────────────────────────────────────────────────────
# Pass 1: parent_uuid
# ─────────────────────────────────────────────────────────────────────────────

def _pass_parent_uuid(conn: sqlite3.Connection, jsonl_files: list[Path],
                      dry_run: bool = False, limit: int = 0):
    """Backfill parent_uuid on existing _types_message rows."""
    print(f"\n[backfill] Pass 1: parent_uuid", file=sys.stderr)

    # Count current gaps
    total = conn.execute("SELECT COUNT(*) FROM _types_message").fetchone()[0]
    populated = conn.execute(
        "SELECT COUNT(*) FROM _types_message WHERE parent_uuid IS NOT NULL"
    ).fetchone()[0]
    print(f"  Current: {populated:,} / {total:,} populated", file=sys.stderr)

    if dry_run:
        # Estimate from a sample
        sample = jsonl_files[:min(50, len(jsonl_files))]
        has_parent = 0
        for path in sample:
            for _, entry in parse_jsonl_entries(path):
                if entry.get('parentUuid'):
                    has_parent += 1
        rate = has_parent / max(len(sample), 1)
        est = int(rate * len(jsonl_files))
        print(f"  Estimated entries with parentUuid: ~{est:,} (sampled {len(sample)} files)", file=sys.stderr)
        return

    files = jsonl_files[:limit] if limit else jsonl_files
    updated = skipped = no_chunk = sessions_done = 0

    for i, jsonl_path in enumerate(files):
        session_id = jsonl_path.stem
        entries = parse_jsonl_entries(jsonl_path)

        session_updated = 0
        for line_num, entry in entries:
            parent_uuid = entry.get('parentUuid')
            if not parent_uuid:
                skipped += 1
                continue

            chunk_id = f"{session_id}_{line_num}"

            # Try direct chunk_id match first
            row = conn.execute(
                "SELECT 1 FROM _types_message WHERE chunk_id = ? AND parent_uuid IS NULL",
                (chunk_id,)
            ).fetchone()

            if row:
                conn.execute(
                    "UPDATE _types_message SET parent_uuid = ? WHERE chunk_id = ? AND parent_uuid IS NULL",
                    (parent_uuid, chunk_id)
                )
                session_updated += 1
            else:
                # Try chunk_number match for ep-format / queue-format chunk_ids
                result = conn.execute("""
                    UPDATE _types_message SET parent_uuid = ?
                    WHERE chunk_id IN (
                        SELECT tm.chunk_id FROM _types_message tm
                        JOIN _edges_source es ON tm.chunk_id = es.chunk_id
                        WHERE es.source_id = ? AND tm.chunk_number = ? AND tm.parent_uuid IS NULL
                    )
                """, (parent_uuid, session_id, line_num))
                if result.rowcount > 0:
                    session_updated += result.rowcount
                else:
                    no_chunk += 1

        updated += session_updated
        sessions_done += 1

        if sessions_done % 100 == 0:
            conn.commit()
            print(f"  Progress: {sessions_done:,} / {len(files):,} sessions, {updated:,} updated",
                  file=sys.stderr)

    conn.commit()
    print(f"  Sessions: {sessions_done:,}", file=sys.stderr)
    print(f"  Updated: {updated:,}", file=sys.stderr)
    print(f"  No parent: {skipped:,}", file=sys.stderr)
    print(f"  No matching chunk: {no_chunk:,}", file=sys.stderr)


# ─────────────────────────────────────────────────────────────────────────────
# Pass 2: tool_content
# ─────────────────────────────────────────────────────────────────────────────

def _pass_tool_content(conn: sqlite3.Connection, jsonl_files: list[Path],
                       dry_run: bool = False, limit: int = 0):
    """Extract tool_use inputs + tool_result outputs to _raw_content."""
    print(f"\n[backfill] Pass 2: tool_content", file=sys.stderr)

    # Current state
    current = conn.execute("SELECT COUNT(*) FROM _raw_content").fetchone()[0]
    print(f"  Current _raw_content rows: {current:,}", file=sys.stderr)

    if dry_run:
        # Sample to estimate
        sample = jsonl_files[:min(50, len(jsonl_files))]
        tool_use_count = tool_result_count = 0
        for path in sample:
            for _, entry in parse_jsonl_entries(path):
                message = entry.get('message', {})
                content = message.get('content', [])
                if not isinstance(content, list):
                    continue
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    if item.get('type') == 'tool_use':
                        tool_use_count += 1
                    elif item.get('type') == 'tool_result':
                        tool_result_count += 1

        ratio = len(jsonl_files) / max(len(sample), 1)
        print(f"  Estimated tool_use items: ~{int(tool_use_count * ratio):,}", file=sys.stderr)
        print(f"  Estimated tool_result items: ~{int(tool_result_count * ratio):,}", file=sys.stderr)
        return

    _ensure_tables(conn)
    files = jsonl_files[:limit] if limit else jsonl_files
    tool_use_stored = tool_result_stored = sessions_done = 0
    bytes_total = 0

    for i, jsonl_path in enumerate(files):
        session_id = jsonl_path.stem
        entries = parse_jsonl_entries(jsonl_path)

        # Build tool_use_id -> tool_name map for this session
        tool_use_id_map = {}

        for line_num, entry in entries:
            message = entry.get('message', {})
            content = message.get('content', [])
            if not isinstance(content, list):
                continue

            # Extract timestamp
            ts_int = int(time.time())
            timestamp = entry.get('timestamp')
            if timestamp:
                try:
                    from datetime import datetime
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    ts_int = int(dt.timestamp())
                except Exception:
                    pass

            chunk_id = f"{session_id}_{line_num}"

            for item in content:
                if not isinstance(item, dict):
                    continue

                if item.get('type') == 'tool_use':
                    tool_name = item.get('name', 'unknown')
                    tool_use_id_map[item.get('id', '')] = tool_name

                    # Store input as JSON
                    raw = json.dumps(item.get('input', {}))
                    if len(raw) > 10:
                        _store_content(conn, chunk_id, raw, tool_name, ts_int)
                        tool_use_stored += 1
                        bytes_total += len(raw)

                elif item.get('type') == 'tool_result':
                    tool_use_id = item.get('tool_use_id', '')
                    tool_name = tool_use_id_map.get(tool_use_id, 'unknown')

                    raw = _normalize_tool_result_content(item.get('content'))
                    if raw and len(raw) > 10:
                        _store_content(conn, chunk_id, raw, tool_name, ts_int)
                        tool_result_stored += 1
                        bytes_total += len(raw)

        sessions_done += 1
        conn.commit()

        if sessions_done % 100 == 0:
            print(f"  Progress: {sessions_done:,} / {len(files):,} sessions, "
                  f"{tool_use_stored:,} inputs + {tool_result_stored:,} results, "
                  f"{bytes_total / (1024**3):.2f} GB",
                  file=sys.stderr)

    print(f"  Sessions: {sessions_done:,}", file=sys.stderr)
    print(f"  tool_use stored: {tool_use_stored:,}", file=sys.stderr)
    print(f"  tool_result stored: {tool_result_stored:,}", file=sys.stderr)
    print(f"  Total bytes: {bytes_total / (1024**3):.2f} GB", file=sys.stderr)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="JSONL content backfill — parent_uuid + tool content extraction"
    )
    parser.add_argument("--dry-run", action="store_true", help="Report gaps only")
    parser.add_argument("--limit", type=int, default=0, help="Limit sessions processed")
    parser.add_argument("--pass", dest="pass_name",
                        choices=["parent_uuid", "tool_content"],
                        help="Run specific pass only")
    args = parser.parse_args()

    cell_path = resolve_cell('claude_code')
    if not cell_path:
        print("[backfill] FATAL: claude_code cell not found", file=sys.stderr)
        sys.exit(1)

    conn = sqlite3.connect(str(cell_path), timeout=30.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")

    t0 = time.time()
    jsonl_files = find_all_jsonl()
    print(f"[backfill] JSONL Content Backfill — {cell_path}", file=sys.stderr)
    print(f"[backfill] Found {len(jsonl_files):,} JSONL files", file=sys.stderr)
    if args.dry_run:
        print("[backfill] DRY RUN — no writes", file=sys.stderr)

    passes = {
        'parent_uuid': _pass_parent_uuid,
        'tool_content': _pass_tool_content,
    }

    if args.pass_name:
        passes[args.pass_name](conn, jsonl_files, dry_run=args.dry_run, limit=args.limit)
    else:
        for fn in passes.values():
            fn(conn, jsonl_files, dry_run=args.dry_run, limit=args.limit)

    elapsed = time.time() - t0
    print(f"\n[backfill] Done in {elapsed:.1f}s", file=sys.stderr)
    conn.close()


if __name__ == "__main__":
    main()
