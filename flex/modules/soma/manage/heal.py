#!/usr/bin/env python3
"""
SOMA Identity Heal — Backfill missing identity edges.

Port of Thread's backfill-identity.py + backfill-content.py + backfill-url-uuid.py,
retargeted from flat columns to Flex edge tables.

Three passes:
  Pass 1: file_uuid + repo_root (from _edges_tool_ops with file ops)
  Pass 2: content_hash (from _edges_tool_ops with mutations)
  Pass 3: url_uuid (from _edges_tool_ops with WebFetch)

Usage:
  python -m flex.modules.soma.manage.heal              # run all passes
  python -m flex.modules.soma.manage.heal --dry-run     # report gaps only
  python -m flex.modules.soma.manage.heal --limit 100   # test subset
  python -m flex.modules.soma.manage.heal --pass file   # file_uuid only
  python -m flex.modules.soma.manage.heal --pass content
  python -m flex.modules.soma.manage.heal --pass url
"""

import re
import sys
import time
import argparse
import sqlite3
from pathlib import Path

from flex.registry import resolve_cell
from flex.modules.soma.compile import ensure_tables


# ─────────────────────────────────────────────────────────────────────────────
# Gap queries
# ─────────────────────────────────────────────────────────────────────────────

FILE_GAP_SQL = """
    SELECT t.chunk_id, t.target_file
    FROM _edges_tool_ops t
    LEFT JOIN _edges_file_identity fi ON t.chunk_id = fi.chunk_id
    WHERE fi.chunk_id IS NULL
      AND t.target_file IS NOT NULL
      AND t.target_file NOT LIKE '/tmp/%'
      AND t.target_file NOT LIKE '/var/tmp/%'
      AND t.tool_name IN ('Write','Edit','MultiEdit','Read','Glob','Grep')
"""

REPO_GAP_SQL = """
    SELECT t.chunk_id, t.target_file
    FROM _edges_tool_ops t
    LEFT JOIN _edges_repo_identity ri ON t.chunk_id = ri.chunk_id
    WHERE ri.chunk_id IS NULL
      AND t.target_file IS NOT NULL
      AND t.target_file NOT LIKE '/tmp/%'
      AND t.tool_name IN ('Write','Edit','MultiEdit','Read','Glob','Grep','Bash')
"""

CONTENT_GAP_SQL = """
    SELECT t.chunk_id, t.target_file
    FROM _edges_tool_ops t
    LEFT JOIN _edges_content_identity ci ON t.chunk_id = ci.chunk_id
    WHERE ci.chunk_id IS NULL
      AND t.target_file IS NOT NULL
      AND t.tool_name IN ('Write','Edit','MultiEdit')
"""

URL_GAP_SQL = """
    SELECT c.id as chunk_id, c.content
    FROM _raw_chunks c
    JOIN _edges_tool_ops t ON c.id = t.chunk_id
    LEFT JOIN _edges_url_identity ui ON c.id = ui.chunk_id
    WHERE ui.chunk_id IS NULL
      AND t.tool_name = 'WebFetch'
"""


# ─────────────────────────────────────────────────────────────────────────────
# Passes
# ─────────────────────────────────────────────────────────────────────────────

def _pass_file(conn, dry_run=False, limit=0):
    """Pass 1: file_uuid + repo_root backfill."""
    try:
        from soma.identity.file_identity import FileIdentity
        from soma.identity.repo_identity import RepoIdentity
        file_id = FileIdentity()
        repo_id = RepoIdentity()
    except ImportError:
        print("[heal] SOMA identity not available — skipping file pass", file=sys.stderr)
        return

    # file_uuid
    sql = FILE_GAP_SQL + (f" LIMIT {limit}" if limit else "")
    gaps = conn.execute(sql).fetchall()
    print(f"\n[heal] Pass 1a: file_uuid", file=sys.stderr)
    print(f"  Gaps: {len(gaps)}", file=sys.stderr)

    if dry_run:
        return

    resolved = unresolvable = 0
    for chunk_id, target_file in gaps:
        try:
            file_uuid = file_id.assign(target_file)
            if file_uuid:
                conn.execute(
                    "INSERT OR IGNORE INTO _edges_file_identity (chunk_id, file_uuid) VALUES (?, ?)",
                    (chunk_id, file_uuid)
                )
                resolved += 1
            else:
                unresolvable += 1
        except Exception:
            unresolvable += 1

    conn.commit()
    print(f"  Resolved: {resolved}", file=sys.stderr)
    print(f"  Unresolvable: {unresolvable}", file=sys.stderr)

    # repo_root
    sql = REPO_GAP_SQL + (f" LIMIT {limit}" if limit else "")
    gaps = conn.execute(sql).fetchall()
    print(f"\n[heal] Pass 1b: repo_root", file=sys.stderr)
    print(f"  Gaps: {len(gaps)}", file=sys.stderr)

    if not gaps:
        return

    resolved = unresolvable = 0
    for chunk_id, target_file in gaps:
        try:
            result = repo_id.resolve_file(target_file)
            if result:
                _, repo = result
                if repo.root_commit:
                    conn.execute(
                        "INSERT OR IGNORE INTO _edges_repo_identity (chunk_id, repo_root, is_tracked) VALUES (?, ?, 1)",
                        (chunk_id, repo.root_commit)
                    )
                    resolved += 1
                else:
                    unresolvable += 1
            else:
                unresolvable += 1
        except Exception:
            unresolvable += 1

    conn.commit()
    print(f"  Resolved: {resolved}", file=sys.stderr)
    print(f"  Unresolvable: {unresolvable}", file=sys.stderr)


def _pass_content(conn, dry_run=False, limit=0):
    """Pass 2: content_hash backfill."""
    try:
        from soma.identity.content_identity import ContentIdentity
        content_id = ContentIdentity()
    except ImportError:
        print("[heal] SOMA ContentIdentity not available — skipping content pass", file=sys.stderr)
        return

    sql = CONTENT_GAP_SQL + (f" LIMIT {limit}" if limit else "")
    gaps = conn.execute(sql).fetchall()
    print(f"\n[heal] Pass 2: content_hash", file=sys.stderr)
    print(f"  Gaps: {len(gaps)}", file=sys.stderr)

    if dry_run:
        return

    resolved = unresolvable = 0
    for chunk_id, target_file in gaps:
        try:
            path = Path(target_file)
            if not path.is_file():
                unresolvable += 1
                continue
            content_hash = content_id.store(path.read_bytes())
            if content_hash:
                conn.execute(
                    "INSERT OR IGNORE INTO _edges_content_identity (chunk_id, content_hash) VALUES (?, ?)",
                    (chunk_id, content_hash)
                )
                resolved += 1
            else:
                unresolvable += 1
        except Exception:
            unresolvable += 1

    conn.commit()
    print(f"  Resolved: {resolved}", file=sys.stderr)
    print(f"  Unresolvable: {unresolvable} (file no longer exists)", file=sys.stderr)


def _pass_url(conn, dry_run=False, limit=0):
    """Pass 3: url_uuid backfill."""
    try:
        from soma.identity.url_identity import URLIdentity
        url_id = URLIdentity()
    except ImportError:
        print("[heal] SOMA URLIdentity not available — skipping url pass", file=sys.stderr)
        return

    sql = URL_GAP_SQL + (f" LIMIT {limit}" if limit else "")
    gaps = conn.execute(sql).fetchall()
    print(f"\n[heal] Pass 3: url_uuid", file=sys.stderr)
    print(f"  Gaps: {len(gaps)}", file=sys.stderr)

    if dry_run:
        return

    resolved = unresolvable = 0
    for chunk_id, content in gaps:
        try:
            match = re.search(r'https?://[^\s]+', content or '')
            if not match:
                unresolvable += 1
                continue
            url = match.group(0)
            url_uuid = url_id.assign(url)
            if url_uuid:
                conn.execute(
                    "INSERT OR IGNORE INTO _edges_url_identity (chunk_id, url_uuid) VALUES (?, ?)",
                    (chunk_id, url_uuid)
                )
                resolved += 1
            else:
                unresolvable += 1
        except Exception:
            unresolvable += 1

    conn.commit()
    print(f"  Resolved: {resolved}", file=sys.stderr)
    print(f"  Unresolvable: {unresolvable} (no URL in content)", file=sys.stderr)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SOMA identity heal — backfill missing edges")
    parser.add_argument("--dry-run", action="store_true", help="Report gaps only")
    parser.add_argument("--limit", type=int, default=0, help="Limit per pass")
    parser.add_argument("--pass", dest="pass_name", choices=["file", "content", "url"],
                        help="Run specific pass only")
    args = parser.parse_args()

    cell_path = resolve_cell('claude_code')
    if not cell_path:
        print("[heal] FATAL: claude_code cell not found", file=sys.stderr)
        sys.exit(1)

    conn = sqlite3.connect(str(cell_path), timeout=30.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    ensure_tables(conn)

    t0 = time.time()
    print(f"[heal] SOMA Identity Heal — {cell_path}", file=sys.stderr)
    if args.dry_run:
        print("[heal] DRY RUN — no writes", file=sys.stderr)

    passes = {
        'file': _pass_file,
        'content': _pass_content,
        'url': _pass_url,
    }

    if args.pass_name:
        passes[args.pass_name](conn, dry_run=args.dry_run, limit=args.limit)
    else:
        for fn in passes.values():
            fn(conn, dry_run=args.dry_run, limit=args.limit)

    elapsed = time.time() - t0
    print(f"\n[heal] Done in {elapsed:.1f}s", file=sys.stderr)
    conn.close()


def heal(conn):
    """Run all heal passes on an open connection. For use by the worker daemon."""
    t0 = time.time()
    print("[heal] SOMA identity heal...", file=sys.stderr)
    for fn in (_pass_file, _pass_content, _pass_url):
        fn(conn, dry_run=False, limit=0)
    print(f"[heal] Done in {time.time() - t0:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
