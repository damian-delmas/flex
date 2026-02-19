#!/usr/bin/env python3
"""Migrate cells to ~/.flex/cells/{uuid}.db with full registry.

Phase 1: Re-register existing cells (assigns UUIDs, adds corpus_path)
Phase 2: Copy .db files to ~/.flex/cells/{uuid}.db
Phase 3: Move history JSONL alongside
Phase 4: Update registry paths
Phase 5: Create SQLite queue at ~/.flex/queue.db

Idempotent: safe to run multiple times. Skips cells already at ~/.flex/cells/.
"""

import shutil
import sqlite3
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flex.registry import (
    CELLS_ROOT, FLEX_HOME, REGISTRY_DB,
    register_cell, list_cells, _open_registry,
)

CELLS_DIR = FLEX_HOME / "cells"


def _detect_corpus_path(db_path: str) -> str | None:
    """Auto-detect corpus_path from a docpac cell's source_path column."""
    try:
        conn = sqlite3.connect(db_path, timeout=5)
        # Check if it's a docpac cell
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        if '_types_docpac' not in tables:
            conn.close()
            return None

        # Get a source_path and walk up to context/ root
        row = conn.execute(
            "SELECT source_path FROM _raw_sources WHERE source_path IS NOT NULL LIMIT 1"
        ).fetchone()
        conn.close()

        if not row or not row[0]:
            return None

        # Walk up from source_path to find context/ root
        p = Path(row[0])
        for parent in p.parents:
            if parent.name == 'context':
                return str(parent)
            if parent.name in ('docs', 'documentation'):
                return str(parent)
        return None
    except Exception:
        return None


def phase1_register():
    """Re-register cells from legacy dir. Assigns UUIDs and corpus_path."""
    if not CELLS_ROOT.exists():
        print(f"CELLS_ROOT not found: {CELLS_ROOT}")
        return []

    cells = sorted(
        d for d in CELLS_ROOT.iterdir()
        if d.is_dir() and (d / "main.db").exists()
    )

    print(f"Phase 1: Found {len(cells)} cells at {CELLS_ROOT}")
    registered = []
    for cell_dir in cells:
        name = cell_dir.name
        db_path = cell_dir / "main.db"
        corpus_path = _detect_corpus_path(str(db_path))
        cell_id = register_cell(name, db_path, corpus_path=corpus_path)
        print(f"  {name:<25} id={cell_id[:8]}... corpus={corpus_path or 'NULL'}")
        registered.append((name, cell_id, str(db_path), corpus_path))
    return registered


def phase2_move_cells():
    """Copy .db files to ~/.flex/cells/{uuid}.db."""
    CELLS_DIR.mkdir(parents=True, exist_ok=True)

    cells = list_cells()
    print(f"\nPhase 2: Moving {len(cells)} cells to {CELLS_DIR}")

    for cell in cells:
        old_path = Path(cell['path'])
        if not old_path.exists():
            print(f"  SKIP {cell['name']}: source not found at {old_path}")
            continue

        # Skip if already at target location
        if str(old_path).startswith(str(CELLS_DIR)):
            print(f"  SKIP {cell['name']}: already at {old_path}")
            continue

        cell_id = cell['id']
        if not cell_id:
            cell_id = str(uuid.uuid4())

        new_path = CELLS_DIR / f"{cell_id}.db"
        print(f"  {cell['name']}: {old_path} -> {new_path}")
        shutil.copy2(str(old_path), str(new_path))

        # Move history JSONL if it exists
        old_history = old_path.parent / "flex-history.jsonl"
        if old_history.exists():
            new_history = CELLS_DIR / f"{cell_id}-history.jsonl"
            shutil.copy2(str(old_history), str(new_history))
            print(f"    history: {old_history.name} -> {new_history.name}")


def phase3_update_paths():
    """Update registry paths to point to new locations."""
    db = _open_registry()
    cells = [dict(r) for r in db.execute(
        "SELECT id, name, path FROM cells"
    ).fetchall()]

    print(f"\nPhase 3: Updating {len(cells)} registry paths")
    for cell in cells:
        old_path = Path(cell['path'])
        if str(old_path).startswith(str(CELLS_DIR)):
            continue  # already migrated

        cell_id = cell['id']
        new_path = CELLS_DIR / f"{cell_id}.db"
        if new_path.exists():
            db.execute(
                "UPDATE cells SET path = ? WHERE name = ?",
                (str(new_path), cell['name'])
            )
            print(f"  {cell['name']}: -> {new_path}")
        else:
            print(f"  SKIP {cell['name']}: {new_path} not found (run phase2 first)")

    db.commit()
    db.close()


def phase4_create_queue():
    """Create SQLite queue at ~/.flex/queue.db."""
    queue_path = FLEX_HOME / "queue.db"
    print(f"\nPhase 4: Creating queue at {queue_path}")

    conn = sqlite3.connect(str(queue_path), timeout=5)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS pending (
            path TEXT PRIMARY KEY,
            ts   INTEGER NOT NULL
        )
    """)
    conn.commit()
    conn.close()
    print("  Queue ready.")


def phase5_verify():
    """Verify all cells accessible via registry."""
    from flex.registry import resolve_cell

    print("\nPhase 5: Verification")
    cells = list_cells()
    ok = 0
    for cell in cells:
        result = resolve_cell(cell['name'])
        if result and result.exists():
            size_mb = result.stat().st_size / (1024 * 1024)
            print(f"  OK  {cell['name']:<25} {size_mb:>7.1f}MB  {cell['cell_type'] or '':<12} corpus={cell.get('corpus_path') or 'NULL'}")
            ok += 1
        else:
            print(f"  FAIL {cell['name']}: resolve returned {result}")

    print(f"\n{ok}/{len(cells)} cells verified.")
    return ok == len(cells)


def main():
    print(f"Registry: {REGISTRY_DB}")
    print(f"Target:   {CELLS_DIR}\n")

    phase1_register()
    phase2_move_cells()
    phase3_update_paths()
    phase4_create_queue()
    success = phase5_verify()

    if success:
        print("\nMigration complete. Safe to remove ~/.qmem/cells/projects/ after verifying MCP server.")
    else:
        print("\nMigration incomplete — check failures above.")
        sys.exit(1)


if __name__ == '__main__':
    main()
