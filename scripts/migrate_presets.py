#!/usr/bin/env python3
"""
One-shot migration: bake .sql preset files into _presets table on all production cells.

Usage:
    python scripts/migrate_presets.py

Creates _presets table if missing, INSERTs general presets into all cells,
INSERTs thread-specific presets into thread/claude cells.
"""
import sqlite3
import sys
from pathlib import Path

# Preset source directories
PRESET_ROOT = Path(__file__).resolve().parent.parent / "flexsearch" / "retrieve" / "presets"
GENERAL_DIR = PRESET_ROOT / "general"
THREAD_DIR = PRESET_ROOT / "thread"

# Cell paths
CELLS_ROOT = Path.home() / ".qmem/cells/projects"

# Which cells get which presets
CELL_CONFIG = {
    'thread': [GENERAL_DIR, THREAD_DIR],
    'claude': [GENERAL_DIR, THREAD_DIR],  # claude has messages/sessions views too
    'qmem': [GENERAL_DIR],
    'inventory': [GENERAL_DIR],
    'thread-codebase': [GENERAL_DIR],
}


def migrate_cell(cell_name: str, preset_dirs: list[Path]):
    db_path = CELLS_ROOT / cell_name / "main.db"
    if not db_path.exists():
        print(f"  {cell_name}: SKIP (not found at {db_path})")
        return

    try:
        conn = sqlite3.connect(str(db_path), timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")

        # Create table if missing, add params column if needed
        conn.execute("""
            CREATE TABLE IF NOT EXISTS _presets (
                name TEXT PRIMARY KEY,
                description TEXT,
                params TEXT DEFAULT '',
                sql TEXT
            )
        """)
        # Migration: add params column to existing tables
        try:
            conn.execute("ALTER TABLE _presets ADD COLUMN params TEXT DEFAULT ''")
        except sqlite3.OperationalError:
            pass  # column already exists

        # Install presets from each dir
        from flexsearch.retrieve.presets import install_presets
        for pd in preset_dirs:
            if pd.exists():
                install_presets(conn, pd)

        # Report
        count = conn.execute("SELECT COUNT(*) FROM _presets").fetchone()[0]
        names = [r[0] for r in conn.execute("SELECT name FROM _presets ORDER BY name").fetchall()]
        print(f"  {cell_name}: {count} presets [{', '.join(names)}]")

        conn.close()
    except sqlite3.OperationalError as e:
        print(f"  {cell_name}: LOCKED ({e}) — retry after stopping thread-worker")


def main():
    print("Migrating presets to _presets table...")
    for cell_name, dirs in CELL_CONFIG.items():
        migrate_cell(cell_name, dirs)
    print("Done.")


if __name__ == "__main__":
    main()
