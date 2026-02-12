"""
Flexsearch Core — cell loading, SQL execution, view generation.

Three functions:
- open_cell()        → load .db, register FTS, return conn
- run_sql()          → execute SQL, return list[dict]
- regenerate_views() → discover tables, read renames from _meta, emit CREATE VIEW
"""

import sqlite3
from pathlib import Path
from typing import Optional


def open_cell(db_path: str) -> sqlite3.Connection:
    """
    Open a cell database with optimized settings.

    Args:
        db_path: Path to .db file

    Returns:
        SQLite connection with row_factory = sqlite3.Row
    """
    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row

    # Performance pragmas
    db.execute("PRAGMA synchronous=NORMAL")
    db.execute("PRAGMA cache_size=-20000")
    db.execute("PRAGMA temp_store=MEMORY")
    db.execute("PRAGMA journal_mode=WAL")

    return db


def run_sql(db: sqlite3.Connection, query: str,
            params: tuple = ()) -> list[dict]:
    """Execute SQL, return list of dicts. The AI writes the query."""
    rows = db.execute(query, params).fetchall()
    return [dict(r) for r in rows]


def get_meta(db: sqlite3.Connection, key: str) -> Optional[str]:
    """Read a single value from _meta table."""
    try:
        row = db.execute(
            "SELECT value FROM _meta WHERE key = ?", (key,)
        ).fetchone()
        return row[0] if row else None
    except sqlite3.OperationalError:
        return None


def set_meta(db: sqlite3.Connection, key: str, value: str):
    """Write a key-value pair to _meta table."""
    db.execute("""
        CREATE TABLE IF NOT EXISTS _meta (key TEXT PRIMARY KEY, value TEXT)
    """)
    db.execute(
        "INSERT OR REPLACE INTO _meta (key, value) VALUES (?, ?)",
        (key, value)
    )
    db.commit()


def regenerate_views(db: sqlite3.Connection):
    """
    Discover tables via sqlite_master, read renames from _meta,
    emit CREATE VIEW statements.

    Rules:
    - _edges_* and _enrich_* with PK on chunk_id → LEFT JOIN into view
    - _edges_* without PK on chunk_id → skip (1:N, AI JOINs manually)
    - Column renames from _meta WHERE key LIKE 'view:%'
    - COALESCE for graceful NULL handling
    """
    # Discover edge and enrichment tables
    edge_tables = _discover_tables(db, '_edges_%')
    enrich_tables = _discover_tables(db, '_enrich_%')

    # Read renames from _meta
    renames = _read_renames(db)

    # Get view definitions from _meta
    view_names = set()
    try:
        rows = db.execute(
            "SELECT DISTINCT substr(key, 6, instr(substr(key, 6), ':') - 1) "
            "FROM _meta WHERE key LIKE 'view:%'"
        ).fetchall()
        view_names = {r[0] for r in rows if r[0]}
    except sqlite3.OperationalError:
        pass

    # Default: build 'chunks' view if no views configured
    if not view_names:
        view_names = {'chunks'}

    for view_name in view_names:
        view_renames = renames.get(view_name, {})
        _build_view(db, view_name, edge_tables, enrich_tables, view_renames)


def _discover_tables(db: sqlite3.Connection, pattern: str) -> list[dict]:
    """Discover tables matching a LIKE pattern with their columns and PK info."""
    tables = []
    rows = db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE ?",
        (pattern,)
    ).fetchall()

    for row in rows:
        table_name = row[0]
        cols = db.execute(f"PRAGMA table_info([{table_name}])").fetchall()
        col_info = []
        has_chunk_id_pk = False
        for c in cols:
            col_info.append({
                'name': c[1],  # column name
                'type': c[2],  # data type
                'pk': bool(c[5]),  # is primary key
            })
            if c[1] == 'chunk_id' and c[5]:
                has_chunk_id_pk = True

        tables.append({
            'name': table_name,
            'columns': col_info,
            'has_chunk_id_pk': has_chunk_id_pk,
            'has_chunk_id': any(c['name'] == 'chunk_id' for c in col_info),
        })

    return tables


def _read_renames(db: sqlite3.Connection) -> dict[str, dict[str, str]]:
    """
    Read view column renames from _meta.

    Keys: view:{view_name}:rename:{raw_col} → domain_name
    Returns: {view_name: {raw_col: domain_name}}
    """
    renames = {}
    try:
        rows = db.execute(
            "SELECT key, value FROM _meta WHERE key LIKE 'view:%:rename:%'"
        ).fetchall()
        for row in rows:
            parts = row[0].split(':')
            if len(parts) == 4:
                _, view_name, _, raw_col = parts
                renames.setdefault(view_name, {})[raw_col] = row[1]
    except sqlite3.OperationalError:
        pass
    return renames


def _build_view(db: sqlite3.Connection, view_name: str,
                edge_tables: list[dict], enrich_tables: list[dict],
                renames: dict[str, str]):
    """Build and execute a CREATE VIEW statement."""
    # Start with _raw_chunks as base
    has_raw = db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='_raw_chunks'"
    ).fetchone()

    if not has_raw:
        return  # No raw chunks, nothing to compose

    # Collect columns and joins
    select_cols = []
    joins = []

    # Base table columns
    raw_cols = db.execute("PRAGMA table_info(_raw_chunks)").fetchall()
    for c in raw_cols:
        col_name = c[1]
        alias = renames.get(col_name, col_name)
        if col_name == alias:
            select_cols.append(f"r.[{col_name}]")
        else:
            select_cols.append(f"r.[{col_name}] AS [{alias}]")

    # Join eligible tables (1:1 on chunk_id)
    joinable = [t for t in edge_tables + enrich_tables
                if t['has_chunk_id'] and t['has_chunk_id_pk']]

    for i, table in enumerate(joinable):
        alias = f"t{i}"
        joins.append(
            f"LEFT JOIN [{table['name']}] {alias} ON r.id = {alias}.chunk_id"
        )
        for col in table['columns']:
            if col['name'] == 'chunk_id':
                continue  # Skip the FK itself
            col_name = col['name']
            domain_name = renames.get(col_name, col_name)
            if col_name == domain_name:
                select_cols.append(f"{alias}.[{col_name}]")
            else:
                select_cols.append(f"{alias}.[{col_name}] AS [{domain_name}]")

    # Build SQL
    select_str = ",\n    ".join(select_cols)
    join_str = "\n".join(joins)

    sql = f"""CREATE VIEW IF NOT EXISTS [{view_name}] AS
SELECT
    {select_str}
FROM _raw_chunks r
{join_str}"""

    # Drop and recreate
    db.execute(f"DROP VIEW IF EXISTS [{view_name}]")
    db.execute(sql)
    db.commit()
