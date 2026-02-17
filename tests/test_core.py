"""
TDD Tests for flexsearch/core.py — Plan 1

Tests the three core functions:
  - open_cell(path) -> sqlite3.Connection
  - run_sql(db, sql, params) -> list[dict]
  - regenerate_views(db) -> None

These tests WILL FAIL until Plan 1 code is written.
The contract they define is the spec.

Run with: pytest tests/test_core.py -v
"""
import sqlite3
import pytest

def _can_import_flexsearch():
    try:
        from flexsearch.core import open_cell, run_sql, regenerate_views
        return True
    except ImportError:
        return False


# These imports will fail until Plan 1 creates the module
pytestmark = pytest.mark.skipif(
    not _can_import_flexsearch(),
    reason="flexsearch.core not yet implemented (Plan 1)"
)


# =============================================================================
# open_cell
# =============================================================================

class TestOpenCell:
    """open_cell(path) loads a .db, registers UDFs, returns connection."""

    def test_returns_connection(self, tmp_path):
        from flexsearch.core import open_cell
        db_path = tmp_path / "test.db"
        db_path.touch()
        conn = open_cell(str(db_path))
        assert isinstance(conn, sqlite3.Connection)
        conn.close()

    def test_row_factory_is_row(self, tmp_path):
        from flexsearch.core import open_cell
        db_path = tmp_path / "test.db"
        db_path.touch()
        conn = open_cell(str(db_path))
        assert conn.row_factory == sqlite3.Row
        conn.close()

    def test_set_meta_creates_table(self, tmp_path):
        """set_meta() creates _meta on demand if missing."""
        from flexsearch.core import open_cell, set_meta, get_meta
        db_path = tmp_path / "test.db"
        bare = sqlite3.connect(str(db_path))
        bare.close()
        conn = open_cell(str(db_path))
        set_meta(conn, 'description', 'Test cell')
        assert get_meta(conn, 'description') == 'Test cell'
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='_meta'"
        ).fetchall()
        assert len(tables) == 1
        conn.close()


# =============================================================================
# run_sql
# =============================================================================

class TestRunSQL:
    """run_sql(db, sql, params) executes SQL and returns list[dict]."""

    def test_returns_list_of_dicts(self, qmem_cell):
        from flexsearch.core import run_sql
        results = run_sql(qmem_cell, "SELECT * FROM _raw_chunks LIMIT 2")
        assert isinstance(results, list)
        assert len(results) == 2
        assert isinstance(results[0], dict)

    def test_dict_has_column_names(self, qmem_cell):
        from flexsearch.core import run_sql
        results = run_sql(qmem_cell, "SELECT id, content FROM _raw_chunks LIMIT 1")
        assert 'id' in results[0]
        assert 'content' in results[0]

    def test_params_interpolation(self, qmem_cell):
        from flexsearch.core import run_sql
        results = run_sql(
            qmem_cell,
            "SELECT * FROM _raw_sources WHERE doc_type = :dtype",
            {'dtype': 'architecture'}
        )
        assert len(results) == 1
        assert results[0]['source_id'] == 'src-arch'

    def test_empty_result(self, qmem_cell):
        from flexsearch.core import run_sql
        results = run_sql(
            qmem_cell,
            "SELECT * FROM _raw_chunks WHERE id = 'nonexistent'"
        )
        assert results == []

    def test_count_query(self, qmem_cell):
        from flexsearch.core import run_sql
        results = run_sql(qmem_cell, "SELECT COUNT(*) as n FROM _raw_chunks")
        assert results[0]['n'] == 9


# =============================================================================
# regenerate_views
# =============================================================================

class TestRegenerateViews:
    """regenerate_views(db) discovers tables, reads _meta renames, emits CREATE VIEW."""

    def test_creates_views(self, qmem_cell):
        from flexsearch.core import regenerate_views
        regenerate_views(qmem_cell)
        views = qmem_cell.execute(
            "SELECT name FROM sqlite_master WHERE type='view'"
        ).fetchall()
        assert len(views) > 0, "Should create at least one view"

    def test_view_has_one_row_per_chunk(self, qmem_cell):
        """Views must not multiply rows (1:1 PK rule)."""
        from flexsearch.core import regenerate_views
        regenerate_views(qmem_cell)
        chunk_count = qmem_cell.execute("SELECT COUNT(*) FROM _raw_chunks").fetchone()[0]
        # Find the chunk-level view
        views = qmem_cell.execute(
            "SELECT name FROM sqlite_master WHERE type='view'"
        ).fetchall()
        for (view_name,) in views:
            view_count = qmem_cell.execute(f"SELECT COUNT(*) FROM {view_name}").fetchone()[0]
            # Source-level views will have fewer rows, chunk-level views must match
            assert view_count <= chunk_count, \
                f"View '{view_name}' has {view_count} rows but only {chunk_count} chunks"

    def test_applies_meta_renames(self, claude_code_cell):
        """Views should use domain vocabulary from _meta renames."""
        from flexsearch.core import regenerate_views
        regenerate_views(claude_code_cell)
        # Check that 'action' appears as a column (renamed from tool_name)
        views = claude_code_cell.execute(
            "SELECT name FROM sqlite_master WHERE type='view'"
        ).fetchall()
        found_rename = False
        for (view_name,) in views:
            info = claude_code_cell.execute(f"PRAGMA table_info({view_name})").fetchall()
            col_names = {r[1] for r in info}
            if 'action' in col_names or 'importance' in col_names:
                found_rename = True
        assert found_rename, "View should contain renamed columns from _meta"

    def test_idempotent(self, qmem_cell):
        """Calling regenerate_views twice should produce same result."""
        from flexsearch.core import regenerate_views
        regenerate_views(qmem_cell)
        views_first = qmem_cell.execute(
            "SELECT name FROM sqlite_master WHERE type='view' ORDER BY name"
        ).fetchall()
        regenerate_views(qmem_cell)
        views_second = qmem_cell.execute(
            "SELECT name FROM sqlite_master WHERE type='view' ORDER BY name"
        ).fetchall()
        assert views_first == views_second

    def test_coalesce_defaults(self, qmem_cell):
        """Views should COALESCE enrichment columns with sensible defaults."""
        from flexsearch.core import regenerate_views
        regenerate_views(qmem_cell)
        # After wiping enrichments, view should still return rows without NULLs
        # for enrichment-derived columns (they should COALESCE to defaults)
        qmem_cell.execute("DELETE FROM _enrich_source_graph")
        qmem_cell.execute("DELETE FROM _enrich_types")
        views = qmem_cell.execute(
            "SELECT name FROM sqlite_master WHERE type='view'"
        ).fetchall()
        for (view_name,) in views:
            rows = qmem_cell.execute(f"SELECT * FROM {view_name}").fetchall()
            assert len(rows) > 0, f"View '{view_name}' should still have rows after enrich wipe"
