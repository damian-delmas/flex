"""
test_init_schema.py — Schema completeness tests for flex init pipeline.

Validates that after bootstrap + enrichment stubs are applied, every table
that views, presets, or enrichment chains reference actually exists.

No embedding, no sessions, no ONNX model required.
Runs in < 2 seconds.

Run with: pytest tests/test_init_schema.py -v
"""
import os
import sqlite3
import pytest
from pathlib import Path

pytestmark = [pytest.mark.unit]


# All tables that must exist after bootstrap + stubs, before any enrichment.
# Divided by source so failures are easy to diagnose.
BOOTSTRAP_TABLES = [
    # _ensure_core_tables
    "_raw_chunks",
    "_raw_sources",
    "_edges_source",
    "_edges_tool_ops",
    "_types_message",
    "_edges_delegations",
    "_edges_soft_ops",
    "_meta",
    "_presets",
    "chunks_fts",
    # _ensure_content_tables
    "_raw_content",
    "_edges_raw_content",
    "content_fts",
]

STUB_TABLES = [
    "_enrich_source_graph",
    "_types_source_warmup",
    "_enrich_session_summary",
    "_enrich_repo_identity",
    "_enrich_file_graph",
    "_enrich_delegation_graph",
    "_ops",
    "_views",
]

REQUIRED_TABLES = BOOTSTRAP_TABLES + STUB_TABLES

# Queries that crashed before stubs were added. Must execute without error
# on an empty (post-bootstrap) cell.
CRASH_QUERIES = [
    "SELECT * FROM _enrich_source_graph LIMIT 1",
    "SELECT * FROM _enrich_session_summary LIMIT 1",
    "SELECT * FROM _enrich_repo_identity LIMIT 1",
    "SELECT * FROM _enrich_file_graph LIMIT 1",
    "SELECT * FROM _enrich_delegation_graph LIMIT 1",
    "SELECT * FROM _types_source_warmup LIMIT 1",
    "SELECT * FROM _ops LIMIT 1",
    "SELECT * FROM _views LIMIT 1",
    # JOIN pattern used by sessions view
    "SELECT s.source_id, g.centrality FROM _raw_sources s "
    "LEFT JOIN _enrich_source_graph g ON s.source_id = g.source_id LIMIT 1",
    # health preset pattern
    "SELECT COUNT(*) FROM _ops WHERE operation = 'build_similarity_graph'",
    # community_labels pattern
    "SELECT repo_path FROM _enrich_repo_identity WHERE repo_path IS NOT NULL",
]


@pytest.fixture
def init_cell(tmp_path, monkeypatch):
    """
    Bootstrap a real claude_code cell into tmp_path, apply enrichment stubs.
    Returns open sqlite3 connection.
    """
    monkeypatch.setenv("FLEX_HOME", str(tmp_path))
    # Reload registry so it picks up the new FLEX_HOME
    import flex.registry as registry_mod
    import importlib
    importlib.reload(registry_mod)

    from flex.modules.claude_code.compile.worker import bootstrap_claude_code_cell
    from flex.cli import _ENRICHMENT_STUBS

    cell_path = bootstrap_claude_code_cell()
    conn = sqlite3.connect(str(cell_path), timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")

    for ddl in _ENRICHMENT_STUBS.get("claude-code", []):
        conn.execute(ddl)
    conn.commit()

    yield conn
    conn.close()


class TestBootstrapTables:
    """Every table created by bootstrap_claude_code_cell must exist."""

    def test_bootstrap_tables_exist(self, init_cell):
        existing = {
            r[0]
            for r in init_cell.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
            ).fetchall()
        }
        missing = [t for t in BOOTSTRAP_TABLES if t not in existing]
        assert not missing, f"Missing bootstrap tables: {missing}"

    def test_meta_cell_type(self, init_cell):
        row = init_cell.execute(
            "SELECT value FROM _meta WHERE key = 'cell_type'"
        ).fetchone()
        assert row is not None
        assert row[0] == "claude-code"

    def test_meta_description(self, init_cell):
        row = init_cell.execute(
            "SELECT value FROM _meta WHERE key = 'description'"
        ).fetchone()
        assert row is not None
        assert len(row[0]) > 10


class TestStubTables:
    """Every enrichment stub table must exist and be queryable after init."""

    def test_stub_tables_exist(self, init_cell):
        existing = {
            r[0]
            for r in init_cell.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        missing = [t for t in STUB_TABLES if t not in existing]
        assert not missing, f"Missing stub tables: {missing}"

    def test_stub_tables_are_empty(self, init_cell):
        """Stubs must exist but be empty — enrichment hasn't run."""
        for table in STUB_TABLES:
            count = init_cell.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            assert count == 0, f"{table} should be empty stub, has {count} rows"

    @pytest.mark.parametrize("query", CRASH_QUERIES)
    def test_crash_query_executes(self, init_cell, query):
        """Queries that crashed before stubs were added must not raise."""
        try:
            init_cell.execute(query).fetchall()
        except sqlite3.OperationalError as e:
            pytest.fail(f"Query raised OperationalError: {e}\nQuery: {query}")


class TestAllRequiredTables:
    """Full surface — every table required by the pipeline must exist."""

    def test_all_required_tables(self, init_cell):
        existing = {
            r[0]
            for r in init_cell.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
            ).fetchall()
        }
        missing = [t for t in REQUIRED_TABLES if t not in existing]
        assert not missing, (
            f"Tables missing after bootstrap + stubs:\n"
            + "\n".join(f"  - {t}" for t in missing)
        )

    def test_stub_tables_parity_with_enrichment_stubs_constant(self, init_cell):
        """
        _ENRICHMENT_STUBS must cover all tables referenced by presets/views.
        If a new enrichment table is added, it must also be added to stubs.
        """
        from flex.cli import _ENRICHMENT_STUBS
        # Extract table names from stub DDL
        import re
        stub_ddl = " ".join(_ENRICHMENT_STUBS.get("claude-code", []))
        stub_table_names = set(
            re.findall(r"CREATE TABLE IF NOT EXISTS (\w+)", stub_ddl)
        )
        for t in STUB_TABLES:
            assert t in stub_table_names, (
                f"'{t}' is in STUB_TABLES test list but missing from "
                f"_ENRICHMENT_STUBS in cli.py — add it to both."
            )
