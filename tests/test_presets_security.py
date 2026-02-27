"""
Security and edge case tests for PresetLoader.

Covers: SQL injection via _interpolate, empty presets, caching, error handling.

Run with: pytest tests/test_presets_security.py -v
"""
import sqlite3
import pytest
from pathlib import Path


def _can_import():
    try:
        from flex.retrieve.presets import PresetLoader
        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(
    not _can_import(),
    reason="flex.retrieve.presets not yet implemented"
)


@pytest.fixture
def secure_db():
    """DB with _presets table and a data table to test injection against."""
    conn = sqlite3.connect(':memory:')
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE _raw_chunks (id TEXT PRIMARY KEY, content TEXT)")
    conn.execute("INSERT INTO _raw_chunks VALUES ('c1', 'hello world')")
    conn.execute("INSERT INTO _raw_chunks VALUES ('c2', 'foo bar')")
    conn.execute("CREATE TABLE _presets (name TEXT PRIMARY KEY, description TEXT, params TEXT DEFAULT '', sql TEXT)")
    conn.execute("INSERT INTO _presets (name, description, params, sql) VALUES (?, ?, ?, ?)", (
        'search', 'Search chunks by content', 'term (required)',
        "SELECT id, content FROM _raw_chunks WHERE content LIKE :term"))
    conn.commit()
    return conn


class TestSQLInjection:
    """Verify _interpolate escapes dangerous param values."""

    def test_single_quote_escaped(self, secure_db):
        from flex.retrieve.presets import PresetLoader
        loader = PresetLoader(secure_db)
        # This should NOT cause an error — quotes should be escaped
        results = loader.execute(secure_db, 'search', {'term': "it's"})
        assert isinstance(results, list)

    def test_injection_attempt_does_not_destroy_data(self, secure_db):
        from flex.retrieve.presets import PresetLoader
        loader = PresetLoader(secure_db)
        # Classic injection attempt
        try:
            loader.execute(secure_db, 'search',
                           {'term': "'; DROP TABLE _raw_chunks; --"})
        except sqlite3.OperationalError:
            pass  # Expected — the SQL is malformed after injection
        # Table must still exist with data
        count = secure_db.execute(
            "SELECT COUNT(*) FROM _raw_chunks"
        ).fetchone()[0]
        assert count == 2, "Data should survive injection attempt"


class TestPresetEdgeCases:
    """Edge cases: empty presets, caching, missing table."""

    def test_empty_sql_text(self, secure_db):
        from flex.retrieve.presets import PresetLoader
        secure_db.execute("INSERT OR REPLACE INTO _presets (name, description, params, sql) VALUES ('empty', '', '', '')")
        secure_db.commit()
        loader = PresetLoader(secure_db)
        preset = loader.load('empty')
        assert preset['queries'] == []

    def test_cache_returns_same_object(self, secure_db):
        from flex.retrieve.presets import PresetLoader
        loader = PresetLoader(secure_db)
        p1 = loader.load('search')
        p2 = loader.load('search')
        assert p1 is p2  # same cached object

    def test_list_presets_no_table(self):
        from flex.retrieve.presets import PresetLoader
        conn = sqlite3.connect(':memory:')
        conn.row_factory = sqlite3.Row
        loader = PresetLoader(conn)
        assert loader.list_presets() == []
        conn.close()

    def test_multi_query_error_captured(self, secure_db):
        from flex.retrieve.presets import PresetLoader
        secure_db.execute("INSERT OR REPLACE INTO _presets (name, description, params, sql) VALUES (?, ?, ?, ?)", (
            'bad-multi', 'Multi with bad query', '',
            "-- @multi: true\n-- @query: good\nSELECT COUNT(*) as n FROM _raw_chunks;\n-- @query: bad\nSELECT * FROM nonexistent_table;"))
        secure_db.commit()
        loader = PresetLoader(secure_db)
        results = loader.execute(secure_db, 'bad-multi')
        good = next(r for r in results if r['query'] == 'good')
        bad = next(r for r in results if r['query'] == 'bad')
        assert 'results' in good
        assert 'error' in bad
