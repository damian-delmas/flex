"""Tests for the keyword() FTS5 materializer."""

import json
import sqlite3

import pytest

try:
    from flex.retrieve.keyword import materialize_keyword
    _SKIP = False
except ImportError:
    _SKIP = True

pytestmark = pytest.mark.skipif(_SKIP, reason="flex not importable")


@pytest.fixture
def db():
    """In-memory DB with minimal FTS5 setup."""
    conn = sqlite3.connect(':memory:')
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE _raw_chunks (id TEXT PRIMARY KEY, content TEXT)")
    conn.execute("""
        CREATE VIRTUAL TABLE chunks_fts USING fts5(
            content, content='_raw_chunks', content_rowid='rowid'
        )
    """)
    # Insert test data
    test_data = [
        ('chunk_0', 'authentication security bug fix session zero'),
        ('chunk_1', 'authentication middleware handler for login'),
        ('chunk_2', 'database migration script for users table'),
        ('chunk_3', 'security audit report findings summary'),
        ('chunk_4', 'vector search implementation notes'),
    ]
    for cid, content in test_data:
        conn.execute("INSERT INTO _raw_chunks VALUES (?, ?)", (cid, content))
    conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
    conn.commit()
    return conn


class TestBasicRewrite:
    def test_basic_rewrite(self, db):
        sql = "SELECT k.id, k.rank FROM keyword('authentication') k LIMIT 5"
        result = materialize_keyword(db, sql)
        assert 'keyword' not in result
        assert '_kw_results_' in result
        rows = db.execute(result).fetchall()
        assert len(rows) > 0

    def test_rewrite_preserves_join(self, db):
        sql = "SELECT k.id, k.rank FROM keyword('security') k ORDER BY k.rank DESC LIMIT 3"
        result = materialize_keyword(db, sql)
        assert '_kw_results_' in result
        rows = db.execute(result).fetchall()
        assert len(rows) > 0
        # Rank should be positive (negated BM25)
        assert rows[0]['rank'] > 0

    def test_rank_positive_higher_better(self, db):
        sql = "SELECT k.id, k.rank FROM keyword('authentication') k ORDER BY k.rank DESC"
        result = materialize_keyword(db, sql)
        rows = db.execute(result).fetchall()
        for row in rows:
            assert row['rank'] > 0

    def test_snippet_present(self, db):
        sql = "SELECT k.id, k.snippet FROM keyword('authentication') k LIMIT 1"
        result = materialize_keyword(db, sql)
        rows = db.execute(result).fetchall()
        assert len(rows) > 0
        assert rows[0]['snippet'] is not None


class TestPassthrough:
    def test_no_keyword_passthrough(self, db):
        sql = "SELECT * FROM _raw_chunks LIMIT 5"
        assert materialize_keyword(db, sql) == sql

    def test_scalar_position_passthrough(self, db):
        """keyword in SELECT position (not FROM/JOIN) should pass through."""
        sql = "SELECT 'keyword' as label FROM _raw_chunks LIMIT 1"
        assert materialize_keyword(db, sql) == sql


class TestFromJoinGuard:
    def test_from_position(self, db):
        sql = "SELECT k.id FROM keyword('auth') k"
        result = materialize_keyword(db, sql)
        assert '_kw_results_' in result

    def test_join_position(self, db):
        sql = ("SELECT k.id FROM _raw_chunks c "
               "JOIN keyword('auth') k ON c.id = k.id")
        result = materialize_keyword(db, sql)
        assert '_kw_results_' in result

    def test_scalar_blocked(self, db):
        """SELECT keyword(...) should NOT rewrite — not a table source."""
        sql = "SELECT keyword('test') as x"
        result = materialize_keyword(db, sql)
        # Should pass through unchanged (no rewrite for scalar position)
        assert result == sql


class TestEmptyInput:
    def test_empty_string_error(self, db):
        sql = "SELECT * FROM keyword('') k"
        result = materialize_keyword(db, sql)
        parsed = json.loads(result)
        assert 'error' in parsed

    def test_whitespace_only_error(self, db):
        sql = "SELECT * FROM keyword('   ') k"
        result = materialize_keyword(db, sql)
        parsed = json.loads(result)
        assert 'error' in parsed


class TestParenBalancing:
    def test_nested_parens(self, db):
        sql = "SELECT k.id FROM keyword('term (with parens)') k"
        result = materialize_keyword(db, sql)
        assert '_kw_results_' in result

    def test_escaped_quotes(self, db):
        sql = "SELECT k.id FROM keyword('it''s a test') k"
        result = materialize_keyword(db, sql)
        assert '_kw_results_' in result

    def test_unmatched_parens_passthrough(self, db):
        sql = "SELECT k.id FROM keyword('test'"
        result = materialize_keyword(db, sql)
        assert result == sql  # unmatched — pass through


class TestEmptyResults:
    def test_no_matches_returns_empty(self, db):
        sql = "SELECT k.id FROM keyword('xyznonexistentterm999') k"
        result = materialize_keyword(db, sql)
        assert '_kw_results_' in result  # temp table created
        rows = db.execute(result).fetchall()
        assert len(rows) == 0


class TestSpecialChars:
    def test_cpp_fallback(self, db):
        """FTS5 parse error on C++ → fallback to quoted."""
        sql = "SELECT k.id FROM keyword('C++') k"
        result = materialize_keyword(db, sql)
        # Should not be an error — fallback handles it
        assert '"error"' not in result
        assert '_kw_results_' in result

    def test_dots_fallback(self, db):
        sql = "SELECT k.id FROM keyword('axp.systems') k"
        result = materialize_keyword(db, sql)
        assert '"error"' not in result
        assert '_kw_results_' in result


class TestHybridWithVecOps:
    def test_keyword_leaves_vec_ops_untouched(self, db):
        """keyword() only rewrites its own pattern, leaves vec_ops alone."""
        sql = ("SELECT k.id FROM keyword('auth') k "
               "JOIN vec_ops('_raw_chunks', 'test') v ON k.id = v.id")
        result = materialize_keyword(db, sql)
        assert '_kw_results_' in result
        assert 'vec_ops' in result  # keyword doesn't touch vec_ops


class TestModifiers:
    def test_limit_modifier(self, db):
        sql = "SELECT k.id FROM keyword('authentication', 'limit:2') k"
        result = materialize_keyword(db, sql)
        assert '_kw_results_' in result
        rows = db.execute(result).fetchall()
        assert len(rows) <= 2
