"""
Tests for the SOMA identity module.

Tests compile.py (import, ensure_tables, insert_edges),
audit.py (coverage queries), and table DDL correctness.
"""

import sqlite3
import struct
import pytest


def _can_import():
    try:
        import flex.modules.soma.compile
        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(not _can_import(), reason="flex not importable")


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def soma_cell():
    """Minimal cell with SOMA identity tables."""
    from tests.conftest import CHUNK_ATOM_DDL, CLAUDE_CODE_MODULE_DDL, SOMA_MODULE_DDL
    conn = sqlite3.connect(':memory:')
    conn.row_factory = sqlite3.Row
    conn.executescript(CHUNK_ATOM_DDL)
    conn.executescript(CLAUDE_CODE_MODULE_DDL)
    conn.executescript(SOMA_MODULE_DDL)

    # Insert a session
    conn.execute(
        "INSERT INTO _raw_sources (source_id, title) VALUES ('sess-001', 'Test session')"
    )

    # Insert chunks with tool ops
    for i, (tool, path) in enumerate([
        ('Edit', '/home/user/project/main.py'),
        ('Read', '/home/user/project/utils.py'),
        ('Write', '/home/user/project/new.py'),
        ('WebFetch', None),
        ('Bash', '/home/user/project'),
    ]):
        cid = f"sess-001_{i}"
        conn.execute(
            "INSERT INTO _raw_chunks (id, content, timestamp) VALUES (?, ?, ?)",
            (cid, f"{tool} {path or 'web'}", 1707000000 + i)
        )
        conn.execute(
            "INSERT INTO _edges_source (chunk_id, source_id, position) VALUES (?, 'sess-001', ?)",
            (cid, i)
        )
        conn.execute(
            "INSERT INTO _types_message (chunk_id, type, role, chunk_number) VALUES (?, 'tool_call', 'assistant', ?)",
            (cid, i)
        )
        conn.execute(
            "INSERT INTO _edges_tool_ops (chunk_id, tool_name, target_file) VALUES (?, ?, ?)",
            (cid, tool, path)
        )

    conn.commit()
    yield conn
    conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# Test: Module imports
# ─────────────────────────────────────────────────────────────────────────────

class TestSomaImport:
    def test_compile_importable(self):
        from flex.modules.soma import compile
        assert hasattr(compile, 'enrich')
        assert hasattr(compile, 'insert_edges')
        assert hasattr(compile, 'ensure_tables')

    def test_constants_present(self):
        from flex.modules.soma.compile import (
            IDENTITY_APPLICABILITY, FILE_TOOLS, MUTATION_TOOLS, REPO_TOOLS,
            APPLICABLE_FILE_TOOLS, APPLICABLE_MUTATION_TOOLS, APPLICABLE_REPO_TOOLS,
        )
        assert 'file_uuid' in IDENTITY_APPLICABILITY
        assert 'Write' in FILE_TOOLS
        assert 'Edit' in MUTATION_TOOLS
        assert 'Bash' in REPO_TOOLS

    def test_old_enrich_deleted(self):
        """enrich.py should no longer exist in claude_code/compile/."""
        with pytest.raises(ImportError):
            from flex.modules.claude_code.compile.enrich import enrich_event


# ─────────────────────────────────────────────────────────────────────────────
# Test: ensure_tables
# ─────────────────────────────────────────────────────────────────────────────

class TestEnsureTables:
    def test_creates_all_four_tables(self):
        from flex.modules.soma.compile import ensure_tables
        conn = sqlite3.connect(':memory:')
        ensure_tables(conn)

        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}

        assert '_edges_file_identity' in tables
        assert '_edges_repo_identity' in tables
        assert '_edges_content_identity' in tables
        assert '_edges_url_identity' in tables
        conn.close()

    def test_idempotent(self):
        from flex.modules.soma.compile import ensure_tables
        conn = sqlite3.connect(':memory:')
        ensure_tables(conn)
        ensure_tables(conn)  # should not raise
        conn.close()

    def test_old_blob_hash_column_exists(self):
        from flex.modules.soma.compile import ensure_tables
        conn = sqlite3.connect(':memory:')
        ensure_tables(conn)

        cols = {r[1] for r in conn.execute("PRAGMA table_info(_edges_content_identity)")}
        assert 'old_blob_hash' in cols
        conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# Test: insert_edges
# ─────────────────────────────────────────────────────────────────────────────

class TestInsertEdges:
    def test_file_identity(self, soma_cell):
        from flex.modules.soma.compile import insert_edges
        chunk = {'id': 'sess-001_0', 'file_uuid': 'uuid-abc'}
        insert_edges(soma_cell, chunk)
        soma_cell.commit()

        row = soma_cell.execute(
            "SELECT file_uuid FROM _edges_file_identity WHERE chunk_id = 'sess-001_0'"
        ).fetchone()
        assert row['file_uuid'] == 'uuid-abc'

    def test_repo_identity(self, soma_cell):
        from flex.modules.soma.compile import insert_edges
        chunk = {'id': 'sess-001_0', 'repo_root': 'root-xyz', 'is_tracked': 1}
        insert_edges(soma_cell, chunk)
        soma_cell.commit()

        row = soma_cell.execute(
            "SELECT repo_root, is_tracked FROM _edges_repo_identity WHERE chunk_id = 'sess-001_0'"
        ).fetchone()
        assert row['repo_root'] == 'root-xyz'
        assert row['is_tracked'] == 1

    def test_content_identity_with_old_blob(self, soma_cell):
        from flex.modules.soma.compile import insert_edges
        chunk = {
            'id': 'sess-001_0',
            'content_hash': 'sha256-abc',
            'blob_hash': 'blob-def',
            'old_blob_hash': 'old-blob-ghi',
        }
        insert_edges(soma_cell, chunk)
        soma_cell.commit()

        row = soma_cell.execute(
            "SELECT content_hash, blob_hash, old_blob_hash FROM _edges_content_identity WHERE chunk_id = 'sess-001_0'"
        ).fetchone()
        assert row['content_hash'] == 'sha256-abc'
        assert row['blob_hash'] == 'blob-def'
        assert row['old_blob_hash'] == 'old-blob-ghi'

    def test_url_identity(self, soma_cell):
        from flex.modules.soma.compile import insert_edges
        chunk = {'id': 'sess-001_3', 'url_uuid': 'url-uuid-xyz'}
        insert_edges(soma_cell, chunk)
        soma_cell.commit()

        row = soma_cell.execute(
            "SELECT url_uuid FROM _edges_url_identity WHERE chunk_id = 'sess-001_3'"
        ).fetchone()
        assert row['url_uuid'] == 'url-uuid-xyz'

    def test_no_fields_no_inserts(self, soma_cell):
        from flex.modules.soma.compile import insert_edges
        chunk = {'id': 'sess-001_0'}
        insert_edges(soma_cell, chunk)
        soma_cell.commit()

        for table in ('_edges_file_identity', '_edges_repo_identity',
                      '_edges_content_identity', '_edges_url_identity'):
            count = soma_cell.execute(
                f"SELECT COUNT(*) FROM {table} WHERE chunk_id = 'sess-001_0'"
            ).fetchone()[0]
            assert count == 0, f"Unexpected row in {table}"

    def test_all_four_tables(self, soma_cell):
        from flex.modules.soma.compile import insert_edges
        chunk = {
            'id': 'sess-001_0',
            'file_uuid': 'uuid-f',
            'repo_root': 'root-r',
            'is_tracked': 1,
            'content_hash': 'hash-c',
            'blob_hash': 'blob-b',
            'old_blob_hash': 'old-o',
            'url_uuid': 'url-u',
        }
        insert_edges(soma_cell, chunk)
        soma_cell.commit()

        assert soma_cell.execute("SELECT COUNT(*) FROM _edges_file_identity WHERE chunk_id='sess-001_0'").fetchone()[0] == 1
        assert soma_cell.execute("SELECT COUNT(*) FROM _edges_repo_identity WHERE chunk_id='sess-001_0'").fetchone()[0] == 1
        assert soma_cell.execute("SELECT COUNT(*) FROM _edges_content_identity WHERE chunk_id='sess-001_0'").fetchone()[0] == 1
        assert soma_cell.execute("SELECT COUNT(*) FROM _edges_url_identity WHERE chunk_id='sess-001_0'").fetchone()[0] == 1


# ─────────────────────────────────────────────────────────────────────────────
# Test: Audit queries
# ─────────────────────────────────────────────────────────────────────────────

class TestAudit:
    def test_empty_coverage(self, soma_cell):
        from flex.modules.soma.manage.audit import audit
        results = audit(soma_cell)

        # No identity edges yet — coverage should be 0%
        for field, (covered, applicable, pct) in results.items():
            assert covered == 0
            assert pct == 0.0

    def test_applicable_counts(self, soma_cell):
        from flex.modules.soma.manage.audit import audit
        results = audit(soma_cell)

        # file_uuid: Edit + Read + Write = 3 applicable (all have target_file, none in /tmp/)
        assert results['file_uuid'][1] == 3
        # content_hash: Edit + Write = 2 applicable (mutations only)
        assert results['content_hash'][1] == 2
        # url_uuid: WebFetch = 1 applicable
        assert results['url_uuid'][1] == 1
        # repo_root: Edit + Read + Write + Bash = 4 (Bash has target_file set)
        assert results['repo_root'][1] == 4

    def test_partial_coverage(self, soma_cell):
        from flex.modules.soma.manage.audit import audit
        from flex.modules.soma.compile import insert_edges

        # Add file identity for one chunk
        insert_edges(soma_cell, {'id': 'sess-001_0', 'file_uuid': 'uuid-test'})
        soma_cell.commit()

        results = audit(soma_cell)
        assert results['file_uuid'][0] == 1  # 1 covered
        assert results['file_uuid'][2] > 0   # pct > 0


# ─────────────────────────────────────────────────────────────────────────────
# Test: 1:N contract (identity tables NOT in views)
# ─────────────────────────────────────────────────────────────────────────────

class TestIdentityContract:
    def test_no_pk_on_chunk_id(self, soma_cell):
        """Identity tables should NOT have PK on chunk_id (they're 1:N)."""
        for table in ('_edges_file_identity', '_edges_repo_identity',
                      '_edges_content_identity', '_edges_url_identity'):
            cols = soma_cell.execute(f"PRAGMA table_info({table})").fetchall()
            pk_cols = [c for c in cols if c[5] > 0]  # pk field > 0 means primary key
            pk_names = [c[1] for c in pk_cols]
            assert 'chunk_id' not in pk_names, f"{table} should not have PK on chunk_id"

    def test_multiple_edges_per_chunk(self, soma_cell):
        """1:N: a chunk can have multiple file_uuid edges."""
        soma_cell.execute(
            "INSERT INTO _edges_file_identity VALUES ('sess-001_0', 'uuid-1')"
        )
        soma_cell.execute(
            "INSERT INTO _edges_file_identity VALUES ('sess-001_0', 'uuid-2')"
        )
        soma_cell.commit()

        count = soma_cell.execute(
            "SELECT COUNT(*) FROM _edges_file_identity WHERE chunk_id = 'sess-001_0'"
        ).fetchone()[0]
        assert count == 2
