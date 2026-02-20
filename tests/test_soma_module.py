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


# ─────────────────────────────────────────────────────────────────────────────
# Test: old_blob_hash heal pass
# ─────────────────────────────────────────────────────────────────────────────

class TestOldBlobHashHeal:
    """_pass_old_blob_hash backfills from file-history backups."""

    def test_git_blob_hash(self):
        """_git_blob_hash matches git hash-object output."""
        from flex.modules.soma.manage.heal import _git_blob_hash
        content = b"hello world\n"
        # Known git blob hash for "hello world\n" (12 bytes)
        # echo "hello world" | git hash-object --stdin
        expected = "3b18e512dba79e4c8300dd08aeb37f8e728b8dad"
        assert _git_blob_hash(content) == expected

    def test_build_snapshot_map(self, tmp_path):
        """_build_snapshot_map extracts file hashes from JSONL + backup files."""
        import json
        from flex.modules.soma.manage.heal import _build_snapshot_map

        session_id = "heal-test-001"
        assistant_uuid = "assist-heal-001"
        target_file = "/home/test/main.py"
        backup_name = "hash123@v1"

        # Create file-history backup
        fh_dir = tmp_path / ".claude" / "file-history" / session_id
        fh_dir.mkdir(parents=True)
        backup_content = b"original content\n"
        (fh_dir / backup_name).write_bytes(backup_content)

        # Create JSONL
        entries = [
            {"type": "user", "uuid": "u1", "timestamp": "2026-02-19T15:00:00Z",
             "message": {"role": "user", "content": "edit it"}},
            {
                "type": "file-history-snapshot",
                "messageId": assistant_uuid,
                "timestamp": "2026-02-19T15:01:00Z",
                "snapshot": {
                    "messageId": assistant_uuid,
                    "trackedFileBackups": {
                        target_file: {
                            "backupFileName": backup_name,
                            "version": 1,
                            "backupTime": "2026-02-19T15:01:00Z",
                        }
                    },
                    "timestamp": "2026-02-19T15:01:00Z",
                },
            },
            {"type": "assistant", "uuid": assistant_uuid,
             "timestamp": "2026-02-19T15:01:01Z",
             "message": {"role": "assistant", "content": [
                 {"type": "tool_use", "id": "t1", "name": "Edit",
                  "input": {"file_path": target_file, "old_string": "x", "new_string": "y"}}
             ]}},
        ]
        jsonl_path = tmp_path / f"{session_id}.jsonl"
        with open(jsonl_path, 'w') as f:
            for e in entries:
                f.write(json.dumps(e) + '\n')

        # Monkeypatch Path.home for file-history lookup
        from pathlib import Path
        original_home = Path.home
        try:
            Path.home = classmethod(lambda cls: tmp_path)
            result = _build_snapshot_map(jsonl_path, session_id)
        finally:
            Path.home = original_home

        # Line 3 (1-indexed) is the assistant entry
        assert 3 in result
        assert target_file in result[3]
        # Verify it's a valid sha1 hex
        assert len(result[3][target_file]) == 40

    def test_pass_updates_existing_row(self, tmp_path, monkeypatch):
        """_pass_old_blob_hash UPDATEs rows that already have content_hash."""
        import json
        from tests.conftest import CHUNK_ATOM_DDL, CLAUDE_CODE_MODULE_DDL, SOMA_MODULE_DDL
        from flex.modules.soma.manage.heal import _pass_old_blob_hash, _git_blob_hash

        conn = sqlite3.connect(':memory:')
        conn.executescript(CHUNK_ATOM_DDL)
        conn.executescript(CLAUDE_CODE_MODULE_DDL)
        conn.executescript(SOMA_MODULE_DDL)

        session_id = "heal-sess-001"
        target_file = "/home/user/project/main.py"
        assistant_uuid = "assist-heal-upd"
        # chunk_id = {session}_{line_num} — line 2 is the assistant entry
        chunk_id = f"{session_id}_2"

        # Set up cell data
        conn.execute("INSERT INTO _raw_sources (source_id, title) VALUES (?, 'test')", (session_id,))
        conn.execute("INSERT INTO _raw_chunks (id, content, timestamp) VALUES (?, 'edit main.py', 1707000000)", (chunk_id,))
        conn.execute("INSERT INTO _edges_source (chunk_id, source_id, position) VALUES (?, ?, 0)", (chunk_id, session_id))
        conn.execute("INSERT INTO _edges_tool_ops (chunk_id, tool_name, target_file) VALUES (?, 'Edit', ?)", (chunk_id, target_file))
        conn.execute(
            "INSERT INTO _edges_content_identity (chunk_id, content_hash, blob_hash, old_blob_hash) VALUES (?, 'hash123', NULL, NULL)",
            (chunk_id,)
        )
        conn.commit()

        # Create backup file
        fh_dir = tmp_path / ".claude" / "file-history" / session_id
        fh_dir.mkdir(parents=True)
        backup_content = b"original main.py\n"
        (fh_dir / "main@v1").write_bytes(backup_content)
        expected_hash = _git_blob_hash(backup_content)

        # Create JSONL: line 1 = snapshot, line 2 = assistant
        entries = [
            {
                "type": "file-history-snapshot",
                "messageId": assistant_uuid,
                "snapshot": {
                    "messageId": assistant_uuid,
                    "trackedFileBackups": {
                        target_file: {"backupFileName": "main@v1", "version": 1, "backupTime": "..."},
                    },
                    "timestamp": "2026-02-19T15:01:00Z",
                },
            },
            {"type": "assistant", "uuid": assistant_uuid,
             "timestamp": "2026-02-19T15:01:01Z",
             "message": {"role": "assistant", "content": [
                 {"type": "tool_use", "id": "t1", "name": "Edit",
                  "input": {"file_path": target_file}}
             ]}},
        ]
        jsonl_path = tmp_path / f"{session_id}.jsonl"
        with open(jsonl_path, 'w') as f:
            for e in entries:
                f.write(json.dumps(e) + '\n')

        from pathlib import Path
        monkeypatch.setattr(Path, 'home', classmethod(lambda cls: tmp_path))
        monkeypatch.setattr(
            'flex.modules.claude_code.compile.worker.find_jsonl',
            lambda sid: jsonl_path if sid == session_id else None
        )

        _pass_old_blob_hash(conn)

        row = conn.execute(
            "SELECT old_blob_hash FROM _edges_content_identity WHERE chunk_id = ?",
            (chunk_id,)
        ).fetchone()
        assert row is not None
        assert row[0] == expected_hash
        conn.close()
