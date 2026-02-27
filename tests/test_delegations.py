"""
Tests for delegation edge detection in sync_session_messages.

Covers: progress entry detection, tool_result detection, dedup,
agent_type extraction, heal backfill, unique index.
"""

import json
import sqlite3
import struct
import pytest


def _can_import():
    try:
        import flex.modules.claude_code.compile.worker
        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(not _can_import(), reason="flex not importable")

EMBED_DIM = 128


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_embedding(dim=EMBED_DIM):
    return struct.pack(f'{dim}f', *([0.1] * dim))


def _make_cell(tmp_path):
    """Chunk-atom cell with production delegation schema."""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE _raw_chunks (
            id TEXT PRIMARY KEY, content TEXT, embedding BLOB, timestamp INTEGER
        );
        CREATE TABLE _raw_sources (
            source_id TEXT PRIMARY KEY, source TEXT, project TEXT,
            git_root TEXT, start_time INTEGER, primary_cwd TEXT,
            message_count INTEGER DEFAULT 0, episode_count INTEGER DEFAULT 0,
            end_time INTEGER, duration_minutes INTEGER,
            title TEXT, embedding BLOB
        );
        CREATE TABLE _edges_source (
            chunk_id TEXT NOT NULL, source_id TEXT NOT NULL,
            source_type TEXT DEFAULT 'claude-code', position INTEGER
        );
        CREATE TABLE _types_message (
            chunk_id TEXT PRIMARY KEY, type TEXT, role TEXT,
            chunk_number INTEGER, parent_uuid TEXT,
            is_sidechain INTEGER, entry_uuid TEXT
        );
        CREATE TABLE _edges_tool_ops (
            chunk_id TEXT NOT NULL, tool_name TEXT, target_file TEXT,
            success INTEGER, cwd TEXT, git_branch TEXT
        );
        CREATE TABLE _edges_delegations (
            id INTEGER PRIMARY KEY,
            chunk_id TEXT,
            child_session_id TEXT,
            agent_type TEXT,
            created_at INTEGER,
            parent_source_id TEXT
        );
        CREATE UNIQUE INDEX IF NOT EXISTS idx_deleg_chunk_child
            ON _edges_delegations(chunk_id, child_session_id);
        CREATE TABLE _edges_soft_ops (
            id INTEGER PRIMARY KEY, chunk_id TEXT, file_path TEXT,
            file_uuid TEXT, inferred_op TEXT, confidence TEXT
        );
        CREATE TABLE _raw_content (
            hash TEXT PRIMARY KEY, content TEXT NOT NULL,
            tool_name TEXT, byte_length INTEGER, first_seen INTEGER
        );
        CREATE TABLE _edges_raw_content (
            chunk_id TEXT NOT NULL, content_hash TEXT NOT NULL,
            PRIMARY KEY (chunk_id, content_hash)
        );
    """)
    conn.commit()
    return conn


def _write_jsonl(tmp_path, session_id, entries):
    session_dir = tmp_path / "claude" / "projects" / "test"
    session_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = session_dir / f"{session_id}.jsonl"
    with open(jsonl_path, 'w') as f:
        for entry in entries:
            f.write(json.dumps(entry) + '\n')
    return jsonl_path


def _patch_worker(monkeypatch, tmp_path, session_id, jsonl_path):
    import flex.modules.claude_code.compile.worker as w
    monkeypatch.setattr(w, 'find_jsonl', lambda sid: jsonl_path if sid == session_id else None)
    monkeypatch.setattr(w, 'get_embedder', lambda: None)
    monkeypatch.setattr(w, 'encode', lambda texts: [[0.1] * EMBED_DIM for _ in texts])
    monkeypatch.setattr(w, 'serialize_f32', lambda v: _make_embedding())
    monkeypatch.setattr(w, 'soma_enrich', None)
    monkeypatch.setattr(w, 'soma_insert_edges', None)
    return w


# ─────────────────────────────────────────────────────────────────────────────
# JSONL builders
# ─────────────────────────────────────────────────────────────────────────────

def _user_entry(text, uuid="uuid-001"):
    return {
        "type": "user",
        "uuid": uuid,
        "timestamp": "2026-02-19T15:00:00Z",
        "message": {"role": "user", "content": text},
        "cwd": "/home/test/projects/myapp",
    }


def _assistant_with_task(uuid="uuid-002", tool_use_id="toolu_task_001",
                         subagent_type="Explore", prompt="search for auth"):
    """Assistant entry spawning a Task agent."""
    return {
        "type": "assistant",
        "uuid": uuid,
        "timestamp": "2026-02-19T15:01:00Z",
        "message": {"role": "assistant", "content": [
            {"type": "text", "text": "Let me search for that."},
            {
                "type": "tool_use",
                "id": tool_use_id,
                "name": "Task",
                "input": {
                    "prompt": prompt,
                    "subagent_type": subagent_type,
                    "description": "search auth",
                },
            },
        ]},
        "cwd": "/home/test/projects/myapp",
    }


def _progress_entry(agent_id, parent_tool_use_id):
    """Progress entry from a spawned agent."""
    return {
        "type": "progress",
        "timestamp": "2026-02-19T15:01:30Z",
        "parentToolUseID": parent_tool_use_id,
        "data": {
            "agentId": agent_id,
            "content": [{"type": "text", "text": "Working on it..."}],
        },
    }


def _user_tool_result_entry(agent_id, tool_use_id="toolu_task_001", uuid="uuid-003"):
    """User entry containing tool_result with agentId (old format)."""
    return {
        "type": "user",
        "uuid": uuid,
        "timestamp": "2026-02-19T15:02:00Z",
        "message": {"role": "user", "content": [
            {
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": f"Agent completed. agentId: {agent_id}\nDone.",
            },
        ]},
        "cwd": "/home/test/projects/myapp",
    }


def _assistant_text(text, uuid="uuid-004"):
    return {
        "type": "assistant",
        "uuid": uuid,
        "timestamp": "2026-02-19T15:03:00Z",
        "message": {"role": "assistant", "content": [
            {"type": "text", "text": text},
        ]},
        "cwd": "/home/test/projects/myapp",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestProgressEntryDetection:
    """Progress entries with agentId + parentToolUseID produce delegation edges."""

    def test_progress_creates_delegation_edge(self, tmp_path, monkeypatch):
        conn = _make_cell(tmp_path)
        session_id = "test-deleg-progress"
        agent_id = "abc123def456"
        tuid = "toolu_task_001"

        entries = [
            _user_entry("find auth code"),
            _assistant_with_task(tool_use_id=tuid, subagent_type="Explore"),
            _progress_entry(agent_id, tuid),
            _progress_entry(agent_id, tuid),  # duplicate — should not create 2nd edge
            _assistant_text("Here's what I found"),
        ]
        jsonl_path = _write_jsonl(tmp_path, session_id, entries)
        w = _patch_worker(monkeypatch, tmp_path, session_id, jsonl_path)

        w.sync_session_messages(session_id, conn)
        conn.commit()

        rows = conn.execute(
            "SELECT chunk_id, child_session_id, agent_type, parent_source_id "
            "FROM _edges_delegations"
        ).fetchall()
        assert len(rows) == 1
        row = dict(rows[0])
        assert row['child_session_id'] == f"agent-{agent_id}"
        assert row['parent_source_id'] == session_id
        # chunk_id should point to assistant Task line (line 2), not progress line
        assert row['chunk_id'] == f"{session_id}_2"


class TestUserToolResultDetection:
    """Old-format: user entry with tool_result containing agentId."""

    def test_tool_result_creates_delegation_edge(self, tmp_path, monkeypatch):
        conn = _make_cell(tmp_path)
        session_id = "test-deleg-toolresult"
        agent_id = "ff00112233aa"
        tuid = "toolu_task_002"

        entries = [
            _user_entry("do the thing"),
            _assistant_with_task(tool_use_id=tuid, subagent_type="general-purpose"),
            _user_tool_result_entry(agent_id, tool_use_id=tuid),
            _assistant_text("Done"),
        ]
        jsonl_path = _write_jsonl(tmp_path, session_id, entries)
        w = _patch_worker(monkeypatch, tmp_path, session_id, jsonl_path)

        w.sync_session_messages(session_id, conn)
        conn.commit()

        rows = conn.execute(
            "SELECT chunk_id, child_session_id FROM _edges_delegations"
        ).fetchall()
        assert len(rows) == 1
        row = dict(rows[0])
        assert row['child_session_id'] == f"agent-{agent_id}"
        # Should resolve to the Task tool_use chunk, not the user line
        assert row['chunk_id'] == f"{session_id}_2"


class TestDedupMultipleProgress:
    """Dozens of progress entries for same agent produce exactly 1 edge."""

    def test_many_progress_one_edge(self, tmp_path, monkeypatch):
        conn = _make_cell(tmp_path)
        session_id = "test-deleg-dedup"
        agent_id = "dedup0000aaaa"
        tuid = "toolu_task_003"

        entries = [
            _user_entry("lots of progress"),
            _assistant_with_task(tool_use_id=tuid),
        ]
        # 20 progress entries for same agent
        for _ in range(20):
            entries.append(_progress_entry(agent_id, tuid))
        entries.append(_assistant_text("All done"))

        jsonl_path = _write_jsonl(tmp_path, session_id, entries)
        w = _patch_worker(monkeypatch, tmp_path, session_id, jsonl_path)

        w.sync_session_messages(session_id, conn)
        conn.commit()

        count = conn.execute(
            "SELECT COUNT(*) FROM _edges_delegations"
        ).fetchone()[0]
        assert count == 1


class TestAgentTypeExtracted:
    """subagent_type from Task tool_input populates agent_type column."""

    def test_agent_type_populated(self, tmp_path, monkeypatch):
        conn = _make_cell(tmp_path)
        session_id = "test-deleg-agenttype"
        agent_id = "type0000bbbb"
        tuid = "toolu_task_004"

        entries = [
            _user_entry("explore codebase"),
            _assistant_with_task(tool_use_id=tuid, subagent_type="Explore"),
            _progress_entry(agent_id, tuid),
            _assistant_text("Found it"),
        ]
        jsonl_path = _write_jsonl(tmp_path, session_id, entries)
        w = _patch_worker(monkeypatch, tmp_path, session_id, jsonl_path)

        w.sync_session_messages(session_id, conn)
        conn.commit()

        agent_type = conn.execute(
            "SELECT agent_type FROM _edges_delegations"
        ).fetchone()[0]
        assert agent_type == "Explore"

    def test_general_purpose_type(self, tmp_path, monkeypatch):
        conn = _make_cell(tmp_path)
        session_id = "test-deleg-gp"
        agent_id = "gp00001111cc"
        tuid = "toolu_task_005"

        entries = [
            _user_entry("do research"),
            _assistant_with_task(tool_use_id=tuid, subagent_type="general-purpose"),
            _progress_entry(agent_id, tuid),
            _assistant_text("Research complete"),
        ]
        jsonl_path = _write_jsonl(tmp_path, session_id, entries)
        w = _patch_worker(monkeypatch, tmp_path, session_id, jsonl_path)

        w.sync_session_messages(session_id, conn)
        conn.commit()

        agent_type = conn.execute(
            "SELECT agent_type FROM _edges_delegations"
        ).fetchone()[0]
        assert agent_type == "general-purpose"


class TestHealBackfillsGap:
    """_heal_delegations backfills edges for sessions synced without delegation detection."""

    def test_heal_restores_deleted_edges(self, tmp_path, monkeypatch):
        conn = _make_cell(tmp_path)
        session_id = "test-deleg-heal"
        agent_id = "heal0000dddd"
        tuid = "toolu_task_006"

        entries = [
            _user_entry("heal test"),
            _assistant_with_task(tool_use_id=tuid, subagent_type="Plan"),
            _progress_entry(agent_id, tuid),
            _assistant_text("Planned"),
        ]
        jsonl_path = _write_jsonl(tmp_path, session_id, entries)
        w = _patch_worker(monkeypatch, tmp_path, session_id, jsonl_path)

        # Step 1: sync normally — edges should exist
        w.sync_session_messages(session_id, conn)
        conn.commit()
        assert conn.execute("SELECT COUNT(*) FROM _edges_delegations").fetchone()[0] == 1

        # Step 2: delete edges to simulate pre-detection sync
        conn.execute("DELETE FROM _edges_delegations")
        conn.commit()
        assert conn.execute("SELECT COUNT(*) FROM _edges_delegations").fetchone()[0] == 0

        # Step 3: heal should restore
        healed = w._heal_delegations(conn)
        assert healed >= 1

        rows = conn.execute(
            "SELECT chunk_id, child_session_id, agent_type, parent_source_id "
            "FROM _edges_delegations"
        ).fetchall()
        assert len(rows) >= 1
        row = dict(rows[0])
        assert row['child_session_id'] == f"agent-{agent_id}"
        assert row['agent_type'] == "Plan"
        assert row['parent_source_id'] == session_id


class TestUniqueIndexPreventsDupes:
    """UNIQUE INDEX on (chunk_id, child_session_id) prevents duplicate edges."""

    def test_insert_or_ignore_deduplicates(self, tmp_path):
        conn = _make_cell(tmp_path)

        # Manually insert a delegation edge
        conn.execute(
            "INSERT INTO _edges_delegations (chunk_id, child_session_id, agent_type, created_at) "
            "VALUES (?, ?, ?, ?)",
            ("session_2", "agent-abc123", "Explore", 1707000000)
        )
        conn.commit()

        # Second insert with same (chunk_id, child_session_id) should be ignored
        conn.execute(
            "INSERT OR IGNORE INTO _edges_delegations "
            "(chunk_id, child_session_id, agent_type, created_at) "
            "VALUES (?, ?, ?, ?)",
            ("session_2", "agent-abc123", "general-purpose", 1707000001)
        )
        conn.commit()

        count = conn.execute(
            "SELECT COUNT(*) FROM _edges_delegations"
        ).fetchone()[0]
        assert count == 1

        # Original agent_type preserved (not overwritten)
        agent_type = conn.execute(
            "SELECT agent_type FROM _edges_delegations"
        ).fetchone()[0]
        assert agent_type == "Explore"

    def test_different_child_allowed(self, tmp_path):
        conn = _make_cell(tmp_path)

        # Same chunk, different children — both allowed
        conn.execute(
            "INSERT INTO _edges_delegations (chunk_id, child_session_id, agent_type, created_at) "
            "VALUES (?, ?, ?, ?)",
            ("session_2", "agent-aaa", "Explore", 1707000000)
        )
        conn.execute(
            "INSERT INTO _edges_delegations (chunk_id, child_session_id, agent_type, created_at) "
            "VALUES (?, ?, ?, ?)",
            ("session_2", "agent-bbb", "Plan", 1707000000)
        )
        conn.commit()

        count = conn.execute(
            "SELECT COUNT(*) FROM _edges_delegations"
        ).fetchone()[0]
        assert count == 2
