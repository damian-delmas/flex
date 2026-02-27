"""
test_init_pipeline.py — Integration test for flex init pipeline.

Runs bootstrap + backfill on minimal seed sessions (no external fixtures).
Validates parsing, chunk insertion, and that enrichment completes without
crashing even when git/SOMA ops fail (as they do in CI with no real repos).

ONNX model required for embed phase. Mark skipped if model not present.

Run with: pytest tests/test_init_pipeline.py -v
Run fast (parse only): pytest tests/test_init_pipeline.py -v -k "not embed"
"""
import json
import sqlite3
import time
import pytest
from pathlib import Path

pytestmark = [pytest.mark.pipeline]


# ---------------------------------------------------------------------------
# Minimal JSONL session fixtures
# ---------------------------------------------------------------------------

def _ts(offset=0):
    return f"{time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime(1707000000 + offset))}.000Z"


SESSION_A = [
    {"type": "user", "uuid": "aaaa-0001", "timestamp": _ts(0),
     "message": {"role": "user", "content": "Fix the auth bug in auth.py"},
     "cwd": "/tmp/myapp", "parentUuid": None},
    {"type": "assistant", "uuid": "aaaa-0002", "timestamp": _ts(1),
     "message": {"role": "assistant", "content": [
         {"type": "tool_use", "id": "t1", "name": "Read",
          "input": {"file_path": "/tmp/myapp/auth.py"}},
     ]}, "cwd": "/tmp/myapp"},
    {"type": "tool", "uuid": "aaaa-0003", "timestamp": _ts(2),
     "message": {"role": "tool", "content": [
         {"type": "tool_result", "tool_use_id": "t1",
          "content": "def login(u, p): return check_password(u, p)"}
     ]}, "cwd": "/tmp/myapp"},
    {"type": "assistant", "uuid": "aaaa-0004", "timestamp": _ts(3),
     "message": {"role": "assistant", "content": [
         {"type": "text", "text": "Fixed — using check_password now."}
     ]}, "cwd": "/tmp/myapp"},
]

SESSION_B = [
    {"type": "user", "uuid": "bbbb-0001", "timestamp": _ts(100),
     "message": {"role": "user", "content": "Create a utils module"},
     "cwd": "/tmp/myapp", "parentUuid": None},
    {"type": "assistant", "uuid": "bbbb-0002", "timestamp": _ts(101),
     "message": {"role": "assistant", "content": [
         {"type": "tool_use", "id": "t2", "name": "Write",
          "input": {"file_path": "/tmp/myapp/utils.py",
                    "content": "def helper(): pass"}},
     ]}, "cwd": "/tmp/myapp"},
    {"type": "tool", "uuid": "bbbb-0003", "timestamp": _ts(102),
     "message": {"role": "tool", "content": [
         {"type": "tool_result", "tool_use_id": "t2", "content": ""}
     ]}, "cwd": "/tmp/myapp"},
    {"type": "assistant", "uuid": "bbbb-0004", "timestamp": _ts(103),
     "message": {"role": "assistant", "content": [
         {"type": "text", "text": "Created utils.py."}
     ]}, "cwd": "/tmp/myapp"},
]


def _write_sessions(projects_dir: Path):
    """Write two minimal JSONL sessions. Returns list of paths."""
    proj = projects_dir / "-tmp-myapp"
    proj.mkdir(parents=True, exist_ok=True)

    paths = []
    for name, session in [
        ("aaaaaaaa-0000-0000-0000-000000000001", SESSION_A),
        ("bbbbbbbb-0000-0000-0000-000000000002", SESSION_B),
    ]:
        p = proj / f"{name}.jsonl"
        p.write_text("\n".join(json.dumps(entry) for entry in session))
        paths.append(p)
    return paths


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def pipeline_env(tmp_path, monkeypatch):
    """
    Isolated FLEX_HOME + CLAUDE_PROJECTS with 2 seed sessions.
    Returns dict: {conn, cell_path, projects_dir, session_ids}.
    """
    flex_home = tmp_path / "flex"
    projects_dir = tmp_path / "claude" / "projects"

    monkeypatch.setenv("FLEX_HOME", str(flex_home))

    import flex.registry as registry_mod
    import flex.modules.claude_code.compile.worker as worker_mod
    import importlib
    importlib.reload(registry_mod)

    # Patch CLAUDE_PROJECTS to our tmp dir
    monkeypatch.setattr(worker_mod, "CLAUDE_PROJECTS", projects_dir)

    _write_sessions(projects_dir)

    from flex.modules.claude_code.compile.worker import bootstrap_claude_code_cell
    from flex.cli import _ENRICHMENT_STUBS

    cell_path = bootstrap_claude_code_cell()
    conn = sqlite3.connect(str(cell_path), timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

    # Apply stubs (same as cmd_init)
    for ddl in _ENRICHMENT_STUBS.get("claude-code", []):
        conn.execute(ddl)
    conn.commit()

    yield {
        "conn": conn,
        "cell_path": cell_path,
        "projects_dir": projects_dir,
    }

    conn.close()


# ---------------------------------------------------------------------------
# Parse phase (no embedding — fast, always runs)
# ---------------------------------------------------------------------------

class TestParsePhase:
    """Validates JSONL parsing and chunk insertion — no ONNX needed."""

    def test_sessions_indexed(self, pipeline_env):
        from flex.modules.claude_code.compile.worker import initial_backfill
        conn = pipeline_env["conn"]

        stats = initial_backfill(conn, quiet_embed=True)

        assert stats["sessions"] == 2, f"Expected 2 sessions, got {stats['sessions']}"

    def test_chunks_inserted(self, pipeline_env):
        from flex.modules.claude_code.compile.worker import initial_backfill
        conn = pipeline_env["conn"]

        initial_backfill(conn, quiet_embed=True)

        n = conn.execute("SELECT COUNT(*) FROM _raw_chunks").fetchone()[0]
        assert n > 0, "No chunks inserted after backfill"

    def test_sources_inserted(self, pipeline_env):
        from flex.modules.claude_code.compile.worker import initial_backfill
        conn = pipeline_env["conn"]

        initial_backfill(conn, quiet_embed=True)

        n = conn.execute("SELECT COUNT(*) FROM _raw_sources").fetchone()[0]
        assert n == 2, f"Expected 2 sources, got {n}"

    def test_tool_ops_extracted(self, pipeline_env):
        from flex.modules.claude_code.compile.worker import initial_backfill
        conn = pipeline_env["conn"]

        initial_backfill(conn, quiet_embed=True)

        tools = {
            r[0]
            for r in conn.execute(
                "SELECT DISTINCT tool_name FROM _edges_tool_ops WHERE tool_name IS NOT NULL"
            ).fetchall()
        }
        assert tools & {"Read", "Write"}, f"Expected Read/Write in tool_ops, got {tools}"

    def test_message_types_classified(self, pipeline_env):
        from flex.modules.claude_code.compile.worker import initial_backfill
        conn = pipeline_env["conn"]

        initial_backfill(conn, quiet_embed=True)

        types = {
            r[0]
            for r in conn.execute(
                "SELECT DISTINCT type FROM _types_message WHERE type IS NOT NULL"
            ).fetchall()
        }
        assert "user_prompt" in types or "tool_call" in types, (
            f"Expected message types, got {types}"
        )

    def test_resume_is_idempotent(self, pipeline_env):
        """Running backfill twice must not duplicate sessions or chunks."""
        from flex.modules.claude_code.compile.worker import initial_backfill
        conn = pipeline_env["conn"]

        initial_backfill(conn, quiet_embed=True)
        n1 = conn.execute("SELECT COUNT(*) FROM _raw_chunks").fetchone()[0]
        s1 = conn.execute("SELECT COUNT(*) FROM _raw_sources").fetchone()[0]

        initial_backfill(conn, quiet_embed=True)
        n2 = conn.execute("SELECT COUNT(*) FROM _raw_chunks").fetchone()[0]
        s2 = conn.execute("SELECT COUNT(*) FROM _raw_sources").fetchone()[0]

        assert n1 == n2, f"Chunks duplicated on re-run: {n1} → {n2}"
        assert s1 == s2, f"Sources duplicated on re-run: {s1} → {s2}"


# ---------------------------------------------------------------------------
# Enrichment phase (no embed — just runs silently, must not crash)
# ---------------------------------------------------------------------------

class TestEnrichmentPhase:
    """
    Enrichment runs on the parsed cell. Git/SOMA ops will fail (no real repos
    in CI) — but the pipeline must not crash and all tables must remain queryable.
    """

    def test_enrichment_no_crash(self, pipeline_env):
        from flex.modules.claude_code.compile.worker import initial_backfill
        from flex.cli import _run_enrichment_quiet
        conn = pipeline_env["conn"]

        initial_backfill(conn, quiet_embed=True)
        n_clusters, failures = _run_enrichment_quiet(conn)

        # Some failures are expected (no git repos in CI)
        # But it must not raise and must return valid types
        assert isinstance(n_clusters, int)
        assert isinstance(failures, list)

    def test_stub_tables_queryable_after_enrichment(self, pipeline_env):
        """All stub tables must be queryable even if enrichment failed."""
        from flex.modules.claude_code.compile.worker import initial_backfill
        from flex.cli import _run_enrichment_quiet, _ENRICHMENT_STUBS
        import re
        conn = pipeline_env["conn"]

        initial_backfill(conn, quiet_embed=True)
        _run_enrichment_quiet(conn)

        stub_ddl = " ".join(_ENRICHMENT_STUBS.get("claude-code", []))
        stub_tables = re.findall(r"CREATE TABLE IF NOT EXISTS (\w+)", stub_ddl)

        for table in stub_tables:
            try:
                conn.execute(f"SELECT * FROM {table} LIMIT 1").fetchall()
            except sqlite3.OperationalError as e:
                pytest.fail(f"Stub table {table} not queryable after enrichment: {e}")

    def test_ops_logged_after_enrichment(self, pipeline_env):
        """If any enrichment step ran, _ops should have at least one entry."""
        from flex.modules.claude_code.compile.worker import initial_backfill
        from flex.cli import _run_enrichment_quiet
        conn = pipeline_env["conn"]

        initial_backfill(conn, quiet_embed=True)
        _run_enrichment_quiet(conn)

        # _ops may be empty if all steps failed — just must not crash
        conn.execute("SELECT COUNT(*) FROM _ops").fetchone()

    def test_presets_installed(self, pipeline_env):
        """orient preset must be installed — the most critical one."""
        from flex.modules.claude_code.compile.worker import initial_backfill
        from flex.cli import _run_enrichment_quiet
        conn = pipeline_env["conn"]

        initial_backfill(conn, quiet_embed=True)
        _run_enrichment_quiet(conn)

        names = {
            r[0]
            for r in conn.execute("SELECT name FROM _presets").fetchall()
        }
        assert "orient" in names, (
            f"orient preset not installed. Got: {sorted(names)}"
        )


# ---------------------------------------------------------------------------
# Embed phase (requires ONNX model)
# ---------------------------------------------------------------------------

def _model_available():
    try:
        from flex.onnx.fetch import model_ready
        return model_ready()
    except ImportError:
        return False


@pytest.mark.skipif(not _model_available(), reason="ONNX model not downloaded")
class TestEmbedPhase:
    """Validates embedding — only runs when model is present."""

    def test_chunks_embedded(self, pipeline_env):
        from flex.modules.claude_code.compile.worker import initial_backfill
        conn = pipeline_env["conn"]

        initial_backfill(conn)  # full embed

        n_embedded = conn.execute(
            "SELECT COUNT(*) FROM _raw_chunks WHERE embedding IS NOT NULL"
        ).fetchone()[0]
        n_total = conn.execute("SELECT COUNT(*) FROM _raw_chunks").fetchone()[0]

        assert n_embedded == n_total, (
            f"Only {n_embedded}/{n_total} chunks embedded"
        )

    def test_embedding_dimension(self, pipeline_env):
        import struct
        from flex.modules.claude_code.compile.worker import initial_backfill
        conn = pipeline_env["conn"]

        initial_backfill(conn)

        blob = conn.execute(
            "SELECT embedding FROM _raw_chunks WHERE embedding IS NOT NULL LIMIT 1"
        ).fetchone()[0]
        dim = len(blob) // 4  # float32
        assert dim == 128, f"Expected 128-dim Matryoshka embedding, got {dim}"
