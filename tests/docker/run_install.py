"""
Install-path E2E — tests the real `pip install getflex` + `flex init` flow.

Validates:
  1. flex binary works after pip install
  2. flex init completes (model download, cell creation, embeddings, views, presets)
  3. Cell has correct structure (chunks, views, presets, embeddings)
  4. flex search works (SQL, FTS, vec_ops)
  5. Authorizer blocks writes

Exit 0 = pass, exit 1 = fail.
"""
import json
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

from harness import Harness

h = Harness("install")
FLEX_HOME = Path.home() / ".flex"


# ── Phase 1: Package smoke ────────────────────────────────────────────────────
h.phase("Phase 1: Package smoke")

r = subprocess.run(["flex", "--help"], capture_output=True, text=True, timeout=15)
h.check("flex --help works", r.returncode == 0, r.stderr[:200] if r.stderr else "")

r = subprocess.run(
    ["python3", "-c", "import flex; print(flex.__file__)"],
    capture_output=True, text=True, timeout=15,
)
h.check("import flex works", r.returncode == 0, r.stderr[:200] if r.stderr else "")

# ── Phase 2: flex init ────────────────────────────────────────────────────────
h.phase("Phase 2: flex init --local")

t0 = time.time()
init_result = subprocess.run(
    ["flex", "init", "--local"],
    capture_output=False,
    timeout=600,
)
elapsed = time.time() - t0

print()
h.check("flex init exit 0", init_result.returncode == 0,
        f"exit code {init_result.returncode}")
print(f"  (completed in {elapsed:.1f}s)")

# ── Phase 3: Filesystem ──────────────────────────────────────────────────────
h.phase("Phase 3: Filesystem")

h.check("~/.flex/ created", FLEX_HOME.exists())
h.check("~/.flex/cells/ exists", (FLEX_HOME / "cells").exists())
h.check("registry.db exists", (FLEX_HOME / "registry.db").exists())
h.check("model.onnx exists", (FLEX_HOME / "models" / "model.onnx").exists())
h.check("model.onnx.data exists", (FLEX_HOME / "models" / "model.onnx.data").exists())

model_data = FLEX_HOME / "models" / "model.onnx.data"
if model_data.exists():
    size_mb = model_data.stat().st_size / (1024 * 1024)
    h.check("model.onnx.data size > 80MB", size_mb > 80, f"got {size_mb:.1f}MB")
else:
    h.check("model.onnx.data size > 80MB", False, "file missing")

settings = Path.home() / ".claude" / "settings.json"
h.check("settings.json exists", settings.exists())
claude_json = Path.home() / ".claude.json"
h.check(".claude.json exists", claude_json.exists())

# ── Phase 4: Hooks ────────────────────────────────────────────────────────────
h.phase("Phase 4: Hooks")

if settings.exists():
    s = json.loads(settings.read_text())
    hooks = s.get("hooks", {})
    h.check("PostToolUse hooked", "PostToolUse" in hooks)
    h.check("UserPromptSubmit hooked", "UserPromptSubmit" in hooks)
    all_cmds = []
    for group in hooks.get("PostToolUse", []):
        for hook in group.get("hooks", []):
            all_cmds.append(hook.get("command", ""))
    h.check("capture hook registered",
            any("claude-code-capture" in c for c in all_cmds),
            f"cmds: {all_cmds}")
else:
    h.skip("hooks check", "settings.json missing")

# ── Phase 5: Cell contents ────────────────────────────────────────────────────
h.phase("Phase 5: Cell contents")

sys.path.insert(0, "/test")
import flex.registry as reg

cell_path = reg.resolve_cell("claude_code")
h.check("cell registered", cell_path is not None)
h.check("cell db on disk", cell_path is not None and cell_path.exists())

if cell_path and cell_path.exists():
    conn = sqlite3.connect(str(cell_path))

    n_chunks = conn.execute("SELECT COUNT(*) FROM _raw_chunks").fetchone()[0]
    n_sessions = conn.execute("SELECT COUNT(*) FROM _raw_sources").fetchone()[0]
    n_embedded = conn.execute(
        "SELECT COUNT(*) FROM _raw_chunks WHERE embedding IS NOT NULL"
    ).fetchone()[0]

    h.check("chunks indexed", n_chunks > 0, f"got {n_chunks}")
    h.check("sessions indexed", n_sessions >= 4, f"got {n_sessions} (expect >=4)")
    h.check("embeddings present", n_embedded > 0, f"got {n_embedded}/{n_chunks}")
    h.check("all chunks embedded", n_embedded == n_chunks,
            f"{n_embedded}/{n_chunks}")

    tool_names = {r[0] for r in conn.execute(
        "SELECT DISTINCT tool_name FROM _edges_tool_ops"
    ).fetchall()}
    h.check("tool_ops extracted",
            bool({"Read", "Edit", "Write"} & tool_names),
            f"got {tool_names}")

    msg_types = {r[0] for r in conn.execute(
        "SELECT DISTINCT type FROM _types_message"
    ).fetchall()}
    h.check("message types classified",
            bool({"user_prompt", "tool_call"} & msg_types),
            f"got {msg_types}")

    preset_names = {r[0] for r in conn.execute(
        "SELECT name FROM _presets"
    ).fetchall()}
    h.check("@orient installed", "orient" in preset_names,
            f"presets: {sorted(preset_names)}")

    view_names = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='view'"
    ).fetchall()}
    h.check("messages view exists", "messages" in view_names, f"views: {view_names}")
    h.check("sessions view exists", "sessions" in view_names, f"views: {view_names}")

    conn.row_factory = sqlite3.Row
    try:
        n_msg = conn.execute("SELECT COUNT(*) as n FROM messages").fetchone()[0]
        h.check("messages view queryable", n_msg > 0, f"got {n_msg}")
    except Exception as e:
        h.check("messages view queryable", False, str(e))

    try:
        n_sess = conn.execute("SELECT COUNT(*) as n FROM sessions").fetchone()[0]
        h.check("sessions view queryable", n_sess > 0, f"got {n_sess}")
    except Exception as e:
        h.check("sessions view queryable", False, str(e))

    dims = {len(r[0]) // 4 for r in conn.execute(
        "SELECT embedding FROM _raw_chunks WHERE embedding IS NOT NULL LIMIT 500"
    ).fetchall()}
    h.check("uniform embedding dims", len(dims) == 1, f"got dims: {dims}")
    if dims:
        h.check("embedding dim = 128", 128 in dims, f"got {dims}")

    n_deleg = conn.execute(
        "SELECT COUNT(*) FROM _edges_delegations"
    ).fetchone()[0]
    h.check("delegation edges exist", n_deleg > 0, f"got {n_deleg}")

    try:
        fts_count = conn.execute(
            "SELECT COUNT(*) FROM chunks_fts WHERE chunks_fts MATCH 'auth'"
        ).fetchone()[0]
        h.check("FTS works", fts_count > 0, f"got {fts_count} matches")
    except Exception as e:
        h.check("FTS works", False, str(e))

    conn.close()

# ── Phase 6: flex search ─────────────────────────────────────────────────────
h.phase("Phase 6: flex search")

r = subprocess.run(
    ["flex", "search", "--json",
     "SELECT COUNT(*) as n FROM _raw_chunks"],
    capture_output=True, text=True, timeout=15,
)
h.check("flex search SQL exit 0", r.returncode == 0,
        r.stderr[:200] if r.stderr else "")

for label, sql in [
    ("DELETE blocked", "DELETE FROM _raw_chunks WHERE 1=0"),
    ("DROP blocked", "DROP TABLE _raw_chunks"),
]:
    r_auth = subprocess.run(
        ["flex", "search", "--json", sql],
        capture_output=True, text=True, timeout=15,
    )
    if r_auth.returncode == 0:
        try:
            out = json.loads(r_auth.stdout)
            blocked = isinstance(out, dict) and "error" in out
            h.check(label, blocked, f"got: {r_auth.stdout[:200]}")
        except Exception:
            h.check(label, True)
    else:
        h.check(label, True)

# ── Summary ───────────────────────────────────────────────────────────────────
sys.exit(h.finish())
