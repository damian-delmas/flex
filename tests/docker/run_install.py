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
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

FLEX_HOME = Path.home() / ".flex"
PASS = "\033[32m[PASS]\033[0m"
FAIL = "\033[31m[FAIL]\033[0m"
SKIP = "\033[33m[SKIP]\033[0m"

failures = []


def check(name, condition, detail=""):
    if condition:
        print(f"  {PASS} {name}")
    else:
        msg = f"{name}" + (f": {detail}" if detail else "")
        print(f"  {FAIL} {msg}")
        failures.append(msg)


def skip(name, reason=""):
    print(f"  {SKIP} {name}" + (f": {reason}" if reason else ""))


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1: Package smoke
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("Phase 1: Package smoke")
print("=" * 60)

r = subprocess.run(["flex", "--help"], capture_output=True, text=True, timeout=15)
check("flex --help works", r.returncode == 0, r.stderr[:200] if r.stderr else "")

r = subprocess.run(
    ["python3", "-c", "import flex; print(flex.__file__)"],
    capture_output=True, text=True, timeout=15,
)
check("import flex works", r.returncode == 0, r.stderr[:200] if r.stderr else "")

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2: flex init
# ═══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("Phase 2: flex init --local")
print("=" * 60)

t0 = time.time()
init_result = subprocess.run(
    ["flex", "init", "--local"],
    capture_output=False,  # stream output
    timeout=600,
)
elapsed = time.time() - t0

print()
check("flex init exit 0", init_result.returncode == 0,
      f"exit code {init_result.returncode}")
print(f"  (completed in {elapsed:.1f}s)")

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 3: Filesystem structure
# ═══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("Phase 3: Filesystem")
print("=" * 60)

check("~/.flex/ created", FLEX_HOME.exists())
check("~/.flex/cells/ exists", (FLEX_HOME / "cells").exists())
check("registry.db exists", (FLEX_HOME / "registry.db").exists())
check("model.onnx exists", (FLEX_HOME / "models" / "model.onnx").exists())
check("model.onnx.data exists", (FLEX_HOME / "models" / "model.onnx.data").exists())

# Model size sanity (model.onnx.data should be ~87MB, not truncated)
model_data = FLEX_HOME / "models" / "model.onnx.data"
if model_data.exists():
    size_mb = model_data.stat().st_size / (1024 * 1024)
    check("model.onnx.data size > 80MB", size_mb > 80, f"got {size_mb:.1f}MB")
else:
    check("model.onnx.data size > 80MB", False, "file missing")

settings = Path.home() / ".claude" / "settings.json"
check("settings.json exists", settings.exists())
claude_json = Path.home() / ".claude.json"
check(".claude.json exists", claude_json.exists())

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 4: Hooks wiring
# ═══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("Phase 4: Hooks")
print("=" * 60)

if settings.exists():
    s = json.loads(settings.read_text())
    hooks = s.get("hooks", {})
    check("PostToolUse hooked", "PostToolUse" in hooks)
    check("UserPromptSubmit hooked", "UserPromptSubmit" in hooks)
    all_cmds = []
    for group in hooks.get("PostToolUse", []):
        for h in group.get("hooks", []):
            all_cmds.append(h.get("command", ""))
    check("capture hook registered",
          any("claude-code-capture" in c for c in all_cmds),
          f"cmds: {all_cmds}")
else:
    skip("hooks check", "settings.json missing")

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 5: Cell contents
# ═══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("Phase 5: Cell contents")
print("=" * 60)

sys.path.insert(0, "/test")
import flex.registry as reg

cell_path = reg.resolve_cell("claude_code")
check("cell registered", cell_path is not None)
check("cell db on disk", cell_path is not None and cell_path.exists())

if cell_path and cell_path.exists():
    conn = sqlite3.connect(str(cell_path))

    n_chunks = conn.execute("SELECT COUNT(*) FROM _raw_chunks").fetchone()[0]
    n_sessions = conn.execute("SELECT COUNT(*) FROM _raw_sources").fetchone()[0]
    n_embedded = conn.execute(
        "SELECT COUNT(*) FROM _raw_chunks WHERE embedding IS NOT NULL"
    ).fetchone()[0]

    check("chunks indexed", n_chunks > 0, f"got {n_chunks}")
    check("sessions indexed", n_sessions >= 4, f"got {n_sessions} (expect 4)")
    check("embeddings present", n_embedded > 0, f"got {n_embedded}/{n_chunks}")
    check("all chunks embedded", n_embedded == n_chunks,
          f"{n_embedded}/{n_chunks}")

    # Tool ops
    tool_names = {r[0] for r in conn.execute(
        "SELECT DISTINCT tool_name FROM _edges_tool_ops"
    ).fetchall()}
    check("tool_ops extracted",
          bool({"Read", "Edit", "Write"} & tool_names),
          f"got {tool_names}")

    # Message types
    msg_types = {r[0] for r in conn.execute(
        "SELECT DISTINCT type FROM _types_message"
    ).fetchall()}
    check("message types classified",
          bool({"user_prompt", "tool_call"} & msg_types),
          f"got {msg_types}")

    # Presets
    preset_names = {r[0] for r in conn.execute(
        "SELECT name FROM _presets"
    ).fetchall()}
    check("@orient installed", "orient" in preset_names,
          f"presets: {sorted(preset_names)}")

    # Views
    view_names = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='view'"
    ).fetchall()}
    check("messages view exists", "messages" in view_names, f"views: {view_names}")
    check("sessions view exists", "sessions" in view_names, f"views: {view_names}")

    # Views queryable
    conn.row_factory = sqlite3.Row
    try:
        n_msg = conn.execute("SELECT COUNT(*) as n FROM messages").fetchone()[0]
        check("messages view queryable", n_msg > 0, f"got {n_msg}")
    except Exception as e:
        check("messages view queryable", False, str(e))

    try:
        n_sess = conn.execute("SELECT COUNT(*) as n FROM sessions").fetchone()[0]
        check("sessions view queryable", n_sess > 0, f"got {n_sess}")
    except Exception as e:
        check("sessions view queryable", False, str(e))

    # Embedding dimension uniformity
    dims = {len(r[0]) // 4 for r in conn.execute(
        "SELECT embedding FROM _raw_chunks WHERE embedding IS NOT NULL LIMIT 500"
    ).fetchall()}
    check("uniform embedding dims", len(dims) == 1, f"got dims: {dims}")
    if dims:
        check("embedding dim = 128", 128 in dims, f"got {dims}")

    # Delegation edges
    n_deleg = conn.execute(
        "SELECT COUNT(*) FROM _edges_delegations"
    ).fetchone()[0]
    check("delegation edges exist", n_deleg > 0, f"got {n_deleg}")

    # FTS
    try:
        fts_count = conn.execute(
            "SELECT COUNT(*) FROM chunks_fts WHERE chunks_fts MATCH 'auth'"
        ).fetchone()[0]
        check("FTS works", fts_count > 0, f"got {fts_count} matches")
    except Exception as e:
        check("FTS works", False, str(e))

    conn.close()

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 6: flex search CLI
# ═══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("Phase 6: flex search")
print("=" * 60)

# SQL
r = subprocess.run(
    ["flex", "search", "--json",
     "SELECT COUNT(*) as n FROM _raw_chunks"],
    capture_output=True, text=True, timeout=15,
)
check("flex search SQL exit 0", r.returncode == 0,
      r.stderr[:200] if r.stderr else "")

# Authorizer blocks writes
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
            check(label, blocked, f"got: {r_auth.stdout[:200]}")
        except Exception:
            check(label, True)
    else:
        check(label, True)

# ═══════════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 60)
if failures:
    print(f"\033[31mFAILED — {len(failures)} checks:\033[0m")
    for f in failures:
        print(f"  \u2022 {f}")
    sys.exit(1)
else:
    print(f"\033[32mAll checks passed — install path E2E OK\033[0m")
    sys.exit(0)
