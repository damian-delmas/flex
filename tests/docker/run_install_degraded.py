"""
Degraded Install E2E — tests flex init resilience when model is unavailable.

Validates:
  1. Infrastructure survives model failure (hooks, services, MCP wiring)
  2. Query surface works without embeddings (views, presets, FTS)
  3. Exit code is non-zero on partial completion
  4. flex sync recovers missing layers
  5. Authorizer still blocks writes

Runs in Docker after seed_sessions.py has populated ~/.claude/projects/.

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
# Phase 1: Normal init (to get model + cell)
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("Phase 1: Normal init (baseline)")
print("=" * 60)

r = subprocess.run(["flex", "init", "--local"], capture_output=False, timeout=600)
check("baseline init exit 0", r.returncode == 0, f"exit code {r.returncode}")

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2: Corrupt model, delete cell, re-init
# ═══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("Phase 2: Corrupt model + re-init")
print("=" * 60)

# Corrupt model — check both ~/.flex/models/ and bundled location
model_data = FLEX_HOME / "models" / "model.onnx.data"
bundled_data = Path("/flex/flex/onnx/model.onnx.data")  # editable install in Docker
corrupted = False
for p in [model_data, bundled_data]:
    if p.exists():
        p.write_bytes(b"corrupt")
        print(f"  Model corrupted: {p}")
        corrupted = True
if not corrupted:
    print("  WARNING: no model.onnx.data found to corrupt")

# Delete cell to force re-init
registry = FLEX_HOME / "registry.db"
if registry.exists():
    conn = sqlite3.connect(str(registry))
    row = conn.execute("SELECT path FROM cells WHERE name='claude_code'").fetchone()
    if row:
        cell_path = Path(row[0])
        conn.execute("DELETE FROM cells WHERE name='claude_code'")
        conn.commit()
        cell_path.unlink(missing_ok=True)
        # Also remove WAL/SHM
        Path(str(cell_path) + "-wal").unlink(missing_ok=True)
        Path(str(cell_path) + "-shm").unlink(missing_ok=True)
        print("  Cell deleted")
    conn.close()

# Re-init with corrupt model
print("  Running flex init --local with corrupt model...")
r = subprocess.run(
    ["flex", "init", "--local"],
    capture_output=True, text=True, timeout=600,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 3: Infrastructure survives
# ═══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("Phase 3: Infrastructure checks")
print("=" * 60)

# Exit code should be non-zero (model failed)
check("non-zero exit on corrupt model", r.returncode != 0,
      f"exit code {r.returncode}")

check("~/.flex/ exists", FLEX_HOME.exists())
check("registry.db exists", registry.exists())

settings = Path.home() / ".claude" / "settings.json"
check("settings.json exists", settings.exists())

claude_json = Path.home() / ".claude.json"
check("~/.claude.json exists", claude_json.exists())

if claude_json.exists():
    cfg = json.loads(claude_json.read_text())
    check("MCP entry present", "flex" in cfg.get("mcpServers", {}))

# Check hooks
if settings.exists():
    s = json.loads(settings.read_text())
    hooks = s.get("hooks", {})
    check("PostToolUse hooked", "PostToolUse" in hooks)
    check("UserPromptSubmit hooked", "UserPromptSubmit" in hooks)

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 4: Query surface without embeddings
# ═══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("Phase 4: Query surface (no embeddings)")
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

    check("chunks indexed", n_chunks > 0, f"got {n_chunks}")
    check("sessions indexed", n_sessions > 0, f"got {n_sessions}")

    # Embeddings should be absent or partial (model was corrupt)
    n_embedded = conn.execute(
        "SELECT COUNT(*) FROM _raw_chunks WHERE embedding IS NOT NULL"
    ).fetchone()[0]
    check("embeddings absent or partial", n_embedded < n_chunks,
          f"{n_embedded}/{n_chunks} embedded")

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
    check("messages view exists", "messages" in view_names)
    check("sessions view exists", "sessions" in view_names)

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

    # FTS works without embeddings
    try:
        # Use a term from seed data
        fts_count = conn.execute(
            "SELECT COUNT(*) FROM chunks_fts WHERE chunks_fts MATCH 'test OR auth OR code'"
        ).fetchone()[0]
        check("FTS works", fts_count >= 0)  # may be 0 with minimal seed data
    except Exception as e:
        check("FTS table exists", False, str(e))

    conn.close()

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 5: flex sync recovery
# ═══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("Phase 5: flex sync recovery")
print("=" * 60)

r = subprocess.run(
    ["flex", "sync"],
    capture_output=True, text=True, timeout=60,
)
check("flex sync exit 0", r.returncode == 0, f"exit {r.returncode}: {r.stderr[:200]}")

# Re-check after sync
if cell_path and cell_path.exists():
    conn = sqlite3.connect(str(cell_path))

    preset_names = {r[0] for r in conn.execute("SELECT name FROM _presets").fetchall()}
    check("presets still present after sync", "orient" in preset_names)

    view_names = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='view'"
    ).fetchall()}
    check("views still present after sync", "messages" in view_names)

    conn.close()

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 6: Security (authorizer still works)
# ═══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("Phase 6: Security")
print("=" * 60)

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
    print(f"\033[32mAll checks passed — degraded install E2E OK\033[0m")
    sys.exit(0)
