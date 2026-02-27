"""
Upgrade path E2E — installs old PyPI version, populates cell,
upgrades to current local source, verifies data integrity.

Validates:
  1. Old version installs and inits successfully
  2. Upgrade to current version preserves data
  3. flex sync installs new views/presets
  4. flex search works after upgrade

Exit 0 = pass, exit 1 = fail.
"""
import importlib
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

from harness import Harness

h = Harness("upgrade")
FLEX_HOME = Path.home() / ".flex"
OLD_VERSION = "0.1.43"  # last stable pre-0.2.0


# ── Phase 1: Install old version ──────────────────────────────────────────────
h.phase("Phase 1: Install old version")

r = subprocess.run(
    ["pip3", "install", "--break-system-packages", f"getflex=={OLD_VERSION}"],
    capture_output=True, text=True, timeout=120,
)
h.check("old version installed", r.returncode == 0, r.stderr[:300])

r_ver = subprocess.run(
    ["python3", "-c", "import flex; print(flex.__file__)"],
    capture_output=True, text=True, timeout=10,
)
h.check("old version importable", r_ver.returncode == 0,
        r_ver.stderr[:200] if r_ver.stderr else "")

# ── Phase 2: flex init with old version ───────────────────────────────────────
h.phase("Phase 2: flex init (old version)")

r = subprocess.run(["flex", "init", "--local"], capture_output=False, timeout=600)
h.check("old init exit 0", r.returncode == 0, f"exit code {r.returncode}")

# Snapshot old state
old_state = {}
cell_path = None
try:
    import flex.registry as reg
    cell_path = reg.resolve_cell("claude_code")
    if cell_path and cell_path.exists():
        conn = sqlite3.connect(str(cell_path))
        old_state["chunks"] = conn.execute("SELECT COUNT(*) FROM _raw_chunks").fetchone()[0]
        old_state["sessions"] = conn.execute("SELECT COUNT(*) FROM _raw_sources").fetchone()[0]
        old_state["tables"] = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        conn.close()
        h.check("old cell has chunks", old_state["chunks"] > 0, f"got {old_state['chunks']}")
        h.check("old cell has sessions", old_state["sessions"] > 0, f"got {old_state['sessions']}")
except Exception as e:
    h.check("old state captured", False, str(e))

# ── Phase 3: Upgrade ─────────────────────────────────────────────────────────
h.phase("Phase 3: Upgrade to current version")

r = subprocess.run(
    ["pip3", "install", "--break-system-packages", "-e", "/flex[mcp,cluster]"],
    capture_output=True, text=True, timeout=120,
)
h.check("upgrade installed", r.returncode == 0, r.stderr[:300])

# ── Phase 4: Data integrity ──────────────────────────────────────────────────
h.phase("Phase 4: Data integrity after upgrade")

# Reload registry module
importlib.reload(reg)

cell_path = reg.resolve_cell("claude_code")
h.check("cell still registered", cell_path is not None)

if cell_path and cell_path.exists():
    conn = sqlite3.connect(str(cell_path))
    new_chunks = conn.execute("SELECT COUNT(*) FROM _raw_chunks").fetchone()[0]
    new_sessions = conn.execute("SELECT COUNT(*) FROM _raw_sources").fetchone()[0]

    h.check("chunks preserved",
            new_chunks >= old_state.get("chunks", 0),
            f"old={old_state.get('chunks')} new={new_chunks}")
    h.check("sessions preserved",
            new_sessions >= old_state.get("sessions", 0),
            f"old={old_state.get('sessions')} new={new_sessions}")

    new_tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    # Core tables should survive upgrade
    for t in ["_raw_chunks", "_raw_sources", "_edges_source", "_presets"]:
        if t in old_state.get("tables", set()):
            h.check(f"table {t} preserved", t in new_tables)

    conn.close()

# ── Phase 5: flex sync ───────────────────────────────────────────────────────
h.phase("Phase 5: flex sync after upgrade")

r = subprocess.run(["flex", "sync"], capture_output=True, text=True, timeout=120)
h.check("sync exit 0", r.returncode == 0, r.stderr[:200])

if cell_path and cell_path.exists():
    conn = sqlite3.connect(str(cell_path))
    views = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='view'"
    ).fetchall()}
    h.check("messages view after sync", "messages" in views, f"views: {views}")
    h.check("sessions view after sync", "sessions" in views, f"views: {views}")

    presets = {r[0] for r in conn.execute(
        "SELECT name FROM _presets"
    ).fetchall()}
    h.check("orient preset after sync", "orient" in presets,
            f"presets: {sorted(presets)}")
    conn.close()

# ── Phase 6: Search works ────────────────────────────────────────────────────
h.phase("Phase 6: Search after upgrade")

r = subprocess.run(
    ["flex", "search", "--json", "SELECT COUNT(*) as n FROM _raw_chunks"],
    capture_output=True, text=True, timeout=15,
)
h.check("search works after upgrade", r.returncode == 0,
        r.stderr[:200] if r.stderr else "")

sys.exit(h.finish())
