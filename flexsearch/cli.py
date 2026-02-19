#!/usr/bin/env python3
"""
Flex CLI — flex init + flex search.

pip install getflex
flex init              # hooks + daemon + MCP wiring
flex search "query"    # query your sessions
"""

import argparse
import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

FLEX_HOME = Path.home() / ".flex"
CLAUDE_DIR = Path.home() / ".claude"
CLAUDE_JSON = Path.home() / ".claude.json"
HOOKS_DIR = CLAUDE_DIR / "hooks"
SYSTEMD_DIR = Path.home() / ".config" / "systemd" / "user"

# Package data locations (relative to this file)
PKG_ROOT = Path(__file__).parent
CAPTURE_HOOKS_DIR = PKG_ROOT / "modules" / "claude_code" / "compile" / "hooks"
DATA_HOOKS_DIR = PKG_ROOT / "data" / "hooks"
SYSTEMD_TMPL_DIR = PKG_ROOT / "data" / "systemd"

# PostToolUse matcher — all tool types that produce indexable events
POST_TOOL_MATCHER = (
    "Write|Edit|Read|MultiEdit|NotebookEdit|Grep|Glob|Bash|"
    "WebFetch|WebSearch|Task|TaskOutput|mcp__.*"
)

# Hooks to install
HOOKS = {
    "PostToolUse": [
        {"src": CAPTURE_HOOKS_DIR / "claude-code-capture.sh", "name": "claude-code-capture.sh"},
        {"src": DATA_HOOKS_DIR / "flex-index.sh", "name": "flex-index.sh"},
    ],
    "UserPromptSubmit": [
        {"src": CAPTURE_HOOKS_DIR / "user-prompt-capture.sh", "name": "user-prompt-capture.sh"},
    ],
}


# ============================================================
# flex init
# ============================================================

def _install_hooks():
    """Copy hook scripts to ~/.claude/hooks/ and set executable."""
    HOOKS_DIR.mkdir(parents=True, exist_ok=True)
    installed = []
    for event, hooks in HOOKS.items():
        for hook in hooks:
            dest = HOOKS_DIR / hook["name"]
            shutil.copy2(hook["src"], dest)
            dest.chmod(dest.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
            installed.append(hook["name"])
    return installed


def _patch_settings_json():
    """Non-destructively add hook entries to ~/.claude/settings.json."""
    settings_path = CLAUDE_DIR / "settings.json"
    if settings_path.exists():
        settings = json.loads(settings_path.read_text())
    else:
        CLAUDE_DIR.mkdir(parents=True, exist_ok=True)
        settings = {}

    hooks = settings.setdefault("hooks", {})

    # --- PostToolUse ---
    post_hooks = hooks.setdefault("PostToolUse", [])
    our_commands = {str(HOOKS_DIR / h["name"]) for h in HOOKS["PostToolUse"]}
    # Check if our hooks are already registered
    already = set()
    for group in post_hooks:
        for h in group.get("hooks", []):
            if h.get("command") in our_commands:
                already.add(h["command"])
    missing = our_commands - already
    if missing:
        new_group = {
            "matcher": POST_TOOL_MATCHER,
            "hooks": [
                {"type": "command", "command": cmd, "timeout": 5}
                for cmd in sorted(missing)
            ],
        }
        post_hooks.append(new_group)

    # --- UserPromptSubmit ---
    user_hooks = hooks.setdefault("UserPromptSubmit", [])
    our_commands = {str(HOOKS_DIR / h["name"]) for h in HOOKS["UserPromptSubmit"]}
    already = set()
    for group in user_hooks:
        for h in group.get("hooks", []):
            if h.get("command") in our_commands:
                already.add(h["command"])
    missing = our_commands - already
    if missing:
        new_group = {
            "hooks": [
                {"type": "command", "command": cmd, "timeout": 5}
                for cmd in sorted(missing)
            ],
        }
        user_hooks.append(new_group)

    settings_path.write_text(json.dumps(settings, indent=2) + "\n")


def _install_systemd():
    """Generate and install systemd user units. Returns True if installed."""
    if sys.platform != "linux":
        print("  [skip] systemd not available (not Linux)")
        print("         macOS launchd support coming in v0.1")
        return False

    SYSTEMD_DIR.mkdir(parents=True, exist_ok=True)
    python = sys.executable

    for tmpl_name, service_name in [
        ("flex-worker.service.tmpl", "flex-worker.service"),
        ("flex-mcp.service.tmpl", "flex-mcp.service"),
    ]:
        tmpl = (SYSTEMD_TMPL_DIR / tmpl_name).read_text()
        rendered = tmpl.replace("{{PYTHON}}", python)
        (SYSTEMD_DIR / service_name).write_text(rendered)

    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
    subprocess.run(
        ["systemctl", "--user", "enable", "--now", "flex-worker", "flex-mcp"],
        check=True,
    )
    return True


def _patch_claude_json():
    """Add MCP server entry to ~/.claude.json."""
    if CLAUDE_JSON.exists():
        data = json.loads(CLAUDE_JSON.read_text())
    else:
        data = {}

    servers = data.setdefault("mcpServers", {})
    if "flex" not in servers:
        servers["flex"] = {"type": "sse", "url": "http://localhost:8081/sse"}
        CLAUDE_JSON.write_text(json.dumps(data, indent=2) + "\n")
        return True
    return False


def cmd_init(args):
    """Wire hooks, daemon, and MCP for Claude Code capture."""
    print("flex init")
    print()

    # 1. Create ~/.flex/
    FLEX_HOME.mkdir(parents=True, exist_ok=True)
    (FLEX_HOME / "cells").mkdir(exist_ok=True)
    print("  [ok] ~/.flex/ created")

    # 2. Install hooks
    installed = _install_hooks()
    print(f"  [ok] hooks installed: {', '.join(installed)}")

    # 3. Patch settings.json
    _patch_settings_json()
    print("  [ok] ~/.claude/settings.json patched")

    # 4. Install systemd services
    if _install_systemd():
        print("  [ok] flex-worker + flex-mcp services started")

    # 5. Patch .claude.json
    if _patch_claude_json():
        print("  [ok] MCP server wired (localhost:8081)")
    else:
        print("  [ok] MCP server already wired")

    print()
    print("Done. Claude Code will now capture your sessions automatically.")
    print("Use 'flex search \"your query\"' to search, or ask Claude directly.")


# ============================================================
# flex search
# ============================================================

def _open_cell_for_search(cell_name: str):
    """Open a cell with vec_ops UDF registered. Returns (db, cleanup) or exits."""
    from flexsearch.registry import resolve_cell
    from flexsearch.core import open_cell

    path = resolve_cell(cell_name)
    if path is None:
        print(f"Cell '{cell_name}' not found.", file=sys.stderr)
        print("Run 'flex init' first, then use Claude Code to build your index.", file=sys.stderr)
        sys.exit(1)

    db = open_cell(str(path))

    # Try to register vec_ops (needs ONNX + embeddings)
    try:
        from flexsearch.retrieve.vec_ops import VectorCache, register_vec_ops
        from flexsearch.onnx import get_model

        embedder = get_model()
        caches = {}
        for table, id_col in [("_raw_chunks", "id"), ("_raw_sources", "source_id")]:
            try:
                cache = VectorCache()
                cache.load_from_db(db, table, "embedding", id_col)
                if cache.size > 0:
                    cache.load_columns(db, table, id_col)
                    caches[table] = cache
            except Exception:
                pass

        if caches and embedder:
            # Read vec config from _meta
            config = {}
            try:
                rows = db.execute(
                    "SELECT key, value FROM _meta WHERE key LIKE 'vec:%'"
                ).fetchall()
                config = {r[0]: r[1] for r in rows}
            except Exception:
                pass
            register_vec_ops(db, caches, embedder.encode, config)
    except ImportError:
        pass  # vec_ops won't work but plain SQL is fine

    return db


def _format_results(result_json: str, as_json: bool = False) -> str:
    """Format query results for terminal output."""
    if as_json:
        return result_json

    data = json.loads(result_json)
    if isinstance(data, dict) and "error" in data:
        return f"Error: {data['error']}"
    if not isinstance(data, list) or len(data) == 0:
        return "No results."

    # Simple table format
    keys = list(data[0].keys())
    # Compute column widths
    widths = {k: len(k) for k in keys}
    for row in data:
        for k in keys:
            val = str(row.get(k, ""))
            if len(val) > 80:
                val = val[:77] + "..."
            widths[k] = max(widths[k], len(val))

    # Cap total width
    lines = []
    header = "  ".join(k.ljust(widths[k]) for k in keys)
    lines.append(header)
    lines.append("  ".join("-" * widths[k] for k in keys))
    for row in data:
        vals = []
        for k in keys:
            val = str(row.get(k, ""))
            if len(val) > 80:
                val = val[:77] + "..."
            vals.append(val.ljust(widths[k]))
        lines.append("  ".join(vals))

    return "\n".join(lines)


def cmd_search(args):
    """Execute a query against a cell."""
    # Lazy import — avoids pulling in mcp deps at CLI startup
    from flexsearch.mcp_server import execute_query

    db = _open_cell_for_search(args.cell)
    try:
        result = execute_query(db, args.query)
        print(_format_results(result, as_json=args.json))
    finally:
        db.close()


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        prog="flex",
        description="Your AI sessions, searchable forever.",
    )
    sub = parser.add_subparsers(dest="command")

    # flex init
    sub.add_parser("init", help="Wire hooks, daemon, and MCP for Claude Code")

    # flex search
    search_p = sub.add_parser("search", help="Search your sessions")
    search_p.add_argument("query", help="SQL query, @preset, or vec_ops expression")
    search_p.add_argument("--cell", default="claude_code", help="Cell to query (default: claude_code)")
    search_p.add_argument("--json", action="store_true", help="Output raw JSON")

    args = parser.parse_args()
    if args.command == "init":
        cmd_init(args)
    elif args.command == "search":
        cmd_search(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
