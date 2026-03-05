"""
Claude MCP integration test — verifies the full stack:
flex init → MCP server → Claude Code headless → meaningful response.

Requires:
  - flex init already ran (cell with chunks + embeddings)
  - ANTHROPIC_API_KEY env var set
  - claude binary on PATH (npm install -g @anthropic-ai/claude-code)

Runs 5 canned prompts via `claude -p` with flex MCP tool restriction.
Each prompt escalates complexity: raw SQL → preset → semantic → FTS.

Exit 0 = pass, exit 1 = fail.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from harness import Harness

h = Harness("mcp-claude")

# ── Prerequisites ────────────────────────────────────────────────────────────

h.phase("Prerequisites")

claude_bin = shutil.which("claude")
h.check("claude binary on PATH", claude_bin is not None,
        "install: npm install -g @anthropic-ai/claude-code")

api_key = os.environ.get("ANTHROPIC_API_KEY", "")
creds_file = Path.home() / ".claude" / ".credentials.json"
has_creds = bool(api_key) or creds_file.exists()
h.check("API key or credentials", has_creds,
        "set ANTHROPIC_API_KEY or run 'claude' to authenticate")

cell_db = Path.home() / ".flex" / "cells"
h.check("flex cell exists", cell_db.exists() and any(cell_db.iterdir()),
        "run flex init --local first")

claude_json = Path.home() / ".claude.json"
h.check("MCP config exists", claude_json.exists())

if not claude_bin or not has_creds:
    print("\n  Cannot run MCP tests without claude + credentials. Exiting.")
    sys.exit(h.finish())


# ── Ensure MCP server running ────────────────────────────────────────────────

h.phase("MCP server")

_mcp_proc = None  # track if we started it ourselves

# Check if already running
mcp_ready = False
try:
    r = subprocess.run(
        ["curl", "-sf", "http://localhost:7134/mcp"],
        capture_output=True, timeout=3,
    )
    mcp_ready = True
except Exception:
    pass

if not mcp_ready:
    # Try flex-serve (Docker), fall back to direct start
    if shutil.which("flex-serve"):
        subprocess.run(["flex-serve", "--stop"], capture_output=True)
        subprocess.run(["flex-serve"], capture_output=True, timeout=15)
    else:
        _mcp_proc = subprocess.Popen(
            [sys.executable, "-m", "flex.mcp_server", "--http", "--port", "7134"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

    # Wait for MCP to respond
    for _ in range(15):
        try:
            r = subprocess.run(
                ["curl", "-sf", "http://localhost:7134/mcp"],
                capture_output=True, timeout=3,
            )
            mcp_ready = True
            break
        except Exception:
            time.sleep(0.5)

h.check("MCP server responsive", mcp_ready, "localhost:7134 not responding")

if not mcp_ready:
    print("\n  MCP server not responding. Exiting.")
    if _mcp_proc:
        _mcp_proc.kill()
    sys.exit(h.finish())


# ── Canned prompts ───────────────────────────────────────────────────────────


def _check_positive_int(result):
    """Result contains at least one number > 0."""
    numbers = re.findall(r'\d+', result)
    return any(int(n) > 0 for n in numbers) if numbers else False


PROMPTS = [
    {
        "name": "sql_count",
        "prompt": (
            "Use the flex MCP tool (mcp__flex__flex_search) to run this exact SQL query: "
            "SELECT COUNT(*) as n FROM sessions. "
            'The cell parameter should be "claude_code". '
            "Return ONLY the number from the result, nothing else."
        ),
        "check": _check_positive_int,
        "check_label": "returned session count > 0",
    },
    {
        "name": "preset_orient",
        "prompt": (
            "Use the flex MCP tool (mcp__flex__flex_search) to run the query: @orient "
            'with cell "claude_code". '
            "Summarize what query surface is available (views, tables, presets)."
        ),
        "check": lambda r: any(kw in r.lower() for kw in ["view", "query", "preset", "table", "session", "message"]),
        "check_label": "mentions query surface elements",
    },
    {
        "name": "preset_health",
        "prompt": (
            "Use the flex MCP tool (mcp__flex__flex_search) to run: @health "
            'with cell "claude_code". '
            "Report the chunk count, embedding coverage, and whether the graph is built."
        ),
        "check": lambda r: _check_positive_int(r) and any(kw in r.lower() for kw in ["chunk", "embed", "graph"]),
        "check_label": "reports chunks/embeddings/graph status",
    },
    {
        "name": "semantic_projects",
        "prompt": (
            "Use the flex MCP tool (mcp__flex__flex_search) to find what projects exist. "
            'Cell is "claude_code". '
            "Try: SELECT DISTINCT project, COUNT(*) as n FROM sessions GROUP BY project ORDER BY n DESC LIMIT 10. "
            "List the project names and counts."
        ),
        "check": lambda r: _check_positive_int(r),
        "check_label": "returned project names with counts",
    },
    {
        "name": "fts_keyword",
        "prompt": (
            "Use the flex MCP tool (mcp__flex__flex_search) to search for a keyword. "
            'Cell is "claude_code". '
            "Run: SELECT k.id, k.rank, k.snippet FROM keyword('Read') k LIMIT 5. "
            "Report the results."
        ),
        "check": lambda r: any(kw in r.lower() for kw in ["rank", "snippet", "result", "id", "read"]),
        "check_label": "FTS returned ranked results",
    },
]


h.phase("Claude MCP prompts")

# Build env: unset CLAUDECODE (nesting guard), pass API key if set
_env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}
if api_key:
    _env["ANTHROPIC_API_KEY"] = api_key

for p in PROMPTS:
    name = p["name"]
    prompt = p["prompt"]

    r = subprocess.run(
        ["claude", "-p",
         "--output-format", "json",
         "--allowedTools", "mcp__flex__flex_search",
         "--max-turns", "3",
         prompt],
        capture_output=True, text=True, timeout=120,
        env=_env,
    )

    # Parse response
    try:
        claude_out = json.loads(r.stdout) if r.stdout else {}
    except json.JSONDecodeError:
        claude_out = {}

    is_error = claude_out.get("is_error", True)
    result_text = str(claude_out.get("result", ""))
    num_turns = claude_out.get("num_turns", 0)

    # Store full response for agent audit
    h.artifact(f"claude_{name}", result_text)

    # Checks
    h.check(f"{name}: exit 0", r.returncode == 0,
            r.stderr[:200] if r.stderr else "")
    h.check(f"{name}: no error", not is_error,
            f"result: {result_text[:200]}")
    h.check(f"{name}: tool called", num_turns >= 2,
            f"num_turns={num_turns}")
    h.check(f"{name}: {p['check_label']}", p["check"](result_text),
            f"result: {result_text[:300]}")


# ── Cleanup ──────────────────────────────────────────────────────────────────

if _mcp_proc:
    _mcp_proc.terminate()
    try:
        _mcp_proc.wait(timeout=5)
    except Exception:
        _mcp_proc.kill()
elif shutil.which("flex-serve"):
    subprocess.run(["flex-serve", "--stop"], capture_output=True)

sys.exit(h.finish())
