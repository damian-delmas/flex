"""
UX verification for flex init + flex search output.

Tier A: Regex-based terminal capture (deterministic, always runs)
  - Onboarding panel structure
  - Progress bar format
  - Phase markers
  - Search output format
  - Error message actionability

Tier B: Agent-verified (optional, needs ANTHROPIC_API_KEY)
  - Natural language assessment of output quality

Exit 0 = pass, exit 1 = fail.
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from harness import Harness

h = Harness("ux")


# ── Tier A: flex init terminal output ─────────────────────────────────────────
h.phase("Tier A: flex init output")

r = subprocess.run(
    ["flex", "init", "--local"],
    capture_output=True, text=True, timeout=600,
)
output = r.stdout + r.stderr

# Onboarding panel
h.check("panel: 'Flex is ready'", "Flex is ready" in output)
h.check("panel: MCP endpoint", "localhost:7134" in output)
h.check("panel: usage examples",
        "flex:" in output.lower() or "flex search" in output.lower()
        or "flex:local" in output)

# Progress feedback
h.check("progress: session count",
        bool(re.search(r'\d+\s+sessions?\s+scanned', output)))
h.check("progress: chunk count",
        bool(re.search(r'\d+\s+chunks?\s+embedded', output)))

# Phase markers
for phase in ["storage", "model", "capture"]:
    h.check(f"phase marker: '{phase}'", phase in output.lower())

# Summary line
h.check("summary: session + chunk counts",
        bool(re.search(r'\d+\s+sessions?\s+.+\d+\s+chunks?', output)))


# ── Tier A: flex search output ────────────────────────────────────────────────
h.phase("Tier A: flex search output")

# @orient returns valid JSON
r_orient = subprocess.run(
    ["flex", "search", "--json", "@orient"],
    capture_output=True, text=True, timeout=30,
)
h.check("orient exit 0", r_orient.returncode == 0)
if r_orient.returncode == 0:
    try:
        data = json.loads(r_orient.stdout)
        h.check("orient json valid", True)
        h.check("orient json is list", isinstance(data, list))
    except json.JSONDecodeError as e:
        h.check("orient json valid", False, str(e))

# SQL query returns JSON
r_sql = subprocess.run(
    ["flex", "search", "--json", "SELECT COUNT(*) as n FROM _raw_chunks"],
    capture_output=True, text=True, timeout=15,
)
h.check("sql query exit 0", r_sql.returncode == 0)
if r_sql.returncode == 0:
    try:
        rows = json.loads(r_sql.stdout)
        h.check("sql returns list", isinstance(rows, list))
        h.check("sql has rows", len(rows) > 0)
    except json.JSONDecodeError as e:
        h.check("sql json valid", False, str(e))

# Error messages are actionable
r_bad = subprocess.run(
    ["flex", "search", "SELECT * FROM nonexistent_table_xyz"],
    capture_output=True, text=True, timeout=15,
)
combined = r_bad.stdout + r_bad.stderr
h.check("error for bad query",
        "error" in combined.lower() or "no such table" in combined.lower(),
        f"output: {combined[:200]}")


# ── Tier B: Agent UX verification (optional) ──────────────────────────────────
api_key = os.environ.get("ANTHROPIC_API_KEY", "")
claude_bin = subprocess.run(
    ["which", "claude"], capture_output=True, text=True
).stdout.strip()

if api_key and claude_bin:
    h.phase("Tier B: Agent UX verification")

    prompt = (
        "You are verifying the UX quality of a CLI tool's output. "
        "Here is the output from running 'flex init --local':\n\n"
        f"```\n{output[:3000]}\n```\n\n"
        "Evaluate these criteria (YES or NO):\n"
        "1. progress: Is the progress feedback clear? Can user tell what's happening?\n"
        "2. completion: Is the completion message clear? Does user know it worked?\n"
        "3. next_steps: Are the next steps obvious? Does user know what to do next?\n"
        "4. errors: If there were warnings, are they actionable?\n\n"
        'Return ONLY a JSON object: {"progress": true/false, "completion": true/false, '
        '"next_steps": true/false, "errors": true/false}'
    )

    r_agent = subprocess.run(
        ["claude", "-p", "--output-format", "json",
         "--max-turns", "1", prompt],
        capture_output=True, text=True, timeout=60,
        env={**os.environ, "ANTHROPIC_API_KEY": api_key},
    )

    try:
        agent_out = json.loads(r_agent.stdout) if r_agent.stdout else {}
        result_text = agent_out.get("result", "")
        # Extract JSON from result text (may have markdown fencing)
        json_match = re.search(r'\{[^}]+\}', result_text)
        if json_match:
            ux_eval = json.loads(json_match.group())
            for criterion in ["progress", "completion", "next_steps", "errors"]:
                h.check(f"agent: {criterion} UX ok",
                        ux_eval.get(criterion, False))
        else:
            h.skip("agent UX evaluation", f"no JSON in response: {result_text[:200]}")
    except Exception as e:
        h.skip("agent UX evaluation", f"parse error: {e}")
else:
    h.phase("Tier B: Agent UX verification (SKIPPED)")
    h.skip("agent UX checks", "no ANTHROPIC_API_KEY or claude binary")


sys.exit(h.finish())
