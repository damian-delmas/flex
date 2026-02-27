"""
Seed ~/.claude/projects/ with realistic JSONL sessions for E2E testing.

Configurable via NUM_SESSIONS env var (default: 400).
Generates sessions across a 5-day window with realistic variety:
  - Mixed tool patterns (Read, Edit, Write, Bash, Grep, Glob, Task)
  - Cross-session file overlap (graph edges)
  - Delegation chains (parent→child agent sessions)
  - Thinking blocks, sidechains
  - Varying session lengths (10-40 chunks)

Target: ~30s embedding on CPU (for testing Ctrl+C interrupt + resume).
"""
import json
import os
import random
import time
from datetime import datetime, timedelta
from pathlib import Path

NUM_SESSIONS = int(os.environ.get("NUM_SESSIONS", "800"))
NUM_DAYS = 5
PROJ_DIR = Path.home() / ".claude" / "projects" / "-home-testuser-projects-myapp"
PROJ_DIR.mkdir(parents=True, exist_ok=True)

random.seed(42)  # deterministic for reproducibility

# ── Project structure (shared files create graph edges) ───────────────────────

PROJECTS = {
    "myapp": {
        "cwd": "/home/testuser/projects/myapp",
        "files": [
            "auth.py", "config.py", "utils.py", "db.py",
            "models/user.py", "models/session.py", "models/order.py",
            "api/routes.py", "api/middleware.py", "api/auth.py",
            "tests/test_auth.py", "tests/test_api.py", "tests/test_db.py",
            "tests/test_utils.py", "tests/conftest.py",
            "migrations/001_init.py", "migrations/002_sessions.py",
            "README.md", "setup.py", "requirements.txt",
        ],
    },
    "frontend": {
        "cwd": "/home/testuser/projects/frontend",
        "files": [
            "src/App.tsx", "src/index.tsx", "src/api/client.ts",
            "src/components/Login.tsx", "src/components/Dashboard.tsx",
            "src/components/UserList.tsx", "src/hooks/useAuth.ts",
            "src/hooks/useApi.ts", "src/types.ts", "src/utils.ts",
            "tests/Login.test.tsx", "tests/Dashboard.test.tsx",
            "package.json", "tsconfig.json", "vite.config.ts",
        ],
    },
    "infra": {
        "cwd": "/home/testuser/projects/infra",
        "files": [
            "terraform/main.tf", "terraform/variables.tf", "terraform/outputs.tf",
            "docker/Dockerfile", "docker/docker-compose.yml",
            "scripts/deploy.sh", "scripts/backup.sh", "scripts/migrate.sh",
            "monitoring/alerts.yml", "monitoring/dashboards.json",
            "README.md",
        ],
    },
}

# ── Session templates ─────────────────────────────────────────────────────────

TASK_TYPES = [
    {"desc": "Fix bug in {file}", "tools": ["Read", "Edit", "Read", "Bash"]},
    {"desc": "Add tests for {file}", "tools": ["Read", "Write", "Bash", "Read"]},
    {"desc": "Refactor {file}", "tools": ["Read", "Read", "Edit", "Edit", "Bash"]},
    {"desc": "Create {file}", "tools": ["Write", "Read", "Edit"]},
    {"desc": "Debug failing tests", "tools": ["Bash", "Read", "Read", "Edit", "Bash"]},
    {"desc": "Review and update {file}", "tools": ["Read", "Grep", "Edit", "Read"]},
    {"desc": "Search for usage of pattern", "tools": ["Grep", "Glob", "Read", "Read"]},
    {"desc": "Update docs and config", "tools": ["Read", "Edit", "Write", "Read"]},
    {"desc": "Performance optimization", "tools": ["Read", "Bash", "Edit", "Bash", "Read"]},
    {"desc": "Add new feature", "tools": ["Read", "Read", "Write", "Edit", "Bash", "Edit"]},
]

CODE_SNIPPETS = {
    ".py": [
        "def authenticate(username, password):\n    user = get_user(username)\n    return user and verify_hash(password, user.password_hash)\n",
        "class UserModel:\n    def __init__(self, id, name, email):\n        self.id = id\n        self.name = name\n        self.email = email\n",
        "import pytest\n\ndef test_create_user():\n    user = create_user('test', 'test@example.com')\n    assert user.name == 'test'\n",
        "from flask import Flask, jsonify, request\n\napp = Flask(__name__)\n\n@app.route('/api/health')\ndef health():\n    return jsonify({'status': 'ok'})\n",
        "import sqlite3\n\ndef get_connection(db_path='app.db'):\n    conn = sqlite3.connect(db_path)\n    conn.row_factory = sqlite3.Row\n    return conn\n",
        "async def fetch_data(url, timeout=30):\n    async with httpx.AsyncClient() as client:\n        response = await client.get(url, timeout=timeout)\n        return response.json()\n",
        "def validate_input(data, schema):\n    errors = []\n    for field, rules in schema.items():\n        if rules.get('required') and field not in data:\n            errors.append(f'Missing required field: {field}')\n    return errors\n",
    ],
    ".tsx": [
        "import React from 'react';\n\nexport const Login: React.FC = () => {\n  const [email, setEmail] = React.useState('');\n  return <form><input value={email} onChange={e => setEmail(e.target.value)} /></form>;\n};\n",
        "export function useAuth() {\n  const [user, setUser] = React.useState(null);\n  const login = async (email: string, password: string) => {\n    const res = await fetch('/api/auth', { method: 'POST', body: JSON.stringify({email, password}) });\n    setUser(await res.json());\n  };\n  return { user, login };\n}\n",
    ],
    ".ts": [
        "export interface User {\n  id: number;\n  name: string;\n  email: string;\n  createdAt: string;\n}\n\nexport interface ApiResponse<T> {\n  data: T;\n  error?: string;\n}\n",
    ],
    ".tf": [
        'resource "aws_instance" "web" {\n  ami           = var.ami_id\n  instance_type = "t3.medium"\n  tags = { Name = "web-server" }\n}\n',
    ],
    ".sh": [
        "#!/bin/bash\nset -euo pipefail\n\necho 'Deploying...'\ndocker-compose up -d\necho 'Done'\n",
    ],
    ".yml": [
        "alerts:\n  - name: high_cpu\n    condition: cpu > 80\n    duration: 5m\n    severity: warning\n",
    ],
    ".json": [
        '{\n  "name": "frontend",\n  "version": "1.0.0",\n  "scripts": { "dev": "vite", "build": "tsc && vite build" }\n}\n',
    ],
    ".md": [
        "# Project Setup\n\n## Prerequisites\n- Python 3.12+\n- Docker\n\n## Installation\n```bash\npip install -r requirements.txt\npython manage.py migrate\n```\n",
    ],
}

BASH_COMMANDS = [
    ("pytest tests/ -v --tb=short", "5 passed in 0.42s"),
    ("python -m pytest --cov=. 2>&1 | tail -5", "TOTAL    124    18    85%\n5 passed"),
    ("git status", "On branch main\nnothing to commit, working tree clean"),
    ("git diff --stat", " auth.py | 4 ++--\n config.py | 3 +++\n 2 files changed, 5 insertions(+), 2 deletions(-)"),
    ("docker-compose up -d", "Creating network... done\nStarting db... done\nStarting web... done"),
    ("curl -s localhost:8000/health | jq .", '{\n  "status": "ok"\n}'),
    ("pip install -r requirements.txt", "Successfully installed flask-2.3.0 sqlalchemy-2.0.0"),
    ("flake8 --count . 2>&1 | tail -3", "0 warnings"),
    ("mypy . --ignore-missing-imports 2>&1 | tail -3", "Success: no issues found"),
    ("ls -la", "total 48\ndrwxr-xr-x  8 user user 4096 Feb 22 14:00 .\n-rw-r--r--  1 user user 1234 Feb 22 13:55 auth.py"),
]

GREP_RESULTS = [
    "auth.py:5:def authenticate(username, password):\nauth.py:12:def check_permission(user, resource):",
    "tests/test_auth.py:1:from auth import authenticate\ntests/test_api.py:3:from auth import check_permission",
    "config.py:1:DATABASE_URL = 'sqlite:///app.db'\nconfig.py:5:SECRET_KEY = 'dev-key'",
]

EDIT_CHANGES = [
    ("return True", "return verify_hash(password, user.password_hash)"),
    ("pass", "raise NotImplementedError('TODO')"),
    ("import os", "import os\nimport logging\n\nlogger = logging.getLogger(__name__)"),
    ("def process(data):", "def process(data: dict) -> dict:"),
    ("# TODO: implement", "result = compute_value(input_data)\nreturn result"),
]

THINKING_TEXTS = [
    "I need to understand the current architecture before making changes. Let me trace the call chain from the API route to the database layer.",
    "The user wants to refactor this module. I should check for all callers first to avoid breaking changes.",
    "This test is failing because the mock isn't matching the updated function signature. Let me check the function definition.",
    "I should split this into two tasks: one for the database migration and one for the API changes. The migration needs to land first.",
    "Looking at the error trace, the issue is in the connection pool. The pool returns stale connections after timeout.",
]

# ── Generator ─────────────────────────────────────────────────────────────────

_uuid_counter = 0

def _uuid():
    global _uuid_counter
    _uuid_counter += 1
    return f"uuid-{_uuid_counter:06d}"


def _session_id(n):
    return f"{'%08x' % n}-0000-0000-0000-{'%012x' % n}"


def _agent_id(n):
    return f"agent-{'%08x' % n}-0000-0000-0000-{'%012x' % n}"


def _snippet_for(filepath):
    """Get a realistic code snippet for a file extension."""
    ext = "." + filepath.rsplit(".", 1)[-1] if "." in filepath else ".py"
    snippets = CODE_SNIPPETS.get(ext, CODE_SNIPPETS[".py"])
    return random.choice(snippets)


def _tool_entry(tool_name, filepath, cwd):
    """Generate a (tool_name, input, result) tuple."""
    full_path = f"{cwd}/{filepath}"
    snippet = _snippet_for(filepath)

    if tool_name == "Read":
        return (tool_name, {"file_path": full_path}, snippet)
    elif tool_name == "Write":
        return (tool_name, {"file_path": full_path, "content": snippet}, "File written.")
    elif tool_name == "Edit":
        old, new = random.choice(EDIT_CHANGES)
        return (tool_name, {"file_path": full_path, "old_string": old, "new_string": new}, "File edited successfully.")
    elif tool_name == "Bash":
        cmd, result = random.choice(BASH_COMMANDS)
        return (tool_name, {"command": f"cd {cwd} && {cmd}"}, result)
    elif tool_name == "Grep":
        return (tool_name, {"pattern": random.choice(["def ", "import ", "class ", "TODO"]), "path": cwd}, random.choice(GREP_RESULTS))
    elif tool_name == "Glob":
        ext = filepath.rsplit(".", 1)[-1] if "." in filepath else "py"
        return (tool_name, {"pattern": f"**/*.{ext}"}, "\n".join(random.sample([f for f in PROJECTS[list(PROJECTS.keys())[0]]["files"]], min(5, len(PROJECTS[list(PROJECTS.keys())[0]]["files"])))))
    else:
        return ("Read", {"file_path": full_path}, snippet)


def generate_session(session_num, base_time, project_name, project, is_child=False, parent_session=None):
    """Generate a single session's JSONL entries."""
    entries = []
    task = random.choice(TASK_TYPES)
    files = random.sample(project["files"], min(len(task["tools"]) + 2, len(project["files"])))
    cwd = project["cwd"]
    primary_file = files[0]

    # Session timestamp: offset within the day
    ts = base_time + timedelta(minutes=random.randint(0, 480))

    # User prompt
    desc = task["desc"].format(file=primary_file)
    entries.append({
        "type": "user",
        "uuid": _uuid(),
        "timestamp": ts.isoformat() + "Z",
        "message": {"role": "user", "content": desc},
        "cwd": cwd,
        "parentUuid": None,
    })
    ts += timedelta(seconds=5)

    # Should this session have thinking blocks?
    use_thinking = random.random() < 0.15  # 15% of sessions

    # Generate tool interactions (each produces ~2-3 chunks)
    tool_sequence = task["tools"]
    # Pad with extra tools for longer sessions
    extra = random.randint(0, 4)
    tool_sequence = tool_sequence + random.choices(["Read", "Edit", "Bash", "Grep"], k=extra)

    for i, tool_name in enumerate(tool_sequence):
        file_for_tool = files[i % len(files)]
        tool_data = _tool_entry(tool_name, file_for_tool, cwd)

        content_blocks = []

        # Add thinking block occasionally
        if use_thinking and i == 0:
            content_blocks.append({
                "type": "thinking",
                "text": random.choice(THINKING_TEXTS),
            })

        # Text before tools
        if i == 0:
            content_blocks.append({"type": "text", "text": f"Let me {task['desc'].format(file=file_for_tool).lower()}."})
        else:
            content_blocks.append({"type": "text", "text": random.choice([
                "Continuing with the changes.",
                "Let me check this next.",
                "Now updating the implementation.",
                "Running verification.",
                "Looking at the related code.",
            ])})

        tool_id = f"toolu_{session_num:04d}_{i:03d}"
        content_blocks.append({
            "type": "tool_use",
            "id": tool_id,
            "name": tool_data[0],
            "input": tool_data[1],
        })
        content_blocks.append({
            "type": "tool_result",
            "tool_use_id": tool_id,
            "content": tool_data[2],
        })

        entries.append({
            "type": "assistant",
            "uuid": _uuid(),
            "timestamp": (ts + timedelta(seconds=i * 10)).isoformat() + "Z",
            "message": {"role": "assistant", "content": content_blocks},
            "cwd": cwd,
        })

    # Add follow-up user message (creates user_prompt chunk)
    ts += timedelta(seconds=len(tool_sequence) * 10 + 5)
    entries.append({
        "type": "user",
        "uuid": _uuid(),
        "timestamp": ts.isoformat() + "Z",
        "message": {"role": "user", "content": random.choice([
            "Looks good, thanks.",
            "Can you also run the tests?",
            "Perfect, let's move on.",
            "Nice, what about the edge cases?",
            "That works. Commit it.",
        ])},
        "cwd": cwd,
        "parentUuid": None,
    })

    # Final assistant response
    entries.append({
        "type": "assistant",
        "uuid": _uuid(),
        "timestamp": (ts + timedelta(seconds=5)).isoformat() + "Z",
        "message": {"role": "assistant", "content": [
            {"type": "text", "text": random.choice([
                "Done. All changes are in place.",
                "Tests passing. The fix is ready.",
                "Changes committed. Moving on.",
                "Everything looks good.",
            ])},
        ]},
        "cwd": cwd,
    })

    return entries


def generate_delegation_session(session_num, base_time, project_name, project, child_nums):
    """Generate a parent session that delegates to child agents."""
    entries = []
    cwd = project["cwd"]
    ts = base_time + timedelta(minutes=random.randint(0, 480))

    # User prompt
    entries.append({
        "type": "user",
        "uuid": _uuid(),
        "timestamp": ts.isoformat() + "Z",
        "message": {"role": "user", "content": "Refactor the codebase and run all tests in parallel"},
        "cwd": cwd,
        "parentUuid": None,
    })
    ts += timedelta(seconds=5)

    # Thinking + initial reads
    files = random.sample(project["files"], min(3, len(project["files"])))
    tool_calls = []
    for f in files[:2]:
        tool_calls.append(_tool_entry("Read", f, cwd))

    content_blocks = [
        {"type": "thinking", "text": "I need to split this into parallel tasks. Let me read the code first, then delegate."},
        {"type": "text", "text": "Let me review the code and then delegate tasks."},
    ]
    for i, (name, inp, result) in enumerate(tool_calls):
        tid = f"toolu_{session_num:04d}_r{i}"
        content_blocks.extend([
            {"type": "tool_use", "id": tid, "name": name, "input": inp},
            {"type": "tool_result", "tool_use_id": tid, "content": result},
        ])

    entries.append({
        "type": "assistant",
        "uuid": _uuid(),
        "timestamp": ts.isoformat() + "Z",
        "message": {"role": "assistant", "content": content_blocks},
        "cwd": cwd,
    })
    ts += timedelta(seconds=10)

    # Delegate to child agents
    task_ids = []
    for ci, child_num in enumerate(child_nums):
        tid = f"toolu_{session_num:04d}_t{ci}"
        task_ids.append(tid)

        entries.append({
            "type": "assistant",
            "uuid": _uuid(),
            "timestamp": (ts + timedelta(seconds=ci * 5)).isoformat() + "Z",
            "message": {"role": "assistant", "content": [
                {"type": "text", "text": f"Delegating task {ci + 1} to a sub-agent."},
                {"type": "tool_use", "id": tid, "name": "Task", "input": {
                    "description": f"Sub-task {ci + 1}",
                    "prompt": random.choice(["Run tests", "Update docs", "Fix lint", "Check coverage"]),
                    "subagent_type": "general-purpose",
                }},
            ]},
            "cwd": cwd,
        })

        # Progress entry with agentId
        entries.append({
            "type": "progress",
            "uuid": _uuid(),
            "timestamp": (ts + timedelta(seconds=ci * 5 + 2)).isoformat() + "Z",
            "parentToolUseID": tid,
            "data": {"agentId": _agent_id(child_num)},
        })

    # Tool results from children
    ts += timedelta(seconds=30)
    result_blocks = [
        {"type": "tool_result", "tool_use_id": tid, "content": "Task completed successfully."}
        for tid in task_ids
    ]
    entries.append({
        "type": "user",
        "uuid": _uuid(),
        "timestamp": ts.isoformat() + "Z",
        "message": {"role": "user", "content": result_blocks},
        "cwd": cwd,
    })

    # Final summary with more tool calls
    ts += timedelta(seconds=5)
    summary_tools = []
    for f in files:
        summary_tools.append(_tool_entry("Read", f, cwd))
    summary_tools.append(_tool_entry("Bash", files[0], cwd))

    content_blocks = [{"type": "text", "text": "All tasks complete. Verifying final state."}]
    for i, (name, inp, result) in enumerate(summary_tools):
        tid = f"toolu_{session_num:04d}_s{i}"
        content_blocks.extend([
            {"type": "tool_use", "id": tid, "name": name, "input": inp},
            {"type": "tool_result", "tool_use_id": tid, "content": result},
        ])

    entries.append({
        "type": "assistant",
        "uuid": _uuid(),
        "timestamp": ts.isoformat() + "Z",
        "message": {"role": "assistant", "content": content_blocks},
        "cwd": cwd,
    })

    return entries


def generate_sidechain_session(session_num, base_time, project_name, project):
    """Generate a session with a sidechain branch (parentUuid branching)."""
    entries = generate_session(session_num, base_time, project_name, project)

    # Add a sidechain: user branches from the first message
    root_uuid = entries[0]["uuid"]
    ts_str = entries[-1]["timestamp"]
    cwd = project["cwd"]
    files = random.sample(project["files"], 2)

    branch_uuid = _uuid()
    entries.append({
        "type": "user",
        "uuid": branch_uuid,
        "timestamp": ts_str,
        "message": {"role": "user", "content": "Actually, let me try a different approach"},
        "cwd": cwd,
        "parentUuid": root_uuid,  # branches from root
    })

    # A couple tool calls on the branch
    for i, f in enumerate(files):
        tool_data = _tool_entry("Read", f, cwd)
        tid = f"toolu_{session_num:04d}_br{i}"
        entries.append({
            "type": "assistant",
            "uuid": _uuid(),
            "timestamp": ts_str,
            "message": {"role": "assistant", "content": [
                {"type": "text", "text": "Taking the alternative approach."},
                {"type": "tool_use", "id": tid, "name": tool_data[0], "input": tool_data[1]},
                {"type": "tool_result", "tool_use_id": tid, "content": tool_data[2]},
            ]},
            "cwd": cwd,
        })

    return entries


# ═══════════════════════════════════════════════════════════════════════════════
# Main generation
# ═══════════════════════════════════════════════════════════════════════════════

t0 = time.time()
base_date = datetime(2026, 2, 19, 9, 0, 0)
project_names = list(PROJECTS.keys())
total_entries = 0
total_chunks_est = 0

# Distribute sessions across days
sessions_per_day = NUM_SESSIONS // NUM_DAYS
remainder = NUM_SESSIONS % NUM_DAYS

session_num = 0
delegation_parents = []  # track which sessions are delegation parents

for day in range(NUM_DAYS):
    day_base = base_date + timedelta(days=day)
    n_today = sessions_per_day + (1 if day < remainder else 0)

    for s in range(n_today):
        session_num += 1
        proj_name = random.choice(project_names)
        proj = PROJECTS[proj_name]
        sid = _session_id(session_num)

        # Decide session type
        roll = random.random()
        if roll < 0.08 and session_num + 2 <= NUM_SESSIONS:
            # 8% delegation sessions (parent + 2 children)
            child1 = session_num + NUM_SESSIONS + 1000  # offset to avoid collision
            child2 = session_num + NUM_SESSIONS + 2000
            entries = generate_delegation_session(session_num, day_base, proj_name, proj, [child1, child2])

            # Write child sessions
            for ci, child_num in enumerate([child1, child2]):
                child_sid = _session_id(child_num)
                child_entries = generate_session(child_num, day_base, proj_name, proj, is_child=True)
                p = PROJ_DIR / f"{child_sid}.jsonl"
                with open(p, "w") as f:
                    for entry in child_entries:
                        f.write(json.dumps(entry) + "\n")
                total_entries += len(child_entries)
                total_chunks_est += len(child_entries)
        elif roll < 0.15:
            # 7% sidechain sessions
            entries = generate_sidechain_session(session_num, day_base, proj_name, proj)
        else:
            # 85% normal sessions
            entries = generate_session(session_num, day_base, proj_name, proj)

        p = PROJ_DIR / f"{sid}.jsonl"
        with open(p, "w") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")
        total_entries += len(entries)
        total_chunks_est += len(entries)

elapsed = time.time() - t0
print(f"Seeded {session_num} sessions + children ({total_entries} entries, ~{total_chunks_est} est chunks) in {elapsed:.1f}s")
print(f"  {PROJ_DIR}")
