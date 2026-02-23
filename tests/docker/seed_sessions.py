"""
Seed ~/.claude/projects/ with two minimal JSONL sessions
that simulate a real new-user environment.
"""
import json
from pathlib import Path

PROJ_DIR = Path.home() / ".claude" / "projects" / "-home-testuser-projects-myapp"
PROJ_DIR.mkdir(parents=True, exist_ok=True)

SESSION_A = "aaaaaaaa-0000-0000-0000-000000000001"
SESSION_B = "aaaaaaaa-0000-0000-0000-000000000002"

sessions = {
    SESSION_A: [
        {
            "type": "user",
            "uuid": "uuid-a001",
            "timestamp": "2026-02-19T10:00:00Z",
            "message": {"role": "user", "content": "Fix the auth bug in auth.py"},
            "cwd": "/home/testuser/projects/myapp",
            "parentUuid": None,
        },
        {
            "type": "assistant",
            "uuid": "uuid-a002",
            "timestamp": "2026-02-19T10:00:05Z",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "I'll look at auth.py and fix it."},
                    {
                        "type": "tool_use",
                        "id": "toolu_0001",
                        "name": "Read",
                        "input": {"file_path": "/home/testuser/projects/myapp/auth.py"},
                    },
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_0001",
                        "content": "def auth(u, p):\n    return u == 'admin'\n",
                    },
                    {
                        "type": "tool_use",
                        "id": "toolu_0002",
                        "name": "Edit",
                        "input": {
                            "file_path": "/home/testuser/projects/myapp/auth.py",
                            "old_string": "return u == 'admin'",
                            "new_string": "return check_password(u, p)",
                        },
                    },
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_0002",
                        "content": "File edited successfully.",
                    },
                ],
            },
            "cwd": "/home/testuser/projects/myapp",
        },
    ],
    SESSION_B: [
        {
            "type": "user",
            "uuid": "uuid-b001",
            "timestamp": "2026-02-20T09:00:00Z",
            "message": {"role": "user", "content": "Create a utils module"},
            "cwd": "/home/testuser/projects/myapp",
            "parentUuid": None,
        },
        {
            "type": "assistant",
            "uuid": "uuid-b002",
            "timestamp": "2026-02-20T09:00:10Z",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "I'll create utils.py."},
                    {
                        "type": "tool_use",
                        "id": "toolu_0003",
                        "name": "Write",
                        "input": {
                            "file_path": "/home/testuser/projects/myapp/utils.py",
                            "content": "def helper():\n    pass\n",
                        },
                    },
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_0003",
                        "content": "File written.",
                    },
                ],
            },
            "cwd": "/home/testuser/projects/myapp",
        },
    ],
}

for sid, entries in sessions.items():
    p = PROJ_DIR / f"{sid}.jsonl"
    with open(p, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

print(f"Seeded {len(sessions)} sessions in {PROJ_DIR}")
