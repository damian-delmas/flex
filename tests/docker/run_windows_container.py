#!/usr/bin/env python3
"""Provider-general Docker distribution proof for the Windows container package.

Runs on a Docker host (Linux CI is sufficient for this layer). It builds or uses
an exact Flex image, initializes Codex, Claude Code, and Goose in separate
`docker compose run --rm` containers, starts one worker, proves live JSONL and
SQLite updates, exercises stdio MCP, and verifies named-volume persistence.

The clean-Windows bind-mount/reboot gate is owned by deploy/windows/verify.ps1.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from harness import Harness

REPO_ROOT = Path(__file__).resolve().parents[2]
COMPOSE_TEMPLATE = REPO_ROOT / "deploy" / "windows" / "compose.yaml"
DOCKERFILE = REPO_ROOT / "deploy" / "windows" / "Dockerfile"


def run(cmd: list[str], *, cwd: Path | None = None, timeout: int = 1200,
        input_text: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=cwd,
        input=input_text,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def must(h: Harness, name: str, cmd: list[str], *, cwd: Path | None = None,
         timeout: int = 1200, input_text: str | None = None) -> subprocess.CompletedProcess:
    try:
        result = run(cmd, cwd=cwd, timeout=timeout, input_text=input_text)
    except Exception as exc:
        h.check(name, False, str(exc))
        raise
    h.check(name, result.returncode == 0, (result.stderr or result.stdout)[-600:])
    if result.returncode != 0:
        raise RuntimeError(f"{name} failed: {result.stderr or result.stdout}")
    return result


def write_jsonl(path: Path, entries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8")


def seed_codex(root: Path, marker: str) -> Path:
    session_id = "11111111-1111-4111-8111-111111111111"
    path = root / "2026" / "08" / "25" / f"rollout-test-{session_id}.jsonl"
    write_jsonl(path, [
        {"type": "session_meta", "timestamp": "2026-08-25T12:00:00Z",
         "payload": {"id": session_id, "cwd": "/workspace/codex"}},
        {"type": "turn_context", "payload": {"turn_id": "turn-1", "cwd": "/workspace/codex",
          "model": "gpt-test", "approval_policy": "never", "sandbox_policy": {"mode": "read-only"}}},
        {"type": "response_item", "timestamp": "2026-08-25T12:00:01Z",
         "payload": {"type": "message", "role": "user",
                     "content": [{"type": "input_text", "text": marker}]}},
        {"type": "response_item", "timestamp": "2026-08-25T12:00:02Z",
         "payload": {"type": "message", "role": "assistant",
                     "content": [{"type": "output_text", "text": "Codex fixture ready"}]}},
    ])
    return path


def append_codex(path: Path, marker: str) -> None:
    entry = {"type": "response_item", "timestamp": datetime.now(timezone.utc).isoformat(),
             "payload": {"type": "message", "role": "user",
                         "content": [{"type": "input_text", "text": marker}]}}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")


def seed_claude(root: Path, marker: str) -> Path:
    session_id = "22222222-2222-4222-8222-222222222222"
    path = root / "-workspace-claude" / f"{session_id}.jsonl"
    write_jsonl(path, [
        {"type": "user", "uuid": "claude-user-1", "timestamp": "2026-08-25T12:00:00Z",
         "message": {"role": "user", "content": marker}, "cwd": "/workspace/claude", "parentUuid": None},
        {"type": "assistant", "uuid": "claude-assistant-1", "timestamp": "2026-08-25T12:00:01Z",
         "message": {"role": "assistant", "content": [{"type": "text", "text": "Claude fixture ready"}]},
         "cwd": "/workspace/claude", "parentUuid": "claude-user-1"},
    ])
    return path


def append_claude(path: Path, marker: str) -> None:
    entry = {"type": "user", "uuid": f"claude-user-{uuid.uuid4().hex[:8]}",
             "timestamp": datetime.now(timezone.utc).isoformat(),
             "message": {"role": "user", "content": marker},
             "cwd": "/workspace/claude", "parentUuid": "claude-assistant-1"}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")


def seed_external_module(path: Path, marker: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "install.py").write_text(
        "\n".join([
            'CLI_NAME = "external-toy"',
            'MODULE_SUMMARY = "external container proof module"',
            'MODULE = {"cell_type": "external-toy", "description": "external container proof"}',
            'def register_args(parser):',
            '    return None',
            'def run(args, console):',
            '    from flex.sdk import index',
            f'    db = index("external_toy", [{marker!r}], "external module persistence proof")',
            '    db.close()',
            '',
        ]),
        encoding="utf-8",
    )


def seed_goose(path: Path, marker: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    db.executescript("""
        PRAGMA journal_mode=WAL;
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY, name TEXT, description TEXT, session_type TEXT,
            working_dir TEXT, created_at TEXT, updated_at TEXT, provider_name TEXT,
            model_config_json TEXT, goose_mode TEXT, thread_id TEXT,
            total_tokens INTEGER, input_tokens INTEGER, output_tokens INTEGER,
            accumulated_total_tokens INTEGER, accumulated_input_tokens INTEGER,
            accumulated_output_tokens INTEGER, recipe_json TEXT, user_recipe_values_json TEXT
        );
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY, session_id TEXT, role TEXT,
            content_json TEXT, created_timestamp INTEGER, message_id TEXT
        );
    """)
    db.execute(
        "INSERT INTO sessions(id,name,working_dir,created_at,updated_at,provider_name) VALUES(?,?,?,?,?,?)",
        ("goose-session-1", "Goose fixture", "/workspace/goose", "2026-08-25T12:00:00Z",
         "2026-08-25T12:00:01Z", "test"),
    )
    db.execute(
        "INSERT INTO messages(id,session_id,role,content_json,created_timestamp,message_id) VALUES(?,?,?,?,?,?)",
        (1, "goose-session-1", "user", json.dumps([{"type": "text", "text": marker}]),
         1770000000, "goose-message-1"),
    )
    db.commit()
    db.close()


def append_goose(path: Path, marker: str) -> None:
    db = sqlite3.connect(path)
    db.execute(
        "INSERT INTO messages(id,session_id,role,content_json,created_timestamp,message_id) VALUES(?,?,?,?,?,?)",
        (2, "goose-session-1", "user", json.dumps([{"type": "text", "text": marker}]),
         int(time.time()), "goose-message-2"),
    )
    db.execute("UPDATE sessions SET updated_at=? WHERE id='goose-session-1'",
               (datetime.now(timezone.utc).isoformat(),))
    db.commit()
    db.close()


def yaml_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def write_override(
    path: Path,
    mounts: list[tuple[Path, str, bool]],
    init_only_mounts: list[tuple[Path, str, bool]] | None = None,
) -> None:
    lines = ["services:"]
    init_only_mounts = init_only_mounts or []
    for service in ("init", "worker"):
        lines.extend([f"  {service}:", "    volumes:"])
        service_mounts = mounts + (init_only_mounts if service == "init" else [])
        for source, target, read_only in service_mounts:
            lines.extend([
                "      - type: bind",
                f"        source: {yaml_quote(str(source))}",
                f"        target: {yaml_quote(target)}",
                f"        read_only: {'true' if read_only else 'false'}",
            ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def wait_for_marker(container: str, cell: str, marker: str, timeout: int = 150) -> tuple[bool, str]:
    escaped = marker.replace("'", "''")
    query = f"SELECT k.id, k.snippet FROM keyword('\"{escaped}\"') k LIMIT 5"
    deadline = time.time() + timeout
    last = ""
    while time.time() < deadline:
        result = run(["docker", "exec", container, "flex", "search", "--cell", cell, query], timeout=60)
        last = (result.stdout or "") + (result.stderr or "")
        if result.returncode == 0 and marker in last:
            return True, last
        time.sleep(3)
    return False, last


def build_image(h: Harness, image: str, wheel: Path) -> None:
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    wheel_name = wheel.name
    if wheel.parent != REPO_ROOT / "dist":
        (REPO_ROOT / "dist").mkdir(exist_ok=True)
        shutil.copy2(wheel, REPO_ROOT / "dist" / wheel_name)
    dist_version = wheel_name.removeprefix("getflex-").split("-")[0]
    result = must(h, "build exact Flex image", [
        "docker", "build", "-f", str(DOCKERFILE), "-t", image,
        "--build-arg", f"FLEX_VERSION={dist_version}",
        "--build-arg", f"FLEX_WHEEL_SHA256={digest}",
        "--build-arg", f"FLEX_SOURCE_REVISION={run(['git','rev-parse','HEAD'], cwd=REPO_ROOT).stdout.strip()}",
        "--build-arg", "FLEX_IMAGE_REVISION=windows-container-test",
        ".",
    ], cwd=REPO_ROOT, timeout=1800)
    h.artifact("docker-build-tail", "\n".join((result.stdout + result.stderr).splitlines()[-80:]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="getflex-windows:test")
    ap.add_argument("--wheel", type=Path)
    ap.add_argument("--no-build", action="store_true")
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    h = Harness("windows-container")
    suffix = uuid.uuid4().hex[:10]
    project = f"getflex-test-{suffix}"
    container = f"getflex-worker-{suffix}"
    volume = f"getflex-data-{suffix}"
    temp = Path(tempfile.mkdtemp(prefix="getflex-windows-container-"))
    source_root = temp / "sources"
    codex_root = source_root / "codex" / "sessions"
    claude_root = source_root / "claude" / "projects"
    goose_db = source_root / "goose" / "sessions.db"
    external_module = source_root / "external-toy"

    compose = temp / "compose.yaml"
    override = temp / "compose.sources.yaml"
    compose_text = COMPOSE_TEMPLATE.read_text(encoding="utf-8")
    compose_text = compose_text.replace("getflex-worker", container).replace("getflex-data", volume)
    compose.write_text(compose_text, encoding="utf-8")
    (temp / ".env").write_text(f"FLEX_IMAGE={args.image}\n", encoding="utf-8")

    initial = {
        "codex": f"WINDOWS_CONTAINER_CODEX_INITIAL_{suffix}",
        "claude_code": f"WINDOWS_CONTAINER_CLAUDE_INITIAL_{suffix}",
        "goose": f"WINDOWS_CONTAINER_GOOSE_INITIAL_{suffix}",
        "external_toy": f"WINDOWS_CONTAINER_EXTERNAL_INITIAL_{suffix}",
    }
    live = {
        "codex": f"WINDOWS_CONTAINER_CODEX_LIVE_{suffix}",
        "claude_code": f"WINDOWS_CONTAINER_CLAUDE_LIVE_{suffix}",
        "goose": f"WINDOWS_CONTAINER_GOOSE_LIVE_{suffix}",
    }
    codex_file = seed_codex(codex_root, initial["codex"])
    claude_file = seed_claude(claude_root, initial["claude_code"])
    seed_goose(goose_db, initial["goose"])
    seed_external_module(external_module, initial["external_toy"])
    write_override(override, [
        (codex_root, "/root/.codex/sessions", True),
        (claude_root, "/root/.claude/projects", True),
        # Goose remains application-read-only, but SQLite may need writable
        # -shm sidecars beside a WAL database.
        (goose_db.parent, "/root/.local/share/goose/sessions", False),
    ], init_only_mounts=[
        (external_module, "/module-packages/external-toy", True),
    ])

    compose_cmd = [
        "docker", "compose", "--project-name", project,
        "--project-directory", str(temp), "-f", str(compose), "-f", str(override),
    ]

    try:
        h.phase("Preflight")
        h.check("Dockerfile exists", DOCKERFILE.exists())
        h.check("Compose template exists", COMPOSE_TEMPLATE.exists())
        must(h, "docker daemon", ["docker", "info"], timeout=60)
        must(h, "docker compose", ["docker", "compose", "version"], timeout=60)

        if not args.no_build:
            wheel = args.wheel
            if wheel is None:
                wheels = sorted((REPO_ROOT / "dist").glob("getflex-*.whl"))
                if len(wheels) != 1:
                    h.check("exactly one wheel available", False, f"found {len(wheels)} under dist/")
                    return h.finish()
                wheel = wheels[0]
            build_image(h, args.image, wheel.resolve())

        h.phase("Disposable module initialization")
        modules = [
            ("codex", "codex"),
            ("claude-code", "claude_code"),
            ("goose", "goose"),
        ]
        for module, cell in modules:
            must(h, f"init {module}", compose_cmd + [
                "run", "--rm", "-e", "FLEX_SKILL_MODE=none", "init",
                "flex", "init", "--module", module,
            ], timeout=1800)
            must(h, f"orient {cell}", compose_cmd + [
                "run", "--rm", "init", "flex", "search", "--cell", cell, "@orient",
            ], timeout=180)
            listed = run(["docker", "ps", "-a", "--filter", f"label=com.docker.compose.project={project}",
                          "--filter", "label=com.docker.compose.service=init", "--format", "{{.ID}}"])
            h.check(f"{module} init container removed", not listed.stdout.strip(), listed.stdout.strip())

        must(h, "install external module snapshot", compose_cmd + [
            "run", "--rm", "init", "flex", "module", "install", "/module-packages/external-toy",
        ], timeout=300)
        must(h, "init external module", compose_cmd + [
            "run", "--rm", "init", "flex", "init", "--module", "external-toy",
        ], timeout=900)
        must(h, "orient external_toy", compose_cmd + [
            "run", "--rm", "init", "flex", "search", "--cell", "external_toy", "@orient",
        ], timeout=180)
        listed = run(["docker", "ps", "-a", "--filter", f"label=com.docker.compose.project={project}",
                      "--filter", "label=com.docker.compose.service=init", "--format", "{{.ID}}"])
        h.check("external init container removed", not listed.stdout.strip(), listed.stdout.strip())

        h.phase("Single worker and initial retrieval")
        must(h, "start worker", compose_cmd + ["up", "-d", "--wait", "worker"], timeout=300)
        running = must(h, "worker inventory", ["docker", "ps", "--filter", f"name=^{container}$", "--format", "{{.ID}}"])
        h.check("exactly one worker", len(running.stdout.splitlines()) == 1, running.stdout)
        for cell, marker in initial.items():
            ok, output = wait_for_marker(container, cell, marker, timeout=90)
            h.check(f"initial marker in {cell}", ok, output[-400:])

        h.phase("Live JSONL and SQLite updates")
        append_codex(codex_file, live["codex"])
        append_claude(claude_file, live["claude_code"])
        append_goose(goose_db, live["goose"])
        for cell, marker in live.items():
            ok, output = wait_for_marker(container, cell, marker, timeout=180)
            h.check(f"live marker in {cell}", ok, output[-400:])

        h.phase("Stdio MCP")
        initialize = json.dumps({
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                       "clientInfo": {"name": "windows-container-test", "version": "1"}},
        }) + "\n"
        mcp = run(["docker", "exec", "-i", container, "python", "-m", "flex.serve"],
                  input_text=initialize, timeout=90)
        h.check("stdio MCP handshake", mcp.returncode == 0 and '"id":1' in mcp.stdout.replace(" ", ""),
                (mcp.stdout + mcp.stderr)[-600:])

        h.phase("Restart and volume persistence")
        must(h, "restart worker", compose_cmd + ["restart", "worker"], timeout=180)
        ok, output = wait_for_marker(container, "codex", live["codex"], timeout=90)
        h.check("query after worker restart", ok, output[-400:])
        before = must(h, "inspect volume before down", ["docker", "volume", "inspect", volume,
                                                           "--format", "{{.Name}}:{{.CreatedAt}}"])
        must(h, "compose down retains volume", compose_cmd + ["down"], timeout=180)
        must(h, "recreate worker", compose_cmd + ["up", "-d", "--wait", "worker"], timeout=300)
        after = must(h, "inspect volume after recreate", ["docker", "volume", "inspect", volume,
                                                                "--format", "{{.Name}}:{{.CreatedAt}}"])
        h.check("named volume identity preserved", before.stdout.strip() == after.stdout.strip(),
                f"before={before.stdout.strip()} after={after.stdout.strip()}")
        ok, output = wait_for_marker(container, "claude_code", live["claude_code"], timeout=90)
        h.check("query after container recreation", ok, output[-400:])
        external = must(h, "external module snapshot persisted", [
            "docker", "exec", container, "flex", "module", "list",
        ])
        h.check("external module listed after recreation", "external-toy" in external.stdout, external.stdout)
        ok, output = wait_for_marker(container, "external_toy", initial["external_toy"], timeout=90)
        h.check("external cell after recreation", ok, output[-400:])

    except Exception as exc:
        h.check("container proof completed", False, str(exc)[:1000])
    finally:
        if not args.keep:
            run(compose_cmd + ["down", "-v", "--remove-orphans"], timeout=180)
            shutil.rmtree(temp, ignore_errors=True)
        else:
            h.artifact("kept-test-directory", str(temp))

    return h.finish()


if __name__ == "__main__":
    raise SystemExit(main())
