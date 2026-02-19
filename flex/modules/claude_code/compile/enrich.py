"""
Event enrichment for claude_code episodes.

Shared between hook/worker (real-time) and backfill (batch).

Hybrid content hashing:
  - Git-tracked files: blob_hash (SHA-1) + content_hash (SHA-256)
  - Untracked files: content_hash (SHA-256) only
  - Both stored for maximum recoverability
"""

import base64
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Identity Applicability Rules
# ─────────────────────────────────────────────────────────────────────────────

IDENTITY_APPLICABILITY = {
    'file_uuid': {
        'description': 'Stable UUID from SOMA FileIdentity (survives renames)',
        'applicable_tools': ['Write', 'Edit', 'MultiEdit', 'Read', 'Glob', 'Grep'],
        'exclude_paths': [r'^/tmp/', r'^/var/tmp/', r'^/dev/'],
        'requires': 'file exists at enrichment time',
    },
    'repo_root': {
        'description': 'Git root commit hash (survives repo moves)',
        'applicable_tools': ['Write', 'Edit', 'MultiEdit', 'Read', 'Glob', 'Grep', 'Bash'],
        'exclude_paths': [r'^/tmp/', r'^/var/tmp/'],
        'requires': 'file is inside a git repository',
    },
    'content_hash': {
        'description': 'SHA-256 of file content (content-addressable)',
        'applicable_tools': ['Write', 'Edit', 'MultiEdit'],
        'exclude_paths': [],
        'requires': 'file exists at enrichment time',
    },
    'blob_hash': {
        'description': 'Git SHA-1 blob hash',
        'applicable_tools': ['Write', 'Edit', 'MultiEdit', 'Read', 'Glob', 'Grep'],
        'exclude_paths': [r'^/tmp/'],
        'requires': 'file is inside a git repository',
    },
    'is_tracked': {
        'description': 'Boolean: file is tracked by git',
        'applicable_tools': ['Write', 'Edit', 'MultiEdit', 'Read', 'Glob', 'Grep'],
        'exclude_paths': [],
        'requires': 'file is inside a git repository',
    },
    'url_uuid': {
        'description': 'Stable URL identity from SOMA URLIdentity',
        'applicable_tools': ['WebFetch'],
        'exclude_paths': [],
        'requires': 'URL present in content',
    },
}

NON_FILE_TOOLS = [
    'user_prompt', 'assistant',
    'Task', 'TaskOutput',
    'WebSearch',
    'TodoWrite',
    'AskUserQuestion',
    'Skill',
]

APPLICABLE_FILE_TOOLS = "('Write', 'Edit', 'MultiEdit', 'Read', 'Glob', 'Grep')"
APPLICABLE_MUTATION_TOOLS = "('Write', 'Edit', 'MultiEdit')"
APPLICABLE_REPO_TOOLS = "('Write', 'Edit', 'MultiEdit', 'Read', 'Glob', 'Grep', 'Bash')"


# ─────────────────────────────────────────────────────────────────────────────
# Optional dependencies — try/except instead of sys.path hacks
# ─────────────────────────────────────────────────────────────────────────────

GIT_REGISTRY = None
try:
    from registry import GitRegistry
    GIT_REGISTRY = GitRegistry()
except ImportError:
    try:
        # Fallback: try the home git-registry path
        _gr_path = Path.home() / "projects/home/git-registry/main"
        if _gr_path.exists():
            sys.path.insert(0, str(_gr_path))
            from registry import GitRegistry
            GIT_REGISTRY = GitRegistry()
    except (ImportError, Exception):
        pass

FILE_IDENTITY = None
CONTENT_IDENTITY = None
URL_IDENTITY = None

try:
    from soma.identity.file_identity import FileIdentity
    from soma.identity.content_identity import ContentIdentity
    from soma.identity.url_identity import URLIdentity
    FILE_IDENTITY = FileIdentity()
    CONTENT_IDENTITY = ContentIdentity()
    URL_IDENTITY = URLIdentity()
except ImportError:
    pass


def is_git_tracked(file_path: str, repo: str) -> bool:
    """Check if a file is tracked by git (not just in a git repo)."""
    if not file_path or not repo:
        return False

    try:
        rel_path = os.path.relpath(file_path, repo)
        result = subprocess.run(
            ["git", "-C", repo, "ls-files", "--error-unmatch", rel_path],
            capture_output=True, text=True, timeout=5
        )
        return result.returncode == 0
    except Exception:
        return False


def get_git_info(file_path: str, cwd: str = "", tool: str = "") -> dict:
    """Get git repo info for a file."""
    result = {"repo": "", "blob_hash": "", "old_blob_hash": "", "is_tracked": False}

    if not file_path and not cwd:
        return result

    check_path = file_path if file_path and os.path.exists(file_path) else cwd
    if not check_path:
        return result

    try:
        if os.path.isdir(check_path):
            repo_dir = check_path
        else:
            repo_dir = os.path.dirname(check_path)

        repo_result = subprocess.run(
            ["git", "-C", repo_dir, "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, timeout=5
        )

        if repo_result.returncode != 0:
            return result

        repo = repo_result.stdout.strip()
        result["repo"] = repo

        if file_path and os.path.isfile(file_path):
            blob_result = subprocess.run(
                ["git", "hash-object", file_path],
                capture_output=True, text=True, timeout=5
            )
            if blob_result.returncode == 0:
                result["blob_hash"] = blob_result.stdout.strip()

            result["is_tracked"] = is_git_tracked(file_path, repo)

            if tool in ("Edit", "Write", "MultiEdit") and result["is_tracked"]:
                rel_path = file_path.replace(repo + "/", "")
                old_result = subprocess.run(
                    ["git", "-C", repo, "rev-parse", f"HEAD:{rel_path}"],
                    capture_output=True, text=True, timeout=5
                )
                if old_result.returncode == 0:
                    old_blob = old_result.stdout.strip()
                    if not old_blob.startswith("fatal"):
                        result["old_blob_hash"] = old_blob
    except Exception:
        pass

    return result


def get_registry_info(file_path: str) -> dict:
    """Get git-registry info for a file."""
    result = {"file_relative": "", "repo_root": "", "repo_remote": ""}

    if not file_path or not GIT_REGISTRY:
        return result

    try:
        resolved = GIT_REGISTRY.resolve_file(file_path)
        if resolved:
            relative, repo = resolved
            result["file_relative"] = relative
            if repo.root_commit:
                result["repo_root"] = repo.root_commit
            if repo.remote_url:
                result["repo_remote"] = repo.remote_url
    except Exception:
        pass

    return result


def get_content_hash(file_path: str, session: str = "", msg: int = 0,
                     blob_hash: str = "") -> Optional[str]:
    """Compute and store content hash (SHA-256)."""
    if not file_path or not CONTENT_IDENTITY:
        return None

    if not os.path.isfile(file_path):
        return None

    try:
        content = Path(file_path).read_bytes()
        content_hash = CONTENT_IDENTITY.store(content)

        if session:
            CONTENT_IDENTITY.add_ref(content_hash, "episode", f"{session}:{msg}")

        if blob_hash:
            CONTENT_IDENTITY.add_ref(content_hash, "blob", blob_hash)

        return content_hash
    except Exception:
        return None


def enrich_event(event: dict, allow_slow: bool = False) -> dict:
    """Add git info to an event.

    Args:
        event: The event dict with tool, file, cwd, etc.
        allow_slow: If True, allow slow operations (backfill mode).

    Returns:
        Enriched event dict with git info added.
    """
    file_path = event.get("file", "")
    cwd = event.get("cwd", "")
    tool = event.get("tool", "")

    # Always: get basic git info
    git_info = get_git_info(file_path, cwd, tool)
    for key, value in git_info.items():
        if value or key == "is_tracked":
            event[key] = value

    # Always: get git-registry info
    if file_path:
        registry_info = get_registry_info(file_path)
        for key, value in registry_info.items():
            if value:
                event[key] = value

    # Always: get file-identity UUID
    if file_path and FILE_IDENTITY:
        try:
            file_uuid = FILE_IDENTITY.assign(file_path)
            if file_uuid:
                event["file_uuid"] = file_uuid
        except Exception:
            pass

    # URL identity for WebFetch/WebSearch
    url = event.get("url", "")
    web_content = event.get("web_content", "")
    if url and URL_IDENTITY:
        try:
            is_search = tool == "WebSearch"
            url_uuid = URL_IDENTITY.assign(url, is_search=is_search)
            if url_uuid:
                event["url_uuid"] = url_uuid

                if web_content and tool == "WebFetch":
                    content_hash = URL_IDENTITY.record_fetch(
                        url_uuid,
                        content=web_content,
                        status_code=event.get("web_status", 200),
                        session_id=event.get("session", ""),
                        prompt=event.get("prompt", "")
                    )
                    if content_hash:
                        event["web_content_hash"] = content_hash
                    event.pop("web_content", None)
        except Exception:
            pass

    # Hybrid content hashing for Write/Edit operations
    if file_path and tool in ("Write", "Edit", "MultiEdit"):
        content_hash = get_content_hash(
            file_path,
            session=event.get("session", ""),
            msg=event.get("msg", 0),
            blob_hash=git_info.get("blob_hash", "")
        )
        if content_hash:
            event["content_hash"] = content_hash

    # Slow mode only: register unknown repos
    if allow_slow and file_path and not event.get("repo_root") and GIT_REGISTRY:
        try:
            if os.path.isfile(file_path):
                dir_path = os.path.dirname(file_path)
            else:
                dir_path = file_path

            result = subprocess.run(
                ["git", "-C", dir_path, "rev-parse", "--show-toplevel"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                repo_path = result.stdout.strip()
                repo = GIT_REGISTRY.register(repo_path)
                if repo:
                    registry_info = get_registry_info(file_path)
                    for key, value in registry_info.items():
                        if value:
                            event[key] = value
        except Exception:
            pass

    event.pop("cwd", None)
    return event


# ─────────────────────────────────────────────────────────────────────────────
# Lookup utilities
# ─────────────────────────────────────────────────────────────────────────────

def find_content_by_blob(blob_hash: str) -> Optional[str]:
    """Find content_hash from blob_hash."""
    if not CONTENT_IDENTITY or not blob_hash:
        return None
    return CONTENT_IDENTITY.find_by_ref("blob", blob_hash)


def find_episodes_by_content(content_hash: str) -> list[str]:
    """Find all episode references for a content_hash."""
    if not CONTENT_IDENTITY or not content_hash:
        return []
    refs = CONTENT_IDENTITY.get_refs(content_hash)
    return [ref_id for ref_type, ref_id in refs if ref_type == "episode"]


# ─────────────────────────────────────────────────────────────────────────────
# Image extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_tool_result_images(tools_used: str, session: str = "", msg: int = 0) -> tuple[str, list]:
    """Extract base64 images from tool_result content, store in content-store."""
    if not tools_used or not CONTENT_IDENTITY:
        return tools_used, []

    try:
        tools = json.loads(tools_used)
    except (json.JSONDecodeError, TypeError):
        return tools_used, []

    if not isinstance(tools, list):
        return tools_used, []

    image_hashes = []
    modified = False

    for tool in tools:
        if not isinstance(tool, dict):
            continue
        if 'tool_use_id' not in tool:
            continue

        content = tool.get('content')
        if not isinstance(content, list):
            continue

        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get('type') != 'image':
                continue

            source = item.get('source', {})
            if source.get('type') != 'base64':
                continue
            if 'data' not in source:
                continue

            media_type = source.get('media_type', 'image/png')
            try:
                image_bytes = base64.b64decode(source['data'])
            except Exception:
                continue

            try:
                content_hash = CONTENT_IDENTITY.store(image_bytes, mime_type=media_type)

                if session:
                    CONTENT_IDENTITY.add_ref(content_hash, "image", f"{session}:{msg}")

                image_hashes.append({
                    "hash": content_hash,
                    "media_type": media_type,
                    "size": len(image_bytes)
                })

                del source['data']
                source['content_hash'] = content_hash
                modified = True

            except Exception:
                pass

    if modified:
        return json.dumps(tools), image_hashes
    return tools_used, image_hashes
