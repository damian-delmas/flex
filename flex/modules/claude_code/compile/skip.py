"""
Skip logic for agentdb events.

Shared between hook and backfill to ensure consistent filtering.
"""

import re
from typing import Optional


# Paths to skip
SKIP_PATH_PATTERNS = [
    r'^/tmp/',
    r'node_modules',
    r'\.git/',
    r'__pycache__',
    r'\.pyc$',
    # Infrastructure paths that appear across virtually all sessions.
    # Nexus knowledge injection and Claude Code hook scripts carry no project
    # signal — indexing them pollutes project attribution and file graphs.
    r'/\.nexus/',
    r'/\.claude/hooks/',
]

# Noisy bash commands to skip
SKIP_BASH_PATTERNS = [
    r'^(ls|pwd|which|type|file|stat|echo|cd)( |$)',
    r'^(cat|head|tail) ',  # Should use Read tool
]


def should_skip_file(file_path: Optional[str]) -> bool:
    """Check if a file path should be skipped.

    Args:
        file_path: The file path to check.

    Returns:
        True if the file should be skipped.
    """
    if not file_path:
        return False

    for pattern in SKIP_PATH_PATTERNS:
        if re.search(pattern, file_path):
            return True

    return False


def should_skip_bash(command: Optional[str]) -> bool:
    """Check if a bash command should be skipped.

    Args:
        command: The bash command to check.

    Returns:
        True if the command should be skipped.
    """
    if not command:
        return False

    for pattern in SKIP_BASH_PATTERNS:
        if re.search(pattern, command):
            return True

    return False


def should_skip_event(event: dict) -> bool:
    """Check if an entire event should be skipped.

    Args:
        event: The event dict with tool, file, command, etc.

    Returns:
        True if the event should be skipped.
    """
    tool = event.get("tool", "")
    file_path = event.get("file", "")
    command = event.get("command", "")

    # Skip temp/internal files
    if should_skip_file(file_path):
        return True

    # Skip noisy bash commands
    if tool == "Bash" and should_skip_bash(command):
        return True

    return False
