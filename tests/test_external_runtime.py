"""Container/runtime-ownership contract for module installers."""
from __future__ import annotations

from pathlib import Path

import pytest

from flex import cli

pytestmark = pytest.mark.unit


def test_runtime_setup_defaults_to_native_owner(monkeypatch):
    monkeypatch.delenv("FLEX_RUNTIME_OWNER", raising=False)
    assert cli._runtime_setup_enabled() is True


@pytest.mark.parametrize("value", ["external", "EXTERNAL", " external "])
def test_external_runtime_owner_disables_machine_setup(monkeypatch, value):
    monkeypatch.setenv("FLEX_RUNTIME_OWNER", value)
    assert cli._runtime_setup_enabled() is False


def test_unknown_runtime_owner_preserves_native_behavior(monkeypatch):
    monkeypatch.setenv("FLEX_RUNTIME_OWNER", "something-else")
    assert cli._runtime_setup_enabled() is True


def test_external_init_does_not_stop_existing_runtime(monkeypatch):
    calls = []
    monkeypatch.setenv("FLEX_RUNTIME_OWNER", "external")
    monkeypatch.setattr(cli, "_kill_pid_services", lambda: calls.append("kill"))
    cli._quiesce_runtime_for_init("claude-code")
    assert calls == []


def test_native_init_still_quiesces_existing_runtime(monkeypatch):
    calls = []
    monkeypatch.delenv("FLEX_RUNTIME_OWNER", raising=False)
    monkeypatch.setattr(cli, "_kill_pid_services", lambda: calls.append("kill"))
    cli._quiesce_runtime_for_init("claude-code")
    assert calls == ["kill"]


def test_every_legacy_runtime_owner_honors_external_contract():
    root = Path(__file__).resolve().parents[1]
    owners = [
        "flex/modules/claude_code/install.py",
        "flex/modules/fs/install.py",
        "flex/modules/hn/install.py",
        "flex/modules/markdown/install.py",
        "flex/modules/reddit/install.py",
    ]
    for relative in owners:
        source = (root / relative).read_text(encoding="utf-8")
        assert "_runtime_setup_enabled" in source, relative
        assert "externally managed" in source, relative
