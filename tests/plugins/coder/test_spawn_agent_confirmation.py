# SPDX-License-Identifier: AGPL-3.0-or-later
"""spawn_agent must not let a child bypass the parent's confirmation posture.

Hardcoding the child Agent's ``auto_approve=True`` and not passing through
``dry_run``, ``always_confirm`` or ``confirm_handler`` from the parent session
means a parent that required confirmation before destructive tools
(``auto_approve=False``), was running under ``--dry-run``, or had a GUI confirm
handler wired up would still spawn a child that freely executed
write_file/edit_file/patch_file/run_shell/git_commit/git_push with zero
confirmation.

These tests use REAL parent and child Agent instances (no Agent mocking) and
drive an actual destructive tool call through the child's own ``_execute_tool``
- the exact code path ``run_task``/``_loop`` uses internally, always with
``interactive=False`` - rather than merely inspecting constructor kwargs.
"""

from unittest.mock import patch

from localm.plugins.coder.agent import Agent
from localm.plugins.coder.parser import ToolCall
from localm.plugins.coder.tools import tool_spawn_agent


class _StubBackend:
    model_id = "stub-model"
    native_tools = False

    def set_tools(self, defs):
        pass


def _write_call(rel="child_output.txt"):
    return ToolCall(name="write_file", args={"path": rel, "content": "hi"},
                    raw="", start=0, end=0)


def _spawn_and_capture_child(tmp_path, parent_kwargs):
    """Spawn a real child via tool_spawn_agent, short-circuiting run_task so
    no LLM call is needed, and return the constructed child Agent for direct
    inspection of the code path spawn_agent actually exercises."""
    parent = Agent(_StubBackend(), cwd=tmp_path, **parent_kwargs)
    captured = {}

    def _fake_run_task(self, task):
        captured["child"] = self
        return "child done"

    with patch.object(Agent, "run_task", _fake_run_task):
        result = tool_spawn_agent(tmp_path, "do work", _parent_agent=parent)
    assert result.ok, result.output
    return captured["child"]


class TestSpawnAgentConfirmationPropagation:
    def test_child_denies_destructive_tool_when_parent_requires_confirmation(self, tmp_path):
        child = _spawn_and_capture_child(tmp_path, {"auto_approve": False})
        res = child._execute_tool(_write_call(), interactive=False)
        assert not res.ok
        assert "confirmation" in res.output.lower() or "denied" in res.output.lower()
        assert not (tmp_path / "child_output.txt").exists()

    def test_child_respects_parent_dry_run(self, tmp_path):
        child = _spawn_and_capture_child(tmp_path, {"dry_run": True})
        res = child._execute_tool(_write_call(), interactive=False)
        assert res.ok
        assert "dry-run" in res.output.lower()
        assert not (tmp_path / "child_output.txt").exists()

    def test_child_uses_parent_confirm_handler(self, tmp_path):
        seen = []

        def deny(call):
            seen.append(call.name)
            return False

        child = _spawn_and_capture_child(
            tmp_path, {"auto_approve": False, "confirm_handler": deny})
        res = child._execute_tool(_write_call(), interactive=False)
        assert not res.ok
        assert seen == ["write_file"]

    def test_child_inherits_always_confirm(self, tmp_path):
        child = _spawn_and_capture_child(
            tmp_path, {"auto_approve": True, "always_confirm": {"write_file"}})
        res = child._execute_tool(_write_call(), interactive=False)
        assert not res.ok

    def test_baseline_child_still_auto_approves_when_parent_does(self, tmp_path):
        # A parent with auto_approve=True spawns a child that executes destructive
        # tools without a prompt.
        child = _spawn_and_capture_child(tmp_path, {"auto_approve": True})
        res = child._execute_tool(_write_call(), interactive=False)
        assert res.ok, res.output
        assert (tmp_path / "child_output.txt").exists()
