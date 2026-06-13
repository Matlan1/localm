"""
Tests for the agent loop guards added for foundation hardening:

  - Self-verification: agent is nudged once to verify unverified code writes
    before its final answer is accepted.
  - Uncertainty escalation: non-interactive tasks that exceed the per-task
    turn budget get a [turn budget] message telling the model to surface
    blockers instead of guessing.
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from localm.plugins.coder.tools import ToolResult


def _make_agent(tmp_path: Path, **kwargs) -> object:
    from localm.plugins.coder.agent import Agent
    backend = MagicMock()
    backend.model_id = "test-model"
    backend.native_tools = False
    with patch("localm.plugins.coder.agent.ProjectMap") as MockPM, \
         patch("localm.plugins.coder.agent.make_audit_log"), \
         patch("localm.plugins.coder.agent.load_memory", return_value=""):
        MockPM.build.return_value.file_count.return_value = 0
        agent = Agent(backend=backend, cwd=tmp_path, **kwargs)
    return agent


def _make_call(name: str, **args):
    c = MagicMock()
    c.name = name
    c.args = args
    return c


# ---------------------------------------------------------------------------
#  Unverified-write tracking in _execute_tool
# ---------------------------------------------------------------------------

class TestUnverifiedWriteTracking:
    def _run_tool(self, agent, name, result=None, **args):
        call = _make_call(name, **args)
        tool_def = MagicMock()
        tool_def.destructive = False
        tool_def.fn = MagicMock(return_value=result or ToolResult.success("ok"))
        with patch.dict(
            "localm.plugins.coder.agent.TOOL_REGISTRY", {name: tool_def}
        ):
            return agent._execute_tool(call, interactive=False)

    def test_write_file_python_tracked(self, tmp_path):
        agent = _make_agent(tmp_path)
        self._run_tool(agent, "write_file", path="src/app.py", content="x = 1")
        assert "src/app.py" in agent._unverified_writes

    def test_non_code_file_not_tracked(self, tmp_path):
        agent = _make_agent(tmp_path)
        self._run_tool(agent, "write_file", path="notes.md", content="hi")
        assert agent._unverified_writes == set()

    def test_failed_write_not_tracked(self, tmp_path):
        agent = _make_agent(tmp_path)
        self._run_tool(
            agent, "write_file",
            result=ToolResult.error("disk full"),
            path="src/app.py", content="x",
        )
        assert agent._unverified_writes == set()

    def test_edit_and_patch_tracked(self, tmp_path):
        agent = _make_agent(tmp_path)
        self._run_tool(agent, "edit_file", path="a.py", old="x", new="y")
        self._run_tool(agent, "patch_file", path="b.ts", diff="--- a\n+++ b\n")
        assert agent._unverified_writes == {"a.py", "b.ts"}

    def test_run_tests_clears_tracking(self, tmp_path):
        agent = _make_agent(tmp_path)
        agent._unverified_writes = {"a.py"}
        self._run_tool(agent, "run_tests")
        assert agent._unverified_writes == set()

    def test_pytest_shell_command_clears_tracking(self, tmp_path):
        agent = _make_agent(tmp_path)
        agent._unverified_writes = {"a.py"}
        self._run_tool(agent, "run_shell", command="python -m pytest tests/")
        assert agent._unverified_writes == set()

    def test_unrelated_shell_command_keeps_tracking(self, tmp_path):
        agent = _make_agent(tmp_path)
        agent._unverified_writes = {"a.py"}
        self._run_tool(agent, "run_shell", command="git status")
        assert agent._unverified_writes == {"a.py"}

    def test_dry_run_not_tracked(self, tmp_path):
        agent = _make_agent(tmp_path, dry_run=True)
        call = _make_call("write_file", path="a.py", content="x")
        tool_def = MagicMock()
        tool_def.destructive = True
        with patch.dict(
            "localm.plugins.coder.agent.TOOL_REGISTRY", {"write_file": tool_def}
        ):
            agent._execute_tool(call, interactive=False)
        assert agent._unverified_writes == set()

    def test_reset_clears_tracking(self, tmp_path):
        agent = _make_agent(tmp_path)
        agent._unverified_writes = {"a.py"}
        agent.reset()
        assert agent._unverified_writes == set()


# ---------------------------------------------------------------------------
#  Self-verification nudge in _loop
# ---------------------------------------------------------------------------

class TestSelfVerificationNudge:
    def test_nudge_injected_before_final_answer(self, tmp_path):
        agent = _make_agent(tmp_path)
        agent._unverified_writes = {"src/app.py"}
        # Backend returns a plain final answer (no tool calls) every time
        responses = iter(["All done!", "Verified, all done!"])
        with patch.object(agent, "_call_llm", side_effect=lambda *a, **k: next(responses)), \
             patch("localm.plugins.coder.agent.parse_tool_calls", return_value=[]):
            result = agent.run_task("change something")

        assert result == "Verified, all done!"
        # The nudge message must be in history
        nudges = [
            m for m in agent._messages
            if m["role"] == "user" and "[self-verification]" in str(m.get("content", ""))
        ]
        assert len(nudges) == 1
        assert "src/app.py" in str(nudges[0]["content"])

    def test_nudge_fires_only_once(self, tmp_path):
        agent = _make_agent(tmp_path)
        agent._unverified_writes = {"a.py"}
        # Agent never verifies - second final answer must be accepted anyway
        with patch.object(agent, "_call_llm", return_value="done"), \
             patch("localm.plugins.coder.agent.parse_tool_calls", return_value=[]):
            result = agent.run_task("task")

        assert result == "done"
        nudges = [
            m for m in agent._messages
            if m["role"] == "user" and "[self-verification]" in str(m.get("content", ""))
        ]
        assert len(nudges) == 1

    def test_no_nudge_without_unverified_writes(self, tmp_path):
        agent = _make_agent(tmp_path)
        with patch.object(agent, "_call_llm", return_value="done"), \
             patch("localm.plugins.coder.agent.parse_tool_calls", return_value=[]):
            result = agent.run_task("task")
        assert result == "done"
        assert not any(
            "[self-verification]" in str(m.get("content", ""))
            for m in agent._messages
        )

    def test_no_nudge_when_disabled(self, tmp_path):
        agent = _make_agent(tmp_path, self_verify=False)
        agent._unverified_writes = {"a.py"}
        with patch.object(agent, "_call_llm", return_value="done"), \
             patch("localm.plugins.coder.agent.parse_tool_calls", return_value=[]):
            result = agent.run_task("task")
        assert result == "done"
        assert not any(
            "[self-verification]" in str(m.get("content", ""))
            for m in agent._messages
        )


# ---------------------------------------------------------------------------
#  Uncertainty escalation (turn budget)
# ---------------------------------------------------------------------------

class TestTurnBudgetEscalation:
    def test_default_budget_is_two_thirds_of_max_turns(self, tmp_path):
        agent = _make_agent(tmp_path, max_turns=30)
        assert agent.turn_budget == 20

    def test_explicit_budget_respected(self, tmp_path):
        agent = _make_agent(tmp_path, max_turns=40, turn_budget=5)
        assert agent.turn_budget == 5

    def test_non_interactive_escalation_message_injected(self, tmp_path):
        agent = _make_agent(tmp_path, max_turns=10, turn_budget=2)
        # Always return a tool call so the loop keeps spinning past the budget
        call = _make_call("read_file", path="x.py")
        with patch.object(agent, "_call_llm", return_value="<tool/>"), \
             patch("localm.plugins.coder.agent.parse_tool_calls", return_value=[call]), \
             patch.object(agent, "_execute_tools", return_value=["<result>ok</result>"]), \
             patch("localm.plugins.coder.agent.print_warning"):
            agent.run_task("endless task")

        budget_msgs = [
            m for m in agent._messages
            if m["role"] == "user" and "[turn budget]" in str(m.get("content", ""))
        ]
        assert len(budget_msgs) == 1

    def test_no_escalation_under_budget(self, tmp_path):
        agent = _make_agent(tmp_path, max_turns=10, turn_budget=5)
        with patch.object(agent, "_call_llm", return_value="done"), \
             patch("localm.plugins.coder.agent.parse_tool_calls", return_value=[]):
            agent.run_task("quick task")
        assert not any(
            "[turn budget]" in str(m.get("content", ""))
            for m in agent._messages
        )

    def test_budget_is_per_task_not_per_session(self, tmp_path):
        """Turns from a previous task must not count against the next task."""
        agent = _make_agent(tmp_path, max_turns=100, turn_budget=3)
        agent._turns = 50  # simulate a long previous session
        with patch.object(agent, "_call_llm", return_value="done"), \
             patch("localm.plugins.coder.agent.parse_tool_calls", return_value=[]):
            agent._add_user("next task")
            agent._loop(interactive=False)
        assert not any(
            "[turn budget]" in str(m.get("content", ""))
            for m in agent._messages
        )
