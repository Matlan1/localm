# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Tests for the agent loop guards added for foundation hardening:

  - Self-verification: agent is nudged once to verify unverified code writes
    before its final answer is accepted.
  - Uncertainty escalation: non-interactive tasks that exceed the per-task
    turn budget get a [turn budget] message telling the model to surface
    blockers instead of guessing.
"""

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
#  Tool-call repair turn (parser runs for real here)
# ---------------------------------------------------------------------------

class TestRepairTurn:
    def _repairs(self, agent):
        return [
            m for m in agent._messages
            if m["role"] == "user" and "[tool-call format]" in str(m.get("content", ""))
        ]

    def test_repair_fires_then_accepts_reformatted_answer(self, tmp_path):
        agent = _make_agent(tmp_path)
        # First: a truncated tool call that cannot parse but clearly tried.
        # Then: a plain final answer.
        responses = iter(['{"name": "read_file", "args": {"path"', "Done."])
        with patch.object(agent, "_call_llm",
                          side_effect=lambda *a, **k: next(responses)):
            result = agent.run_task("read a file")
        assert result == "Done."
        assert len(self._repairs(agent)) == 1

    def test_repair_capped_then_surfaces_raw_attempt(self, tmp_path):
        agent = _make_agent(tmp_path)
        # The model keeps emitting the same unparseable attempt: repair fires up to
        # the cap (no infinite loop), then the raw attempt is SURFACED rather than
        # silently finalised as a hidden tool-call block (the old empty-bubble no-op).
        raw = '{"name": "read_file", "args": {"path"'
        with patch.object(agent, "_call_llm", return_value=raw):
            result = agent.run_task("task")
        assert len(self._repairs(agent)) == 2          # capped at _MAX_TOOL_REPAIRS
        assert "could not produce valid tool-call JSON" in result
        assert raw in result                           # the raw attempt is not lost

    def test_no_repair_on_plain_answer(self, tmp_path):
        agent = _make_agent(tmp_path)
        with patch.object(agent, "_call_llm", return_value="Here is the answer."):
            result = agent.run_task("task")
        assert result == "Here is the answer."
        assert self._repairs(agent) == []

    def test_bare_json_call_parses_without_repair(self, tmp_path):
        """A well-formed bare JSON call now parses and runs - no repair turn."""
        agent = _make_agent(tmp_path)
        responses = iter([
            '{"name": "read_file", "args": {"path": "a.py"}}',
            "Done.",
        ])
        with patch.object(agent, "_call_llm",
                          side_effect=lambda *a, **k: next(responses)), \
             patch.object(agent, "_execute_tools",
                          return_value=["<result>ok</result>"]) as ex:
            result = agent.run_task("read a.py")
        assert result == "Done."
        ex.assert_called_once()          # the bare JSON was dispatched as a call
        assert self._repairs(agent) == []


class TestNoProgressBreaker:
    """Global no-progress breaker: many VARIED failing tool calls (which the
    per-tool 4-identical breaker misses) must trip the breaker so a weak model
    cannot spin and burn the whole budget."""

    def test_varied_failures_trip_global_breaker(self, tmp_path):
        agent = _make_agent(tmp_path)
        # 6 failing calls, no single tool reaching 4 (read_file x3, list_dir x3).
        failing = [
            _make_call("read_file", path="missing1.txt"),
            _make_call("read_file", path="missing2.txt"),
            _make_call("list_dir",  path="nope_dir1"),
            _make_call("list_dir",  path="nope_dir2"),
            _make_call("read_file", path="missing3.txt"),
            _make_call("list_dir",  path="nope_dir3"),
        ]
        for c in failing:
            r = agent._execute_tool(c, interactive=False)
            assert not r.ok                          # each call really failed
        assert agent._abort_no_progress is True       # global breaker tripped
        assert agent._abort_streak_tool is None       # per-tool breaker did NOT

    def test_success_resets_global_streak(self, tmp_path):
        agent = _make_agent(tmp_path)
        (tmp_path / "real.txt").write_text("hi", encoding="utf-8")
        agent._global_error_streak = 4
        r = agent._execute_tool(_make_call("read_file", path="real.txt"),
                                interactive=False)
        assert r.ok
        assert agent._global_error_streak == 0


class TestDormantToolGrammar:
    """The TOOL_CALLS_ONLY grammar is WIRED but dormant: off unless the config flag
    is set AND the backend supports grammar."""

    def test_dormant_by_default(self, tmp_path):
        agent = _make_agent(tmp_path)
        agent.backend.supports_grammar = True
        assert agent._tool_call_grammar() is None        # flag off by default
        assert "grammar" not in agent._llm_kwargs()

    def test_active_when_flag_on_and_supported(self, tmp_path, monkeypatch):
        agent = _make_agent(tmp_path)
        agent.backend.supports_grammar = True
        monkeypatch.setattr("localm.config.load_config",
                            lambda: {"coder_tool_grammar": True})
        g = agent._tool_call_grammar()
        assert g is not None and "tool_call" in g
        assert agent._llm_kwargs().get("grammar") == g

    def test_off_when_backend_unsupported(self, tmp_path, monkeypatch):
        agent = _make_agent(tmp_path)
        agent.backend.supports_grammar = False
        monkeypatch.setattr("localm.config.load_config",
                            lambda: {"coder_tool_grammar": True})
        assert agent._tool_call_grammar() is None


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
