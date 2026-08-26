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

import pytest

from localm.plugins.coder.tools import ToolResult
from tests.conftest import final_answer as _final_answer


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
    # start/end are int text offsets; 0/0 means the whole response is trailing text.
    c.start = 0
    c.end = 0
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

        assert _final_answer(result) == "Verified, all done!"
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

        assert _final_answer(result) == "done"
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
        assert _final_answer(result) == "done"
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
        assert _final_answer(result) == "done"
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
        """First a truncated call that cannot parse but clearly tried, so the
        repair re-prompt fires; then the model RE-EMITS it correctly and that
        call runs; then a plain final answer, which is accepted.

        The middle step is load-bearing: without a scripted reformatted call
        the model never calls a tool at all on a "read a file" request, which
        the loop escalates rather than accepting in silence."""
        agent = _make_agent(tmp_path)
        (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
        responses = iter([
            '{"name": "read_file", "args": {"path"',
            '<tool_call>\n{"name": "read_file", "args": {"path": "a.py"}}\n</tool_call>',
            "Done.",
        ])
        with patch.object(agent, "_call_llm",
                          side_effect=lambda *a, **k: next(responses)):
            result = agent.run_task("read a file")
        assert _final_answer(result) == "Done."
        assert len(self._repairs(agent)) == 1

    def test_repair_capped_then_surfaces_raw_attempt(self, tmp_path):
        agent = _make_agent(tmp_path)
        # The model keeps emitting the same unparseable attempt: repair fires up to
        # the cap, then the raw attempt is surfaced.
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
        assert _final_answer(result) == "Here is the answer."
        assert self._repairs(agent) == []

    def test_hallucinated_xml_tool_tag_repairs_instead_of_silently_finishing(self, tmp_path):
        """The model hallucinates <edit_file>/<read_file path="..."> tags using
        this project's REAL tool names, instead of its own
        <tool_call>{"name":...} wrapper. parse_tool_calls() recovers nothing,
        and a looks_like_tool_attempt() without tool_names also misses it: that
        form checks only for the literal "tool_call"/"tool_code" markers and a
        "name"+"args" JSON key pair, and this response has neither. Such a
        response must NOT fall through _handle_no_tool_calls's final branch and
        be accepted as the finished answer; it is recognised via the
        tool-name-tagged form and routed through the SAME repair-then-surface
        path proven above, so the user is told plainly that nothing was run or
        written."""
        agent = _make_agent(tmp_path)
        raw = (
            '<edit_file>\n{"path": "sample.py", "old": "def add(a, b): pass", '
            '"new": "..."}\n\nLet me verify the file exists before making changes:'
            '\n\n<read_file path="sample.py">\n\nPlease confirm if add is in this '
            "file or provide more context about what you're trying to modify."
        )
        with patch.object(agent, "_call_llm", return_value=raw):
            result = agent.run_task("Add a docstring to the add function in sample.py")
        assert len(self._repairs(agent)) == 2          # capped at _MAX_TOOL_REPAIRS
        assert "could not produce valid tool-call JSON" in result
        assert "nothing was run or written" in result
        assert raw in result                           # the raw attempt is not lost

    def test_bare_json_call_parses_without_repair(self, tmp_path):
        """A well-formed bare JSON call parses and runs, with no repair turn."""
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
        assert _final_answer(result) == "Done."
        ex.assert_called_once()          # the bare JSON was dispatched as a call
        assert self._repairs(agent) == []


class TestPartialParseSurfacing:
    """A tool-call-shaped block that fails to parse alongside a SIBLING call
    that parses fine must not vanish silently.

    parse_tool_calls only signals a parse problem via an EMPTY calls list
    (which routes to TestRepairTurn's mechanism above) - it has no way to say
    "N calls were attempted, only N-1 recovered", so an edit_file that fails to
    parse alongside a sibling write_file and run_shell that execute for real
    would vanish with nothing in the session recording it. loop.py checks the
    text NOT consumed by the calls that did parse (split_response) for a
    leftover tool-call-shaped fragment and, if found, appends a notice to the
    results fed back.

    The first <tool_call> below brace-balances fine, so it is NOT a
    pairing/scan failure (that class is recovered by the marker-variant
    fallback), but it is semantically rejected by _try_parse_body (name=123 is
    not a string), which no pass in parser.py can recover - a stable,
    fix-independent way to force exactly one call in a batch to be
    unrecoverable.
    """
    def _partial_notices(self, agent):
        return [
            m for m in agent._messages
            if m["role"] == "user"
            and "Part of this response looked like another tool call" in str(m.get("content", ""))
        ]

    def _cap_announcements(self, agent):
        return [
            m for m in agent._messages
            if m["role"] == "user"
            and "further occurrences will not be reported individually" in str(m.get("content", ""))
        ]

    def test_sibling_success_does_not_hide_an_unrecoverable_call(self, tmp_path):
        agent = _make_agent(tmp_path)
        broken_and_good = (
            '<tool_call>\n{"name": 123, "args": {"path": "a.py"}}\n</tool_call>\n\n'
            '<tool_call>\n{"name": "read_file", "args": {"path": "b.py"}}\n</tool_call>\n'
        )
        responses = iter([broken_and_good, "Done."])
        with patch.object(agent, "_call_llm",
                          side_effect=lambda *a, **k: next(responses)), \
             patch.object(agent, "_execute_tools",
                          return_value=["<result>ok</result>"]) as ex:
            result = agent.run_task("read two files")
        assert _final_answer(result) == "Done."
        ex.assert_called_once()
        (dispatched,), _ = ex.call_args
        assert len(dispatched) == 1
        assert dispatched[0].name == "read_file"
        assert dispatched[0].args["path"] == "b.py"
        assert len(self._partial_notices(agent)) == 1

    def test_partial_notice_is_capped_not_repeated_every_turn(self, tmp_path):
        """The notice's own example text is itself tool-call-shaped (a literal
        <tool_call> block with "name"/"args" keys) - a model that echoes it
        back as commentary while also making one real call each turn would
        otherwise re-trigger this notice forever, once per turn. Capped at
        _MAX_TOOL_REPAIRS, the same bound and the same reasoning as the
        repair-turn mechanism above."""
        from localm.plugins.coder.agent.constants import _MAX_TOOL_REPAIRS

        def _broken_and_good(i):
            # Varied per turn: an identical response would trip _REPEAT_RESPONSE_ABORT.
            return (
                f'<tool_call>\n{{"name": 123, "args": {{"path": "a{i}.py"}}}}\n</tool_call>\n\n'
                f'<tool_call>\n{{"name": "read_file", "args": {{"path": "b{i}.py"}}}}\n</tool_call>\n'
            )

        # More turns of the same shape than the cap allows, then a plain final answer.
        responses = iter([_broken_and_good(i) for i in range(_MAX_TOOL_REPAIRS + 3)]
                         + ["Done."])
        agent = _make_agent(tmp_path, max_turns=_MAX_TOOL_REPAIRS + 5)
        with patch.object(agent, "_call_llm",
                          side_effect=lambda *a, **k: next(responses)), \
             patch.object(agent, "_execute_tools",
                          # A fresh list per call; loop.py appends the notice to it in place.
                          side_effect=lambda *a, **k: ["<result>ok</result>"]), \
             patch.object(agent, "_maybe_compact"):
            result = agent.run_task("read two files repeatedly")
        assert _final_answer(result) == "Done."
        assert len(self._partial_notices(agent)) == _MAX_TOOL_REPAIRS
        # Exactly one final notice announces that further occurrences are not
        # individually reported.
        assert len(self._cap_announcements(agent)) == 1

    def test_partial_notice_cap_logs_every_drop_even_once_silent_to_the_model(self, tmp_path):
        """The turns AFTER the one-time cap announcement must still leave a
        durable trace: going fully silent (nothing in the fed-back message,
        nothing logged) would be an invisible drop inside the very mechanism
        built to surface one."""
        from localm.plugins.coder.agent.constants import _MAX_TOOL_REPAIRS

        def _broken_and_good(i):
            return (
                f'<tool_call>\n{{"name": 123, "args": {{"path": "a{i}.py"}}}}\n</tool_call>\n\n'
                f'<tool_call>\n{{"name": "read_file", "args": {{"path": "b{i}.py"}}}}\n</tool_call>\n'
            )

        n_turns = _MAX_TOOL_REPAIRS + 3   # more than the cap
        responses = iter([_broken_and_good(i) for i in range(n_turns)] + ["Done."])
        agent = _make_agent(tmp_path, max_turns=n_turns + 2)
        debug_calls = []
        with patch.object(agent, "_call_llm",
                          side_effect=lambda *a, **k: next(responses)), \
             patch.object(agent, "_execute_tools",
                          side_effect=lambda *a, **k: ["<result>ok</result>"]), \
             patch.object(agent, "_maybe_compact"), \
             patch("localm.debuglog.logger.debug",
                   side_effect=lambda *a, **k: debug_calls.append(a)):
            result = agent.run_task("read two files repeatedly")
        assert _final_answer(result) == "Done."
        # Filtered to the per-drop trace message specifically, not any debug call.
        per_drop_traces = [
            a for a in debug_calls
            if "notice cap already reached" in str(a[0])
        ]
        still_silent_turns = n_turns - _MAX_TOOL_REPAIRS - 1
        assert len(per_drop_traces) == still_silent_turns, (
            f"expected exactly {still_silent_turns} per-drop debug trace(s) "
            f"for the drops that stopped being reported to the model, got "
            f"{len(per_drop_traces)} (out of {len(debug_calls)} total debug calls)")

    def test_no_partial_notice_when_everything_parsed(self, tmp_path):
        agent = _make_agent(tmp_path)
        responses = iter([
            '<tool_call>\n{"name": "read_file", "args": {"path": "a.py"}}\n</tool_call>\n',
            "Done.",
        ])
        with patch.object(agent, "_call_llm",
                          side_effect=lambda *a, **k: next(responses)), \
             patch.object(agent, "_execute_tools",
                          return_value=["<result>ok</result>"]):
            result = agent.run_task("read a file")
        assert _final_answer(result) == "Done."
        assert self._partial_notices(agent) == []

    def test_no_partial_notice_when_leftover_is_plain_prose(self, tmp_path):
        agent = _make_agent(tmp_path)
        responses = iter([
            'Sure, reading it now.\n\n'
            '<tool_call>\n{"name": "read_file", "args": {"path": "a.py"}}\n</tool_call>\n'
            '\n\nThat should tell us what we need.',
            "Done.",
        ])
        with patch.object(agent, "_call_llm",
                          side_effect=lambda *a, **k: next(responses)), \
             patch.object(agent, "_execute_tools",
                          return_value=["<result>ok</result>"]):
            result = agent.run_task("read a file")
        assert _final_answer(result) == "Done."
        assert self._partial_notices(agent) == []


class TestGroundingFooter:
    """The final answer must carry a factual "what actually happened" line
    from the session's own record, unconditionally - never gated on what the
    response text itself claims.

    The footer grounds on the observable artifact (a diff, an exit code)
    rather than the model's own self-report: a model confirming its own claim,
    or a keyword scan over its prose, inherits the same unreliability being
    guarded against. It reuses changed_files() and _last_verify_state (already
    tracked for the self-verify nudge and the exit-code oracle) and never reads
    the response text, so it cannot be gamed by phrasing.
    """

    def test_footer_reports_no_files_changed_when_nothing_was_written(self, tmp_path):
        agent = _make_agent(tmp_path)
        with patch.object(agent, "_call_llm", return_value="Here is the answer."):
            result = agent.run_task("what does this function do?")
        assert result == (
            "Here is the answer.\n\n[session record: no files changed]")

    def test_footer_reports_the_real_changed_files(self, tmp_path):
        # self_verify=False keeps the self-verify-nudge turn out of this run.
        agent = _make_agent(tmp_path, auto_approve=True, self_verify=False)
        responses = iter([
            '<tool_call>\n{"name": "write_file", "args": '
            '{"path": "app.py", "content": "x = 1\\n"}}\n</tool_call>\n',
            "Done.",
        ])
        with patch.object(agent, "_call_llm",
                          side_effect=lambda *a, **k: next(responses)):
            result = agent.run_task("write app.py")
        assert result == (
            "Done.\n\n[session record: 1 file(s) changed: app.py]")
        assert (tmp_path / "app.py").read_text(encoding="utf-8") == "x = 1\n"

    def test_footer_reports_search_replace_changes(self, tmp_path):
        """A task that uses ONLY search_replace to edit a real file must not
        leave changed_files() - and so this footer - empty. search_replace's
        targets are a glob+regex sweep rather than a `path` arg the pre-call
        snapshot tracker can see, so the changes are carried on
        ToolResult.changes (see execution.py's search_replace branch in
        _post_tool_success). A false "no files changed" is the under-claiming
        mirror of the over-claiming the footer exists to catch."""
        (tmp_path / "app.py").write_text("x = 1\n", encoding="utf-8")
        agent = _make_agent(tmp_path, auto_approve=True, self_verify=False)
        responses = iter([
            '<tool_call>\n{"name": "search_replace", "args": '
            '{"pattern": "x = 1", "replacement": "x = 2", "glob": "*.py"}}'
            '\n</tool_call>\n',
            "Done.",
        ])
        with patch.object(agent, "_call_llm",
                          side_effect=lambda *a, **k: next(responses)):
            result = agent.run_task("bump the constant")
        assert result == (
            "Done.\n\n[session record: 1 file(s) changed: app.py]")
        assert (tmp_path / "app.py").read_text(encoding="utf-8") == "x = 2\n"

    def test_footer_names_multiple_changed_files_sorted(self, tmp_path):
        """The single-file test above cannot catch a count/join/sort bug -
        "1 file(s)" is right whether or not pluralization, joining, and
        sorting are implemented at all. Writes b.py then a.py, in that
        order, so a footer that just echoed write order (instead of
        sorting) would also be caught."""
        agent = _make_agent(tmp_path, auto_approve=True, self_verify=False)
        responses = iter([
            '<tool_call>\n{"name": "write_file", "args": '
            '{"path": "b.py", "content": "b = 1\\n"}}\n</tool_call>\n',
            '<tool_call>\n{"name": "write_file", "args": '
            '{"path": "a.py", "content": "a = 1\\n"}}\n</tool_call>\n',
            "Done.",
        ])
        with patch.object(agent, "_call_llm",
                          side_effect=lambda *a, **k: next(responses)):
            result = agent.run_task("write two files")
        assert result == (
            "Done.\n\n[session record: 2 file(s) changed: a.py, b.py]")

    def test_footer_not_stored_in_assistant_history(self, tmp_path):
        """The stored conversation history must stay an ACCURATE record of
        what the model actually said - the footer is harness commentary
        appended to what is shown/returned, not something the model said,
        and must not be replayed back to the model as its own prior turn on
        a later call in the same session.

        Asserts BOTH halves, not just the negative one: the footer IS in the
        returned text (proves the feature exists and this test would fail if
        it silently stopped appending anything at all) AND is NOT in stored
        history (the actual property under test). A test that only checked
        history-is-clean would pass identically whether the footer feature
        works, is broken, or was never implemented - which is exactly what
        a footer feature that works, is broken, or was never implemented."""
        agent = _make_agent(tmp_path)
        with patch.object(agent, "_call_llm", return_value="Here is the answer."):
            result = agent.run_task("what does this function do?")
        assert "[session record:" in result   # the positive half
        assistant_msgs = [m for m in agent._messages if m["role"] == "assistant"]
        assert len(assistant_msgs) == 1
        assert assistant_msgs[0]["content"] == "Here is the answer."
        assert "session record" not in assistant_msgs[0]["content"]   # the negative half

    def test_footer_includes_verify_state_when_verify_cmd_configured(self, tmp_path):
        """_last_verify_state is reset at the start of every _loop() call (it
        is per-RUN: "this run has not been verified until its own gate says
        so"), so it cannot be pre-set before run_task() and
        must instead be earned for real within the same run: write a file
        (so _write_total() moves), then let the REAL exit-code oracle run a
        trivially-passing command on the next no-tool-calls turn - the exit-
        code oracle runs BEFORE the self-verify nudge, so no self_verify=False
        is needed here."""
        import sys
        agent = _make_agent(tmp_path, auto_approve=True)
        agent.verify_cmd = [sys.executable, "-c", "pass"]
        responses = iter([
            '<tool_call>\n{"name": "write_file", "args": '
            '{"path": "app.py", "content": "x = 1\\n"}}\n</tool_call>\n',
            "Done.",
        ])
        with patch.object(agent, "_call_llm",
                          side_effect=lambda *a, **k: next(responses)):
            result = agent.run_task("write and verify app.py")
        assert result == (
            "Done.\n\n[session record: 1 file(s) changed: app.py; verify: passed]")
        assert agent._last_verify_state == "passed"

    def test_footer_reports_an_inconclusive_verify_state(self, tmp_path):
        """The passing-case test above cannot catch a footer that always
        prints "passed" regardless of the real oracle result - inconclusive
        is a materially different code path (a command that never started).
        A nonexistent executable is the simplest way to earn a REAL
        "inconclusive" verdict without a real failing check.

        self_verify=False for the same reason as the multi-file test above,
        but load-bearing here in a way it wasn't there: unlike a PASS
        (loop.py clears _unverified_writes only on code == 0), an
        inconclusive verdict leaves _unverified_writes populated, so
        without this the self-verify nudge gate would also fire and this
        script would need a third scripted turn to reach the footer."""
        agent = _make_agent(tmp_path, auto_approve=True, self_verify=False)
        agent.verify_cmd = ["definitely-not-a-real-executable-xyz"]
        responses = iter([
            '<tool_call>\n{"name": "write_file", "args": '
            '{"path": "app.py", "content": "x = 1\\n"}}\n</tool_call>\n',
            "Done.",
        ])
        with patch.object(agent, "_call_llm",
                          side_effect=lambda *a, **k: next(responses)):
            result = agent.run_task("write app.py")
        assert result == (
            "Done.\n\n[session record: 1 file(s) changed: app.py; "
            "verify: inconclusive]")
        assert agent._last_verify_state == "inconclusive"

    def test_a_failed_verify_still_gets_the_footer(self, tmp_path):
        """_run_verify_gate's exhausted-retries branch (loop.py, the "Retries
        exhausted" block) returns its own explicit "[verification FAILED] ...
        NOT verified" notice, bypassing _handle_no_tool_calls's later
        fall-through, so it must append the footer itself or a verified FAILURE
        would be the one case with no session record. _last_verify_state is
        already "failed" by this point, so the SAME _grounding_footer() call
        includes "verify: failed"."""
        import sys
        agent = _make_agent(tmp_path, auto_approve=True, verify_max_retries=0)
        agent.verify_cmd = [sys.executable, "-c", "import sys; sys.exit(1)"]
        responses = iter([
            '<tool_call>\n{"name": "write_file", "args": '
            '{"path": "app.py", "content": "x = 1\\n"}}\n</tool_call>\n',
            # A second turn is required: verification runs in _handle_no_tool_calls,
            # which only sees a turn after the write.
            "Done.",
        ])
        with patch.object(agent, "_call_llm",
                          side_effect=lambda *a, **k: next(responses)):
            result = agent.run_task("write app.py")
        assert "[verification FAILED]" in result
        assert result.endswith(
            "[session record: 1 file(s) changed: app.py; verify: failed]")
        assert agent._last_verify_state == "failed"

    def test_no_verify_state_omitted_when_verify_cmd_not_configured(self, tmp_path):
        agent = _make_agent(tmp_path)
        assert agent.verify_cmd is None
        with patch.object(agent, "_call_llm", return_value="Done."):
            result = agent.run_task("task")
        assert result == "Done.\n\n[session record: no files changed]"
        assert "verify" not in result


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


class TestLazyToolGrammar:
    """The LAZY tool-call grammar is ON by default: free text and thinking
    flow, and a started <tool_call> must be valid. Off when the user disables
    the flag or the backend cannot enforce grammar."""

    def test_on_by_default_when_supported(self, tmp_path, monkeypatch):
        agent = _make_agent(tmp_path)
        agent.backend.supports_grammar = True
        monkeypatch.setattr("localm.config.load_config", lambda: {})
        pair = agent._tool_call_grammar()
        assert pair is not None
        grammar, triggers = pair
        assert "tool_call" in grammar
        assert triggers and "<tool_call>" in triggers[0]
        kw = agent._llm_kwargs()
        assert kw.get("grammar") == grammar
        assert kw.get("grammar_lazy") is True
        assert kw.get("grammar_triggers") == triggers

    def test_off_when_flag_disabled(self, tmp_path, monkeypatch):
        agent = _make_agent(tmp_path)
        agent.backend.supports_grammar = True
        monkeypatch.setattr("localm.config.load_config",
                            lambda: {"coder_tool_grammar": False})
        assert agent._tool_call_grammar() is None
        kw = agent._llm_kwargs()
        assert "grammar" not in kw and "grammar_lazy" not in kw

    def test_off_when_backend_unsupported(self, tmp_path, monkeypatch):
        agent = _make_agent(tmp_path)
        agent.backend.supports_grammar = False
        monkeypatch.setattr("localm.config.load_config",
                            lambda: {"coder_tool_grammar": True})
        assert agent._tool_call_grammar() is None

    @pytest.mark.parametrize("mode", ["event_sink", "interactive", "silent"])
    def test_all_three_llm_dispatch_branches_apply_the_grammar(
            self, tmp_path, monkeypatch, mode):
        agent = _make_agent(tmp_path)
        agent.backend.supports_grammar = True
        monkeypatch.setattr("localm.config.load_config", lambda: {})
        agent.backend.chat_stream.return_value = iter([])
        agent.backend.chat.return_value = "done"

        if mode == "event_sink":
            agent.on_event = lambda *a, **k: None
            agent._call_llm([{"role": "user", "content": "hi"}], interactive=False)
            kw = agent.backend.chat_stream.call_args.kwargs
        elif mode == "interactive":
            agent.on_event = None
            agent._call_llm([{"role": "user", "content": "hi"}], interactive=True)
            kw = agent.backend.chat_stream.call_args.kwargs
        else:
            agent.on_event = None
            agent._call_llm([{"role": "user", "content": "hi"}], interactive=False)
            kw = agent.backend.chat.call_args.kwargs

        assert kw.get("grammar_lazy") is True, f"{mode} branch must carry the grammar"
        assert "tool_call" in kw.get("grammar", "")
        assert kw.get("grammar_triggers")


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


class TestEventSinkFailureIsVisible:
    """A sink that raises must not kill the agent loop, and the event must not
    vanish in silence either: a consumer wired up before it is ready would drop
    every event with nothing left in the log to find it by."""

    @staticmethod
    def _agent_with_broken_sink(tmp_path):
        # Set the sink after construction so constructor startup notices are excluded.
        agent = _make_agent(tmp_path)
        agent.on_event = MagicMock(side_effect=RuntimeError("sink not ready"))
        return agent

    def test_a_raising_sink_does_not_propagate(self, tmp_path):
        agent = self._agent_with_broken_sink(tmp_path)
        agent._emit("info", text="hello")        # must not raise
        assert agent.on_event.called

    def test_a_raising_sink_leaves_a_debug_trace(self, tmp_path):
        agent = self._agent_with_broken_sink(tmp_path)
        with patch("localm.debuglog.logger") as logger:
            agent._emit("info", text="hello")
        assert logger.debug.called, "a dropped event left no trace at all"
        logged = " ".join(str(a)
                          for call in logger.debug.call_args_list
                          for a in call.args)
        assert "info" in logged and "sink not ready" in logged, logged

    def test_a_working_sink_is_not_reported_as_a_failure(self, tmp_path):
        """Control: the debug line marks a real drop, not every emit."""
        agent = _make_agent(tmp_path)
        agent.on_event = MagicMock()
        with patch("localm.debuglog.logger") as logger:
            agent._emit("info", text="hello")
        assert agent.on_event.called
        assert not logger.debug.called
