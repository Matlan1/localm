# SPDX-License-Identifier: AGPL-3.0-or-later
"""The exit-code oracle in interactive REPL/GUI sessions.

Goal mode's judge is the HARNESS running a command, with its exit code deciding.
These tests cover it at the pre-done boundary of an interactive session, plus
the project-check auto-detection that supplies the command when the user gives
none.

The verification commands here are REAL subprocesses with real exit codes; only
the LLM backend is scripted.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from localm.plugins.coder import verify


# --------------------------------------------------------------------------- #
#  Helpers                                                                     #
# --------------------------------------------------------------------------- #

class _ScriptedBackend:
    """Yields a scripted response per LLM turn; the last one repeats forever."""

    model_id = "test-model"
    native_tools = False
    supports_grammar = False

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0
        self.last_usage: dict = {}
        self.last_reasoning = ""

    def _next(self) -> str:
        idx = min(self.calls, len(self._responses) - 1)
        self.calls += 1
        return self._responses[idx]

    def chat_stream(self, messages, on_reasoning=None, **kwargs):
        yield self._next()

    def chat(self, messages, **kwargs):
        return self._next()

    def set_tools(self, tool_defs):
        pass

    def context_capacity(self):
        return None


def _make_agent(tmp_path: Path, responses=("done",), **kwargs):
    from localm.plugins.coder.agent import Agent
    with patch("localm.plugins.coder.agent.ProjectMap") as MockPM, \
         patch("localm.plugins.coder.agent.make_audit_log"), \
         patch("localm.plugins.coder.agent.load_memory", return_value=""):
        MockPM.build.return_value.file_count.return_value = 0
        return Agent(backend=_ScriptedBackend(responses), cwd=tmp_path, **kwargs)


def _fresh_state(agent, verify_checked_at=0):
    """The per-task state namespace _loop builds, for direct gate tests."""
    return SimpleNamespace(verify_nudged=False, review_done=False, repair_count=0,
                           verify_retries=0, verify_settled=False,
                           verify_checked_at=verify_checked_at)


def _touch_cmd(marker: Path):
    """An argv command that creates *marker* - a probe for "did the check run?"
    that needs no shell quoting on either platform."""
    return [sys.executable, "-c", f"open(r'{marker}', 'w').close()"]


def _record_write(agent, path="mod.py", writes=1):
    agent._changed_files[path] = {"original": None, "writes": writes,
                                  "last_tool": "write_file"}


def _tool_call(name, **args):
    return ("<tool_call>\n"
            + json.dumps({"name": name, "args": args})
            + "\n</tool_call>")


_FAKE_BIN = "/fake/bin/%s"


@pytest.fixture
def runners_installed(monkeypatch):
    """Pretend every test runner is installed, at a stable fake path.

    Detection gates on the runner actually resolving and carries its resolved
    path, so without this the branch-order tests would assert one thing on a box
    with cargo and another on a box without. Availability itself is the subject
    of TestDetectionConfirmsTheRunnerCanRun; these tests are about which branch
    wins and what shape it returns."""
    monkeypatch.setattr(
        "localm.plugins.coder.tools.shell.resolve_runner",
        lambda name: _FAKE_BIN % name)


# --------------------------------------------------------------------------- #
#  run_verify: the primitive                                                   #
# --------------------------------------------------------------------------- #

class TestRunVerify:
    def test_shell_string_reports_exit_code(self, tmp_path):
        assert verify.run_verify("exit 0", tmp_path)[0] == 0
        assert verify.run_verify("exit 3", tmp_path)[0] == 3

    def test_argv_list_runs_without_a_shell(self, tmp_path):
        """An auto-detected command arrives as argv, so an interpreter path with
        spaces cannot be mangled by shell quoting."""
        code, out = verify.run_verify(
            [sys.executable, "-c", "print('marker_text')"], tmp_path)
        assert code == 0
        assert "marker_text" in out

    def test_argv_list_reports_nonzero(self, tmp_path):
        code, _ = verify.run_verify(
            [sys.executable, "-c", "raise SystemExit(4)"], tmp_path)
        assert code == 4

    def test_missing_command_is_not_a_pass(self, tmp_path):
        code, out = verify.run_verify(
            ["definitely-not-a-real-binary-xyz"], tmp_path)
        assert code != 0
        assert "failed to run verification command" in out

    def test_cli_still_exports_the_moved_names(self, tmp_path):
        """cli._run_verify / cli._goal_feedback are the CLI's patch points."""
        import localm.plugins.coder.cli as cli
        assert cli._run_verify("exit 0", tmp_path)[0] == 0
        assert "Do not modify the check itself" in cli._goal_feedback(
            "pytest", 1, "boom")


class TestInconclusive:
    def test_pytest_exit_5_is_inconclusive(self):
        assert verify.is_inconclusive("pytest -q", 5) is True
        assert verify.is_inconclusive([sys.executable, "-m", "pytest"], 5) is True

    def test_other_commands_exit_5_is_a_real_failure(self):
        assert verify.is_inconclusive("npm test", 5) is False

    def test_pytest_other_codes_are_not_inconclusive(self):
        assert verify.is_inconclusive("pytest -q", 1) is False
        assert verify.is_inconclusive("pytest -q", 0) is False


class TestLaunchFailureIsInconclusive:
    """A command that never started is not a code defect to bill the model for:
    it takes the INCONCLUSIVE branch, not the FAILURE one."""

    def test_the_launch_fact_travels_on_the_outcome_not_in_the_exit_code(
            self, tmp_path):
        """Through the REAL primitive: run_verify is the only thing that knows
        the launch raised, so it has to say so directly."""
        outcome = verify.run_verify(["definitely-not-a-real-binary-xyz"],
                                    tmp_path)
        code, out = outcome                       # still unpacks as a 2-tuple
        assert verify.launch_failed(outcome) is True
        assert "failed to run verification command" in out
        assert verify.is_inconclusive(["x"], code, True) is True

    def test_a_command_that_ran_reports_no_launch_failure(self, tmp_path):
        outcome = verify.run_verify([sys.executable, "-c",
                                     "raise SystemExit(1)"], tmp_path)
        assert verify.launch_failed(outcome) is False
        assert verify.is_inconclusive(["x"], outcome[0],
                                      verify.launch_failed(outcome)) is False

    @pytest.mark.parametrize("code", [1, 2, 5, 125, 126, 127, 124, 128])
    def test_no_exit_code_alone_makes_a_run_inconclusive(self, code):
        """125/126/127 are POSIX's "could not execute" codes, but a command that
        ran perfectly well can return them: npm exits 127 when a test script's
        binary is missing, and `npm test` is what auto-detection produces. Only
        the out-of-band launch failure counts as inconclusive (exit 5 is a
        separate, pytest-specific rule, covered above)."""
        assert verify.is_inconclusive(["npm", "test"], code) is False

    def test_the_reason_distinguishes_the_two_inconclusive_cases(self):
        assert verify.inconclusive_reason(
            ["npm", "test"], 125, True) == "could not run"
        assert verify.inconclusive_reason("pytest -q", 5) == "collected no tests"

    def test_a_plain_tuple_never_claims_a_launch_failure(self):
        """A patched cli._run_verify returns a bare 2-tuple; it must read as
        "the command ran" rather than silently disarming the oracle."""
        assert verify.launch_failed((127, "boom")) is False


# --------------------------------------------------------------------------- #
#  Auto-detection: the right command per project type, or none at all          #
# --------------------------------------------------------------------------- #

class TestDetectVerifyCommand:
    def test_empty_project_has_no_check(self, tmp_path):
        """The gate that keeps the oracle from running pytest (exit 5, forever)
        in a project that simply has no tests."""
        assert verify.detect_verify_command(tmp_path) is None

    def test_cargo_project(self, tmp_path, runners_installed):
        (tmp_path / "Cargo.toml").write_text("[package]\nname='x'\n")
        assert verify.detect_verify_command(tmp_path) == [
            _FAKE_BIN % "cargo", "test", "--color=never"]

    def test_go_project(self, tmp_path, runners_installed):
        (tmp_path / "go.mod").write_text("module x\n")
        assert verify.detect_verify_command(tmp_path) == [
            _FAKE_BIN % "go", "test", "./..."]

    def test_npm_project_with_a_test_script(self, tmp_path, runners_installed):
        (tmp_path / "package.json").write_text(
            json.dumps({"scripts": {"test": "jest"}}))
        assert verify.detect_verify_command(tmp_path) == [
            _FAKE_BIN % "npm", "test", "--", "--passWithNoTests"]

    def test_yarn_lock_selects_yarn(self, tmp_path, runners_installed):
        (tmp_path / "package.json").write_text(
            json.dumps({"scripts": {"test": "jest"}}))
        (tmp_path / "yarn.lock").write_text("")
        assert verify.detect_verify_command(tmp_path)[0] == _FAKE_BIN % "yarn"

    def test_npm_project_without_a_test_script_is_not_a_check(
            self, tmp_path, runners_installed):
        """`npm test` with no test script fails every run with "missing script",
        which no code change can fix - so it must not become the oracle. npm is
        present here, so this pins the test-script half specifically."""
        (tmp_path / "package.json").write_text(json.dumps({"name": "x"}))
        assert verify.detect_verify_command(tmp_path) is None

    def test_malformed_package_json_is_not_a_check(self, tmp_path,
                                                   runners_installed):
        (tmp_path / "package.json").write_text("{not json")
        assert verify.detect_verify_command(tmp_path) is None

    @pytest.mark.parametrize("name,body", [
        ("pytest.ini", "[pytest]\n"),
        ("pyproject.toml", "[tool.pytest.ini_options]\n"),
        ("tox.ini", "[pytest]\n"),
        ("setup.cfg", "[tool:pytest]\n"),
    ])
    def test_pytest_config_marks_a_python_project(self, tmp_path, name, body):
        (tmp_path / name).write_text(body)
        cmd = verify.detect_verify_command(tmp_path)
        assert cmd is not None and "pytest" in verify.command_text(cmd)

    def test_pyproject_without_a_pytest_section_is_not_a_check(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
        assert verify.detect_verify_command(tmp_path) is None

    def test_tests_directory_marks_a_python_project(self, tmp_path):
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_thing.py").write_text("def test_x(): pass\n")
        cmd = verify.detect_verify_command(tmp_path)
        assert cmd is not None and "pytest" in verify.command_text(cmd)

    def test_detected_python_command_uses_this_interpreter(self, tmp_path):
        """Not a bare `python` off PATH - that is a different env on many boxes."""
        (tmp_path / "pytest.ini").write_text("[pytest]\n")
        assert verify.detect_verify_command(tmp_path)[0] == sys.executable

    def test_config_verify_key_overrides_detection(self, tmp_path):
        (tmp_path / "Cargo.toml").write_text("[package]\nname='x'\n")
        (tmp_path / ".localcoder").mkdir()
        (tmp_path / ".localcoder" / "config.toml").write_text(
            'verify = "make check"\n')
        assert verify.detect_verify_command(tmp_path) == "make check"

    def test_config_verify_key_works_without_any_detection(self, tmp_path):
        (tmp_path / ".localcoder").mkdir()
        (tmp_path / ".localcoder" / "config.toml").write_text(
            'verify = "make check"\n')
        assert verify.detect_verify_command(tmp_path) == "make check"


class TestDetectionConfirmsTheRunnerCanRun:
    """X4's first half: a project file proves the project's SHAPE, not that its
    runner is installed. Detecting a command that cannot start hands the oracle
    a permanent 125 and bills it to the model as a code defect."""

    @staticmethod
    def _fake_which(monkeypatch, table):
        """Patch runner resolution at its single source (tools/shell)."""
        monkeypatch.setattr(
            "localm.plugins.coder.tools.shell.resolve_runner",
            lambda name: table.get(name))

    @pytest.mark.parametrize("marker,body,runner", [
        ("Cargo.toml", "[package]\nname='x'\n", "cargo"),
        ("go.mod", "module x\n", "go"),
    ])
    def test_absent_runner_means_no_oracle(self, tmp_path, monkeypatch,
                                           marker, body, runner):
        (tmp_path / marker).write_text(body)
        self._fake_which(monkeypatch, {})
        assert verify.detect_verify_command(tmp_path) is None

    @pytest.mark.parametrize("marker,body,runner,resolved", [
        ("Cargo.toml", "[package]\nname='x'\n", "cargo", "/opt/bin/cargo"),
        ("go.mod", "module x\n", "go", "/opt/bin/go"),
    ])
    def test_present_runner_is_used_at_its_resolved_path(
            self, tmp_path, monkeypatch, marker, body, runner, resolved):
        (tmp_path / marker).write_text(body)
        self._fake_which(monkeypatch, {runner: resolved})
        assert verify.detect_verify_command(tmp_path)[0] == resolved

    def test_npm_project_without_npm_installed_is_not_a_check(
            self, tmp_path, monkeypatch):
        (tmp_path / "package.json").write_text(
            json.dumps({"scripts": {"test": "jest"}}))
        self._fake_which(monkeypatch, {})
        assert verify.detect_verify_command(tmp_path) is None

    def test_npm_is_used_at_its_resolved_path_not_its_bare_name(
            self, tmp_path, monkeypatch):
        """`shutil.which('npm')` finds `npm.CMD` on Windows, but an argv list
        naming it "npm" still cannot start: argv execution goes through
        CreateProcess, which will not launch a .CMD shim. The resolved path
        does, so detection carries the path, not the name."""
        (tmp_path / "package.json").write_text(
            json.dumps({"scripts": {"test": "jest"}}))
        self._fake_which(monkeypatch, {"npm": r"Z:\Program Files\nodejs\npm.CMD"})
        cmd = verify.detect_verify_command(tmp_path)
        assert cmd == [r"Z:\Program Files\nodejs\npm.CMD", "test",
                       "--", "--passWithNoTests"]

    def test_yarn_project_gates_on_yarn_not_npm(self, tmp_path, monkeypatch):
        """The lockfile picks the runner, so the availability check has to follow
        it - npm being installed says nothing about yarn."""
        (tmp_path / "package.json").write_text(
            json.dumps({"scripts": {"test": "jest"}}))
        (tmp_path / "yarn.lock").write_text("")
        self._fake_which(monkeypatch, {"npm": "/usr/bin/npm"})
        assert verify.detect_verify_command(tmp_path) is None
        self._fake_which(monkeypatch, {"yarn": "/usr/bin/yarn"})
        assert verify.detect_verify_command(tmp_path)[0] == "/usr/bin/yarn"

    def test_python_project_without_pytest_importable_is_not_a_check(
            self, tmp_path, monkeypatch):
        """The interpreter always launches, so the launch check cannot see this
        one: with no pytest importable the check exits 1 with "No module named
        pytest" on every run, unfixable by the model and wearing an exit code
        that looks like a genuine test failure."""
        (tmp_path / "pytest.ini").write_text("[pytest]\n")
        monkeypatch.setattr(
            "importlib.util.find_spec",
            lambda name, *a, **k: None if name == "pytest" else object())
        assert verify.detect_verify_command(tmp_path) is None

    def test_python_project_with_pytest_still_detects(self, tmp_path):
        """The other direction, unpatched: pytest is importable here (it is
        running this test), so the oracle must still be offered."""
        (tmp_path / "pytest.ini").write_text("[pytest]\n")
        assert verify.detect_verify_command(tmp_path)[0] == sys.executable

    def test_whatever_is_detected_here_can_actually_be_launched(self, tmp_path):
        """The invariant, against the REAL environment rather than a fake: for
        every project shape, detection returns either None or a command whose
        argv[0] this platform can genuinely start."""
        import shutil
        import subprocess
        (tmp_path / "package.json").write_text(
            json.dumps({"scripts": {"test": "jest"}}))
        if shutil.which("npm") is None:
            pytest.skip("npm not installed on this box")
        cmd = verify.detect_verify_command(tmp_path)
        assert cmd is not None, "npm is installed, so the oracle should exist"
        try:
            subprocess.run([cmd[0], "--version"], capture_output=True,
                           timeout=120)
        except FileNotFoundError as exc:
            pytest.fail(
                f"detection returned a command that cannot be launched: "
                f"{cmd!r} ({exc})")


# --------------------------------------------------------------------------- #
#  The gate at the pre-done boundary                                           #
# --------------------------------------------------------------------------- #

class TestVerifyGate:
    _NO_TOOL_SCRIPT = ["Nothing to do here."]
    _SCRIPT_THEN_ANSWER = [
        _tool_call("write_file", path="mod.py", content="x = 1\n"),
        "All done.",
    ]

    def test_no_command_configured_is_a_no_op(self, tmp_path):
        agent = _make_agent(tmp_path)
        _record_write(agent)
        assert agent._run_verify_gate("done", False, _fresh_state(agent)) is None

    def test_task_that_changed_nothing_does_not_run_the_check(self, tmp_path):
        """A follow-up question in a REPL session must not trigger the suite."""
        marker = tmp_path / "ran"
        agent = _make_agent(tmp_path, verify_cmd=_touch_cmd(marker))
        _record_write(agent, writes=2)
        st = _fresh_state(agent, verify_checked_at=2)  # the writes predate this task
        assert agent._run_verify_gate("done", False, st) is None
        assert not marker.exists()

    def test_passing_check_falls_through(self, tmp_path):
        agent = _make_agent(tmp_path, verify_cmd="exit 0")
        _record_write(agent)
        st = _fresh_state(agent)
        assert agent._run_verify_gate("done", False, st) is None
        assert st.verify_checked_at == 1

    def test_a_pass_does_not_exempt_later_writes(self, tmp_path):
        """A fix made after the check went green (the reviewer can prompt one)
        must be checked too, not ride in on the earlier pass."""
        marker = tmp_path / "ran"
        agent = _make_agent(tmp_path, verify_cmd="exit 0")
        _record_write(agent)
        st = _fresh_state(agent)
        agent._run_verify_gate("done", False, st)          # passes at 1 write
        agent.verify_cmd = _touch_cmd(marker)              # prove it runs again
        _record_write(agent, path="other.py")              # a NEW write lands
        agent._run_verify_gate("done", False, st)
        assert marker.exists()

    def test_passing_check_suppresses_the_self_verify_nudge(self, tmp_path):
        """The nudge is a self-graded proxy for exactly this check; once the real
        one passes it has nothing left to ask for."""
        agent = _make_agent(tmp_path, verify_cmd="exit 0")
        _record_write(agent)
        agent._unverified_writes.add("mod.py")
        st = _fresh_state(agent)
        agent._run_verify_gate("done", False, st)
        assert st.verify_nudged is True
        assert agent._unverified_writes == set()

    def test_failing_check_feeds_the_failure_back(self, tmp_path):
        agent = _make_agent(tmp_path, verify_cmd="exit 1", verify_max_retries=2)
        _record_write(agent)
        st = _fresh_state(agent)
        result = agent._run_verify_gate("all done", False, st)
        assert result == (False, "")          # keep looping, do not finish
        assert st.verify_retries == 1
        fed_back = agent._messages[-1]["content"]
        assert "failed with exit code 1" in fed_back

    def test_feedback_forbids_editing_the_check(self, tmp_path):
        """The anti-gaming instruction: a weak model's cheapest green is to
        weaken the check."""
        agent = _make_agent(tmp_path, verify_cmd="exit 1")
        _record_write(agent)
        agent._run_verify_gate("done", False, _fresh_state(agent))
        assert "Do not modify the check itself" in agent._messages[-1]["content"]

    def test_exhausted_retries_report_failure_not_success(self, tmp_path):
        agent = _make_agent(tmp_path, verify_cmd="exit 1", verify_max_retries=1)
        _record_write(agent)
        st = _fresh_state(agent)
        agent._run_verify_gate("done", False, st)          # retry 1
        should_break, text = agent._run_verify_gate("done", False, st)
        assert should_break is True
        assert "verification FAILED" in text
        assert agent.last_run_ok is False

    def test_inconclusive_pytest_does_not_loop_and_does_not_claim_success(
            self, tmp_path):
        agent = _make_agent(
            tmp_path,
            verify_cmd=[sys.executable, "-c",
                        "import sys; sys.stderr.write('pytest\\n'); "
                        "raise SystemExit(5)"])
        _record_write(agent)
        st = _fresh_state(agent)
        with patch("localm.plugins.coder.agent.print_warning") as warn:
            assert agent._run_verify_gate("done", False, st) is None
        assert st.verify_settled is True                    # no pointless retries
        assert agent.last_run_ok is True                    # not a failure either
        assert "inconclusive" in warn.call_args[0][0]       # and never called a pass

    def test_a_check_that_could_not_run_is_neither_a_failure_nor_a_pass(
            self, tmp_path):
        """X4 END TO END at the gate. The command names a binary that does not
        exist, so it never starts. The model must not be asked to fix it, the
        task must not be marked failed, and nothing may report a pass."""
        agent = _make_agent(tmp_path,
                            verify_cmd=["definitely-not-a-real-binary-xyz"],
                            verify_max_retries=2)
        _record_write(agent)
        st = _fresh_state(agent)
        with patch("localm.plugins.coder.agent.print_warning") as warn:
            assert agent._run_verify_gate("done", False, st) is None
        assert st.verify_settled is True          # not a retry loop
        assert st.verify_retries == 0             # not one attempt was billed
        assert agent.last_run_ok is True          # not reported as task failure
        assert agent.last_verify_state == "inconclusive"   # and not as a pass
        assert "could not run" in warn.call_args[0][0]
        assert "nothing was actually verified" in warn.call_args[0][0]

    def test_a_genuinely_failing_check_still_reports_failure(self, tmp_path):
        """FIRES-CONTROL for the test above, in the same run: the oracle must
        still fail the things it is there to fail. If widening "inconclusive"
        had disarmed it, this is what would go quiet."""
        agent = _make_agent(tmp_path, verify_cmd="exit 1", verify_max_retries=1)
        _record_write(agent)
        st = _fresh_state(agent)
        assert agent._run_verify_gate("done", False, st) == (False, "")
        assert st.verify_retries == 1             # the model IS asked to fix it
        should_break, text = agent._run_verify_gate("done", False, st)
        assert should_break is True
        assert "verification FAILED" in text
        assert agent.last_run_ok is False
        assert agent.last_verify_state == "failed"

    def test_a_run_that_collected_nothing_is_never_reported_as_a_pass(
            self, tmp_path):
        """A mangled invocation (args split wrong, a filter matching nothing)
        makes pytest collect zero tests and exit 5 in about a second, and the
        wrapper log looks clean. Nothing about that run verified anything, so
        the machine-readable answer reports INCONCLUSIVE rather than a pass."""
        agent = _make_agent(
            tmp_path,
            verify_cmd=[sys.executable, "-c",
                        "import sys; sys.stderr.write('pytest: no tests ran\\n');"
                        " raise SystemExit(5)"])
        _record_write(agent)
        with patch("localm.plugins.coder.agent.print_warning"):
            agent._run_verify_gate("done", False, _fresh_state(agent))
        assert agent.last_verify_state == "inconclusive"

    def test_verify_state_records_a_pass(self, tmp_path):
        agent = _make_agent(tmp_path, verify_cmd="exit 0")
        _record_write(agent)
        agent._run_verify_gate("done", False, _fresh_state(agent))
        assert agent.last_verify_state == "passed"

    def test_verify_state_is_none_when_no_check_ran(self, tmp_path):
        """Not just "None from __init__": drive it to a real verdict first, so a
        field that is never assigned cannot pass this by accident."""
        agent = _make_agent(tmp_path, self._NO_TOOL_SCRIPT, verify_cmd="exit 0")
        _record_write(agent)
        agent._run_verify_gate("done", False, _fresh_state(agent))
        assert agent.last_verify_state == "passed"      # a verdict is now set
        agent.verify_cmd = None                         # this run has no oracle
        agent.chat("just answer")
        assert agent.last_verify_state is None

    def test_clearing_the_session_drops_the_stale_verdict(self, tmp_path):
        """reset() re-arms last_run_ok; the new field has to go with it, or /clear
        leaves a verdict about a conversation that no longer exists."""
        agent = _make_agent(tmp_path, verify_cmd="exit 0")
        _record_write(agent)
        agent._run_verify_gate("done", False, _fresh_state(agent))
        assert agent.last_verify_state == "passed"
        agent.reset()
        assert agent.last_verify_state is None

    def test_verify_state_is_per_run_like_last_run_ok(self, tmp_path):
        """A later clean turn must not keep reporting the earlier turn's verdict
        (the same per-run reset last_run_ok has)."""
        agent = _make_agent(tmp_path, self._SCRIPT_THEN_ANSWER,
                            verify_cmd="exit 1", verify_max_retries=1)
        agent.chat("write mod.py")
        assert agent.last_verify_state == "failed"
        agent.chat("thanks, what does it do?")    # writes nothing -> no gate
        assert agent.last_verify_state is None

    def test_gate_is_skipped_for_a_malformed_tool_call(self, tmp_path):
        """A response that only looks like a broken tool call is mid-call, not a
        finish - the repair turn owns it, and the suite must not run for it."""
        marker = tmp_path / "ran"
        agent = _make_agent(tmp_path, verify_cmd=_touch_cmd(marker))
        _record_write(agent)
        agent._handle_no_tool_calls(
            '<tool_call>{"name": "read_file", "args"', False, _fresh_state(agent))
        assert not marker.exists()

    def test_restricted_session_never_gets_an_oracle(self, tmp_path):
        """A restricted session has no process execution at all; an oracle would
        hand a scoped key exactly what the restriction removes."""
        agent = _make_agent(tmp_path, verify_cmd="exit 0", restricted=True)
        assert agent.verify_cmd is None


# --------------------------------------------------------------------------- #
#  End to end: the model cannot declare success past a failing check           #
# --------------------------------------------------------------------------- #

class TestInteractiveSessionEndToEnd:
    """Drives the REAL agent loop with a scripted model: it writes a file, then
    claims it is finished. Only the exit code decides whether it is."""

    _SCRIPT = [_tool_call("write_file", path="mod.py", content="x = 1\n"),
               "All done - the code is correct and complete."]

    def test_model_cannot_declare_success_while_the_check_fails(self, tmp_path):
        agent = _make_agent(tmp_path, self._SCRIPT,
                            verify_cmd="exit 1", verify_max_retries=1)
        final = agent.chat("write mod.py")
        assert "verification FAILED" in final
        assert agent.last_run_ok is False
        assert (tmp_path / "mod.py").exists()      # the write really happened

    def test_failed_verification_is_recorded_session_wide(self, tmp_path):
        """Coexistence with the per-run last_run_ok reset: the gate's write
        happens mid-run, inside _handle_no_tool_calls, so _loop's finally still
        folds it into the session-level _had_any_failure the close-time episodic
        reflection reads. Asserted as a call chain, not by line position."""
        agent = _make_agent(tmp_path, self._SCRIPT,
                            verify_cmd="exit 1", verify_max_retries=1)
        agent.chat("write mod.py")
        assert agent.last_run_ok is False
        assert agent._had_any_failure is True

    def test_clean_turn_after_a_failed_verification_reports_ok(self, tmp_path):
        """The other half of that coexistence: the per-run reset must still
        clear a PREVIOUS turn's verification failure, while the session-level
        record of it survives."""
        agent = _make_agent(tmp_path, self._SCRIPT,
                            verify_cmd="exit 1", verify_max_retries=1)
        agent.chat("write mod.py")
        assert agent.last_run_ok is False
        # Second turn writes nothing, so the gate does not fire and the run is clean.
        agent.chat("thanks, what does it do?")
        assert agent.last_run_ok is True          # not poisoned by the earlier turn
        assert agent._had_any_failure is True     # but the session still remembers

    def test_passing_check_lets_the_answer_through_clean(self, tmp_path):
        agent = _make_agent(tmp_path, self._SCRIPT,
                            verify_cmd="exit 0", verify_max_retries=1)
        final = agent.chat("write mod.py")
        assert "verification FAILED" not in final
        assert "All done" in final
        assert agent.last_run_ok is True

    def test_check_actually_runs_against_the_written_code(self, tmp_path):
        """The oracle sees the file the agent just wrote - a real check, not a
        rerun of whatever was on disk before the turn."""
        agent = _make_agent(
            tmp_path, self._SCRIPT,
            verify_cmd=[sys.executable, "-c",
                        "import pathlib,sys; "
                        "sys.exit(0 if pathlib.Path('mod.py').read_text().strip()"
                        " == 'x = 1' else 1)"],
            verify_max_retries=1)
        final = agent.chat("write mod.py")
        assert "verification FAILED" not in final
        assert agent.last_run_ok is True

    def test_no_command_leaves_the_old_behaviour_untouched(self, tmp_path):
        agent = _make_agent(tmp_path, self._SCRIPT)
        final = agent.chat("write mod.py")
        assert "verification FAILED" not in final
        assert agent.last_run_ok is True

    def test_gui_shaped_session_emits_the_failure(self, tmp_path):
        """A GUI session has an event sink instead of a terminal; the failure has
        to reach it, not only the console."""
        events: list = []
        agent = _make_agent(tmp_path, self._SCRIPT, verify_cmd="exit 1",
                            verify_max_retries=1, on_event=events.append)
        agent.chat("write mod.py")
        texts = [e.get("text", "") for e in events if e.get("type") == "info"]
        assert any("verification FAILED" in t for t in texts)
        assert any("verification: running" in t for t in texts)


# --------------------------------------------------------------------------- #
#  Wiring: sessions, the CLI, and the self-verify nudge                        #
# --------------------------------------------------------------------------- #

class TestSessionWiring:
    def _session(self, tmp_path, **kwargs):
        from localm.plugins.coder.sessions import CoderSession
        with patch("localm.plugins.coder.agent.ProjectMap") as MockPM, \
             patch("localm.plugins.coder.agent.make_audit_log"), \
             patch("localm.plugins.coder.agent.load_memory", return_value=""):
            MockPM.build.return_value.file_count.return_value = 0
            return CoderSession(tmp_path, _ScriptedBackend(["done"]), **kwargs)

    def test_gui_session_auto_detects(self, tmp_path, runners_installed):
        (tmp_path / "Cargo.toml").write_text("[package]\nname='x'\n")
        session = self._session(tmp_path)
        assert session.agent.verify_cmd == [
            _FAKE_BIN % "cargo", "test", "--color=never"]
        assert session.info()["verify"] == (
            f"{_FAKE_BIN % 'cargo'} test --color=never")

    def test_gui_session_explicit_command_wins(self, tmp_path):
        (tmp_path / "Cargo.toml").write_text("[package]\nname='x'\n")
        session = self._session(tmp_path, verify="make check")
        assert session.agent.verify_cmd == "make check"

    def test_gui_session_auto_verify_off(self, tmp_path):
        (tmp_path / "Cargo.toml").write_text("[package]\nname='x'\n")
        session = self._session(tmp_path, auto_verify=False)
        assert session.agent.verify_cmd is None
        assert session.info()["verify"] is None

    def test_restricted_gui_session_gets_none_even_when_asked(self, tmp_path):
        (tmp_path / "Cargo.toml").write_text("[package]\nname='x'\n")
        session = self._session(tmp_path, restricted=True, verify="make check")
        assert session.agent.verify_cmd is None

    def test_final_event_distinguishes_unverified_from_a_clean_finish(
            self, tmp_path):
        """The GUI is the consumer that reads the gate's verdict as a
        boolean. A check that could not run leaves ok true, so without a second
        field the one machine-readable answer says "clean finish" about a task
        nothing verified."""
        import queue as _queue
        import time as _time
        from localm.plugins.coder.sessions import CoderSession

        script = [_tool_call("write_file", path="mod.py", content="x = 1\n"),
                  "All done."]
        session = CoderSession(
            tmp_path, _ScriptedBackend(script), auto_approve=True,
            verify=["definitely-not-a-real-binary-xyz"], max_turns=10)
        try:
            assert session.send_message("write mod.py") == "started"
            deadline = _time.monotonic() + 15.0
            final = None
            while _time.monotonic() < deadline and final is None:
                try:
                    ev = session.events.get(timeout=0.2)
                except _queue.Empty:
                    continue
                if ev["type"] == "final":
                    final = ev
            assert final is not None, "no final event within 15s"
            assert final["ok"] is True            # the run itself was fine
            assert final["verify_state"] == "inconclusive"   # but unverified
        finally:
            session.close()


class TestSelfVerifyNudge:
    def _nudge_text(self, agent):
        agent._unverified_writes.add("mod.py")
        agent._handle_no_tool_calls("done", False, _fresh_state(agent))
        return agent._messages[-1]["content"]

    def test_nudge_names_the_detected_command(self, tmp_path):
        agent = _make_agent(tmp_path, verify_cmd="pytest -x")
        assert "pytest -x" in self._nudge_text(agent)

    def test_nudge_falls_back_when_no_command_is_known(self, tmp_path):
        agent = _make_agent(tmp_path)
        text = self._nudge_text(agent)
        assert "run_tests" in text and "re-read the changed files" in text


class TestCliWiring:
    def test_repl_session_gets_the_detected_command(self, tmp_path, monkeypatch,
                                                    runners_installed):
        """`localcoder` with no TASK is the interactive path the oracle targets."""
        from localm.plugins.coder.cli import _main
        (tmp_path / "Cargo.toml").write_text("[package]\nname='x'\n")
        captured = {}

        def _fake_repl(agent):
            captured["verify_cmd"] = agent.verify_cmd

        monkeypatch.setattr(_main, "_repl", _fake_repl)
        self._run_cli(monkeypatch, tmp_path, [])
        assert captured["verify_cmd"] == [
            _FAKE_BIN % "cargo", "test", "--color=never"]

    def test_no_verify_flag_turns_it_off(self, tmp_path, monkeypatch):
        from localm.plugins.coder.cli import _main
        (tmp_path / "Cargo.toml").write_text("[package]\nname='x'\n")
        captured = {}
        monkeypatch.setattr(
            _main, "_repl",
            lambda agent: captured.__setitem__("verify_cmd", agent.verify_cmd))
        self._run_cli(monkeypatch, tmp_path, ["--no-verify"])
        assert captured["verify_cmd"] is None

    def test_explicit_verify_flag_wins(self, tmp_path, monkeypatch):
        from localm.plugins.coder.cli import _main
        (tmp_path / "Cargo.toml").write_text("[package]\nname='x'\n")
        captured = {}
        monkeypatch.setattr(
            _main, "_repl",
            lambda agent: captured.__setitem__("verify_cmd", agent.verify_cmd))
        self._run_cli(monkeypatch, tmp_path, ["--verify", "make check"])
        assert captured["verify_cmd"] == "make check"

    def test_one_shot_task_has_no_in_loop_oracle(self, tmp_path, monkeypatch):
        """--until owns the one-shot flow; running both would execute the check
        twice per iteration."""
        (tmp_path / "Cargo.toml").write_text("[package]\nname='x'\n")
        captured = {}

        def _fake_run_task(self_agent, task):
            captured["verify_cmd"] = self_agent.verify_cmd
            return "done"

        monkeypatch.setattr(
            "localm.plugins.coder.agent.Agent.run_task", _fake_run_task)
        self._run_cli(monkeypatch, tmp_path, ["do the thing", "--yes"])
        assert captured["verify_cmd"] is None

    @staticmethod
    def _run_cli(monkeypatch, tmp_path, extra_args):
        from click.testing import CliRunner
        from localm.plugins.coder.cli import _main

        monkeypatch.setattr(
            "localm.plugins.engine.PluginManager.is_active", lambda self, n: True)
        monkeypatch.setattr(
            _main, "_build_backend",
            lambda *a, **k: _ScriptedBackend(["done"]))
        monkeypatch.setattr(_main, "print_banner", lambda *a, **k: None)
        with patch("localm.plugins.coder.agent.ProjectMap") as MockPM, \
             patch("localm.plugins.coder.agent.make_audit_log"), \
             patch("localm.plugins.coder.agent.load_memory", return_value=""):
            MockPM.build.return_value.file_count.return_value = 0
            result = CliRunner().invoke(
                _main.main, ["--model", "m", "-c", str(tmp_path)] + extra_args)
        assert result.exit_code == 0, result.output
        return result
