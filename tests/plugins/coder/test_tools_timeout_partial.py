# SPDX-License-Identifier: AGPL-3.0-or-later
"""Timed-out subprocesses surface the output they produced before the kill.

subprocess.TimeoutExpired carries the partial stdout/stderr captured up to the
timeout. Discarding it leaves a command that printed real diagnostics and then
hung reporting only "timed out". The helper lives in tools/base.py."""

import subprocess
from unittest.mock import patch

from localm.plugins.coder.tools import (
    tool_git_status,
    tool_run_shell,
    tool_run_tests,
)
from localm.plugins.coder.tools.base import _partial_on_timeout

_RUN = "localm.plugins.coder.tools.subprocess.run"


def _timeout_exc(output=None, stderr=None, cmd="cmd", timeout=30):
    return subprocess.TimeoutExpired(cmd, timeout, output=output, stderr=stderr)


class TestHelper:
    def test_no_capture_returns_empty(self):
        assert _partial_on_timeout(_timeout_exc()) == ""

    def test_str_output_formatted(self):
        s = _partial_on_timeout(_timeout_exc(output="Running test 1...\n"))
        assert "partial output captured before the timeout" in s
        assert "Running test 1..." in s

    def test_bytes_output_decoded(self):
        # POSIX quirk: TimeoutExpired can carry bytes even in text mode.
        s = _partial_on_timeout(_timeout_exc(output=b"caf\xc3\xa9 done\n"))
        assert "café done" in s

    def test_stderr_labelled(self):
        s = _partial_on_timeout(_timeout_exc(output="out\n", stderr="warn\n"))
        assert "STDERR:\nwarn" in s

    def test_whitespace_only_capture_returns_empty(self):
        assert _partial_on_timeout(_timeout_exc(output="  \n")) == ""

    def test_huge_capture_truncated(self):
        s = _partial_on_timeout(_timeout_exc(output="x" * 50_000))
        assert "chars truncated" in s
        assert len(s) < 20_000


class TestErrorSummaryStaysOneLine:
    def test_multiline_error_summary_is_first_line_only(self):
        from localm.plugins.coder.tools.base import ToolResult
        r = ToolResult.error("headline\n" + "x" * 9000)
        assert r.summary == "ERROR: headline"
        assert "x" * 9000 in r.output          # full detail stays in output

    def test_long_single_line_summary_capped(self):
        from localm.plugins.coder.tools.base import ToolResult
        r = ToolResult.error("y" * 500)
        assert len(r.summary) < 220
        assert r.summary.endswith("...")
        assert r.output == "y" * 500

    def test_short_error_summary_unchanged(self):
        from localm.plugins.coder.tools.base import ToolResult
        r = ToolResult.error("it broke")
        assert r.summary == "ERROR: it broke"

    def test_timeout_partial_not_in_summary(self, tmp_path):
        exc = _timeout_exc(output="blob " * 500)
        with patch(_RUN, side_effect=exc):
            r = tool_run_shell(tmp_path, "slow-command", timeout=30)
        assert r.summary == "ERROR: Command timed out after 30s"
        assert "blob" in r.output              # detail only in the output body


class TestRunShellTimeout:
    def test_partial_output_included(self, tmp_path):
        exc = _timeout_exc(output="step 1 done\nstep 2 running", stderr="late warn")
        with patch(_RUN, side_effect=exc):
            r = tool_run_shell(tmp_path, "slow-command", timeout=30)
        assert r.ok is False
        assert "timed out after 30s" in r.output
        assert "step 2 running" in r.output
        assert "STDERR:" in r.output and "late warn" in r.output

    def test_no_partial_keeps_plain_message(self, tmp_path):
        with patch(_RUN, side_effect=_timeout_exc()):
            r = tool_run_shell(tmp_path, "slow-command", timeout=30)
        assert r.ok is False
        assert "timed out after 30s" in r.output
        assert "partial output" not in r.output


class TestRunTestsTimeout:
    def test_partial_output_included(self, tmp_path):
        exc = _timeout_exc(output="3 passed, 2 failed so far")
        with patch(_RUN, side_effect=exc):
            r = tool_run_tests(tmp_path, runner="pytest")
        assert r.ok is False
        assert "timed out after 120s" in r.output
        assert "3 passed, 2 failed so far" in r.output


class TestGitTimeout:
    def test_partial_output_included(self, tmp_path):
        exc = _timeout_exc(output="On branch master")
        with patch(_RUN, side_effect=exc):
            r = tool_git_status(tmp_path)
        assert r.ok is False
        assert "timed out" in r.output
        assert "On branch master" in r.output


class TestGoalVerifyTimeout:
    """cli/goal.py's _run_verify must preserve any output the verification
    command produced before a timeout, the same as run_shell/run_tests/git do via
    _partial_on_timeout. All four go through the shared run_subprocess()
    primitive, so goal.py gets the same partial-output behaviour rather than a
    bare 'timed out' message."""

    def test_partial_output_included(self, tmp_path):
        import localm.plugins.coder.cli as cli

        exc = _timeout_exc(output="1 passed\n2 running", stderr="slow fixture")
        with patch("localm.plugins.coder.tools.base.subprocess.run", side_effect=exc):
            code, output = cli._run_verify("slow-command", tmp_path, timeout=30)

        assert code == 124
        assert "timed out after 30s" in output
        assert "2 running" in output
        assert "STDERR:" in output and "slow fixture" in output

    def test_no_partial_keeps_plain_message(self, tmp_path):
        import localm.plugins.coder.cli as cli

        with patch("localm.plugins.coder.tools.base.subprocess.run",
                   side_effect=_timeout_exc()):
            code, output = cli._run_verify("slow-command", tmp_path, timeout=30)

        assert code == 124
        assert "timed out after 30s" in output
        assert "partial output" not in output

    def test_command_not_found_reports_failure_not_a_crash(self, tmp_path):
        import localm.plugins.coder.cli as cli

        with patch("localm.plugins.coder.tools.base.subprocess.run",
                   side_effect=FileNotFoundError("no such file")):
            code, output = cli._run_verify("ghost-command", tmp_path)

        assert code == 125
        assert "failed to run verification command" in output
