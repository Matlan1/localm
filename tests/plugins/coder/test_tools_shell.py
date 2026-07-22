# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Tests for shell and test-runner tools in localm.plugins.coder.tools:
  tool_run_shell, tool_run_tests, _detect_test_runner
"""

import subprocess
from unittest.mock import patch, MagicMock

from localm.plugins.coder.tools import (
    tool_run_shell,
    tool_run_tests,
    _detect_test_runner,
    _needs_shell,
)


def _launcher(launched) -> str:
    """The program *launched* actually starts.

    The platform shell's launch form differs on purpose: POSIX gets an argv list
    (execv receives it verbatim), Windows a raw command-line STRING, because an
    argv list is re-quoted by list2cmdline in syntax cmd.exe misreads. See
    tools/base.py:platform_shell.
    """
    return launched.split()[0] if isinstance(launched, str) else launched[0]


# ---------------------------------------------------------------------------
#  tool_run_shell
# ---------------------------------------------------------------------------

class TestRunShell:
    def _make_proc(self, stdout="", stderr="", returncode=0):
        p = MagicMock()
        p.stdout = stdout
        p.stderr = stderr
        p.returncode = returncode
        return p

    def _patch_run(self, **kwargs):
        return patch(
            "localm.plugins.coder.tools.subprocess.run",
            return_value=self._make_proc(**kwargs),
        )

    def test_success_returns_ok(self, tmp_path):
        with self._patch_run(stdout="hello\n", returncode=0):
            r = tool_run_shell(tmp_path, "echo hello")
        assert r.ok
        assert "hello" in r.output

    def test_nonzero_exit_returns_not_ok(self, tmp_path):
        with self._patch_run(stdout="", stderr="error", returncode=1):
            r = tool_run_shell(tmp_path, "false")
        assert not r.ok
        assert "exit_code" in r.output

    def test_output_contains_exit_code_xml(self, tmp_path):
        with self._patch_run(stdout="out\n", returncode=0):
            r = tool_run_shell(tmp_path, "echo out")
        assert "<exit_code>0</exit_code>" in r.output

    def test_stderr_included_in_output(self, tmp_path):
        with self._patch_run(stdout="", stderr="warning!\n", returncode=0):
            r = tool_run_shell(tmp_path, "cmd")
        assert "STDERR" in r.output
        assert "warning!" in r.output

    def test_timeout_returns_error(self, tmp_path):
        with patch(
            "localm.plugins.coder.tools.subprocess.run",
            side_effect=subprocess.TimeoutExpired(["cmd"], 30),
        ):
            r = tool_run_shell(tmp_path, "sleep 100", timeout=30)
        assert not r.ok
        assert "timed out" in r.output.lower()

    def test_no_output_shows_placeholder(self, tmp_path):
        with self._patch_run(stdout="", stderr="", returncode=0):
            r = tool_run_shell(tmp_path, "true")
        assert "(no output)" in r.output

    def test_summary_contains_command_prefix(self, tmp_path):
        with self._patch_run(stdout="", returncode=0):
            r = tool_run_shell(tmp_path, "echo test")
        assert "echo test" in r.summary

    def test_large_output_truncated(self, tmp_path):
        big = "x" * 50_000
        with self._patch_run(stdout=big, returncode=0):
            r = tool_run_shell(tmp_path, "bigcmd")
        assert r.ok
        assert r.truncated

    def test_privacy_mode_passes_env(self, tmp_path):
        captured_env = {}

        def fake_run(cmd, **kwargs):
            captured_env.update(kwargs.get("env") or {})
            return self._make_proc(stdout="ok")

        with patch("localm.plugins.coder.tools.subprocess.run", side_effect=fake_run):
            tool_run_shell(tmp_path, "echo hi", _privacy=True)

        assert captured_env.get("HISTFILE") in ("NUL", "/dev/null")
        assert captured_env.get("HISTSIZE") == "0"

    def test_simple_command_uses_arg_list(self, tmp_path):
        """Commands whose executable exists on PATH run as an argument list."""
        captured_cmd = []

        def fake_run(cmd, **kwargs):
            captured_cmd.extend(cmd)
            return self._make_proc(stdout="")

        with patch("localm.plugins.coder.tools.subprocess.run", side_effect=fake_run):
            tool_run_shell(tmp_path, "git status")   # git resolves via PATH

        # Must NOT go through cmd/sh - first token is the executable
        assert captured_cmd[0] == "git"
        assert "status" in captured_cmd

    def test_shell_builtin_routed_through_shell(self, tmp_path):
        """echo/dir/type have no executable on disk - must use the shell,
        otherwise argument-list mode fails with 'file not found'."""
        captured = {}

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            return self._make_proc(stdout="hi\n")

        with patch("localm.plugins.coder.tools.subprocess.run", side_effect=fake_run), \
             patch("shutil.which", return_value=None):
            r = tool_run_shell(tmp_path, "echo hi")

        assert r.ok
        assert _launcher(captured["cmd"]) in ("cmd", "/bin/sh"), captured["cmd"]

    def test_pipe_uses_shell(self, tmp_path):
        """Commands with pipe operators are routed through the system shell."""
        captured = {}

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            return self._make_proc(stdout="")

        with patch("localm.plugins.coder.tools.subprocess.run", side_effect=fake_run):
            tool_run_shell(tmp_path, "echo hi | cat")

        launched = captured["cmd"]
        assert _launcher(launched) in ("cmd", "/bin/sh"), launched
        # The command text must reach the shell UNCHANGED - re-quoting it is the
        # defect that made a quoted path unopenable.
        tail = launched if isinstance(launched, str) else launched[-1]
        assert tail.endswith("echo hi | cat"), tail


# ---------------------------------------------------------------------------
#  _needs_shell
# ---------------------------------------------------------------------------

class TestNeedsShell:
    def test_simple_command_no_shell(self):
        assert not _needs_shell("python -m pytest")
        assert not _needs_shell("echo hello")
        assert not _needs_shell("git status")

    def test_pipe_needs_shell(self):
        assert _needs_shell("echo hi | cat")

    def test_redirect_needs_shell(self):
        assert _needs_shell("echo hi > out.txt")
        assert _needs_shell("cat < in.txt")

    def test_and_operator_needs_shell(self):
        assert _needs_shell("make && echo done")

    def test_subshell_needs_shell(self):
        assert _needs_shell("echo $(pwd)")

    def test_quoted_pipe_no_shell(self):
        # A pipe inside quotes is not a shell operator
        assert not _needs_shell("python -c 'print(\"|\")'")

    def test_glob_star_needs_shell(self):
        assert _needs_shell("ls *.py")

    def test_semicolon_needs_shell(self):
        assert _needs_shell("cd src; ls")


# ---------------------------------------------------------------------------
#  _detect_test_runner
# ---------------------------------------------------------------------------

class TestDetectTestRunner:
    def test_detects_pytest_for_python_project(self, tmp_path):
        # Empty dir - defaults to pytest
        cmd = _detect_test_runner(tmp_path)
        assert "pytest" in " ".join(cmd)

    def test_python_runner_uses_current_interpreter(self, tmp_path):
        # Regression: the pytest runner must invoke sys.executable (the interpreter
        # running localm), NOT a bare "python" off PATH. On machines where PATH
        # `python` is a different env (uv/conda/system) without pytest, a bare
        # "python -m pytest" reports "No module named pytest" on a passing suite.
        import sys
        cmd = _detect_test_runner(tmp_path)
        assert cmd[0] == sys.executable
        assert cmd[1:3] == ["-m", "pytest"]

    def test_detects_cargo_for_rust(self, tmp_path):
        (tmp_path / "Cargo.toml").write_text('[package]\nname = "x"\n')
        cmd = _detect_test_runner(tmp_path)
        assert "cargo" in cmd[0]

    def test_detects_go_test(self, tmp_path):
        (tmp_path / "go.mod").write_text("module example.com/x\n")
        cmd = _detect_test_runner(tmp_path)
        assert "go" in cmd[0]

    def test_detects_npm_when_no_yarn_lock(self, tmp_path):
        (tmp_path / "package.json").write_text('{"name": "x"}')
        cmd = _detect_test_runner(tmp_path)
        assert "npm" in cmd[0]

    def test_detects_yarn_when_yarn_lock_present(self, tmp_path):
        (tmp_path / "package.json").write_text('{"name": "x"}')
        (tmp_path / "yarn.lock").write_text("")
        cmd = _detect_test_runner(tmp_path)
        assert "yarn" in cmd[0]


# ---------------------------------------------------------------------------
#  tool_run_tests
# ---------------------------------------------------------------------------

class TestRunTests:
    def _make_proc(self, stdout="", stderr="", returncode=0):
        p = MagicMock()
        p.stdout = stdout
        p.stderr = stderr
        p.returncode = returncode
        return p

    def _patch_run(self, **kwargs):
        return patch(
            "localm.plugins.coder.tools.subprocess.run",
            return_value=self._make_proc(**kwargs),
        )

    def test_auto_runner_runs_tests(self, tmp_path):
        with self._patch_run(stdout="1 passed\n", returncode=0):
            r = tool_run_tests(tmp_path, runner="auto")
        assert r.ok
        assert "passed" in r.summary

    def test_explicit_pytest_runner(self, tmp_path):
        captured = []

        def fake_run(cmd, **kwargs):
            captured.extend(cmd)
            return self._make_proc(stdout="2 passed")

        with patch("localm.plugins.coder.tools.subprocess.run", side_effect=fake_run):
            tool_run_tests(tmp_path, runner="pytest")
        assert "pytest" in " ".join(captured)

    def test_failure_returns_not_ok(self, tmp_path):
        with self._patch_run(stdout="1 failed\n", returncode=1):
            r = tool_run_tests(tmp_path)
        assert not r.ok
        assert "failed" in r.summary

    def test_timeout_returns_error(self, tmp_path):
        with patch(
            "localm.plugins.coder.tools.subprocess.run",
            side_effect=subprocess.TimeoutExpired(["pytest"], 120),
        ):
            r = tool_run_tests(tmp_path)
        assert not r.ok
        assert "timed out" in r.output.lower()

    def test_runner_not_found_returns_error(self, tmp_path):
        with patch(
            "localm.plugins.coder.tools.subprocess.run",
            side_effect=FileNotFoundError,
        ):
            r = tool_run_tests(tmp_path, runner="pytest")
        assert not r.ok
        assert "not found" in r.output.lower()

    def test_unknown_runner_returns_error(self, tmp_path):
        r = tool_run_tests(tmp_path, runner="not_a_runner")
        assert not r.ok
        assert "unknown runner" in r.output.lower()

    def test_path_arg_appended_to_command(self, tmp_path):
        captured = []

        def fake_run(cmd, **kwargs):
            captured.extend(cmd)
            return self._make_proc(stdout="ok")

        with patch("localm.plugins.coder.tools.subprocess.run", side_effect=fake_run):
            tool_run_tests(tmp_path, runner="pytest", path="tests/")
        assert "tests/" in captured

    def test_extra_args_appended(self, tmp_path):
        captured = []

        def fake_run(cmd, **kwargs):
            captured.extend(cmd)
            return self._make_proc(stdout="ok")

        with patch("localm.plugins.coder.tools.subprocess.run", side_effect=fake_run):
            tool_run_tests(tmp_path, runner="pytest", extra_args="-v -k test_foo")
        assert "-v" in captured
        assert "-k" in captured
        assert "test_foo" in captured

    def test_output_contains_runner_and_status(self, tmp_path):
        with self._patch_run(stdout="all good\n", returncode=0):
            r = tool_run_tests(tmp_path, runner="pytest")
        assert "<runner>" in r.output
        assert "<status>" in r.output
