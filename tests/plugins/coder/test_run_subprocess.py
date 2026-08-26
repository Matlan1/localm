# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for localm.plugins.coder.tools.base.run_subprocess - the canonical
subprocess-execution primitive shared by tools/shell.py, tools/git.py and
cli/goal.py. Each caller's own behaviour is covered where that caller is tested;
this file drives run_subprocess itself, directly."""

import subprocess
from unittest.mock import MagicMock, patch

from localm.plugins.coder.tools.base import SubprocessResult, run_subprocess

_RUN = "localm.plugins.coder.tools.base.subprocess.run"


def _make_proc(stdout="", stderr="", returncode=0):
    p = MagicMock()
    p.stdout = stdout
    p.stderr = stderr
    p.returncode = returncode
    return p


class TestArgvMode:
    def test_success_returns_ok_result(self, tmp_path):
        with patch(_RUN, return_value=_make_proc(stdout="hi\n", returncode=0)):
            r = run_subprocess(["echo", "hi"], tmp_path, timeout=10)
        assert isinstance(r, SubprocessResult)
        assert r.ok is True
        assert r.returncode == 0
        assert r.stdout == "hi\n"

    def test_nonzero_exit_is_not_ok(self, tmp_path):
        with patch(_RUN, return_value=_make_proc(returncode=1)):
            r = run_subprocess(["false"], tmp_path, timeout=10)
        assert r.ok is False
        assert r.returncode == 1

    def test_argv_passed_through_unwrapped(self, tmp_path):
        captured = []

        def fake_run(argv, **kwargs):
            captured.extend(argv)
            return _make_proc()

        with patch(_RUN, side_effect=fake_run):
            run_subprocess(["git", "status"], tmp_path, timeout=10)
        assert captured == ["git", "status"]


class TestShellWrapMode:
    def test_wraps_command_string_through_platform_shell(self, tmp_path):
        import sys
        captured = {}

        def fake_run(argv, **kwargs):
            captured["argv"] = argv
            return _make_proc()

        with patch(_RUN, side_effect=fake_run):
            run_subprocess("echo hi && echo bye", tmp_path, timeout=10, shell_wrap=True)

        launched = captured["argv"]
        if sys.platform == "win32":
            # The command is passed as a raw command line, not an argv list.
            assert launched == "cmd /C echo hi && echo bye"
        else:
            assert launched[0] == "/bin/sh" and launched[1] == "-c"
            assert launched[-1] == "echo hi && echo bye"

    def test_shell_wrap_passes_a_quoted_argument_through_unchanged(self, tmp_path):
        """The command text must reach the shell verbatim - re-quoting it is
        exactly the defect this wrapping was fixed for."""
        captured = {}

        def fake_run(argv, **kwargs):
            captured["argv"] = argv
            return _make_proc()

        command = 'type "a dir with spaces/f.txt"'
        with patch(_RUN, side_effect=fake_run):
            run_subprocess(command, tmp_path, timeout=10, shell_wrap=True)

        launched = captured["argv"]
        tail = launched if isinstance(launched, str) else launched[-1]
        assert tail.endswith(command), tail

    def test_default_does_not_shell_wrap(self, tmp_path):
        captured = []

        def fake_run(argv, **kwargs):
            captured.extend(argv)
            return _make_proc()

        with patch(_RUN, side_effect=fake_run):
            run_subprocess(["echo", "hi"], tmp_path, timeout=10)
        assert captured == ["echo", "hi"]


class TestTimeoutHandling:
    def test_timeout_sets_timed_out_flag(self, tmp_path):
        with patch(_RUN, side_effect=subprocess.TimeoutExpired(["cmd"], 5)):
            r = run_subprocess(["cmd"], tmp_path, timeout=5)
        assert r.timed_out is True
        assert r.ok is False

    def test_timeout_preserves_partial_output(self, tmp_path):
        exc = subprocess.TimeoutExpired(["cmd"], 5, output="partial stdout",
                                        stderr="partial stderr")
        with patch(_RUN, side_effect=exc):
            r = run_subprocess(["cmd"], tmp_path, timeout=5)
        assert r.stdout == "partial stdout"
        assert r.stderr == "partial stderr"


class TestLaunchFailures:
    def test_missing_executable_sets_not_found(self, tmp_path):
        with patch(_RUN, side_effect=FileNotFoundError("no such file")):
            r = run_subprocess(["ghost-binary"], tmp_path, timeout=5)
        assert r.not_found is True
        assert r.ok is False

    def test_generic_exception_sets_error(self, tmp_path):
        with patch(_RUN, side_effect=OSError("permission denied")):
            r = run_subprocess(["cmd"], tmp_path, timeout=5)
        assert r.ok is False
        assert r.not_found is False
        assert "permission denied" in r.error


class TestEnvForwarding:
    def test_env_kwarg_passed_to_subprocess_run(self, tmp_path):
        captured = {}

        def fake_run(argv, **kwargs):
            captured.update(kwargs)
            return _make_proc()

        with patch(_RUN, side_effect=fake_run):
            run_subprocess(["cmd"], tmp_path, timeout=5, env={"FOO": "bar"})
        assert captured.get("env") == {"FOO": "bar"}
