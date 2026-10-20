# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for localm.plugins.coder.tools.base.run_subprocess - the canonical
subprocess-execution primitive shared by tools/shell.py, tools/git.py and
cli/goal.py. Each caller's own behaviour is covered where that caller is tested;
this file drives run_subprocess itself, directly."""

import os
import sys

from localm.plugins.coder.tools.base import SubprocessResult, run_subprocess
from tests.plugins.coder.launch_stub import patch_launch

_PY = sys.executable


class _Done:
    stdout = ""
    stderr = ""
    returncode = 0


class TestArgvMode:
    def test_success_returns_ok_result(self, tmp_path):
        r = run_subprocess([_PY, "-c", "print('hi')"], tmp_path, timeout=30)
        assert isinstance(r, SubprocessResult)
        assert r.ok is True
        assert r.returncode == 0
        assert r.stdout == "hi\n"

    def test_nonzero_exit_is_not_ok(self, tmp_path):
        r = run_subprocess([_PY, "-c", "import sys; sys.exit(1)"], tmp_path, timeout=30)
        assert r.ok is False
        assert r.returncode == 1

    def test_stderr_is_captured_separately(self, tmp_path):
        r = run_subprocess(
            [_PY, "-c", "import sys; print('o'); print('e', file=sys.stderr)"],
            tmp_path, timeout=30)
        assert r.stdout == "o\n"
        assert r.stderr == "e\n"

    def test_crlf_output_is_normalised_to_newlines(self, tmp_path):
        r = run_subprocess(
            [_PY, "-c",
             r"import sys; sys.stdout.buffer.write(b'a\r\nb\rc\n')"],
            tmp_path, timeout=30)
        assert r.stdout == "a\nb\nc\n"

    def test_undecodable_bytes_are_replaced(self, tmp_path):
        r = run_subprocess(
            [_PY, "-c", r"import sys; sys.stdout.buffer.write(b'ok\xff')"],
            tmp_path, timeout=30)
        assert r.stdout == "ok\ufffd"

    def test_argv_is_passed_through_unwrapped(self, tmp_path):
        r = run_subprocess(
            [_PY, "-c", "import sys; print(sys.argv[1])", "a b && c"],
            tmp_path, timeout=30)
        assert r.stdout.strip() == "a b && c"

    def test_runs_in_the_given_directory(self, tmp_path):
        r = run_subprocess(
            [_PY, "-c", "import os; print(os.getcwd())"], tmp_path, timeout=30)
        assert os.path.samefile(r.stdout.strip(), tmp_path)


class TestShellWrapMode:
    def test_wraps_command_string_through_platform_shell(self, tmp_path):
        captured = {}

        def fake_run(argv, **kwargs):
            captured["argv"] = argv
            return _Done()

        with patch_launch(side_effect=fake_run):
            run_subprocess("echo hi && echo bye", tmp_path, timeout=30, shell_wrap=True)

        launched = captured["argv"]
        if sys.platform == "win32":
            # The command is passed as a raw command line, not an argv list.
            assert launched == 'cmd /S /C "echo hi && echo bye"'
        else:
            assert launched[0] == "/bin/sh" and launched[1] == "-c"
            assert launched[-1] == "echo hi && echo bye"

    def test_the_wrapped_command_really_runs(self, tmp_path):
        r = run_subprocess("echo hi && echo bye", tmp_path, timeout=30,
                           shell_wrap=True)
        assert r.ok is True
        assert r.stdout.split() == ["hi", "bye"]

    def test_shell_wrap_passes_a_quoted_argument_through_unchanged(self, tmp_path):
        """The command text must reach the shell verbatim - re-quoting it is
        exactly the defect this wrapping was fixed for."""
        captured = {}

        def fake_run(argv, **kwargs):
            captured["argv"] = argv
            return _Done()

        command = 'type "a dir with spaces/f.txt"'
        with patch_launch(side_effect=fake_run):
            run_subprocess(command, tmp_path, timeout=30, shell_wrap=True)

        launched = captured["argv"]
        tail = launched if isinstance(launched, str) else launched[-1]
        if isinstance(launched, str):
            tail = tail[len('cmd /S /C "'):-1]
        assert tail == command, launched

    def test_a_quoted_path_argument_really_reaches_the_shell(self, tmp_path):
        """A path with spaces, quoted, is read by the real shell."""
        d = tmp_path / "a dir with spaces"
        d.mkdir()
        (d / "f.txt").write_text("found-it", encoding="utf-8")
        reader = "type" if sys.platform == "win32" else "cat"
        sep = "\\" if sys.platform == "win32" else "/"
        r = run_subprocess(f'{reader} "a dir with spaces{sep}f.txt"', tmp_path,
                           timeout=30, shell_wrap=True)
        assert r.stdout.strip() == "found-it", (r.stdout, r.stderr)

    def test_default_does_not_shell_wrap(self, tmp_path):
        r = run_subprocess([_PY, "-c", "import sys; print(sys.argv[1])", "x && y"],
                           tmp_path, timeout=30)
        assert r.stdout.strip() == "x && y"


class TestTimeoutHandling:
    def test_timeout_sets_timed_out_flag(self, tmp_path):
        r = run_subprocess([_PY, "-c", "import time; time.sleep(30)"], tmp_path,
                           timeout=0.5)
        assert r.timed_out is True
        assert r.ok is False

    def test_timeout_preserves_partial_output(self, tmp_path):
        r = run_subprocess(
            [_PY, "-c",
             "import sys, time; print('partial stdout', flush=True); "
             "print('partial stderr', file=sys.stderr, flush=True); time.sleep(30)"],
            tmp_path, timeout=1.5)
        assert r.timed_out is True
        assert r.stdout == "partial stdout\n"
        assert r.stderr == "partial stderr\n"


class TestLaunchFailures:
    def test_missing_executable_sets_not_found(self, tmp_path):
        r = run_subprocess(["ghost-binary-that-does-not-exist"], tmp_path, timeout=5)
        assert r.not_found is True
        assert r.ok is False

    def test_generic_exception_sets_error(self, tmp_path):
        with patch_launch(side_effect=OSError("permission denied")):
            r = run_subprocess(["cmd"], tmp_path, timeout=5)
        assert r.ok is False
        assert r.not_found is False
        assert "permission denied" in r.error


class TestEnvForwarding:
    def test_env_kwarg_reaches_the_child(self, tmp_path):
        env = dict(os.environ, LOCALM_RUN_SUBPROCESS_PROBE="bar")
        r = run_subprocess(
            [_PY, "-c", "import os; print(os.environ['LOCALM_RUN_SUBPROCESS_PROBE'])"],
            tmp_path, timeout=30, env=env)
        assert r.stdout.strip() == "bar"


class TestCancellable:
    """With a cancel check the process is killed the moment it answers True."""

    def test_a_cancel_kills_the_process_and_keeps_its_output(self, tmp_path):
        import sys
        import threading
        import time
        flag = threading.Event()
        threading.Timer(0.4, flag.set).start()
        t0 = time.monotonic()
        result = run_subprocess(
            [sys.executable, "-c",
             "import sys, time; print('started', flush=True); time.sleep(8)"],
            tmp_path, timeout=60, cancel=flag.is_set)
        elapsed = time.monotonic() - t0
        assert result.cancelled is True
        assert result.ok is False and result.timed_out is False
        assert "started" in (result.stdout or "")
        assert elapsed < 4.0, f"the process outlived the cancel ({elapsed:.1f}s)"

    def test_without_a_cancel_the_timeout_still_bounds_the_process(self, tmp_path):
        import sys
        result = run_subprocess(
            [sys.executable, "-c", "import time; time.sleep(8)"],
            tmp_path, timeout=0.5, cancel=lambda: False)
        assert result.timed_out is True and result.cancelled is False

    def test_a_finished_process_is_reported_as_before(self, tmp_path):
        import sys
        result = run_subprocess(
            [sys.executable, "-c", "print('hi')"], tmp_path, timeout=30,
            cancel=lambda: False)
        assert result.ok is True and result.returncode == 0
        assert result.stdout.strip() == "hi"


    def test_a_shell_routed_command_is_killed_with_its_shell(self, tmp_path):
        """`cmd /C` (or `sh -c`) is only the parent: the process doing the work
        must die with it, or the cancel reports "killed" only once the work has
        finished on its own."""
        import sys
        import threading
        import time
        flag = threading.Event()
        threading.Timer(0.5, flag.set).start()
        cmd = f'{sys.executable} -c "import time; time.sleep(8)" && echo done'
        t0 = time.monotonic()
        result = run_subprocess(cmd, tmp_path, timeout=60, shell_wrap=True,
                                cancel=flag.is_set)
        elapsed = time.monotonic() - t0
        assert result.cancelled is True
        assert elapsed < 4.0, f"the shell's child outlived the cancel ({elapsed:.1f}s)"

    def test_an_interrupt_during_the_wait_kills_the_process(self, tmp_path):
        import sys
        import time
        import psutil
        import pytest
        pid_file = tmp_path / "pid.txt"
        calls = []

        def cancel():
            calls.append(1)
            if len(calls) == 3:
                raise KeyboardInterrupt
            return False

        with pytest.raises(KeyboardInterrupt):
            run_subprocess(
                [sys.executable, "-c",
                 f"import os, time; open(r'{pid_file}', 'w').write(str(os.getpid())); "
                 "time.sleep(8)"],
                tmp_path, timeout=60, cancel=cancel)
        assert pid_file.exists(), "the child never started"
        pid = int(pid_file.read_text())
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and psutil.pid_exists(pid):
            time.sleep(0.05)
        assert not psutil.pid_exists(pid), "the child survived the interrupt"


_ESCAPEE = (
    "import os, subprocess, sys\n"
    "pid_file = sys.argv[1]\n"
    "kw = dict(stdin=subprocess.DEVNULL, stdout=1, stderr=2)\n"
    "if sys.platform == 'win32':\n"
    "    kw['creationflags'] = 0x00000008\n"
    "else:\n"
    "    kw['start_new_session'] = True\n"
    "code = ('import os, time; open(%r, \"w\").write(str(os.getpid())); time.sleep(15)'\n"
    "        % pid_file)\n"
    "subprocess.Popen([sys.executable, '-c', code], **kw)\n"
)


def _escapee_command(tmp_path, tail):
    """A command whose middle process starts a detached grandchild that
    inherits stdout/stderr and sleeps 15 s, then exits (so the grandchild is no
    longer a descendant of the launched process), followed by *tail*."""
    import sys
    script = tmp_path / "escapee.py"
    script.write_text(_ESCAPEE, encoding="utf-8")
    pid_file = tmp_path / "escapee.pid"
    outer = (
        "import subprocess, sys, time\n"
        f"subprocess.run([sys.executable, r'{script}', r'{pid_file}'], check=True)\n"
        "print('started', flush=True)\n"
        f"{tail}\n"
    )
    outer_file = tmp_path / "outer.py"
    outer_file.write_text(outer, encoding="utf-8")
    return [sys.executable, str(outer_file)], pid_file


def _reap(pid_file):
    import psutil
    try:
        psutil.Process(int(pid_file.read_text())).kill()
    except Exception:
        pass


class TestEscapedPipeHolder:
    """A grandchild that left the process tree but kept the output pipe open
    must not stretch a timeout or a cancel."""

    _BOUND = 6.0

    def _run(self, tmp_path, tail, cancel, timeout=1.0):
        import time
        argv, pid_file = _escapee_command(tmp_path, tail)
        try:
            t0 = time.monotonic()
            result = run_subprocess(argv, tmp_path, timeout=timeout, cancel=cancel)
            return result, time.monotonic() - t0
        finally:
            _reap(pid_file)

    def test_timeout_is_not_outlived_by_an_escaped_holder(self, tmp_path):
        result, elapsed = self._run(tmp_path, "time.sleep(30)", cancel=None)
        assert result.timed_out is True
        assert "started" in (result.stdout or "")
        assert elapsed < self._BOUND, f"timeout outlived by the escapee ({elapsed:.1f}s)"

    def test_cancellable_timeout_is_not_outlived_by_an_escaped_holder(self, tmp_path):
        result, elapsed = self._run(tmp_path, "time.sleep(30)", cancel=lambda: False)
        assert result.timed_out is True
        assert "started" in (result.stdout or "")
        assert elapsed < self._BOUND, f"timeout outlived by the escapee ({elapsed:.1f}s)"

    def test_cancel_is_not_outlived_by_an_escaped_holder(self, tmp_path):
        import threading
        flag = threading.Event()
        threading.Timer(1.5, flag.set).start()
        result, elapsed = self._run(
            tmp_path, "time.sleep(30)", cancel=flag.is_set, timeout=60)
        assert result.cancelled is True
        assert "started" in (result.stdout or "")
        assert elapsed < self._BOUND + 1.5, f"cancel outlived by the escapee ({elapsed:.1f}s)"

    def test_a_finished_process_is_not_held_by_an_escaped_holder(self, tmp_path):
        for cancel in (None, lambda: False):
            sub = tmp_path / ("c" if cancel else "n")
            sub.mkdir()
            result, elapsed = self._run(sub, "pass", cancel=cancel, timeout=30)
            assert result.ok is True and result.returncode == 0
            assert result.stdout.strip() == "started"
            assert elapsed < self._BOUND, f"exit held by the escapee ({elapsed:.1f}s)"
