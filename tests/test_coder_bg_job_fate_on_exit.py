# SPDX-License-Identifier: AGPL-3.0-or-later
"""A stop or restart must not abandon the coder plugin's own background jobs.

``localm.plugins.coder.background.JobRegistry`` reaps its running shell/agent
jobs through an atexit hook (``JobRegistry.__init__`` ->
``atexit.register(shutdown_all)``). Both server exit paths bypass atexit:

  * ``_do_shutdown`` ends at ``os._exit(0)``.
  * ``_do_restart`` ends at ``os.execv``.

Mirrors ``tests/test_job_child_fate_on_exit.py`` (the GUI's own job registry),
which the same ``os._exit``/``os.execv`` bypass affects identically. The first
two tests spawn a genuine child and assert on genuine process liveness, since
a recording double's ``kill()`` being CALLED would also pass against a kill
that never reaches the process.
"""

from __future__ import annotations

import os
import sys
import time

import pytest

from localm.inference import http_server as hs
from localm.plugins.coder import background as bg
from localm.plugins.coder.tools.shell import tool_run_shell_background


@pytest.fixture(autouse=True)
def _clean_registry():
    bg.reset_registry()
    yield
    bg.reset_registry()


@pytest.fixture
def _sleeper(tmp_path):
    """A shell command that sleeps, via a script file rather than ``-c`` - a
    Windows cmd /C quoting limitation, not something this test is about."""
    def _make(seconds: int = 120) -> str:
        script = tmp_path / "_bg_sleeper.py"
        script.write_text(f"import time; time.sleep({seconds})\n", encoding="utf-8")
        return f"{sys.executable} -u {script.name}"
    return _make


def _job_id(result) -> str:
    import re
    m = re.search(r"<job>(job_[0-9a-f]+)</job>", result.output)
    assert m, f"no job id in: {result.output}"
    return m.group(1)


def _pid_alive(pid: int) -> bool:
    """Ground truth from the OS, not from the job's own bookkeeping."""
    if sys.platform == "win32":
        import subprocess
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                             capture_output=True, text=True).stdout
        return str(pid) in out
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _force_kill(pid: int) -> None:
    import subprocess
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
    else:
        try:
            os.kill(pid, 9)
        except OSError:
            pass


def _wait_for(predicate, timeout=15.0, interval=0.05) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def _spawn_background_job(tmp_path, _sleeper):
    res = tool_run_shell_background(tmp_path, _sleeper())
    assert res.ok, res.output
    job = bg.get_registry().get(_job_id(res))
    pid = job.pid
    assert _pid_alive(pid), "the background job did not start"
    return pid


class TestRealChildIsKilled:

    def test_do_shutdown_kills_a_real_background_shell_job(
            self, tmp_path, _sleeper, monkeypatch):
        monkeypatch.setattr(hs, "_engine", None)
        pid = _spawn_background_job(tmp_path, _sleeper)
        monkeypatch.setattr(os, "_exit",
                            lambda code: (_ for _ in ()).throw(SystemExit(code)))
        try:
            with pytest.raises(SystemExit):
                hs._do_shutdown()
            assert _wait_for(lambda: not _pid_alive(pid)), (
                f"pid {pid} survived _do_shutdown: os._exit bypasses the "
                "atexit hook JobRegistry relies on, so nothing else reaps it "
                "(measured)")
        finally:
            if _pid_alive(pid):
                _force_kill(pid)

    def test_do_restart_kills_a_real_background_shell_job(
            self, tmp_path, _sleeper, monkeypatch):
        monkeypatch.setattr(hs, "_engine", None)
        pid = _spawn_background_job(tmp_path, _sleeper)
        monkeypatch.setattr(os, "execv",
                            lambda exe, argv: (_ for _ in ()).throw(SystemExit(0)))
        try:
            with pytest.raises(SystemExit):
                hs._do_restart()
            assert _wait_for(lambda: not _pid_alive(pid)), (
                f"pid {pid} survived _do_restart: os.execv bypasses atexit "
                "too (measured)")
        finally:
            if _pid_alive(pid):
                _force_kill(pid)


class TestExitPathsInvokeIt:
    """Double-based: which paths CALL the teardown, without paying for a real
    subprocess each time."""

    @pytest.fixture
    def _spy(self, monkeypatch):
        calls = []
        monkeypatch.setattr(bg, "terminate_all_for_exit",
                            lambda: calls.append(1) or 1)
        return calls

    def test_do_shutdown_calls_it(self, monkeypatch, _spy):
        monkeypatch.setattr(hs, "_engine", None)
        monkeypatch.setattr(os, "_exit",
                            lambda code: (_ for _ in ()).throw(SystemExit(code)))
        with pytest.raises(SystemExit):
            hs._do_shutdown()
        assert _spy, "shutdown never called terminate_all_for_exit"

    def test_do_restart_calls_it(self, monkeypatch, _spy):
        monkeypatch.setattr(hs, "_engine", None)
        monkeypatch.setattr(os, "execv",
                            lambda exe, argv: (_ for _ in ()).throw(SystemExit(0)))
        with pytest.raises(SystemExit):
            hs._do_restart()
        assert _spy, "restart never called terminate_all_for_exit"

    def test_a_failure_in_the_kill_does_not_block_the_stop(self, monkeypatch):
        """The stop the user asked for outranks tidying up after a coder job."""
        monkeypatch.setattr(hs, "_engine", None)
        monkeypatch.setattr(bg, "terminate_all_for_exit",
                            lambda: (_ for _ in ()).throw(RuntimeError("boom")))
        monkeypatch.setattr(os, "_exit",
                            lambda code: (_ for _ in ()).throw(SystemExit(code)))
        with pytest.raises(SystemExit):
            hs._do_shutdown()
