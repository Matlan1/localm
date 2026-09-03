# SPDX-License-Identifier: AGPL-3.0-or-later
"""A stop or restart must not abandon a background job's child process.

``JobManager.start_cli`` runs ``python -m localm <cmd>`` as a real subprocess - a
model pull, a llama.cpp runtime provision, a ComfyUI setup. Nothing else reaps
that child:

  * ``_do_shutdown`` ends at ``os._exit(0)`` and ``_do_restart`` at ``os.execv``.
    Both bypass atexit.
  * the job's worker thread is a daemon, so its ``finally`` may never run.
  * the ``Popen`` carries no ``creationflags`` and no ``start_new_session``.

An abandoned child that writes NOTHING to stdout keeps running untracked, while
one that writes and flushes dies at its next write on the broken pipe, mid
operation with no cleanup and no record. Terminating deliberately leaves one
known state for the next start to report as "interrupted".

The first test here spawns a genuine child and asserts on genuine process
liveness, since a recording double's ``terminate`` being CALLED would also pass
against a kill that never reaches the process (the wrong pid, a signal the child
ignores, a tree that survives its root). The double-based tests below only cover
which exit paths CALL it.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import weakref

import pytest

from localm.inference import http_server as hs
from localm.plugins.gui import jobs as J


# --------------------------------------------------------------------------- #
#  helpers
# --------------------------------------------------------------------------- #

@pytest.fixture(autouse=True)
def isolated_managers(monkeypatch, tmp_path):
    """A fresh manager registry and activity dir per test.

    _MANAGERS is module state and a WeakSet's membership is only cleared by the
    garbage collector, so without this a manager built by an earlier test could
    still be reachable and the module-level terminate would act on ITS jobs."""
    monkeypatch.setattr(J, "_MANAGERS", weakref.WeakSet())
    monkeypatch.setattr(J, "activity_dir", lambda: tmp_path / "activity")


def _spawn_sleeper(seconds: int = 120) -> subprocess.Popen:
    return subprocess.Popen([sys.executable, "-c",
                             f"import time; time.sleep({seconds})"])


def _running_cli_job(manager, proc, *, kind="pull", label="Pull tiny/model.gguf"):
    """Register a job in *manager* the way start_cli does, wrapping *proc*.

    Built directly rather than by calling start_cli with a real localm command,
    every one of which would download something or provision a runtime. The
    child itself is not faked."""
    job = J.Job(id=f"{len(manager._jobs):012d}", kind=kind, argv=[], label=label)
    job._proc = proc
    manager._register(job)
    return job


def _dead(proc, timeout=15.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            return True
        time.sleep(0.05)
    return proc.poll() is not None


class _FakeProc:
    """A Popen stand-in for the tests that only ask WHICH PATHS call the kill."""

    def __init__(self, *, alive=True, pid=4242):
        self.pid = pid
        self._alive = alive
        self.terminated = 0
        self.killed = 0

    def poll(self):
        return None if self._alive else 0

    def terminate(self):
        self.terminated += 1
        self._alive = False

    def kill(self):
        self.killed += 1
        self._alive = False

    def wait(self, timeout=None):
        self._alive = False
        return 0


# --------------------------------------------------------------------------- #
#  the real property: a real child process is actually dead afterwards
# --------------------------------------------------------------------------- #

class TestRealChildIsKilled:

    def test_a_real_child_is_dead_after_the_exit_terminate(self):
        m = J.JobManager()
        proc = _spawn_sleeper()
        try:
            _running_cli_job(m, proc)
            assert proc.poll() is None, "the helper child did not start"
            assert J.terminate_children_for_exit() == 1
            assert _dead(proc), (
                "the job's child survived the stop: os._exit bypasses atexit and "
                "the worker thread is a daemon, so nothing else will ever reap it "
                "- a quiet child then keeps working untracked (measured)")
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=10)

    def test_a_grandchild_is_killed_too(self):
        """A start_cli child spawns its own children (comfy setup runs git and
        pip). Killing only the direct child strands those - which on Windows is
        why taskkill /T is used, and on POSIX why the tree is walked with psutil."""
        if sys.platform != "win32":
            pytest.importorskip(
                "psutil", reason="without psutil only the direct child is "
                                 "reachable on POSIX, which the code logs")
        script = (
            "import subprocess, sys, time\n"
            "k = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
            "print(k.pid, flush=True)\n"
            "time.sleep(120)\n"
        )
        parent = subprocess.Popen([sys.executable, "-c", script],
                                  stdout=subprocess.PIPE, text=True)
        kid_pid = None
        try:
            line = parent.stdout.readline().strip()
            kid_pid = int(line)
            from localm.instances import pid_alive
            assert pid_alive(kid_pid), "the grandchild did not start"
            m = J.JobManager()
            _running_cli_job(m, parent, kind="comfy-setup")
            assert J.terminate_children_for_exit() == 1
            assert _dead(parent)
            deadline = time.time() + 15
            while time.time() < deadline and pid_alive(kid_pid):
                time.sleep(0.05)
            assert not pid_alive(kid_pid), (
                f"grandchild {kid_pid} outlived the stop; a comfy setup's git or "
                "pip would keep running against a server that is gone")
        finally:
            if parent.poll() is None:
                parent.kill()
                parent.wait(timeout=10)
            if kid_pid:
                try:
                    from localm.instances import kill_pid
                    kill_pid(kid_pid)
                except Exception:
                    pass

    def test_the_registry_still_says_running_afterwards(self):
        """The row stays "running", and the next start reconciles it to
        "interrupted". Marking it "cancelled" here would claim the USER asked to
        stop it."""
        m = J.JobManager()
        proc = _spawn_sleeper()
        try:
            job = _running_cli_job(m, proc)
            J.terminate_children_for_exit()
            assert job.status == "running"
            assert not job.cancel_requested
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=10)


# --------------------------------------------------------------------------- #
#  what is and is not in scope for the kill
# --------------------------------------------------------------------------- #

class TestSelection:

    def test_a_finished_job_is_not_touched(self):
        m = J.JobManager()
        job = _running_cli_job(m, _FakeProc(alive=False))
        job.status = "done"
        assert J.terminate_children_for_exit() == 0
        assert job._proc.terminated == 0

    def test_a_child_that_already_exited_is_not_signalled(self):
        m = J.JobManager()
        _running_cli_job(m, _FakeProc(alive=False))
        assert J.terminate_children_for_exit() == 0

    def test_an_in_thread_job_has_no_child_to_kill(self):
        """start_fn jobs cooperate through cancel_event; there is no subprocess."""
        m = J.JobManager()
        job = J.Job(id="000000000001", kind="rag-index", argv=[])
        m._register(job)
        assert J.terminate_children_for_exit() == 0

    def test_every_running_child_is_signalled_not_just_the_first(self):
        m = J.JobManager()
        procs = [_FakeProc(pid=100 + i) for i in range(3)]
        for p in procs:
            _running_cli_job(m, p)
        assert J.terminate_children_for_exit() == 3

    def test_managers_are_reached_through_module_state(self):
        """The exit paths take no app and hold no manager, so they reach every
        manager through the module-level registry."""
        a, b = J.JobManager(), J.JobManager()
        _running_cli_job(a, _FakeProc(pid=1))
        _running_cli_job(b, _FakeProc(pid=2))
        assert J.terminate_children_for_exit() == 2

    def test_one_broken_manager_does_not_stop_the_others(self):
        good = J.JobManager()
        _running_cli_job(good, _FakeProc(pid=7))

        class _Broken:
            def terminate_children_for_exit(self, *, grace=0):
                raise RuntimeError("boom")

        broken = _Broken()
        J._MANAGERS.add(broken)
        assert J.terminate_children_for_exit() == 1


class TestTreeWalkOwnership:
    """The tree walk sends SIGTERM to every process it enumerates, so it may only
    ever be pointed at a process THIS one spawned.

    Both arms are needed: the first alone is satisfied by a walk that never runs,
    which would silently stop grandchildren being terminated at all."""

    @pytest.mark.skipif(sys.platform == "win32",
                        reason="the psutil tree walk is the POSIX arm; Windows "
                               "terminates the tree with taskkill /T")
    def test_tree_walk_refuses_a_pid_we_did_not_spawn(self, monkeypatch):
        psutil = pytest.importorskip("psutil")
        walked: list[int] = []
        monkeypatch.setattr(psutil.Process, "children",
                            lambda self, *a, **kw: (walked.append(self.pid) or []))
        # pid 1 is a real, live process that this test did not spawn. Its
        # recursive children are every process this user owns.
        J._terminate_process_tree(_FakeProc(pid=1), grace=0)
        assert walked == [], (
            f"walked the process tree of pid(s) {walked}, which this process did "
            "not spawn - every descendant would have been sent SIGTERM")

    @pytest.mark.skipif(sys.platform == "win32",
                        reason="the psutil tree walk is the POSIX arm; Windows "
                               "terminates the tree with taskkill /T")
    def test_tree_walk_still_happens_for_our_own_child(self, monkeypatch):
        psutil = pytest.importorskip("psutil")
        walked: list[int] = []
        proc = _spawn_sleeper(30)
        try:
            monkeypatch.setattr(psutil.Process, "children",
                                lambda self, *a, **kw: (walked.append(self.pid) or []))
            J._terminate_process_tree(proc, grace=0)
            assert walked == [proc.pid], (
                "the tree of our OWN child was not walked, so any grandchildren "
                "it spawned would be left running")
        finally:
            proc.kill()
            proc.wait(timeout=10)


# --------------------------------------------------------------------------- #
#  both exit paths must call it
# --------------------------------------------------------------------------- #

class TestExitPathsInvokeIt:

    @pytest.fixture
    def a_running_child(self, monkeypatch):
        monkeypatch.setattr(hs, "_engine", None)
        m = J.JobManager()
        proc = _FakeProc()
        _running_cli_job(m, proc)
        return proc

    def test_do_shutdown_terminates_the_child(self, a_running_child, monkeypatch):
        monkeypatch.setattr(os, "_exit",
                            lambda code: (_ for _ in ()).throw(SystemExit(code)))
        with pytest.raises(SystemExit):
            hs._do_shutdown()
        assert a_running_child.terminated or a_running_child.killed, (
            "stop left the job's child running: os._exit bypasses atexit, so "
            "nothing else reaps it")

    def test_do_restart_terminates_the_child(self, a_running_child, monkeypatch):
        monkeypatch.setattr(os, "execv",
                            lambda exe, argv: (_ for _ in ()).throw(SystemExit(0)))
        with pytest.raises(SystemExit):
            hs._do_restart()
        assert a_running_child.terminated or a_running_child.killed, (
            "restart left the OLD child running: os.execv replaces this process "
            "image but never touches a separate child, and bypasses atexit")

    def test_a_failure_in_the_kill_does_not_block_the_stop(self, monkeypatch):
        """The stop the user asked for outranks tidying up after a job."""
        monkeypatch.setattr(hs, "_engine", None)
        monkeypatch.setattr(J, "terminate_children_for_exit",
                            lambda **kw: (_ for _ in ()).throw(RuntimeError("boom")))
        monkeypatch.setattr(os, "_exit",
                            lambda code: (_ for _ in ()).throw(SystemExit(code)))
        with pytest.raises(SystemExit):
            hs._do_shutdown()
