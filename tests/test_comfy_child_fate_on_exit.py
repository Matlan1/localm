# SPDX-License-Identifier: AGPL-3.0-or-later
"""A stop or restart must not abandon a ComfyUI instance localm itself
launched.

localm spawns ComfyUI (its own managed instance, or a user's via
comfy_launch_cmd) in a DETACHED process group specifically so stop_comfy()
can kill its whole tree on demand (see comfy_client.py's _launch_and_wait).
That same detachment means the child does NOT die on its own when the
localm server process exits or re-execs:

  * _do_shutdown ends at os._exit(0) and _do_restart at os.execv. Both
    bypass atexit.
  * CREATE_NEW_PROCESS_GROUP (Windows) / start_new_session (POSIX) is exactly
    what keeps a Ctrl+C or an ordinary parent-death signal from reaching it.

An abandoned ComfyUI keeps running, holding whatever VRAM/RAM its last job
loaded, invisible to the next server start (a fresh process has an empty
_spawned_procs).

The first test class here spawns a genuine child and asserts on genuine
process liveness, for the same reason test_job_child_fate_on_exit.py's does:
a recording double's stop_comfy being CALLED would also pass against a kill
that never reaches the process (the wrong pid, a signal the child ignores).
The double-based tests below only cover which exit paths call it.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

import pytest

from localm.inference import http_server as hs
from localm.media import comfy_client as cc


class _RecordingProc:
    """A Popen stand-in that records which handle call it received."""

    def __init__(self, pid: int = 4242) -> None:
        self.pid = pid
        self.terminated = 0
        self.killed = 0

    def terminate(self) -> None:
        self.terminated += 1

    def kill(self) -> None:
        self.killed += 1

    def wait(self, timeout=None):
        return 0


class TestKillProcessTreeNeverBroadcasts:
    """killpg(pgid, sig) is kill(-pgid, sig).

    pgid 1 is therefore the kill(2) BROADCAST - every process this user may
    signal - and pgid 0 is this process's own group. Either one takes localm
    down with the child it meant to stop, and on a CI runner it also signals the
    runner agent, which then reports a shutdown signal and cancels the job."""

    @pytest.mark.skipif(sys.platform == "win32",
                        reason="the killpg arm is POSIX; Windows uses taskkill /T")
    @pytest.mark.parametrize("pgid", [1, 0])
    def test_kill_process_tree_never_broadcasts(self, monkeypatch, pgid):
        sent = []
        monkeypatch.setattr(os, "getpgid", lambda _pid: pgid)
        monkeypatch.setattr(os, "killpg",
                            lambda g, s: sent.append((g, s)))
        proc = _RecordingProc()
        cc._kill_process_tree(proc)
        assert sent == [], (
            f"sent killpg{sent} - pgid {pgid} signals every process this user "
            "owns, including the test runner and the CI runner agent")
        assert proc.terminated == 1, (
            "fell back to no signal at all; the child must still be stopped "
            "through its own handle")

    @pytest.mark.skipif(sys.platform == "win32",
                        reason="the killpg arm is POSIX; Windows uses taskkill /T")
    def test_kill_process_tree_still_signals_a_real_child_group(self, monkeypatch):
        """The guard must not disable group termination for a genuine group."""
        sent = []
        monkeypatch.setattr(os, "getpgid", lambda _pid: 987654)
        monkeypatch.setattr(os, "killpg", lambda g, s: sent.append((g, s)))
        proc = _RecordingProc()
        cc._kill_process_tree(proc)
        assert [g for g, _ in sent] == [987654], (
            "a real, non-broadcast child group was not signalled, so the tree "
            "would be left running")


def _spawn_sleeper(seconds: int = 120) -> subprocess.Popen:
    """Spawn a real child the same way _launch_and_wait spawns ComfyUI: its own
    process group/session on POSIX (start_new_session), its own process group
    on Windows (CREATE_NEW_PROCESS_GROUP). _kill_process_tree's POSIX arm does
    os.killpg(os.getpgid(pid), ...) on the assumption that the target has its
    OWN group - a plain Popen() without this shares the CALLER's group, so
    killpg would signal the test runner's own process group too."""
    popen_kw: dict = {}
    if sys.platform == "win32":
        popen_kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen_kw["start_new_session"] = True
    return subprocess.Popen([sys.executable, "-c",
                             f"import time; time.sleep({seconds})"], **popen_kw)


def _dead(proc, timeout=15.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            return True
        time.sleep(0.05)
    return proc.poll() is not None


@pytest.fixture(autouse=True)
def isolated_spawned_procs(monkeypatch):
    """A fresh _spawned_procs dict per test - module-level state shared
    across every caller (see test_comfy_console_warnings.py's own note on
    this same hazard)."""
    monkeypatch.setattr(cc, "_spawned_procs", {})


class _FakeProc:
    """A Popen stand-in for the tests that only ask WHICH PATHS call the stop."""
    def __init__(self, *, alive=True, pid=4242):
        self.pid = pid
        self._alive = alive

    def poll(self):
        return None if self._alive else 0


def _patch_network(monkeypatch, *, alive_after=False):
    """Stub the network side-effects stop_comfy makes so a fake-proc test
    never touches a real ComfyUI - mirrors test_stopcomfy_2026_07_01.py's
    _patch_common."""
    monkeypatch.setattr(cc, "interrupt_comfy", lambda url: True)
    monkeypatch.setattr(cc, "free_comfy_vram", lambda url=None: True)
    monkeypatch.setattr(cc, "_comfy_alive", lambda url, timeout=3.0: alive_after)


# --------------------------------------------------------------------------- #
#  the real property: a real spawned ComfyUI is actually dead afterwards
# --------------------------------------------------------------------------- #

class TestRealChildIsKilled:

    def test_a_real_spawned_comfy_is_dead_after_shutdown(self, monkeypatch):
        monkeypatch.setattr(hs, "_engine", None)
        _patch_network(monkeypatch)
        proc = _spawn_sleeper()
        try:
            cc._remember_spawned("http://127.0.0.1:8188", proc)
            assert proc.poll() is None, "the helper child did not start"
            monkeypatch.setattr(
                os, "_exit", lambda code: (_ for _ in ()).throw(SystemExit(code)))
            with pytest.raises(SystemExit):
                hs._do_shutdown()
            assert _dead(proc), (
                "the ComfyUI localm launched survived the stop: a detached "
                "process group does not die with the parent, and os._exit "
                "bypasses atexit - a quiet child then keeps running orphaned "
                "(measured)")
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=10)

    def test_a_real_spawned_comfy_is_dead_after_restart(self, monkeypatch):
        monkeypatch.setattr(hs, "_engine", None)
        _patch_network(monkeypatch)
        proc = _spawn_sleeper()
        try:
            cc._remember_spawned("http://127.0.0.1:8188", proc)
            assert proc.poll() is None, "the helper child did not start"
            monkeypatch.setattr(
                os, "execv", lambda exe, argv: (_ for _ in ()).throw(SystemExit(0)))
            with pytest.raises(SystemExit):
                hs._do_restart()
            assert _dead(proc), (
                "the OLD ComfyUI survived the restart: os.execv replaces "
                "this process image but never touches a detached child")
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=10)


# --------------------------------------------------------------------------- #
#  stop_all_spawned_comfy: multiple instances, partial failure
# --------------------------------------------------------------------------- #

class TestStopAllSpawned:

    def test_stops_every_tracked_instance_not_just_the_first(self, monkeypatch):
        _patch_network(monkeypatch)
        killed = []
        monkeypatch.setattr(cc, "_kill_process_tree",
                            lambda proc: killed.append(proc.pid))
        for i, url in enumerate(("http://127.0.0.1:8188",
                                 "http://127.0.0.1:8189",
                                 "http://127.0.0.1:8190")):
            cc._remember_spawned(url, _FakeProc(pid=100 + i))
        assert cc.stop_all_spawned_comfy() == 3
        assert sorted(killed) == [100, 101, 102]

    def test_no_spawned_instances_is_a_noop(self, monkeypatch):
        _patch_network(monkeypatch)
        assert cc.stop_all_spawned_comfy() == 0

    def test_one_instance_failing_to_stop_does_not_block_the_others(self, monkeypatch):
        _patch_network(monkeypatch)
        killed = []

        def _kill(proc):
            if proc.pid == 200:
                raise RuntimeError("boom")
            killed.append(proc.pid)
        monkeypatch.setattr(cc, "_kill_process_tree", _kill)
        cc._remember_spawned("http://127.0.0.1:8188", _FakeProc(pid=200))
        cc._remember_spawned("http://127.0.0.1:8189", _FakeProc(pid=201))
        assert cc.stop_all_spawned_comfy() == 1   # only the healthy one counted
        assert killed == [201]


# --------------------------------------------------------------------------- #
#  both exit paths must call it
# --------------------------------------------------------------------------- #

class TestExitPathsInvokeIt:

    @pytest.fixture
    def a_spawned_instance(self, monkeypatch):
        monkeypatch.setattr(hs, "_engine", None)
        _patch_network(monkeypatch)
        killed = []
        monkeypatch.setattr(cc, "_kill_process_tree",
                            lambda proc: killed.append(proc.pid))
        cc._remember_spawned("http://127.0.0.1:8188", _FakeProc(pid=321))
        return killed

    def test_do_shutdown_stops_the_spawned_comfy(self, a_spawned_instance, monkeypatch):
        monkeypatch.setattr(os, "_exit",
                            lambda code: (_ for _ in ()).throw(SystemExit(code)))
        with pytest.raises(SystemExit):
            hs._do_shutdown()
        assert a_spawned_instance == [321], (
            "stop left the launched ComfyUI running: os._exit bypasses "
            "atexit, so nothing else reaps it")

    def test_do_restart_stops_the_spawned_comfy(self, a_spawned_instance, monkeypatch):
        monkeypatch.setattr(os, "execv",
                            lambda exe, argv: (_ for _ in ()).throw(SystemExit(0)))
        with pytest.raises(SystemExit):
            hs._do_restart()
        assert a_spawned_instance == [321], (
            "restart left the OLD ComfyUI running: os.execv replaces this "
            "process image but never touches a separate detached child")

    def test_a_failure_in_the_comfy_stop_does_not_block_the_shutdown(self, monkeypatch):
        """The stop the user asked for outranks tidying up ComfyUI."""
        monkeypatch.setattr(hs, "_engine", None)
        monkeypatch.setattr(cc, "stop_all_spawned_comfy",
                            lambda: (_ for _ in ()).throw(RuntimeError("boom")))
        monkeypatch.setattr(os, "_exit",
                            lambda code: (_ for _ in ()).throw(SystemExit(code)))
        with pytest.raises(SystemExit):
            hs._do_shutdown()
