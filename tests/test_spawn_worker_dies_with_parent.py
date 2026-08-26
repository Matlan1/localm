# SPDX-License-Identifier: AGPL-3.0-or-later
"""A spawned model worker must die when its parent process dies - however the
parent died, including an uncatchable HARD kill (Windows TerminateProcess / Task
Manager "End Task", POSIX SIGKILL) where NO parent-side code runs.

Every isolated worker is spawned ``daemon=True``, and multiprocessing's
daemon-child reclamation is an atexit hook
(``multiprocessing.util._exit_function``), which never runs under a hard kill, so
the daemon flag does not save the worker. The graceful paths
(``ModelRunner.shutdown()`` / ``release_for_exit()``) only run when parent-side
code runs, which here it does not.

This test drives the REAL failure mode: a REAL worker under a REAL subprocess
parent, hard-killed via ``Popen.kill()`` (TerminateProcess on Windows, SIGKILL on
POSIX). It does NOT mock the worker, the spawn, or the kill. Idle workers only
(no model load), so it needs no model, GPU, or native runtime.

Under test: ``install_parent_death_watchdog()`` (localm/_mp_spawn.py), installed
at the top of each worker main, which blocks on multiprocessing's already-open
parent sentinel and ``os._exit(0)``s the instant the parent dies.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

import psutil
import pytest

# Import the WORKTREE's edited localm, not the venv's editable install of the MAIN
# checkout: point the helper subprocess at the repo root that contains THIS test.
_WORKTREE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_HELPER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_orphan_worker_helper.py")


def _read_worker_pid(parent: subprocess.Popen, timeout: float = 60.0) -> int:
    """Block until the helper prints its worker PID (its first stdout line)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        line = parent.stdout.readline()
        if line:
            return int(line.strip())
        if parent.poll() is not None:
            raise AssertionError(
                f"helper exited (code {parent.returncode}) before printing a PID; "
                f"stderr:\n{parent.stderr.read()}")
    raise AssertionError("helper never printed a worker PID")


def _is_gone(proc: psutil.Process) -> bool:
    """True if *proc* is no longer a live process. psutil's is_running() is
    identity-safe (it matches creation time), so a recycled PID reads as gone.
    A process that exits between is_running() and status() raises NoSuchProcess -
    which also means gone, so treat it as such rather than letting it flake."""
    try:
        return not proc.is_running() or proc.status() == psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return True


def _wait_gone(proc: psutil.Process, timeout: float) -> bool:
    """Poll until *proc* is gone or *timeout* elapses."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _is_gone(proc):
            return True
        time.sleep(0.2)
    return _is_gone(proc)


@pytest.mark.parametrize("kind", ["gguf", "embedder", "voice"])
def test_spawn_worker_dies_when_parent_is_hard_killed(kind: str) -> None:
    env = dict(os.environ, PYTHONPATH=_WORKTREE_ROOT, PYTHONUNBUFFERED="1")
    parent = subprocess.Popen(
        [sys.executable, _HELPER, kind],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)

    worker = None
    try:
        wpid = _read_worker_pid(parent)
        worker = psutil.Process(wpid)

        # The worker is genuinely alive BEFORE the parent is touched, so a later
        # disappearance can only come from the parent's death.
        time.sleep(1.0)
        assert not _is_gone(worker), (
            f"{kind} worker was not alive before the kill; nothing to prove")
        assert parent.poll() is None, "parent died on its own before the kill"

        # The uncatchable hard kill - no atexit, no finally, no signal handler runs.
        parent.kill()
        parent.wait(timeout=10)

        # The whole property: the worker must die on its OWN, via the watchdog.
        assert _wait_gone(worker, timeout=20), (
            f"{kind} worker (pid {wpid}) SURVIVED a hard kill of its parent - it "
            "would keep its model resident in VRAM indefinitely as an orphan")
    finally:
        # Clean up only the exact processes we spawned (identity-guarded by psutil).
        if worker is not None:
            try:
                if worker.is_running():
                    worker.kill()
            except psutil.Error:
                pass
        if parent.poll() is None:
            parent.kill()
        try:
            parent.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
