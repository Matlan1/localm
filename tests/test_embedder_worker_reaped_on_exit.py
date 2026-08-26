# SPDX-License-Identifier: AGPL-3.0-or-later
"""Restart/stop must not leak the embedder worker.

_do_shutdown and _do_restart skip releasing the embedder while a request is
mid-embed(). A daemon child is NOT reclaimed on the two paths those functions
take:

  - _do_shutdown ends at os._exit(0), and _do_restart at os.execv(). Both
    bypass atexit - and multiprocessing's daemon-child reclamation IS an atexit
    hook (multiprocessing.util._exit_function). os.execv additionally keeps the
    same PID and replaces only THIS process's image; it never touches the
    separate `localm-embedder-worker` child.

So a restart mid-index would leave the worker orphaned with its model resident
in VRAM, and a "stopped" server would leave a GPU-resident zombie.
"""

from __future__ import annotations

import os
import threading

import pytest

from localm.inference import embedder as emb
from localm.inference import http_server as hs
from localm.inference._embedder_runner import EmbedderRunner


@pytest.fixture(autouse=True)
def _no_singleton():
    """Never leave this module's _EMBEDDER pointing at a test double."""
    before = emb._EMBEDDER
    yield
    emb._EMBEDDER = before


def _isolated_embedder_with(runner, *, active):
    """An IsolatedEmbedder wrapping *runner*, built without a real model load."""
    e = emb.IsolatedEmbedder.__new__(emb.IsolatedEmbedder)
    e.model_path = "does-not-matter.gguf"
    e.active_requests = active
    e._runner = runner
    e.dim = 4
    return e


# --------------------------------------------------------------------------- #
# A real worker process is dead afterwards.
# --------------------------------------------------------------------------- #

def test_release_for_exit_kills_a_real_busy_worker_process():
    """The release must terminate the real child. Uses a REAL spawned worker
    (as the crash-containment tests do) and asserts on real process liveness,
    not on a call being recorded."""
    runner = EmbedderRunner()
    runner._spawn()
    try:
        assert runner.is_alive(), "worker did not start"
        proc = runner._proc
        # A request is mid-embed().
        emb._EMBEDDER = _isolated_embedder_with(runner, active=1)

        assert emb.release_for_exit() is True
        proc.join(timeout=10)
        assert not proc.is_alive(), (
            "the embedder worker survived the release and would outlive an "
            "os._exit/os.execv as a VRAM-resident zombie")
    finally:
        runner.shutdown(grace=0)


def test_release_for_exit_never_takes_the_load_lock():
    """Every other way to make the release decision - active_requests(),
    reset_embedder() - takes _LOCK, which get_embedder() holds for the FULL
    duration of an embedding-model load. A stop or restart during a load must
    not block on it.

    Holds _LOCK from another thread (exactly as an in-progress load does) and
    requires the release to complete anyway."""
    runner = _RecordingRunner()
    emb._EMBEDDER = _isolated_embedder_with(runner, active=1)
    released = []
    holding = threading.Event()
    finished = threading.Event()

    def _hold_lock_like_a_load():
        with emb._LOCK:
            holding.set()
            finished.wait(10)          # a real load holds this for its whole run

    t = threading.Thread(target=_hold_lock_like_a_load, daemon=True)
    t.start()
    try:
        assert holding.wait(5), "helper never took _LOCK"
        caller = threading.Thread(target=lambda: released.append(emb.release_for_exit()))
        caller.start()
        caller.join(5)
        assert not caller.is_alive(), (
            "release_for_exit blocked on the embedder load lock: a stop or "
            "restart issued during an embedding-model load would hang here and "
            "never reach the worker teardown, leaking the worker (REG-650)")
        assert released == [True]
        assert runner.shutdown_calls == [0], "a busy worker must not be waited on"
    finally:
        finished.set()
        t.join(5)


def test_release_for_exit_is_safe_with_no_embedder():
    emb._EMBEDDER = None
    assert emb.release_for_exit() is False


def test_release_for_exit_is_safe_with_no_runner():
    emb._EMBEDDER = _isolated_embedder_with(None, active=1)
    assert emb.release_for_exit() is False


# --------------------------------------------------------------------------- #
# The exit paths must invoke it (with the real os._exit/os.execv stubbed out).
# --------------------------------------------------------------------------- #

class _RecordingRunner:
    def __init__(self):
        self.shutdown_calls = []

    def shutdown(self, grace=5.0):
        self.shutdown_calls.append(grace)


@pytest.fixture
def pinned(monkeypatch):
    """A pinned embedder (mid-embed) whose worker teardown is observable."""
    runner = _RecordingRunner()
    monkeypatch.setattr(emb, "_EMBEDDER", _isolated_embedder_with(runner, active=1))
    monkeypatch.setattr(hs, "_engine", None)
    reset_calls = []
    monkeypatch.setattr(emb, "reset_embedder", lambda: reset_calls.append(1))
    return runner, reset_calls


def test_stop_during_an_embedder_load_does_not_hang(monkeypatch):
    """The exit path must never block on the embedder's load lock.

    get_embedder() holds _LOCK for a whole embedding-model load, and
    active_requests() takes _LOCK too, so a guard built on it would block a stop
    issued mid-load and never reach the worker teardown. _EMBEDDER is still None
    mid-load, so the idle branch's reset_embedder() - which also takes _LOCK -
    is the one taken.
    """
    monkeypatch.setattr(hs, "_engine", None)
    monkeypatch.setattr(os, "_exit", lambda code: (_ for _ in ()).throw(SystemExit(code)))
    holding = threading.Event()
    finished = threading.Event()

    def _hold_lock_like_a_load():
        with emb._LOCK:
            holding.set()
            finished.wait(10)

    t = threading.Thread(target=_hold_lock_like_a_load, daemon=True)
    t.start()
    done = []

    def _stop():
        try:
            hs._do_shutdown()
        except SystemExit:
            done.append("exited")

    try:
        assert holding.wait(5), "helper never took _LOCK"
        s = threading.Thread(target=_stop, daemon=True)
        s.start()
        s.join(8)
        assert not s.is_alive(), (
            "_do_shutdown blocked while an embedding-model load held the "
            "embedder lock: the stop hangs and never reaches the worker "
            "teardown (REG-650)")
        assert done == ["exited"]
    finally:
        finished.set()
        t.join(5)


def test_do_shutdown_reaps_the_pinned_embedder_worker(pinned, monkeypatch):
    runner, reset_calls = pinned
    monkeypatch.setattr(os, "_exit", lambda code: (_ for _ in ()).throw(SystemExit(code)))

    with pytest.raises(SystemExit):
        hs._do_shutdown()

    assert reset_calls == [], "a pinned embedder must not be reset_embedder()'d"
    assert runner.shutdown_calls, (
        "stop left the embedder worker running: os._exit bypasses atexit, so the "
        "daemon child is NOT reclaimed and keeps its model resident in VRAM")
    assert runner.shutdown_calls == [0], (
        "the reap must not wait on the in-flight embed (grace=0), or the stop "
        f"stalls for the grace period: {runner.shutdown_calls}")


def test_do_restart_reaps_the_pinned_embedder_worker(pinned, monkeypatch):
    runner, reset_calls = pinned
    monkeypatch.setattr(os, "execv", lambda exe, argv: (_ for _ in ()).throw(SystemExit(0)))

    with pytest.raises(SystemExit):
        hs._do_restart()

    assert reset_calls == [], "a pinned embedder must not be reset_embedder()'d"
    assert runner.shutdown_calls == [0], (
        "restart left the old embedder worker running: os.execv replaces this "
        "process image but not the worker child, and bypasses atexit - the "
        "restarted server then spawns a SECOND worker beside the orphan")


def test_idle_embedder_is_closed_politely(monkeypatch):
    """An IDLE worker still gets the polite shutdown command (so the child frees
    the model cleanly), without taking the load lock: the exiting process has no
    use for reset_embedder()'s singleton clearing."""
    runner = _RecordingRunner()
    monkeypatch.setattr(emb, "_EMBEDDER", _isolated_embedder_with(runner, active=0))
    monkeypatch.setattr(hs, "_engine", None)
    monkeypatch.setattr(os, "_exit", lambda code: (_ for _ in ()).throw(SystemExit(code)))

    with pytest.raises(SystemExit):
        hs._do_shutdown()

    assert runner.shutdown_calls == [5.0], (
        "an idle worker should be asked to close cleanly, not hard-killed: "
        f"{runner.shutdown_calls}")
