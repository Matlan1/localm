# SPDX-License-Identifier: AGPL-3.0-or-later
"""Restart/stop must not leak the embedder worker (REG-650).

_do_shutdown and _do_restart skip releasing the embedder while a request is
mid-embed(), on the premise that "the worker is a daemon child, so it is still
reclaimed when this process exits". That premise is FALSE on exactly the two
paths those functions take:

  - _do_shutdown ends at os._exit(0), and _do_restart at os.execv(). Both
    bypass atexit - and multiprocessing's daemon-child reclamation IS an atexit
    hook (multiprocessing.util._exit_function). os.execv additionally keeps the
    same PID and replaces only THIS process's image; it never touches the
    separate `localm-embedder-worker` child.

Verified live (2026-07-15) with a real daemon child blocked on a queue, exactly
like the real worker: reclaimed on a normal interpreter exit, but SURVIVES both
os._exit and os.execv. So a restart mid-index left the worker orphaned with its
model resident in VRAM, and the restarted server spawned a second one beside it;
a "stopped" server left a GPU-resident zombie, defeating that path's stated goal
of freeing ALL resident VRAM.

The existing pin tests (test_embedder_unload_honors_pin.py) only assert that
reset_embedder was NOT called - they bless the skip and never check that the
worker is actually reclaimed, and they monkeypatch os._exit/os.execv to raise
before any real teardown runs, so no test ever spawned a real child across a
real exit.
"""

from __future__ import annotations

import os

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
# The real property: a REAL worker process is actually dead afterwards.
# --------------------------------------------------------------------------- #

def test_reap_worker_for_exit_kills_a_real_busy_worker_process():
    """The reaper must terminate the real child. Uses a REAL spawned worker
    (as the crash-containment tests do) and asserts on real process liveness,
    not on a call being recorded."""
    runner = EmbedderRunner()
    runner._spawn()
    try:
        assert runner.is_alive(), "worker did not start"
        proc = runner._proc
        # Pinned: a request is mid-embed(), which is exactly when the old code
        # skipped the release entirely and leaked this process.
        emb._EMBEDDER = _isolated_embedder_with(runner, active=1)

        assert emb.reap_worker_for_exit() is True
        proc.join(timeout=10)
        assert not proc.is_alive(), (
            "the embedder worker survived the reap and would outlive an "
            "os._exit/os.execv as a VRAM-resident zombie")
    finally:
        runner.shutdown(grace=0)


def test_reap_worker_for_exit_is_safe_with_no_embedder():
    emb._EMBEDDER = None
    assert emb.reap_worker_for_exit() is False


def test_reap_worker_for_exit_is_safe_with_no_runner():
    emb._EMBEDDER = _isolated_embedder_with(None, active=1)
    assert emb.reap_worker_for_exit() is False


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
    monkeypatch.setattr(emb, "active_requests", lambda: 1)
    monkeypatch.setattr(hs, "_engine", None)
    reset_calls = []
    monkeypatch.setattr(emb, "reset_embedder", lambda: reset_calls.append(1))
    return runner, reset_calls


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


def test_idle_embedder_still_uses_the_graceful_reset(monkeypatch):
    """The unpinned path is unchanged: a clean reset_embedder(), which closes the
    model in the child rather than hard-killing it."""
    runner = _RecordingRunner()
    monkeypatch.setattr(emb, "_EMBEDDER", _isolated_embedder_with(runner, active=0))
    monkeypatch.setattr(emb, "active_requests", lambda: 0)
    monkeypatch.setattr(hs, "_engine", None)
    reset_calls = []
    monkeypatch.setattr(emb, "reset_embedder", lambda: reset_calls.append(1))
    monkeypatch.setattr(os, "_exit", lambda code: (_ for _ in ()).throw(SystemExit(code)))

    with pytest.raises(SystemExit):
        hs._do_shutdown()

    assert reset_calls == [1]
    assert runner.shutdown_calls == [], "no hard reap needed when nothing is in flight"
