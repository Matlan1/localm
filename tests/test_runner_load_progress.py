# SPDX-License-Identifier: AGPL-3.0-or-later
"""ModelRunner load wait: a NON-TERMINAL ``progress`` envelope.

``spawn_and_load``'s parent-side poll loop must deliver a progress envelope and
keep waiting, so the terminal envelope behind it is still what decides the load.

These drive the REAL parent-side poll loop over REAL queues, substituting only
the child process's liveness check.

A ``progress`` envelope keeps ``resp_q.get()`` returning, so a deadline check
that only ran on ``queue.Empty`` would never be reached again once a child
started emitting them, and a wedged native call could hang a load forever. The
load-deadline test is written to FAIL, never HANG, if that regresses.
"""

from __future__ import annotations

import multiprocessing as mp
import threading

import pytest

from localm.inference.backends.llamacpp._runner import ModelRunner


class _AliveProc:
    """Stands in for the worker process's liveness check only; the queues and
    the parent-side load loop under test are real."""

    def __init__(self):
        self.terminated = False

    def is_alive(self):
        return not self.terminated

    def terminate(self):
        self.terminated = True

    def join(self, timeout=None):
        return None

    exitcode = 0


def _make_runner():
    """A runner whose queues are real but whose _spawn is inert; spawn_and_load
    calls _spawn() first thing."""
    ctx = mp.get_context("spawn")
    r = ModelRunner()
    r._req_q, r._resp_q, r._ctrl_q = ctx.Queue(), ctx.Queue(), ctx.Queue()
    r._proc = _AliveProc()
    r._spawn = lambda: None
    return r


def test_progress_envelope_does_not_end_the_load():
    """A progress envelope is delivered and the wait CONTINUES, so the terminal
    envelope behind it is still the one that decides the load."""
    r = _make_runner()
    seen = []
    r._resp_q.put(("progress", {"fraction": 0.25}))
    r._resp_q.put(("progress", {"fraction": 0.75}))
    r._resp_q.put(("ok", {"n_layers": 32}))

    meta = r.spawn_and_load({}, timeout=10, on_progress=seen.append)

    assert meta == {"n_layers": 32}
    assert seen == [{"fraction": 0.25}, {"fraction": 0.75}], seen


def test_progress_payload_is_passed_through_uninterpreted():
    """The runner guarantees delivery, not meaning: the producer owns the
    payload's shape, and anything picklable must survive unchanged."""
    r = _make_runner()
    seen = []
    for payload in (0.5, "loading tensors", None, {"a": [1, 2]}):
        r._resp_q.put(("progress", payload))
    r._resp_q.put(("ok", {}))

    r.spawn_and_load({}, timeout=10, on_progress=seen.append)

    assert seen == [0.5, "loading tensors", None, {"a": [1, 2]}], seen


def test_progress_without_a_sink_is_simply_dropped():
    """No caller passes on_progress today, so the default path must swallow a
    progress envelope and still load - not raise, and not hang."""
    r = _make_runner()
    r._resp_q.put(("progress", 0.5))
    r._resp_q.put(("ok", {"n_layers": 1}))

    assert r.spawn_and_load({}, timeout=10) == {"n_layers": 1}


def test_a_raising_progress_sink_never_fails_the_load():
    """A reporting callback is not the work: the load succeeded, so it is
    reported as succeeded."""
    r = _make_runner()

    def _boom(_payload):
        raise ValueError("sink is broken")

    r._resp_q.put(("progress", 0.5))
    r._resp_q.put(("ok", {"n_layers": 7}))

    assert r.spawn_and_load({}, timeout=10, on_progress=_boom) == {"n_layers": 7}


def test_an_unknown_envelope_kind_is_still_a_loud_error():
    """Adding one non-terminal kind must not soften the strict check: an
    unrecognised kind is still a loud error."""
    r = _make_runner()
    r._resp_q.put(("surprise", 1))

    with pytest.raises(RuntimeError, match="Unexpected response"):
        r.spawn_and_load({}, timeout=10)


def test_a_non_tuple_envelope_is_not_re_read_as_progress():
    """A non-tuple cannot be indexed for a kind, so it goes down the TERMINAL
    path and fails loudly instead of being mistaken for progress.

    The bare string is "progress": the substring is present, so anything matching
    loosely rather than on the tuple's own shape would let it through."""
    r = _make_runner()
    r._resp_q.put("progress")

    with pytest.raises(RuntimeError, match="Unexpected response"):
        r.spawn_and_load({}, timeout=10)


def test_progress_without_a_payload_is_delivered_as_none():
    """A payload-less ``("progress",)`` is a well-formed KIND with a missing
    value, not a malformed envelope: it is delivered as None and does not fail
    the load. The deadline still bounds the wait."""
    r = _make_runner()
    seen = []
    r._resp_q.put(("progress",))
    r._resp_q.put(("ok", {"n_layers": 3}))

    assert r.spawn_and_load({}, timeout=10, on_progress=seen.append) == {
        "n_layers": 3}
    assert seen == [None], seen


def test_progress_does_not_extend_the_load_deadline():
    """A child emitting progress in a tight loop must NOT keep a hung load alive:
    the deadline bounds the WHOLE load, and it is the only guard between a wedged
    native call and a server that never answers.

    Asserted via a watchdog, so a deadline checked only on queue.Empty - which a
    steady progress stream starves - FAILS this test instead of hanging the
    suite."""
    r = _make_runner()
    stop = threading.Event()

    def _flood():
        while not stop.is_set():
            try:
                r._resp_q.put(("progress", 0.1))
            except Exception:       # queue closed by shutdown() on timeout
                return
            stop.wait(0.01)

    flood = threading.Thread(target=_flood, daemon=True)
    flood.start()

    box: dict = {}

    def _drive():
        try:
            box["meta"] = r.spawn_and_load({}, timeout=0.4)
        except BaseException as e:      # noqa: BLE001 - re-asserted below
            box["exc"] = e

    driver = threading.Thread(target=_drive, daemon=True)
    driver.start()
    driver.join(timeout=30)
    finished = not driver.is_alive()
    stop.set()
    flood.join(timeout=5)

    assert finished, (
        "spawn_and_load never returned while progress kept arriving: the load "
        "deadline is being starved, so a hung load can now run forever")
    exc = box.get("exc")
    assert isinstance(exc, RuntimeError), f"expected a timeout, got {box!r}"
    assert "timed out" in str(exc), exc
