# SPDX-License-Identifier: AGPL-3.0-or-later
"""ModelRunner stream timeouts: slow prefill is not a stall, and a killed worker does not leave the backend wedged (REG-606)."""

from __future__ import annotations

import multiprocessing as mp
import queue as _queue
import threading

import pytest

from localm.inference.backends.llamacpp import _runner as runner_mod
from localm.inference.backends.llamacpp._runner import ModelRunner


class _AliveProc:
    """Stands in for the worker process's liveness check only; the queues and the parent-side poll loop under test are real."""

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
    ctx = mp.get_context("spawn")
    r = ModelRunner()
    r._req_q, r._resp_q, r._ctrl_q = ctx.Queue(), ctx.Queue(), ctx.Queue()
    r._proc = _AliveProc()
    return r


def _fake_child(r, stop, *, first_delay, tokens):
    """Mimics the worker's observable protocol: waits *first_delay* (the prompt prefill) before the first chunk, then streams tokens promptly."""
    while not stop.is_set():
        try:
            cmd = r._req_q.get(timeout=0.05)
        except (_queue.Empty, OSError, ValueError):
            continue
        if cmd[0] != "chat_stream":
            continue
        if stop.wait(first_delay):
            return
        for t in tokens:
            r._resp_q.put(("chunk", t))
        r._resp_q.put(("done", {"finish_reason": "stop"}))


class TestFirstTokenPrefill:
    def test_slow_prefill_is_not_reported_as_a_stall(self, monkeypatch):
        """A prefill slower than the PER-TOKEN ceiling must still produce tokens: it is the first token's own latency, not a stalled process."""
        # Per-token ceiling far below the prefill; first-chunk budget above it.
        monkeypatch.setattr(runner_mod, "_STREAM_CHUNK_TIMEOUT", 0.5)
        r = _make_runner()
        stop = threading.Event()
        child = threading.Thread(
            target=_fake_child, args=(r, stop),
            kwargs=dict(first_delay=1.5, tokens=["Hel", "lo"]), daemon=True)
        child.start()
        try:
            out = list(r.chat_stream(first_chunk_timeout=10.0, messages=[]))
            assert out == ["Hel", "lo"], out
            assert not r._proc.terminated, (
                "a legitimately slow prefill killed the worker as a false stall")
        finally:
            stop.set(); child.join(2)

    def test_a_genuinely_wedged_prefill_still_raises(self, monkeypatch):
        """The negative case: raising the first-token budget must not remove the bound."""
        monkeypatch.setattr(runner_mod, "_STREAM_CHUNK_TIMEOUT", 0.5)
        r = _make_runner()
        # No child at all: nothing will ever answer.
        with pytest.raises(RuntimeError, match="stalled"):
            list(r.chat_stream(first_chunk_timeout=1.0, messages=[]))
        assert r._proc is None, "a wedged worker must still be killed"

    def test_per_token_ceiling_still_applies_after_the_first_token(self, monkeypatch):
        """The generous budget is for PREFILL only: once tokens flow, a stalled child must still be caught by the per-token ceiling."""
        monkeypatch.setattr(runner_mod, "_STREAM_CHUNK_TIMEOUT", 0.5)
        r = _make_runner()
        stop = threading.Event()

        def _child():
            r._req_q.get(timeout=5)
            r._resp_q.put(("chunk", "first"))
            stop.wait(30)          # then wedge, mid-stream

        child = threading.Thread(target=_child, daemon=True)
        child.start()
        try:
            gen = r.chat_stream(first_chunk_timeout=10.0, messages=[])
            assert next(gen) == "first"
            with pytest.raises(RuntimeError, match="stalled"):
                next(gen)
        finally:
            stop.set(); child.join(2)

    def test_first_chunk_timeout_defaults_to_the_generous_module_default(self, monkeypatch):
        """Callers that pass nothing must still get the prefill budget, not the per-token ceiling."""
        monkeypatch.setattr(runner_mod, "_STREAM_CHUNK_TIMEOUT", 0.5)
        r = _make_runner()
        stop = threading.Event()
        child = threading.Thread(
            target=_fake_child, args=(r, stop),
            kwargs=dict(first_delay=1.5, tokens=["ok"]), daemon=True)
        child.start()
        try:
            assert list(r.chat_stream(messages=[])) == ["ok"]
        finally:
            stop.set(); child.join(2)


class TestBackendRecoversAfterDrainTimeoutKill:
    def test_next_request_after_a_drain_timeout_kill_reloads_instead_of_crashing(
            self, monkeypatch):
        """A mid-stream cancel whose drain times out kills the worker and nulls the runner's queues."""
        from localm.inference.backends.gguf import GgufBackend

        monkeypatch.setattr(runner_mod, "_CANCEL_DRAIN_TIMEOUT", 0.3)
        b = GgufBackend.__new__(GgufBackend)
        b._loaded = True
        b._llm = None
        b._grammar_unsupported = False
        b.last_finish_reason = "stop"
        r = _make_runner()
        b._runner = r

        stop = threading.Event()

        def _child():
            r._req_q.get(timeout=5)
            r._resp_q.put(("chunk", "hi"))
            stop.wait(30)      # wedged inside a native call: never confirms cancel

        child = threading.Thread(target=_child, daemon=True)
        child.start()
        try:
            gen = b.chat_stream(messages=[])
            assert next(gen) == "hi"
            gen.close()        # the mid-stream cancel; its drain will time out
        finally:
            stop.set(); child.join(2)

        assert r._req_q is None, "precondition: the drain timeout killed the worker"
        assert not b.loaded, (
            "the backend still reports loaded after its worker was killed, so "
            "Engine.chat_stream skips the auto-reload and the next request "
            "dereferences a None queue (AttributeError) - permanently")
