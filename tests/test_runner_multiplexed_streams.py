# SPDX-License-Identifier: AGPL-3.0-or-later
"""The GGUF runner's multiplexed protocol, used when a model loads with more
than one parallel slot: several streams share one worker, each envelope tagged
with its stream id, cancelled one at a time, failing together when the worker
dies.

Parent side: a real ModelRunner against a scripted child on in-process queues.
Child side: the real ``_runner_main`` dispatch loop with a stand-in worker.
"""
from __future__ import annotations

import queue
import threading
import time

import pytest

from localm.inference.backends.base import stream_stop_check
from localm.inference.backends.llamacpp import _runner as runner_mod
from localm.inference.backends.llamacpp._runner import ModelRunner


class _Proc:
    def __init__(self):
        self.alive = True
        self.exitcode = None

    def is_alive(self):
        return self.alive

    def terminate(self):
        self.alive = False

    def join(self, timeout=None):
        return None


class _Child:
    """Scripted worker: records each multiplexed stream request by sid and
    answers simple requests with *answer*."""

    def __init__(self, runner, answer=7):
        self.r = runner
        self.answer = answer
        self.streams = queue.Queue()
        self.simple = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while not self._stop.is_set():
            try:
                cmd = self.r._req_q.get(timeout=0.05)
            except queue.Empty:
                continue
            if cmd is None or cmd[0] == "shutdown":
                return
            if cmd[0] == "chat_stream_mux":
                self.streams.put(cmd[1]["sid"])
            else:
                self.simple.append(cmd)
                self.r._resp_q.put(("ok", self.answer))

    def next_sid(self, timeout=5):
        return self.streams.get(timeout=timeout)

    def send(self, sid, envelope):
        self.r._resp_q.put(("stream", sid, envelope))

    def stop(self):
        self._stop.set()


def _mux_runner():
    r = ModelRunner()
    r._req_q, r._resp_q, r._ctrl_q = queue.Queue(), queue.Queue(), queue.Queue()
    r._proc = _Proc()
    r._start_mux()
    return r


def _consume(r, out, key, **kw):
    try:
        out[key] = {"tokens": list(r.chat_stream(messages=[], **kw)),
                    "done": r.last_done}
    except Exception as exc:
        out[key] = {"error": exc}


def _start(target):
    t = threading.Thread(target=target, daemon=True)
    t.start()
    return t


def test_two_streams_each_receive_only_their_own_envelopes():
    r = _mux_runner()
    child = _Child(r)
    out = {}
    a = _start(lambda: _consume(r, out, "a"))
    sid_a = child.next_sid()
    b = _start(lambda: _consume(r, out, "b"))
    sid_b = child.next_sid()
    for i in range(3):
        child.send(sid_b, ("chunk", f"b{i}"))
        child.send(sid_a, ("chunk", f"a{i}"))
    child.send(sid_a, ("done", {"finish_reason": "length"}))
    child.send(sid_b, ("done", {"finish_reason": "stop"}))
    a.join(5)
    b.join(5)
    assert out["a"] == {"tokens": ["a0", "a1", "a2"], "done": {"finish_reason": "length"}}
    assert out["b"] == {"tokens": ["b0", "b1", "b2"], "done": {"finish_reason": "stop"}}
    child.stop()
    r.shutdown(grace=0)


def test_closing_one_stream_cancels_only_that_stream():
    r = _mux_runner()
    child = _Child(r)
    gen_a = r.chat_stream(messages=[])
    out = {}
    first = {}

    def drive_a():
        first["tok"] = next(gen_a)

    t = _start(drive_a)
    sid_a = child.next_sid()
    b = _start(lambda: _consume(r, out, "b"))
    sid_b = child.next_sid()
    child.send(sid_a, ("chunk", "a0"))
    t.join(5)
    closer = _start(gen_a.close)
    assert r._ctrl_q.get(timeout=5) == ("cancel_stream", sid_a)
    child.send(sid_a, ("done", {"finish_reason": "stop"}))
    closer.join(5)
    assert not closer.is_alive()
    child.send(sid_b, ("chunk", "b0"))
    child.send(sid_b, ("done", {"finish_reason": "length"}))
    b.join(5)
    assert first["tok"] == "a0"
    assert out["b"]["tokens"] == ["b0"]
    assert r._ctrl_q.empty()
    child.stop()
    r.shutdown(grace=0)


def test_a_token_count_during_a_stream_gets_the_real_answer():
    r = _mux_runner()
    child = _Child(r, answer=42)
    out = {}
    s = _start(lambda: _consume(r, out, "s"))
    sid = child.next_sid()
    assert r.count_tokens("hello") == 42
    child.send(sid, ("done", {"finish_reason": "stop"}))
    s.join(5)
    assert out["s"]["tokens"] == []
    child.stop()
    r.shutdown(grace=0)


def test_a_dead_worker_fails_every_stream_with_the_same_report(monkeypatch):
    r = _mux_runner()
    child = _Child(r)
    traces = iter(["Fatal Python error: Segmentation fault\nmore", ""])
    monkeypatch.setattr(r, "_native_crash_trace", lambda: next(traces, ""))
    out = {}
    a = _start(lambda: _consume(r, out, "a"))
    child.next_sid()
    b = _start(lambda: _consume(r, out, "b"))
    child.next_sid()
    r._proc.exitcode = -11
    r._proc.alive = False
    a.join(5)
    b.join(5)
    for key in ("a", "b"):
        err = out[key]["error"]
        assert isinstance(err, RuntimeError)
        assert "Segmentation fault" in str(err), str(err)
    child.stop()


def test_a_repeated_status_reaches_on_status_once():
    r = _mux_runner()
    child = _Child(r)
    seen = []
    out = {}
    s = _start(lambda: _consume(r, out, "s", on_status=seen.append))
    sid = child.next_sid()
    for _ in range(3):
        child.send(sid, ("status", "Waiting for another request to finish..."))
    child.send(sid, ("status", "Generating response..."))
    child.send(sid, ("done", {"finish_reason": "stop"}))
    s.join(5)
    assert seen == ["Waiting for another request to finish...", "Generating response..."]
    child.stop()
    r.shutdown(grace=0)


def test_a_published_stop_cancels_a_stream_that_has_not_started():
    r = _mux_runner()
    child = _Child(r)
    stop = threading.Event()
    out = {}

    def consume():
        with stream_stop_check(stop.is_set):
            _consume(r, out, "s")

    s = _start(consume)
    sid = child.next_sid()
    stop.set()
    assert r._ctrl_q.get(timeout=5) == ("cancel_stream", sid)
    child.send(sid, ("done", {"finish_reason": "stop"}))
    s.join(5)
    assert out["s"]["tokens"] == []
    child.stop()
    r.shutdown(grace=0)


def test_an_envelope_for_an_ended_stream_is_dropped():
    r = _mux_runner()
    child = _Child(r)
    out = {}
    s = _start(lambda: _consume(r, out, "s"))
    sid = child.next_sid()
    child.send(sid, ("done", {"finish_reason": "stop"}))
    s.join(5)
    child.send(sid, ("chunk", "late"))
    s2 = _start(lambda: _consume(r, out, "s2"))
    sid2 = child.next_sid()
    child.send(sid2, ("chunk", "x"))
    child.send(sid2, ("done", {"finish_reason": "stop"}))
    s2.join(5)
    assert out["s2"]["tokens"] == ["x"]
    child.stop()
    r.shutdown(grace=0)


@pytest.mark.parametrize("slots, mux", [(1, False), (2, True), (None, False)])
def test_a_load_switches_to_the_multiplexed_protocol_only_with_slots(monkeypatch, slots, mux):
    r = ModelRunner()

    def fake_spawn():
        r._mux = False
        r._req_q, r._resp_q, r._ctrl_q = queue.Queue(), queue.Queue(), queue.Queue()
        r._proc = _Proc()
        meta = {} if slots is None else {"parallel_slots": slots}
        r._resp_q.put(("ok", meta))

    monkeypatch.setattr(r, "_spawn", fake_spawn)
    r.spawn_and_load({})
    assert r.multiplexed is mux
    r.shutdown(grace=0)


# ---------------------------------------------------------------- child side


class _Worker:
    """Stand-in GgufWorker: each stream waits (up to 5 s) until *together*
    streams are running, records how many ran at once, then yields its
    tokens."""

    last_finish_reason = "stop"
    grammar_unsupported_this_call = False
    chatml_fallback_reason = None
    mtp_status = None
    mtp_active_this_call = False
    mtp_call_status = ""
    mtp_drafted = 0
    mtp_accepted = 0
    mtp_steps = 0
    mtp_paused_steps = 0
    mtp_skipped = ""
    spec_report = None
    together = 2

    def __init__(self, cancel_event=None, **payload):
        self.stream_cancel = None
        self._lock = threading.Lock()
        self.running = 0
        self.max_running = 0

    def load(self):
        return {"parallel_slots": 2}

    def _join(self):
        with self._lock:
            self.running += 1
            self.max_running = max(self.max_running, self.running)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            with self._lock:
                if self.max_running >= self.together:
                    return
            time.sleep(0.01)

    def chat_stream(self, on_status=None, **payload):
        self._join()
        for tok in payload["tokens"]:
            yield tok
            time.sleep(0.01)

    def close(self):
        pass


def _child_envelopes(resp_q, n_done, timeout=10):
    got = []
    deadline = time.monotonic() + timeout
    done = 0
    while done < n_done and time.monotonic() < deadline:
        try:
            item = resp_q.get(timeout=0.1)
        except queue.Empty:
            continue
        got.append(item)
        if item[0] == "stream" and item[2][0] == "done":
            done += 1
    return got


def test_the_child_runs_multiplexed_streams_at_the_same_time(monkeypatch):
    made = []

    class _Seen(_Worker):
        def __init__(self, **kw):
            super().__init__(**kw)
            made.append(self)

    monkeypatch.setattr("localm.inference.backends.llamacpp._worker.GgufWorker", _Seen)
    req_q, resp_q, ctrl_q = queue.Queue(), queue.Queue(), queue.Queue()
    req_q.put(("load", {}))
    req_q.put(("chat_stream_mux", {"sid": 1, "kwargs": {"tokens": ["a", "b", "c"]}}))
    req_q.put(("chat_stream_mux", {"sid": 2, "kwargs": {"tokens": ["x", "y"]}}))
    req_q.put(None)
    runner_mod._runner_main(req_q, resp_q, ctrl_q)
    assert resp_q.get(timeout=5) == ("ok", {"parallel_slots": 2})
    got = _child_envelopes(resp_q, 2)
    ctrl_q.put(None)
    by_sid = {1: [], 2: []}
    for kind, sid, env in got:
        assert kind == "stream"
        by_sid[sid].append(env)
    assert [e[1] for e in by_sid[1] if e[0] == "chunk"] == ["a", "b", "c"]
    assert [e[1] for e in by_sid[2] if e[0] == "chunk"] == ["x", "y"]
    assert by_sid[1][-1][0] == "done" and by_sid[2][-1][0] == "done"
    assert made[0].max_running == 2


def test_the_child_cancels_only_the_named_stream(monkeypatch):
    class _Slow(_Worker):
        def chat_stream(self, on_status=None, **payload):
            self._join()
            for tok in payload["tokens"]:
                yield tok
                time.sleep(0.05)

    monkeypatch.setattr("localm.inference.backends.llamacpp._worker.GgufWorker", _Slow)
    req_q, resp_q, ctrl_q = queue.Queue(), queue.Queue(), queue.Queue()
    req_q.put(("load", {}))
    req_q.put(("chat_stream_mux", {"sid": 1, "kwargs": {"tokens": list("abcdefghij")}}))
    req_q.put(("chat_stream_mux", {"sid": 2, "kwargs": {"tokens": list("vwxyz")}}))
    req_q.put(None)
    runner_mod._runner_main(req_q, resp_q, ctrl_q)
    resp_q.get(timeout=5)
    ctrl_q.put(("cancel_stream", 1))
    got = _child_envelopes(resp_q, 2)
    ctrl_q.put(None)
    one = [env[1] for kind, sid, env in got if sid == 1 and env[0] == "chunk"]
    two = [env[1] for kind, sid, env in got if sid == 2 and env[0] == "chunk"]
    assert two == list("vwxyz")
    assert len(one) < 10


def test_an_unexpected_fault_in_a_stream_thread_exits_the_worker(monkeypatch):
    class _Exit(Exception):
        pass

    codes = []

    def fake_exit(code):
        codes.append(code)
        raise _Exit

    monkeypatch.setattr(runner_mod.os, "_exit", fake_exit)

    class _Broken(_Worker):
        def chat_stream(self, on_status=None, **payload):
            raise ValueError("boom")
            yield

    cancels = {7: threading.Event()}
    with pytest.raises(_Exit):
        runner_mod._serve_mux_stream(_Broken(), 7, {}, queue.Queue(),
                                     cancels[7], cancels)
    assert codes == [1]
    assert 7 not in cancels


def test_a_typed_refusal_in_a_stream_thread_is_an_error_envelope(monkeypatch):
    from localm.inference.backends.base import InvalidGrammarError

    class _Refusing(_Worker):
        def chat_stream(self, on_status=None, **payload):
            raise InvalidGrammarError("bad grammar")
            yield

    resp_q = queue.Queue()
    runner_mod._serve_mux_stream(_Refusing(), 3, {}, resp_q, threading.Event(), {})
    assert resp_q.get_nowait() == ("stream", 3, ("error", "bad grammar", "InvalidGrammarError"))
