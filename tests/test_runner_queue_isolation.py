# SPDX-License-Identifier: AGPL-3.0-or-later
"""Parent-side response-queue isolation for llamacpp/_runner.py's ModelRunner.

One ``GgufBackend`` (a module-level singleton per model in http_server.py) holds
ONE ``ModelRunner`` with ONE shared ``self._resp_q``, and two PARENT threads can
reach it: a live ``chat_stream`` drive on the stream's producer thread, and a
``count_tokens`` / ``count_messages_tokens`` RPC on an executor thread (token
counting runs OUTSIDE the per-model generation semaphore - the prompt count fires
before ``async with sem`` and the completion count after it releases). Without
arbitration a count RPC in flight during another request's stream can
``resp_q.get()`` an envelope meant for the stream: a stolen ``("chunk",..)``
drops a token, a stolen ``("done",..)`` spins the stream to its 120s ceiling, and
a delayed reply trips the simple command's 30s timeout, which kills the worker
mid-generation and misreports it as a native fault.

Parent-side queue use is serialised with a per-ModelRunner lock, held across a
whole stream drive and per simple command; token counts acquire it NON-blocking
and raise :class:`RunnerBusy` (the caller falls back to its documented chars/4
heuristic) instead of queueing behind a live generation.

These tests exercise the REAL ModelRunner parent-side code (``chat_stream`` /
``count_tokens`` / ``_simple_request`` / ``_q_lock``) over real thread-safe queues
and real threads. Only the leaf worker - the child GgufWorker dispatch loop, which
is strictly serial - is replaced by a protocol-faithful in-process thread, so the
tests need no native runtime or model. The race being guarded is entirely
parent-side (two threads arbitrating one ``resp_q.get()``), which is identical
whether the transport is a ``multiprocessing.Queue`` or the ``queue.Queue`` used
here; both deliver each enqueued item to exactly one getter and raise the same
``queue.Empty`` on the timeout path ModelRunner polls.
"""

import queue
import threading
import time

from localm.inference.backends.llamacpp._runner import ModelRunner, RunnerBusy


class _FakeProc:
    """Stand-in for the worker Process: always reports alive (the fake worker is
    a live thread) so ModelRunner's is_alive() crash checks never short-circuit
    the queue polling under test."""

    exitcode = None

    def is_alive(self) -> bool:
        return True

    def kill(self) -> None:  # pragma: no cover - teardown only
        pass

    def terminate(self) -> None:  # pragma: no cover - teardown only
        pass

    def join(self, timeout=None) -> None:  # pragma: no cover - teardown only
        pass


def _fake_worker(req_q, resp_q, ctrl_q, *, chunks, gate=None, gate_after=1,
                 count_value=7, chunk_delay=0.0):
    """Protocol-faithful, strictly serial stand-in for the child's dispatch loop
    (``_runner_main``): reads req_q one command at a time and answers on resp_q
    with the same tagged envelopes. Strictly serial, exactly like the real worker,
    so the ONLY concurrency the tests exercise is the PARENT-side arbitration of
    resp_q.

    On ``chat_stream`` it emits each of *chunks* as a ``("chunk", tok)`` then a
    ``("done", ...)``. If *gate* is given, it blocks on it after emitting
    ``gate_after`` chunks, so a test can hold the stream mid-flight (the producer
    is then parked in ``resp_q.get()`` holding ``_q_lock``) while it fires a
    concurrent token count. Count/grammar commands answer immediately."""
    while True:
        try:
            cmd = req_q.get(timeout=5)
        except queue.Empty:  # pragma: no cover - safety valve if a test forgets to stop us
            return
        if cmd is None or cmd[0] == "shutdown":
            return
        name = cmd[0]
        if name == "chat_stream":
            for i, tok in enumerate(chunks):
                resp_q.put(("chunk", tok))
                if chunk_delay:
                    time.sleep(chunk_delay)
                if gate is not None and i == gate_after - 1:
                    gate.wait()
            resp_q.put(("done", {"finish_reason": "stop", "grammar_unsupported": False}))
        elif name in ("count_tokens", "count_messages_tokens"):
            resp_q.put(("ok", count_value))
        elif name == "check_grammar":
            resp_q.put(("ok", None))
        else:  # pragma: no cover - unreachable in these tests
            resp_q.put(("error", f"unknown: {name!r}"))


def _make_runner_with_fake_worker(*, chunks, gate=None, gate_after=1,
                                  count_value=7, chunk_delay=0.0):
    """Build a REAL ModelRunner wired to in-process queues and a live worker
    thread - no spawned process, no native model, but the parent-side code under
    test is genuine and unmocked."""
    r = ModelRunner()
    r._req_q = queue.Queue()
    r._resp_q = queue.Queue()
    r._ctrl_q = queue.Queue()
    r._proc = _FakeProc()
    t = threading.Thread(
        target=_fake_worker, args=(r._req_q, r._resp_q, r._ctrl_q),
        kwargs=dict(chunks=chunks, gate=gate, gate_after=gate_after,
                    count_value=count_value, chunk_delay=chunk_delay),
        name="fake-gguf-worker", daemon=True)
    t.start()
    return r, t


def _stop(r, worker_t, gate=None):
    if gate is not None:
        gate.set()  # release a parked worker so it can read the stop sentinel
    try:
        r._req_q.put(None)
    except Exception:
        pass
    worker_t.join(timeout=5)


def _wait_until(pred, timeout=5.0, interval=0.005) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(interval)
    return pred()


def test_token_count_declines_immediately_during_active_stream():
    """While a stream is being driven on the runner, a concurrent token count must
    decline AT ONCE (RunnerBusy -> heuristic) instead of blocking behind it or
    stealing its envelopes - and the stream must arrive intact.

    Without a parent-side lock count_tokens blocks in ``resp_q.get()`` behind the
    parked worker, so the counting thread is still alive after the join
    timeout."""
    gate = threading.Event()
    chunks = ["A", "B", "C", "D"]
    r, wt = _make_runner_with_fake_worker(chunks=chunks, gate=gate, gate_after=1)

    collected: list = []
    stream_exc: list = []

    def run_stream():
        try:
            for tok in r.chat_stream(messages=[{"role": "user", "content": "hi"}]):
                collected.append(tok)
        except BaseException as e:  # noqa: BLE001 - record any failure for the assert
            stream_exc.append(e)

    s = threading.Thread(target=run_stream, name="stream", daemon=True)
    s.start()

    # Stream is now mid-flight: chunk 0 consumed, producer parked in resp_q.get()
    # for chunk 1 while holding _q_lock.
    assert _wait_until(lambda: len(collected) >= 1, timeout=5), \
        "stream never produced its first token"

    c_result: list = []

    def run_count():
        try:
            c_result.append(("value", r.count_tokens("hello world")))
        except RunnerBusy:
            c_result.append(("busy", None))
        except BaseException as e:  # noqa: BLE001
            c_result.append(("exc", e))

    c = threading.Thread(target=run_count, name="count", daemon=True)
    c.start()
    c.join(timeout=3)
    declined_promptly = not c.is_alive()

    # Let the stream finish regardless, so everything unwinds for teardown.
    gate.set()
    s.join(timeout=10)
    c.join(timeout=5)
    _stop(r, wt, gate)

    assert declined_promptly, (
        "count_tokens blocked behind the live stream instead of declining - the "
        "parent-side queue lock is missing or not held across the stream")
    assert c_result == [("busy", None)], f"expected RunnerBusy, got {c_result!r}"
    assert not stream_exc, f"stream raised under a concurrent count: {stream_exc!r}"
    assert collected == chunks, f"stream tokens corrupted: {collected!r}"


def test_multiple_concurrent_counts_decline_and_stream_arrives_intact():
    """Several token counts of BOTH kinds (count_tokens + count_messages_tokens)
    fired from separate threads while a longer stream is provably parked
    mid-flight must ALL decline immediately (RunnerBusy), and the whole stream
    must then arrive in order. Deterministic (gate-driven, no timing luck): the
    producer holds ``_q_lock`` across the parked stream, so every non-blocking
    count acquire fails at once.

    Pre-fix (no lock) the counts would instead queue their RPCs behind the parked
    worker and block in ``resp_q.get()``, so the counting threads would still be
    alive after the join - the RED this asserts against."""
    gate = threading.Event()
    chunks = [f"t{i}" for i in range(10)]
    r, wt = _make_runner_with_fake_worker(chunks=chunks, gate=gate, gate_after=5)

    collected: list = []
    stream_exc: list = []

    def run_stream():
        try:
            for tok in r.chat_stream(messages=[{"role": "user", "content": "go"}]):
                collected.append(tok)
        except BaseException as e:  # noqa: BLE001
            stream_exc.append(e)

    s = threading.Thread(target=run_stream, name="stream", daemon=True)
    s.start()
    assert _wait_until(lambda: len(collected) >= 5, timeout=5), \
        "stream never reached its mid-point"

    results: list = []
    results_lock = threading.Lock()

    def do_count(which: int):
        try:
            if which % 2 == 0:
                v = r.count_tokens("x" * (which + 1))
            else:
                v = r.count_messages_tokens([{"role": "user", "content": "x"}])
            with results_lock:
                results.append(("value", v))
        except RunnerBusy:
            with results_lock:
                results.append(("busy", None))
        except BaseException as e:  # noqa: BLE001
            with results_lock:
                results.append(("exc", e))

    counters = [threading.Thread(target=do_count, args=(i,), name=f"count-{i}",
                                 daemon=True) for i in range(6)]
    for t in counters:
        t.start()
    for t in counters:
        t.join(timeout=3)
    all_declined_promptly = all(not t.is_alive() for t in counters)

    gate.set()
    s.join(timeout=10)
    for t in counters:
        t.join(timeout=5)
    _stop(r, wt, gate)

    assert all_declined_promptly, (
        "a token count blocked behind the live stream instead of declining")
    assert results and all(x[0] == "busy" for x in results), \
        f"expected every concurrent count to decline (RunnerBusy), got {results!r}"
    assert not stream_exc, f"stream raised under concurrent counts: {stream_exc!r}"
    assert collected == chunks, f"stream tokens corrupted: {collected!r}"


def test_check_grammar_declines_during_active_stream():
    """validate_grammar -> ModelRunner.check_grammar is invoked SYNCHRONOUSLY on
    the server's async event loop (routes/chat.py, before the per-model
    semaphore), so it must NOT block behind a live stream - that would freeze the
    whole event loop for the full duration of a concurrent same-model generation.
    While a stream is parked mid-flight holding _q_lock, check_grammar must
    decline promptly (RunnerBusy), leaving the caller to defer validation to
    generation time (where a malformed grammar still raises the same clean
    InvalidGrammarError).

    With a blocking acquire the check_grammar thread is still alive after the join
    timeout."""
    gate = threading.Event()
    chunks = ["A", "B", "C"]
    r, wt = _make_runner_with_fake_worker(chunks=chunks, gate=gate, gate_after=1)

    collected: list = []
    stream_exc: list = []

    def run_stream():
        try:
            for tok in r.chat_stream(messages=[{"role": "user", "content": "hi"}]):
                collected.append(tok)
        except BaseException as e:  # noqa: BLE001
            stream_exc.append(e)

    s = threading.Thread(target=run_stream, name="stream", daemon=True)
    s.start()
    assert _wait_until(lambda: len(collected) >= 1, timeout=5), "stream never started"

    result: list = []

    def run_check():
        try:
            r.check_grammar('root ::= "a"')
            result.append(("ok", None))
        except RunnerBusy:
            result.append(("busy", None))
        except BaseException as e:  # noqa: BLE001
            result.append(("exc", e))

    g = threading.Thread(target=run_check, name="grammar", daemon=True)
    g.start()
    g.join(timeout=3)
    declined_promptly = not g.is_alive()

    gate.set()
    s.join(timeout=10)
    g.join(timeout=5)
    _stop(r, wt, gate)

    assert declined_promptly, (
        "check_grammar blocked behind the live stream instead of declining - "
        "validate_grammar on the event loop would freeze the whole server")
    assert result == [("busy", None)], f"expected RunnerBusy, got {result!r}"
    assert not stream_exc, f"stream raised: {stream_exc!r}"
    assert collected == chunks, f"stream tokens corrupted: {collected!r}"


def test_check_grammar_still_works_when_idle():
    """A blocking simple command (grammar validation) still round-trips normally
    when nothing else holds the queue - the lock must not deadlock the common,
    uncontended path."""
    r, wt = _make_runner_with_fake_worker(chunks=["x"])
    try:
        r.check_grammar("root ::= \"a\"")  # fake worker answers ("ok", None)
        assert r.count_tokens("hello") == 7
        assert r.count_messages_tokens([{"role": "user", "content": "hi"}]) == 7
    finally:
        _stop(r, wt)
