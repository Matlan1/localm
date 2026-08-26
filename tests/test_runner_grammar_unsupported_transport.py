# SPDX-License-Identifier: AGPL-3.0-or-later
"""``GrammarUnsupportedError`` must SURVIVE the GGUF worker IPC as a type.

``_build_sampler`` REFUSES a lazy grammar it cannot apply rather than building
a chain with no grammar stage and answering with unconstrained text. That raise
happens in the isolated CHILD process, and exceptions do not cross a
``multiprocessing.Queue`` - only tagged tuples do.

The TYPE is what has to survive, not just the message. The untagged fallback in
both decoders is ``RuntimeError``, and ``GgufBackend.chat_stream`` reads a
``RuntimeError`` from the runner as "the isolated worker faulted": it UNLOADS
the model and reports a 503. Hence the explicit
``not isinstance(exc, RuntimeError)`` assertions below;
``pytest.raises(GrammarUnsupportedError)`` alone would be satisfied by a
RuntimeError subclass.

These drive the REAL parent-side decoders over REAL queues, substituting only
the child process's liveness check.
"""

from __future__ import annotations

import multiprocessing as mp

import pytest

from localm.inference.backends.base import (
    GRAMMAR_LAZY_UNSUPPORTED_MESSAGE,
    GrammarUnsupportedError,
)
from localm.inference.backends.llamacpp._runner import ModelRunner


class _AliveProc:
    """Stands in for the worker process's liveness check only; the queues and
    the parent-side decoders under test are real."""

    def is_alive(self):
        return True

    def terminate(self):
        return None

    def join(self, timeout=None):
        return None

    exitcode = 0


def _make_runner():
    ctx = mp.get_context("spawn")
    r = ModelRunner()
    r._req_q, r._resp_q, r._ctrl_q = ctx.Queue(), ctx.Queue(), ctx.Queue()
    r._proc = _AliveProc()
    r._spawn = lambda: None
    return r


def test_chat_stream_reraises_the_tagged_type_not_a_runtimeerror():
    r = _make_runner()
    r._resp_q.put(("error", GRAMMAR_LAZY_UNSUPPORTED_MESSAGE,
                   "GrammarUnsupportedError"))

    with pytest.raises(GrammarUnsupportedError) as ei:
        list(r.chat_stream(messages=[{"role": "user", "content": "hi"}]))

    assert GRAMMAR_LAZY_UNSUPPORTED_MESSAGE in str(ei.value)
    assert not isinstance(ei.value, RuntimeError), (
        "a RuntimeError out of chat_stream means 'the worker died' to "
        "GgufBackend, which unloads the model - a per-request refusal by a "
        "healthy worker must never cost the user their loaded model")


def test_simple_request_reraises_the_tagged_type_not_a_runtimeerror():
    """The second decoder reads the SAME protocol off the SAME queue, so it
    must honour the same tag."""
    r = _make_runner()
    r._resp_q.put(("error", GRAMMAR_LAZY_UNSUPPORTED_MESSAGE,
                   "GrammarUnsupportedError"))

    with pytest.raises(GrammarUnsupportedError) as ei:
        r._simple_request("check_grammar", "root ::= \"x\"")

    assert GRAMMAR_LAZY_UNSUPPORTED_MESSAGE in str(ei.value)
    assert not isinstance(ei.value, RuntimeError)


def test_child_dispatch_reports_the_refusal_instead_of_dying(monkeypatch):
    """The PRODUCER half, on the dispatch loop.

    ``_runner_main``'s chat_stream branch catches InvalidGrammarError and
    UnsupportedInputError and lets everything else escape, which kills the
    worker process and makes GgufBackend unload the model, so
    GrammarUnsupportedError needs its own arm there. This drives the real
    dispatch loop and asserts BOTH halves: it reports a tagged envelope, and it
    is still alive to serve the next command.
    """
    import queue
    import threading

    import localm._mp_spawn as mp_spawn
    from localm.inference.backends.llamacpp import _runner, _worker

    monkeypatch.setattr(mp_spawn, "install_parent_death_watchdog", lambda *a: None)
    monkeypatch.setattr(mp_spawn, "suppress_native_error_dialogs", lambda *a: None)

    class _FakeWorker:
        last_finish_reason = "stop"
        grammar_unsupported_this_call = False
        chatml_fallback_reason = None

        def __init__(self, **kw):
            pass

        def load(self):
            return {"ok": True}

        def chat_stream(self, **kw):
            raise GrammarUnsupportedError(GRAMMAR_LAZY_UNSUPPORTED_MESSAGE)
            yield   # pragma: no cover - makes this a generator, as the real one is

        def count_tokens(self, payload):
            return 42

        def close(self):
            pass

    monkeypatch.setattr(_worker, "GgufWorker", _FakeWorker)

    req_q, resp_q, ctrl_q = queue.Queue(), queue.Queue(), queue.Queue()
    died: list = []

    def _run():
        try:
            _runner._runner_main(req_q, resp_q, ctrl_q)
        except BaseException as e:      # noqa: BLE001 - escaping = the worker dies
            died.append(e)

    t = threading.Thread(target=_run, name="dispatch-under-test", daemon=True)
    t.start()
    try:
        req_q.put(("load", {}))
        assert resp_q.get(timeout=5)[0] == "ok"

        req_q.put(("chat_stream", {"messages": [{"role": "user", "content": "hi"}]}))
        envelope = resp_q.get(timeout=5)
        assert envelope[0] == "error", (
            f"expected a clean error envelope, got {envelope!r}")
        # len BEFORE index, so a missing tag reports the LOSS rather than an
        # IndexError.
        assert len(envelope) > 2, (
            f"the refusal crossed the IPC with NO type tag ({envelope!r}), so the "
            f"parent will raise RuntimeError and GgufBackend will unload the "
            f"user's model over a request the caller could simply resend")
        assert envelope[2] == "GrammarUnsupportedError", (
            f"untagged errors become RuntimeError, which makes GgufBackend unload "
            f"the model; got tag {envelope[2:]!r}")

        # Still serving.
        req_q.put(("count_tokens", "still alive?"))
        assert resp_q.get(timeout=5) == ("ok", 42)
        assert not died, f"the dispatch loop died instead of reporting: {died!r}"
    finally:
        req_q.put(None)
        t.join(timeout=5)


def test_an_untagged_error_is_still_a_runtimeerror_on_both_decoders():
    """Adding a tag arm must not soften an UNTAGGED error: that fallback is
    what reports a genuine worker fault, and a recoverable ValueError would keep
    serving from a model left in an unknown state."""
    r = _make_runner()
    r._resp_q.put(("error", "something native went wrong"))
    with pytest.raises(RuntimeError):
        list(r.chat_stream(messages=[{"role": "user", "content": "hi"}]))

    r2 = _make_runner()
    r2._resp_q.put(("error", "something native went wrong"))
    with pytest.raises(RuntimeError):
        r2._simple_request("check_grammar", "root ::= \"x\"")
