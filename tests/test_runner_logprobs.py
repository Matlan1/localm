# SPDX-License-Identifier: AGPL-3.0-or-later
"""Token log probability records across the GGUF worker boundary.

Child side: ``_serve_stream`` sends the records gathered since the last chunk
as one ``logprobs`` envelope ahead of each chunk and ahead of ``done``, and
only for a stream that asked. Parent side: ``ModelRunner.chat_stream`` (inline
and multiplexed) hands each envelope's records to ``on_logprobs`` before the
chunk that follows, treats an unasked envelope as a protocol error, and
cancels and drains the child when ``on_logprobs`` raises."""
from __future__ import annotations

import queue
import threading

import pytest

from localm.inference.backends.llamacpp import _runner as runner_mod
from localm.inference.backends.llamacpp._runner import ModelRunner
from localm.inference.backends.llamacpp._worker import GgufWorker


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


def _runner(mux=False):
    r = ModelRunner()
    r._req_q, r._resp_q, r._ctrl_q = queue.Queue(), queue.Queue(), queue.Queue()
    r._proc = _Proc()
    if mux:
        r._start_mux()
    return r


def _script(r, envelopes, mux=False, on_cancel=None):
    """Answer the first stream request with *envelopes*; after a cancel, send
    *on_cancel* (default: a done)."""
    def run():
        cmd = r._req_q.get(timeout=5)
        sid = cmd[1]["sid"] if mux else None

        def put(env):
            r._resp_q.put(("stream", sid, env) if mux else env)
        for env in envelopes:
            put(env)
        try:
            msg = r._ctrl_q.get(timeout=5)
        except queue.Empty:
            return
        if msg[0] == "cancel_stream":
            for env in (on_cancel if on_cancel is not None else [("done", {"finish_reason": "stop"})]):
                put(env)
    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t


REC_A = (b"Hel", -0.1, ((b"Hel", -0.1),))
REC_B = (b"lo", -0.2, ((b"lo", -0.2),))


@pytest.mark.parametrize("mux", [False, True])
def test_records_reach_on_logprobs_before_the_chunk_they_belong_to(mux):
    r = _runner(mux)
    _script(r, [("logprobs", [REC_A]), ("chunk", "Hel"), ("logprobs", [REC_B]),
                ("chunk", "lo"), ("done", {"finish_reason": "stop"})], mux=mux)
    seen: list = []
    for chunk in r.chat_stream(messages=[], logprobs=1,
                               on_logprobs=lambda recs: seen.append(("records", recs))):
        seen.append(("chunk", chunk))
    assert seen == [("records", [REC_A]), ("chunk", "Hel"),
                    ("records", [REC_B]), ("chunk", "lo")]


@pytest.mark.parametrize("mux", [False, True])
def test_records_for_a_stream_that_did_not_ask_are_a_protocol_error(mux):
    r = _runner(mux)
    _script(r, [("logprobs", [REC_A]), ("chunk", "Hel"),
                ("done", {"finish_reason": "stop"})], mux=mux)
    with pytest.raises(RuntimeError, match="did not request them"):
        list(r.chat_stream(messages=[]))


@pytest.mark.parametrize("mux", [False, True])
def test_a_failing_on_logprobs_cancels_and_drains_the_child(mux):
    r = _runner(mux)
    child = _script(r, [("logprobs", [REC_A]), ("chunk", "late")], mux=mux,
                    on_cancel=[("chunk", "stray"), ("done", {"finish_reason": "stop"})])

    def boom(_records):
        raise ValueError("sink broke")
    with pytest.raises(ValueError, match="sink broke"):
        list(r.chat_stream(messages=[], logprobs=1, on_logprobs=boom))
    child.join(5)
    assert not child.is_alive(), "the child never saw a cancel_stream"
    assert r._resp_q.empty()


class _Worker:
    last_finish_reason = "stop"
    grammar_unsupported_this_call = False
    chatml_fallback_reason = None
    mtp_status = None
    mtp_active_this_call = False
    mtp_call_status = ""
    mtp_drafted = mtp_accepted = mtp_steps = mtp_paused_steps = 0
    mtp_skipped = ""
    spec_report = None

    def __init__(self, script):
        self.script = script
        self.kwargs = None

    def chat_stream(self, on_status=None, **kwargs):
        self.kwargs = kwargs
        sink = kwargs.get("logprob_sink")
        for records, text in self.script:
            if sink is not None:
                sink.extend(records)
            if text is not None:
                yield text


def _serve(worker, payload):
    out: list = []
    runner_mod._serve_stream(worker, payload, out.append, threading.Event())
    return out


def test_the_child_sends_records_ahead_of_each_chunk_and_of_done():
    worker = _Worker([([REC_A], None), ([], "Hel"), ([REC_B], "lo"), ([(b"!", -0.3, ())], None)])
    out = _serve(worker, {"messages": [], "logprobs": 2})
    assert [e[0] for e in out] == ["logprobs", "chunk", "logprobs", "chunk", "logprobs", "done"]
    assert out[0][1] == [REC_A] and out[2][1] == [REC_B]
    assert out[4][1] == [(b"!", -0.3, ())]
    assert worker.kwargs["logprobs"] == 2 and worker.kwargs["logprob_sink"] == []


def test_the_child_sends_no_records_and_no_sink_when_not_asked():
    worker = _Worker([([], "Hi")])
    out = _serve(worker, {"messages": []})
    assert [e[0] for e in out] == ["chunk", "done"]
    assert "logprob_sink" not in worker.kwargs


class _FaultingLlm:
    """create_chat_completion that scores a token, then faults in the grammar
    sampler before any text; the retry without the grammar answers."""

    def __init__(self):
        self.calls = []

    def create_chat_completion(self, **kw):
        self.calls.append(kw)
        sink = kw.get("logprob_sink")
        if kw.get("grammar"):
            sink.append((b"bad", -9.0, ()))
            raise OSError("grammar sampler fault")
        sink.append((b"ok", -0.1, ()))
        yield {"choices": [{"delta": {"content": "ok"}}]}
        yield {"choices": [{"delta": {}, "finish_reason": "stop"}]}


def test_a_retry_without_the_grammar_drops_the_failed_attempts_records():
    worker = GgufWorker.__new__(GgufWorker)
    worker._llm = _FaultingLlm()
    worker.stream_cancel = None
    sink: list = []
    out = list(worker.chat_stream([], grammar="root ::= x", logprobs=0, logprob_sink=sink))
    assert out == ["ok"]
    assert sink == [(b"ok", -0.1, ())]
    assert worker.grammar_unsupported_this_call is True
    assert all(c["logprobs"] == 0 for c in worker._llm.calls)
