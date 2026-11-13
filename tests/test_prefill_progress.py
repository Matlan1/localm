# SPDX-License-Identifier: AGPL-3.0-or-later
"""A long prompt that is still being read is not killed as a stall: each prompt
batch the GGUF worker finishes reports progress, the status reaches the parent
process intact, and every status restarts the first-token deadline. A worker
that reports nothing is still caught."""

from __future__ import annotations

import pickle
import threading
import time

import pytest

from localm.inference import protocol
from localm.inference.backends.llamacpp import _runner as runner_mod
from localm.inference.backends.llamacpp import llama as llama_mod
from tests.test_runner_stream_timeouts import _make_runner


def _fake_llama(monkeypatch):
    llm = object.__new__(llama_mod.LlamaCpp)
    llm._ctx_ptr = object()
    llm._prefill_progress = None
    llm._cached_tokens = []
    monkeypatch.setattr(llm, "_state_checkpoints_wanted", lambda: False, raising=False)
    monkeypatch.setattr(llm, "_create_batch", lambda chunk, i, logits_at_last_only=True: chunk,
                        raising=False)
    monkeypatch.setattr(llm, "_after_prefill_chunk", lambda chunk, i: None, raising=False)
    monkeypatch.setattr(llama_mod.api, "llama_decode", lambda ctx, batch: 0)
    monkeypatch.setattr(llama_mod.api, "llama_batch_free", lambda batch: None)
    return llm


def test_each_prompt_batch_reports_progress(monkeypatch):
    monkeypatch.setattr(llama_mod, "PREFILL_PROGRESS_INTERVAL_S", 0.0)
    llm = _fake_llama(monkeypatch)
    seen = []
    with llm._prefill_progress_to(seen.append):
        assert llm._decode_prompt(list(range(10)), 0, 4) == 0
    assert [str(s) for s in seen] == [
        "Processing prompt... (4 of 10 tokens)",
        "Processing prompt... (8 of 10 tokens)",
        "Processing prompt... (10 of 10 tokens)"]
    assert all(s.code == "processing_progress" for s in seen)
    assert seen[-1].args == {"done": "10", "total": "10"}
    assert llm._prefill_progress is None, "the reporter does not outlive its reply"


def test_progress_is_throttled_but_the_last_batch_always_reports(monkeypatch):
    monkeypatch.setattr(llama_mod, "PREFILL_PROGRESS_INTERVAL_S", 3600.0)
    llm = _fake_llama(monkeypatch)
    seen = []
    with llm._prefill_progress_to(seen.append):
        llm._decode_prompt(list(range(10)), 0, 2)
    assert [str(s) for s in seen] == ["Processing prompt... (10 of 10 tokens)"]


def test_without_a_status_callback_nothing_is_reported(monkeypatch):
    llm = _fake_llama(monkeypatch)
    with llm._prefill_progress_to(None):
        assert llm._decode_prompt(list(range(10)), 0, 4) == 0
    assert llm._prefill_progress is None


def test_the_progress_status_survives_the_process_boundary():
    s = protocol.prefill_progress_status(2048, 30000)
    back = pickle.loads(pickle.dumps(s))
    assert str(back) == str(s)
    assert back.code == "processing_progress"
    assert back.args == {"done": "2048", "total": "30000"}
    assert protocol.status_code_of(back) == "processing_progress"
    assert protocol.STATUS_ARGS_BY_CODE["processing_progress"] == ("done", "total")


def test_a_prefill_longer_than_the_budget_that_reports_progress_is_not_killed(monkeypatch):
    monkeypatch.setattr(runner_mod, "_STREAM_CHUNK_TIMEOUT", 0.5)
    r = _make_runner()
    stop = threading.Event()

    def _child():
        r._req_q.get(timeout=5)
        for done in range(1, 9):              # 8 x 0.3 s = 2.4 s, budget 1 s
            if stop.wait(0.3):
                return
            r._resp_q.put(("status", protocol.prefill_progress_status(done, 8)))
        r._resp_q.put(("chunk", "ok"))
        r._resp_q.put(("done", {"finish_reason": "stop"}))

    child = threading.Thread(target=_child, daemon=True)
    child.start()
    statuses = []
    try:
        t0 = time.monotonic()
        out = list(r.chat_stream(first_chunk_timeout=1.0, on_status=statuses.append,
                                 messages=[]))
        assert out == ["ok"]
        assert time.monotonic() - t0 > 2.0
        assert not r._proc.terminated
        assert statuses and getattr(statuses[-1], "code", None) == "processing_progress"
    finally:
        stop.set(); child.join(2)


def test_a_prefill_that_stops_reporting_is_still_caught(monkeypatch):
    monkeypatch.setattr(runner_mod, "_STREAM_CHUNK_TIMEOUT", 0.5)
    r = _make_runner()

    def _child():
        r._req_q.get(timeout=5)
        r._resp_q.put(("status", protocol.prefill_progress_status(1, 8)))

    child = threading.Thread(target=_child, daemon=True)
    child.start()
    with pytest.raises(RuntimeError, match="made no progress for 1s"):
        list(r.chat_stream(first_chunk_timeout=1.0, messages=[]))
    child.join(2)
