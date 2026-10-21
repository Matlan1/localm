# SPDX-License-Identifier: AGPL-3.0-or-later
"""Unattended model calls (memory consolidation, scheduled jobs) must not hold
the model against a chat request that is waiting for it.

Layers covered: the runner counting streams queued behind a running one, the
helper that stops a background call once one is queued, and the three callers
(automatic pass, manual route, scheduled job).
"""

from __future__ import annotations

import asyncio
import json
import multiprocessing as mp
import threading
import time

import pytest

from localm.inference.background_generation import (
    BACKGROUND_MAX_TOKENS, BackgroundYield, background_complete, foreground_waiting)
from localm.inference.backends.llamacpp._runner import ModelRunner
from localm.plugins.builtin.jobs import runner as jobs_runner
from localm.plugins.builtin.jobs.store import Job
from localm.plugins.builtin.memory import plug


class _FakeProc:
    exitcode = -11

    def is_alive(self):
        return True

    def terminate(self):
        pass

    def join(self, timeout=None):
        return None


def _make_runner() -> ModelRunner:
    ctx = mp.get_context("spawn")
    r = ModelRunner()
    r._req_q, r._resp_q, r._ctrl_q = ctx.Queue(), ctx.Queue(), ctx.Queue()
    r._proc = _FakeProc()
    return r


class _Engine:
    """Engine stand-in whose stream records how it was driven."""

    loaded = True
    display_name = "stub-model"

    def __init__(self, chunks=("a", "b", "c", "d"), reply=None):
        self.chunks = list(chunks)
        self.reply = reply
        self.streams_waiting = 0
        self.active_requests = 0
        self.last_finish_reason = "stop"
        self.calls: list[dict] = []
        self.consumed = 0
        self.closed = False
        self.wait_after: int | None = None

    def chat_stream(self, messages, **kw):
        self.calls.append({"messages": messages, **kw})
        return self._stream()

    def _stream(self):
        try:
            for i, chunk in enumerate(self.chunks if self.reply is None else [self.reply]):
                if self.wait_after is not None and i + 1 >= self.wait_after:
                    self.streams_waiting = 1
                self.consumed += 1
                yield chunk
        finally:
            self.closed = True


# --------------------------------------------------------------------------- #
#  The runner counts streams queued behind a running one                        #
# --------------------------------------------------------------------------- #

def test_runner_counts_a_stream_waiting_for_the_worker():
    r = _make_runner()
    assert r.streams_waiting == 0
    out: list[str] = []
    r._q_lock.acquire()
    t = threading.Thread(target=lambda: out.extend(r.chat_stream(messages=[])), daemon=True)
    t.start()
    try:
        deadline = time.monotonic() + 5
        while r.streams_waiting != 1 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert r.streams_waiting == 1
    finally:
        r._q_lock.release()

    r._req_q.get(timeout=5)
    r._resp_q.put(("chunk", "x"))
    r._resp_q.put(("done", {"finish_reason": "stop"}))
    t.join(5)

    assert not t.is_alive()
    assert out == ["x"]
    assert r.streams_waiting == 0


def test_runner_reports_nothing_waiting_on_a_multiplexed_runner():
    r = _make_runner()
    r._mux = True
    assert r.streams_waiting == 0


# --------------------------------------------------------------------------- #
#  background_complete                                                         #
# --------------------------------------------------------------------------- #

def test_background_complete_turns_thinking_off_and_caps_the_reply():
    eng = _Engine(reply="the answer")
    assert background_complete(eng, "decide this") == "the answer"
    (call,) = eng.calls
    assert call["thinking"] is False
    assert call["max_tokens"] == BACKGROUND_MAX_TOKENS
    assert call["messages"] == [{"role": "user", "content": "decide this"}]


def test_background_complete_stops_between_tokens_once_a_request_is_waiting():
    eng = _Engine(chunks=list("abcdefgh"))
    eng.wait_after = 2
    with pytest.raises(BackgroundYield):
        background_complete(eng, "p", yield_to_foreground=True)
    assert eng.consumed == 2
    assert eng.closed


def test_background_complete_does_not_start_while_a_request_is_waiting():
    eng = _Engine()
    eng.streams_waiting = 1
    with pytest.raises(BackgroundYield):
        background_complete(eng, "p", yield_to_foreground=True)
    assert eng.calls == []


def test_background_complete_ignores_a_waiting_request_unless_asked_to_yield():
    eng = _Engine(chunks=list("abcd"))
    eng.streams_waiting = 1
    assert background_complete(eng, "p") == "abcd"
    assert eng.consumed == 4


def test_background_yield_is_not_an_exception_a_step_can_swallow():
    assert not issubclass(BackgroundYield, Exception)
    from localm.memory.consolidate import judge_episode

    def complete(_prompt):
        raise BackgroundYield()

    with pytest.raises(BackgroundYield):
        judge_episode(complete, "Planned the billing migration")


def test_background_complete_logs_a_reply_cut_at_the_limit(caplog):
    eng = _Engine(reply="cut sho")
    eng.last_finish_reason = "length"
    with caplog.at_level("WARNING", logger="localm"):
        assert background_complete(eng, "p") == "cut sho"
    assert any("token limit" in r.getMessage() for r in caplog.records)


def test_foreground_waiting_reads_the_engine_count():
    eng = _Engine()
    assert foreground_waiting(eng) is False
    eng.streams_waiting = 2
    assert foreground_waiting(eng) is True
    assert foreground_waiting(object()) is False


def test_engine_reports_the_backends_waiting_count():
    from localm.inference.engine import Engine
    eng = Engine.__new__(Engine)

    class _Backend:
        streams_waiting = 3

    eng._backend = _Backend()
    assert eng.streams_waiting == 3
    eng._backend = object()
    assert eng.streams_waiting == 0


# --------------------------------------------------------------------------- #
#  The callers                                                                 #
# --------------------------------------------------------------------------- #

@pytest.fixture
def memhome(tmp_path, monkeypatch):
    import localm.config as cfg
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path))
    monkeypatch.setenv("LOCALM_MODE", "log")
    monkeypatch.setattr(cfg, "HOME_DIR", tmp_path)
    monkeypatch.setattr(cfg, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", tmp_path / "registry.json")
    monkeypatch.setattr(cfg, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(plug, "_home", lambda: tmp_path)
    monkeypatch.setattr(plug, "_memory_root", lambda: tmp_path / "memory")
    monkeypatch.setattr(plug, "_embed_fn", lambda: None)
    plug._auto_running = False
    sdir = tmp_path / "sessions"
    sdir.mkdir()
    rows = [{"type": "user", "data": {"content": "my name is Ada and I prefer metric units"}},
            {"type": "llm", "data": {"content": "Noted, Ada!"}}]
    (sdir / "s.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return tmp_path


_FACTS = json.dumps({"facts": [{"fact": "User's name is Ada", "confidence": 0.9}]})


def test_the_automatic_pass_runs_without_reasoning_and_capped(memhome, monkeypatch):
    eng = _Engine(reply=_FACTS)
    monkeypatch.setattr(plug, "_live_engine", lambda: eng)

    plug._auto_consolidate_bg()

    assert eng.calls, "the pass never called the model"
    assert all(c["thinking"] is False and c["max_tokens"] == BACKGROUND_MAX_TOKENS
               for c in eng.calls)
    assert any("Ada" in r.text for r in plug._chat_store().all())
    assert plug._auto_running is False


def test_the_automatic_pass_stops_when_a_chat_request_is_waiting(memhome, monkeypatch, caplog):
    eng = _Engine(chunks=list(_FACTS))
    eng.wait_after = 3
    monkeypatch.setattr(plug, "_live_engine", lambda: eng)
    plug._auto_running = True

    with caplog.at_level("INFO", logger="localm"):
        plug._auto_consolidate_bg()

    assert eng.closed
    assert len(eng.calls) == 1, "the pass went on to another model call"
    assert plug._chat_store().all() == []
    assert plug._auto_running is False
    assert any("a chat request is waiting" in r.getMessage() for r in caplog.records)
    assert not any(r.levelname == "WARNING" for r in caplog.records)
    due_in = plug.MEMORY_AUTO_MIN_INTERVAL - (time.time() - plug._auto_last_run())
    assert plug.MEMORY_AUTO_RETRY_AFTER - 5 <= due_in <= plug.MEMORY_AUTO_RETRY_AFTER + 5


def test_the_manual_route_runs_without_reasoning_and_does_not_yield(memhome, monkeypatch):
    eng = _Engine(reply=_FACTS)
    eng.streams_waiting = 1
    monkeypatch.setattr(plug, "_live_engine", lambda: eng)

    res = asyncio.run(plug.memory_consolidate(request=None))

    assert res["status"] == "ok"
    assert eng.calls and all(c["thinking"] is False for c in eng.calls)
    assert any("Ada" in r.text for r in plug._chat_store().all())


def test_the_scheduled_memory_job_stops_when_a_chat_request_is_waiting(memhome):
    eng = _Engine(chunks=list(_FACTS))
    eng.wait_after = 3
    job = Job(name="t", task_kind="memory", prompt="", schedule_kind="interval", schedule=60)

    out = jobs_runner._run_memory(job, engine=eng)

    assert "chat request was waiting" in out
    assert eng.closed
    assert eng.active_requests == 0
    assert all(c["thinking"] is False for c in eng.calls)
    assert plug._chat_store().all() == []


def test_hf_runner_counts_a_stream_waiting_for_the_worker():
    import queue

    from localm.inference.backends._hf_runner import HFRunner
    r = HFRunner()
    r._req_q, r._resp_q = queue.Queue(), queue.Queue()
    r._proc = _FakeProc()
    assert r.streams_waiting == 0
    out: list[str] = []
    r._q_lock.acquire()
    t = threading.Thread(target=lambda: out.extend(r.chat_stream()), daemon=True)
    t.start()
    try:
        deadline = time.monotonic() + 5
        while r.streams_waiting != 1 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert r.streams_waiting == 1
    finally:
        r._q_lock.release()
    r._resp_q.put(("chunk", "x"))
    r._resp_q.put(("done", {"finish_reason": "stop"}))
    t.join(5)

    assert not t.is_alive()
    assert out == ["x"]
    assert r.streams_waiting == 0


def test_backends_report_the_runners_waiting_count():
    from localm.inference.backends.gguf import GgufBackend
    from localm.inference.backends.hf import HFBackend

    class _Runner:
        streams_waiting = 2

    for cls in (GgufBackend, HFBackend):
        backend = cls.__new__(cls)
        backend._runner = None
        assert backend.streams_waiting == 0
        backend._runner = _Runner()
        assert backend.streams_waiting == 2


def test_a_yield_mid_episodic_pass_keeps_the_sessions_already_summarised(memhome):
    import os
    for stem, topic in (("a1", "the billing service Postgres migration"),
                        ("a2", "the release checklist")):
        rows = [{"type": "user", "data": {"content": f"let us work through {topic} in "
                                                     "detail and decide the next steps for it"}},
                {"type": "llm", "data": {"content": f"Here is a plan for {topic}."}}]
        p = memhome / "sessions" / f"{stem}.jsonl"
        p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
        old = time.time() - 7200 + (0 if stem == "a1" else 10)
        os.utime(p, (old, old))
    store = plug._chat_store()
    calls = []

    def complete(prompt):
        calls.append(prompt)
        if len(calls) == 2:
            raise BackgroundYield()
        return "Planned the billing service Postgres migration"

    with pytest.raises(BackgroundYield):
        plug._store_episodes(store, complete, embed_fn=None)
    assert len(calls) == 2

    retry_prompts = []

    def retry(prompt):
        retry_prompts.append(prompt)
        return "Worked through the release checklist"

    assert plug._store_episodes(store, retry, embed_fn=None) == 1
    assert len(retry_prompts) == 1 and "release checklist" in retry_prompts[0]


def test_a_deferred_pass_retries_soon_whatever_the_debounce_interval(memhome, monkeypatch):
    monkeypatch.setenv("LOCALM_MEMORY_AUTO_INTERVAL", "3600")
    eng = _Engine(chunks=list(_FACTS))
    eng.wait_after = 3
    monkeypatch.setattr(plug, "_live_engine", lambda: eng)
    plug._auto_running = True

    plug._auto_consolidate_bg()

    due_in = 3600 - (time.time() - plug._auto_last_run())
    assert plug.MEMORY_AUTO_RETRY_AFTER - 5 <= due_in <= plug.MEMORY_AUTO_RETRY_AFTER + 5


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
