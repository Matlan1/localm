# SPDX-License-Identifier: AGPL-3.0-or-later
"""A streaming chat request whose preparation (model load, inlet hooks, request
checks) takes noticeable time opens its stream early and reports each phase as
a status chunk; the reply, the response headers and any refusal follow
in-stream."""

from __future__ import annotations

import asyncio
import json
import threading
import time

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import localm.inference.http_server as hs
from localm.inference.backends.base import LOADING_MODEL_STATUS
from localm.inference.engine import Engine
from localm.inference.protocol import (
    RECALLING_MEMORY_STATUS, STATUS_CODE_BY_TEXT,
)

SLOW_LOAD_S = 1.0


class FakeEngine:
    def __init__(self, name, load_s=0.0, load_error=None):
        self.display_name = name
        self.loaded = False
        self.supports_images = False
        self.can_be_multimodal = False
        self.last_finish_reason = "stop"
        self.unloading = False
        self.load_s = load_s
        self.load_error = load_error

    def load(self):
        time.sleep(self.load_s)
        if self.load_error is not None:
            raise self.load_error
        self.loaded = True

    def unload(self):
        self.loaded = False

    def chat_stream(self, messages, on_status=None, **kw):
        yield f"answered-by-{self.display_name}"

    def count_tokens(self, text):
        return 3

    def count_messages_tokens(self, messages):
        return 5

    def context_capacity(self):
        return 8192


def _reg(*names):
    return {n: {"path": f"Z:/models/{n}.gguf", "source": "local",
                "model_type": "llm", "context_length": 8192} for n in names}


@pytest.fixture
def serve(monkeypatch):
    """``serve(engines)`` returns a TestClient over a server whose registry
    holds *engines* (name -> FakeEngine); "plain" is loaded at startup."""
    clients = []

    def _make(engines):
        registry = _reg(*engines)
        monkeypatch.setattr("localm.config.load_registry", lambda: registry)
        monkeypatch.setattr("localm.model_manager.load_registry", lambda: registry)
        monkeypatch.setattr("localm.model_manager.get_model_info",
                            lambda name: (f"Z:/models/{name}.gguf", "hint"))
        monkeypatch.setattr("localm.model_manager.get_model_mmproj", lambda name: None)
        monkeypatch.setattr(hs, "_engine_factory", lambda name: engines[name])
        hs._engines.clear()
        hs._engines_lru.clear()
        hs._inference_sems.clear()
        hs._last_activity_per_model.clear()
        hs._active_model_name = None
        hs._engine = None
        hs._inference_sem = None
        startup = engines["plain"]
        startup.load()
        client = TestClient(hs.create_app(startup))
        client.__enter__()
        clients.append(client)
        return client

    yield _make
    for c in clients:
        c.__exit__(None, None, None)


def _events(text: str) -> list:
    """The stream's data payloads, parsed, plus keepalive comments as None."""
    out = []
    for line in text.splitlines():
        if line.startswith("data: ") and line != "data: [DONE]":
            out.append(json.loads(line[len("data: "):]))
        elif line.startswith(":"):
            out.append(None)
    return out


def _delta(ev) -> dict:
    return ((ev or {}).get("choices") or [{}])[0].get("delta") or {}


def _post(client, model, **extra):
    return client.post("/v1/chat/completions", json={
        "model": model, "stream": True,
        "messages": [{"role": "user", "content": "hi"}], **extra})


class TestSlowLoadOpensTheStreamEarly:
    def test_loading_status_precedes_the_reply(self, serve):
        client = serve({"plain": FakeEngine("plain"),
                        "slow": FakeEngine("slow", load_s=SLOW_LOAD_S)})
        r = _post(client, "slow")
        assert r.status_code == 200
        events = [e for e in _events(r.text) if e is not None]
        statuses = [(_delta(e).get("status"), _delta(e).get("status_code"))
                    for e in events if _delta(e).get("status")]
        content = "".join(_delta(e).get("content") or "" for e in events)
        assert content == "answered-by-slow"
        assert (LOADING_MODEL_STATUS, "loading_model") in statuses
        first_status = next(i for i, e in enumerate(events) if _delta(e).get("status"))
        first_content = next(i for i, e in enumerate(events) if _delta(e).get("content"))
        assert first_status < first_content
        assert sum(1 for e in events if _delta(e).get("role") == "assistant") == 1
        assert len({e["id"] for e in events}) == 1

    def test_a_fast_request_keeps_its_headers_and_sends_no_meta_chunk(self, serve):
        client = serve({"plain": FakeEngine("plain")})
        r = _post(client, "plain")
        assert r.status_code == 200
        events = [e for e in _events(r.text) if e is not None]
        assert not any("localm_headers" in e for e in events)
        assert not any(_delta(e).get("status") == LOADING_MODEL_STATUS for e in events)

    def test_a_failed_slow_load_ends_the_stream_with_its_status_and_detail(self, serve):
        client = serve({"plain": FakeEngine("plain"),
                        "broken": FakeEngine("broken", load_s=SLOW_LOAD_S,
                                             load_error=OSError("weights missing"))})
        r = _post(client, "broken")
        assert r.status_code == 200
        events = [e for e in _events(r.text) if e is not None]
        last = events[-1]
        assert last["choices"][0]["finish_reason"] == "error"
        err = last["localm_error"]
        assert isinstance(err["status"], int) and err["status"] >= 500
        assert "weights missing" in err["detail"]
        content = "".join(_delta(e).get("content") or "" for e in events)
        assert content == err["detail"]
        assert r.text.rstrip().endswith("data: [DONE]")


class TestStreamAfterPrep:
    """The early stream itself, driven directly."""

    def _run(self, coro):
        return asyncio.run(coro)

    def test_statuses_follow_progress_and_headers_arrive_in_stream(self):
        async def _drive():
            progress = hs.PrepProgress()

            async def _prep():
                progress.set(LOADING_MODEL_STATUS)
                await asyncio.sleep(0.05)
                progress.set(RECALLING_MEMORY_STATUS)
                await asyncio.sleep(0.05)
                return "prepared"

            async def _reply(prepared, chunk_id):
                yield f"data: reply-{prepared}-{chunk_id}\n\n"

            task = asyncio.ensure_future(_prep())
            await asyncio.sleep(0)
            lines = [line async for line in hs.stream_after_prep(
                task, progress, "m", _reply, engine_of=lambda p: None,
                headers_of=lambda p: {"X-Localm-Memory": "{\"n\":1}"})]
            return lines

        lines = self._run(_drive())
        payloads = [json.loads(line[6:]) for line in lines
                    if line.startswith("data: {")]
        statuses = [_delta(p).get("status") for p in payloads if _delta(p).get("status")]
        assert statuses == [LOADING_MODEL_STATUS, RECALLING_MEMORY_STATUS]
        codes = [_delta(p).get("status_code") for p in payloads if _delta(p).get("status")]
        assert codes == [STATUS_CODE_BY_TEXT[LOADING_MODEL_STATUS],
                         STATUS_CODE_BY_TEXT[RECALLING_MEMORY_STATUS]]
        meta = [p["localm_headers"] for p in payloads if "localm_headers" in p]
        assert meta == [{"X-Localm-Memory": "{\"n\":1}"}]
        chunk_id = payloads[0]["id"]
        assert lines[-1] == f"data: reply-prepared-{chunk_id}\n\n"

    def test_keepalive_while_a_phase_runs_long(self, monkeypatch):
        monkeypatch.setattr(hs, "PREP_KEEPALIVE_S", 0.05)

        async def _drive():
            progress = hs.PrepProgress()

            async def _prep():
                progress.set(LOADING_MODEL_STATUS)
                await asyncio.sleep(0.3)
                return "p"

            async def _reply(prepared, chunk_id):
                yield "data: done\n\n"

            task = asyncio.ensure_future(_prep())
            await asyncio.sleep(0)
            return [line async for line in hs.stream_after_prep(
                task, progress, "m", _reply, engine_of=lambda p: None,
                headers_of=lambda p: {})]

        lines = self._run(_drive())
        assert ": keepalive\n\n" in lines
        assert sum(1 for line in lines if LOADING_MODEL_STATUS in line) == 1

    def test_an_http_refusal_becomes_an_error_reply(self):
        async def _drive():
            progress = hs.PrepProgress()

            async def _prep():
                await asyncio.sleep(0.01)
                raise HTTPException(413, "too long for the model")

            async def _reply(prepared, chunk_id):
                yield "data: never\n\n"

            task = asyncio.ensure_future(_prep())
            return [line async for line in hs.stream_after_prep(
                task, progress, "m", _reply, engine_of=lambda p: None,
                headers_of=lambda p: {})]

        lines = self._run(_drive())
        assert "data: never\n\n" not in lines
        done = json.loads(lines[-2][6:])
        assert done["choices"][0]["finish_reason"] == "error"
        assert done["localm_error"] == {"status": 413, "detail": "too long for the model"}
        assert lines[-1] == "data: [DONE]\n\n"

    def test_a_client_gone_before_the_reply_releases_the_pin_when_prep_ends(
            self, monkeypatch):
        released = []
        monkeypatch.setattr(hs, "_unpin", lambda engine: released.append(engine))

        async def _drive():
            progress = hs.PrepProgress()
            gate = asyncio.Event()

            async def _prep():
                progress.set(LOADING_MODEL_STATUS)
                await gate.wait()
                return "the-engine"

            async def _reply(prepared, chunk_id):
                yield "data: never\n\n"

            task = asyncio.ensure_future(_prep())
            gen = hs.stream_after_prep(
                task, progress, "m", _reply, engine_of=lambda p: p,
                headers_of=lambda p: {})
            await gen.__anext__()
            await gen.__anext__()
            await gen.aclose()
            assert released == []
            gate.set()
            await task
            await asyncio.sleep(0)
            return released

        assert self._run(_drive()) == ["the-engine"]


class TestEngineReloadReportsLoading:
    class _Backend:
        def __init__(self, loaded):
            self.loaded = loaded
            self.loads = 0

        def load(self):
            self.loads += 1
            self.loaded = True

        def chat_stream(self, messages, **kw):
            return iter(["x"])

    def _engine(self, loaded):
        engine = Engine.__new__(Engine)
        engine._backend = self._Backend(loaded)
        engine.display_name = "m"
        return engine

    def test_an_unloaded_model_reports_loading_before_it_reloads(self):
        engine = self._engine(loaded=False)
        seen = []

        def _on_status(text):
            seen.append((text, engine._backend.loads))

        list(engine.chat_stream([{"role": "user", "content": "hi"}], on_status=_on_status))
        assert seen == [(LOADING_MODEL_STATUS, 0)]
        assert engine._backend.loads == 1

    def test_a_loaded_model_reports_no_loading(self):
        engine = self._engine(loaded=True)
        seen = []
        list(engine.chat_stream([{"role": "user", "content": "hi"}], on_status=seen.append))
        assert seen == []


class TestMemoryInletAnnouncesRecall:
    def test_the_hook_reports_recalling_on_the_loop_thread(self, monkeypatch):
        from localm.plugins.builtin.memory import plug

        def _fake_body(messages, ctx, announce=None):
            announce()
            return None

        monkeypatch.setattr(plug, "_memory_inlet", _fake_body)

        class _Ctx:
            state: dict = {}

        seen = []

        async def _drive():
            ctx = _Ctx()
            loop_thread = threading.get_ident()
            ctx.on_status = lambda text: seen.append((text, threading.get_ident() == loop_thread))
            await plug._memory_inlet_hook([{"role": "user", "content": "hi"}], ctx)
            await asyncio.sleep(0)

        asyncio.run(_drive())
        assert seen == [(RECALLING_MEMORY_STATUS, True)]

    def test_a_recall_that_runs_announces_once_before_the_lookup(self, monkeypatch):
        from localm.plugins.builtin.memory import plug
        order = []

        class _Store:
            def recall(self, query, **kw):
                order.append("recall")
                return []

        monkeypatch.setattr(plug, "_recall_enabled", lambda: True)
        monkeypatch.setattr(plug, "_persist_enabled", lambda: False)
        monkeypatch.setattr(plug, "_recall_in_privacy", lambda surface: True)
        monkeypatch.setattr(plug, "_chat_store", lambda principal=None: _Store())
        monkeypatch.setattr(plug, "_embed_fn", lambda: None)
        monkeypatch.setattr(plug, "_legacy_bullets", lambda: [])
        plug._memory_inlet([{"role": "user", "content": "what is my name"}], None,
                           announce=lambda: order.append("announce"))
        assert order == ["announce", "recall"]

    def test_recall_that_does_not_run_announces_nothing(self, monkeypatch):
        from localm.plugins.builtin.memory import plug
        monkeypatch.setattr(plug, "_recall_enabled", lambda: False)
        calls = []
        plug._memory_inlet([{"role": "user", "content": "hi"}], None,
                           announce=lambda: calls.append(1))
        assert calls == []
