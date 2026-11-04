# SPDX-License-Identifier: AGPL-3.0-or-later
"""A request that does not fit the loaded model's context window grows the
window (a reload with a larger reserved window) instead of being refused, and
a refusal that remains names the setting or limit that holds the window, never
a command.

Driven through the real chat and completions routes and switch_engine with a
backend stand-in that reports its window, the way tests/
test_context_capacity_guard.py drives the capacity guard."""

import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from localm.inference import compact, http_server as hs
from localm.inference.http_server import create_app
from localm.inference.protocol import GROWING_CONTEXT_STATUS
from tests.test_context_capacity_guard import (_SKIP_ROLE_NUDGE_SYSTEM, _EngineWithBackend,
                                               _MockBackend)


class _GrowableBackend(_MockBackend):
    """A backend that reserves context windows: a load after
    ``Engine.reserve_context`` reports the reserved window."""

    def __init__(self, capacity, limit=(None, None)):
        super().__init__(capacity)
        self.ctx_reserve = 0
        self._limit = limit
        self.loads = 0

    def ctx_reserve_limit(self):
        return self._limit

    def load(self):
        super().load()
        self.loads += 1
        if self.ctx_reserve:
            self.effective_ctx_max = max(self.effective_ctx_max, self.ctx_reserve)


@pytest.fixture(autouse=True)
def _clean_registries():
    yield
    for d in (hs._engines, hs._engines_lru, hs._inference_sems, hs._requested_context):
        d.clear()


def _post_chat(client, content, **extra):
    return client.post("/v1/chat/completions", json={
        "model": "test-cap-model", "stream": False,
        "messages": [_SKIP_ROLE_NUDGE_SYSTEM, {"role": "user", "content": content}],
        **extra})


class TestGrowInsteadOfRefuse:
    def test_an_overflowing_prompt_reloads_with_a_larger_window_and_is_answered(self):
        backend = _GrowableBackend(capacity=100)
        client = TestClient(create_app(_EngineWithBackend(backend)))

        resp = _post_chat(client, "x" * 150)

        assert resp.status_code == 200, resp.text
        assert resp.json()["choices"][0]["message"]["content"] == "Hello world!"
        assert backend.ctx_reserve == hs._CONTEXT_GROW_STEP
        assert backend.effective_ctx_max == hs._CONTEXT_GROW_STEP
        assert backend.loads == 1, "one reload"

    def test_a_declared_context_reserve_grows_a_window_the_prompt_fits(self):
        backend = _GrowableBackend(capacity=8192)
        client = TestClient(create_app(_EngineWithBackend(backend)))

        resp = _post_chat(client, "hi", context_reserve=32768)

        assert resp.status_code == 200, resp.text
        assert backend.effective_ctx_max == 32768

    def test_a_window_that_already_holds_the_request_is_not_reloaded(self):
        backend = _GrowableBackend(capacity=32768)
        client = TestClient(create_app(_EngineWithBackend(backend)))

        resp = _post_chat(client, "hi", context_reserve=32768)

        assert resp.status_code == 200
        assert backend.loads == 0

    def test_the_trained_window_stops_the_growth_and_the_refusal_says_so(self):
        backend = _GrowableBackend(capacity=100, limit=(150, "trained"))
        client = TestClient(create_app(_EngineWithBackend(backend)))

        resp = _post_chat(client, "x" * 150)

        assert resp.status_code == 413
        detail = resp.json()["detail"]
        assert "150-token context window this model was trained for" in detail
        assert "localm config" not in detail
        assert backend.loads == 1

    def test_a_fixed_ceiling_names_the_setting_without_a_command(self):
        backend = _GrowableBackend(capacity=100, limit=(100, "n_ctx_max"))
        client = TestClient(create_app(_EngineWithBackend(backend)))

        resp = _post_chat(client, "x" * 150)

        assert resp.status_code == 413
        detail = resp.json()["detail"]
        assert "set by Context window (max) (n_ctx_max)" in detail
        assert "localm config" not in detail
        assert backend.loads == 0, "nothing to grow to"

    def test_completions_grow_too(self):
        backend = _GrowableBackend(capacity=100)
        client = TestClient(create_app(_EngineWithBackend(backend)))

        resp = client.post("/v1/completions", json={
            "model": "test-cap-model", "prompt": "x" * 150, "stream": False})

        assert resp.status_code == 200, resp.text
        assert backend.effective_ctx_max == hs._CONTEXT_GROW_STEP

    def test_a_backend_that_cannot_reserve_is_refused_as_before(self):
        backend = _MockBackend(capacity=100)
        client = TestClient(create_app(_EngineWithBackend(backend)))

        resp = _post_chat(client, "x" * 150)

        assert resp.status_code == 413
        assert "the model's 100-token context window" in resp.json()["detail"]


class TestGrowDue:
    def _engine(self, capacity=100, busy=0):
        engine = _EngineWithBackend(_GrowableBackend(capacity))
        engine.active_requests = busy
        return engine

    def test_due_when_idle_and_smaller(self):
        assert hs._context_grow_due(self._engine(), 200) is True

    def test_not_due_while_another_request_holds_it(self):
        assert hs._context_grow_due(self._engine(busy=1), 200) is False

    def test_not_due_when_the_window_already_holds_it(self):
        assert hs._context_grow_due(self._engine(), 100) is False

    def test_not_due_for_a_backend_without_reserved_windows(self):
        assert hs._context_grow_due(_EngineWithBackend(_MockBackend(100)), 200) is False

    def test_switch_engine_reloads_the_same_engine_and_reports_the_status(self):
        engine = self._engine()
        hs._engines["m"] = engine
        hs._engines_lru.append("m")
        said = []

        async def _go():
            return await hs.switch_engine("m", None, preempt=False, activate=False,
                                          reserve_ctx=4096, on_status=said.append)

        from unittest.mock import patch
        with patch("localm.config.load_registry", return_value={}), \
                patch("localm.model_manager.get_model_info", return_value=None):
            result = asyncio.run(_go())

        assert result["status"] == "loaded"
        assert hs._engines["m"] is engine and engine.loaded
        assert engine._backend.effective_ctx_max == 4096
        assert said[0] == GROWING_CONTEXT_STATUS


class TestWanted:
    @pytest.mark.parametrize("prompt,min_context,capacity,expected", [
        (100, None, 8192, None),
        (7000, None, 8192, 2 * 8192),
        (20000, None, 8192, 20000 + 2048),
        (29500, None, 32768, 2 * 32768),
        (100, 32768, 8192, 32768),
        (100, 9000, 8192, 9000),
        (40000, 32768, 8192, 40000 + 2048),
        (100, None, None, None),
        (100, 4096, 8192, None),
    ])
    def test_the_prompt_plus_its_reply_buffer_at_least_doubled_or_the_declared_window(
            self, prompt, min_context, capacity, expected):
        engine = _EngineWithBackend(_MockBackend(capacity))
        assert hs.context_window_wanted(engine, prompt, min_context, capacity) == expected


class TestGrowthSafety:
    def test_a_reload_that_does_not_grow_the_window_is_not_repeated(self):
        backend = _GrowableBackend(capacity=100)

        def _load_without_growing():
            backend._loaded = True
            backend.unloaded = False
            backend.loads += 1

        backend.load = _load_without_growing
        client = TestClient(create_app(_EngineWithBackend(backend)))

        first = _post_chat(client, "x" * 150)
        second = _post_chat(client, "x" * 150)

        assert first.status_code == second.status_code == 413
        assert backend.loads == 1, "the second request does not reload again"

    def test_a_model_whose_files_cannot_be_resolved_is_used_as_loaded(self, monkeypatch):
        backend = _GrowableBackend(capacity=100)
        monkeypatch.setattr("localm.config.load_registry",
                            lambda: {"other": {"path": "models/other.gguf"}})
        monkeypatch.setattr("localm.model_manager.get_model_info", lambda name: None)
        client = TestClient(create_app(_EngineWithBackend(backend)))

        resp = _post_chat(client, "x" * 150)

        assert resp.status_code == 413, resp.text
        assert backend.loads == 0 and backend.ctx_reserve == 0

    def test_the_request_keeps_its_pin_through_the_reload(self):
        backend = _GrowableBackend(capacity=100)
        engine = _EngineWithBackend(backend)
        seen = []
        real_load = backend.load

        def _load():
            seen.append(engine.active_requests)
            real_load()

        backend.load = _load
        client = TestClient(create_app(engine))

        resp = _post_chat(client, "x" * 150)

        assert resp.status_code == 200
        assert seen == [1], "the reload ran while the request still held the engine"
        assert engine.active_requests == 0

    def test_a_failed_growth_reload_loads_the_model_again_with_its_window(self):
        backend = _GrowableBackend(capacity=100)
        real_load = backend.load
        calls = []

        def _load():
            calls.append(backend.ctx_reserve)
            if len(calls) == 1:
                raise RuntimeError("out of memory")
            real_load()

        backend.load = _load
        engine = _EngineWithBackend(backend)
        client = TestClient(create_app(engine))

        resp = _post_chat(client, "x" * 150)

        assert resp.status_code == 413, resp.text
        assert calls == [hs._CONTEXT_GROW_STEP, 0]
        assert hs._engines["test-cap-model"].loaded

    @staticmethod
    def _failing_first_load(backend, during=None):
        real_load = backend.load
        calls = []

        def _load():
            calls.append(backend.ctx_reserve)
            if len(calls) == 1:
                if during is not None:
                    during()
                raise RuntimeError("out of memory")
            real_load()

        backend.load = _load
        return calls

    def test_a_failed_growth_reload_leaves_the_active_model_active(self):
        backend = _GrowableBackend(capacity=100)
        calls = self._failing_first_load(backend)
        engine = _EngineWithBackend(backend)
        client = TestClient(create_app(engine))
        assert client.get("/health").json()["loaded"] is True

        resp = _post_chat(client, "x" * 150)

        assert resp.status_code == 413, resp.text
        assert len(calls) == 2, "the fault fired and the model loaded again"
        assert hs._active_model_name == "test-cap-model"
        assert hs._engine is engine
        assert client.get("/health").json()["loaded"] is True
        listed = {m["id"]: m for m in client.get("/v1/models").json()["data"]}
        assert listed["test-cap-model"]["active"] is True

    def test_a_failed_growth_reload_of_another_model_does_not_make_it_active(self, monkeypatch):
        backend = _GrowableBackend(capacity=100)
        calls = self._failing_first_load(backend)
        engine = _EngineWithBackend(backend, display_name="m")
        engine.active_requests = 1
        other = _EngineWithBackend(_GrowableBackend(capacity=100), display_name="other")
        hs._engines.update({"m": engine, "other": other})
        monkeypatch.setattr(hs, "_active_model_name", "other")
        monkeypatch.setattr(hs, "_engine", other)
        monkeypatch.setattr("localm.config.load_registry", lambda: {})
        monkeypatch.setattr("localm.model_manager.get_model_info", lambda name: None)

        answering = asyncio.run(hs.grow_context_window(engine, 4096))

        assert answering is engine and engine.loaded
        assert len(calls) == 2
        assert hs._active_model_name == "other"
        assert hs._engine is other

    def test_a_model_made_active_during_a_failed_reload_stays_active(self, monkeypatch):
        backend = _GrowableBackend(capacity=100)
        calls = self._failing_first_load(
            backend, during=lambda: setattr(hs, "_active_model_name", "other"))
        client = TestClient(create_app(_EngineWithBackend(backend)))
        hs._engines["other"] = _EngineWithBackend(_GrowableBackend(capacity=100),
                                                  display_name="other")
        monkeypatch.setattr(hs, "_active_model_name", hs._active_model_name)

        resp = _post_chat(client, "x" * 150)

        assert resp.status_code == 413, resp.text
        assert len(calls) == 2
        assert hs._active_model_name == "other"


def _long_chat(turns, size=400):
    """A chat of *turns* user and assistant pairs of *size* tokens each (one
    token a character on the stand-in backend), ending on a new user message."""
    messages = [_SKIP_ROLE_NUDGE_SYSTEM]
    for i in range(turns):
        messages.append({"role": "user", "content": f"{i} " + "u" * size})
        messages.append({"role": "assistant", "content": f"{i} " + "a" * size})
    messages.append({"role": "user", "content": "and now?"})
    return messages


def _post_messages(client, messages, stream=False, **extra):
    return client.post("/v1/chat/completions", json={
        "model": "test-cap-model", "stream": stream, "messages": messages, **extra})


def _stream_statuses(resp):
    """The status codes of an SSE reply, and whether it ended on an error."""
    codes, errored = [], False
    for line in resp.text.splitlines():
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        chunk = json.loads(line[6:])
        errored = errored or bool(chunk.get("localm_error"))
        for choice in chunk.get("choices") or []:
            code = (choice.get("delta") or {}).get("status_code")
            if code:
                codes.append(code)
    return codes, errored


class _SummaryHogBackend(_GrowableBackend):
    """A model whose summary ignores its token budget, so a compacted
    conversation can still overflow the window."""

    def chat_stream(self, messages, **kwargs):
        if "Summarise the following" in str(messages[0].get("content", "")):
            yield "s" * 9000
            return
        yield from super().chat_stream(messages, **kwargs)


class TestCompactionComesBeforeGrowth:
    def test_a_long_chat_compacts_and_keeps_the_model_loaded(self):
        backend = _GrowableBackend(capacity=16384)
        client = TestClient(create_app(_EngineWithBackend(backend)))
        messages = _long_chat(19)
        assert 16384 - 2048 < backend.count_messages_tokens(messages) <= 16384

        resp = _post_messages(client, messages)

        assert resp.status_code == 200, resp.text
        assert resp.headers["X-Localm-Context-Compacted"] == "1"
        assert backend.loads == 0
        assert backend.effective_ctx_max == 16384

    def test_a_streamed_long_chat_compacts_and_keeps_the_model_loaded(self):
        backend = _GrowableBackend(capacity=16384)
        client = TestClient(create_app(_EngineWithBackend(backend)))

        resp = _post_messages(client, _long_chat(19), stream=True)

        codes, errored = _stream_statuses(resp)
        assert "compacting" in codes
        assert "growing_context" not in codes
        assert not errored
        assert backend.loads == 0

    @pytest.mark.parametrize("stream", [False, True])
    def test_a_chat_far_past_the_window_never_reloads(self, stream):
        backend = _GrowableBackend(capacity=16384)
        client = TestClient(create_app(_EngineWithBackend(backend)))

        for turns in range(15, 90, 5):
            resp = _post_messages(client, _long_chat(turns), stream=stream)
            assert resp.status_code == 200, resp.text
            if stream:
                assert not _stream_statuses(resp)[1]

        assert backend.loads == 0
        assert backend.effective_ctx_max == 16384

    @pytest.mark.parametrize("stream", [False, True])
    def test_a_message_larger_than_the_window_grows_without_compacting(self, stream):
        backend = _GrowableBackend(capacity=16384)
        client = TestClient(create_app(_EngineWithBackend(backend)))
        messages = _long_chat(5)
        messages[-1]["content"] = "p" * 20000

        resp = _post_messages(client, messages, stream=stream)

        assert resp.status_code == 200, resp.text
        if stream:
            codes, errored = _stream_statuses(resp)
            assert "compacting" not in codes
            assert not errored
        else:
            assert "X-Localm-Context-Compacted" not in resp.headers
        assert backend.loads == 1
        assert backend.effective_ctx_max == 2 * 16384

    @pytest.mark.parametrize("field", ["min_context", "context_reserve"])
    def test_a_declared_window_grows_a_chat_that_could_compact(self, field):
        backend = _GrowableBackend(capacity=16384)
        client = TestClient(create_app(_EngineWithBackend(backend)))

        resp = _post_messages(client, _long_chat(19), **{field: 8192})

        assert resp.status_code == 200, resp.text
        assert "X-Localm-Context-Compacted" not in resp.headers
        assert backend.loads == 1
        assert backend.effective_ctx_max == 2 * 16384

    def test_a_short_conversation_grows_since_nothing_can_compact(self):
        backend = _GrowableBackend(capacity=8192)
        client = TestClient(create_app(_EngineWithBackend(backend)))

        resp = _post_chat(client, "x" * 7000)

        assert resp.status_code == 200, resp.text
        assert backend.loads == 1
        assert backend.effective_ctx_max == 2 * 8192

    def test_growth_doubles_so_the_next_turns_do_not_reload(self):
        backend = _GrowableBackend(capacity=8192)
        client = TestClient(create_app(_EngineWithBackend(backend)))

        for size in (7000, 8000, 9000, 10000, 12000):
            assert _post_chat(client, "x" * size).status_code == 200

        assert backend.loads == 1

    def test_a_window_at_its_limit_still_compacts(self):
        backend = _GrowableBackend(capacity=16384, limit=(16384, "trained"))
        client = TestClient(create_app(_EngineWithBackend(backend)))
        messages = _long_chat(5)
        messages[-1]["content"] = "p" * 13000

        resp = _post_messages(client, messages)

        assert resp.status_code == 200, resp.text
        assert resp.headers["X-Localm-Context-Compacted"] == "1"
        assert backend.loads == 0

    def test_a_compacted_prompt_that_still_overflows_grows(self):
        backend = _SummaryHogBackend(capacity=8192)
        client = TestClient(create_app(_EngineWithBackend(backend)))
        messages = _long_chat(9)
        assert backend.count_messages_tokens(messages) > 8192 - 2048

        resp = _post_messages(client, messages)

        assert resp.status_code == 200, resp.text
        assert resp.headers["X-Localm-Context-Compacted"] == "1"
        assert backend.loads == 1
        assert backend.effective_ctx_max > 9000


class TestRoomAfterCompaction:
    @staticmethod
    def _count(messages):
        return sum(len(str(m.get("content", ""))) for m in messages)

    def _chat_with_tail(self, tail_tokens):
        return [{"role": "user", "content": "older"},
                {"role": "assistant", "content": "older"},
                {"role": "user", "content": "x" * tail_tokens},
                {"role": "assistant", "content": ""},
                {"role": "user", "content": ""},
                {"role": "assistant", "content": ""},
                {"role": "user", "content": ""}]

    def test_the_kept_tail_the_summary_and_the_reply_must_fit(self):
        capacity = 16384
        bound = (compact.reply_reserve(capacity) + compact.template_margin(capacity)
                 + max(compact.summary_reply_tokens(capacity),
                       compact.DIGEST_MAX_CHARS // compact.CHARS_PER_TOKEN))
        room = capacity - bound
        assert compact.room_after_compaction(self._chat_with_tail(room), capacity, self._count)
        assert not compact.room_after_compaction(
            self._chat_with_tail(room + 1), capacity, self._count)

    def test_it_measures_what_compaction_keeps(self):
        messages = self._chat_with_tail(100)
        kept = []

        def _count(ms):
            kept.append(ms)
            return self._count(ms)

        compact.room_after_compaction(messages, 16384, _count)
        assert kept == [messages[2:]]


    def test_a_switch_that_takes_the_model_during_the_reload_asks_for_a_retry(self, monkeypatch):
        backend = _GrowableBackend(capacity=100)
        engine = _EngineWithBackend(backend)
        client = TestClient(create_app(engine))

        async def _superseded(name, *args, **kwargs):
            hs._engines.pop(name, None)
            return {"status": "superseded", "model": name}

        monkeypatch.setattr(hs, "switch_engine", _superseded)

        resp = _post_chat(client, "x" * 150)

        assert resp.status_code == 503, resp.text
        assert "retry the request" in resp.json()["detail"]
        assert engine.active_requests == 0, "the request gave up its pin"


class TestRequestedWindowSizesTheLoad:
    def test_the_next_load_reserves_the_requested_window_once(self):
        engine = _EngineWithBackend(_GrowableBackend(capacity=100))
        hs.request_context_window("m", 32768)
        hs.request_context_window("m", 4096)

        hs._apply_requested_context("m", engine, None)

        assert engine.ctx_reserve == 32768
        engine._backend.ctx_reserve = 0
        hs._apply_requested_context("m", engine, None)
        assert engine.ctx_reserve == 0, "the request is used once"

    def test_an_engine_that_cannot_reserve_ignores_it(self):
        engine = _EngineWithBackend(_MockBackend(capacity=100))
        hs.request_context_window("m", 32768)

        hs._apply_requested_context("m", engine, None)

        assert not hasattr(engine._backend, "ctx_reserve")
        assert "m" not in hs._requested_context

    def test_switch_engine_applies_it_before_the_load(self):
        engine = _EngineWithBackend(_GrowableBackend(capacity=100))
        engine._backend._loaded = False
        seen = []
        real_load = engine._backend.load

        def _load():
            seen.append(engine._backend.ctx_reserve)
            real_load()

        engine._backend.load = _load
        hs.request_context_window("m", 8192)

        from unittest.mock import patch
        with patch("localm.config.load_registry", return_value={}), \
                patch("localm.model_manager.get_model_info", return_value=None):
            asyncio.run(hs.switch_engine("m", lambda name: engine))

        assert seen == [8192]
        assert engine.context_capacity() == 8192
