# SPDX-License-Identifier: AGPL-3.0-or-later
"""A request that does not fit the loaded model's context window grows the
window (a reload with a larger reserved window) instead of being refused, and
a refusal that remains names the setting or limit that holds the window, never
a command.

Driven through the real chat and completions routes and switch_engine with a
backend stand-in that reports its window, the way tests/
test_context_capacity_guard.py drives the capacity guard."""

import asyncio

import pytest
from fastapi.testclient import TestClient

from localm.inference import http_server as hs
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
    for d in (hs._engines, hs._engines_lru, hs._inference_sems):
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
        (7000, None, 8192, 7000 + 2048),
        (100, 32768, 8192, 32768),
        (40000, 32768, 8192, 40000 + 2048),
        (100, None, None, None),
        (100, 4096, 8192, None),
    ])
    def test_the_prompt_plus_its_reply_buffer_or_the_declared_window(
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
