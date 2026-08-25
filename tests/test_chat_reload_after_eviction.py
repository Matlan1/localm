# SPDX-License-Identifier: AGPL-3.0-or-later
"""An evicted chat model must come back on the next turn, unnamed turns included.

Reported against the 0.1.4 candidate: "chat wasnt able to automatically reload
the model after it was evicted by vram manager for embedding tests". The user
had to load it by hand from the Models page.

The mechanism, live-reproduced on a real server with a real GGUF before this
test was written:

    chat before eviction            -> 200
    unload every loaded model       -> 200 (what vram.evict_chat_for_embedder
                                       does, via http_server.unload_all_models)
    chat, model ""                  -> 400 in 4 ms, no reload   <-- the bug
    chat, model named               -> 200 in 9.7 s (reloaded)

get_engine already knows how to recover: it resolves an unnamed request through
``_active_model_name or _default_model_name`` and reloads the result, and its
own 503 documents "the transient window during an active-model eviction/unload
where _active_model_name was just cleared". None of that was reachable, because
the route refused an empty model on its first line, before get_engine ran.

FIXTURE PREMISE (diff-review-discipline.md item 19). The failing case needs
THREE things at once: a server that CAN resolve a model, the active pointer
CLEARED by a real eviction, and a request that does not name the model. The
pre-existing empty-model test builds its app with ``create_app(None)``, so
``_default_model_name`` is None and nothing is resolvable - that fixture can
never take the value this defect needs, which is why it stayed green through
the bug. These tests start from an app that HAS a startup model and run the real
unload_all_models coroutine to clear the pointer.
"""

from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

from localm.inference import http_server as hs
from tests.conftest import probe_double


class FakeEngine:
    """Stands in for the model backend only. What is under test is the ROUTE's
    name resolution and the reload handoff, not llama.cpp - the end-to-end
    behaviour with a real GGUF is covered by the live reproduction quoted above."""

    def __init__(self, display_name: str):
        self.display_name = display_name
        self._loaded = False
        self.active_requests = 0
        self.supports_images = False
        self.model_path = f"Z:/models/{display_name}.gguf"
        self.load_calls = 0

    @property
    def loaded(self):
        return self._loaded

    def load(self):
        self.load_calls += 1
        self._loaded = True

    def unload(self):
        self._loaded = False

    def set_load_cancel(self, cancel):
        pass

    def count_tokens(self, text):
        return len(text.split())

    def count_messages_tokens(self, messages):
        return 10

    def context_capacity(self):
        return 4096

    def chat_stream(self, messages, **gen_kwargs):
        yield "Hello from " + self.display_name

    def embed(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]


@pytest.fixture
def server(monkeypatch):
    """A server started WITH a model - the setup the report came from."""
    engines: dict[str, FakeEngine] = {}

    def factory(name):
        return engines.setdefault(name, FakeEngine(name))

    monkeypatch.setattr("localm.config.load_registry",
                        lambda: {"model-a": {"path": "Z:/models/model-a.gguf",
                                             "source": "local"}})
    monkeypatch.setattr("localm.model_manager.get_model_info",
                        lambda name: (f"Z:/models/{name}.gguf", "hint"))
    monkeypatch.setattr("localm.model_manager.get_model_mmproj", lambda name: None)
    monkeypatch.setattr("localm.discover.vram_info",
                        probe_double({"free": 10 * 1024 ** 3,
                                      "total": 16 * 1024 ** 3}))
    monkeypatch.setattr(hs, "_engine_factory", factory)

    hs._engines.clear()
    hs._engines_lru.clear()
    hs._inference_sems.clear()
    hs._last_activity_per_model.clear()
    hs._active_model_name = None
    hs._default_model_name = None
    hs._engine = None
    hs._inference_sem = None

    startup = factory("model-a")
    startup.load()
    app = hs.create_app(startup)
    return TestClient(app), engines


def _evict():
    """The real eviction, through the real function.

    vram.evict_chat_for_embedder does NOT round-trip over HTTP - its docstring
    is explicit that it submits ``http_server.unload_all_models`` onto the
    server's captured loop, precisely so the unload honours the in-flight
    request pin and serialises with get_engine. Calling that same coroutine here
    reproduces the eviction the report came from; driving the
    ``/v1/models/unload`` ROUTE instead would test a management-auth path the
    embedder never takes."""
    result = asyncio.run(hs.unload_all_models())
    # Eviction clears the active pointer.
    assert hs._active_model_name is None, (
        "test premise: eviction must clear the active model pointer")
    return result


def test_unnamed_turn_reloads_the_evicted_model(server):
    client, engines = server
    _evict()
    assert not engines["model-a"].loaded

    r = client.post("/v1/chat/completions",
                    json={"model": "", "stream": False,
                          "messages": [{"role": "user", "content": "hi"}]})

    assert r.status_code == 200, (
        f"an unnamed turn did not recover the evicted model: "
        f"{r.status_code} {r.text[:200]}")
    assert engines["model-a"].loaded, "the evicted model was never reloaded"
    # The reply names the model that answered: model_id is echoed straight into
    # the response envelope, and an unresolved field would return "model": "".
    assert r.json()["model"] == "model-a", (
        f'unnamed turn reported model {r.json()["model"]!r}')


def test_named_turn_still_reloads_the_evicted_model(server):
    """The half that already worked. Kept so a fix that recovers the unnamed
    case by breaking the named one cannot pass."""
    client, engines = server
    _evict()

    r = client.post("/v1/chat/completions",
                    json={"model": "model-a", "stream": False,
                          "messages": [{"role": "user", "content": "hi"}]})

    assert r.status_code == 200, r.text
    assert engines["model-a"].loaded


def test_completions_route_recovers_too(server):
    """/v1/completions carried the identical premature refusal."""
    client, engines = server
    _evict()

    r = client.post("/v1/completions",
                    json={"model": "", "stream": False, "prompt": "hi"})

    assert r.status_code == 200, r.text
    assert engines["model-a"].loaded
    assert r.json()["model"] == "model-a", (
        f'unnamed completion reported model {r.json()["model"]!r}')


def test_a_chat_hook_sees_the_model_actually_in_use(server):
    """A plugin hook that branches on ctx.model_id must not be handed "".

    chat/plug.py's thinking inlet does exactly that
    (`is_thinking_model(getattr(ctx, "model_id", "") or "")`). Before this
    change an unnamed request could not reach a hook at all, so letting one
    through with the raw empty string would have silently disabled that
    handling on the very recovery path being added here - a capability lost
    with nothing reporting it."""
    client, engines = server
    _evict()

    seen = {}

    def _record_inlet(messages, ctx):
        # Record, never assert, in here: run_inlet wraps every hook in
        # `except Exception`, so an assertion raised inside a hook is swallowed.
        seen["model_id"] = getattr(ctx, "model_id", None)
        return None

    pipeline = client.app.state.chat_pipeline
    pipeline.add_hook("inlet", _record_inlet, plugin="_test_probe")
    try:
        r = client.post("/v1/chat/completions",
                        json={"model": "", "stream": False,
                              "messages": [{"role": "user", "content": "hi"}]})
    finally:
        pipeline.remove_plugin("_test_probe")

    assert r.status_code == 200, r.text
    assert seen, "the inlet never ran, so this test proves nothing"
    assert seen["model_id"] == "model-a", (
        f"the inlet saw model_id={seen['model_id']!r}")


def test_empty_model_is_still_400_when_nothing_can_be_resolved(monkeypatch):
    """The contract that must NOT change: with no model loaded and none ever
    configured, an unnamed request is genuinely unserveable and the caller does
    have to name one. Only the RECOVERABLE case moved."""
    monkeypatch.setattr("localm.config.load_registry", lambda: {})
    hs._engines.clear()
    hs._engines_lru.clear()
    hs._active_model_name = None
    hs._default_model_name = None
    hs._engine = None

    client = TestClient(hs.create_app(None))
    r = client.post("/v1/chat/completions",
                    json={"model": "", "messages": [{"role": "user",
                                                     "content": "hi"}]})

    assert r.status_code == 400
    assert "Model parameter is required" in r.text
