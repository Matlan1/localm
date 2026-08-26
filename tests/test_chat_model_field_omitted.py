# SPDX-License-Identifier: AGPL-3.0-or-later
"""An omitted "model" field must be distinguishable from an explicit "localm".

protocol.py's ChatRequest/CompletionRequest.model must not default to a TRUTHY
sentinel like the string "localm": with one, a request that OMITTED the field
entirely is indistinguishable from one explicitly asking for "localm" at
``reported_model = req.model or engine.display_name`` (routes/chat.py), the "or"
never falls through, and the client is told the literal string instead of the
model that answered.

FIXTURE PREMISE: the other tests for this fallback
(test_chat_reload_after_eviction.py) all send ``{"model": ""}`` explicitly, so the
value that distinguishes "omitted" from "explicitly emptied" is never in their
test data. These tests build the request body with no "model" key at all.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from localm.inference import http_server as hs
from tests.conftest import probe_double


class FakeEngine:
    def __init__(self, display_name):
        self.display_name = display_name
        self._loaded = False
        self.active_requests = 0
        self.supports_images = False
        self.model_path = f"Z:/models/{display_name}.gguf"

    @property
    def loaded(self):
        return self._loaded

    def load(self):
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


@pytest.fixture
def server(monkeypatch):
    """A server started WITH a model already loaded and active."""
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


def test_omitted_model_field_reports_the_serving_model(server):
    """The natural request shape: no "model" key in the body at all."""
    client, engines = server

    r = client.post("/v1/chat/completions",
                    json={"stream": False,
                          "messages": [{"role": "user", "content": "hi"}]})

    assert r.status_code == 200, r.text
    assert r.json()["model"] == "model-a", (
        f'omitted-field turn reported model {r.json()["model"]!r}, '
        f'expected the real serving model, not the "localm" sentinel')


def test_omitted_model_field_reports_the_serving_model_on_completions_route(server):
    """CompletionRequest carries the identical model default, so it is asserted
    separately and does not ride on the chat route's coverage."""
    client, engines = server

    r = client.post("/v1/completions", json={"stream": False, "prompt": "hi"})

    assert r.status_code == 200, r.text
    assert r.json()["model"] == "model-a", (
        f'omitted-field completion reported model {r.json()["model"]!r}, '
        f'expected the real serving model, not the "localm" sentinel')


def test_explicit_localm_still_echoes_localm(server):
    """The documented behaviour that must NOT change: a client that actually
    sends "localm" gets "localm" back, not the resolved engine's real name."""
    client, engines = server

    r = client.post("/v1/chat/completions",
                    json={"model": "localm", "stream": False,
                          "messages": [{"role": "user", "content": "hi"}]})

    assert r.status_code == 200, r.text
    assert r.json()["model"] == "localm", (
        f'explicit "localm" request reported model {r.json()["model"]!r}')


def test_omitted_model_still_400_when_nothing_can_be_resolved(monkeypatch):
    """The contract that must NOT change, omitted-field variant of
    test_empty_model_is_still_400_when_nothing_can_be_resolved
    (test_chat_reload_after_eviction.py): with no model loaded and none ever
    configured, an unnamed request is genuinely unserveable regardless of
    whether "unnamed" means the field was left out or sent as ""."""
    monkeypatch.setattr("localm.config.load_registry", lambda: {})
    hs._engines.clear()
    hs._engines_lru.clear()
    hs._active_model_name = None
    hs._default_model_name = None
    hs._engine = None

    client = TestClient(hs.create_app(None))
    r = client.post("/v1/chat/completions",
                    json={"messages": [{"role": "user", "content": "hi"}]})

    assert r.status_code == 400
    assert "Model parameter is required" in r.text
