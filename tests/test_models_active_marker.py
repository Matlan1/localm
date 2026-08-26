# SPDX-License-Identifier: AGPL-3.0-or-later
"""GET /v1/models must mark the active model, and remote_model_status must read
that marker, never data[0].

GET /v1/models returns EVERY registered model (sorted) with a per-model `loaded`
flag, so data[0] is the alphabetically-first REGISTERED model regardless of what
is loaded. Reading it as "the loaded model" makes `localm run <model>` attach to
a running server and chat with (and force-load) the wrong model.
"""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from localm.inference import http_server as hs
from localm.inference.http_engine import remote_model_status
from tests.conftest import probe_double


class FakeEngine:
    def __init__(self, display_name):
        self.display_name = display_name
        self._loaded = False
        self.active_requests = 0
        self.supports_images = False
        self.can_be_multimodal = False

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

    def chat_stream(self, messages, **kw):
        yield "hi from " + self.display_name


@pytest.fixture
def multi(monkeypatch):
    reg = {
        "aaa-first": {"path": "models/aaa-first.gguf", "source": "local"},
        "zzz-loaded": {"path": "models/zzz-loaded.gguf", "source": "local"},
    }
    monkeypatch.setattr("localm.config.load_registry", lambda: reg)
    monkeypatch.setattr("localm.model_manager.get_model_info",
                        lambda name: (f"models/{name}.gguf", "hint"))
    monkeypatch.setattr("localm.model_manager.get_model_mmproj", lambda name: None)
    # probe_double, not a bare lambda: the gate on this path (vram_capacity ->
    # vram_info) passes return_status / deadline / wait_for_inflight, and a
    # zero-arg double rejects those kwargs with a TypeError that 500s the
    # chat-completion-driven load this file tests.
    monkeypatch.setattr("localm.discover.vram_info",
                        probe_double({"free": 10 * 1024 ** 3,
                                      "total": 16 * 1024 ** 3}))
    monkeypatch.setattr(hs, "_engine_factory", lambda name: FakeEngine(name))
    for d in (hs._engines, hs._engines_lru, hs._inference_sems, hs._last_activity_per_model):
        d.clear()
    hs._active_model_name = None
    hs._default_model_name = None
    hs._engine = None
    hs._inference_sem = None


def test_v1_models_marks_the_active_model(multi):
    app = hs.create_app(None)
    client = TestClient(app)
    # Load zzz-loaded (alphabetically LAST, so data[0] would be aaa-first).
    r = client.post("/v1/chat/completions", json={
        "model": "zzz-loaded",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": False,
    })
    assert r.status_code == 200

    data = client.get("/v1/models").json()["data"]
    by_id = {m["id"]: m for m in data}
    assert by_id["zzz-loaded"]["active"] is True, "the loaded model must be marked active"
    assert by_id["aaa-first"]["active"] is False, "an idle registered model is not active"


def _resp(payload):
    m = MagicMock()
    m.status_code = 200
    m.json = lambda: payload
    return m


def test_remote_model_status_picks_loaded_not_first(monkeypatch):
    """data[0] is the alphabetically-first REGISTERED model; the loaded one is
    later in the list. remote_model_status must return the loaded id."""
    payload = {"object": "list", "data": [
        {"id": "aaa-first", "loaded": False, "active": False},
        {"id": "zzz-loaded", "loaded": True, "active": True},
    ]}
    monkeypatch.setattr("requests.get", lambda *a, **k: _resp(payload))
    state, active = remote_model_status("http://x/v1", "tok")
    assert (state, active) == ("loaded", "zzz-loaded"), \
        "must report the LOADED model, not data[0] (the alphabetically-first registered)"


def test_remote_model_status_none_loaded_is_empty(monkeypatch):
    payload = {"object": "list", "data": [
        {"id": "aaa-first", "loaded": False, "active": False},
        {"id": "bbb-second", "loaded": False, "active": False},
    ]}
    monkeypatch.setattr("requests.get", lambda *a, **k: _resp(payload))
    assert remote_model_status("http://x/v1")[0] == "empty"


def test_remote_model_status_single_loaded_still_works(monkeypatch):
    """Back-compat: the common single-model reply still resolves."""
    payload = {"object": "list", "data": [{"id": "gemma-4", "loaded": True}]}
    monkeypatch.setattr("requests.get", lambda *a, **k: _resp(payload))
    assert remote_model_status("http://x/v1") == ("loaded", "gemma-4")
