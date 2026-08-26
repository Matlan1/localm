# SPDX-License-Identifier: AGPL-3.0-or-later
"""An omitted "model" field on /v1/embeddings must be distinguishable from an
explicit "localm".

``EmbeddingRequest.model`` is ``Optional[str] = None``, and the route's
required-model gate refuses only when nothing can be resolved at all, matching
/v1/chat/completions and /v1/completions for THAT part of the contract.

What an omitted model resolves TO is not a mirror of those two routes:
embeddings from different models are not comparable the way two chat replies
are, so an unnamed request resolves to the CONFIGURED embedder first, when one
exists, and only falls through to the general active-model resolution when no
embedder is configured at all. There Engine.embed()'s own contract applies: a
resolved engine that cannot itself embed falls back to the dedicated embedder,
and raises NotImplementedError (-> 422) only when no embedding path exists at
all. The fall-through case can therefore still land on a different embedding
space when the active model is an embed-capable HF encoder, which is only
reachable with no dedicated embedder configured.

Every pre-existing embeddings test (test_embeddings_no_force_load.py,
test_embeddings_route_configured_model.py, test_server_embeddings_phase3.py)
sends an explicit "model" key. These tests build the request body without a
"model" key at all.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from localm.inference import http_server as hs
from tests.conftest import probe_double


class FakeBackend:
    def __init__(self, can_embed):
        self.can_embed = can_embed


class FakeEngine:
    """Stands in for Engine. Real Engine.embed() (engine.py) either uses the
    backend directly (can_embed=True) or falls back to the dedicated on-device
    embedder, raising NotImplementedError only when neither is available - this
    double is told which of those outcomes to produce via embed_result /
    embed_error rather than re-implementing that fallback chain itself."""

    def __init__(self, name, can_embed=True, embed_result=None, embed_error=None):
        self.display_name = name
        self._loaded = False
        self._backend = FakeBackend(can_embed)
        self.active_requests = 0
        self.load_calls = 0
        self.embed_calls = 0
        self._embed_result = embed_result if embed_result is not None else [0.1, 0.2]
        self._embed_error = embed_error

    @property
    def loaded(self):
        return self._loaded

    def load(self):
        self.load_calls += 1
        self._loaded = True

    def unload(self):
        self._loaded = False

    def set_load_cancel(self, ev):
        pass

    def embed(self, texts):
        self.embed_calls += 1
        if self._embed_error is not None:
            raise self._embed_error
        return [self._embed_result for _ in texts]

    def count_tokens(self, text):
        return len(text.split())


@pytest.fixture
def server(monkeypatch):
    """A server started with a chat model already loaded and active, that CAN
    embed directly (no dedicated embedder configured) - so an unnamed request
    must reach the general get_engine() fallback and use this engine's own
    .embed(), the same shape /v1/chat/completions' fallback reaches."""
    engines: dict[str, FakeEngine] = {}

    def factory(name):
        return engines.setdefault(
            name, FakeEngine(name, can_embed=True, embed_result=[1.0, 2.0, 3.0]))

    monkeypatch.setattr("localm.config.load_registry",
                        lambda: {"model-a": {"path": "Z:/models/model-a.gguf",
                                             "source": "local"}})
    monkeypatch.setattr("localm.config.load_config", lambda: {})
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


def test_omitted_model_field_resolves_to_the_active_engine_and_reports_it(server):
    """The natural request shape: no "model" key in the body at all."""
    client, engines = server

    r = client.post("/v1/embeddings", json={"input": "hello world"})

    assert r.status_code == 200, r.text
    assert r.json()["model"] == "model-a", (
        f'omitted-field embeddings request reported model {r.json()["model"]!r}, '
        f'expected the real serving model, not the "localm" sentinel')
    assert r.json()["data"][0]["embedding"] == [1.0, 2.0, 3.0]


def test_explicit_localm_still_echoes_localm_on_embeddings(server):
    """The documented behaviour that must NOT change (matches the
    chat/completions contract): a client that actually sends "localm" gets
    "localm" back, not the resolved engine's real name."""
    client, engines = server

    r = client.post("/v1/embeddings", json={"model": "localm", "input": "hello"})

    assert r.status_code == 200, r.text
    assert r.json()["model"] == "localm", (
        f'explicit "localm" embeddings request reported model {r.json()["model"]!r}')


def test_omitted_model_still_400_when_nothing_can_be_resolved(monkeypatch):
    """With no model loaded and none ever configured, an unnamed embeddings
    request is refused with a 400, whether "unnamed" means the field was left
    out or sent as ""."""
    monkeypatch.setattr("localm.config.load_registry", lambda: {})
    hs._engines.clear()
    hs._engines_lru.clear()
    hs._active_model_name = None
    hs._default_model_name = None
    hs._engine = None

    client = TestClient(hs.create_app(None))
    r = client.post("/v1/embeddings", json={"input": "hi"})

    assert r.status_code == 400
    assert "Model parameter is required" in r.text


def test_omitted_model_honest_422_when_resolved_engine_cannot_embed(monkeypatch):
    """When the resolved (active) engine cannot embed and there is no dedicated
    embedder configured, Engine.embed() raises NotImplementedError and the route
    turns that into a 422, rather than returning vectors from a chat decoder
    that cannot embed."""
    monkeypatch.setattr("localm.config.load_registry",
                        lambda: {"model-b": {"path": "Z:/models/model-b.gguf",
                                             "source": "local"}})
    monkeypatch.setattr("localm.config.load_config", lambda: {})
    monkeypatch.setattr("localm.model_manager.get_model_info",
                        lambda name: (f"Z:/models/{name}.gguf", "hint"))
    monkeypatch.setattr("localm.model_manager.get_model_mmproj", lambda name: None)
    monkeypatch.setattr("localm.discover.vram_info",
                        probe_double({"free": 10 * 1024 ** 3,
                                      "total": 16 * 1024 ** 3}))

    engine = FakeEngine(
        "model-b", can_embed=False,
        embed_error=NotImplementedError(
            "No embedding model available. Run 'localm setup-embeddings'."))
    engines = {"model-b": engine}
    monkeypatch.setattr(hs, "_engine_factory", lambda n: engines[n])

    hs._engines.clear()
    hs._engines_lru.clear()
    hs._inference_sems.clear()
    hs._last_activity_per_model.clear()
    hs._active_model_name = None
    hs._default_model_name = None
    hs._engine = None
    hs._inference_sem = None

    engine.load()
    app = hs.create_app(engine)
    client = TestClient(app)

    r = client.post("/v1/embeddings", json={"input": "hi"})

    assert r.status_code == 422, r.text
    assert "setup-embeddings" in r.text


def test_omitted_model_prefers_the_configured_embedder_over_an_active_chat_model(monkeypatch):
    """An unnamed request resolves to the CONFIGURED embedder deterministically,
    even with a DIFFERENT, embed-CAPABLE chat model active, so two
    otherwise-identical omitted-model requests cannot return vectors from
    different embedding spaces depending on unrelated chat activity.
    """
    registry = {
        "embedding-bge-small-en-v1.5": {
            "path": "models/embeddings/bge-small-en-v1.5-q4_k_m.gguf",
            "source": "setup-embeddings", "model_type": "embedding",
        },
        "some-chat-model": {"path": "models/some-chat-model.gguf", "source": "local"},
    }
    monkeypatch.setattr("localm.config.load_registry", lambda: registry)
    monkeypatch.setattr("localm.config.load_config",
                        lambda: {"embedding_model": "bge-small-en-v1.5"})
    monkeypatch.setattr("localm.inference.embedder.embed_texts",
                        lambda texts: [[9.9, 8.8] for _ in texts])

    # can_embed=True and a DISTINCT result: this engine could itself serve the
    # request (unlike a GGUF double, which cannot), so a wrong result here means a
    # wrong CODE PATH rather than a wrong number.
    active_engine = FakeEngine("some-chat-model", can_embed=True,
                               embed_result=[1.1, 2.2])
    monkeypatch.setattr(hs, "_engine_factory", lambda n: active_engine)

    hs._engines.clear()
    hs._engines_lru.clear()
    hs._inference_sems.clear()
    hs._last_activity_per_model.clear()
    hs._engine = None
    hs._inference_sem = None

    # create_app(engine) is what sets _active_model_name (to engine.display_name);
    # create_app(None) unconditionally clears it, so setting the global by hand
    # beforehand would be overwritten.
    active_engine.load()
    app = hs.create_app(active_engine)
    client = TestClient(app)

    r = client.post("/v1/embeddings", json={"input": "hello"})

    assert r.status_code == 200, r.text
    assert r.json()["data"][0]["embedding"] == [9.9, 8.8], (
        "omitted-model request did not use the configured embedder")
    assert r.json()["model"] == "bge-small-en-v1.5"
    assert active_engine.embed_calls == 0, (
        "the active chat model's own .embed() was called - the omitted "
        "request fell through to the general resolution instead of "
        "preferring the configured embedder")
