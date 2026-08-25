# SPDX-License-Identifier: AGPL-3.0-or-later
"""An omitted "model" field on /v1/embeddings must be distinguishable from an
explicit "localm" - the sibling fix to test_chat_model_field_omitted.py
(checkup item 17), deliberately split out because this route also has its own
unconditional required-model gate that the sentinel default made unreachable.

EmbeddingRequest.model used to default to the truthy string "localm", so an
omitted field was indistinguishable from an explicit "localm" request AND
always passed the route's ``if not req.model: raise 400`` check. Fixing only
the type (Optional[str] = None) without also relaxing that check would have
turned every previously-served omitted-field request into a hard 400 - a
behaviour change made silently instead of decided on purpose.

DECISION (stated once here, not duplicated at every call site - see the
matching comments in routes/chat.py's embeddings handler): an omitted model
refuses only when nothing can be resolved at all, matching /v1/chat/completions
and /v1/completions for THAT part of the contract. But WHAT it resolves to is
deliberately NOT a blind mirror of those two routes, on a distinction raised
in review: embeddings from different models are not comparable the way two
chat replies are, so "no preference" must not silently drift onto whatever
chat model happens to be active - that is shared, mutable state unrelated
chat traffic can change between two otherwise-identical requests, and unlike
a rejection, a wrong-model embedding returns 200 with no signal anything is
wrong (the exact NEW-RAG-DIM-NO-REEMBED hazard, reached a new way). So an
omitted model resolves to the CONFIGURED embedder first, when one exists -
deterministic, independent of chat activity - and only falls through to the
general active-model resolution (the same chain chat/completions use) when no
embedder is configured at all, where Engine.embed()'s own contract still
degrades honestly: a resolved engine that cannot itself embed transparently
falls back to the dedicated embedder, raising NotImplementedError (-> 422)
only when no embedding path exists at all. So the remaining "whatever's
active" case can still land on a genuinely different embedding space if the
active model happens to be an embed-capable HF encoder rather than a GGUF or
HF chat decoder (both of which always degrade to the dedicated embedder
regardless of which one is active) - a narrower, pre-existing-in-kind risk
(explicitly naming an arbitrary chat model has always had it) rather than a
new one, and only reachable when the user has not configured a dedicated
embedder at all.

FIXTURE PREMISE (diff-review-discipline.md item 19). Every pre-existing
embeddings test (test_embeddings_no_force_load.py,
test_embeddings_route_configured_model.py, test_server_embeddings_phase3.py)
sends an explicit "model" key - not one omits it - so that suite is
structurally incapable of failing on this defect: the value that distinguishes
"omitted" from "explicitly emptied" was never in the test data. These tests
build the request body without a "model" key at all.
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
    """The gate decision's negative half, identical contract to
    test_chat_model_field_omitted.py's equivalent test: with no model loaded
    and none ever configured, an unnamed embeddings request is genuinely
    unserveable regardless of whether "unnamed" means the field was left out
    or sent as ""."""
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
    """The gate decision's other guardrail: relaxing the gate must not turn
    into a silently-wrong 200. When the resolved (active) engine cannot embed
    and there is no dedicated embedder configured, Engine.embed() itself
    raises NotImplementedError - the route already turns that into a clean 422
    (the except NotImplementedError arm) - so an omitted model must still fail
    HONESTLY, never fabricate vectors from a chat decoder that cannot embed."""
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
    """The load-bearing guarantee raised in review: an omitted model must not
    silently drift onto whatever chat model happens to be active. Unlike chat,
    embeddings from different models are not comparable, so this pins that an
    unnamed request resolves to the CONFIGURED embedder deterministically -
    even with a DIFFERENT, embed-CAPABLE chat model active (the exact unstable
    shape flagged: two otherwise-identical omitted-model requests must not be
    able to return vectors from different embedding spaces depending on
    unrelated chat activity). Mirrors
    test_default_embedding_model_routes_even_with_a_chat_model_active in
    test_embeddings_route_configured_model.py, but with the model field
    OMITTED rather than sent as the explicit configured name - that existing
    test could not have failed on THIS defect, since it never omits the field.
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
