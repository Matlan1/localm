# SPDX-License-Identifier: AGPL-3.0-or-later
"""Per-request size cap on /v1/embeddings for HF-backed models.

HFBackend.embed() loops over texts one at a time with no batching (or, for a
sentence-transformer model, batches with no truncation at all), against a model
that is always loaded full bf16/fp32, so an oversized batch has no bound of its
own beyond the generous hf_embed_timeout_s. These tests exercise the cap
directly against HFBackend (no real model or subprocess: a real HFBackend with
_loaded=True and a fake _runner), plus the route-level mapping to a 413.

Not applicable to GGUF (can_embed is a fixed False there) or the dedicated
on-device embedder (embedder.py's IsolatedEmbedder, a separate, purpose-built
path).
"""

import logging

import pytest
from fastapi.testclient import TestClient

from localm.inference import http_server as hs
from localm.inference.backends._hf_runner import (
    EMBED_MAX_CHARS_DEFAULT,
    EMBED_MAX_TEXTS_DEFAULT,
)
from localm.inference.backends.base import EmbedBatchTooLargeError
from localm.inference.backends.hf import HFBackend
from tests.conftest import probe_double


class _FakeRunner:
    """Records what it was asked to embed; never touches a real model/process."""

    def __init__(self):
        self.calls = []

    def embed(self, texts, timeout=None):
        self.calls.append(list(texts))
        return [[0.1, 0.2, 0.3] for _ in texts]


class _PoisonRunner:
    """Fails the test if a request reaches the runner - i.e. crosses into the
    isolated worker - proving the cap check runs first, in the parent."""

    def embed(self, texts, timeout=None):
        raise AssertionError(
            "HFBackend.embed() reached the runner - the size cap should have "
            "rejected this request before it crossed the process boundary")


def _loaded_backend(runner):
    backend = HFBackend("does-not-need-to-exist")
    backend._loaded = True
    backend._runner = runner
    return backend


class TestTextCountCap:
    def test_within_default_cap_succeeds_and_reaches_runner(self):
        runner = _FakeRunner()
        backend = _loaded_backend(runner)
        texts = ["hello"] * 10
        vecs = backend.embed(texts)
        assert len(vecs) == 10
        assert runner.calls == [texts]

    def test_exact_boundary_at_default_max_texts_succeeds(self):
        runner = _FakeRunner()
        backend = _loaded_backend(runner)
        texts = ["x"] * EMBED_MAX_TEXTS_DEFAULT
        vecs = backend.embed(texts)
        assert len(vecs) == EMBED_MAX_TEXTS_DEFAULT
        assert runner.calls, "a batch at the exact limit should reach the runner"

    def test_one_over_default_max_texts_is_rejected_before_the_runner(self):
        backend = _loaded_backend(_PoisonRunner())
        texts = ["x"] * (EMBED_MAX_TEXTS_DEFAULT + 1)
        with pytest.raises(EmbedBatchTooLargeError) as ei:
            backend.embed(texts)
        msg = str(ei.value)
        assert str(EMBED_MAX_TEXTS_DEFAULT + 1) in msg
        assert str(EMBED_MAX_TEXTS_DEFAULT) in msg


class TestCharCountCap:
    def test_within_default_char_cap_succeeds(self):
        runner = _FakeRunner()
        backend = _loaded_backend(runner)
        vecs = backend.embed(["a" * 1000])
        assert len(vecs) == 1
        assert runner.calls

    def test_exact_boundary_at_default_max_chars_succeeds(self):
        runner = _FakeRunner()
        backend = _loaded_backend(runner)
        vecs = backend.embed(["a" * EMBED_MAX_CHARS_DEFAULT])
        assert len(vecs) == 1
        assert runner.calls, "a batch at the exact limit should reach the runner"

    def test_one_char_over_default_max_chars_is_rejected_before_the_runner(self):
        backend = _loaded_backend(_PoisonRunner())
        with pytest.raises(EmbedBatchTooLargeError) as ei:
            backend.embed(["a" * (EMBED_MAX_CHARS_DEFAULT + 1)])
        msg = str(ei.value)
        assert str(EMBED_MAX_CHARS_DEFAULT + 1) in msg
        assert str(EMBED_MAX_CHARS_DEFAULT) in msg

    def test_aggregate_across_many_small_texts_is_rejected(self):
        # No single text is near the cap, but the SUM is: the check is on TOTAL
        # characters across the batch, not per-text length. per_text is chosen
        # so the char total (212_500) exceeds EMBED_MAX_CHARS_DEFAULT (200_000)
        # while count (250) stays under EMBED_MAX_TEXTS_DEFAULT (256).
        backend = _loaded_backend(_PoisonRunner())
        count = 250
        per_text = 850
        assert count <= EMBED_MAX_TEXTS_DEFAULT, "test setup: stay under the text cap"
        assert count * per_text > EMBED_MAX_CHARS_DEFAULT, "test setup: must exceed the char cap"
        texts = ["a" * per_text] * count
        with pytest.raises(EmbedBatchTooLargeError):
            backend.embed(texts)


class TestConfigOverride:
    def test_smaller_configured_max_texts_is_honored(self, monkeypatch):
        monkeypatch.setattr("localm.config.load_config",
                            lambda: {"hf_embed_max_texts": 3})
        backend = _loaded_backend(_PoisonRunner())
        with pytest.raises(EmbedBatchTooLargeError) as ei:
            backend.embed(["a", "b", "c", "d"])
        assert "4" in str(ei.value) and "3" in str(ei.value)

    def test_smaller_configured_max_chars_is_honored(self, monkeypatch):
        monkeypatch.setattr("localm.config.load_config",
                            lambda: {"hf_embed_max_chars": 5})
        backend = _loaded_backend(_PoisonRunner())
        with pytest.raises(EmbedBatchTooLargeError):
            backend.embed(["123456"])

    def test_larger_configured_max_texts_allows_a_bigger_batch(self, monkeypatch):
        monkeypatch.setattr(
            "localm.config.load_config",
            lambda: {"hf_embed_max_texts": EMBED_MAX_TEXTS_DEFAULT + 50})
        runner = _FakeRunner()
        backend = _loaded_backend(runner)
        texts = ["x"] * (EMBED_MAX_TEXTS_DEFAULT + 10)
        vecs = backend.embed(texts)
        assert len(vecs) == len(texts)

    def test_malformed_max_texts_config_falls_back_to_default(self, monkeypatch, caplog):
        monkeypatch.setattr(
            "localm.config.load_config", lambda: {"hf_embed_max_texts": "not-a-number"})
        with caplog.at_level(logging.WARNING, logger="localm"):
            assert HFBackend._embed_max_texts() == EMBED_MAX_TEXTS_DEFAULT
        assert "hf_embed_max_texts" in caplog.text

    def test_malformed_max_chars_config_falls_back_to_default(self, monkeypatch, caplog):
        monkeypatch.setattr(
            "localm.config.load_config", lambda: {"hf_embed_max_chars": "not-a-number"})
        with caplog.at_level(logging.WARNING, logger="localm"):
            assert HFBackend._embed_max_chars() == EMBED_MAX_CHARS_DEFAULT
        assert "hf_embed_max_chars" in caplog.text


# --------------------------------------------------------------------------- #
# Route level: does /v1/embeddings turn the backend's rejection into a 413?
# --------------------------------------------------------------------------- #

class FakeBackend:
    can_embed = True   # HF-encoder-style: CAN embed, unlike GGUF's fixed False


class FakeEngine:
    def __init__(self, name):
        self.display_name = name
        self._loaded = True
        self._backend = FakeBackend()
        self.load_calls = 0
        self.active_requests = 0

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
        # Simulates what a real HFBackend.embed() does for an oversized batch;
        # this test covers the ROUTE's exception-to-HTTP mapping.
        raise EmbedBatchTooLargeError(
            f"Too many texts in one /v1/embeddings request: {len(texts)} "
            f"(max {EMBED_MAX_TEXTS_DEFAULT}). Split the batch across multiple "
            f"requests.")

    def count_tokens(self, text):
        return len(text.split())


@pytest.fixture
def setup(monkeypatch):
    reg = {"model-hf": {"path": "models/model-hf", "source": "local"}}
    monkeypatch.setattr("localm.config.load_registry", lambda: reg)
    monkeypatch.setattr("localm.config.load_config", lambda: {})
    monkeypatch.setattr("localm.model_manager.get_model_info",
                        lambda n: (f"models/{n}", "h"))
    monkeypatch.setattr("localm.model_manager.get_model_mmproj", lambda n: None)
    monkeypatch.setattr("localm.discover.vram_info",
                        probe_double({"free": 10 * 1024 ** 3, "total": 16 * 1024 ** 3}))
    engines = {"model-hf": FakeEngine("model-hf")}
    monkeypatch.setattr(hs, "_engine_factory", lambda n: engines[n])
    for d in (hs._engines, hs._engines_lru, hs._inference_sems, hs._last_activity_per_model):
        d.clear()
    hs._active_model_name = None
    hs._default_model_name = None
    hs._engine = None
    hs._inference_sem = None
    return engines


def test_embeddings_route_returns_413_when_backend_rejects_an_oversized_batch(setup):
    app = hs.create_app(None)
    client = TestClient(app)

    r = client.post("/v1/embeddings", json={"model": "model-hf", "input": ["x"] * 5})

    assert r.status_code == 413, r.text
    assert "Too many texts" in r.json()["detail"]
