# SPDX-License-Identifier: AGPL-3.0-or-later
"""/v1/embeddings for a model registered as a Hugging Face FOLDER of type
``embedding`` (a sentence-transformers folder) is answered by that model's own
engine, never by the dedicated GGUF embedder labelled with the folder's name."""

import types

import pytest
from fastapi.testclient import TestClient

from localm.inference import http_server as hs


@pytest.fixture
def app_client(monkeypatch):
    for d in (hs._engines, hs._engines_lru, hs._inference_sems, hs._last_activity_per_model):
        d.clear()
    hs._active_model_name = None
    hs._engine = None
    hs._inference_sem = None
    return TestClient(hs.create_app(None))


def test_a_registered_folder_embedder_is_served_by_its_own_engine(app_client, monkeypatch, tmp_path):
    folder = tmp_path / "bge-st"
    folder.mkdir()
    registry = {"bge-st": {"path": str(folder), "source": "local", "model_type": "embedding",
                           "reranker": False}}
    monkeypatch.setattr("localm.config.load_registry", lambda: registry)
    monkeypatch.setattr("localm.config.load_config",
                        lambda: {"embedding_model": "bge-small-en-v1.5"})

    def dedicated_embedder_must_not_run(texts):
        raise AssertionError("the dedicated GGUF embedder answered for a folder model")

    monkeypatch.setattr("localm.inference.embedder.embed_texts", dedicated_embedder_must_not_run)
    engine = types.SimpleNamespace(
        display_name="bge-st", _backend=types.SimpleNamespace(can_embed=True),
        embed=lambda texts: [[1.0, 2.0] for _ in texts], count_tokens=lambda text: 3)

    async def fake_get_engine(name, load=True):
        assert name == "bge-st"
        return engine

    monkeypatch.setattr(hs, "get_engine", fake_get_engine)
    r = app_client.post("/v1/embeddings", json={"model": "bge-st", "input": "hello"})
    assert r.status_code == 200, r.text
    assert r.json()["data"][0]["embedding"] == [1.0, 2.0]
    assert r.json()["model"] == "bge-st"


def test_a_registered_gguf_embedder_still_uses_the_dedicated_embedder(app_client, monkeypatch, tmp_path):
    gguf = tmp_path / "e.gguf"
    gguf.write_bytes(b"x")
    registry = {"emb": {"path": str(gguf), "source": "local", "model_type": "embedding"}}
    monkeypatch.setattr("localm.config.load_registry", lambda: registry)
    monkeypatch.setattr("localm.config.load_config", lambda: {"embedding_model": "other"})
    monkeypatch.setattr("localm.inference.embedder.embed_texts",
                        lambda texts: [[0.5, 0.25] for _ in texts])
    r = app_client.post("/v1/embeddings", json={"model": "emb", "input": "hello"})
    assert r.status_code == 200, r.text
    assert r.json()["data"][0]["embedding"] == [0.5, 0.25]
