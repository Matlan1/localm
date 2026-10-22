# SPDX-License-Identifier: AGPL-3.0-or-later
"""/v1/embeddings for a model registered as a Hugging Face FOLDER of type
``embedding`` (a sentence-transformers folder) is answered by that model's own
engine, never by the dedicated GGUF embedder labelled with the folder's name. The
one exception is a folder named like the pinned embedding model, where the pinned
model answers."""

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


def test_a_registered_gguf_embedder_is_embedded_with_its_own_file(app_client, monkeypatch, tmp_path):
    gguf = tmp_path / "e.gguf"
    gguf.write_bytes(b"x")
    registry = {"emb": {"path": str(gguf), "source": "local", "model_type": "embedding"}}
    monkeypatch.setattr("localm.config.load_registry", lambda: registry)
    monkeypatch.setattr("localm.config.load_config", lambda: {"embedding_model": "other"})

    async def folder_engine_must_not_run(name, load=True):
        raise AssertionError("a GGUF embedder was served as a folder model")

    monkeypatch.setattr(hs, "get_engine", folder_engine_must_not_run)
    configured = []
    monkeypatch.setattr("localm.inference.embedder.embed_texts",
                        lambda texts: configured.append(texts) or [[9.0] for _ in texts])
    used = []
    monkeypatch.setattr("localm.inference.reranker.embed",
                        lambda path, texts: used.append(path) or [[0.5, 0.25] for _ in texts])
    r = app_client.post("/v1/embeddings", json={"model": "emb", "input": "hello"})
    assert r.status_code == 200, r.text
    assert r.json()["data"][0]["embedding"] == [0.5, 0.25]
    assert used == [str(gguf.resolve())] and configured == []


def test_a_folder_named_like_the_pinned_embedder_does_not_shadow_it(app_client, monkeypatch, tmp_path):
    folder = tmp_path / "bge-small-en-v1.5"
    folder.mkdir()
    registry = {"bge-small-en-v1.5": {"path": str(folder), "source": "local",
                                      "model_type": "embedding", "reranker": False}}
    monkeypatch.setattr("localm.config.load_registry", lambda: registry)
    monkeypatch.setattr("localm.config.load_config", lambda: {})

    async def folder_engine_must_not_run(name, load=True):
        raise AssertionError("the registered folder answered for the pinned embedder's name")

    monkeypatch.setattr(hs, "get_engine", folder_engine_must_not_run)
    monkeypatch.setattr("localm.inference.embedder.embed_texts",
                        lambda texts: [[0.5, 0.25] for _ in texts])
    r = app_client.post("/v1/embeddings", json={"model": "bge-small-en-v1.5", "input": "hello"})
    assert r.status_code == 200, r.text
    assert r.json()["data"][0]["embedding"] == [0.5, 0.25]


def test_a_folder_named_differently_from_the_pinned_embedder_is_still_served_by_its_engine(
        app_client, monkeypatch, tmp_path):
    folder = tmp_path / "bge-st"
    folder.mkdir()
    registry = {"bge-st": {"path": str(folder), "source": "local", "model_type": "embedding",
                           "reranker": False}}
    monkeypatch.setattr("localm.config.load_registry", lambda: registry)
    monkeypatch.setattr("localm.config.load_config", lambda: {})
    monkeypatch.setattr("localm.inference.embedder.embed_texts",
                        lambda texts: pytest.fail("the dedicated embedder answered for a folder"))
    engine = types.SimpleNamespace(
        display_name="bge-st", _backend=types.SimpleNamespace(can_embed=True),
        embed=lambda texts: [[1.0, 2.0] for _ in texts], count_tokens=lambda text: 3)

    async def fake_get_engine(name, load=True):
        return engine

    monkeypatch.setattr(hs, "get_engine", fake_get_engine)
    r = app_client.post("/v1/embeddings", json={"model": "bge-st", "input": "hello"})
    assert r.status_code == 200, r.text
    assert r.json()["data"][0]["embedding"] == [1.0, 2.0]


@pytest.mark.parametrize("raw", ["", None])
def test_an_embedder_entry_with_no_path_is_not_read_as_the_current_folder(
        app_client, monkeypatch, raw):
    registry = {"emb": {"path": raw, "source": "local", "model_type": "embedding"}}
    monkeypatch.setattr("localm.config.load_registry", lambda: registry)
    monkeypatch.setattr("localm.config.load_config", lambda: {"embedding_model": "other"})

    async def folder_engine_must_not_run(name, load=True):
        raise AssertionError("an entry with no path was served as a folder model")

    monkeypatch.setattr(hs, "get_engine", folder_engine_must_not_run)
    configured = []
    monkeypatch.setattr("localm.inference.embedder.embed_texts",
                        lambda texts: configured.append(texts) or [[0.5, 0.25] for _ in texts])
    r = app_client.post("/v1/embeddings", json={"model": "emb", "input": "hello"})
    assert r.status_code == 422, r.text
    assert "'emb'" in r.json()["detail"] and "no file path" in r.json()["detail"]
    assert configured == []


def test_the_folder_check_runs_off_the_event_loop(app_client, monkeypatch, tmp_path):
    import asyncio
    import pathlib
    folder = tmp_path / "bge-st"
    folder.mkdir()
    registry = {"bge-st": {"path": str(folder), "source": "local", "model_type": "embedding",
                           "reranker": False}}
    monkeypatch.setattr("localm.config.load_registry", lambda: registry)
    monkeypatch.setattr("localm.config.load_config", lambda: {})
    on_loop = []
    real_is_dir = pathlib.Path.is_dir

    def _is_dir(self):
        if self == folder:
            try:
                asyncio.get_running_loop()
                on_loop.append(True)
            except RuntimeError:
                on_loop.append(False)
        return real_is_dir(self)

    monkeypatch.setattr(pathlib.Path, "is_dir", _is_dir)
    engine = types.SimpleNamespace(
        display_name="bge-st", _backend=types.SimpleNamespace(can_embed=True),
        embed=lambda texts: [[1.0, 2.0] for _ in texts], count_tokens=lambda text: 3)

    async def fake_get_engine(name, load=True):
        return engine

    monkeypatch.setattr(hs, "get_engine", fake_get_engine)
    r = app_client.post("/v1/embeddings", json={"model": "bge-st", "input": "hello"})
    assert r.status_code == 200, r.text
    assert on_loop == [False]
