# SPDX-License-Identifier: AGPL-3.0-or-later
"""A reranker is never accepted as the RAG embedding model: ``POST
/api/rag/embedding`` (dry run and confirmed) and ``PATCH /v1/config`` answer 400
for a registered reranker while an embedding model is still accepted, and the
GUI's ``/api/models`` marks the reranker row so the Knowledge picker can skip it."""
from __future__ import annotations

import json
import struct

import pytest
from fastapi.testclient import TestClient

OWNER_KEY = "owner-key-reranker-not-an-embedder"


def _owner():
    return {"Authorization": f"Bearer {OWNER_KEY}"}


def _gguf(path, kvs):
    def kv(key, vtype, value):
        k = key.encode()
        out = struct.pack("<Q", len(k)) + k + struct.pack("<I", vtype)
        if vtype == 8:
            v = value.encode()
            return out + struct.pack("<Q", len(v)) + v
        return out + struct.pack("<I", value)

    body = b"".join(kv(*x) for x in kvs)
    path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, len(kvs)) + body)
    return path


@pytest.fixture
def home(tmp_path, monkeypatch):
    import localm.config as cfg
    home = tmp_path / ".localm"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.setattr(cfg, "HOME_DIR", home)
    monkeypatch.setattr(cfg, "MODELS_DIR", home / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", home / "registry.json")
    monkeypatch.setenv("LOCALM_API_KEY", OWNER_KEY)
    rr = _gguf(tmp_path / "rerank.gguf",
               [("general.architecture", 8, "qwen3"), ("qwen3.pooling_type", 4, 4)])
    emb = _gguf(tmp_path / "embed.gguf",
                [("general.architecture", 8, "bert"), ("bert.pooling_type", 4, 2)])
    (home / "registry.json").write_text(json.dumps({
        "rerank-q3": {"path": str(rr), "source": "local", "model_type": "embedding",
                      "reranker": True},
        "embed-bge": {"path": str(emb), "source": "local", "model_type": "embedding",
                      "reranker": False},
    }), encoding="utf-8")
    return home


@pytest.fixture
def client(home, tmp_path):
    from localm.inference.http_server import create_app
    from localm.plugins.engine import PluginManager
    app = create_app(None)
    PluginManager(app, external_root=tmp_path / "noplugins").install("rag")
    with TestClient(app) as c:
        yield c


def _embedding_model(home):
    cfg = home / "config.json"
    if not cfg.exists():
        return None
    return json.loads(cfg.read_text(encoding="utf-8")).get("embedding_model")


class TestRagEmbeddingRouteRefusesAReranker:
    @pytest.mark.parametrize("confirm", [False, True])
    def test_a_registered_reranker_is_refused(self, client, home, confirm):
        before = _embedding_model(home)
        r = client.post("/api/rag/embedding", headers=_owner(),
                        json={"model": "rerank-q3", "confirm": confirm})
        assert r.status_code == 400, r.text
        assert "is a reranker" in r.json()["detail"]
        assert _embedding_model(home) == before

    def test_a_reranker_gguf_path_is_refused(self, client, home, tmp_path):
        r = client.post("/api/rag/embedding", headers=_owner(),
                        json={"model": str(tmp_path / "rerank.gguf")})
        assert r.status_code == 400, r.text
        assert "reranker GGUF" in r.json()["detail"]

    def test_an_embedding_model_is_accepted(self, client, home):
        r = client.post("/api/rag/embedding", headers=_owner(),
                        json={"model": "embed-bge"})
        assert r.status_code == 200, r.text
        assert r.json()["needs_confirm"] is True
        assert r.json()["model"] == "embed-bge"


class TestPatchConfigRefusesAReranker:
    @pytest.mark.parametrize("confirm", [False, True])
    def test_a_registered_reranker_is_refused_and_not_written(self, client, home, confirm):
        before = _embedding_model(home)
        r = client.patch("/v1/config", headers=_owner(),
                         json={"embedding_model": "rerank-q3", "confirm": confirm})
        assert r.status_code == 400, r.text
        assert "is a reranker" in r.json()["detail"]
        assert _embedding_model(home) == before

    def test_an_embedding_model_is_written(self, client, home):
        r = client.patch("/v1/config", headers=_owner(),
                         json={"embedding_model": "embed-bge", "confirm": True})
        assert r.status_code == 200, r.text
        assert _embedding_model(home) == "embed-bge"


class TestModelsInventoryMarksRerankers:
    def test_only_the_reranker_row_carries_the_flag(self, home):
        from fastapi import FastAPI

        from localm.plugins.gui.web import attach_gui
        app = FastAPI()
        attach_gui(app, self_url="http://127.0.0.1:9/v1",
                   switch_model=lambda name: None, active_model=lambda: None)
        with TestClient(app) as c:
            r = c.get("/api/models", headers=_owner())
        assert r.status_code == 200, r.text
        rows = {m["name"]: m for m in r.json()["models"]}
        assert rows["rerank-q3"]["reranker"] is True
        assert rows["rerank-q3"]["model_type"] == "embedding"
        assert "reranker" not in rows["embed-bge"]
