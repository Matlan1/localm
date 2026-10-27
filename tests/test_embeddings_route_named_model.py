# SPDX-License-Identifier: AGPL-3.0-or-later
"""/v1/embeddings, /api/embed and /api/embeddings for a registered embedding GGUF
that is not the configured ``embedding_model``: the request is answered by THAT
model, the configured embedder keeps serving its own name, and an entry whose
file is missing or whose path is empty is refused with a 4xx naming it.

Registry entries are real synthetic GGUF headers registered through the model
manager. The native worker is replaced at ``embedder.IsolatedEmbedder``.
"""

import asyncio
import json
import os
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from localm import model_manager as mm
from localm.inference import embedder as emb
from localm.inference import reranker as rr
from localm.model_manager import registry as R
from tests._gguf_builder import embedding_bytes

CONFIGURED_DIM = 384


class FakeEmbedWorker:
    instances = []
    dims = {}

    def __init__(self, path, **kw):
        self.model_path = path
        self.kw = kw
        self.dim = FakeEmbedWorker.dims.get(os.path.basename(path), 768)
        self.cls_labels = []
        self.effective_pooling = kw.get("_report_pooling", 1)
        self.active_requests = 0
        self.closed = False
        self.during_embed = None
        self._runner = SimpleNamespace(grace=[])
        self._runner.shutdown = lambda grace=5.0: self._runner.grace.append(grace)
        FakeEmbedWorker.instances.append(self)

    def embed(self, texts):
        if self.during_embed is not None:
            self.during_embed(self)
        return [[float(i + 1)] * self.dim for i, _t in enumerate(texts)]

    def close(self):
        self.closed = True


@pytest.fixture
def home(tmp_path, monkeypatch):
    import localm.config as cfg
    home = tmp_path / ".localm"
    (home / "models").mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.setattr(cfg, "HOME_DIR", home)
    monkeypatch.setattr(cfg, "MODELS_DIR", home / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", home / "registry.json")
    monkeypatch.setattr(mm, "MODELS_DIR", home / "models")
    monkeypatch.setattr(mm, "REGISTRY_FILE", home / "registry.json")
    (home / "config.json").write_text(json.dumps({"embedding_model": "bge-small-en-v1.5"}))
    rr.reset_reranker()
    emb.reset_embedder()
    yield home
    rr.reset_reranker()
    emb.reset_embedder()


@pytest.fixture
def worker(monkeypatch, home):
    FakeEmbedWorker.instances = []
    FakeEmbedWorker.dims = {"mini.gguf": 256}
    swaps = []
    from localm.inference.engine import _LOAD_LOCK

    def swap(path, ngl):
        swaps.append((path, ngl))
        held = [name for name, lock in (("engine _LOAD_LOCK", _LOAD_LOCK._lock),
                                        ("reranker _LOCK", rr._LOCK),
                                        ("embedder _LOCK", emb._LOCK))
                if lock._is_owned()]
        FakeEmbedWorker.held_at_swap.append(held)
    FakeEmbedWorker.held_at_swap = []
    monkeypatch.setattr(emb, "IsolatedEmbedder", FakeEmbedWorker)
    monkeypatch.setattr(emb, "_maybe_swap_for_embedder", swap)
    monkeypatch.setattr(emb, "_choose_embedder_gpu_layers", lambda path, cfg: (99, None))
    FakeEmbedWorker.swaps = swaps
    return FakeEmbedWorker


@pytest.fixture
def configured(monkeypatch):
    calls = []

    def fake_embed_texts(texts):
        calls.append(list(texts))
        return [[0.5] * CONFIGURED_DIM for _ in texts]
    monkeypatch.setattr(emb, "embed_texts", fake_embed_texts)
    return calls


def register(tmp_path, name, filename=None):
    path = tmp_path / (filename or f"{name}.gguf")
    path.write_bytes(embedding_bytes("nomic-bert"))
    R._register(name, path, model_type="embedding")
    return path


def raw_entry(home, name, path):
    reg_file = home / "registry.json"
    reg = json.loads(reg_file.read_text()) if reg_file.exists() else {}
    reg[name] = {"path": path, "source": "local", "model_type": "embedding"}
    reg_file.write_text(json.dumps(reg))


@pytest.fixture
def client(home):
    from localm.inference.http_server import create_app
    os.environ.pop("LOCALM_API_KEY", None)
    with TestClient(create_app(None)) as c:
        yield c


def embed(client, model, text="hello"):
    return client.post("/v1/embeddings", json={"model": model, "input": text})


class TestARegisteredEmbedderIsAnsweredByItself:
    def test_its_own_model_and_dimension_answer(self, client, worker, configured, tmp_path):
        path = register(tmp_path, "nomic")
        r = embed(client, "nomic")
        assert r.status_code == 200, r.text
        body = r.json()
        assert len(body["data"][0]["embedding"]) == 768
        assert body["model"] == "nomic"
        assert configured == []
        assert [w.model_path for w in worker.instances] == [str(path.resolve())]

    def test_a_list_input_keeps_its_order(self, client, worker, configured, tmp_path):
        register(tmp_path, "nomic")
        r = client.post("/v1/embeddings", json={"model": "nomic", "input": ["a", "b"]})
        assert r.status_code == 200, r.text
        assert [d["embedding"][0] for d in r.json()["data"]] == [1.0, 2.0]
        assert [d["index"] for d in r.json()["data"]] == [0, 1]

    def test_base64_encoding_is_applied(self, client, worker, configured, tmp_path):
        import base64
        import struct
        register(tmp_path, "nomic")
        r = client.post("/v1/embeddings", json={"model": "nomic", "input": "x",
                                                "encoding_format": "base64"})
        assert r.status_code == 200, r.text
        raw = base64.b64decode(r.json()["data"][0]["embedding"])
        assert len(struct.unpack("<768f", raw)) == 768

    def test_two_registered_embedders_each_answer_with_their_own(self, client, worker, configured, tmp_path):
        register(tmp_path, "nomic")
        register(tmp_path, "mini", "mini.gguf")
        a = embed(client, "nomic")
        b = embed(client, "mini")
        assert a.status_code == 200 and b.status_code == 200, (a.text, b.text)
        assert len(a.json()["data"][0]["embedding"]) == 768
        assert len(b.json()["data"][0]["embedding"]) == 256
        first, second = worker.instances
        assert first.closed and not second.closed
        assert configured == []

    def test_the_same_model_is_loaded_once(self, client, worker, configured, tmp_path):
        register(tmp_path, "nomic")
        assert embed(client, "nomic").status_code == 200
        assert embed(client, "nomic").status_code == 200
        assert len(worker.instances) == 1

    def test_the_load_runs_the_vram_swap_check_holding_no_lock(self, client, worker, configured, tmp_path):
        path = register(tmp_path, "nomic")
        assert embed(client, "nomic").status_code == 200
        assert worker.swaps == [(str(path.resolve()), 99)]
        assert worker.held_at_swap == [[]]

    def test_the_swap_check_takes_an_explicit_gpu_layer_choice(self, client, worker, configured, tmp_path, home):
        path = register(tmp_path, "nomic")
        (home / "config.json").write_text(json.dumps(
            {"embedding_model": "bge-small-en-v1.5", "embedding_gpu_layers": 0}))
        assert embed(client, "nomic").status_code == 200
        assert worker.swaps == [(str(path.resolve()), 0)]

    def test_the_placement_choice_reaches_the_worker(self, client, worker, configured, tmp_path, monkeypatch):
        register(tmp_path, "nomic")
        monkeypatch.setattr(emb, "_choose_embedder_gpu_layers", lambda path, cfg: (7, "tight VRAM"))
        assert embed(client, "nomic").status_code == 200
        kw = worker.instances[0].kw
        assert kw["n_gpu_layers"] == 7 and kw["gpu_fallback_reason"] == "tight VRAM"

    def test_the_model_pools_as_it_declares_whatever_embedding_pooling_says(self, client, worker, configured, tmp_path, home):
        register(tmp_path, "nomic")
        cfg = json.loads((home / "config.json").read_text())
        cfg["embedding_pooling"] = "cls"
        (home / "config.json").write_text(json.dumps(cfg))
        assert embed(client, "nomic").status_code == 200
        assert worker.instances[0].kw["pooling_type"] == emb.POOLING_AUTO

    def test_a_model_that_loads_as_a_reranker_is_refused(self, client, worker, configured, tmp_path, monkeypatch):
        register(tmp_path, "sneaky")

        class RankWorker(FakeEmbedWorker):
            def __init__(self, path, **kw):
                super().__init__(path, **kw)
                self.effective_pooling = emb._POOLING_RANK
        monkeypatch.setattr(emb, "IsolatedEmbedder", RankWorker)
        r = embed(client, "sneaky")
        assert r.status_code == 503, r.text
        assert "reranker" in r.json()["detail"]
        assert worker.instances[0].closed
        assert not rr.is_loaded()
        assert configured == []

    def test_a_load_failure_is_a_503_naming_the_cause(self, client, worker, configured, home, monkeypatch):
        register(home / "models", "broken")

        def boom(path, **kw):
            raise RuntimeError(f"native load failed for {path}")
        monkeypatch.setattr(emb, "IsolatedEmbedder", boom)
        r = embed(client, "broken")
        assert r.status_code == 503, r.text
        detail = r.json()["detail"]
        assert "native load failed" in detail and "broken.gguf" in detail
        assert str(home) not in detail
        assert configured == []


class TestTheConfiguredEmbedderIsUnchanged:
    def test_the_configured_name_uses_the_configured_embedder(self, client, worker, configured, tmp_path):
        register(tmp_path, "nomic")
        r = embed(client, "bge-small-en-v1.5")
        assert r.status_code == 200, r.text
        assert len(r.json()["data"][0]["embedding"]) == CONFIGURED_DIM
        assert configured == [["hello"]]
        assert worker.instances == []

    def test_an_alias_of_the_configured_file_uses_the_configured_embedder(self, client, worker, configured, home):
        dest = emb._embeddings_dir() / emb.KNOWN_EMBEDDING_MODELS["bge-small-en-v1.5"][1]
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(embedding_bytes("bert"))
        R._register("embedding-bge-small-en-v1.5", dest, model_type="embedding")
        r = embed(client, "embedding-bge-small-en-v1.5")
        assert r.status_code == 200, r.text
        assert len(r.json()["data"][0]["embedding"]) == CONFIGURED_DIM
        assert configured == [["hello"]]
        assert worker.instances == []

    def test_a_named_request_leaves_the_configured_embedder_loaded(self, client, worker, home, tmp_path, monkeypatch):
        dest = tmp_path / "configured.gguf"
        dest.write_bytes(embedding_bytes("bert"))
        (home / "config.json").write_text(json.dumps({"embedding_model": str(dest)}))
        FakeEmbedWorker.dims["configured.gguf"] = CONFIGURED_DIM
        register(tmp_path, "nomic")
        rag_vecs = emb.embed_texts(["rag text"])
        assert len(rag_vecs[0]) == CONFIGURED_DIM
        assert embed(client, "nomic").status_code == 200
        assert emb.loaded_path() == str(dest)
        assert len(emb.embed_texts(["rag text"])[0]) == CONFIGURED_DIM

    def test_no_registry_entry_and_not_configured_is_not_routed_to_the_embedder(self, client, worker, configured):
        r = embed(client, "nothing-by-this-name")
        assert r.status_code >= 400, r.text
        assert configured == [] and worker.instances == []


class TestABrokenEntryIsRefusedByName:
    def test_a_missing_file_is_a_422_naming_the_entry(self, client, worker, configured, home, tmp_path):
        raw_entry(home, "nomic-gone", str(tmp_path / "nowhere.gguf"))
        r = embed(client, "nomic-gone")
        assert r.status_code == 422, r.text
        detail = r.json()["detail"]
        assert "nomic-gone" in detail and "missing" in detail
        assert configured == [] and worker.instances == []

    def test_an_empty_path_is_a_422_naming_the_entry(self, client, worker, configured, home):
        raw_entry(home, "nomic-empty", "")
        r = embed(client, "nomic-empty")
        assert r.status_code == 422, r.text
        assert "nomic-empty" in r.json()["detail"]
        assert configured == [] and worker.instances == []

    def test_a_unc_path_is_refused_without_a_load(self, client, worker, configured, home):
        raw_entry(home, "nomic-unc", r"\\server\share\nomic.gguf")
        r = embed(client, "nomic-unc")
        assert r.status_code == 422, r.text
        assert "nomic-unc" in r.json()["detail"]
        assert "network or device path" in r.json()["detail"]
        assert configured == [] and worker.instances == []

    def test_ollama_embed_reports_the_same_refusal(self, client, worker, configured, home, tmp_path):
        raw_entry(home, "nomic-gone", str(tmp_path / "nowhere.gguf"))
        r = client.post("/api/embed", json={"model": "nomic-gone", "input": "hi"})
        assert r.status_code == 422, r.text
        assert "nomic-gone" in r.text
        assert configured == []


class TestMcpEmbedTool:
    @staticmethod
    def _mcp(default_model="chat-model"):
        from unittest.mock import MagicMock

        from localm.plugins.mcpserver.server import (EngineCache, MCPStdioServer,
                                                      build_tools)
        engines = EngineCache(default_model=default_model,
                              engine_factory=lambda name: MagicMock())
        return MCPStdioServer(build_tools(engines)), engines

    @staticmethod
    def _call(server, model, texts=("hello",)):
        args = {"texts": list(texts)}
        if model is not None:
            args["model"] = model
        resp = server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                              "params": {"name": "embed", "arguments": args}})
        return resp["result"]

    def test_a_named_embedding_gguf_embeds_with_its_own_file(self, home, worker, configured, tmp_path):
        path = register(tmp_path, "nomic")
        server, engines = self._mcp()
        resident = list(engines.resident)
        made_room = []
        engines._make_room_for = lambda name: made_room.append(name)
        result = self._call(server, "nomic", ["a", "b"])
        assert not result.get("isError"), result
        vecs = json.loads(result["content"][0]["text"])
        assert [len(v) for v in vecs] == [768, 768]
        assert configured == []
        assert [w.model_path for w in worker.instances] == [str(path.resolve())]
        assert made_room == [] and engines.resident == resident

    def test_the_resident_chat_model_is_not_unloaded_for_it(self, home, worker, configured, tmp_path):
        register(tmp_path, "nomic")
        server, engines = self._mcp()
        resident = list(engines.resident)
        assert resident == ["chat-model"]
        engines._make_room_for = lambda name: pytest.fail(f"made room for {name}")
        assert not self._call(server, "nomic").get("isError")
        assert engines.resident == resident

    def test_the_configured_name_and_an_unnamed_call_are_left_to_the_engine_path(self, home, worker, configured, tmp_path):
        from localm.plugins.mcpserver.tools.chat import _named_embedding_path
        dest = emb._embeddings_dir() / emb.KNOWN_EMBEDDING_MODELS["bge-small-en-v1.5"][1]
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(embedding_bytes("bert"))
        R._register("embedding-bge-small-en-v1.5", dest, model_type="embedding")
        register(tmp_path, "nomic")
        assert _named_embedding_path(None) is None
        assert _named_embedding_path("") is None
        assert _named_embedding_path("not-registered") is None
        assert _named_embedding_path("embedding-bge-small-en-v1.5") is None
        assert _named_embedding_path("nomic") is not None

    def test_a_missing_file_is_refused_naming_the_entry(self, home, worker, configured, tmp_path):
        raw_entry(home, "nomic-gone", str(tmp_path / "nowhere.gguf"))
        server, _ = self._mcp()
        result = self._call(server, "nomic-gone")
        assert result["isError"] is True
        assert "nomic-gone" in result["content"][0]["text"]
        assert configured == [] and worker.instances == []

    def test_a_worker_failure_is_a_tool_error(self, home, worker, configured, tmp_path, monkeypatch):
        register(tmp_path, "nomic")

        def boom(path, **kw):
            raise RuntimeError("native load failed")
        monkeypatch.setattr(emb, "IsolatedEmbedder", boom)
        server, _ = self._mcp()
        result = self._call(server, "nomic")
        assert result["isError"] is True
        assert "native load failed" in result["content"][0]["text"]


class TestOllamaRoutes:
    def test_api_embed_uses_the_named_model(self, client, worker, configured, tmp_path):
        register(tmp_path, "nomic")
        r = client.post("/api/embed", json={"model": "nomic", "input": ["a", "b"]})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["model"] == "nomic"
        assert [len(v) for v in body["embeddings"]] == [768, 768]
        assert configured == []

    def test_api_embeddings_uses_the_named_model(self, client, worker, configured, tmp_path):
        register(tmp_path, "nomic")
        r = client.post("/api/embeddings", json={"model": "nomic", "prompt": "a"})
        assert r.status_code == 200, r.text
        assert len(r.json()["embedding"]) == 768
        assert configured == []

    def test_api_embed_for_the_configured_name_is_unchanged(self, client, worker, configured, tmp_path):
        register(tmp_path, "nomic")
        r = client.post("/api/embed", json={"model": "bge-small-en-v1.5", "input": "a"})
        assert r.status_code == 200, r.text
        assert len(r.json()["embeddings"][0]) == CONFIGURED_DIM
        assert worker.instances == []


class TestLifecycle:
    def test_unload_all_releases_the_named_embedder(self, client, worker, configured, tmp_path):
        from localm.inference import http_server as hs
        register(tmp_path, "nomic")
        assert embed(client, "nomic").status_code == 200
        assert rr.is_loaded()
        asyncio.run(hs.unload_all_models())
        assert worker.instances[0].closed
        assert not rr.is_loaded()

    def test_an_in_flight_embed_is_not_released_under_it(self, client, worker, configured, tmp_path):
        register(tmp_path, "nomic")
        seen = []

        def try_release(w):
            seen.append(rr.reset_reranker(force=False))
        assert embed(client, "nomic").status_code == 200
        w = worker.instances[0]
        w.during_embed = try_release
        assert embed(client, "nomic").status_code == 200
        assert seen == [False]
        assert not w.closed and rr.is_loaded()
        assert rr.active_requests() == 0

    def test_the_resident_path_is_reported_for_the_models_page(self, client, worker, configured, tmp_path):
        path = register(tmp_path, "nomic")
        assert embed(client, "nomic").status_code == 200
        assert rr.reranker_info()["path"] == str(path.resolve())

    def test_exit_release_reaches_the_named_embedder(self, client, worker, configured, tmp_path):
        register(tmp_path, "nomic")
        assert embed(client, "nomic").status_code == 200
        assert rr.release_for_exit() is True
        assert worker.instances[0]._runner.grace == [5.0]

    def test_exit_release_does_not_wait_for_an_embed_in_flight(self, client, worker, configured, tmp_path):
        register(tmp_path, "nomic")
        released = []
        assert embed(client, "nomic").status_code == 200
        w = worker.instances[0]
        w.during_embed = lambda _w: released.append(rr.release_for_exit())
        assert embed(client, "nomic").status_code == 200
        assert released == [True] and w._runner.grace == [0]

    def test_a_request_whose_model_is_replaced_before_it_is_counted_retries(self, home, worker, tmp_path, monkeypatch):
        path = register(tmp_path, "nomic")
        real = rr._get_resident
        calls = []

        def replaced_once(p, purpose):
            model = real(p, purpose)
            calls.append(model)
            if len(calls) == 1:
                rr.reset_reranker()
            return model
        monkeypatch.setattr(rr, "_get_resident", replaced_once)
        vecs = rr.embed(str(path.resolve()), ["a"])
        assert len(vecs[0]) == 768
        assert len(calls) == 2 and calls[0].closed and not calls[1].closed

    def test_a_request_replaced_twice_is_refused(self, home, worker, tmp_path, monkeypatch):
        path = register(tmp_path, "nomic")
        real = rr._get_resident

        def always_replaced(p, purpose):
            model = real(p, purpose)
            rr.reset_reranker()
            return model
        monkeypatch.setattr(rr, "_get_resident", always_replaced)
        with pytest.raises(rr.RerankerUnavailableError, match="was replaced"):
            rr.embed(str(path.resolve()), ["a"])

    def test_a_rerank_waits_for_a_named_embed_in_flight_then_replaces_it(self, home, worker, tmp_path, monkeypatch):
        import threading
        from tests._gguf_builder import reranker_bytes
        emb_path = register(tmp_path, "nomic")
        rer_path = tmp_path / "rer.gguf"
        rer_path.write_bytes(reranker_bytes("bert"))

        class Worker(FakeEmbedWorker):
            def rerank(self, pairs):
                return [{"scores": [1.0], "tokens": 1, "truncated": False} for _ in pairs]
        monkeypatch.setattr(emb, "IsolatedEmbedder", Worker)
        in_embed, finish = threading.Event(), threading.Event()
        named = rr.get_embedding_model(str(emb_path.resolve()))

        def slow(_w):
            in_embed.set()
            assert finish.wait(10)
        named.during_embed = slow
        out = {}
        t = threading.Thread(target=lambda: out.setdefault(
            "vecs", rr.embed(str(emb_path.resolve()), ["a"])))
        t.start()
        assert in_embed.wait(10)
        r = threading.Thread(target=lambda: out.setdefault(
            "rerank", rr.rerank(str(rer_path), "q", ["d"])))
        r.start()
        r.join(0.5)
        assert r.is_alive() and not named.closed
        finish.set()
        t.join(10)
        r.join(10)
        assert len(out["vecs"][0]) == 768
        assert out["rerank"].scored[0]["scores"] == [1.0]
        assert named.closed and rr.reranker_info()["kind"] == "reranker model"

    def test_a_wait_for_the_other_model_is_bounded(self, home, worker, tmp_path, monkeypatch):
        from tests._gguf_builder import reranker_bytes
        emb_path = register(tmp_path, "nomic")
        rer_path = tmp_path / "rer.gguf"
        rer_path.write_bytes(reranker_bytes("bert"))
        monkeypatch.setattr(rr, "_REPLACE_WAIT_S", 0.2)
        named = rr.get_embedding_model(str(emb_path.resolve()))
        with rr._LOCK:
            rr._RESIDENT_REQUESTS += 1
        with pytest.raises(rr.RerankerUnavailableError, match="still scoring"):
            rr.get_reranker(str(rer_path))
        assert not named.closed

    def test_a_busy_named_embedder_is_reported_by_kind_by_unload_all(self, client, worker, configured, tmp_path):
        from localm.inference import http_server as hs
        register(tmp_path, "nomic")
        results = []

        def unload_during_embed(w):
            loop = asyncio.new_event_loop()
            try:
                results.append(loop.run_until_complete(hs.unload_all_models()))
            finally:
                loop.close()
        assert embed(client, "nomic").status_code == 200
        worker.instances[0].during_embed = unload_during_embed
        assert embed(client, "nomic").status_code == 200
        assert "embedding model" in results[0]["skipped_in_use"]
        assert not worker.instances[0].closed

    def test_a_rerank_in_flight_is_not_released_under_it(self, home, worker, tmp_path, monkeypatch):
        from tests._gguf_builder import reranker_bytes
        path = tmp_path / "rer.gguf"
        path.write_bytes(reranker_bytes("bert"))
        seen = []

        class RerankWorker(FakeEmbedWorker):
            def rerank(self, pairs):
                seen.append(rr.reset_reranker(force=False))
                return [{"scores": [1.0], "tokens": 1, "truncated": False} for _ in pairs]
        monkeypatch.setattr(emb, "IsolatedEmbedder", RerankWorker)
        rr.rerank(str(path), "q", ["d"])
        assert seen == [False]
        assert rr.is_loaded() and rr.active_requests() == 0


class TestConfiguredModelPath:
    def _agrees(self, expected):
        assert emb.configured_model_path() == expected
        assert emb.resolve_embedding_model_path(allow_download=False) == expected

    def test_a_known_key_present_on_disk(self, home):
        dest = emb._embeddings_dir() / emb.KNOWN_EMBEDDING_MODELS["bge-small-en-v1.5"][1]
        assert emb.configured_model_path() is None
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(embedding_bytes("bert"))
        self._agrees(str(dest))

    def test_a_registered_name(self, home, tmp_path):
        path = register(tmp_path, "mine")
        (home / "config.json").write_text(json.dumps({"embedding_model": "mine"}))
        assert os.path.samefile(emb.configured_model_path(), path)
        assert os.path.samefile(emb.resolve_embedding_model_path(allow_download=False), path)

    def test_a_file_path(self, home, tmp_path):
        path = tmp_path / "direct.gguf"
        path.write_bytes(embedding_bytes("bert"))
        (home / "config.json").write_text(json.dumps({"embedding_model": str(path)}))
        self._agrees(str(path))

    def test_a_unc_spec_is_none_without_a_filesystem_call(self, home, monkeypatch):
        unc = "\\\\server\\share\\x.gguf"
        (home / "config.json").write_text(json.dumps({"embedding_model": unc}))
        assert emb._current_spec() == unc
        calls = []
        real = emb.Path.is_file

        def spy(self):
            if "server" in str(self):
                calls.append(str(self))
            return real(self)
        monkeypatch.setattr(emb.Path, "is_file", spy)
        assert emb.configured_model_path() is None
        assert calls == []
        monkeypatch.setattr(emb, "_nonlocal_spec_reason", lambda spec: None)
        emb.configured_model_path()
        assert calls, "the spy must see the UNC path once the refusal is bypassed"

    def test_it_records_nothing_in_last_error(self, home):
        (home / "config.json").write_text(json.dumps({"embedding_model": "not-a-thing"}))
        emb.reset_embedder()
        assert emb.configured_model_path() is None
        assert emb.last_error() is None
        emb.resolve_embedding_model_path(allow_download=False)
        assert emb.last_error() is not None
