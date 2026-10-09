# SPDX-License-Identifier: AGPL-3.0-or-later
"""Reranking surfaces: model resolution, ranking, the resident-reranker lifecycle,
POST /v1/rerank, ``localm rerank`` and the MCP ``rerank`` tool.

Registry entries come from real (synthetic) GGUF headers registered through the
model manager, so the reranker flag is the one the product records. The native
scoring is replaced where the worker would run; the real worker is covered by
tests/test_rerank_real_gguf.py.
"""

import json
import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from localm import model_manager as mm
from localm.inference import reranker as rr
from localm.inference.backends.base import (PretokenizerUnsafeInputError,
                                            RerankerHeadMissingError, RerankInputError)
from localm.model_manager import registry as R
from tests._gguf_builder import build_gguf, embedding_bytes, reranker_bytes


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
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
    rr.reset_reranker()
    yield home
    rr.reset_reranker()


def register(tmp_path, name, data, model_type):
    path = tmp_path / f"{name}.gguf"
    path.write_bytes(data)
    R._register(name, path, model_type=model_type)
    return path


@pytest.fixture
def library(isolated_home, tmp_path):
    """One reranker, one plain embedding model and one chat model, registered."""
    paths = {
        "rerank-a": register(tmp_path, "rerank-a", reranker_bytes("bert"), "embedding"),
        "embed-a": register(tmp_path, "embed-a", embedding_bytes("bert"), "embedding"),
        "chat-a": register(tmp_path, "chat-a", build_gguf("llama"), "llm"),
    }
    return paths


def scored(*rows):
    """RerankOutcome rows: each row is a score or a (scores, tokens, truncated) tuple."""
    out = []
    for row in rows:
        if isinstance(row, tuple):
            scores, tokens, truncated = row
        else:
            scores, tokens, truncated = [row], 3, False
        out.append({"scores": list(scores), "tokens": tokens, "truncated": truncated})
    return out


class TestResolve:
    def test_a_named_reranker_resolves_to_its_file(self, library):
        name, path = rr.resolve_reranker("rerank-a")
        assert name == "rerank-a" and path == str(library["rerank-a"].resolve())

    def test_an_omitted_model_resolves_to_the_only_reranker(self, library):
        assert rr.resolve_reranker(None)[0] == "rerank-a"
        assert rr.resolve_reranker("")[0] == "rerank-a"
        assert rr.resolve_reranker("localm")[0] == "rerank-a"

    def test_an_omitted_model_with_no_reranker_says_so(self, isolated_home, tmp_path):
        register(tmp_path, "embed-a", embedding_bytes("bert"), "embedding")
        with pytest.raises(rr.RerankerModelError, match="No reranker model is registered") as e:
            rr.resolve_reranker(None)
        assert e.value.status == 404

    def test_an_omitted_model_with_several_rerankers_asks_for_one(self, library, tmp_path):
        register(tmp_path, "rerank-b", reranker_bytes("qwen3", rank_key=True), "embedding")
        with pytest.raises(rr.RerankerModelError, match="rerank-a, rerank-b") as e:
            rr.resolve_reranker(None)
        assert e.value.status == 400
        assert rr.registered_rerankers() == ["rerank-a", "rerank-b"]

    def test_an_unregistered_name_is_not_found_and_a_path_is_never_accepted(self, library, tmp_path):
        with pytest.raises(rr.RerankerModelError) as e:
            rr.resolve_reranker("nope")
        assert e.value.status == 404
        with pytest.raises(rr.RerankerModelError) as e:
            rr.resolve_reranker(str(library["rerank-a"]))
        assert e.value.status == 404

    def test_an_embedding_model_is_refused_with_what_it_is(self, library):
        with pytest.raises(rr.RerankerModelError, match="is an embedding model, not a reranker") as e:
            rr.resolve_reranker("embed-a")
        assert e.value.status == 422

    def test_a_chat_model_is_refused_with_what_it_is(self, library):
        with pytest.raises(rr.RerankerModelError, match="is a chat model, not a reranker") as e:
            rr.resolve_reranker("chat-a")
        assert e.value.status == 422

    def test_a_reranker_whose_file_is_gone_is_refused(self, library):
        library["rerank-a"].unlink()
        with pytest.raises(rr.RerankerModelError, match="missing") as e:
            rr.resolve_reranker("rerank-a")
        assert e.value.status == 422


class TestRankResults:
    def test_best_first_with_ties_in_request_order(self):
        rows = rr.rank_results(scored(0.2, 0.9, 0.9, -1.0))
        assert [r["index"] for r in rows] == [1, 2, 0, 3]
        assert rows[0] == {"index": 1, "relevance_score": 0.9}

    def test_top_n_cuts_the_list_and_larger_than_the_list_keeps_it_all(self):
        assert [r["index"] for r in rr.rank_results(scored(1, 3, 2), top_n=2)] == [1, 2]
        assert len(rr.rank_results(scored(1, 3, 2), top_n=50)) == 3

    def test_a_cut_document_is_flagged(self):
        rows = rr.rank_results(scored(([1.0], 4, True), 0.5))
        assert rows[0]["truncated"] is True and "truncated" not in rows[1]

    def test_several_outputs_add_label_scores_and_the_first_is_the_relevance(self):
        rows = rr.rank_results(scored(([0.9, 0.1], 4, False)), labels=["yes", "no"])
        assert rows[0]["relevance_score"] == 0.9
        assert rows[0]["label_scores"] == {"yes": 0.9, "no": 0.1}

    def test_unnamed_labels_fall_back_to_their_position(self):
        rows = rr.rank_results(scored(([0.9, 0.1], 4, False)), labels=["yes"])
        assert rows[0]["label_scores"] == {"yes": 0.9, "1": 0.1}


class FakeIsolated:
    instances = []

    def __init__(self, path, **kw):
        self.model_path = path
        self.kw = kw
        self.cls_labels = ["yes", "no"]
        self.active_requests = 0
        self.closed = False
        self._runner = SimpleNamespace(grace=[])
        self._runner.shutdown = lambda grace=5.0: self._runner.grace.append(grace)
        FakeIsolated.instances.append(self)

    def rerank(self, pairs):
        return [{"scores": [float(len(d))], "tokens": 1, "truncated": False}
                for _q, d in pairs]

    def close(self):
        self.closed = True


@pytest.fixture
def fake_worker(monkeypatch, isolated_home):
    from localm.inference import embedder as emb
    FakeIsolated.instances = []
    monkeypatch.setattr(emb, "IsolatedEmbedder", FakeIsolated)
    monkeypatch.setattr(emb, "_maybe_swap_for_embedder", lambda *a, **k: None)
    monkeypatch.setattr(emb, "_choose_embedder_gpu_layers", lambda path, cfg: (99, None))
    return FakeIsolated


class TestResidentReranker:
    def test_the_worker_is_loaded_once_per_file_and_reused(self, library, fake_worker):
        path = str(library["rerank-a"])
        first = rr.get_reranker(path)
        assert rr.get_reranker(path) is first
        assert len(fake_worker.instances) == 1
        assert first.kw["pooling_type"] == 4

    def test_the_worker_is_asked_to_pool_for_ranking_whatever_the_user_configured(self, library, fake_worker):
        from localm.inference import embedder as emb
        rr.get_reranker(str(library["rerank-a"]))
        assert fake_worker.instances[0].kw["pooling_type"] == emb._POOLING_RANK

    def test_a_different_reranker_replaces_the_resident_one(self, library, fake_worker, tmp_path):
        other = register(tmp_path, "rerank-b", reranker_bytes("qwen3", rank_key=True), "embedding")
        a = rr.get_reranker(str(library["rerank-a"]))
        b = rr.get_reranker(str(other))
        assert a.closed and not b.closed and b is not a
        assert rr.reranker_info()["path"] == str(other)

    def test_a_changed_file_is_loaded_again(self, library, fake_worker):
        path = library["rerank-a"]
        first = rr.get_reranker(str(path))
        path.write_bytes(path.read_bytes() + b"\x00" * 64)
        second = rr.get_reranker(str(path))
        assert second is not first and first.closed

    def test_a_model_without_a_classifier_head_is_refused_before_any_load(self, isolated_home, fake_worker, tmp_path):
        path = tmp_path / "headless.gguf"
        path.write_bytes(reranker_bytes("bert", rank_key=True, tensors=("token_embd.weight",)))
        with pytest.raises(RerankerHeadMissingError, match="no classifier head"):
            rr.get_reranker(str(path))
        assert fake_worker.instances == []

    def test_a_header_that_cannot_be_read_does_not_refuse(self, isolated_home, fake_worker, tmp_path):
        path = tmp_path / "odd.gguf"
        path.write_bytes(b"GGUF" + b"\x03\x00\x00\x00" + b"\x00" * 600)
        rr.get_reranker(str(path))
        assert len(fake_worker.instances) == 1

    def test_a_failed_load_is_reported_once_and_not_retried_until_the_file_changes(self, library, isolated_home, monkeypatch):
        from localm.inference import embedder as emb
        attempts = []

        def boom(path, **kw):
            attempts.append(path)
            raise RuntimeError(f"cannot load {path}")

        monkeypatch.setattr(emb, "IsolatedEmbedder", boom)
        monkeypatch.setattr(emb, "_maybe_swap_for_embedder", lambda *a, **k: None)
        monkeypatch.setattr(emb, "_choose_embedder_gpu_layers", lambda path, cfg: (99, None))
        path = str(library["rerank-a"])
        with pytest.raises(rr.RerankerUnavailableError):
            rr.get_reranker(path)
        with pytest.raises(rr.RerankerUnavailableError):
            rr.get_reranker(path)
        assert len(attempts) == 1
        rr.reset_reranker()
        with pytest.raises(rr.RerankerUnavailableError):
            rr.get_reranker(path)
        assert len(attempts) == 2

    def test_a_load_failure_message_goes_through_the_path_scrubber(self, isolated_home, monkeypatch):
        from localm import pathscrub
        from localm.inference import embedder as emb

        def boom(path, **kw):
            raise RuntimeError(f"failed to load {path}")

        monkeypatch.setattr(emb, "IsolatedEmbedder", boom)
        monkeypatch.setattr(emb, "_maybe_swap_for_embedder", lambda *a, **k: None)
        monkeypatch.setattr(emb, "_choose_embedder_gpu_layers", lambda path, cfg: (99, None))
        monkeypatch.setattr(pathscrub, "scrub_paths", lambda text: text.replace(str(isolated_home), "<data>"))
        path = isolated_home / "models" / "rr.gguf"
        path.write_bytes(reranker_bytes("bert"))
        with pytest.raises(rr.RerankerUnavailableError) as e:
            rr.get_reranker(str(path))
        assert str(isolated_home) not in str(e.value) and "<data>" in str(e.value)

    def test_rerank_returns_scores_and_the_head_labels(self, library, fake_worker):
        outcome = rr.rerank(str(library["rerank-a"]), "q", ["a", "bbb"])
        assert [s["scores"] for s in outcome.scored] == [[1.0], [3.0]]
        assert outcome.labels == ["yes", "no"]

    def test_reset_leaves_a_busy_reranker_alone_unless_forced(self, library, fake_worker):
        w = rr.get_reranker(str(library["rerank-a"]))
        w.active_requests = 1
        assert rr.active_requests() == 1
        assert rr.reset_reranker(force=False) is False and not w.closed and rr.is_loaded()
        assert rr.reset_reranker() is True and w.closed and not rr.is_loaded()

    def test_reset_with_nothing_loaded_reports_false(self, isolated_home):
        assert rr.reset_reranker() is False and rr.active_requests() == 0
        assert rr.reranker_info() is None

    def test_exit_release_closes_an_idle_worker_politely_and_stops_a_busy_one_at_once(self, library, fake_worker):
        w = rr.get_reranker(str(library["rerank-a"]))
        assert rr.release_for_exit() is True and w._runner.grace == [5.0]
        w.active_requests = 2
        assert rr.release_for_exit() is True and w._runner.grace == [5.0, 0]

    def test_exit_release_with_nothing_loaded_reports_false(self, isolated_home):
        assert rr.release_for_exit() is False


@pytest.fixture
def client(library):
    from localm.inference.http_server import create_app
    os.environ.pop("LOCALM_API_KEY", None)
    with TestClient(create_app(None)) as c:
        yield c


def fake_rerank(rows, labels=None):
    def _rerank(path, query, documents):
        assert len(rows) == len(documents)
        return rr.RerankOutcome(rows, list(labels or []))
    return _rerank


class TestRerankRoute:
    def test_shape_ordering_and_usage(self, client, monkeypatch):
        monkeypatch.setattr(rr, "rerank", fake_rerank(scored(0.1, 0.9, 0.5)))
        r = client.post("/v1/rerank", json={
            "model": "rerank-a", "query": "q", "documents": ["a", "b", "c"]})
        assert r.status_code == 200
        body = r.json()
        assert body["object"] == "list" and body["model"] == "rerank-a"
        assert body["usage"] == {"prompt_tokens": 9, "total_tokens": 9}
        assert [x["index"] for x in body["results"]] == [1, 2, 0]
        assert body["results"][0] == {"index": 1, "relevance_score": 0.9}

    def test_the_query_and_documents_reach_the_scorer(self, client, monkeypatch):
        seen = {}

        def _rerank(path, query, documents):
            seen.update(path=path, query=query, documents=documents)
            return rr.RerankOutcome(scored(*[0.5] * len(documents)), [])

        monkeypatch.setattr(rr, "rerank", _rerank)
        client.post("/v1/rerank", json={
            "model": "rerank-a", "query": "what", "documents": ["x", {"text": "y"}]})
        assert seen["query"] == "what" and seen["documents"] == ["x", "y"]
        assert seen["path"].endswith("rerank-a.gguf")

    def test_top_n_and_return_documents(self, client, monkeypatch):
        monkeypatch.setattr(rr, "rerank", fake_rerank(scored(0.1, 0.9, 0.5)))
        body = client.post("/v1/rerank", json={
            "model": "rerank-a", "query": "q", "documents": ["a", "b", {"text": "c"}],
            "top_n": 2, "return_documents": True}).json()
        assert [x["index"] for x in body["results"]] == [1, 2]
        assert body["results"][0]["document"] == {"text": "b"}
        assert body["results"][1]["document"] == {"text": "c"}

    def test_an_omitted_model_uses_the_only_reranker(self, client, monkeypatch):
        monkeypatch.setattr(rr, "rerank", fake_rerank(scored(0.3)))
        r = client.post("/v1/rerank", json={"query": "q", "documents": ["a"]})
        assert r.status_code == 200 and r.json()["model"] == "rerank-a"

    def test_a_head_with_several_outputs_reports_each_label(self, client, monkeypatch):
        monkeypatch.setattr(rr, "rerank", fake_rerank(scored(([0.8, 0.2], 5, False)), ["yes", "no"]))
        body = client.post("/v1/rerank", json={
            "model": "rerank-a", "query": "q", "documents": ["a"]}).json()
        assert body["results"][0]["label_scores"] == {"yes": 0.8, "no": 0.2}

    def test_a_cut_document_is_flagged_in_its_result(self, client, monkeypatch):
        monkeypatch.setattr(rr, "rerank", fake_rerank(scored(([0.8], 5, True))))
        body = client.post("/v1/rerank", json={
            "model": "rerank-a", "query": "q", "documents": ["a"]}).json()
        assert body["results"][0]["truncated"] is True

    @pytest.mark.parametrize("payload,status", [
        ({"model": "rerank-a", "query": "   ", "documents": ["a"]}, 400),
        ({"model": "rerank-a", "query": "q", "documents": []}, 422),
        ({"model": "rerank-a", "query": "q"}, 422),
        ({"model": "rerank-a", "documents": ["a"]}, 422),
        ({"model": "rerank-a", "query": "q", "documents": ["a"], "top_n": 0}, 422),
        ({"model": "rerank-a", "query": "q", "documents": [{"nope": 1}]}, 422),
        ({"model": "nope", "query": "q", "documents": ["a"]}, 404),
        ({"model": "embed-a", "query": "q", "documents": ["a"]}, 422),
        ({"model": "chat-a", "query": "q", "documents": ["a"]}, 422),
    ])
    def test_a_request_that_cannot_be_served_is_refused_clearly(self, client, monkeypatch, payload, status):
        def must_not_run(*a, **k):
            raise AssertionError("the model must not be touched for a refused request")

        monkeypatch.setattr(rr, "rerank", must_not_run)
        r = client.post("/v1/rerank", json=payload)
        assert r.status_code == status, r.text

    def test_a_non_reranker_is_refused_by_name_and_kind(self, client):
        detail = client.post("/v1/rerank", json={
            "model": "embed-a", "query": "q", "documents": ["a"]}).json()["detail"]
        assert "embed-a" in detail and "embedding model" in detail

    @pytest.mark.parametrize("error,status", [
        (RerankInputError("query too long"), 400),
        (PretokenizerUnsafeInputError("unsafe text"), 400),
        (RerankerHeadMissingError("no head"), 422),
        (rr.RerankerUnavailableError("could not load"), 503),
        (RuntimeError("worker died"), 503),
    ])
    def test_scoring_failures_map_to_distinct_statuses(self, client, monkeypatch, error, status):
        def failing(*a, **k):
            raise error

        monkeypatch.setattr(rr, "rerank", failing)
        r = client.post("/v1/rerank", json={"model": "rerank-a", "query": "q", "documents": ["a"]})
        assert r.status_code == status
        assert str(error) in r.json()["detail"]

    def test_a_key_is_required_once_one_is_configured(self, client, monkeypatch):
        monkeypatch.setattr(rr, "rerank", fake_rerank(scored(0.3)))
        with patch.dict(os.environ, {"LOCALM_API_KEY": "secret-key-for-rerank-test"}):
            r = client.post("/v1/rerank", json={"model": "rerank-a", "query": "q", "documents": ["a"]})
            assert r.status_code == 401
            ok = client.post("/v1/rerank", json={"model": "rerank-a", "query": "q", "documents": ["a"]},
                             headers={"Authorization": "Bearer secret-key-for-rerank-test"})
            assert ok.status_code == 200

    def test_the_route_is_exempt_from_the_cross_origin_refusal_like_embeddings(self, client, monkeypatch):
        monkeypatch.setattr(rr, "rerank", fake_rerank(scored(0.3)))
        r = client.post("/v1/rerank", json={"model": "rerank-a", "query": "q", "documents": ["a"]},
                        headers={"Origin": "http://localhost:9999"})
        assert r.status_code == 200


class TestCli:
    def invoke(self, args):
        from click.testing import CliRunner

        from localm import cli
        return CliRunner().invoke(cli.main, ["rerank", *args])

    def test_ranks_the_documents_best_first(self, library, monkeypatch):
        monkeypatch.setattr(rr, "rerank", fake_rerank(scored(0.1, 0.9)))
        res = self.invoke(["what", "alpha", "beta", "--model", "rerank-a"])
        assert res.exit_code == 0, res.output
        lines = [ln for ln in res.output.splitlines() if "#" in ln]
        assert "beta" in lines[0] and "alpha" in lines[1]

    def test_json_output_has_the_route_shape_with_documents(self, library, monkeypatch):
        monkeypatch.setattr(rr, "rerank", fake_rerank(scored(0.1, 0.9)))
        res = self.invoke(["what", "alpha", "beta", "--model", "rerank-a", "--json", "--top-n", "1"])
        body = json.loads(res.output)
        assert body["model"] == "rerank-a"
        assert body["results"] == [{"index": 1, "relevance_score": 0.9,
                                    "document": {"text": "beta"}}]

    def test_documents_can_come_from_a_file(self, library, monkeypatch, tmp_path):
        docs = tmp_path / "docs.txt"
        docs.write_text("first\n\nsecond\n", encoding="utf-8")
        seen = {}

        def _rerank(path, query, documents):
            seen["documents"] = documents
            return rr.RerankOutcome(scored(0.2, 0.4, 0.1), [])

        monkeypatch.setattr(rr, "rerank", _rerank)
        res = self.invoke(["q", "arg", "--file", str(docs), "--model", "rerank-a"])
        assert res.exit_code == 0, res.output
        assert seen["documents"] == ["arg", "first", "second"]

    def test_no_documents_is_a_usage_error(self, library):
        res = self.invoke(["q", "--model", "rerank-a"])
        assert res.exit_code == 2 and "at least one document" in res.output

    def test_a_model_that_is_not_a_reranker_exits_nonzero_naming_why(self, library):
        res = self.invoke(["q", "d", "--model", "chat-a"])
        assert res.exit_code == 1 and "not a reranker" in res.output

    def test_a_scoring_failure_exits_nonzero_with_the_reason(self, library, monkeypatch):
        def failing(*a, **k):
            raise RerankInputError("query too long")

        monkeypatch.setattr(rr, "rerank", failing)
        res = self.invoke(["q", "d", "--model", "rerank-a"])
        assert res.exit_code == 1 and "query too long" in res.output


class TestMcpTool:
    def call(self, args):
        from localm.plugins.mcpserver.tools import chat as chat_tools
        tools = chat_tools.build(object())
        assert "rerank" in tools
        return tools["rerank"]["handler"](args)

    def text(self, result):
        return result["content"][0]["text"]

    def test_the_tool_is_listed_with_its_schema(self):
        from localm.plugins.mcpserver.tools import chat as chat_tools
        spec = chat_tools.build(object())["rerank"]
        assert spec["inputSchema"]["required"] == ["query", "documents"]
        assert "top_n" in spec["inputSchema"]["properties"]

    def test_returns_the_ranking_as_json(self, library, monkeypatch):
        monkeypatch.setattr(rr, "rerank", fake_rerank(scored(0.1, 0.9)))
        result = self.call({"query": "q", "documents": ["a", "b"], "model": "rerank-a", "top_n": 1})
        assert not result.get("isError")
        body = json.loads(self.text(result))
        assert body == {"model": "rerank-a", "results": [{"index": 1, "relevance_score": 0.9}]}

    @pytest.mark.parametrize("args", [
        {"documents": ["a"]},
        {"query": " ", "documents": ["a"]},
        {"query": "q"},
        {"query": "q", "documents": []},
        {"query": "q", "documents": [1, 2]},
        {"query": "q", "documents": ["a"], "top_n": 0},
        {"query": "q", "documents": ["a"], "top_n": True},
    ])
    def test_bad_arguments_are_an_error_result(self, library, args):
        assert self.call(args).get("isError") is True

    def test_a_non_reranker_is_an_error_result_naming_why(self, library):
        result = self.call({"query": "q", "documents": ["a"], "model": "chat-a"})
        assert result.get("isError") is True and "not a reranker" in self.text(result)

    def test_a_scoring_failure_is_an_error_result(self, library, monkeypatch):
        def failing(*a, **k):
            raise RerankInputError("query too long")

        monkeypatch.setattr(rr, "rerank", failing)
        result = self.call({"query": "q", "documents": ["a"], "model": "rerank-a"})
        assert result.get("isError") is True and "query too long" in self.text(result)
