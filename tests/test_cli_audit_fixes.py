# SPDX-License-Identifier: AGPL-3.0-or-later
"""CLI defects found by the functional audit: running an embedding model as a
chat model, RAG remove-doc path spelling, native loader output from in-process
embedder and reranker loads, and CLI wording that named the HTTP request."""

from __future__ import annotations

import json
import sys

import pytest
from click.testing import CliRunner

from localm.cli import main


@pytest.fixture(autouse=True)
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path))
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.setenv("COLUMNS", "300")
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", tmp_path)
    monkeypatch.setattr(cfg, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", tmp_path / "registry.json")
    return tmp_path


# ----------------------------------------------------------------- F5

def test_run_with_an_embedding_model_is_refused_before_any_engine_is_built(
        env, monkeypatch):
    model_file = env / "models" / "bge-small.gguf"
    model_file.parent.mkdir()
    model_file.write_bytes(b"GGUF")
    (env / "registry.json").write_text(json.dumps({
        "bge-small": {"path": str(model_file), "source": "local",
                      "model_type": "embedding"}}), encoding="utf-8")
    built = []

    class _Boom:
        def __init__(self, *a, **k):
            built.append(a)
            raise AssertionError("an Engine was built for an embedding model")

    monkeypatch.setattr("localm.inference.engine.Engine", _Boom)
    r = CliRunner().invoke(main, ["run", "bge-small", "--no-server", "-p", "hi"])
    assert r.exit_code == 1, r.output
    assert "bge-small" in r.output
    assert "embedding" in r.output.lower()
    assert built == []


def test_run_refusal_matches_the_http_refusal(env):
    from localm.model_manager.registry import chat_load_refusal
    model_file = env / "bge-small.gguf"
    model_file.write_bytes(b"GGUF")
    reg = {"bge-small": {"path": str(model_file), "source": "local",
                         "model_type": "embedding"}}
    (env / "registry.json").write_text(json.dumps(reg), encoding="utf-8")
    expected = chat_load_refusal("bge-small", reg)
    r = CliRunner().invoke(main, ["run", "bge-small", "--no-server", "-p", "hi"])
    assert expected and " ".join(expected.split()) in " ".join(r.output.split())


# ----------------------------------------------------------------- F6

@pytest.fixture
def indexed(env):
    from localm.rag import Collection
    docs = env / "docs"
    docs.mkdir()
    (docs / "Zebra.txt").write_text("zebras have stripes and graze", encoding="utf-8")
    Collection("kb").create()
    Collection("kb").add_paths([str(docs)])
    return Collection("kb"), docs


def test_remove_doc_accepts_forward_slashes(indexed):
    coll, docs = indexed
    spelled = str(docs / "Zebra.txt").replace("\\", "/")
    assert coll.remove_doc(spelled) is True
    assert coll.docs() == []


def test_remove_doc_accepts_a_different_case_on_windows(indexed):
    coll, docs = indexed
    if sys.platform != "win32":
        pytest.skip("case-insensitive paths are a Windows property")
    assert coll.remove_doc(str(docs / "zebra.TXT").upper()) is True
    assert coll.docs() == []


def test_remove_doc_still_refuses_a_path_that_is_not_indexed(indexed):
    coll, docs = indexed
    assert coll.remove_doc(str(docs / "other.txt")) is False
    assert len(coll.docs()) == 1


def test_remove_doc_removes_the_chunks_of_the_matched_spelling(indexed):
    coll, docs = indexed
    coll.remove_doc(str(docs / "Zebra.txt").replace("\\", "/"))
    assert coll.query("zebras") == [] or all(
        "Zebra" not in str(h.get("source", "")) for h in coll.query("zebras"))


# ----------------------------------------------------------------- F7

def test_memory_embedder_load_runs_with_the_native_console_paused(monkeypatch):
    from localm.debuglog import native_console_is_paused
    from localm.plugins.builtin.memory import cli as memcli
    seen = {}

    class _Emb:
        def embed(self, texts):
            seen["embed"] = native_console_is_paused()
            return [[0.0] for _ in texts]

    def fake_get_embedder(*a, **k):
        seen["load"] = native_console_is_paused()
        return _Emb()

    monkeypatch.setattr("localm.inference.embedder.get_embedder", fake_get_embedder)
    fn = memcli._embed_fn()
    fn(["a"])
    assert seen == {"load": True, "embed": True}
    assert native_console_is_paused() is False


def test_rerank_command_runs_with_the_native_console_paused(monkeypatch):
    from localm.debuglog import native_console_is_paused
    from localm.inference import reranker
    seen = {}

    def fake_resolve(model, *, where=""):
        seen["resolve"] = native_console_is_paused()
        return "m", "m.gguf"

    class _Outcome:
        scored = [{"scores": [0.5], "tokens": 3, "truncated": False}]
        labels = []

    def fake_rerank(path, query, docs):
        seen["rerank"] = native_console_is_paused()
        return _Outcome()

    monkeypatch.setattr(reranker, "resolve_reranker", fake_resolve)
    monkeypatch.setattr(reranker, "rerank", fake_rerank)
    r = CliRunner().invoke(main, ["rerank", "q", "doc one", "--json"])
    assert r.exit_code == 0, r.output
    assert seen == {"resolve": True, "rerank": True}
    assert native_console_is_paused() is False


# ------------------------------------------------------------- wording

def test_speak_without_a_tts_model_does_not_talk_about_a_request_field(env):
    r = CliRunner().invoke(main, ["speak", "hello", "-o", str(env / "out.wav")])
    assert r.exit_code == 1
    assert "request" not in r.output
    assert "--model" in r.output


def test_the_http_speech_message_still_names_the_request_field():
    from localm.inference import speech
    with pytest.raises(speech.SpeechModelError) as ei:
        speech.resolve_speech_model(None)
    assert "request's model field" in str(ei.value)


def test_rerank_without_a_reranker_names_the_cli_option(env):
    r = CliRunner().invoke(main, ["rerank", "q", "doc"])
    assert r.exit_code == 1
    assert "request" not in r.output and "--model" in r.output


def test_rerank_cli_refusal_of_a_non_reranker_names_no_http_route(env):
    (env / "registry.json").write_text(json.dumps({
        "chat-model": {"path": str(env / "c.gguf"), "source": "local"},
        "emb": {"path": str(env / "e.gguf"), "source": "local",
                "model_type": "embedding"}}), encoding="utf-8")
    for name in ("chat-model", "emb"):
        r = CliRunner().invoke(main, ["rerank", "q", "doc", "--model", name])
        assert r.exit_code == 1
        assert "/v1/" not in r.output, r.output


def test_the_http_reranker_refusal_still_names_its_route(env):
    from localm.inference import reranker
    (env / "registry.json").write_text(json.dumps({
        "chat-model": {"path": str(env / "c.gguf"), "source": "local"}}), encoding="utf-8")
    with pytest.raises(reranker.RerankerModelError) as ei:
        reranker.resolve_reranker("chat-model")
    assert "/v1/rerank" in str(ei.value)


# ------------------------------------------- unregistered embedding GGUF path

def _gguf(path, arch, extra=None):
    from tests.test_model_type_detection import _build_gguf_bytes
    path.write_bytes(_build_gguf_bytes(arch, extra))
    return path


def test_an_unregistered_embedding_gguf_path_is_refused_by_its_header(env):
    from localm.model_manager.registry import chat_load_refusal
    emb = _gguf(env / "enc.gguf", "bert", {"bert.pooling_type": "mean"})
    msg = chat_load_refusal(str(emb), {})
    assert msg and "embedding model" in msg and "/v1/embeddings" in msg


def test_an_unregistered_decoder_embedding_is_refused_by_its_pooling_key(env):
    from localm.model_manager.registry import chat_load_refusal
    emb = _gguf(env / "qwen-emb.gguf", "qwen3", {"qwen3.pooling_type": "last"})
    assert chat_load_refusal(str(emb), {}) is not None


def test_an_unregistered_chat_gguf_path_is_not_refused(env):
    from localm.model_manager.registry import chat_load_refusal
    chat = _gguf(env / "chat.gguf", "llama")
    assert chat_load_refusal(str(chat), {}) is None
    assert chat_load_refusal(str(env / "missing.gguf"), {}) is None
    assert chat_load_refusal("not-a-path", {}) is None


def test_a_registered_model_set_to_llm_is_still_loadable(env):
    from localm.model_manager.registry import chat_load_refusal
    emb = _gguf(env / "enc.gguf", "bert", {"bert.pooling_type": "mean"})
    reg = {"enc": {"path": str(emb), "source": "local", "model_type": "llm"}}
    assert chat_load_refusal("enc", reg) is None


def test_run_with_an_unregistered_embedding_path_exits_1(env, monkeypatch):
    emb = _gguf(env / "enc.gguf", "bert", {"bert.pooling_type": "mean"})

    class _Boom:
        def __init__(self, *a, **k):
            raise AssertionError("an Engine was built for an embedding file")

    monkeypatch.setattr("localm.inference.engine.Engine", _Boom)
    r = CliRunner().invoke(main, ["run", str(emb), "--no-server", "-p", "hi"])
    assert r.exit_code == 1, r.output
    assert "embedding" in r.output.lower()


def test_gui_and_serve_startup_refuse_an_embedding_model_by_name_or_path(env):
    from rich.console import Console

    from localm.plugins.gui.cli import _resolve_startup_model_file
    emb = _gguf(env / "enc.gguf", "bert", {"bert.pooling_type": "mean"})
    chat = _gguf(env / "chat.gguf", "llama")
    reg = {"enc": {"path": str(emb), "source": "local", "model_type": "embedding"},
           "chat": {"path": str(chat), "source": "local"}}
    (env / "registry.json").write_text(json.dumps(reg), encoding="utf-8")
    console = Console(record=True, width=200)
    for model in ("enc", str(emb)):
        with pytest.raises(SystemExit) as ei:
            _resolve_startup_model_file(console, model, reg, model_less=False)
        assert ei.value.code == 1
    info, path, name = _resolve_startup_model_file(console, "chat", reg, model_less=False)
    assert name == "chat" and str(path).endswith("chat.gguf")
