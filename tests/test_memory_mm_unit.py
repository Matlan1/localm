# SPDX-License-Identifier: AGPL-3.0-or-later
"""Memory unit MM: several remember requests in one message, no fact taken from a
request to the assistant, a privacy-mode memory job that loads nothing, a
/remember that answers before the embedder loads, and an embedding model that is
never downloaded unasked while one is registered."""

from __future__ import annotations

import asyncio
import hashlib
import json
import threading

import pytest

import localm.config as config
from localm.inference import embedder
from localm.memory import explicit_fact, explicit_facts
from localm.memory.consolidate import extract


# ---- several remember requests -------------------------------------------- #

def test_two_remember_requests_store_both_facts():
    msg = "Remember that my dog is called Rex. Remember that I live in Oslo."
    assert explicit_facts(msg) == ["my dog is called Rex", "I live in Oslo"]
    assert explicit_fact(msg) == "my dog is called Rex"


def test_a_task_after_a_fact_is_not_a_second_fact():
    msg = "remember that Anna's birthday is on May 3rd. Help me pick a gift."
    assert explicit_facts(msg) == ["Anna's birthday is on May 3rd"]


def test_a_repeated_request_is_stored_once_and_a_bad_second_one_is_dropped():
    assert explicit_facts("Remember that I like tea. Remember that I like tea.") == [
        "I like tea"]
    assert explicit_facts("Remember that I like tea. Remember to buy milk.") == [
        "I like tea"]


def test_german_requests_chain_too():
    assert explicit_facts("Merk dir, dass ich Tee mag. Merk dir, dass ich in Oslo wohne.") == [
        "ich Tee mag", "ich in Oslo wohne"]


# ---- distillation grounding ------------------------------------------------ #

def _model(*facts):
    return lambda prompt: json.dumps(
        {"facts": [{"fact": f, "confidence": 0.9} for f in facts]})


def test_a_request_to_the_assistant_grounds_no_fact_about_the_user():
    session = ("User: Tell me one fact about apples.\n"
               "Assistant: You can eat apples and get the vitamin C that is needed "
               "for a strong immune system.")
    facts = ["User should eat apples for vitamin C", "User is interested in apples",
             "You can eat apples and get the vitamin C that is needed for a strong "
             "immune system."]
    assert [c["text"] for c in extract(_model(*facts), session)] == []


def test_a_question_does_not_ground_a_fact_but_a_statement_does():
    session = "User: What do I know about apples? I grow apples in my garden.\nAssistant: Ok."
    kept = [c["text"] for c in extract(_model("User grows apples in the garden"), session)]
    assert kept == ["User grows apples in the garden"]


# ---- the privacy-mode memory job ------------------------------------------- #

def test_a_privacy_mode_memory_job_loads_no_model(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path))
    monkeypatch.delenv("LOCALM_MODE", raising=False)
    monkeypatch.setattr(config, "HOME_DIR", tmp_path)
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(config, "REGISTRY_FILE", tmp_path / "registry.json")
    (tmp_path / "config.json").write_text(json.dumps({"mode": "privacy"}), encoding="utf-8")
    from localm.plugins.builtin.jobs import runner
    from localm.plugins.builtin.jobs.store import Job
    import localm.plugins.builtin.memory.plug as plug
    monkeypatch.setattr(plug, "_persist_enabled", lambda: False)
    monkeypatch.setattr(runner, "_register_live_run", lambda job: None)
    monkeypatch.setattr(runner, "_creator_refusal", lambda job: None)
    called = []
    monkeypatch.setattr(runner, "_served_engine",
                        lambda *a, **k: called.append("served"))
    monkeypatch.setattr(runner, "_load_engine",
                        lambda *a, **k: called.append("load"))
    rec = runner.run_job(Job(name="mem", prompt="", task_kind="memory"))
    assert rec["status"] == "skipped", rec
    assert "privacy" in rec["output"]
    assert called == []


# ---- /remember answers before the embedder loads --------------------------- #

def test_the_append_route_does_not_wait_for_the_embedder(tmp_path, monkeypatch):
    import localm.plugins.builtin.memory.plug as plug
    monkeypatch.setattr(plug, "_home", lambda: tmp_path)
    monkeypatch.setattr(plug, "_memory_root", lambda: tmp_path / "memory")
    monkeypatch.setattr(plug, "_persist_enabled", lambda: True)
    monkeypatch.setattr(plug, "_stop", threading.Event())
    monkeypatch.setattr(plug, "_fill_pending", False)
    release = threading.Event()
    seen = []

    def slow_embed_fn():
        seen.append(threading.current_thread().name)
        release.wait(5)
        return lambda texts: [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr(plug, "_embed_fn", slow_embed_fn)
    monkeypatch.setattr(plug, "_request_store", lambda request: plug._chat_store(None))
    try:
        res = asyncio.run(asyncio.wait_for(
            plug.memory_append(plug.MemoryAppend(text="I live in Hamburg")), 3))
        assert res["status"] == "appended"
        assert "MainThread" not in seen
    finally:
        release.set()
    for t in threading.enumerate():
        if t.name == "memory vector fill":
            t.join(5)
    store = plug._chat_store(None)
    assert any(r.text == "I live in Hamburg" for r in store.all())
    assert not store.lacks_vector()


# ---- an embedding model is never downloaded unasked ------------------------ #

BGE = embedder.KNOWN_EMBEDDING_MODELS["bge-small-en-v1.5"]
REAL = b"the real bytes"


@pytest.fixture
def emb_env(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "REGISTRY_FILE", tmp_path / "registry.json")
    monkeypatch.setattr(embedder, "_embeddings_dir", lambda: tmp_path / "emb")
    monkeypatch.setattr(embedder, "_current_spec", lambda: "bge-small-en-v1.5")
    monkeypatch.setattr(embedder, "_EMBEDDING_PINS",
                        {BGE: ("rev", hashlib.sha256(REAL).hexdigest())})
    monkeypatch.setattr(embedder, "_TRIED_DOWNLOAD", False)
    fetched = []
    monkeypatch.setattr(embedder, "_download_pinned",
                        lambda *a, **k: fetched.append(a) or None)
    import localm.netpolicy as np_
    monkeypatch.setattr(np_, "download_refusal", lambda url: None)
    monkeypatch.setattr(np_, "network_mode", lambda: "allow")
    return tmp_path, fetched


def _register(root, name, path, **extra):
    (root / "registry.json").write_text(json.dumps(
        {name: {"path": str(path), "model_type": "embedding", **extra}}),
        encoding="utf-8")


def test_nothing_registered_still_downloads_under_allow(emb_env):
    _root, fetched = emb_env
    embedder.resolve_embedding_model_path()
    assert len(fetched) == 1


def test_another_registered_embedding_model_blocks_the_automatic_download(emb_env):
    root, fetched = emb_env
    other = root / "embeddinggemma.gguf"
    other.write_bytes(b"x")
    _register(root, "embeddinggemma", other)
    assert embedder.resolve_embedding_model_path() is None
    assert fetched == []
    assert "registered" in (embedder.last_error() or "")
    assert embedder.will_download_on_first_use() is False


def test_the_explicit_download_still_works_with_one_registered(emb_env):
    root, fetched = emb_env
    other = root / "embeddinggemma.gguf"
    other.write_bytes(b"x")
    _register(root, "embeddinggemma", other)
    embedder.resolve_embedding_model_path(allow_download=True)
    assert len(fetched) == 1


def test_a_renamed_copy_of_the_known_file_is_found_by_its_digest(emb_env):
    root, fetched = emb_env
    f = root / "my-embedder.gguf"
    f.write_bytes(REAL)
    _register(root, "mine", f, sha256=hashlib.sha256(REAL).hexdigest())
    assert embedder.resolve_embedding_model_path() == str(f)
    assert fetched == []


def test_the_download_offer_follows_the_network_policy(monkeypatch):
    import localm.netpolicy as np_
    monkeypatch.setattr(np_, "download_refusal",
                        lambda url: "huggingface.co is on the deny list")
    assert embedder.embedding_download_refusal() == "huggingface.co is on the deny list"
    monkeypatch.setattr(np_, "download_refusal", lambda url: None)
    assert embedder.embedding_download_refusal() is None


def test_the_download_button_state_follows_the_refusal(monkeypatch):
    import types
    import localm.inference.http_server as hs
    import localm.plugins.builtin.memory.plug as plug
    monkeypatch.setattr(embedder, "resolve_embedding_model_path", lambda **k: None)
    monkeypatch.setattr(config, "load_config", lambda: {})
    monkeypatch.setattr(hs, "caller_grants", lambda request, scope: True)
    monkeypatch.setattr(hs, "caller_is_operator", lambda request: True)
    request = types.SimpleNamespace(app=types.SimpleNamespace(
        router=types.SimpleNamespace(url_path_for=lambda name: "/api/rag/embedding/download")))
    monkeypatch.setattr(embedder, "embedding_download_refusal",
                        lambda: "huggingface.co is on the deny list")
    refused = plug._embedder_download_status(request)
    assert refused["can_download_embedder"] is False
    assert refused["embedder_missing"] == "bge-small-en-v1.5"
    monkeypatch.setattr(embedder, "embedding_download_refusal", lambda: None)
    offered = plug._embedder_download_status(request)
    assert offered["can_download_embedder"] is True
    assert offered["embedder_model"] == "bge-small-en-v1.5"


def test_a_vector_left_beside_changed_text_is_dropped_on_load(tmp_path):
    from localm.memory import MemoryRecord, MemoryStore
    path = tmp_path / "ns.jsonl"
    store = MemoryStore.open_file(path)
    embed = lambda texts: [[1.0, 0.0] for _ in texts]  # noqa: E731
    keep = store.add(MemoryRecord(text="I live in Oslo", kind="semantic", source="user"),
                     embed_fn=embed)
    edited = store.add(MemoryRecord(text="I like tea", kind="semantic", source="user"),
                       embed_fn=embed)
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    for rec in lines:
        if rec["id"] == edited.id:
            rec["text"] = "I like coffee"
    path.write_text("\n".join(json.dumps(r) for r in lines) + "\n", encoding="utf-8")
    reopened = MemoryStore.open_file(path)
    assert reopened.vectorless_count() == 1
    assert keep.id in reopened._vectors and edited.id not in reopened._vectors
