# SPDX-License-Identifier: AGPL-3.0-or-later
"""The automatic memory pass resolves the shared embedder only when something
needs a vector.

``synthesize_memory`` used to resolve the embedder (download and load) before
the model had extracted anything, so a first chat with nothing worth keeping
still pulled the embedding model in. These tests count the real calls to
``localm.inference.embedder.get_embedder`` made by the real pass.
"""

from __future__ import annotations

import json
import os
import threading

import pytest

from localm.memory import DeferredEmbed, EmbedderUnavailable, MemoryRecord
from localm.plugins.builtin.memory import plug
from tests._shared_model_guard import (  # noqa: F401
    no_embedder_download,
    no_shared_model_left_resident,
)


class _FakeEmbedder:
    def embed(self, texts):
        return [[1.0, 0.0, 0.0] for _ in texts]


class _Resolver:
    """Stands in for get_embedder: counts calls and records whether the memory
    namespace lock was held by the calling thread when it ran."""

    def __init__(self, store_getter, embedder=True):
        self.calls = 0
        self.lock_held = []
        self._embedder = _FakeEmbedder() if embedder else None
        self._store_getter = store_getter

    def _lock_free_for_others(self):
        store = self._store_getter()
        ok = []

        def probe():
            try:
                with store._wlock("lock probe", timeout=0.3):
                    ok.append(True)
            except Exception:
                ok.append(False)
        t = threading.Thread(target=probe)
        t.start()
        t.join()
        return ok == [True]

    def __call__(self, *a, **kw):
        self.calls += 1
        self.lock_held.append(not self._lock_free_for_others())
        return self._embedder


@pytest.fixture
def memhome(tmp_path, monkeypatch):
    monkeypatch.setattr(plug, "_home", lambda: tmp_path)
    monkeypatch.setattr(plug, "_persist_enabled", lambda: True)
    monkeypatch.setenv("LOCALM_MODE", "log")
    sdir = tmp_path / "sessions"
    sdir.mkdir()
    rows = [
        {"type": "user", "data": {"content": "I love sailing, my cat is called Miso."}},
        {"type": "llm", "data": {"content": "Lovely."}},
    ]
    (sdir / "s1.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return tmp_path


@pytest.fixture
def resolver(memhome, monkeypatch):
    import localm.inference.embedder as emb
    r = _Resolver(plug._chat_store)
    monkeypatch.setattr(emb, "get_embedder", r)
    return r


@pytest.fixture
def no_embedder(memhome, monkeypatch):
    import localm.inference.embedder as emb
    r = _Resolver(plug._chat_store, embedder=False)
    monkeypatch.setattr(emb, "get_embedder", r)
    return r


def _stub(facts=(), decision="NO_OP", summary=""):
    payload = json.dumps({"facts": [{"fact": t, "confidence": 0.9} for t in facts]})
    dec = json.dumps({"decision": decision, "confidence": 0.9})

    def complete(prompt):
        if "Extract ONLY durable" in prompt:
            return payload
        if "Decide the single best action" in prompt:
            return dec
        return summary or "{}"
    return complete


def test_pass_with_no_facts_resolves_no_embedder(memhome, resolver):
    out = plug.synthesize_memory(_stub())
    assert out["status"] == "ok" and out["added"] == 0
    assert resolver.calls == 0


def test_pass_with_no_facts_over_a_vectored_store_resolves_no_embedder(memhome, resolver):
    store = plug._chat_store()
    store.add(MemoryRecord(text="User is named Sam", source="user", importance=1.0),
              embed_fn=_FakeEmbedder().embed)
    assert not store.lacks_vector()
    out = plug.synthesize_memory(_stub())
    assert out["added"] == 0
    assert resolver.calls == 0


def test_pass_whose_only_candidate_is_a_duplicate_resolves_no_embedder(memhome, resolver):
    store = plug._chat_store()
    store.add(MemoryRecord(text="User is named Sam", source="user", importance=1.0),
              embed_fn=_FakeEmbedder().embed)
    out = plug.synthesize_memory(_stub(facts=["User is named Sam"]))
    assert out["added"] == 0
    assert resolver.calls == 0


def test_pass_that_saves_a_fact_resolves_the_embedder_and_stores_a_vector(memhome, resolver):
    out = plug.synthesize_memory(_stub(facts=["User loves sailing"]))
    assert out["added"] == 1
    assert resolver.calls >= 1
    store = plug._chat_store()
    rec = next(r for r in store.all() if "sailing" in r.text)
    assert store._vectors.get(rec.id) == [1.0, 0.0, 0.0]


def test_embedder_is_resolved_outside_the_namespace_lock(memhome, resolver):
    plug.synthesize_memory(_stub(facts=["User loves sailing"]))
    assert resolver.calls >= 1
    assert resolver.lock_held and not any(resolver.lock_held)


def test_pass_with_no_embedder_available_stores_the_fact_without_a_vector(memhome, no_embedder):
    out = plug.synthesize_memory(_stub(facts=["User loves sailing"]))
    assert out["status"] == "ok" and out["added"] == 1
    store = plug._chat_store()
    assert any("sailing" in r.text for r in store.all())
    assert store._vectors == {}
    assert no_embedder.calls >= 1


def test_pass_with_no_embedder_keeps_recall_lexical_without_error(memhome, no_embedder):
    plug.synthesize_memory(_stub(facts=["User loves sailing"]))
    hits = plug._chat_store().recall("sailing", embed_fn=None)
    assert [h.text for h in hits] == ["User loves sailing"]


def _aged_session(home, name, content, mtime):
    p = home / "sessions" / f"{name}.jsonl"
    rows = [{"type": "user", "data": {"content": content}},
            {"type": "llm", "data": {"content": "understood"}}]
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    os.utime(p, (mtime, mtime))
    return p


def test_a_session_still_settling_saves_no_episode_and_resolves_no_embedder(memhome, resolver):
    import time
    for f in (memhome / "sessions").glob("*.jsonl"):
        f.unlink()
    _aged_session(memhome, "fresh", "Let us discuss Rust ownership and lifetimes today",
                  time.time())
    plug.synthesize_memory(_stub(summary="Worked through Rust ownership rules"))
    assert [r for r in plug._chat_store().all() if r.kind == "episodic"] == []
    assert resolver.calls == 0


def test_a_settled_session_without_a_summary_resolves_no_embedder(memhome, resolver):
    for f in (memhome / "sessions").glob("*.jsonl"):
        f.unlink()
    _aged_session(memhome, "old", "Let us discuss Rust ownership and lifetimes today", 1000)
    plug.synthesize_memory(_stub(summary=""))
    assert [r for r in plug._chat_store().all() if r.kind == "episodic"] == []
    assert resolver.calls == 0


def test_a_settled_session_with_a_summary_stores_an_episode_with_a_vector(memhome, resolver):
    for f in (memhome / "sessions").glob("*.jsonl"):
        f.unlink()
    _aged_session(memhome, "old", "Let us discuss Rust ownership and lifetimes today", 1000)
    plug.synthesize_memory(_stub(summary="Worked through Rust ownership rules and lifetime errors"))
    store = plug._chat_store()
    eps = [r for r in store.all() if r.kind == "episodic"]
    assert len(eps) == 1
    assert resolver.calls >= 1
    assert store._vectors.get(eps[0].id) == [1.0, 0.0, 0.0]
    assert not any(resolver.lock_held)


def test_vectorless_records_are_backfilled_by_a_pass_that_finds_nothing_new(memhome, resolver):
    store = plug._chat_store()
    store.add(MemoryRecord(text="User is named Sam", source="user", importance=1.0))
    assert store.lacks_vector()
    plug.synthesize_memory(_stub())
    assert resolver.calls >= 1
    assert not plug._chat_store().lacks_vector()


class TestDeferredEmbedContract:
    def test_resolves_once_on_first_call(self):
        n = []

        def resolve():
            n.append(1)
            return lambda texts: [[0.5] for _ in texts]
        d = DeferredEmbed(resolve)
        assert not d.resolved and not d.unavailable and n == []
        assert d(["a"]) == [[0.5]]
        assert d(["b"]) == [[0.5]]
        assert n == [1]

    def test_unavailable_raises_a_dedicated_error_and_is_remembered(self):
        n = []
        d = DeferredEmbed(lambda: n.append(1))
        with pytest.raises(EmbedderUnavailable):
            d(["a"])
        with pytest.raises(EmbedderUnavailable):
            d(["a"])
        assert d.unavailable and n == [1]

    def test_store_writes_without_a_vector_when_unavailable(self, tmp_path):
        from localm.memory import open_store
        store = open_store(None, "chat", "", root=tmp_path)
        d = DeferredEmbed(lambda: None)
        rec = store.add(MemoryRecord(text="User loves sailing"), embed_fn=d)
        assert rec.id not in store._vectors
        assert store.semantic_nearest("sailing", [rec], d) == (-1, 0.0)

    def test_vector_status_does_not_resolve_and_does_not_misreport(self, tmp_path):
        from localm.memory import open_store
        store = open_store(None, "chat", "", root=tmp_path)
        n = []
        d = DeferredEmbed(lambda: n.append(1))
        assert store._vector_status(d) == (False, "no_vectors")
        assert n == []
        with pytest.raises(EmbedderUnavailable):
            d(["x"])
        assert store._vector_status(d) == (False, "no_embedder")
        assert store._vector_status(None) == (False, "no_embedder")
