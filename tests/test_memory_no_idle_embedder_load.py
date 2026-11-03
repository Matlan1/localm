# SPDX-License-Identifier: AGPL-3.0-or-later
"""The embedding model is loaded only when there is something to embed.

A GUI page refresh on a fresh install (empty memory, no legacy file) and a chat
turn over an empty memory must resolve no embedder; a legacy chat-memory.md and
a non-empty memory still do.
"""

from __future__ import annotations

import types

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from localm.memory import MemoryRecord
from localm.plugins.builtin.memory import plug
from localm.plugins.builtin.rag import plug as rag_plug


class _Embedder:
    def embed(self, texts):
        return [[1.0, 0.0, 0.0] for _ in texts]


@pytest.fixture
def loads(monkeypatch):
    """Counts every get_embedder() call made through the module attribute."""
    calls = []

    def _fake_get_embedder(*args, **kwargs):
        calls.append(1)
        return _Embedder()

    monkeypatch.setattr("localm.inference.embedder.get_embedder", _fake_get_embedder)
    return calls


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setattr(plug, "_home", lambda: tmp_path)
    monkeypatch.setenv("LOCALM_MODE", "log")
    return tmp_path


@pytest.fixture
def client(home):
    app = FastAPI()
    app.include_router(plug._router)
    app.include_router(rag_plug._router)
    return TestClient(app)


def _ctx():
    return types.SimpleNamespace(model_id="", principal=None, stream=False,
                                 request_id="r1", state={}, scopes=())


def test_counter_fires_on_a_real_embed(home, loads):
    plug._chat_store().add(MemoryRecord(text="x", source="user"),
                           embed_fn=plug._embed_fn())
    assert len(loads) == 1


def test_get_memory_on_fresh_home_loads_no_embedder(client, home, loads):
    r = client.get("/api/memory")
    assert r.status_code == 200, r.text
    assert r.json()["items"] == []
    assert loads == []
    assert plug._chat_store().path.with_suffix(".legacy-imported").exists()


def test_boot_gets_on_fresh_home_load_no_embedder(client, loads):
    for path in ("/api/memory", "/api/rag/collections", "/api/memory"):
        assert client.get(path).status_code == 200, path
    assert loads == []


def test_get_memory_with_legacy_file_still_migrates(client, home, loads):
    (home / "chat-memory.md").write_text("- likes oolong\n- drives a tram\n",
                                         encoding="utf-8")
    r = client.get("/api/memory")
    assert r.status_code == 200, r.text
    assert sorted(i["text"] for i in r.json()["items"]) == [
        "drives a tram", "likes oolong"]
    assert len(loads) == 1
    assert plug._chat_store().path.with_suffix(".legacy-imported").exists()


def test_already_imported_legacy_bullets_load_no_embedder(client, home, loads):
    (home / "chat-memory.md").write_text("- likes oolong\n", encoding="utf-8")
    plug._chat_store().add(MemoryRecord(text="likes oolong", source="user"),
                           embed_fn=None)
    assert client.get("/api/memory").status_code == 200
    assert loads == []


def test_inlet_over_empty_memory_loads_no_embedder(home, loads):
    messages = [{"role": "user", "content": "what do I like to drink?"}]
    assert plug._memory_inlet(messages, _ctx()) is None
    assert loads == []


def test_inlet_over_existing_memory_loads_the_embedder(home, loads):
    plug._chat_store().add(MemoryRecord(text="I drink oolong tea", source="user",
                                        importance=0.9), embed_fn=None)
    messages = [{"role": "user", "content": "what tea do I drink?"}]
    plug._memory_inlet(messages, _ctx())
    assert len(loads) == 1
