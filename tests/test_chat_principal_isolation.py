# SPDX-License-Identifier: AGPL-3.0-or-later
"""Conversations and personas are stored per caller, and written under a lock."""

import json
import threading

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from localm.plugins.builtin.chat import plug


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(plug, "_home", lambda: tmp_path)
    monkeypatch.setattr(plug, "_persist_enabled", lambda: True)
    import localm.inference.http_server as hs
    monkeypatch.setattr(
        hs, "memory_principal",
        lambda request: request.headers.get("x-test-principal") or None)
    app = FastAPI()
    app.include_router(plug._router)
    return TestClient(app)


OWNER = {}
OTHER = {"x-test-principal": "keyhashother"}


def _conv(title="t", n=1, **kw):
    return {"title": title, "messages": [{"role": "user", "content": str(i)}
                                         for i in range(n)], **kw}


def test_second_key_cannot_read_list_overwrite_or_delete_owner_conversation(client, tmp_path):
    assert client.put("/api/conversations/c1", json=_conv("owner"), headers=OWNER).status_code == 200

    assert client.get("/api/conversations/c1", headers=OTHER).status_code == 404
    assert client.get("/api/conversations", headers=OTHER).json()["conversations"] == []

    assert client.put("/api/conversations/c1", json=_conv("evil"), headers=OTHER).status_code == 200
    assert client.get("/api/conversations/c1", headers=OWNER).json()["title"] == "owner"
    assert client.get("/api/conversations/c1", headers=OTHER).json()["title"] == "evil"

    assert client.delete("/api/conversations/c1", headers=OTHER).json()["status"] == "deleted"
    assert client.get("/api/conversations/c1", headers=OWNER).json()["title"] == "owner"
    assert [c["id"] for c in client.get("/api/conversations", headers=OWNER).json()["conversations"]] == ["c1"]


def test_owner_keeps_the_shared_chat_file(client, tmp_path):
    client.put("/api/conversations/c1", json=_conv("owner"), headers=OWNER)
    assert (tmp_path / "chats" / "c1.json").is_file()


def test_second_key_cannot_read_overwrite_or_delete_owner_persona(client, tmp_path):
    assert client.put("/api/prompts/p", json={"system": "good"}, headers=OWNER).status_code == 200

    assert client.get("/api/prompts", headers=OTHER).json()["prompts"] == []
    assert client.delete("/api/prompts/p", headers=OTHER).status_code == 404

    assert client.put("/api/prompts/p", json={"system": "evil"}, headers=OTHER).status_code == 200
    owner = {e["name"]: e["system"] for e in client.get("/api/prompts", headers=OWNER).json()["prompts"]}
    assert owner == {"p": "good"}
    assert (tmp_path / "prompts.json").is_file()


def test_relabel_reaches_owner_and_principal_conversations(client):
    client.put("/api/conversations/a", json=_conv(pinned_model="old"), headers=OWNER)
    client.put("/api/conversations/b", json=_conv(pinned_model="old"), headers=OTHER)
    rewritten, failed = plug.relabel_pinned_model("old", "new")
    assert sorted(rewritten) == ["a", "b"] and failed == []
    assert client.get("/api/conversations/b", headers=OTHER).json()["pinned_model"] == "new"


def test_put_and_relabel_threads_lose_no_message(client):
    rounds = 60
    client.put("/api/conversations/c", json=_conv(n=0, pinned_model="old"), headers=OWNER)
    errors = []

    def writer():
        try:
            for i in range(1, rounds + 1):
                r = client.put("/api/conversations/c",
                               json=_conv(n=i, pinned_model="old", updated_at=i), headers=OWNER)
                assert r.status_code == 200
        except Exception as e:
            errors.append(e)

    def relabeller():
        try:
            for _ in range(rounds * 3):
                plug.relabel_pinned_model("old", "tmp")
                plug.relabel_pinned_model("tmp", "old")
        except Exception as e:
            errors.append(e)

    t1, t2 = threading.Thread(target=writer), threading.Thread(target=relabeller)
    t1.start(); t2.start(); t1.join(); t2.join()
    assert errors == []
    final = client.get("/api/conversations/c", headers=OWNER).json()
    assert len(final["messages"]) == rounds


def test_relabel_skips_a_file_changed_since_it_was_read(client, tmp_path, monkeypatch):
    client.put("/api/conversations/c", json=_conv(n=1, pinned_model="old", updated_at=5), headers=OWNER)
    path = tmp_path / "chats" / "c.json"
    real = plug._updated_at_of

    def external_writer_lands_first(p):
        path.write_text(json.dumps({"id": "c", "pinned_model": "old", "updated_at": 99,
                                    "messages": [1, 2, 3]}), encoding="utf-8")
        return real(p)

    monkeypatch.setattr(plug, "_updated_at_of", external_writer_lands_first)
    rewritten, failed = plug.relabel_pinned_model("old", "new")
    assert rewritten == [] and failed == ["c"]
    kept = json.loads(path.read_text(encoding="utf-8"))
    assert kept["pinned_model"] == "old" and kept["messages"] == [1, 2, 3]
