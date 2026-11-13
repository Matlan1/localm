# SPDX-License-Identifier: AGPL-3.0-or-later
"""The conversation store's revision check, per-chat parameters and the count
of chats left on disk in privacy mode."""

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from localm.plugins.builtin.chat import plug


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(plug, "_home", lambda: tmp_path)
    state = {"persist": True}
    monkeypatch.setattr(plug, "_persist_enabled", lambda: state["persist"])
    import localm.inference.http_server as hs
    monkeypatch.setattr(hs, "memory_principal", lambda request: None)
    app = FastAPI()
    app.include_router(plug._router)
    return TestClient(app), tmp_path / "chats", state


def _conv(text="hi", **kw):
    return {"title": "t", "messages": [{"role": "user", "content": text}], **kw}


def test_each_save_returns_the_next_revision(store):
    client, _, _ = store
    r1 = client.put("/api/conversations/c1", json=_conv(base_rev=None))
    assert r1.status_code == 200 and r1.json()["rev"] == 1
    r2 = client.put("/api/conversations/c1", json=_conv("two", base_rev=1))
    assert r2.status_code == 200 and r2.json()["rev"] == 2
    assert client.get("/api/conversations/c1").json()["rev"] == 2
    meta = client.get("/api/conversations?meta=true").json()["conversations"]
    assert [(c["id"], c["rev"]) for c in meta] == [("c1", 2)]


def test_a_stale_tab_does_not_overwrite_the_newer_turns(store):
    client, chats, _ = store
    client.put("/api/conversations/c1", json=_conv("first", base_rev=None))
    client.put("/api/conversations/c1", json=_conv("second tab turn", base_rev=1))
    stale = client.put("/api/conversations/c1", json=_conv("first tab turn", base_rev=1))
    assert stale.status_code == 409
    assert stale.json()["detail"] == {"error": "stale", "rev": 2}
    stored = json.loads((chats / "c1.json").read_text(encoding="utf-8"))
    assert stored["messages"][0]["content"] == "second tab turn"
    assert stored["rev"] == 2


def test_a_save_of_a_deleted_chat_does_not_bring_it_back(store):
    client, chats, _ = store
    client.put("/api/conversations/c1", json=_conv(base_rev=None))
    assert client.delete("/api/conversations/c1").json()["status"] == "deleted"
    r = client.put("/api/conversations/c1", json=_conv("late", base_rev=1))
    assert r.status_code == 409
    assert r.json()["detail"] == {"error": "stale", "rev": None}
    assert not (chats / "c1.json").exists()


def test_a_create_does_not_replace_a_stored_chat(store):
    client, _, _ = store
    client.put("/api/conversations/c1", json=_conv("kept", base_rev=None))
    r = client.put("/api/conversations/c1", json=_conv("new", base_rev=None))
    assert r.status_code == 409 and r.json()["detail"]["rev"] == 1
    assert client.get("/api/conversations/c1").json()["messages"][0]["content"] == "kept"


def test_a_file_saved_without_a_revision_counts_as_revision_zero(store):
    client, chats, _ = store
    chats.mkdir(parents=True)
    (chats / "old.json").write_text(json.dumps(_conv("legacy")), encoding="utf-8")
    assert client.get("/api/conversations/old").json()["rev"] == 0
    assert client.get("/api/conversations?meta=true").json()["conversations"][0]["rev"] == 0
    r = client.put("/api/conversations/old", json=_conv("next", base_rev=0))
    assert r.status_code == 200 and r.json()["rev"] == 1


def test_a_save_without_base_rev_is_not_checked(store):
    client, _, _ = store
    client.put("/api/conversations/c1", json=_conv(base_rev=None))
    client.put("/api/conversations/c1", json=_conv(base_rev=1))
    r = client.put("/api/conversations/c1", json=_conv("api client"))
    assert r.status_code == 200 and r.json()["rev"] == 3


def test_per_chat_parameters_are_stored_with_the_chat(store):
    client, _, _ = store
    params = {"system": "You are terse.", "temperature": "0.3", "seed": "42"}
    client.put("/api/conversations/c1", json=_conv(params=params, base_rev=None))
    assert client.get("/api/conversations/c1").json()["params"] == params


def test_parameters_that_are_not_plain_values_are_not_stored(store):
    client, _, _ = store
    params = {"system": "ok", "nested": {"x": 1}, "flag": True, "list": [1]}
    client.put("/api/conversations/c1", json=_conv(params=params))
    assert client.get("/api/conversations/c1").json()["params"] == {"system": "ok"}


def test_a_save_without_params_keeps_the_stored_ones(store):
    client, _, _ = store
    client.put("/api/conversations/c1", json=_conv(params={"system": "kept"}))
    client.put("/api/conversations/c1", json=_conv("api client"))
    assert client.get("/api/conversations/c1").json()["params"] == {"system": "kept"}
    client.put("/api/conversations/c1", json=_conv(params={}))
    assert client.get("/api/conversations/c1").json()["params"] == {}


def test_the_revision_check_sees_a_file_changed_by_another_writer(store):
    client, chats, _ = store
    client.put("/api/conversations/c1", json=_conv(base_rev=None))
    client.get("/api/conversations?meta=true")
    data = json.loads((chats / "c1.json").read_text(encoding="utf-8"))
    data["rev"] = 7
    data["title"] = "changed by another process"
    (chats / "c1.json").write_text(json.dumps(data), encoding="utf-8")
    r = client.put("/api/conversations/c1", json=_conv(base_rev=1))
    assert r.status_code == 409 and r.json()["detail"]["rev"] == 7


def test_relabel_keeps_the_revision(store):
    client, _, _ = store
    client.put("/api/conversations/c1", json=_conv(pinned_model="old", base_rev=None))
    rewritten, failed = plug.relabel_pinned_model("old", "new")
    assert rewritten == ["c1"] and failed == []
    data = client.get("/api/conversations/c1").json()
    assert data["pinned_model"] == "new" and data["rev"] == 1
    assert client.put("/api/conversations/c1", json=_conv(base_rev=1)).status_code == 200


def test_privacy_mode_counts_the_chats_left_on_disk_and_writes_nothing(store):
    client, chats, state = store
    client.put("/api/conversations/a", json=_conv())
    client.put("/api/conversations/b", json=_conv())
    before = sorted(p.name for p in chats.iterdir())
    state["persist"] = False
    data = client.get("/api/conversations?meta=true").json()
    assert data == {"enabled": False, "conversations": [], "stored": 2}
    assert sorted(p.name for p in chats.iterdir()) == before


def test_privacy_mode_with_no_store_counts_zero(store):
    client, chats, state = store
    state["persist"] = False
    assert client.get("/api/conversations").json()["stored"] == 0
    assert not chats.exists()
