# SPDX-License-Identifier: AGPL-3.0-or-later
"""The conversation store's revision check, its record of deleted conversations,
per-chat parameters and the count of chats left on disk in privacy mode."""

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from localm.plugins.builtin.chat import plug


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(plug, "_home", lambda: tmp_path)
    state = {"persist": True, "principal": None}
    monkeypatch.setattr(plug, "_persist_enabled", lambda: state["persist"])
    import localm.inference.http_server as hs
    monkeypatch.setattr(hs, "memory_principal", lambda request: state["principal"])
    app = FastAPI()
    app.include_router(plug._router)
    return TestClient(app), tmp_path / "chats", state


def _msgs(*texts):
    return [{"role": "user" if i % 2 == 0 else "assistant", "content": t}
            for i, t in enumerate(texts)]


def _put(client, cid, *texts, **kw):
    body = {"title": "t", "messages": _msgs(*texts), **kw}
    return client.put(f"/api/conversations/{cid}", json=body)


def test_every_save_returns_a_new_revision_that_the_reads_show(store):
    client, _, _ = store
    r1 = _put(client, "c1", "a", base_rev=None)
    r2 = _put(client, "c1", "a", "b", base_rev=r1.json()["rev"])
    assert r1.status_code == r2.status_code == 200
    assert r1.json()["rev"] != r2.json()["rev"]
    rev = r2.json()["rev"]
    assert client.get("/api/conversations/c1").json()["rev"] == rev
    assert [c["rev"] for c in client.get("/api/conversations?meta=true").json()["conversations"]] == [rev]
    assert [c["rev"] for c in client.get("/api/conversations").json()["conversations"]] == [rev]


def test_a_stale_save_writes_nothing_and_names_the_stored_revision(store):
    client, chats, _ = store
    r1 = _put(client, "c1", "a", base_rev=None).json()["rev"]
    r2 = _put(client, "c1", "a", "from tab two", base_rev=r1).json()["rev"]
    stale = _put(client, "c1", "a", "from tab one", base_rev=r1)
    assert stale.status_code == 409
    detail = stale.json()["detail"]
    assert (detail["error"], detail["rev"]) == ("stale", r2)
    assert detail["row"]["id"] == "c1" and detail["row"]["n_messages"] == 2
    stored = json.loads((chats / "c1.json").read_text(encoding="utf-8"))
    assert stored["messages"][1]["content"] == "from tab two"


def test_a_create_does_not_replace_a_stored_chat(store):
    client, _, _ = store
    rev = _put(client, "c1", "kept", base_rev=None).json()["rev"]
    r = _put(client, "c1", "new", base_rev=None)
    assert r.status_code == 409 and r.json()["detail"]["rev"] == rev
    assert client.get("/api/conversations/c1").json()["messages"][0]["content"] == "kept"


@pytest.mark.parametrize("body", [{}, {"base_rev": None}], ids=["no base_rev", "null base_rev"])
def test_a_deleted_chat_is_not_stored_again(store, body):
    client, chats, _ = store
    rev = _put(client, "c1", "a", base_rev=None).json()["rev"]
    assert client.delete("/api/conversations/c1").json()["status"] == "deleted"
    for r in (_put(client, "c1", "late", **body), _put(client, "c1", "late", base_rev=rev)):
        assert r.status_code == 409
        assert r.json()["detail"]["rev"] is None and r.json()["detail"]["row"] is None
    assert not (chats / "c1.json").exists()
    assert _put(client, "c2", "another").status_code == 200


def test_the_record_of_deleted_chats_holds_ids_only_and_no_conversation(store):
    client, chats, _ = store
    _put(client, "c1", "a secret sentence", title="a secret title")
    client.delete("/api/conversations/c1")
    assert (chats / "deleted-ids.txt").read_text(encoding="utf-8") == "c1\n"
    assert client.get("/api/conversations").json()["conversations"] == []
    assert client.get("/api/conversations?meta=true").json()["unreadable"] == []


def test_the_record_of_deleted_chats_is_bounded_and_keeps_the_newest(store, monkeypatch):
    client, chats, _ = store
    monkeypatch.setattr(plug, "_DELETED_MAX", 3)
    for i in range(5):
        _put(client, f"c{i}", "x")
        client.delete(f"/api/conversations/c{i}")
    assert (chats / "deleted-ids.txt").read_text(encoding="utf-8").split() == ["c2", "c3", "c4"]


def test_a_principal_s_deletions_are_its_own(store):
    client, _, state = store
    state["principal"] = "keyhash1"
    _put(client, "c1", "a")
    client.delete("/api/conversations/c1")
    assert _put(client, "c1", "again").status_code == 409
    state["principal"] = None
    assert _put(client, "c1", "the owner's").status_code == 200


def test_a_client_without_a_revision_may_add_but_not_remove(store):
    client, _, _ = store
    _put(client, "c1", "a", "b")
    assert _put(client, "c1", "a", "b", "c").status_code == 200
    r = _put(client, "c1", "a", "other")
    assert r.status_code == 409 and r.json()["detail"]["rev"]
    stored = client.get("/api/conversations/c1").json()["messages"]
    assert [m["content"] for m in stored] == ["a", "b", "c"]


def test_a_client_without_a_revision_may_park_and_compact_messages(store):
    client, _, _ = store
    _put(client, "c1", "a", "b", "c")
    parked = client.put("/api/conversations/c1", json={
        "title": "t", "messages": _msgs("a", "b2"),
        "branches": [{"parent": "x", "tails": [_msgs("b", "c"), None], "current": 1}]})
    assert parked.status_code == 409, "the roles of the parked tail differ from the stored ones"
    parked = client.put("/api/conversations/c1", json={
        "title": "t", "messages": _msgs("a"),
        "branches": [{"parent": "x", "current": 1, "tails": [
            [{"role": "assistant", "content": "b"}, {"role": "user", "content": "c"}], None]}]})
    assert parked.status_code == 200
    _put(client, "c2", "a", "b")
    compacted = client.put("/api/conversations/c2", json={"title": "t", "messages": [
        {"role": "user", "content": "[Conversation summary]", "bridge": True,
         "compacted": _msgs("a", "b")}]})
    assert compacted.status_code == 200


def test_a_search_result_migrated_from_the_old_format_counts_as_kept(store):
    client, _, _ = store
    client.put("/api/conversations/c1", json={"title": "t", "messages": [
        {"role": "user", "web": True, "content": "[web_search results]\nfound"}]})
    r = client.put("/api/conversations/c1", json={"title": "t", "messages": [
        {"kind": "tool", "tool": "search", "status": "done", "query": "",
         "text": "[web_search results]\nfound"}]})
    assert r.status_code == 200


def test_a_file_saved_before_revisions_is_read_and_taken_over(store):
    client, chats, _ = store
    chats.mkdir(parents=True)
    (chats / "old.json").write_text(json.dumps({"title": "t", "messages": _msgs("a")}),
                                    encoding="utf-8")
    rev = client.get("/api/conversations/old").json()["rev"]
    assert rev == client.get("/api/conversations?meta=true").json()["conversations"][0]["rev"]
    r = _put(client, "old", "a", "b", base_rev=rev)
    assert r.status_code == 200 and r.json()["rev"] != rev


def test_an_unreadable_stored_file_is_not_replaced(store):
    client, chats, _ = store
    chats.mkdir(parents=True)
    (chats / "c1.json").write_text("{broken", encoding="utf-8")
    assert _put(client, "c1", "a").status_code == 409
    assert _put(client, "c1", "a", base_rev="abc").status_code == 409
    assert (chats / "c1.json").read_text(encoding="utf-8") == "{broken"


def test_the_check_sees_a_file_changed_by_another_writer(store):
    client, chats, _ = store
    rev = _put(client, "c1", "a", base_rev=None).json()["rev"]
    client.get("/api/conversations?meta=true")
    data = json.loads((chats / "c1.json").read_text(encoding="utf-8"))
    data["rev"], data["title"] = "elsewhere", "changed by another process"
    (chats / "c1.json").write_text(json.dumps(data), encoding="utf-8")
    r = _put(client, "c1", "a", base_rev=rev)
    assert r.status_code == 409 and r.json()["detail"]["rev"] == "elsewhere"


def test_a_model_rename_keeps_the_revision(store):
    client, _, _ = store
    rev = _put(client, "c1", "a", base_rev=None, pinned_model="old").json()["rev"]
    assert plug.relabel_pinned_model("old", "new") == (["c1"], [])
    data = client.get("/api/conversations/c1").json()
    assert (data["pinned_model"], data["rev"]) == ("new", rev)
    assert _put(client, "c1", "a", "b", base_rev=rev).status_code == 200


def test_drawer_values_are_stored_kept_and_cleared(store):
    client, _, _ = store
    params = {"system": "You are terse.", "temperature": "0.3", "seed": 42}
    _put(client, "c1", "a", params={**params, "flag": True, "nested": {"x": 1}, "list": [1]})
    assert client.get("/api/conversations/c1").json()["params"] == params
    _put(client, "c1", "a", "b")
    assert client.get("/api/conversations/c1").json()["params"] == params, "left out: kept"
    _put(client, "c1", "a", "b", params={})
    assert client.get("/api/conversations/c1").json()["params"] == {}


def test_privacy_mode_counts_the_chats_left_on_disk_and_writes_nothing(store):
    client, chats, state = store
    _put(client, "a", "x")
    _put(client, "b", "x")
    before = sorted(p.name for p in chats.iterdir())
    state["persist"] = False
    assert client.get("/api/conversations?meta=true").json() == {
        "enabled": False, "conversations": [], "stored": 2}
    assert _put(client, "a", "x", base_rev=None).status_code == 403
    assert client.delete("/api/conversations/a").status_code == 403
    assert sorted(p.name for p in chats.iterdir()) == before


def test_privacy_mode_with_no_store_counts_zero_and_creates_nothing(store):
    client, chats, state = store
    state["persist"] = False
    assert client.get("/api/conversations").json()["stored"] == 0
    assert not chats.exists()
