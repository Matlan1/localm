# SPDX-License-Identifier: AGPL-3.0-or-later
"""The conversation store's revision check, its record of deleted conversations,
per-chat parameters and the count of chats left on disk in privacy mode."""

import inspect
import json
import threading
from pathlib import Path

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
        detail = r.json()["detail"]
        assert detail["rev"] is None and detail["row"] is None and detail["deleted"] is True
    assert not (chats / "c1.json").exists()
    assert _put(client, "c2", "another").status_code == 200


def test_the_record_of_deleted_chats_holds_ids_only_and_no_conversation(store):
    client, chats, _ = store
    _put(client, "c1", "a secret sentence", title="a secret title")
    client.delete("/api/conversations/c1")
    assert (chats / "deleted-ids.txt").read_text(encoding="utf-8") == "c1\n"
    assert client.get("/api/conversations").json()["conversations"] == []
    assert client.get("/api/conversations?meta=true").json()["unreadable"] == []


def _deleted(chats):
    return (chats / "deleted-ids.txt").read_text(encoding="utf-8").split()


def test_a_delete_of_a_chat_that_is_not_stored_is_recorded_too(store):
    client, chats, _ = store
    assert client.delete("/api/conversations/onitsway").json()["status"] == "absent"
    assert _deleted(chats) == ["onitsway"]
    assert _put(client, "onitsway", "the late first save").status_code == 409
    assert client.get("/api/conversations/onitsway").status_code == 410
    assert client.get("/api/conversations/never").status_code == 404


def test_the_record_of_deleted_chats_is_appended_to(store, monkeypatch):
    client, chats, _ = store
    client.delete("/api/conversations/a")
    monkeypatch.setattr(plug, "atomic_write", lambda *a, **k: pytest.fail("rewritten"))
    client.delete("/api/conversations/b")
    assert _deleted(chats) == ["a", "b"]


def test_the_record_of_deleted_chats_is_cut_to_the_newest_when_it_is_long(store, monkeypatch):
    client, chats, _ = store
    monkeypatch.setattr(plug, "_DELETED_MAX", 3)
    for i in range(6):
        client.delete(f"/api/conversations/c{i}")
    assert _deleted(chats) == [f"c{i}" for i in range(6)]
    client.delete("/api/conversations/c6")
    assert _deleted(chats) == ["c4", "c5", "c6"]


def test_a_bad_byte_or_a_torn_line_loses_no_other_deletion(store):
    client, chats, _ = store
    for i in range(5):
        client.delete(f"/api/conversations/d{i}")
    path = chats / "deleted-ids.txt"
    raw = path.read_bytes()
    path.write_bytes(raw[:3] + b"\xff\xfe garbage\n" + raw[3:] + b"torn")
    client.delete("/api/conversations/e1")
    for cid in ("d0", "d1", "d4", "e1"):
        assert _put(client, cid, "again").status_code == 409, cid
    assert path.read_bytes().endswith(b"torn\ne1\n")


def test_a_list_that_cannot_be_read_refuses_the_save_that_depends_on_it(store, monkeypatch):
    client, chats, _ = store
    _put(client, "kept", "a")
    client.delete("/api/conversations/gone")
    before = (chats / "deleted-ids.txt").read_bytes()
    real = Path.read_bytes

    def refuse(self):
        if self.name == "deleted-ids.txt":
            raise PermissionError("locked")
        return real(self)

    monkeypatch.setattr(Path, "read_bytes", refuse)
    monkeypatch.setattr(plug, "_transient_backoff", lambda attempt: None)
    assert _put(client, "gone", "late").status_code == 503
    assert _put(client, "new1", "a").status_code == 503
    assert client.get("/api/conversations/gone").status_code == 503
    assert _put(client, "kept", "a", "b").status_code == 200, "a stored chat needs no list"
    assert client.delete("/api/conversations/kept").json()["status"] == "deleted"
    monkeypatch.undo()
    assert (chats / "deleted-ids.txt").read_bytes().startswith(before)


def test_a_deletion_that_cannot_be_recorded_deletes_nothing(store, monkeypatch):
    client, chats, _ = store
    _put(client, "c1", "a")

    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(plug, "_remember_deleted", fail)
    assert client.delete("/api/conversations/c1").status_code == 500
    assert (chats / "c1.json").is_file()


def _refuse(monkeypatch, method, times):
    """Path.<method> of a chat file is refused *times* times like a file another
    process has open."""
    real = getattr(Path, method)
    left = [times]

    def call(self, *args, **kwargs):
        if self.suffix == ".json" and left[0]:
            left[0] -= 1
            raise PermissionError("in use")
        return real(self, *args, **kwargs)

    monkeypatch.setattr(Path, method, call)
    monkeypatch.setattr(plug, "_is_transient_permission_error", lambda e: True)
    monkeypatch.setattr(plug, "_transient_backoff", lambda attempt: None)


@pytest.mark.parametrize("method", ["read_text", "unlink", "stat"])
def test_a_file_another_process_holds_for_a_moment_is_waited_for(store, monkeypatch, method):
    client, chats, _ = store
    _put(client, "c1", "a")
    plug._META_CACHE.clear()
    if method != "stat":
        _refuse(monkeypatch, method, 3)
        assert client.get("/api/conversations/c1").status_code == 200
    else:
        _refuse(monkeypatch, method, 1)
    assert client.get("/api/conversations?meta=true").json()["unreadable"] == []
    assert client.delete("/api/conversations/c1").json()["status"] == "deleted"
    assert not (chats / "c1.json").exists()


def test_a_file_that_cannot_be_removed_stays_and_does_not_count_as_deleted(store, monkeypatch):
    client, chats, _ = store
    _put(client, "c1", "a")
    client.delete("/api/conversations/other")
    with monkeypatch.context() as m:
        _refuse(m, "unlink", 100)
        assert client.delete("/api/conversations/c1").status_code == 500
    assert client.get("/api/conversations/c1").status_code == 200
    (chats / "c1.json").unlink()
    assert client.get("/api/conversations/c1").status_code == 404
    assert _put(client, "c1", "a").status_code == 200
    assert client.get("/api/conversations/other").status_code == 410
    assert client.delete("/api/conversations/c1").json()["status"] == "deleted"
    assert client.get("/api/conversations/c1").status_code == 410


@pytest.mark.parametrize("ask", [
    lambda c: c.get("/api/conversations/c1"), lambda c: c.get("/api/conversations?meta=true"),
    lambda c: c.get("/api/conversations"), lambda c: c.get("/api/conversations?meta=true&q=word")])
def test_a_read_waits_for_the_chat_s_lock(store, ask):
    client, chats, _ = store
    _put(client, "c1", "a word")
    plug._META_CACHE.clear()
    got = []
    with plug._LOCKS.get(str(chats / "c1.json")):
        t = threading.Thread(target=lambda: got.append(ask(client).status_code))
        t.start()
        t.join(0.5)
        assert got == []
    t.join(10)
    assert got == [200]


BAD_BODIES = {
    "a text part that is not text": ('"messages":[{"role":"user","content":[{"type":"text","text":5}]}]',
                                     "messages[0].content[0].text is not text"),
    "NaN as the time": ('"updated_at":NaN', "updated_at is not a finite number"),
    "an infinite time": ('"updated_at":1e999', "updated_at is not a finite number"),
    "NaN in a message": ('"messages":[{"role":"user","content":"a","x":NaN}]', "messages[0].x is not a finite number"),
    "NaN as a drawer value": ('"params":{"seed":NaN}', "params.seed is not a finite number"),
    "half a character": ('"messages":[{"role":"user","content":"\\ud800"}]', "messages[0].content is not valid text"),
    "half a character in the title": ('"title":"\\ud800"', "title is not valid text"),
    "half a character in a name": ('"messages":[{"\\ud800":1}]', "a field name in messages[0] is not valid text"),
}


@pytest.mark.parametrize("part, where", BAD_BODIES.values(), ids=BAD_BODIES.keys())
def test_what_a_chat_file_cannot_hold_is_refused_by_name(store, part, where):
    client, chats, _ = store
    r = client.put("/api/conversations/c1", content="{" + part + "}", headers={"Content-Type": "application/json"})
    assert r.status_code == 422, r.text
    assert where in r.json()["detail"]
    assert not chats.exists() or not list(chats.iterdir())


BAD_FILES = {
    "text": '{"title":"t","updated_at":5,"messages":[{"role":"user","content":[{"type":"text","text":5},'
            '{"type":"text","text":"needle"}]}]}',
    "nan": '{"title":"t","updated_at":NaN,"messages":[{"role":"user","content":"needle"}]}',
    "string time": '{"title":"t","updated_at":"later","messages":[]}',
    "surrogate": '{"title":"\\ud800","updated_at":5,"rev":"\\ud800","messages":[{"role":"user","content":"needle \\ud800"}]}',
    "shapes": '{"title":7,"updated_at":5,"folder":[1],"messages":{"a":1},"branches":3,"save_id":{}}',
    "list": "[1, 2]",
}


@pytest.mark.parametrize("text", BAD_FILES.values(), ids=BAD_FILES.keys())
def test_one_bad_file_breaks_no_list_no_search_and_no_other_chat(store, text):
    client, chats, _ = store
    _put(client, "good", "needle here")
    (chats / "bad.json").write_text(text, encoding="utf-8")
    for url in ("?meta=true", "", "?meta=true&q=needle"):
        r = client.get("/api/conversations" + url)
        assert r.status_code == 200, url
        assert "good" in [c["id"] for c in r.json()["conversations"]], url
    assert client.get("/api/conversations/good").status_code == 200
    assert client.get("/api/conversations/bad").status_code in (200, 500)
    assert _put(client, "bad", "x").status_code in (200, 409)
    assert _put(client, "good", "needle here", "more").status_code == 200


def test_the_search_finds_the_text_beside_a_part_that_is_not_text(store):
    client, chats, _ = store
    chats.mkdir(parents=True)
    (chats / "bad.json").write_text(BAD_FILES["text"], encoding="utf-8")
    got = client.get("/api/conversations?meta=true&q=needle").json()
    assert [c["id"] for c in got["conversations"]] == ["bad"] and got["unreadable"] == []


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


def test_a_client_without_a_revision_may_not_drop_a_picture_or_a_repetition(store):
    client, _, _ = store

    def picture(data, **shown):
        return {"title": "t", "messages": [{"role": "user", "content": [
            {"type": "text", "text": "look"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + data}, **shown}]}]}

    client.put("/api/conversations/p", json=picture("AAAA"))
    assert _put(client, "p", "look").status_code == 409
    assert client.put("/api/conversations/p", json=picture("BBBB")).status_code == 409
    assert client.put("/api/conversations/p", json=picture("AAAA", refusedBy=["m"])).status_code == 200
    _put(client, "r", "ok", "x", "ok", "x")
    assert _put(client, "r", "ok", "x").status_code == 409


@pytest.mark.parametrize("branches", [[{"tails": 5}], ["x"], [{"tails": [None, 3]}], [{}]])
def test_branches_of_another_shape_are_counted_as_holding_nothing(store, branches):
    client, _, _ = store
    assert _put(client, "c1", "a", branches=branches).status_code == 200
    assert _put(client, "c1", "a", "b", branches=branches).status_code == 200
    assert _put(client, "c1", "b", branches=branches).status_code == 409


def test_the_live_timeline_s_own_slot_of_a_fork_is_not_counted_twice(store):
    client, _, _ = store
    live = _msgs("a", "b")
    client.put("/api/conversations/c1", json={"title": "t", "messages": live, "branches": [
        {"parent": "x", "current": 0, "tails": [live[1:], [{"role": "assistant", "content": "other"}]]}]})
    r = client.put("/api/conversations/c1", json={"title": "t", "messages": _msgs("a", "b", "c"), "branches": [
        {"parent": "x", "current": 0, "tails": [None, [{"role": "assistant", "content": "other"}]]}]})
    assert r.status_code == 200


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
        "enabled": False, "conversations": [], "stored": 2, **plug._store_owner(None)}
    assert _put(client, "a", "x", base_rev=None).status_code == 403
    assert client.delete("/api/conversations/a").status_code == 403
    assert sorted(p.name for p in chats.iterdir()) == before


def test_privacy_mode_with_no_store_counts_zero_and_creates_nothing(store):
    client, chats, state = store
    state["persist"] = False
    assert client.get("/api/conversations").json()["stored"] == 0
    assert not chats.exists()


HOSTILE_IDS = {
    "parent": "%2E%2E", "parent with a backslash": "..%5Cevil", "drive": "C%3A",
    "drive path": "C%3A%5Cevil", "stream": "c1%3Astream", "device name": "nul",
    "numbered device name": "COM1", "NUL byte": "c1%00", "line break": "c1%0A",
    "two lines": "c1%0Ac2", "long": "x" * 65, "space": "a%20b", "non-ASCII": "sp%C3%A4t",
    "dot": "c1.json",
}


@pytest.mark.parametrize("bad_id", HOSTILE_IDS.values(), ids=HOSTILE_IDS.keys())
def test_a_hostile_id_is_refused_by_every_route_and_touches_nothing(store, bad_id):
    client, chats, _ = store
    _put(client, "c1", "kept")
    def tree():
        return {str(p): p.is_file() and p.read_bytes() for p in chats.parent.rglob("*")}
    before = tree()
    assert str(chats / "c1.json") in before
    url = f"/api/conversations/{bad_id}"
    body = {"title": "t", "messages": _msgs("x")}
    answers = [client.get(url), client.delete(url), client.put(url, json=body),
               client.put(url, json={**body, "base_rev": None}),
               client.put(url, json={**body, "base_rev": "abc"})]
    assert [r.status_code for r in answers] == [400] * 5
    assert tree() == before
    assert _put(client, "c1", "kept", "more").status_code == 200, "c1 is not counted as deleted"


@pytest.mark.parametrize("name", ["..", "C:", "nul", "c1\x00", "c1\n", "x" * 65, "a/b", "a\\b", ""])
def test_the_id_check_itself_refuses_the_same_ids(name):
    with pytest.raises(plug.HTTPException) as e:
        plug._conv_path(name)
    assert e.value.status_code == 400


def test_the_index_names_the_store_and_a_save_for_another_one_is_refused(store):
    client, chats, state = store
    owner = client.get("/api/conversations?meta=true").json()
    assert owner["principal"] == "owner" and owner["instance"]
    state["principal"] = "a" * 64
    other = client.get("/api/conversations").json()
    assert other["principal"] not in ("owner", "a" * 64)
    assert other["instance"] == owner["instance"]
    state["persist"] = False
    assert client.get("/api/conversations").json()["principal"] == other["principal"]
    state["persist"] = True
    for wrong in ({"principal": "owner"}, {"instance": "elsewhere"}):
        r = _put(client, "c1", "a", **wrong)
        assert r.status_code == 403 and r.json()["detail"]["error"] == "owner"
    assert not chats.exists() or not list(chats.rglob("*.json"))
    assert _put(client, "c1", "a", principal=other["principal"],
                instance=other["instance"]).status_code == 200


def test_the_index_row_names_the_save_that_wrote_it(store):
    client, _, _ = store
    rev = _put(client, "c1", "a", save_id="mine").json()["rev"]
    row = client.get("/api/conversations?meta=true").json()["conversations"][0]
    assert (row["save_id"], row["rev"]) == ("mine", rev)
    detail = _put(client, "c1", "a", "b", base_rev="older").json()["detail"]
    assert detail["row"]["save_id"] == "mine"


@pytest.mark.parametrize("name", ["conversations_list", "conversation_get",
                                  "conversation_upsert", "conversation_delete"])
def test_the_store_s_routes_run_off_the_event_loop(name):
    assert not inspect.iscoroutinefunction(getattr(plug, name))
