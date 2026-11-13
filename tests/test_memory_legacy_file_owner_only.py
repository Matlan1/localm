# SPDX-License-Identifier: AGPL-3.0-or-later
"""The legacy chat-memory.md belongs to the owner: a scoped key's memory never
imports it, shows it or recalls it, while the owner's still does. The memory
dialog's bulk edit matches facts with case and spacing ignored, as every other
way of adding a fact does."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from localm.plugins.builtin.memory import plug

_SCOPED = "0123456789abcdef" * 4
_LEGACY = "- the garage code is 4711\n- prefers green tea\n"


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setattr(plug, "_home", lambda: tmp_path)
    monkeypatch.setenv("LOCALM_MODE", "log")
    monkeypatch.setattr(plug, "_embed_fn", lambda: None)
    (tmp_path / "chat-memory.md").write_text(_LEGACY, encoding="utf-8")
    return tmp_path


def _client(monkeypatch, principal):
    monkeypatch.setattr(plug, "_request_principal", lambda request: principal)
    app = FastAPI()
    app.include_router(plug._router)
    return TestClient(app)


def test_a_scoped_key_never_sees_or_imports_the_owners_legacy_file(home, monkeypatch):
    r = _client(monkeypatch, _SCOPED).get("/api/memory")

    assert r.status_code == 200, r.text
    assert r.json()["items"] == []
    assert "4711" not in r.json()["text"]
    scoped = plug._chat_store(_SCOPED)
    assert scoped.all() == []
    assert not plug._legacy_marker(scoped).exists()
    assert plug._legacy_bullets(scoped) == []


def test_the_owner_still_imports_the_legacy_file(home, monkeypatch):
    r = _client(monkeypatch, None).get("/api/memory")

    assert r.status_code == 200, r.text
    assert sorted(i["text"] for i in r.json()["items"]) == [
        "prefers green tea", "the garage code is 4711"]
    assert plug._legacy_bullets(plug._chat_store()) == [
        "the garage code is 4711", "prefers green tea"]


def test_a_scoped_keys_writes_do_not_pull_the_legacy_file_in(home, monkeypatch):
    client = _client(monkeypatch, _SCOPED)
    assert client.post("/api/memory/append", json={"text": "likes chess"}).status_code == 200

    assert [r.text for r in plug._chat_store(_SCOPED).all()] == ["likes chess"]


def test_the_bulk_edit_matches_facts_ignoring_case_and_spacing(home, monkeypatch):
    client = _client(monkeypatch, None)
    base = client.get("/api/memory").json()["text"]
    before = {i["text"]: i["id"] for i in client.get("/api/memory").json()["items"]}

    r = client.put("/api/memory", json={
        "text": "- Prefers GREEN  tea\n- prefers green tea\n- the garage code is 4711\n",
        "base": base})

    assert r.status_code == 200, r.text
    items = client.get("/api/memory").json()["items"]
    assert sorted(i["text"] for i in items) == ["Prefers GREEN  tea", "the garage code is 4711"]
    kept = next(i for i in items if i["text"] == "Prefers GREEN  tea")
    assert kept["id"] == before["prefers green tea"], "the edited fact keeps its record"
