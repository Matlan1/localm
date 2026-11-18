# SPDX-License-Identifier: AGPL-3.0-or-later
"""The conversation index answers a content search, so the sidebar finds chats
whose bodies the browser has not loaded."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from localm.plugins.builtin.chat import plug


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(plug, "_home", lambda: tmp_path)
    monkeypatch.setattr(plug, "_persist_enabled", lambda: True)
    import localm.inference.http_server as hs
    monkeypatch.setattr(hs, "memory_principal", lambda request: None)
    app = FastAPI()
    app.include_router(plug._router)
    return TestClient(app)


def _put(client, cid, title, content, at):
    r = client.put("/api/conversations/" + cid, json={
        "title": title, "updated_at": at,
        "messages": [{"role": "user", "content": content}]})
    assert r.status_code == 200


def _search(client, q):
    r = client.get("/api/conversations", params={"meta": "true", "q": q})
    assert r.status_code == 200
    return {c["id"]: c for c in r.json()["conversations"]}


def test_search_matches_message_text_and_titles_case_insensitively(client):
    _put(client, "a", "Fruit", [{"type": "text", "text": "I like Cherry pie a lot"}], 1)
    _put(client, "b", "Cherries", "nothing here", 2)
    _put(client, "c", "Other", "no match", 3)
    rows = _search(client, "CHER")
    assert set(rows) == {"a", "b"}
    assert "cherry pie" in rows["a"]["snippet"].lower()
    assert "snippet" not in rows["b"]


def test_search_without_a_term_lists_everything_and_leaves_bodies_out(client):
    _put(client, "a", "One", "alpha", 1)
    _put(client, "b", "Two", "beta", 2)
    rows = _search(client, "")
    assert set(rows) == {"a", "b"}
    assert all("messages" not in r and "snippet" not in r for r in rows.values())
