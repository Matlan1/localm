# SPDX-License-Identifier: AGPL-3.0-or-later
"""Every chat-generating route requires the ``chat`` scope.

Driven over HTTP against the real app with a real scoped key: a key without
``chat`` is refused with 403 before the handler runs, and a key holding ``chat``
passes the gate (it then gets whatever the handler answers with no model
loaded, never the scope refusal).
"""

import pytest
from fastapi.testclient import TestClient

from localm import auth, scopes
from localm.inference.http_server import create_app

CHAT_ROUTES = [
    ("/v1/chat/completions",
     {"model": "x", "messages": [{"role": "user", "content": "hi"}]}),
    ("/v1/completions", {"model": "x", "prompt": "hi"}),
    ("/v1/messages",
     {"model": "x", "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]}),
    ("/v1/messages/count_tokens",
     {"model": "x", "messages": [{"role": "user", "content": "hi"}]}),
    ("/api/chat", {"model": "x", "messages": [{"role": "user", "content": "hi"}]}),
    ("/api/generate", {"model": "x", "prompt": "hi"}),
    ("/v1/responses", {"model": "x", "input": "hi"}),
]
_IDS = [p for p, _ in CHAT_ROUTES]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.delenv("LOCALM_REQUIRE_AUTH", raising=False)


def _client():
    auth.regenerate_key()
    app = create_app(None)
    app.state.bind_host = "127.0.0.1"
    return TestClient(app)


def _bearer(key):
    return {"Authorization": f"Bearer {key}"}


@pytest.mark.parametrize("path,body", CHAT_ROUTES, ids=_IDS)
def test_a_key_without_the_chat_scope_is_refused(path, body):
    c = _client()
    key = auth.create_key("images-only", [scopes.IMAGE])["key"]

    r = c.post(path, json=body, headers=_bearer(key))

    assert r.status_code == 403, (path, r.status_code, r.text)
    assert "chat" in r.text


@pytest.mark.parametrize("path,body", CHAT_ROUTES, ids=_IDS)
def test_a_key_with_the_chat_scope_passes_the_gate(path, body):
    c = _client()
    key = auth.create_key("chatter", [scopes.CHAT])["key"]

    r = c.post(path, json=body, headers=_bearer(key))

    assert r.status_code not in (401, 403), (path, r.status_code, r.text)
