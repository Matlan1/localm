"""A server bound past loopback refuses requests once its last key is removed.

The startup bind guard refuses a keyless network bind, but a key removed while
the server runs used to return it to open mode, reachable from the network. These
drive a real app through a live key removal on a network bind_host.
"""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from localm import auth
from localm.inference import http_server as hs
from localm.inference.http_server import create_app

STRONG_KEY = "w1-strong-test-key-0123456789abcdef"
CHAT = {"model": "test-model", "messages": [{"role": "user", "content": "hi"}]}


def _engine():
    e = MagicMock()
    e.display_name = "test-model"
    e.count_tokens.return_value = 5
    e.loaded = True
    e.chat_stream.side_effect = lambda messages, **k: iter(["ok"])
    return e


@pytest.fixture
def keyed_app(monkeypatch):
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.delenv("LOCALM_REQUIRE_AUTH", raising=False)
    auth.set_api_key(STRONG_KEY)
    app = create_app(_engine())
    app.state.bind_host = "192.168.0.10"
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c, app


def _owner():
    return {"Authorization": f"Bearer {STRONG_KEY}"}


def test_clearing_the_last_key_live_refuses_network_requests(keyed_app):
    c, _ = keyed_app
    assert c.post("/v1/chat/completions", json=CHAT, headers=_owner()).status_code == 200
    assert c.post("/v1/chat/completions", json=CHAT).status_code == 401

    assert auth.clear_api_key() == []
    assert not auth.any_key_configured()

    for method, path, body in (("post", "/v1/chat/completions", CHAT),
                               ("get", "/v1/models", None),
                               ("get", "/api/tags", None),
                               ("get", "/v1/config", None),
                               ("post", "/v1/keys", {"name": "x", "scopes": ["models:read"]})):
        kwargs = {"json": body} if body is not None else {}
        r = getattr(c, method)(path, **kwargs)
        assert r.status_code == 401, (path, r.status_code, r.text)
        assert r.json()["detail"] == hs.KEYLESS_NETWORK_DETAIL
    assert auth.list_keys() == [], "a refused POST /v1/keys must not mint a key"


def test_health_and_whoami_stay_reachable(keyed_app):
    c, _ = keyed_app
    auth.clear_api_key()
    assert c.get("/health").status_code == 200
    assert c.get("/whoami").status_code == 200


def test_instance_attach_token_passes_the_gate(keyed_app):
    c, app = keyed_app
    app.state.instance_token = "w1-instance-attach-token-value"
    auth.clear_api_key()
    assert c.get("/v1/models").status_code == 401
    r = c.get("/v1/models",
              headers={"Authorization": "Bearer w1-instance-attach-token-value"})
    assert r.status_code == 200, r.text
    r = c.get("/v1/models", headers={"Authorization": "Bearer wrong-token"})
    assert r.status_code == 401


def test_setting_a_key_again_lifts_the_gate_without_restart(keyed_app):
    c, _ = keyed_app
    auth.clear_api_key()
    assert c.post("/v1/chat/completions", json=CHAT).status_code == 401
    auth.set_api_key(STRONG_KEY)
    assert c.post("/v1/chat/completions", json=CHAT, headers=_owner()).status_code == 200
    r = c.post("/v1/chat/completions", json=CHAT)
    assert r.status_code == 401
    assert r.json()["detail"] != hs.KEYLESS_NETWORK_DETAIL


@pytest.mark.parametrize("host", ["127.0.0.1", "::1", "localhost"])
def test_a_loopback_bind_keeps_open_mode(keyed_app, host):
    c, app = keyed_app
    app.state.bind_host = host
    auth.clear_api_key()
    assert c.post("/v1/chat/completions", json=CHAT).status_code == 200


def test_an_insecure_start_keeps_open_mode_on_the_network(keyed_app):
    c, app = keyed_app
    app.state.insecure_bind = True
    auth.clear_api_key()
    assert c.post("/v1/chat/completions", json=CHAT).status_code == 200


def test_network_bind_needs_key_reads_bind_host_and_insecure():
    class _App:
        class state:
            pass
    app = _App()
    assert hs.network_bind_needs_key(app) is False
    app.state.bind_host = "0.0.0.0"
    assert hs.network_bind_needs_key(app) is True
    app.state.insecure_bind = True
    assert hs.network_bind_needs_key(app) is False


def test_the_key_clear_route_refuses_on_a_network_bind(keyed_app):
    c, _ = keyed_app
    r = c.post("/api/auth/key/clear", headers=_owner())
    assert r.status_code == 409, r.text
    assert r.json()["detail"] == hs.LAST_KEY_NETWORK_DETAIL
    assert auth.key_file().read_text(encoding="utf-8").strip() == STRONG_KEY
    assert c.post("/v1/chat/completions", json=CHAT, headers=_owner()).status_code == 200


def test_the_key_clear_route_still_clears_on_loopback(keyed_app):
    c, app = keyed_app
    app.state.bind_host = "127.0.0.1"
    r = c.post("/api/auth/key/clear", headers=_owner())
    assert r.status_code == 200, r.text
    assert r.json()["cleared"] is True
    assert not auth.key_file().exists()


def test_revoking_the_last_key_is_refused_on_a_network_bind(keyed_app):
    c, _ = keyed_app
    from localm import scopes
    auth.clear_api_key()
    admin = auth.create_key("admin-device", [scopes.KEYS_ADMIN], allow_privileged=True)
    other = auth.create_key("phone", ["models:read"])
    hdr = {"Authorization": f"Bearer {admin['key']}"}
    r = c.delete(f"/v1/keys/{other['id']}", headers=hdr)
    assert r.status_code == 200, r.text
    r = c.delete(f"/v1/keys/{admin['id']}", headers=hdr)
    assert r.status_code == 409, r.text
    assert r.json()["detail"] == hs.LAST_KEY_NETWORK_DETAIL
    assert [k["id"] for k in auth.list_keys()] == [admin["id"]]


def test_removal_leaves_no_key(monkeypatch):
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    assert auth.removal_leaves_no_key(clear_all=True) is True
    rec = auth.create_key("only", ["models:read"])
    assert auth.removal_leaves_no_key(revoke_id=rec["id"]) is True
    assert auth.removal_leaves_no_key(revoke_id="no-such-id") is False
    other = auth.create_key("other", ["models:read"])
    assert auth.removal_leaves_no_key(revoke_id=rec["id"]) is False
    auth.revoke_key(other["id"])
    auth.set_api_key(STRONG_KEY)
    assert auth.removal_leaves_no_key(revoke_id=rec["id"]) is False
    monkeypatch.setenv("LOCALM_API_KEY", STRONG_KEY)
    assert auth.removal_leaves_no_key(clear_all=True) is False
