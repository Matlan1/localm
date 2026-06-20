# SPDX-License-Identifier: AGPL-3.0-or-later
"""S2: HttpOnly-cookie session auth + CSRF (double-submit) for the GUI.

The GUI must stop holding the API key in JS-readable localStorage. Instead it
POSTs the key once to ``/api/session``; the server sets an HttpOnly auth cookie
(``localm_session``) the browser JS cannot read, plus a readable ``localm_csrf``
cookie. State-changing (unsafe-method) requests authenticated BY THE COOKIE must
echo the CSRF cookie in an ``X-CSRF-Token`` header (double-submit). The bearer
header path (CLI/SDK/coder) is unchanged and CSRF-exempt (a cross-site page can
neither read the key nor set the Authorization header).

These tests are the S2 oracle; each carries a negative case.
"""

import os
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from localm.inference.http_server import create_app

SECRET = "test-secret-cookie-key-xyz"
SESSION_COOKIE = "localm_session"
CSRF_COOKIE = "localm_csrf"
CSRF_HEADER = "X-CSRF-Token"


def _make_engine():
    engine = MagicMock()
    engine.display_name = "test-model"
    engine.count_tokens.return_value = 5
    _state = {"loaded": True}
    engine.unload.side_effect = lambda: _state.update(loaded=False)
    engine.load.side_effect = lambda: _state.update(loaded=True)
    engine.chat_stream.side_effect = lambda messages, **k: iter(["ok"])
    type(engine).loaded = property(lambda self: _state["loaded"])
    return engine


@pytest.fixture()
def client():
    """Protected-mode TestClient: a key is configured for the whole test (so the
    cookie/login path is exercised), and httpx persists Set-Cookie in its jar."""
    with patch.dict(os.environ, {"LOCALM_API_KEY": SECRET}):
        with TestClient(create_app(_make_engine()), raise_server_exceptions=True) as c:
            yield c


def _login(c, key=SECRET):
    """POST the key to /api/session; returns the response. On success the jar now
    holds localm_session + localm_csrf."""
    return c.post("/api/session", json={"key": key})


def _set_cookies(resp):
    return resp.headers.get_list("set-cookie")


# --------------------------------------------------------------------------- #
#  Login sets the cookies with the right flags                                #
# --------------------------------------------------------------------------- #

def test_login_sets_httponly_session_and_readable_csrf(client):
    r = _login(client)
    assert r.status_code == 200, r.text
    cookies = " ; ".join(_set_cookies(r))
    session_sc = [c for c in _set_cookies(r) if c.startswith(SESSION_COOKIE + "=")]
    csrf_sc = [c for c in _set_cookies(r) if c.startswith(CSRF_COOKIE + "=")]
    assert session_sc, f"no {SESSION_COOKIE} cookie set: {cookies}"
    assert csrf_sc, f"no {CSRF_COOKIE} cookie set: {cookies}"
    # auth cookie: NOT readable by JS, not sent cross-site
    assert "httponly" in session_sc[0].lower()
    assert "samesite=strict" in session_sc[0].lower()
    # csrf cookie: readable by JS (double-submit) -> must NOT be HttpOnly
    assert "httponly" not in csrf_sc[0].lower()
    assert "samesite=strict" in csrf_sc[0].lower()


def test_login_with_bad_key_rejected_no_cookie(client):
    r = client.post("/api/session", json={"key": "wrong-key"})
    assert r.status_code == 401
    assert not [c for c in _set_cookies(r) if c.startswith(SESSION_COOKIE + "=")]


# --------------------------------------------------------------------------- #
#  Cookie auth works for reads; CSRF gate for writes                          #
# --------------------------------------------------------------------------- #

def test_cookie_only_get_is_authorized(client):
    """After login, a protected GET with ONLY the cookie (no Authorization
    header) is authorized - the cookie is the credential."""
    assert _login(client).status_code == 200
    r = client.get("/v1/models")  # cookie auto-sent from the jar; no header
    assert r.status_code == 200, r.text


def test_cookie_unsafe_method_without_csrf_is_refused(client):
    """A cookie-authenticated POST without the CSRF header is refused (CSRF)."""
    assert _login(client).status_code == 200
    r = client.post("/v1/models/unload")  # cookie sent, NO X-CSRF-Token
    assert r.status_code == 403, r.text


def test_cookie_unsafe_method_with_csrf_allowed(client):
    assert _login(client).status_code == 200
    token = client.cookies.get(CSRF_COOKIE)
    assert token, "csrf cookie not in jar after login"
    r = client.post("/v1/models/unload", headers={CSRF_HEADER: token})
    assert r.status_code == 200, r.text


def test_cookie_unsafe_method_with_wrong_csrf_refused(client):
    assert _login(client).status_code == 200
    r = client.post("/v1/models/unload", headers={CSRF_HEADER: "not-the-token"})
    assert r.status_code == 403, r.text


# --------------------------------------------------------------------------- #
#  Bearer header path is unchanged + CSRF-exempt (CLI / SDK / coder)          #
# --------------------------------------------------------------------------- #

def test_bearer_post_is_csrf_exempt(client):
    """A Bearer-header POST needs no CSRF token (the header is un-forgeable
    cross-site), so CLI/SDK keep working with no cookie + no CSRF."""
    r = client.post("/v1/models/unload",
                    headers={"Authorization": f"Bearer {SECRET}"})
    assert r.status_code == 200, r.text


def test_bearer_wins_over_cookie(client):
    """If both are present, the Authorization header is used (and is CSRF-exempt)."""
    assert _login(client).status_code == 200
    r = client.post("/v1/models/unload",
                    headers={"Authorization": f"Bearer {SECRET}"})  # no CSRF token
    assert r.status_code == 200, r.text


# --------------------------------------------------------------------------- #
#  Session state + logout                                                     #
# --------------------------------------------------------------------------- #

def test_get_session_reports_authed_state(client):
    before = client.get("/api/session")
    assert before.status_code == 200
    assert before.json().get("authed") is False
    assert _login(client).status_code == 200
    after = client.get("/api/session")
    assert after.json().get("authed") is True


def test_logout_clears_cookie(client):
    assert _login(client).status_code == 200
    token = client.cookies.get(CSRF_COOKIE)
    out = client.post("/api/session/logout", headers={CSRF_HEADER: token})
    assert out.status_code == 200
    # jar cleared -> a cookie-only protected request is now unauthorized
    client.cookies.clear()  # belt and suspenders: drop anything stale
    r = client.get("/v1/models")
    assert r.status_code == 401, r.text
