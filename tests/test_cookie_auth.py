# SPDX-License-Identifier: AGPL-3.0-or-later
"""Opaque-session-cookie auth + session-derived CSRF for the GUI.

The GUI POSTs the key once to ``/api/session``; the server mints an OPAQUE server-
side session and sets its id as the HttpOnly ``localm_session`` cookie (never the
key). CSRF is an HMAC DERIVED from the session, returned in the response body and by
GET /api/session (NOT a separate cookie that could desync), and echoed in the
``X-CSRF-Token`` header on state-changing requests. The bearer header path
(CLI/SDK/coder) is CSRF-exempt: a cross-site page can neither read the key nor set
the Authorization header.

These tests are the oracle; each carries a negative case.
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


def _csrf(c):
    """The current session's CSRF token, from GET /api/session. It is DERIVED from
    the session server-side (an HMAC), not a cookie, so it can never desync from the
    session; the client fetches it here rather than reading a cookie."""
    return c.get("/api/session").json().get("csrf", "")


# --------------------------------------------------------------------------- #
#  Login sets the cookies with the right flags                                #
# --------------------------------------------------------------------------- #

def test_login_sets_httponly_session_and_returns_csrf_token(client):
    r = _login(client)
    assert r.status_code == 200, r.text
    cookies = " ; ".join(_set_cookies(r))
    session_sc = [c for c in _set_cookies(r) if c.startswith(SESSION_COOKIE + "=")]
    assert session_sc, f"no {SESSION_COOKIE} cookie set: {cookies}"
    # auth cookie: NOT readable by JS, not sent cross-site
    assert "httponly" in session_sc[0].lower()
    assert "samesite=strict" in session_sc[0].lower()
    # CSRF is DERIVED from the session and returned in the BODY, not as a
    # separate cookie.
    assert not [c for c in _set_cookies(r) if c.startswith(CSRF_COOKIE + "=")]
    assert r.json().get("csrf"), "login must return a csrf token"


def test_session_cookie_is_persistent(client):
    """The session cookie must PERSIST across a browser/PWA restart, so it carries
    a max-age. A session cookie dropped on close makes the key gate, and its
    'Install certificate' step, reappear every restart."""
    from localm.inference.http_server import SESSION_MAX_AGE
    r = _login(client)
    assert r.status_code == 200, r.text
    session_sc = [c for c in _set_cookies(r)
                  if c.startswith(SESSION_COOKIE + "=")][0].lower()
    assert f"max-age={SESSION_MAX_AGE}" in session_sc, session_sc
    assert SESSION_MAX_AGE > 0


def test_login_with_bad_key_rejected_no_cookie(client):
    r = client.post("/api/session", json={"key": "wrong-key"})
    assert r.status_code == 401
    assert not [c for c in _set_cookies(r) if c.startswith(SESSION_COOKIE + "=")]


def test_session_cookie_is_opaque_not_the_key(client):
    """The session cookie must carry an OPAQUE id, never the raw key: the durable
    secret must not sit in a browser cookie jar (the pre-rework design flaw)."""
    r = _login(client)
    assert r.status_code == 200
    sid = client.cookies.get(SESSION_COOKIE)
    assert sid and sid != SECRET
    assert SECRET not in " ".join(_set_cookies(r))


def test_scoped_key_session_dies_when_the_key_is_revoked(monkeypatch):
    """A browser session minted from a SCOPED key must stop authenticating once that
    key is revoked - parity with the bearer path (a revoke must actually revoke).
    Without this, a paired phone keeps its access for up to 400 days after the owner
    cuts the key off."""
    monkeypatch.setenv("LOCALM_API_KEY", "owner-key-for-revoke-test")
    from localm import auth, sessions
    from localm import scopes as S
    with TestClient(create_app(_make_engine())) as c:
        created = auth.create_key("phone", [S.MODELS_READ])
        sid = sessions.create(scopes={S.MODELS_READ},
                              key_hash=auth._hash_key(created["key"]), fs_access="none")
        c.cookies.set(SESSION_COOKIE, sid)
        assert c.get("/v1/models").status_code == 200      # valid while the key lives
        assert auth.revoke_key(created["id"]) is True
        assert c.get("/v1/models").status_code == 401      # revoked -> session dead


def test_scoped_key_session_does_not_outlive_the_key_expiry(monkeypatch):
    """A short-lived scoped key must not be laundered into a long-lived session: once
    the key's own expiry passes, the cookie session must stop authenticating too
    (the cookie path re-checks the key each request, like the bearer path)."""
    import time
    monkeypatch.setenv("LOCALM_API_KEY", "owner-key-for-expiry-test")
    from localm import auth, sessions
    from localm import scopes as S
    with TestClient(create_app(_make_engine())) as c:
        created = auth.create_key("temp", [S.MODELS_READ], expires=time.time() + 3600)
        sid = sessions.create(scopes={S.MODELS_READ},
                              key_hash=auth._hash_key(created["key"]), fs_access="none")
        c.cookies.set(SESSION_COOKIE, sid)
        assert c.get("/v1/models").status_code == 200
        # Age the key past its deadline in place (no revoke); the session must follow.
        ks = auth._load_keystore()
        for r in ks:
            if r["id"] == created["id"]:
                r["expires"] = time.time() - 1
        auth._save_keystore(ks)
        assert c.get("/v1/models").status_code == 401


def test_owner_session_is_not_gated_on_the_keystore(monkeypatch):
    """The scoped-session keystore re-check must NOT touch OWNER (ADMIN) sessions:
    the owner key is not in the keystore, so gating it there would wrongly log the
    owner out. An ADMIN session stays valid even with an empty keystore, and
    across a key roll."""
    monkeypatch.setenv("LOCALM_API_KEY", "owner-only-key-abcdef")
    from localm import auth, sessions
    from localm import scopes as S
    with TestClient(create_app(_make_engine())) as c:
        assert auth._load_keystore() == []                 # no scoped keys at all
        sid = sessions.create(scopes={S.ADMIN},
                              key_hash=auth._hash_key("owner-only-key-abcdef"),
                              fs_access="host")
        c.cookies.set(SESSION_COOKIE, sid)
        assert c.get("/v1/models").status_code == 200      # ADMIN valid, keystore empty


def test_startup_sweeps_expired_sessions(monkeypatch):
    """The server lifespan prunes expired session rows at startup (design 3.1)."""
    import json
    import time
    monkeypatch.setenv("LOCALM_API_KEY", "owner-sweep-key-123456")
    from localm import sessions
    monkeypatch.setattr(sessions, "_CACHE", {"mtime": None, "records": None})
    sessions.create(scopes={"admin"}, key_hash="KH-GOOD", ttl=10_000)
    sessions.create(scopes={"admin"}, key_hash="KH-DEAD", ttl=10_000)
    data = json.loads(sessions.sessions_file().read_text(encoding="utf-8"))
    for r in data:
        if r["key_hash"] == "KH-DEAD":
            r["expires"] = time.time() - 1
    sessions.sessions_file().write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr(sessions, "_CACHE", {"mtime": None, "records": None})
    with TestClient(create_app(_make_engine())):
        pass                                                # lifespan runs sweep()
    hashes = {r["key_hash"] for r in
              json.loads(sessions.sessions_file().read_text(encoding="utf-8"))}
    assert "KH-DEAD" not in hashes and "KH-GOOD" in hashes


def test_session_survives_owner_key_roll(monkeypatch):
    """At the HTTP layer: after login, rolling the owner key must NOT log the
    browser out. The cookie is a session id decoupled from the key, so a protected
    request over the SAME cookie still authorizes after the roll.

    Standalone (no shared `client` fixture) and monkeypatch-only for
    LOCALM_API_KEY: mixing patch.dict (the fixture) with monkeypatch.setenv on the
    same var leaks it into later open-mode tests via a teardown-order conflict."""
    from localm import auth
    monkeypatch.setenv("LOCALM_API_KEY", "old-owner-key-123456")
    with TestClient(create_app(_make_engine())) as c:
        assert c.post("/api/session", json={"key": "old-owner-key-123456"}
                      ).status_code == 200
        assert c.get("/v1/models").status_code == 200
        # Roll the owner key (what the launcher's Generate + Launch does). verify()
        # now accepts only the NEW key, but the session store is untouched.
        monkeypatch.setenv("LOCALM_API_KEY", "new-owner-key-abcdef")
        assert auth.get_api_key() == "new-owner-key-abcdef"
        # Same session cookie, new key active -> still authorized (no key gate).
        assert c.get("/v1/models").status_code == 200
        # And a state change still works with the session's derived CSRF token.
        token = c.get("/api/session").json().get("csrf", "")
        assert token
        assert c.post("/v1/models/unload",
                      headers={CSRF_HEADER: token}).status_code == 200


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
    """The CSRF token is derived from the session, not a separate readable cookie,
    so a client that clears all readable cookies - which cannot clear the HttpOnly
    session - still gets a usable token from /api/session and its writes keep
    working, with no 403 storm."""
    assert _login(client).status_code == 200
    # There is no readable localm_csrf cookie to clear in the first place.
    assert not client.cookies.get(CSRF_COOKIE)
    token = _csrf(client)
    assert token, "no csrf token from /api/session after login"
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
    token = _csrf(client)
    out = client.post("/api/session/logout", headers={CSRF_HEADER: token})
    assert out.status_code == 200
    # jar cleared -> a cookie-only protected request is now unauthorized
    client.cookies.clear()  # belt and suspenders: drop anything stale
    r = client.get("/v1/models")
    assert r.status_code == 401, r.text


# --------------------------------------------------------------------------- #
#  Open-mode + fail-closed edges                                              #
# --------------------------------------------------------------------------- #

def _open_mode(monkeypatch, tmp_path):
    """Isolate to a clean home with NO key configured anywhere (open mode):
    env unset + auth.key/auth.json resolve into an empty throwaway dir."""
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.delenv("LOCALM_REQUIRE_AUTH", raising=False)
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path))
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", tmp_path)
    monkeypatch.setattr(cfg, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", tmp_path / "registry.json")


def test_login_in_open_mode_no_bypass(tmp_path, monkeypatch):
    """Open mode (no key anywhere): /api/session cannot grant access. It is
    refused either by the route (400 - nothing to log into) or, first, by the
    open-mode management gate (403 - that POST needs the loopback shell token);
    either way it sets NO session cookie, so it is never a keyless bypass."""
    _open_mode(monkeypatch, tmp_path)
    with TestClient(create_app(_make_engine()), raise_server_exceptions=True) as c:
        r = c.post("/api/session", json={"key": "anything"})
    assert r.status_code in (400, 403), r.text
    assert not [x for x in r.headers.get_list("set-cookie")
                if x.startswith(SESSION_COOKIE + "=")]


def test_require_auth_no_key_fails_closed_even_with_forged_cookie(tmp_path, monkeypatch):
    """LOCALM_REQUIRE_AUTH with no key configured must fail CLOSED on a protected
    route, and a forged session cookie cannot bypass that gate. The refusal is 401
    so the GUI shows the key prompt instead of a 'server down' overlay; the
    security property is that it is refused, never 200."""
    _open_mode(monkeypatch, tmp_path)
    monkeypatch.setenv("LOCALM_REQUIRE_AUTH", "1")
    with TestClient(create_app(_make_engine()), raise_server_exceptions=True) as c:
        assert c.get("/v1/models").status_code == 401
        c.cookies.set(SESSION_COOKIE, "forged-value")
        assert c.get("/v1/models").status_code == 401
