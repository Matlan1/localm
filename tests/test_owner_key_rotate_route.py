# SPDX-License-Identifier: AGPL-3.0-or-later
"""``POST /api/auth/key/rotate`` - the GUI form of ``localm key generate`` / ``set``.

Two properties carry the weight here, neither about the happy path:

* **The gate is ADMIN, not config:write.** Setting a key the caller CHOOSES is a
  direct promotion to owner, so a merely ``config:write`` or ``keys:admin``
  holder is refused. The sibling ``/api/auth/key/clear`` takes ``config:write``,
  so there is an explicit test per non-owner scope rather than one generic
  "unauthorised" case.
* **A rotation that did not change the LIVE credential does not report success.**
  ``LOCALM_API_KEY`` outranks the file, so under it the key lands on disk and the
  server keeps accepting the old one.

Assertions read the real ``auth`` state from OUTSIDE the call rather than trusting
the response body, so a route that returned a cheerful shape while writing nothing
still fails.
"""

import pytest

from localm import auth, scopes, sessions

ROTATE = "/api/auth/key/rotate"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.delenv("LOCALM_REQUIRE_AUTH", raising=False)


def _client(bind_host="127.0.0.1"):
    from fastapi.testclient import TestClient

    from localm.inference.http_server import create_app
    app = create_app(None)
    app.state.bind_host = bind_host
    c = TestClient(app)
    # The per-process shell token the server injects into the browser-served SPA.
    # In OPEN mode _origin_guard demands it (or the instance token) for any unsafe
    # method. Kept on the client so the open-mode cases present it exactly as the
    # real GUI does.
    c.shell_token = app.state.shell_token
    return c


def _owner(key):
    return {"Authorization": f"Bearer {key}"}


class TestRotateHappyPath:
    def test_no_body_generates_a_new_key_and_returns_the_live_one(self):
        old = auth.regenerate_key()
        c = _client()

        r = c.post(ROTATE, headers=_owner(old))

        assert r.status_code == 200, r.text
        body = r.json()
        assert body["rotated"] is True and body["active"] is True
        assert body["warnings"] == []
        # The claim is about the SERVER's credential, not about the response.
        assert auth.get_api_key() == body["key"] != old

    def test_explicit_key_is_persisted_verbatim(self):
        old = auth.regenerate_key()
        chosen = "a" * 40
        c = _client()

        r = c.post(ROTATE, json={"key": chosen}, headers=_owner(old))

        assert r.status_code == 200, r.text
        assert r.json()["rotated"] is True
        assert auth.get_api_key() == chosen

    def test_the_new_key_authenticates_and_the_old_one_stops(self):
        """The point of a rotation, asserted end to end over HTTP rather than
        inferred from the file having changed."""
        old = auth.regenerate_key()
        c = _client()
        new = c.post(ROTATE, headers=_owner(old)).json()["key"]

        assert c.get("/api/session", headers=_owner(new)).json()["authed"] is True
        assert c.get("/api/session", headers=_owner(old)).json()["authed"] is False

    def test_empty_key_generates_rather_than_clearing(self):
        """``set_api_key("")`` CLEARS, so this route generates instead: it is
        never a path back to open mode. /api/auth/key/clear is."""
        old = auth.regenerate_key()
        c = _client()

        r = c.post(ROTATE, json={"key": "   "}, headers=_owner(old))

        assert r.status_code == 200, r.text
        assert auth.get_api_key() not in (None, "", old)
        assert auth.any_key_configured() is True


class TestOnlyTheOwnerMayRotate:
    """A non-owner reaching this route promotes itself to owner in one call."""

    @pytest.mark.parametrize("scope", [scopes.CONFIG_WRITE, scopes.KEYS_ADMIN])
    def test_non_owner_scoped_key_is_refused(self, scope):
        auth.regenerate_key()
        made = auth.create_key(f"device-{scope}", [scope], allow_privileged=True)
        c = _client()
        before = auth.get_api_key()

        r = c.post(ROTATE, json={"key": "z" * 40}, headers=_owner(made["key"]))

        assert r.status_code == 403, (
            f"a {scope} key rotating the OWNER credential is privilege escalation")
        assert auth.get_api_key() == before, "and it must not have taken effect"

    def test_config_write_may_still_clear(self):
        """Control for the test above: the refusal is specific to ROTATE, not a
        blanket lockout on the config:write scope."""
        auth.regenerate_key()
        made = auth.create_key("clearer", [scopes.CONFIG_WRITE],
                               allow_privileged=True)
        c = _client()

        r = c.post("/api/auth/key/clear", headers=_owner(made["key"]))

        assert r.status_code == 200, r.text


class TestBadInputIsA400NotA500:
    @pytest.mark.parametrize("bad", ["short", "has spaces in it here ok", "k" * 3])
    def test_rejected_key_leaves_the_previous_one_intact(self, bad):
        old = auth.regenerate_key()
        c = _client()

        r = c.post(ROTATE, json={"key": bad}, headers=_owner(old))

        assert r.status_code == 400, (
            "a key the user mistyped is caller input, not a server fault")
        assert auth.get_api_key() == old, "a refused rotation must change nothing"

    def test_non_string_key_is_refused(self):
        old = auth.regenerate_key()
        c = _client()

        r = c.post(ROTATE, json={"key": 1234}, headers=_owner(old))

        assert r.status_code == 400
        assert auth.get_api_key() == old


class TestRotationHonesty:
    """A rotation that did not change the live credential is not reported as a
    rotation, however cleanly the write succeeded."""

    def test_env_var_override_is_reported_and_rotated_is_false(self, monkeypatch):
        old = auth.regenerate_key()
        c = _client()
        monkeypatch.setenv("LOCALM_API_KEY", old)

        r = c.post(ROTATE, headers=_owner(old))

        assert r.status_code == 200, r.text
        body = r.json()
        assert body["active"] is False
        assert body["rotated"] is False, (
            "rotated:true here tells someone rotating a leaked key they are safe "
            "while the leaked key still authenticates")
        assert body["warnings"], "the caller must learn WHY it did not take"
        assert "LOCALM_API_KEY" in " ".join(body["warnings"])
        # The old key really does still work.
        assert c.get("/api/session", headers=_owner(old)).json()["authed"] is True

    def test_response_leaks_no_path_and_no_exception_text(self, monkeypatch):
        old = auth.regenerate_key()
        home = str(auth.key_file().parent)
        c = _client()
        monkeypatch.setenv("LOCALM_API_KEY", old)

        blob = c.post(ROTATE, headers=_owner(old)).text

        assert home not in blob, "the data dir path carries the account name"
        assert "Error" not in blob, "no raw OS exception text on the wire"


class TestSessionsSurviveARoll:
    def test_a_roll_does_not_sign_the_browser_out(self):
        """Parity with ``localm key generate`` / ``key set``, which leave
        sessions alone: sessions are decoupled from the key value, so a roll does
        not log out the browser doing the rolling. ``localm key recover`` is the
        local compromise path that DOES revoke."""
        old = auth.regenerate_key()
        sid = sessions.create(scopes={scopes.ADMIN},
                              key_hash=auth._hash_key(old), fs_access="host")
        c = _client()

        r = c.post(ROTATE, headers=_owner(old))

        assert r.status_code == 200, r.text
        assert sessions.lookup(sid) is not None, (
            "revoking here would log out the very browser that rotated the key")


class TestFirstKeyDoesNotLockTheLocalBrowserOut:
    def test_open_mode_on_loopback_seeds_an_owner_session_cookie(self):
        """In open mode the loopback GUI is trusted via the shell token, which the
        server stops honouring the instant a key exists, so setting the FIRST key
        hands this browser a session, exactly as the first-key path in
        routes/keys.py does."""
        assert not auth.any_key_configured()
        c = _client(bind_host="127.0.0.1")

        r = c.post(ROTATE, headers=_owner(c.shell_token))

        assert r.status_code == 200, r.text
        assert auth.any_key_configured() is True
        from localm.inference.http_server import SESSION_COOKIE
        assert SESSION_COOKIE in r.cookies, (
            "no cookie means the browser that just set the key is now locked out")
        # And it is a REAL owner session, not merely a cookie-shaped string.
        assert c.get("/api/session").json()["authed"] is True

    def test_network_bind_open_mode_seeds_no_cookie(self):
        """The seed is loopback-only, matching routes/keys.py: a network bind
        already required a key up front."""
        assert not auth.any_key_configured()
        c = _client(bind_host="0.0.0.0")

        r = c.post(ROTATE, headers=_owner(c.shell_token))

        assert r.status_code == 200, r.text
        from localm.inference.http_server import SESSION_COOKIE
        assert SESSION_COOKIE not in r.cookies

    def test_open_mode_without_local_proof_is_refused(self):
        """The ADMIN dependency cannot gate open mode: ``_enforce_request``
        returns early when no key is configured, so what stops a remote caller
        seizing a keyless install is ``_origin_guard``'s demand for the
        shell/instance token. This route depends on that guard for its open-mode
        safety."""
        assert not auth.any_key_configured()
        c = _client(bind_host="0.0.0.0")

        r = c.post(ROTATE)   # no shell token, i.e. not a local process

        assert r.status_code == 403, (
            "a keyless install must not let an unproven caller install an owner key")
        assert not auth.any_key_configured()


class TestCookieCallerNeedsCsrf:
    def test_cookie_session_without_csrf_token_is_refused(self):
        """A cookie-authenticated unsafe request needs X-CSRF-Token, and this
        route is no exception."""
        old = auth.regenerate_key()
        c = _client()
        assert c.post("/api/session", json={"key": old}).status_code == 200

        r = c.post(ROTATE)   # cookie rides along; no X-CSRF-Token header

        assert r.status_code == 403, r.text
        assert auth.get_api_key() == old, "and nothing may have changed"

    def test_cookie_session_with_csrf_token_succeeds(self):
        """Control for the test above: the 403 is the CSRF check, not a cookie
        session being unable to rotate at all."""
        old = auth.regenerate_key()
        c = _client()
        csrf = c.post("/api/session", json={"key": old}).json()["csrf"]

        r = c.post(ROTATE, headers={"X-CSRF-Token": csrf})

        assert r.status_code == 200, r.text
        assert auth.get_api_key() != old
