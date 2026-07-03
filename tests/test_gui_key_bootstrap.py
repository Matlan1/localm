# SPDX-License-Identifier: AGPL-3.0-or-later
"""S2 (GUI half): a loopback `localm gui` must not be locked out when require_auth
is on, WITHOUT putting the API key in JS-readable localStorage. On a loopback bind
the shell route sets the key as an HttpOnly session cookie (protected mode) or
seeds the per-process shell token as a JS global (open mode); a non-loopback LAN
client gets neither (it enters the key in the page -> POST /api/session)."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from localm.plugins.gui.web import (
    attach_gui,
    _index_html_with_shell_token,
    _is_loopback_host,
)

SHELL_GLOBAL = "window.__LOCALM_SHELL_TOKEN__="


def _app(bind_host):
    app = FastAPI()

    async def switch_model(name):
        pass

    attach_gui(app, self_url="http://127.0.0.1:9/v1",
               switch_model=switch_model, active_model=lambda: "m")
    app.state.bind_host = bind_host
    app.state.shell_token = "SHELLTOK123"
    return app


def _set_cookies(resp):
    return " ; ".join(resp.headers.get_list("set-cookie"))


class TestIsLoopbackHost:
    def test_loopback(self):
        assert _is_loopback_host("127.0.0.1")
        assert _is_loopback_host("127.0.0.5")
        assert _is_loopback_host("::1")
        assert _is_loopback_host("localhost")

    def test_not_loopback(self):
        assert not _is_loopback_host("0.0.0.0")
        assert not _is_loopback_host("192.168.1.10")
        assert not _is_loopback_host("")
        assert not _is_loopback_host("testclient")


class TestShellTokenInjection:
    def test_injects_shell_token_global(self):
        html = _index_html_with_shell_token("TOK-abc_123")
        assert SHELL_GLOBAL in html
        assert '"TOK-abc_123"' in html

    def test_no_token_no_injection(self):
        assert SHELL_GLOBAL not in _index_html_with_shell_token("")

    def test_script_breakout_escaped(self):
        # a "<" in the token is unicode-escaped so it can never close the
        # <script> element.
        html = _index_html_with_shell_token("a</script>b")
        assert "a</script>b" not in html
        assert "a\\u003c/script>b" in html


class TestShellRoute:
    def test_loopback_protected_sets_httponly_cookie_not_localstorage(self, monkeypatch):
        # Protected mode + loopback: the key is set as an HttpOnly cookie and is
        # NEVER echoed into the page (S2 - no localStorage seeding).
        monkeypatch.setenv("LOCALM_API_KEY", "REALKEY123")
        r = TestClient(_app("127.0.0.1")).get("/")
        assert r.status_code == 200
        assert "REALKEY123" not in r.text
        assert "localStorage.setItem('localm.apiKey'" not in r.text
        cookies = _set_cookies(r)
        # Decoupled sessions (S2 hardened): the cookie carries an OPAQUE session id,
        # never the raw key, so the key never lands in a cookie jar and rolling it
        # does not invalidate the session. The key must appear NOWHERE in Set-Cookie.
        assert "REALKEY123" not in cookies
        assert "localm_session=" in cookies
        assert "httponly" in cookies.lower()
        assert "samesite=strict" in cookies.lower()
        # CSRF is derived from the session (fetched via GET /api/session), so there
        # is NO separate localm_csrf cookie to set (or to desync from the session).
        assert "localm_csrf=" not in cookies
        # SEAMLESS: the auto-seeded cookie PERSISTS (max-age) so the loopback user
        # stays signed in across a browser restart, matching the /api/session path.
        assert "max-age=" in cookies.lower()

    def test_loopback_open_mode_seeds_shell_token_global(self, monkeypatch):
        # Open mode + loopback: the per-process shell token is injected as a JS
        # global (header-based management), not localStorage, and no auth cookie.
        monkeypatch.delenv("LOCALM_API_KEY", raising=False)
        r = TestClient(_app("127.0.0.1")).get("/")
        assert r.status_code == 200
        assert SHELL_GLOBAL in r.text
        assert "SHELLTOK123" in r.text
        assert "localm_session=" not in _set_cookies(r)

    def test_lan_bind_never_seeds(self, monkeypatch):
        # A non-loopback bind seeds nothing: no key in the page, no auth cookie,
        # no shell-token global. The same-machine user enters the key.
        monkeypatch.setenv("LOCALM_API_KEY", "REALKEY123")
        r = TestClient(_app("0.0.0.0")).get("/")
        assert r.status_code == 200
        assert "REALKEY123" not in r.text
        assert "localm_session=" not in _set_cookies(r)
        assert SHELL_GLOBAL not in r.text


class TestLaunchGrantHandoff:
    """One-time ?localm_token= handoff: the launcher opens the browser at a fresh
    URL that forces a real navigation (a stale tab / warm SW cannot short-circuit
    it); the server redeems the single-use grant, establishes a session, and 303s
    to the clean path. Each test carries its negative case."""

    def _grant(self, app):
        from localm.plugins.gui.web import mint_launch_grant
        return mint_launch_grant(app)

    def test_valid_grant_redirects_and_sets_opaque_session(self, monkeypatch):
        monkeypatch.setenv("LOCALM_API_KEY", "REALKEY123")
        app = _app("127.0.0.1")
        grant = self._grant(app)
        r = TestClient(app).get(f"/?localm_token={grant}", follow_redirects=False)
        assert r.status_code == 303
        assert r.headers["location"] == "/"          # token stripped from the URL
        cookies = _set_cookies(r)                     # joined "set-cookie" string
        assert "localm_session=" in cookies
        assert "REALKEY123" not in cookies            # opaque id, never the key

    def test_grant_is_single_use(self, monkeypatch):
        monkeypatch.setenv("LOCALM_API_KEY", "REALKEY123")
        app = _app("127.0.0.1")
        grant = self._grant(app)
        c = TestClient(app)
        assert c.get(f"/?localm_token={grant}",
                     follow_redirects=False).status_code == 303
        # Reused grant: NOT redeemed again -> no redirect, just the normal shell.
        assert c.get(f"/?localm_token={grant}",
                     follow_redirects=False).status_code == 200

    def test_unknown_grant_falls_through_no_error(self, monkeypatch):
        monkeypatch.setenv("LOCALM_API_KEY", "REALKEY123")
        r = TestClient(_app("127.0.0.1")).get(
            "/?localm_token=bogus-never-minted", follow_redirects=False)
        assert r.status_code == 200                  # normal shell, no leak

    def test_redirect_preserves_other_query_params(self, monkeypatch):
        monkeypatch.setenv("LOCALM_API_KEY", "REALKEY123")
        app = _app("127.0.0.1")
        grant = self._grant(app)
        r = TestClient(app).get(f"/?view=models&localm_token={grant}",
                                follow_redirects=False)
        assert r.status_code == 303
        assert r.headers["location"] == "/?view=models"

    def test_grant_not_redeemed_on_lan_bind(self, monkeypatch):
        # A LAN bind seeds nothing and must not redeem a grant either (only a
        # loopback bind auto-authenticates a browser).
        monkeypatch.setenv("LOCALM_API_KEY", "REALKEY123")
        app = _app("0.0.0.0")
        grant = self._grant(app)
        r = TestClient(app).get(f"/?localm_token={grant}", follow_redirects=False)
        assert r.status_code == 200
        assert "localm_session=" not in _set_cookies(r)
