# SPDX-License-Identifier: AGPL-3.0-or-later
"""A loopback `localm gui` must not be locked out when require_auth
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
    def test_loopback_protected_seeds_no_credential_at_all(self, monkeypatch):
        # The key must NEVER reach page JS or localStorage. No session cookie is
        # auto-seeded either: presenting no key to a keyed instance is the same
        # as presenting an invalid one. The launcher's ?localm_token= grant and
        # the page's key gate are covered by their own classes below.
        monkeypatch.setenv("LOCALM_API_KEY", "REALKEY123")
        r = TestClient(_app("127.0.0.1")).get("/", headers={"Host": "127.0.0.1"})
        assert r.status_code == 200
        assert "REALKEY123" not in r.text
        assert "localStorage.setItem('localm.apiKey'" not in r.text
        cookies = _set_cookies(r)
        assert "REALKEY123" not in cookies
        assert "localm_session=" not in cookies, (
            "a credential-free GET / on a keyed install still minted a session")

    def test_loopback_open_mode_seeds_shell_token_global(self, monkeypatch):
        # Open mode + loopback: the per-process shell token is injected as a JS
        # global (header-based management), not localStorage, and no auth cookie.
        # Host is set to a real loopback literal, which is what a browser
        # navigating to 127.0.0.1 sends; TestClient's default "testserver" Host
        # is a harness artifact this route does not serve the token to.
        monkeypatch.delenv("LOCALM_API_KEY", raising=False)
        r = TestClient(_app("127.0.0.1")).get("/", headers={"Host": "127.0.0.1"})
        assert r.status_code == 200
        assert SHELL_GLOBAL in r.text
        assert "SHELLTOK123" in r.text
        assert "localm_session=" not in _set_cookies(r)

    def test_corrupt_session_store_serves_the_shell_not_a_500(self, monkeypatch):
        # A corrupt or unreadable sessions.json serves the shell with NO session
        # cookie, so the client hits the recoverable key gate rather than a 500.
        monkeypatch.setenv("LOCALM_API_KEY", "REALKEY123")
        from localm import sessions
        monkeypatch.setattr(sessions, "_CACHE", {"mtime": None, "records": None})
        sessions.sessions_file().parent.mkdir(parents=True, exist_ok=True)
        sessions.sessions_file().write_text("{ corrupt not json", encoding="utf-8")
        # A loopback literal Host is load-bearing: the branch is same-origin
        # gated, so with TestClient's default "testserver" Host this would get its
        # no-cookie result from the ORIGIN GATE instead of the corrupt-store
        # fail-safe under test.
        r = TestClient(_app("127.0.0.1")).get("/", headers={"Host": "127.0.0.1"})
        assert r.status_code == 200                      # shell served, not 500
        assert "localm_session=" not in _set_cookies(r)  # fail-safe: no cookie/access

    def test_lan_bind_never_seeds(self, monkeypatch):
        # A non-loopback bind seeds nothing: no key in the page, no auth cookie,
        # no shell-token global. The same-machine user enters the key.
        monkeypatch.setenv("LOCALM_API_KEY", "REALKEY123")
        r = TestClient(_app("0.0.0.0")).get("/")
        assert r.status_code == 200
        assert "REALKEY123" not in r.text
        assert "localm_session=" not in _set_cookies(r)
        assert SHELL_GLOBAL not in r.text

    def test_cross_origin_open_mode_does_not_leak_shell_token(self, monkeypatch):
        # "loopback" describes what the SERVER BOUND TO, not who is asking. A
        # mismatched Origin on a loopback, open-mode bind gets the same
        # empty-token page as the protected-mode and non-loopback branches,
        # independent of the server's CORS config.
        monkeypatch.delenv("LOCALM_API_KEY", raising=False)
        r = TestClient(_app("127.0.0.1")).get(
            "/", headers={"Origin": "https://evil.example"})
        assert r.status_code == 200
        assert SHELL_GLOBAL not in r.text
        assert "SHELLTOK123" not in r.text

    def test_same_origin_explicit_origin_open_mode_still_seeds_shell_token(
            self, monkeypatch):
        # An Origin header that DOES match Host (a real browser fetch or reload,
        # which unlike a bare top-level navigation does send Origin) still gets
        # the token. Only a MISMATCHED Origin is refused.
        monkeypatch.delenv("LOCALM_API_KEY", raising=False)
        r = TestClient(_app("127.0.0.1")).get(
            "/", headers={"Origin": "http://testserver", "Host": "testserver"})
        assert r.status_code == 200
        assert SHELL_GLOBAL in r.text
        assert "SHELLTOK123" in r.text

    def test_dns_rebind_no_origin_attacker_host_does_not_leak_shell_token(
            self, monkeypatch):
        # A DNS-rebinding navigation carries NO Origin header at all: the
        # Same-Origin Policy is computed from the URL string navigated to, never
        # the resolved IP. The request lands on this real loopback server while
        # Host reflects the attacker's domain rather than a loopback literal, and
        # must be refused even with no Origin present.
        monkeypatch.delenv("LOCALM_API_KEY", raising=False)
        r = TestClient(_app("127.0.0.1")).get(
            "/", headers={"Host": "evil.example:8642"})
        assert r.status_code == 200
        assert SHELL_GLOBAL not in r.text
        assert "SHELLTOK123" not in r.text

    def test_no_origin_loopback_literal_host_still_seeds_shell_token(
            self, monkeypatch):
        # A real loopback literal Host (127.0.0.1, localhost, or bracketed IPv6)
        # with no Origin, which is what an actual local navigation sends, still
        # gets the token.
        monkeypatch.delenv("LOCALM_API_KEY", raising=False)
        for host in ("127.0.0.1", "127.0.0.1:8642", "localhost",
                     "localhost:8642", "[::1]:8642"):
            r = TestClient(_app("127.0.0.1")).get("/", headers={"Host": host})
            assert r.status_code == 200
            assert SHELL_GLOBAL in r.text, host
            assert "SHELLTOK123" in r.text, host


class TestKeyedShellNeverMintsForAnAnonymousCaller:
    """On a KEYED install, GET / must not mint a session for a caller that
    presented no credential. Presenting no key to a keyed instance is the same as
    presenting an invalid one.

    There is no origin-gated auto-seed: a top-level browser navigation to
    http://127.0.0.1:PORT/ and a local script calling the same URL are
    byte-identical at the HTTP layer, so no gate can separate them.

    Assertions are on the Set-Cookie ITSELF, never on a later request's status
    code: a downstream 401 can arrive for reasons unrelated to this route, so it
    cannot distinguish "no session was minted" from "one was minted and something
    else refused it"."""

    KEY = "REALKEY123"

    def _get(self, monkeypatch, headers=None, cookies=None, key=KEY, bind="127.0.0.1"):
        monkeypatch.setenv("LOCALM_API_KEY", key)
        return TestClient(_app(bind), cookies=cookies or {}).get(
            "/", headers=headers or {})

    def test_same_origin_first_visit_mints_nothing(self, monkeypatch):
        """THE RULING. The ordinary manually-typed-URL case: a real browser
        navigation, no credential. It gets the shell and NO session - the page
        then shows its key gate, which is the intended outcome."""
        for host in ("127.0.0.1", "127.0.0.1:8642", "localhost", "[::1]:8642"):
            r = self._get(monkeypatch, {"Host": host})
            assert r.status_code == 200, host
            cookies = _set_cookies(r)
            assert "localm_session=" not in cookies, host
            assert self.KEY not in cookies, host
            assert self.KEY not in r.text, host

    def test_same_origin_explicit_matching_origin_mints_nothing(self, monkeypatch):
        r = self._get(monkeypatch,
                      {"Origin": "http://127.0.0.1:8642", "Host": "127.0.0.1:8642"})
        assert r.status_code == 200
        assert "localm_session=" not in _set_cookies(r)

    def test_cross_origin_mints_nothing(self, monkeypatch):
        r = self._get(monkeypatch, {"Origin": "https://evil.example"})
        assert r.status_code == 200
        cookies = _set_cookies(r)
        assert "localm_session=" not in cookies
        assert self.KEY not in cookies and self.KEY not in r.text

    def test_local_process_shaped_request_mints_nothing(self, monkeypatch):
        """The tier the origin gate cannot reach, and why the auto-seed
        was removed rather than gated harder: no Origin and a loopback-literal
        Host is exactly what a local script sends, and it is indistinguishable
        from the legitimate browser navigation above. Neither gets a session now."""
        r = self._get(monkeypatch, {"Host": "127.0.0.1:8642"})
        assert r.status_code == 200
        assert "localm_session=" not in _set_cookies(r)

    def test_lan_bind_mints_nothing(self, monkeypatch):
        r = self._get(monkeypatch, {"Host": "127.0.0.1"}, bind="0.0.0.0")
        assert r.status_code == 200
        assert "localm_session=" not in _set_cookies(r)


class TestKeyedShellStillSupportsEveryLEGITIMATEPath:
    """The three things the removed auto-seed legitimately supported. Each is
    driven through the path that ACTUALLY mints the session now, rather than
    through the removed branch - so if any of these regress, the removal broke a
    real property rather than only the superseded one."""

    KEY = "REALKEY123"

    def _grant_session(self, app):
        """Establish a session the way the LAUNCHER does: redeem a single-use
        ?localm_token= grant. This is a real, still-supported path through this
        same route, so it is the honest way to obtain the "existing session"
        precondition now that the auto-seed cannot supply it."""
        from localm.plugins.gui.web import mint_launch_grant
        grant = mint_launch_grant(app)
        c = TestClient(app)
        r = c.get(f"/?localm_token={grant}", follow_redirects=False)
        assert r.status_code == 303, "precondition: the grant must be redeemed"
        sid = None
        for part in _set_cookies(r).split(";"):
            part = part.strip()
            if part.startswith("localm_session="):
                sid = part.split("=", 1)[1]
        assert sid, "precondition: redeeming the grant must set a session"
        return sid

    def test_the_launch_grant_still_signs_the_owner_in(self, monkeypatch):
        monkeypatch.setenv("LOCALM_API_KEY", self.KEY)
        sid = self._grant_session(_app("127.0.0.1"))
        from localm import sessions as _sessions
        assert _sessions.lookup(sid) is not None

    def test_an_existing_session_is_not_bounced_and_is_not_re_minted(self, monkeypatch):
        """A browser that already holds a valid session must keep working: the
        shell is served, and nothing re-issues or clears the cookie it holds."""
        monkeypatch.setenv("LOCALM_API_KEY", self.KEY)
        app = _app("127.0.0.1")
        sid = self._grant_session(app)
        r = TestClient(app, cookies={"localm_session": sid}).get(
            "/", headers={"Host": "127.0.0.1"})
        assert r.status_code == 200
        assert "localm_session=" not in _set_cookies(r)     # not re-minted
        from localm import sessions as _sessions
        assert _sessions.lookup(sid) is not None            # and not invalidated

    def test_an_existing_session_survives_an_owner_key_roll(self, monkeypatch):
        """The session is decoupled from the key VALUE, so rolling the owner
        key must not sign the browser out."""
        monkeypatch.setenv("LOCALM_API_KEY", self.KEY)
        app = _app("127.0.0.1")
        sid = self._grant_session(app)

        monkeypatch.setenv("LOCALM_API_KEY", "ROLLEDKEY456")      # the roll
        r = TestClient(app, cookies={"localm_session": sid}).get(
            "/", headers={"Host": "127.0.0.1"})
        assert r.status_code == 200
        assert "localm_session=" not in _set_cookies(r)
        from localm import sessions as _sessions
        assert _sessions.lookup(sid) is not None, (
            "an owner-key roll signed the browser out - the exact regression the "
            "removed auto-seed branch existed to prevent")


class TestLaunchGrantHandoff:
    """One-time ?localm_token= handoff: the launcher opens the browser at a fresh
    URL that forces a real navigation (a stale tab / warm SW cannot short-circuit
    it); the server redeems the single-use grant, establishes a session, and 303s
    to the clean path. Each test carries its negative case.

    The boundary the grant does not cross (a plain GET / on a network bind, no
    grant, still seeds no session) is covered by
    TestShellRoute.test_lan_bind_never_seeds, which asserts the same 200/
    no-cookie outcome plus the page-text and shell-token checks."""

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

    def test_grant_IS_redeemed_on_a_network_bind(self, monkeypatch):
        # The launcher is on THIS machine and is handed a session even when the
        # server is exposed on 0.0.0.0: the grant is a single-use secret only the
        # launcher knows, so possessing it is the authorization regardless of
        # bind.
        monkeypatch.setenv("LOCALM_API_KEY", "REALKEY123")
        app = _app("0.0.0.0")
        grant = self._grant(app)
        r = TestClient(app).get(f"/?localm_token={grant}", follow_redirects=False)
        assert r.status_code == 303                          # redeemed on the LAN bind
        assert "localm_session=" in _set_cookies(r)
        assert "REALKEY123" not in _set_cookies(r)           # opaque session, not the key


class TestPullGrant:
    """`localm gui --pull SPEC` mints a single-use, spec-bound secret
    (mint_pull_grant) so its OWN deep link can auto-start the download with
    zero clicks (see init.js), while a forged `?pull=` link elsewhere - which
    cannot know this secret - falls back to an explicit human confirmation. This
    class covers the grant primitive itself; the HTTP redeem endpoint is covered
    by TestPullTokenRedeemEndpoint in test_gui.py, alongside the other pull
    routes."""

    def test_valid_grant_redeems_once(self):
        from localm.plugins.gui.web import consume_pull_grant, mint_pull_grant
        app = _app("127.0.0.1")
        token = mint_pull_grant(app, "owner/repo:m.gguf")
        assert consume_pull_grant(app, "owner/repo:m.gguf", token) is True

    def test_grant_is_single_use(self):
        from localm.plugins.gui.web import consume_pull_grant, mint_pull_grant
        app = _app("127.0.0.1")
        token = mint_pull_grant(app, "owner/repo:m.gguf")
        assert consume_pull_grant(app, "owner/repo:m.gguf", token) is True
        # Replayed: already popped on first redeem.
        assert consume_pull_grant(app, "owner/repo:m.gguf", token) is False

    def test_grant_is_bound_to_the_exact_spec(self):
        # A leaked or observed token must not authorise pulling a DIFFERENT
        # model.
        from localm.plugins.gui.web import consume_pull_grant, mint_pull_grant
        app = _app("127.0.0.1")
        token = mint_pull_grant(app, "owner/repo:m.gguf")
        assert consume_pull_grant(app, "someone-else/other:x.gguf", token) is False
        # Still unredeemed for the RIGHT spec (a mismatched attempt must not burn it).
        assert consume_pull_grant(app, "owner/repo:m.gguf", token) is True

    def test_unknown_or_missing_token_is_rejected(self):
        from localm.plugins.gui.web import consume_pull_grant
        app = _app("127.0.0.1")
        assert consume_pull_grant(app, "owner/repo:m.gguf", "bogus-never-minted") is False
        assert consume_pull_grant(app, "owner/repo:m.gguf", "") is False
        assert consume_pull_grant(app, "owner/repo:m.gguf", None) is False

    def test_expired_grant_is_rejected(self):
        from localm.plugins.gui.web import consume_pull_grant, mint_pull_grant
        app = _app("127.0.0.1")
        # A negative TTL puts the expiry in the past the moment it is minted.
        token = mint_pull_grant(app, "owner/repo:m.gguf", ttl=-1.0)
        assert consume_pull_grant(app, "owner/repo:m.gguf", token) is False
