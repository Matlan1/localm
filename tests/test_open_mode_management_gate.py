"""Open-mode management gate.

In open mode (no API key configured) a no-Origin local client (curl, a script)
could otherwise mint a key, flip config, install a plugin, load or unload a
model, or browse the filesystem. This gate requires such requests to carry the
per-process *shell token* that the loopback GUI shell injects into the SPA,
closing the no-Origin gap without relying on Origin headers. Inference and read
routes stay open; protected mode (a key exists) is unchanged (bearer auth
handles it); an operator-allowlisted CORS origin is exempt.
"""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from localm.inference.http_server import create_app

CHAT_PAYLOAD = {"model": "test-model", "messages": [{"role": "user", "content": "hi"}]}


def _engine():
    e = MagicMock()
    e.display_name = "test-model"
    e.count_tokens.return_value = 5
    st = {"loaded": True}
    e.unload.side_effect = lambda: st.__setitem__("loaded", False)
    e.load.side_effect = lambda: st.__setitem__("loaded", True)
    e.chat_stream.side_effect = lambda messages, **k: iter(["ok"])
    type(e).loaded = property(lambda self: st["loaded"])
    return e


@pytest.fixture
def open_app(monkeypatch):
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.delenv("LOCALM_REQUIRE_AUTH", raising=False)
    app = create_app(_engine())
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c, app


def _bearer(app):
    return {"Authorization": f"Bearer {app.state.shell_token}"}


# --------------------------------------------------------------------------- #
#  The gate                                                                     #
# --------------------------------------------------------------------------- #

class TestOpenModeGate:
    def test_shell_token_minted(self, open_app):
        _, app = open_app
        tok = app.state.shell_token
        assert isinstance(tok, str) and len(tok) >= 20

    # (method, path, json body) for every management endpoint the gate covers,
    # so a route that forgot to register under the gate still fails its own row.
    _MANAGEMENT_ENDPOINTS = [
        ("post", "/v1/keys", {"name": "x", "scopes": ["models:read"]}),
        ("patch", "/v1/config", {"n_ctx": 8192}),
        ("get", "/v1/config", None),
        ("post", "/v1/models/unload", None),
    ]

    @pytest.mark.parametrize("method, path, json_body", _MANAGEMENT_ENDPOINTS)
    def test_management_without_token_refused(self, open_app, method, path, json_body):
        c, _ = open_app
        kwargs = {"json": json_body} if json_body is not None else {}
        r = getattr(c, method)(path, **kwargs)
        assert r.status_code == 403

    @pytest.mark.parametrize("method, path, json_body", _MANAGEMENT_ENDPOINTS)
    def test_management_with_token_allowed(self, open_app, method, path, json_body):
        c, app = open_app
        kwargs = {"json": json_body} if json_body is not None else {}
        r = getattr(c, method)(path, headers=_bearer(app), **kwargs)
        assert r.status_code == 200

    def test_get_session_does_not_need_token(self, open_app):
        c, _ = open_app
        assert c.get("/api/session").status_code == 200

    def test_wrong_token_refused(self, open_app):
        c, _ = open_app
        r = c.post("/v1/models/unload",
                   headers={"Authorization": "Bearer not-the-real-token"})
        assert r.status_code == 403

    def test_cross_origin_with_token_still_refused(self, open_app):
        # The cross-origin guard composes: a stolen token cannot drive management
        # from a foreign origin (the Origin check runs first).
        c, app = open_app
        r = c.post("/v1/models/unload",
                   headers={**_bearer(app), "Origin": "http://evil.example"})
        assert r.status_code == 403


# --------------------------------------------------------------------------- #
#  What the gate must NOT touch                                                 #
# --------------------------------------------------------------------------- #

class TestGateScope:
    def test_inference_stays_open(self, open_app):
        c, _ = open_app
        assert c.post("/v1/chat/completions", json=CHAT_PAYLOAD).status_code == 200

    def test_models_read_stays_open(self, open_app):
        c, _ = open_app
        assert c.get("/v1/models").status_code == 200

    def test_health_stays_open(self, open_app):
        c, _ = open_app
        assert c.get("/health").status_code == 200


class TestProtectedModeUnaffected:
    def test_owner_key_manages_without_shell_token(self, monkeypatch):
        monkeypatch.setenv("LOCALM_API_KEY", "owner-secret-xyz")
        app = create_app(_engine())
        with TestClient(app) as c:
            r = c.post("/v1/models/unload",
                       headers={"Authorization": "Bearer owner-secret-xyz"})
            assert r.status_code == 200            # owner key works, no token needed
            assert c.post("/v1/models/unload").status_code == 401   # missing key

    def test_shell_token_is_not_a_key_in_protected_mode(self, monkeypatch):
        # Once a key exists the shell token must NOT act as a management bypass.
        monkeypatch.setenv("LOCALM_API_KEY", "owner-secret-xyz")
        app = create_app(_engine())
        with TestClient(app) as c:
            r = c.post("/v1/models/unload",
                       headers={"Authorization": f"Bearer {app.state.shell_token}"})
            assert r.status_code == 401            # token is not a valid key


# --------------------------------------------------------------------------- #
#  A keyless LOCAL process (localm status / the MCP tool) authenticates via   #
#  this instance's own attach token, the same "local process, not a browser"  #
#  distinction /v1/surfaces/gui's mount_gui route already uses.               #
# --------------------------------------------------------------------------- #

def _with_instance_token(app, token="the-instance-attach-token"):
    app.state.instance_token = token
    return token


class TestKeylessActivityViaInstanceToken:
    def test_activity_refused_without_any_token(self, open_app):
        """Baseline: a genuinely open server 403s a bare /api/activity read
        with no credential at all."""
        c, _ = open_app
        r = c.get("/api/activity")
        assert r.status_code == 403

    def test_activity_allowed_with_shell_token(self, open_app):
        """The browser's shell token keeps working for /api/activity exactly as
        it does for management routes; the instance token is a second
        acceptable credential, not a replacement."""
        c, app = open_app
        r = c.get("/api/activity", headers=_bearer(app))
        assert r.status_code == 200

    def test_activity_allowed_with_instance_token(self, open_app):
        """A local CLI or MCP process has no shell_token (it never reaches
        outside the browser-served page) but does have this instance's own
        attach token from the 0600 registry file, which is enough."""
        c, app = open_app
        token = _with_instance_token(app)
        r = c.get("/api/activity", headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 200

    def test_wrong_instance_token_refused(self, open_app):
        c, app = open_app
        _with_instance_token(app)
        r = c.get("/api/activity",
                  headers={"Authorization": "Bearer not-the-real-token"})
        assert r.status_code == 403

    def test_cross_origin_with_instance_token_still_refused(self, open_app):
        """THE CRITICAL NEGATIVE: a browser-shaped request (one carrying a
        foreign Origin header) must NOT be able to use this credential to
        bypass the cross-origin protection. A real local CLI or MCP client
        never sends Origin at all, so this costs the legitimate case
        nothing."""
        c, app = open_app
        token = _with_instance_token(app)
        r = c.get("/api/activity", headers={
            "Authorization": f"Bearer {token}",
            "Origin": "http://evil.example",
        })
        assert r.status_code == 403

    def test_instance_token_is_not_a_key_in_protected_mode(self, monkeypatch):
        """Once a real key exists, the instance token must not act as a bypass
        for it."""
        monkeypatch.setenv("LOCALM_API_KEY", "owner-secret-xyz")
        app = create_app(_engine())
        app.state.instance_token = "the-instance-attach-token"
        with TestClient(app) as c:
            r = c.get("/api/activity",
                      headers={"Authorization": "Bearer the-instance-attach-token"})
            assert r.status_code == 401   # not a valid API key

    def test_management_endpoints_also_accept_the_instance_token(self, open_app):
        """The middleware gate is shared by every open-mode management route,
        not special-cased to /api/activity: the same local-process proof works
        everywhere shell_token does."""
        c, app = open_app
        token = _with_instance_token(app)
        r = c.post("/v1/models/unload",
                  headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 200


# --------------------------------------------------------------------------- #
#  The loopback GUI shell injects the token; a LAN bind does not               #
# --------------------------------------------------------------------------- #

@pytest.fixture
def gui_app(monkeypatch):
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    from localm.plugins.gui.web import attach_gui
    app = create_app(_engine())

    async def _switch(name):
        pass
    attach_gui(app, self_url="http://127.0.0.1:9/v1",
               switch_model=_switch, active_model=lambda: "test-model")
    return app


class TestShellTokenInjection:
    def test_loopback_open_mode_injects_token(self, gui_app):
        gui_app.state.bind_host = "127.0.0.1"
        with TestClient(gui_app) as c:
            # Host set to a real loopback literal: TestClient's own default
            # Host ("testserver") is a harness artifact a real browser
            # navigating to 127.0.0.1 never sends, and is not
            # trusted by the no-Origin branch's DNS-rebinding guard.
            html = c.get("/", headers={"Host": "127.0.0.1"}).text
        # The SPA receives the shell token as its apiKey, so app.js sends it as
        # the bearer and the loopback GUI can do management in open mode.
        assert gui_app.state.shell_token in html

    def test_non_loopback_injects_nothing(self, gui_app):
        gui_app.state.bind_host = "0.0.0.0"
        with TestClient(gui_app) as c:
            html = c.get("/").text
        assert gui_app.state.shell_token not in html   # never leak it to a LAN client
        assert "localm.apiKey" not in html             # nothing seeded at all
