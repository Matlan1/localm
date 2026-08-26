# SPDX-License-Identifier: AGPL-3.0-or-later
"""CSRF / drive-by origin guard for state-changing endpoints.

The default CORS policy admits any localhost:PORT origin, so without this guard a
malicious local web page could POST to the management endpoints from the user's
browser even on a keyless install. State-changing methods must be same-origin
(or an explicitly configured cors origin); non-browser clients send no Origin.
"""

import json
import re

import pytest
from fastapi.testclient import TestClient

from localm.inference.http_server import create_app


@pytest.fixture
def client(tmp_path, monkeypatch):
    import localm.config as cfg
    home = tmp_path / ".localm"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.setattr(cfg, "HOME_DIR", home)
    monkeypatch.setattr(cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", home / "registry.json")
    # default config -> localhost CORS regex (the permissive default the guard hardens)
    return TestClient(create_app(None))


def test_cross_origin_patch_refused(client):
    # a page on another localhost port must not drive PATCH /v1/config
    r = client.patch("/v1/config", json={"n_ctx": 8192},
                     headers={"Origin": "http://localhost:9999"})
    assert r.status_code == 403
    assert "cross-origin" in r.json()["detail"].lower()


def test_same_origin_patch_allowed(client):
    # same-origin (Origin host:port matches Host) passes the cross-origin guard;
    # in open mode it must also carry the shell token.
    token = client.app.state.shell_token
    r = client.patch("/v1/config", json={"n_ctx": 8192},
                     headers={"Origin": "http://testserver", "Host": "testserver",
                              "Authorization": f"Bearer {token}"})
    assert r.status_code == 200


def test_no_origin_open_mode_needs_shell_token(client):
    # A no-Origin client (CLI/SDK/curl) can no longer drive open-mode
    # management - the gap this gate closes. Without the shell token -> 403;
    # with it -> 200.
    assert client.patch("/v1/config", json={"n_ctx": 8192}).status_code == 403
    token = client.app.state.shell_token
    r = client.patch("/v1/config", json={"n_ctx": 8192},
                     headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200


def test_cross_origin_safe_get_not_blocked_by_guard(client):
    # GET is a safe method; the guard does not 403 it (CORS handles read exposure)
    r = client.get("/health", headers={"Origin": "http://localhost:9999"})
    assert r.status_code in (200, 503)   # not 403


def test_cross_origin_delete_refused(client):
    r = client.delete("/v1/keys/whatever",
                      headers={"Origin": "http://evil.localhost:8080"})
    assert r.status_code == 403


def test_cross_origin_plugin_data_route_refused(client):
    # A plugin DATA route (e.g. /api/rag) must be same-origin only too, or a
    # localhost-origin page could index and read arbitrary files on a keyless
    # install. The route need not be mounted: the middleware fires before routing.
    r = client.post("/api/rag/collections/x/add",
                    json={"paths": ["whatever"]},
                    headers={"Origin": "http://localhost:9999"})
    assert r.status_code == 403
    assert "cross-origin" in r.json()["detail"].lower()


def test_cross_origin_coder_route_refused(client):
    # The coder agent (shell + file edits) must not be cross-origin drivable.
    r = client.post("/api/coder/sessions", json={},
                    headers={"Origin": "http://evil.localhost:8080"})
    assert r.status_code == 403


def test_inference_api_cross_origin_allowed(client):
    # The OpenAI-compatible inference API stays cross-origin callable so a local
    # app on another port can use it: the guard must NOT 403 it (any other
    # status, 422 or 5xx with no engine, is fine here).
    r = client.post("/v1/chat/completions",
                    json={"model": "x",
                          "messages": [{"role": "user", "content": "hi"}]},
                    headers={"Origin": "http://localhost:9999"})
    assert r.status_code != 403


def test_configured_cors_origin_passes_cross_origin_but_still_needs_token(
        tmp_path, monkeypatch):
    # An allow-listed origin passes the cross-origin guard, but in open mode a
    # state change still needs the shell token: a forgeable Origin header is not
    # a management credential.
    import localm.config as cfg
    home = tmp_path / ".localm"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.setattr(cfg, "HOME_DIR", home)
    monkeypatch.setattr(cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", home / "registry.json")
    from localm.config import save_config
    save_config({"cors_origins": ["https://app.example"]})
    app = create_app(None)
    client = TestClient(app)
    # NEGATIVE: an allow-listed Origin alone no longer bypasses the gate.
    refused = client.patch("/v1/config", json={"n_ctx": 8192},
                           headers={"Origin": "https://app.example"})
    assert refused.status_code == 403
    # With the shell token it goes through.
    ok = client.patch("/v1/config", json={"n_ctx": 8192},
                      headers={"Origin": "https://app.example",
                               "Authorization": f"Bearer {app.state.shell_token}"})
    assert ok.status_code == 200


class TestShellTokenMetadataGetOriginGate:
    """The default CORS policy trusts any http(s)://localhost:PORT /
    127.0.0.1:PORT origin to READ a response, so a hostile local page can steal
    the open-mode shell token from a plain cross-origin GET / and replay it. The
    metadata-GET branch of _origin_guard must reject that replay the same way the
    unsafe-method branch rejects a cross-origin state change: token possession
    alone is not a same-origin proof."""

    def test_cross_origin_metadata_get_with_stolen_token_refused(self, client):
        token = client.app.state.shell_token
        r = client.get("/v1/keys",
                        headers={"Origin": "http://localhost:9999",
                                 "Authorization": f"Bearer {token}"})
        assert r.status_code == 403
        assert "open-mode management" in r.json()["detail"].lower()

    def test_cross_origin_fs_places_with_stolen_token_refused(self, client):
        # the specific route that discloses the real filesystem layout
        token = client.app.state.shell_token
        r = client.get("/api/fs/places",
                        headers={"Origin": "http://localhost:9999",
                                 "Authorization": f"Bearer {token}"})
        assert r.status_code == 403

    def test_same_origin_metadata_get_with_token_still_allowed(self, client):
        # the legitimate loopback GUI shell case must keep working
        token = client.app.state.shell_token
        r = client.get("/v1/keys",
                        headers={"Origin": "http://testserver", "Host": "testserver",
                                 "Authorization": f"Bearer {token}"})
        assert r.status_code == 200

    def test_no_origin_metadata_get_with_token_still_allowed(self, client):
        # a non-browser client (curl/CLI) sends no Origin at all - unaffected
        token = client.app.state.shell_token
        r = client.get("/v1/keys", headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 200

    def test_cross_origin_metadata_get_without_token_still_refused(self, client):
        # sanity: the gate was never bypassable by omitting the token either
        r = client.get("/v1/keys", headers={"Origin": "http://localhost:9999"})
        assert r.status_code == 403

    def test_allowlisted_cors_origin_metadata_get_with_token_allowed(
            self, tmp_path, monkeypatch):
        # an explicitly configured cors_origins entry is trusted the same way
        # the unsafe-method branch already trusts it
        import localm.config as cfg
        home = tmp_path / ".localm"
        home.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("LOCALM_HOME", str(home))
        monkeypatch.setattr(cfg, "HOME_DIR", home)
        monkeypatch.setattr(cfg, "CONFIG_FILE", home / "config.json")
        monkeypatch.setattr(cfg, "REGISTRY_FILE", home / "registry.json")
        from localm.config import save_config
        save_config({"cors_origins": ["https://app.example"]})
        app = create_app(None)
        client = TestClient(app)
        token = app.state.shell_token
        ok = client.get("/v1/keys",
                        headers={"Origin": "https://app.example",
                                 "Authorization": f"Bearer {token}"})
        assert ok.status_code == 200

    def test_wildcard_cors_metadata_get_with_token_still_allowed(
            self, tmp_path, monkeypatch):
        # "cors_origins": "*" opts out of the origin check (an explicit operator
        # choice to trust every origin) but the token itself remains mandatory,
        # for the metadata-GET branch too.
        import localm.config as cfg
        home = tmp_path / ".localm"
        home.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("LOCALM_HOME", str(home))
        monkeypatch.setattr(cfg, "HOME_DIR", home)
        monkeypatch.setattr(cfg, "CONFIG_FILE", home / "config.json")
        monkeypatch.setattr(cfg, "REGISTRY_FILE", home / "registry.json")
        from localm.config import save_config
        save_config({"cors_origins": "*"})
        app = create_app(None)
        client = TestClient(app)
        token = app.state.shell_token
        ok = client.get("/v1/keys",
                        headers={"Origin": "http://localhost:9999",
                                 "Authorization": f"Bearer {token}"})
        assert ok.status_code == 200
        # still refused without the token even under wildcard CORS
        refused = client.get("/v1/keys", headers={"Origin": "http://localhost:9999"})
        assert refused.status_code == 403


class TestShellTokenNotDisclosedByIndexRoute:
    """GET / (the route that carries the shell_token) sits OUTSIDE
    _CROSS_ORIGIN_GET_REFUSED entirely, and _cross_origin_refused itself
    short-circuits to "not refused" under "cors_origins": "*"
    (http_server.py:3355), so neither check runs for GET /.
    test_wildcard_cors_metadata_get_with_token_still_allowed above proves the
    token stays REQUIRED; this asserts the separate property that the token
    cannot be OBTAINED cross-origin under wildcard CORS."""

    def _mounted_app(self, tmp_path, monkeypatch, cors_origins):
        import localm.config as cfg
        home = tmp_path / ".localm"
        home.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("LOCALM_HOME", str(home))
        monkeypatch.setattr(cfg, "HOME_DIR", home)
        monkeypatch.setattr(cfg, "CONFIG_FILE", home / "config.json")
        monkeypatch.setattr(cfg, "REGISTRY_FILE", home / "registry.json")
        from localm.config import save_config
        save_config({"cors_origins": cors_origins})
        from localm.inference.http_server import mount_gui_surface
        app = create_app(None)
        app.state.instance_id = "iid-test"
        app.state.instance_token = "inst-secret-token"
        app.state.instance_mode = "api"
        app.state.instance_port = 8642
        app.state.instance_scheme = "http"
        app.state.bind_host = "127.0.0.1"
        assert mount_gui_surface(app) is True
        return app

    def test_wildcard_cors_cross_origin_get_index_does_not_disclose_token(
            self, tmp_path, monkeypatch):
        # A plain cross-origin fetch("/") under "cors_origins": "*", with no DNS
        # rebinding.
        app = self._mounted_app(tmp_path, monkeypatch, "*")
        client = TestClient(app)
        real_token = app.state.shell_token
        r = client.get("/", headers={"Origin": "https://evil.example"})
        assert r.status_code == 200
        assert real_token not in r.text
        assert "__LOCALM_SHELL_TOKEN__" not in r.text

    def _extract_shell_token(self, html):
        # The real extraction logic an attacker's script would run, attempted
        # rather than asserted as a substring.
        m = re.search(r'__LOCALM_SHELL_TOKEN__=("(?:[^"\\]|\\.)*")', html)
        return json.loads(m.group(1)) if m else None

    def test_wildcard_cors_full_steal_then_mint_chain_fails(
            self, tmp_path, monkeypatch):
        # Extract whatever token a cross-origin GET / actually discloses, using
        # the same extraction an attacker's script would run, then attempt to
        # replay it. Nothing is extractable, so the mint request below carries no
        # credential at all.
        app = self._mounted_app(tmp_path, monkeypatch, "*")
        client = TestClient(app)
        stolen_page = client.get(
            "/", headers={"Origin": "https://evil.example"}).text
        stolen_token = self._extract_shell_token(stolen_page)
        assert stolen_token is None
        headers = {"Origin": "https://evil.example"}
        if stolen_token:
            headers["Authorization"] = f"Bearer {stolen_token}"
        mint = client.post(
            "/v1/keys", json={"name": "stolen", "scopes": ["models:read"]},
            headers=headers)
        assert mint.status_code == 403

    def test_wildcard_cors_legitimate_loopback_gui_still_receives_token(
            self, tmp_path, monkeypatch):
        # The real GUI shell (an ordinary top-level navigation, so no Origin
        # header) still boots with its management token. Host is a real loopback
        # literal: TestClient's own default Host ("testserver") is a harness
        # artefact a real browser never sends, and is not what the no-Origin
        # branch trusts.
        app = self._mounted_app(tmp_path, monkeypatch, "*")
        client = TestClient(app)
        r = client.get("/", headers={"Host": "127.0.0.1:8642"})
        assert r.status_code == 200
        assert app.state.shell_token in r.text
        assert "__LOCALM_SHELL_TOKEN__" in r.text

    def test_wildcard_cors_same_origin_explicit_origin_still_receives_token(
            self, tmp_path, monkeypatch):
        # And not an overcorrection either: an explicit Origin that DOES match
        # Host (a real same-origin fetch/reload) still gets the token.
        app = self._mounted_app(tmp_path, monkeypatch, "*")
        client = TestClient(app)
        r = client.get("/", headers={"Origin": "http://testserver",
                                     "Host": "testserver"})
        assert r.status_code == 200
        assert app.state.shell_token in r.text


class TestDnsRebindingNotDisclosedByIndexRoute:
    """_is_same_origin_document_request's no-Origin branch must not trust ANY
    no-Origin request unconditionally, which is the header shape a DNS-rebinding
    attack produces: the browser considers a follow-up navigation same-origin
    with an attacker's already-open page - Same-Origin Policy is computed from
    the URL STRING navigated to, never the resolved IP - so it sends no Origin
    header even though it lands on this real server under the attacker's own
    domain in Host. This reproduces that chain end to end against the full
    create_app()+mount_gui_surface() wiring, on the DEFAULT cors_origins config;
    the gap does not depend on CORS config at all."""

    def _mounted_app(self, tmp_path, monkeypatch):
        import localm.config as cfg
        home = tmp_path / ".localm"
        home.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("LOCALM_HOME", str(home))
        monkeypatch.setattr(cfg, "HOME_DIR", home)
        monkeypatch.setattr(cfg, "CONFIG_FILE", home / "config.json")
        monkeypatch.setattr(cfg, "REGISTRY_FILE", home / "registry.json")
        from localm.inference.http_server import mount_gui_surface
        app = create_app(None)
        app.state.instance_id = "iid-test"
        app.state.instance_token = "inst-secret-token"
        app.state.instance_mode = "api"
        app.state.instance_port = 8642
        app.state.instance_scheme = "http"
        app.state.bind_host = "127.0.0.1"
        assert mount_gui_surface(app) is True
        return app

    def _extract_shell_token(self, html):
        m = re.search(r'__LOCALM_SHELL_TOKEN__=("(?:[^"\\]|\\.)*")', html)
        return json.loads(m.group(1)) if m else None

    def test_rebound_host_no_origin_does_not_disclose_token(
            self, tmp_path, monkeypatch):
        app = self._mounted_app(tmp_path, monkeypatch)
        client = TestClient(app)
        r = client.get("/", headers={"Host": "evil.example:8642"})
        assert r.status_code == 200
        assert app.state.shell_token not in r.text
        assert "__LOCALM_SHELL_TOKEN__" not in r.text

    def test_rebound_host_full_steal_then_mint_chain_fails(
            self, tmp_path, monkeypatch):
        app = self._mounted_app(tmp_path, monkeypatch)
        client = TestClient(app)
        stolen_page = client.get(
            "/", headers={"Host": "evil.example:8642"}).text
        stolen_token = self._extract_shell_token(stolen_page)
        assert stolen_token is None
        headers = {"Host": "evil.example:8642"}
        if stolen_token:
            headers["Authorization"] = f"Bearer {stolen_token}"
        mint = client.post(
            "/v1/keys", json={"name": "stolen", "scopes": ["models:read"]},
            headers=headers)
        assert mint.status_code == 403

    def test_loopback_literal_host_no_origin_still_receives_token(
            self, tmp_path, monkeypatch):
        # Not an overcorrection: a real loopback-literal Host with no Origin
        # - what an actual local browser navigation sends - still works.
        app = self._mounted_app(tmp_path, monkeypatch)
        client = TestClient(app)
        r = client.get("/", headers={"Host": "127.0.0.1:8642"})
        assert r.status_code == 200
        assert app.state.shell_token in r.text


class TestSensitiveGetCrossOriginRefused:
    """/whoami (root_dir -> the OS username on a loopback bind) and
    /debug/stacks (thread stacks) are UNAUTHENTICATED GETs. The default
    CORS policy hands an ACAO to any http(s)://localhost:PORT origin, so without an
    explicit refusal a drive-by local page could read them cross-origin. They are
    NOT under /api or /v1, so the metadata-GET gate never covered them; and unlike
    those reads they have no route-level auth to fall back on, so the refusal must
    hold in EVERY mode (open and protected)."""

    def test_cross_origin_whoami_refused(self, client):
        # Make the leak concrete: a real root_dir would otherwise be disclosed.
        client.app.state.root_dir = "/home/someuser/project"
        r = client.get("/whoami", headers={"Origin": "http://evil.localhost:1234"})
        assert r.status_code == 403
        assert "cross-origin" in r.json()["detail"].lower()

    def test_cross_origin_debug_stacks_refused(self, client):
        r = client.get("/debug/stacks",
                       headers={"Origin": "http://evil.localhost:1234"})
        assert r.status_code == 403

    def test_same_origin_whoami_allowed(self, client):
        # The legitimate loopback GUI / same-origin discovery case keeps working
        # (and this is exactly what still discloses root_dir to a trusted caller).
        client.app.state.root_dir = "/home/someuser/project"
        r = client.get("/whoami",
                       headers={"Origin": "http://testserver", "Host": "testserver"})
        assert r.status_code == 200
        assert r.json().get("root_dir") == "/home/someuser/project"

    def test_no_origin_whoami_allowed(self, client):
        # A non-browser client (CLI instance discovery) sends no Origin at all -
        # unaffected by the cross-origin refusal.
        assert client.get("/whoami").status_code == 200

    def test_same_origin_debug_stacks_allowed(self, client):
        # Same-origin is not refused by the CROSS-ORIGIN check, which is what
        # this class covers. The shell token is also required in open mode, a
        # separate gate, so it is supplied here.
        r = client.get("/debug/stacks",
                       headers={"Origin": "http://testserver", "Host": "testserver",
                                "Authorization":
                                    f"Bearer {client.app.state.shell_token}"})
        assert r.status_code == 200
        assert "threads" in r.json()

    def test_cross_origin_health_still_not_blocked(self, client):
        # Sanity: only the sensitive GETs are refused; an ordinary safe GET
        # (/health) stays cross-origin readable as before.
        r = client.get("/health", headers={"Origin": "http://evil.localhost:1234"})
        assert r.status_code in (200, 503)

    def test_cross_origin_whoami_refused_in_protected_mode(
            self, tmp_path, monkeypatch):
        # /whoami is unauthenticated in BOTH modes, so the refusal has to hold in
        # protected mode too, not only under the open-mode metadata-GET gate.
        import localm.config as cfg
        home = tmp_path / ".localm"
        home.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("LOCALM_HOME", str(home))
        monkeypatch.setattr(cfg, "HOME_DIR", home)
        monkeypatch.setattr(cfg, "CONFIG_FILE", home / "config.json")
        monkeypatch.setattr(cfg, "REGISTRY_FILE", home / "registry.json")
        from localm.auth import set_api_key
        set_api_key("k" * 32)
        app = create_app(None)
        c = TestClient(app)
        c.app.state.root_dir = "/home/someuser/project"
        r = c.get("/whoami", headers={"Origin": "http://evil.localhost:1234"})
        assert r.status_code == 403
