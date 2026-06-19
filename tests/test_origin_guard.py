# SPDX-License-Identifier: AGPL-3.0-or-later
"""CSRF / drive-by origin guard for state-changing endpoints (SEC-1).

The default CORS policy admits any localhost:PORT origin, so without this guard a
malicious local web page could POST to the management endpoints from the user's
browser even on a keyless install. State-changing methods must be same-origin
(or an explicitly configured cors origin); non-browser clients send no Origin.
"""

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
    # in open mode it must also carry the shell token (H5).
    token = client.app.state.shell_token
    r = client.patch("/v1/config", json={"n_ctx": 8192},
                     headers={"Origin": "http://testserver", "Host": "testserver",
                              "Authorization": f"Bearer {token}"})
    assert r.status_code == 200


def test_no_origin_open_mode_needs_shell_token(client):
    # H5: a no-Origin client (CLI/SDK/curl) can no longer drive open-mode
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
    # SEC: a plugin DATA route (e.g. /api/rag) must be same-origin only too.
    # Before the allowlist flip these were unguarded, letting a localhost-origin
    # page index-and-read arbitrary files via /api/rag on a keyless install.
    # The route need not be mounted - the middleware fires before routing.
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
    # The OpenAI-compatible inference API stays deliberately cross-origin
    # callable so a local app on another port can use it: the guard must NOT
    # 403 it (any other status - 422/5xx with no engine - is fine here).
    r = client.post("/v1/chat/completions",
                    json={"model": "x",
                          "messages": [{"role": "user", "content": "hi"}]},
                    headers={"Origin": "http://localhost:9999"})
    assert r.status_code != 403


def test_configured_cors_origin_allowed(tmp_path, monkeypatch):
    # an explicitly allow-listed origin may drive state-changing endpoints
    import localm.config as cfg
    home = tmp_path / ".localm"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.setattr(cfg, "HOME_DIR", home)
    monkeypatch.setattr(cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", home / "registry.json")
    from localm.config import save_config
    save_config({"cors_origins": ["https://app.example"]})
    client = TestClient(create_app(None))
    r = client.patch("/v1/config", json={"n_ctx": 8192},
                     headers={"Origin": "https://app.example"})
    assert r.status_code == 200
