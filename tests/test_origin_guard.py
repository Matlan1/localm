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
    # same-origin (Origin host:port matches Host) is allowed through the guard
    r = client.patch("/v1/config", json={"n_ctx": 8192},
                     headers={"Origin": "http://testserver", "Host": "testserver"})
    assert r.status_code == 200


def test_no_origin_header_allowed(client):
    # non-browser clients (CLI/SDK) send no Origin and are unaffected
    r = client.patch("/v1/config", json={"n_ctx": 8192})
    assert r.status_code == 200


def test_cross_origin_safe_get_not_blocked_by_guard(client):
    # GET is a safe method; the guard does not 403 it (CORS handles read exposure)
    r = client.get("/health", headers={"Origin": "http://localhost:9999"})
    assert r.status_code in (200, 503)   # not 403


def test_cross_origin_delete_refused(client):
    r = client.delete("/v1/keys/whatever",
                      headers={"Origin": "http://evil.localhost:8080"})
    assert r.status_code == 403


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
