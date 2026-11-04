"""The open-mode gate marks a 403 that rejected the shell token, so the GUI can
tell it from a route's own 403 (a network-policy refusal)."""

import json
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from localm.inference.app_assembly.security import SHELL_REJECTED_HEADER
from localm.inference.http_server import create_app

NET_OFF_ROUTES = [
    ("/api/web/retrieve", {"query": "test"}),
    ("/api/web/search", {"query": "test"}),
    ("/api/web/fetch", {"url": "https://example.invalid/"}),
]


@pytest.fixture
def web_app(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path))
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.delenv("LOCALM_REQUIRE_AUTH", raising=False)
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", tmp_path)
    monkeypatch.setattr(cfg, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", tmp_path / "registry.json")
    (tmp_path / "config.json").write_text(
        json.dumps({"net_mode": "off", "update_url": "", "bugreport_upload_url": ""}),
        encoding="utf-8")
    engine = MagicMock()
    engine.display_name = "test-model"
    app = create_app(engine)
    from localm.plugins.engine import PluginManager
    PluginManager(app, external_root=tmp_path / "noplugins").install("web")
    with TestClient(app) as client:
        yield client, app


@pytest.mark.parametrize("path, body", NET_OFF_ROUTES)
def test_policy_refusal_with_a_valid_shell_token_carries_no_marker(web_app, path, body):
    client, app = web_app
    r = client.post(path, json=body,
                    headers={"Authorization": f"Bearer {app.state.shell_token}"})
    assert r.status_code == 403
    assert "Open-mode management" not in r.json()["detail"]
    assert SHELL_REJECTED_HEADER not in r.headers


@pytest.mark.parametrize("token", [None, "not-the-real-token"])
def test_rejected_shell_token_carries_the_marker(web_app, token):
    client, _ = web_app
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    r = client.post("/api/web/retrieve", json={"query": "test"}, headers=headers)
    assert r.status_code == 403
    assert "Open-mode management" in r.json()["detail"]
    assert r.headers[SHELL_REJECTED_HEADER] == "1"


def test_cross_origin_read_with_a_valid_token_carries_no_marker(web_app):
    client, app = web_app
    r = client.get("/v1/config",
                   headers={"Authorization": f"Bearer {app.state.shell_token}",
                            "Origin": "http://evil.example"})
    assert r.status_code == 403
    assert "Open-mode management" in r.json()["detail"]
    assert SHELL_REJECTED_HEADER not in r.headers
