# SPDX-License-Identifier: AGPL-3.0-or-later
"""GET /v1/keys reports open_mode: true while no key is configured anywhere,
false once one is."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from localm import scopes as S


@pytest.fixture
def auth(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path))
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.delenv("LOCALM_REQUIRE_AUTH", raising=False)
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", tmp_path)
    monkeypatch.setattr(cfg, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", tmp_path / "registry.json")
    import localm.auth as a
    return a


def test_open_mode_true_until_the_first_key_then_false(auth):
    from localm.inference.http_server import create_app
    assert not auth.any_key_configured()
    app = create_app(None)
    with TestClient(app) as client:
        shell = {"Authorization": f"Bearer {app.state.shell_token}"}
        before = client.get("/v1/keys", headers=shell)
        assert before.status_code == 200
        assert before.json()["open_mode"] is True
        assert before.json()["is_owner"] is False
        r = client.post("/v1/keys", json={"name": "phone", "scopes": [S.CHAT]}, headers=shell)
        assert r.status_code == 200
        after = client.get("/v1/keys")          # the owner session cookie from the reply
        assert after.status_code == 200
        assert after.json()["open_mode"] is False
        assert after.json()["is_owner"] is True


def test_open_mode_false_for_a_keyed_server(auth, monkeypatch):
    from localm.inference.http_server import create_app
    monkeypatch.setenv("LOCALM_API_KEY", "ownersecret-ownersecret-ownersecret")
    app = create_app(None)
    with TestClient(app) as client:
        r = client.get("/v1/keys",
                       headers={"Authorization": "Bearer ownersecret-ownersecret-ownersecret"})
        assert r.status_code == 200
        assert r.json()["open_mode"] is False
