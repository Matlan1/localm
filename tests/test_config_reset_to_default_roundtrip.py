# SPDX-License-Identifier: AGPL-3.0-or-later
"""Resetting a saved number setting to its shipped default through PATCH /v1/config.

The Settings page sends the schema's ``shipped_default`` when a saved number is
blanked: a plain number resets to its default value, and a nullable number
resets to null. These drive the real route; the saved override leaves
config.json and the route reports the default again."""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from localm import settings_schema as ss
from localm.config import DEFAULT_CONFIG
from localm.inference.http_server import create_app

KEY = "reset-default-roundtrip-key"


@pytest.fixture
def client(tmp_path, monkeypatch):
    import localm.config as cfg
    home = tmp_path / ".localm"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.setattr(cfg, "HOME_DIR", home)
    monkeypatch.setattr(cfg, "MODELS_DIR", home / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", home / "registry.json")
    monkeypatch.setenv("LOCALM_API_KEY", KEY)
    with TestClient(create_app(None)) as c:
        yield c, home / "config.json"


def _headers():
    return {"Authorization": f"Bearer {KEY}"}


def _shipped(key):
    field = next(f for f in ss.schema_json() if f["key"] == key)
    return field["shipped_default"]


def test_a_saved_n_ctx_is_reset_by_sending_its_shipped_default(client):
    c, config_file = client
    assert c.patch("/v1/config", json={"n_ctx": 3000}, headers=_headers()).status_code == 200
    assert json.loads(config_file.read_text(encoding="utf-8"))["n_ctx"] == 3000

    r = c.patch("/v1/config", json={"n_ctx": _shipped("n_ctx")}, headers=_headers())

    assert r.status_code == 200, r.text
    assert "n_ctx" not in json.loads(config_file.read_text(encoding="utf-8"))
    assert c.get("/v1/config", headers=_headers()).json()["n_ctx"] == DEFAULT_CONFIG["n_ctx"]


def test_a_saved_nullable_number_is_reset_by_sending_null(client):
    c, config_file = client
    assert DEFAULT_CONFIG["embedding_gpu_layers"] is None
    assert c.patch("/v1/config", json={"embedding_gpu_layers": 7},
                   headers=_headers()).status_code == 200
    assert json.loads(config_file.read_text(encoding="utf-8"))["embedding_gpu_layers"] == 7

    r = c.patch("/v1/config", json={"embedding_gpu_layers": _shipped("embedding_gpu_layers")},
                headers=_headers())

    assert r.status_code == 200, r.text
    assert "embedding_gpu_layers" not in json.loads(config_file.read_text(encoding="utf-8"))
    assert c.get("/v1/config", headers=_headers()).json()["embedding_gpu_layers"] is None
