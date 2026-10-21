# SPDX-License-Identifier: AGPL-3.0-or-later
"""Folder and file settings refuse UNC and device paths when they are saved.

Driven over the real PATCH /v1/config and POST /v1/media/config/<plugin>
routes: each refused value is answered 400 and config.json is left as it was;
an ordinary local path is still saved.
"""

from __future__ import annotations

import json
import os

import pytest
from fastapi.testclient import TestClient

from localm.inference.http_server import create_app

_OWNER_KEY = "owner-admin-key-unc-paths-0123456789"
_OWNER = {"Authorization": f"Bearer {_OWNER_KEY}"}

_UNSAFE = [
    pytest.param("\\??\\UNC\\localhost\\c$", marks=pytest.mark.skipif(
        os.name != "nt", reason="the NT object-namespace prefix is refused on Windows")),
    "\\\\localhost\\c$\\work",
    "\\\\?\\C:\\work",
    "\\\\.\\PhysicalDrive0",
]


@pytest.fixture
def client(tmp_path, monkeypatch):
    import localm.config as cfg
    home = tmp_path / ".localm"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.setenv("LOCALM_API_KEY", _OWNER_KEY)
    monkeypatch.setattr(cfg, "HOME_DIR", home)
    monkeypatch.setattr(cfg, "MODELS_DIR", home / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", home / "registry.json")
    with TestClient(create_app(None)) as c:
        yield c, home


def _stored(home) -> dict:
    path = home / "config.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


@pytest.mark.parametrize("key", ["binary_dir", "comfy_workdir", "comfy_output_dir",
                                 "tls_cert", "tls_key"])
@pytest.mark.parametrize("value", _UNSAFE)
def test_a_unc_or_device_path_setting_is_refused(client, key, value):
    c, home = client
    before = _stored(home)
    r = c.patch("/v1/config", json={key: value}, headers=_OWNER)
    assert r.status_code == 400, r.text
    assert key in r.json()["detail"]
    assert "UNC and device paths are not allowed" in r.json()["detail"]
    assert _stored(home) == before
    assert c.get("/v1/config", headers=_OWNER).json()[key] != value


@pytest.mark.parametrize("field", ["workdir", "output_dir"])
@pytest.mark.parametrize("value", _UNSAFE)
def test_a_unc_or_device_path_media_folder_is_refused(client, field, value):
    c, home = client
    before = _stored(home)
    r = c.post("/v1/media/config/image", json={field: value}, headers=_OWNER)
    assert r.status_code == 400, r.text
    assert "UNC and device paths are not allowed" in r.json()["detail"]
    assert _stored(home) == before


def test_a_local_folder_is_still_saved(client, tmp_path):
    c, home = client
    folder = str(tmp_path / "comfy")
    r = c.patch("/v1/config", json={"comfy_workdir": folder}, headers=_OWNER)
    assert r.status_code == 200, r.text
    assert _stored(home)["comfy_workdir"] == folder

    r = c.post("/v1/media/config/image", json={"workdir": folder}, headers=_OWNER)
    assert r.status_code == 200, r.text
    assert _stored(home)["plugins"]["image"]["comfy"]["workdir"] == folder


def test_a_blank_folder_still_clears_the_setting(client, tmp_path):
    c, home = client
    assert c.patch("/v1/config", json={"comfy_workdir": str(tmp_path)}, headers=_OWNER).status_code == 200
    r = c.patch("/v1/config", json={"comfy_workdir": ""}, headers=_OWNER)
    assert r.status_code == 200, r.text
    assert "comfy_workdir" not in _stored(home)


@pytest.mark.parametrize("widget", ["path", "folder"])
def test_a_plugin_path_setting_refuses_a_unc_path(widget):
    from localm.plugins.contract import PluginSettingField
    from localm.settings_schema import validate_plugin_settings_update
    fields = [PluginSettingField("target", widget, "Target")]
    with pytest.raises(ValueError, match="UNC and device paths are not allowed"):
        validate_plugin_settings_update(fields, {"target": "\\\\server\\share\\x"})
    assert validate_plugin_settings_update(fields, {"target": "  "}) == {"target": None}


def test_a_scoped_key_save_does_not_echo_an_owner_only_media_folder(client, tmp_path):
    """A non-owner config:write key's PATCH response leaves out the per-plugin
    ComfyUI folder, an owner-only value, as GET /v1/config does."""
    from localm import auth
    c, home = client
    folder = str(tmp_path / "owner-only-comfy-folder")
    assert c.post("/v1/media/config/image", json={"workdir": folder},
                  headers=_OWNER).status_code == 200
    scoped = auth.create_key("writer", ["config:read", "config:write"],
                             allow_privileged=True)["key"]
    hdr = {"Authorization": f"Bearer {scoped}"}
    r = c.patch("/v1/config", json={"temperature": 0.4}, headers=hdr)
    assert r.status_code == 200, r.text
    assert "owner-only-comfy-folder" not in r.text
    assert "owner-only-comfy-folder" not in c.get("/v1/config", headers=hdr).text
    assert "owner-only-comfy-folder" in c.get("/v1/config", headers=_OWNER).text
