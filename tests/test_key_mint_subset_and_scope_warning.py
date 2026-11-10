# SPDX-License-Identifier: AGPL-3.0-or-later
"""Minting a key: a non-owner caller may only grant scopes it holds, and an
identifier scope that nothing owns warns with the closest known scope."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from localm import scopes
from localm.inference.http_server import create_app

OWNER_KEY = "owner-admin-key-subset-0123456789"


@pytest.fixture
def env(tmp_path, monkeypatch):
    import localm.config as cfg
    home = tmp_path / ".localm"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.setattr(cfg, "HOME_DIR", home)
    monkeypatch.setattr(cfg, "MODELS_DIR", home / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", home / "registry.json")
    monkeypatch.setenv("LOCALM_API_KEY", OWNER_KEY)
    from localm import auth
    admin_only_key = auth.create_key(
        "minter", ["keys:admin"], allow_privileged=True)["key"]
    with TestClient(create_app(None)) as c:
        yield c, admin_only_key


def _h(key):
    return {"Authorization": f"Bearer {key}"}


def _mint(c, key, scope_list):
    return c.post("/v1/keys", headers=_h(key),
                  json={"name": "n", "scopes": scope_list})


def test_a_keys_admin_key_cannot_mint_a_scope_it_does_not_hold(env):
    c, minter = env
    resp = _mint(c, minter, ["rag"])
    assert resp.status_code == 403, resp.text
    assert "rag" in resp.text
    from localm import auth
    assert [k["name"] for k in auth.list_keys()] == ["minter"]


def test_a_keys_admin_key_can_mint_a_scope_it_holds(env):
    c, minter = env
    resp = _mint(c, minter, ["keys:admin"])
    assert resp.status_code == 200, resp.text
    assert "keys:admin" in resp.json()["scopes"]


def test_the_owner_key_can_mint_any_scope(env):
    c, _ = env
    resp = _mint(c, OWNER_KEY, ["rag", "jobs", "coder"])
    assert resp.status_code == 200, resp.text


def test_a_near_miss_scope_warns_with_the_closest_known_scope(env):
    c, _ = env
    resp = _mint(c, OWNER_KEY, ["modles_write"])
    assert resp.status_code == 200, resp.text
    warns = resp.json().get("warnings", [])
    assert any("closest known scope" in w for w in warns), warns


def test_a_known_scope_does_not_warn_about_being_unknown(env):
    c, _ = env
    resp = _mint(c, OWNER_KEY, ["rag"])
    assert not any("closest known scope" in w or "is not a localm scope" in w
                   for w in resp.json().get("warnings", []))


def test_unknown_scope_warning_names_the_closest_scope():
    out = scopes.unknown_scope_warnings(["broswer"])
    assert len(out) == 1 and "'browser'" in out[0]


def test_unknown_scope_warning_skips_known_and_installed_scopes():
    assert scopes.unknown_scope_warnings(["rag", "models:write", "myplug"],
                                         installed={"myplug"}) == []
