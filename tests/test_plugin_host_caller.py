# SPDX-License-Identifier: AGPL-3.0-or-later
"""The plugin Host's caller accessors (``caller_is_owner``, ``caller_principal``,
``caller_owns``) answer for the real request, through a route mounted with
``mount_router`` and real keys from the keystore."""

from __future__ import annotations

from typing import Optional
from unittest.mock import MagicMock

import pytest
from fastapi import APIRouter, FastAPI, Request
from fastapi.testclient import TestClient

from localm import scopes as S
from localm.plugins.contract import Host, PluginSpec
from localm.plugins.engine import PluginHost


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path))
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.delenv("LOCALM_REQUIRE_AUTH", raising=False)
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", tmp_path)
    monkeypatch.setattr(cfg, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", tmp_path / "registry.json")
    return tmp_path


def _client() -> tuple:
    app = FastAPI()
    host = PluginHost(app, MagicMock(), PluginSpec(name="probe", scope=S.CHAT))
    router = APIRouter()

    @router.get("/probe/who")
    def who(request: Request, owner: Optional[str] = None):
        return {"owner": host.caller_is_owner(request),
                "principal": host.caller_principal(request),
                "owns": host.caller_owns(request, owner)}

    host.mount_router(router)
    return TestClient(app), host


def _bearer(key: str) -> dict:
    return {"Authorization": f"Bearer {key}"}


def test_plugin_host_satisfies_the_contract(home):
    _c, host = _client()
    assert isinstance(host, Host)


def test_open_mode_caller_is_the_owner_with_no_principal(home):
    c, _host = _client()
    got = c.get("/probe/who").json()
    assert got == {"owner": True, "principal": None, "owns": True}


def test_keyed_callers_are_told_apart(home, monkeypatch):
    import localm.auth as auth
    monkeypatch.setenv("LOCALM_API_KEY", "owner-secret-for-the-probe-test-0123456789")
    alice = auth.create_key("alice", [S.CHAT])["key"]
    bob = auth.create_key("bob", [S.CHAT])["key"]
    c, _host = _client()

    a = c.get("/probe/who", headers=_bearer(alice)).json()
    b = c.get("/probe/who", headers=_bearer(bob)).json()
    assert a["owner"] is False and b["owner"] is False
    assert a["principal"] and b["principal"] and a["principal"] != b["principal"]
    assert alice not in a["principal"]
    assert a["owns"] is True                                  # owner=None: unrestricted

    assert c.get("/probe/who", params={"owner": a["principal"]},
                 headers=_bearer(alice)).json()["owns"] is True
    assert c.get("/probe/who", params={"owner": a["principal"]},
                 headers=_bearer(bob)).json()["owns"] is False

    owner = c.get("/probe/who", params={"owner": a["principal"]},
                  headers=_bearer("owner-secret-for-the-probe-test-0123456789")).json()
    assert owner["owner"] is True and owner["owns"] is True


def test_no_valid_key_while_keys_exist_is_refused_before_the_route(home, monkeypatch):
    monkeypatch.setenv("LOCALM_API_KEY", "owner-secret-for-the-probe-test-0123456789")
    c, _host = _client()
    assert c.get("/probe/who").status_code == 401
