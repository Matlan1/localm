# SPDX-License-Identifier: AGPL-3.0-or-later
"""Owner-only settings gate.

The RAG indexing settings (rag_indexing_mode / rag_allowed_roots /
rag_denied_roots) are flagged ``admin_only`` because they define a filesystem-READ
boundary: which host folders the RAG indexer may open. Hiding the control in the
GUI is therefore not enough - a non-owner ``config:write`` key must be refused at
the API, or it could add ``Z:\\`` and then index-and-exfil arbitrary files. These
tests prove all three surfaces enforce it server-side:

  - GET /v1/config/schema OMITS the field for a non-owner (no control shipped),
  - GET /v1/config STRIPS its value for a non-owner,
  - PATCH /v1/config 403s the field for a non-owner, while the OWNER can do all.

A plain (non-owner-only) config:write field stays fully usable by the scoped key,
so the gate is specific to admin_only keys, not a blanket block.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from localm.inference.http_server import create_app

OWNER_KEY = "owner-admin-key-abc123"


@pytest.fixture
def app_env(tmp_path, monkeypatch):
    """Protected-mode app: an owner (ADMIN) key via env + a scoped config:write
    key in the keystore, both under an isolated data dir."""
    import localm.config as cfg
    home = tmp_path / ".localm"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.setattr(cfg, "HOME_DIR", home)
    monkeypatch.setattr(cfg, "MODELS_DIR", home / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", home / "registry.json")
    monkeypatch.setenv("LOCALM_API_KEY", OWNER_KEY)   # owner key -> {ADMIN}
    from localm import auth
    scoped = auth.create_key(
        "dev", ["config:read", "config:write"], allow_privileged=True)["key"]
    app = create_app(None)
    with TestClient(app) as c:
        yield c, scoped, tmp_path


def _owner():
    return {"Authorization": f"Bearer {OWNER_KEY}"}


def _scoped(key):
    return {"Authorization": f"Bearer {key}"}


def test_scoped_key_cannot_see_owner_only_fields_in_schema(app_env):
    c, scoped, _ = app_env
    owner_keys = {f["key"] for f in
                  c.get("/v1/config/schema", headers=_owner()).json()["fields"]}
    scoped_keys = {f["key"] for f in
                   c.get("/v1/config/schema", headers=_scoped(scoped)).json()["fields"]}
    rag = {"rag_indexing_mode", "rag_allowed_roots", "rag_denied_roots"}
    assert rag <= owner_keys, "owner must see the owner-only fields"
    assert not (rag & scoped_keys), "scoped key must NOT see any of them"
    # A normal config:write field is still delivered to the scoped key.
    assert "net_allow" in scoped_keys


def test_get_config_strips_owner_only_values_for_scoped_key(app_env):
    c, scoped, tmp_path = app_env
    docs = tmp_path / "docs"
    docs.mkdir()
    assert c.patch("/v1/config", headers=_owner(),
                   json={"rag_allowed_roots": [str(docs)]}).status_code == 200
    owner_cfg = c.get("/v1/config", headers=_owner()).json()
    scoped_cfg = c.get("/v1/config", headers=_scoped(scoped)).json()
    for key in ("rag_indexing_mode", "rag_allowed_roots", "rag_denied_roots"):
        assert key in owner_cfg, f"owner sees {key}"
        assert key not in scoped_cfg, f"{key} value must be hidden from a scoped key"


def test_scoped_key_patch_forbidden_owner_allowed(app_env):
    c, scoped, tmp_path = app_env
    docs = tmp_path / "docs2"
    docs.mkdir()
    body = {"rag_allowed_roots": [str(docs)]}
    denied = c.patch("/v1/config", headers=_scoped(scoped), json=body)
    assert denied.status_code == 403, denied.text
    assert "owner" in denied.text.lower()
    # Even flipping the MODE is owner-only.
    mode_denied = c.patch("/v1/config", headers=_scoped(scoped),
                          json={"rag_indexing_mode": "blacklist"})
    assert mode_denied.status_code == 403, mode_denied.text
    # The SAME scoped key may still change a non-owner-only setting.
    ok = c.patch("/v1/config", headers=_scoped(scoped), json={"net_allow": ["x.com"]})
    assert ok.status_code == 200, ok.text
    # And the owner sets the root successfully.
    owned = c.patch("/v1/config", headers=_owner(), json=body)
    assert owned.status_code == 200, owned.text


def test_net_allow_private_is_owner_only(app_env):
    """net_allow_private DISABLES the SSRF guard server-wide (it permits requests
    to loopback / private / link-local / cloud-metadata targets), so - exactly like
    the rag_* folder settings - it widens a trust boundary and must be owner-only.
    A non-owner config:write key must be refused at the API, or it could flip the
    guard off and then reach internal services via any model-initiated fetch."""
    c, scoped, _ = app_env
    # PATCH is refused for the non-owner config:write key (the guard-disable is
    # owner-only).
    denied = c.patch("/v1/config", headers=_scoped(scoped),
                     json={"net_allow_private": True})
    assert denied.status_code == 403, denied.text
    assert "owner" in denied.text.lower()
    # The control is also hidden from the scoped key's schema (no control shipped).
    scoped_schema = {f["key"] for f in
                     c.get("/v1/config/schema", headers=_scoped(scoped)).json()["fields"]}
    assert "net_allow_private" not in scoped_schema, \
        "guard-disable toggle must not be offered to a non-owner key"
    # ... but the OWNER can still flip it, and the value is visible to the owner.
    ok = c.patch("/v1/config", headers=_owner(), json={"net_allow_private": True})
    assert ok.status_code == 200, ok.text
    assert c.get("/v1/config", headers=_owner()).json().get("net_allow_private") is True
    # A plain (non-owner-only) network field stays fully usable by the scoped key.
    assert c.patch("/v1/config", headers=_scoped(scoped),
                   json={"net_allow": ["x.com"]}).status_code == 200


def test_net_allow_private_value_hidden_from_scoped_key(app_env):
    """The owner-only value is stripped from GET /v1/config for a non-owner key
    (parity with the rag_* keys), so a scoped key cannot even read the current
    SSRF-guard state."""
    c, scoped, _ = app_env
    assert c.patch("/v1/config", headers=_owner(),
                   json={"net_allow_private": True}).status_code == 200
    owner_cfg = c.get("/v1/config", headers=_owner()).json()
    scoped_cfg = c.get("/v1/config", headers=_scoped(scoped)).json()
    assert "net_allow_private" in owner_cfg
    assert "net_allow_private" not in scoped_cfg


def test_cors_origins_is_owner_only(app_env):
    """cors_origins widens trust reach exactly like net_allow_private and the
    bugreport_upload_* / update_* keys do: it names which browser origins may
    call the authenticated API, and "*" opts every unauthenticated-in-every-mode
    sensitive GET (/whoami, /debug/stacks - see _CROSS_ORIGIN_GET_REFUSED in
    http_server.py) out of the cross-origin refusal too. A non-owner
    config:write key must not be able to set it, or it could widen the origin
    trust boundary itself and then read /whoami cross-origin to disclose
    root_dir (the OS username)."""
    c, scoped, _ = app_env
    denied = c.patch("/v1/config", headers=_scoped(scoped),
                     json={"cors_origins": "*"})
    assert denied.status_code == 403, denied.text
    assert "owner" in denied.text.lower()
    scoped_schema = {f["key"] for f in
                     c.get("/v1/config/schema", headers=_scoped(scoped)).json()["fields"]}
    assert "cors_origins" not in scoped_schema, \
        "the origin-widening control must not be offered to a non-owner key"
    ok = c.patch("/v1/config", headers=_owner(), json={"cors_origins": "*"})
    assert ok.status_code == 200, ok.text
    assert c.get("/v1/config", headers=_owner()).json().get("cors_origins") == "*"


def test_patch_response_hides_admin_only_field_values_from_a_scoped_writer(app_env):
    """update_config() returns the FULL merged config (every key, not just the
    ones this call changed), so
    PATCH /v1/config's response must apply the same admin_only_keys() filter
    GET /v1/config already applies above - otherwise a config:write-scoped,
    non-owner key's response echoes back an admin_only secret's CURRENT value
    (e.g. update_token) even though the write itself already refuses to let it
    SET that field."""
    c, scoped, _ = app_env
    from localm.config import update_config
    update_config(lambda cfg: cfg.update({"update_token": "s3cr3t-owner-set-token"}))
    # The scoped (non-owner) key makes an ORDINARY write (something it IS
    # allowed to set) ...
    resp = c.patch("/v1/config", headers=_scoped(scoped), json={"net_allow": ["x.com"]})
    assert resp.status_code == 200, resp.text
    assert "update_token" not in resp.json(), \
        "an admin_only field's value must not be echoed back to a non-owner key"
    # The owner's own PATCH response still carries it.
    owner_resp = c.patch("/v1/config", headers=_owner(), json={"net_allow": ["y.com"]})
    assert owner_resp.json().get("update_token") == "s3cr3t-owner-set-token"


def test_owner_widening_reaches_the_indexer_policy(app_env):
    """The whole point: the owner's API write flows into the RAG indexer's policy,
    so a folder the owner adds becomes indexable. (A folder genuinely outside the
    OS home - e.g. an F:\\ drive - is refused until added; we assert the policy
    linkage, the deciding factor, since pytest's tmp sits under the OS home.)"""
    c, _scoped_key, tmp_path = app_env
    from localm.rag.store import confine_index_path, indexing_policy
    outside = tmp_path / "external-docs"
    outside.mkdir()
    # Before: the folder is not among the indexer's allowed roots.
    assert outside.resolve() not in indexing_policy()["allowed"]
    # Owner widens the allow-list through the API...
    assert c.patch("/v1/config", headers=_owner(),
                   json={"rag_allowed_roots": [str(outside)]}).status_code == 200
    # ...and now it IS an allowed root, and files under it pass the confinement.
    pol = indexing_policy()
    assert outside.resolve() in pol["allowed"]
    target = outside / "note.txt"
    target.write_text("hello", encoding="utf-8")
    assert confine_index_path(target, pol) == target.resolve()
