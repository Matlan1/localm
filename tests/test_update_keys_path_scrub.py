# SPDX-License-Identifier: AGPL-3.0-or-later
"""Absolute paths in the update and key-listing response bodies.

``GET /api/update/rollback`` (``backup``), ``POST /api/update/apply`` (its ``notes``
and an ``error`` text) and ``GET /v1/keys`` (``owner_key_warning``) name files
under the data dir. A caller who is not the local operator (a scoped key) gets
them through ``path_formatter``; the operator (open mode or an owner key) keeps
the full path. ``POST /api/update/rollback`` is reachable only by the operator,
so it is not scrubbed.

Each scrubbed case is paired with its operator control, so a formatter that
scrubs everyone, or no one, fails.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import localm.config as cfg
import localm.inference.http_server as _hs
from localm import auth, scopes, updater
from localm.bugreport import LocalmError
from localm.inference.http_server import create_app

OWNER_KEY = "owner-admin-key-pathscrub-abc123"


@pytest.fixture
def home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.delenv("LOCALM_REQUIRE_AUTH", raising=False)
    monkeypatch.setattr(cfg, "HOME_DIR", home)
    monkeypatch.setattr(cfg, "MODELS_DIR", home / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", home / "registry.json")
    monkeypatch.setenv("LOCALM_API_KEY", OWNER_KEY)
    return home


def _bearer(key):
    return {"Authorization": f"Bearer {key}"}


def _engine():
    e = MagicMock()
    e.display_name = "m"
    e.loaded = True
    return e


def _seed_backup(home):
    backup = home / "updates" / "backup"
    backup.mkdir(parents=True)
    (backup / "VERSION").write_text("0.1.0\n", encoding="utf-8")
    return backup


def test_rollback_probe_scrubs_backup_for_a_scoped_key_not_for_the_owner(home, tmp_path):
    backup = _seed_backup(home)
    scoped = auth.create_key("dev", [scopes.CONFIG_READ])["key"]
    with TestClient(create_app(_engine())) as c:
        owner = c.get("/api/update/rollback", headers=_bearer(OWNER_KEY)).json()
        theirs = c.get("/api/update/rollback", headers=_bearer(scoped)).json()
    assert owner["backup"] == str(backup)
    assert theirs["available"] is True
    assert str(home) not in theirs["backup"], theirs["backup"]
    assert str(tmp_path) not in theirs["backup"], theirs["backup"]
    assert theirs["backup"].endswith("backup"), "the leaf name must survive the scrub"


def test_rollback_probe_without_a_backup_reports_none_for_both(home):
    scoped = auth.create_key("dev", [scopes.CONFIG_READ])["key"]
    with TestClient(create_app(_engine())) as c:
        theirs = c.get("/api/update/rollback", headers=_bearer(scoped)).json()
    assert theirs["available"] is False
    assert theirs["backup"] is None


@pytest.fixture
def applying(home, monkeypatch):
    """An updater that reports a newer release; what ``updater_handoff.prepare`` answers is
    set per test, and no restart really happens."""
    monkeypatch.setattr(updater, "available", lambda: True)
    monkeypatch.setattr(updater, "check", lambda: {
        "newer": True, "current": "0.1.0", "asset": {"id": "a1"}, "signature": None})
    monkeypatch.setattr(_hs, "_request_restart", lambda *a, **k: None)
    return home


def _apply_as(key) -> dict:
    """``POST /api/update/apply`` with *key*, on an app of its own (one update at a time
    per app)."""
    with TestClient(create_app(_engine())) as c:
        return c.post("/api/update/apply", headers=_bearer(key)).json()


def test_apply_scrubs_the_notes_for_a_scoped_key_not_for_the_owner(applying, monkeypatch):
    from pathlib import Path

    from localm import updater_handoff
    kept = str(applying / "updates" / "data-before-0.2.0")

    def prepare(*_a, **_k):
        out = updater_handoff.Prepared(Path("."), "0" * 32, "0.1.0", "0.2.0", None, True,
                                       None, False, None)
        out.notes = [f"your data is copied to {kept}"]
        return out

    monkeypatch.setattr(updater_handoff, "prepare", prepare)
    scoped = auth.create_key("dev", [scopes.CONFIG_WRITE], allow_privileged=True)["key"]
    owner, theirs = _apply_as(OWNER_KEY), _apply_as(scoped)
    assert owner["notes"] == [f"your data is copied to {kept}"]
    assert theirs["applied"] is True
    assert str(applying) not in theirs["notes"][0], theirs["notes"]


def test_apply_failure_text_is_scrubbed_for_a_scoped_key_not_for_the_owner(
        applying, monkeypatch):
    from localm import updater_handoff
    folder = str(applying / "updates" / "new")

    def fail(*a, **k):
        raise LocalmError("the update was not started; nothing was changed",
                          reason=f"{folder} could not be written")

    monkeypatch.setattr(updater_handoff, "prepare", fail)
    scoped = auth.create_key("dev", [scopes.CONFIG_WRITE], allow_privileged=True)["key"]
    owner, theirs = _apply_as(OWNER_KEY), _apply_as(scoped)
    assert owner["applied"] is False and folder in owner["error"]
    assert theirs["applied"] is False
    assert "the update was not started" in theirs["error"]
    assert str(applying) not in theirs["error"], theirs["error"]


def test_keys_listing_scrubs_the_owner_key_warning_for_a_non_owner(home, monkeypatch):
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.setattr(auth, "_empty_owner_key_warned", False, raising=False)
    monkeypatch.setattr(auth, "_empty_owner_key_downgrade_message", None, raising=False)
    manager = auth.create_key("manager", [scopes.KEYS_ADMIN], allow_privileged=True)["key"]
    owner = auth.create_key("owner", [scopes.ADMIN], allow_privileged=True)["key"]
    auth.key_file().write_text("", encoding="utf-8")
    with TestClient(create_app(None)) as c:
        full = c.get("/v1/keys", headers=_bearer(owner)).json()["owner_key_warning"]
        scrubbed = c.get("/v1/keys", headers=_bearer(manager)).json()["owner_key_warning"]
    assert str(auth.key_file()) in full, full
    assert scrubbed and "auth.key" in scrubbed, "the file name must survive the scrub"
    assert str(home) not in scrubbed, scrubbed
