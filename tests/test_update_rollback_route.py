# SPDX-License-Identifier: AGPL-3.0-or-later
"""The GUI form of ``localm update --rollback``: GET/POST /api/update/rollback.

Three properties this file pins, in the order they matter:

1. **The probe never performs the action.** ``rollback_info()`` answers "is
   there a backup" WITHOUT calling ``rollback_last()``, which MOVES THE
   INSTALL. An existence check implemented as try-the-action-and-catch would
   roll a user back for opening a settings page.
2. **A partial restore never reads as success, and never restarts.**
   ``_apply_update.rollback`` reports a half-restored tree by raising; that
   outcome must be the loudest, and must not be confused with the benign "there
   is no backup" refusal.
3. **The route is OWNER-only, not merely config:write.** Restoring an older
   build can put back a fixed defect, so a delegated config:write key - which
   drives the rest of the Updates card - must not reach it.

Assertions about a restore lead with the FILES and only then the status code.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import localm.inference.http_server as _hs
from localm import updater
from localm.inference.http_server import create_app

OWNER_KEY = "owner-admin-key-rollback-abc123"


def _engine():
    e = MagicMock()
    e.display_name = "m"
    e.loaded = True
    return e


def _seed_install(tmp_path, monkeypatch, *, with_backup=True):
    """A fake install plus (optionally) the backup an earlier apply() left behind.

    ``marker.txt`` is the discriminator: "new" in the install, "old" in the
    backup. It is a file the fixture can express BOTH values of, so a rollback
    that silently did nothing cannot pass."""
    import json
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("LOCALM_HOME", str(home))
    install = tmp_path / "install"
    install.mkdir()
    (install / "VERSION").write_text("9.9.9\n", encoding="utf-8")
    (install / "marker.txt").write_text("new", encoding="utf-8")
    monkeypatch.setattr(updater, "repo_root", lambda: install)
    if with_backup:
        backup = home / "updates" / "backup"
        backup.mkdir(parents=True)
        (backup / "VERSION").write_text("0.1.0\n", encoding="utf-8")
        (backup / "marker.txt").write_text("old", encoding="utf-8")
        (home / "updates" / "applied_names.json").write_text(
            json.dumps(["VERSION", "marker.txt"]), encoding="utf-8")
    return home, install


def _open_mode(monkeypatch):
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.delenv("LOCALM_REQUIRE_AUTH", raising=False)


@pytest.fixture
def no_restart(monkeypatch):
    """Capture restart requests instead of re-execing the test runner."""
    calls = []
    monkeypatch.setattr(_hs, "_request_restart",
                        lambda *a, **k: calls.append((a, k)))
    return calls


def _auth(app):
    return {"Authorization": f"Bearer {app.state.shell_token}"}


# --------------------------------------------------------------------------- #
#  (1) the probe is read-only                                                  #
# --------------------------------------------------------------------------- #

def test_rollback_info_reports_nothing_and_creates_nothing(tmp_path, monkeypatch):
    """No backup -> available False. And the probe must not bring an ``updates/``
    tree into existence: a status poll on an install that never updated leaves the
    data dir exactly as it found it."""
    home, _install = _seed_install(tmp_path, monkeypatch, with_backup=False)
    info = updater.rollback_info()
    assert info["available"] is False
    assert info["backup"] is None and info["version"] is None
    assert not (home / "updates").exists(), \
        "a read-only probe created the updates dir"


def test_rollback_info_names_the_backed_up_version_without_restoring_it(
        tmp_path, monkeypatch):
    """After the probe the install is UNCHANGED, which is what makes it safe to
    call on every settings load."""
    _home, install = _seed_install(tmp_path, monkeypatch)
    info = updater.rollback_info()
    assert (install / "marker.txt").read_text(encoding="utf-8") == "new", \
        "the probe restored the backup - it must only look"
    assert (install / "VERSION").read_text(encoding="utf-8").strip() == "9.9.9"
    assert info["available"] is True
    assert info["version"] == "0.1.0"
    assert info["backup"] is not None


def test_rollback_info_reports_an_unknown_version_rather_than_guessing(
        tmp_path, monkeypatch):
    """A backup written before VERSION existed (or unreadable) is still restorable.
    Report the version as unknown; never substitute the running one, which would name
    the build the user is trying to leave."""
    home, _install = _seed_install(tmp_path, monkeypatch)
    (home / "updates" / "backup" / "VERSION").unlink()
    info = updater.rollback_info()
    assert info["available"] is True
    assert info["version"] is None
    assert info["current"] != info["version"]


# --------------------------------------------------------------------------- #
#  (2) the routes                                                              #
# --------------------------------------------------------------------------- #

def test_get_route_mirrors_the_probe(tmp_path, monkeypatch):
    _open_mode(monkeypatch)
    _home, install = _seed_install(tmp_path, monkeypatch)
    app = create_app(_engine())
    with TestClient(app) as c:
        r = c.get("/api/update/rollback", headers=_auth(app))
    assert (install / "marker.txt").read_text(encoding="utf-8") == "new", \
        "a GET rolled the install back"
    assert r.status_code == 200
    assert r.json()["available"] is True and r.json()["version"] == "0.1.0"


def test_post_restores_the_previous_build_and_restarts(
        tmp_path, monkeypatch, no_restart):
    _open_mode(monkeypatch)
    _home, install = _seed_install(tmp_path, monkeypatch)
    app = create_app(_engine())
    with TestClient(app) as c:
        r = c.post("/api/update/rollback", headers=_auth(app))
    # The files first: this is the property, the status code is the proxy.
    assert (install / "marker.txt").read_text(encoding="utf-8") == "old"
    assert (install / "VERSION").read_text(encoding="utf-8").strip() == "0.1.0"
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["rolled_back"] is True
    assert body["version"] == "0.1.0", "the reply names what it put back"
    assert body["restarting"] is True
    assert len(no_restart) == 1, \
        "the process must re-exec: the running code was replaced on disk"


def test_post_without_a_backup_is_a_409_and_restarts_nothing(
        tmp_path, monkeypatch, no_restart):
    """Distinct from the partial-restore 500 below. Nothing was touched here."""
    _open_mode(monkeypatch)
    _home, install = _seed_install(tmp_path, monkeypatch, with_backup=False)
    app = create_app(_engine())
    with TestClient(app) as c:
        r = c.post("/api/update/rollback", headers=_auth(app))
    assert (install / "marker.txt").read_text(encoding="utf-8") == "new"
    assert r.status_code == 409, r.text
    assert "backup" in r.text.lower()
    assert no_restart == [], "nothing was rolled back, so nothing may restart"


def test_a_partial_restore_is_reported_as_such_and_does_not_restart(
        tmp_path, monkeypatch, no_restart):
    """_apply_update.rollback raises to report a HALF-RESTORED tree and keeps the
    backup. That must surface as its own failure - not as success, and not as the
    benign 'no backup' refusal - and must not re-exec into a half-restored install."""
    _open_mode(monkeypatch)
    _home, _install = _seed_install(tmp_path, monkeypatch)
    import localm._apply_update as au

    def boom(backup_dir, installed, names):
        raise RuntimeError("could not restore: marker.txt")

    monkeypatch.setattr(au, "rollback", boom)
    app = create_app(_engine())
    with TestClient(app) as c:
        r = c.post("/api/update/rollback", headers=_auth(app))
    assert r.status_code == 500, r.text
    assert "half-restored" in r.text.lower(), \
        "a partial restore must say so, not read as a generic error"
    assert "marker.txt" in r.text, \
        "the failing restore must be named, not generalised away"
    assert "rolled_back" not in r.text.lower()
    assert no_restart == [], \
        "never re-exec into an install that is half-restored"


def test_rollback_is_refused_while_an_update_holds_the_lock(
        tmp_path, monkeypatch, no_restart):
    """apply() and rollback_last() mutate the SAME install tree, and a route
    makes an apply and a rollback two buttons in one Settings card. A rollback
    that interleaved with a swap would remove names the swap is restoring.

    The lock dir is created DIRECTLY (an atomic mkdir at the real path), not
    monkeypatched, so this exercises the cross-process guard the way a separate
    process would contend with it: the CLI in a terminal is a real contender
    that no in-process lock could see."""
    _open_mode(monkeypatch)
    home, install = _seed_install(tmp_path, monkeypatch)
    (home / "updates" / "apply.lock").mkdir()
    (home / "updates" / "apply.lock" / "pid").write_text(str(__import__("os").getpid()),
                                                         encoding="utf-8")
    app = create_app(_engine())
    with TestClient(app) as c:
        r = c.post("/api/update/rollback", headers=_auth(app))
    assert (install / "marker.txt").read_text(encoding="utf-8") == "new",         "a rollback ran while an update held the lock - the tree could be corrupted"
    assert r.status_code == 409, r.text
    assert "already being applied" in r.text
    assert no_restart == [], "nothing was restored, so nothing may restart"


# --------------------------------------------------------------------------- #
#  (3) the owner gate                                                          #
# --------------------------------------------------------------------------- #

@pytest.fixture
def protected(tmp_path, monkeypatch, no_restart):
    """Protected mode: an owner (ADMIN) key plus a scoped config:read/write key."""
    import localm.config as cfg
    home, install = _seed_install(tmp_path, monkeypatch)
    monkeypatch.setattr(cfg, "HOME_DIR", home)
    monkeypatch.setattr(cfg, "MODELS_DIR", home / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", home / "registry.json")
    monkeypatch.setenv("LOCALM_API_KEY", OWNER_KEY)
    from localm import auth
    scoped = auth.create_key(
        "dev", ["config:read", "config:write"], allow_privileged=True)["key"]
    with TestClient(create_app(None)) as c:
        yield c, scoped, install, no_restart


def test_scoped_config_write_key_cannot_roll_the_install_back(protected):
    """config:write drives Check and Update. It must NOT drive a downgrade:
    restoring an older build can put back a defect the newer one fixed."""
    c, scoped, install, restarts = protected
    r = c.post("/api/update/rollback", headers={"Authorization": f"Bearer {scoped}"})
    assert (install / "marker.txt").read_text(encoding="utf-8") == "new", \
        "a refused rollback that still ran would be the worst of both"
    assert r.status_code == 403, r.text
    assert "owner" in r.text.lower()
    assert restarts == []


def test_owner_key_can_roll_the_install_back(protected):
    """The gate hides the capability from a delegated key; it must not remove it
    from the owner, or the GUI would ship a control nobody can use."""
    c, _scoped, install, restarts = protected
    r = c.post("/api/update/rollback", headers={"Authorization": f"Bearer {OWNER_KEY}"})
    assert (install / "marker.txt").read_text(encoding="utf-8") == "old"
    assert r.status_code == 200, r.text
    assert len(restarts) == 1


def test_the_gate_is_specific_not_a_blanket_block(protected):
    """The same scoped key still reads the probe, so the control can render (and
    correctly stay hidden) for a delegated key - this is a write gate, not a loss of
    the scope."""
    c, scoped, _install, _restarts = protected
    r = c.get("/api/update/rollback", headers={"Authorization": f"Bearer {scoped}"})
    assert r.status_code == 200, r.text
    assert r.json()["available"] is True
