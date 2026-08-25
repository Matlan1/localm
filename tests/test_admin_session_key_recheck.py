# SPDX-License-Identifier: AGPL-3.0-or-later
"""A revocable key's browser session must not outlive the key it was minted from."""

from __future__ import annotations

import json
import time

import pytest

from localm import scopes as S

KEY = "owner-key-admin-recheck-0123456789"


@pytest.fixture
def home(tmp_path, monkeypatch):
    from pathlib import Path
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path))
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.delenv("LOCALM_REQUIRE_AUTH", raising=False)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", tmp_path)
    monkeypatch.setattr(cfg, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", tmp_path / "registry.json")
    from localm import sessions
    monkeypatch.setattr(sessions, "_CACHE", {"mtime": None, "records": None})
    return tmp_path


def _resolves(sid) -> bool:
    """Does this session still authenticate, through the real gate every cookie consumer uses (never a bare sessions.lookup, which skips the re-check)."""
    from localm.inference.http_server import _principal_from_token
    return _principal_from_token(sid, "cookie") is not None


def _admin_device_session(auth, sessions, **create_kw):
    """An ADMIN-scoped KEYSTORE key and a session minted from it, the way a paired device gets one."""
    created = auth.create_key("device", [S.ADMIN], allow_privileged=True,
                              **create_kw)
    sid = sessions.create(scopes={S.ADMIN}, key_hash=auth._hash_key(created["key"]),
                          fs_access="host", owner_key_minted=False)
    return created, sid


# --------------------------------------------------------------------------- #
#  THE DEFECT                                                                  #
# --------------------------------------------------------------------------- #

def test_a_revoked_admin_device_keys_session_stops_resolving(home, monkeypatch):
    """The reported gap. revoke_key also drops the key's sessions as cleanup, but that cleanup can FAIL (it is best-effort and now warns when it does)."""
    from localm import auth, sessions
    auth.set_api_key(KEY)
    created, sid = _admin_device_session(auth, sessions)
    assert _resolves(sid)                                   # control

    # Neutralise the belt-and-suspenders cleanup so this measures ONLY the
    # per-request re-check, which is the property under test.
    monkeypatch.setattr(sessions, "revoke_by_key_hash", lambda *a, **kw: 0)
    assert auth.revoke_key(created["id"]) is True
    assert sessions.lookup(sid) is not None, \
        "the record was dropped, so this is not measuring the re-check"

    assert not _resolves(sid), (
        "a REVOKED admin-scoped keystore key's cookie still authenticated; a "
        "revocable credential's session must not outlive it")


def test_an_expired_admin_device_keys_session_stops_resolving(home):
    """Expiry is the sharper case: unlike revoke it does NOT delete the record and does NOT drop sessions, so the per-request re-check is the ONLY thing standing between an expired ADMIN device key and a permanently valid cookie."""
    from localm import auth, sessions
    auth.set_api_key(KEY)
    _created, sid = _admin_device_session(auth, sessions,
                                          expires=time.time() + 3600)
    assert _resolves(sid)                                   # control

    ks = auth._load_keystore()
    for rec in ks:
        rec["expires"] = time.time() - 10
    auth._save_keystore(ks)

    assert not _resolves(sid)


# --------------------------------------------------------------------------- #
#  WHAT MUST NOT REGRESS: the owner is why the exemption exists                #
# --------------------------------------------------------------------------- #

def test_the_owners_session_survives_a_key_roll(home):
    """The load-bearing negative."""
    from localm import auth, sessions
    auth.set_api_key(KEY)
    sid = sessions.create(scopes={S.ADMIN}, key_hash=auth._hash_key(KEY),
                          fs_access="host", owner_key_minted=True)
    assert _resolves(sid)

    auth.regenerate_key()

    assert _resolves(sid), \
        "an owner-key roll signed the owner out of their own browser"


def test_a_pre_upgrade_owner_session_survives_a_key_roll(home):
    """The back-compat half, and the one a naive fix breaks silently."""
    from localm import auth, sessions
    auth.set_api_key(KEY)
    sid = sessions.create(scopes={S.ADMIN}, key_hash=auth._hash_key(KEY),
                          fs_access="host", owner_key_minted=True)
    raw = json.loads(sessions.sessions_file().read_text(encoding="utf-8"))
    for rec in raw:
        rec.pop("owner_key_minted", None)       # an older build's record
    sessions.sessions_file().write_text(json.dumps(raw), encoding="utf-8")
    sessions._CACHE["mtime"] = None

    assert _resolves(sid)                       # recognised by value, and stamped
    assert sessions.lookup(sid)["owner_key_minted"] is True

    auth.regenerate_key()

    assert _resolves(sid), (
        "a session minted before the stamp existed was signed out by a key roll - "
        "the exact promise the exemption exists to keep")


def test_a_session_from_the_legacy_owner_digest_is_recognised(home):
    """The owner key's identity moved to a salted KDF."""
    from localm import auth, sessions
    auth.set_api_key(KEY)
    sid = sessions.create(scopes={S.ADMIN},
                          key_hash=auth._legacy_owner_identity(KEY),
                          fs_access="host", owner_key_minted=False)
    assert _resolves(sid)


# --------------------------------------------------------------------------- #
#  The traps that produced two escalations in the equivalent jobs check        #
# --------------------------------------------------------------------------- #

def test_an_unreadable_keystore_cannot_grant_the_exemption(home, monkeypatch):
    """_load_keystore() fails OPEN (returns [] on OSError/ValueError)."""
    from localm import auth, sessions
    auth.set_api_key(KEY)
    _created, sid = _admin_device_session(auth, sessions)
    assert _resolves(sid)                                   # control

    monkeypatch.setattr(auth, "_load_keystore", lambda: [])

    assert not _resolves(sid), \
        "a keystore read failure exempted a revocable key's session"


def test_holding_admin_is_not_enough_on_its_own(home, monkeypatch):
    """ADMIN is NECESSARY but not SUFFICIENT."""
    from localm import auth, sessions
    auth.set_api_key(KEY)
    created, sid = _admin_device_session(auth, sessions)
    assert sessions.lookup(sid)["scopes"] == [S.ADMIN]

    assert _resolves(sid)
    monkeypatch.setattr(sessions, "revoke_by_key_hash", lambda *a, **kw: 0)
    auth.revoke_key(created["id"])
    assert sessions.lookup(sid)["scopes"] == [S.ADMIN]      # unchanged
    assert not _resolves(sid)


def test_a_forged_stamp_on_a_narrower_session_is_not_enough_either(home,
                                                                   monkeypatch):
    """The conjunction the other way round."""
    from localm import auth, sessions
    auth.set_api_key(KEY)
    created = auth.create_key("bot", ["jobs"], allow_privileged=False)
    sid = sessions.create(scopes={"jobs"}, key_hash=auth._hash_key(created["key"]),
                          fs_access="none", owner_key_minted=True)   # tampered
    assert _resolves(sid)                                   # control: key is live

    monkeypatch.setattr(sessions, "revoke_by_key_hash", lambda *a, **kw: 0)
    auth.revoke_key(created["id"])

    assert not _resolves(sid), \
        "a forged owner stamp on a non-admin session bought an exemption"


def test_the_backfill_failing_does_not_break_authentication(home, monkeypatch):
    """The back-fill is best-effort: the request is already correct without it, and the next one re-proves the same way."""
    from localm import auth, sessions
    auth.set_api_key(KEY)
    sid = sessions.create(scopes={S.ADMIN}, key_hash=auth._hash_key(KEY),
                          fs_access="host", owner_key_minted=False)

    def boom(_records):
        raise OSError(13, "The process cannot access the file")

    monkeypatch.setattr(sessions, "_save", boom)

    assert _resolves(sid), "a failed back-fill write broke a valid owner session"


def test_a_plain_scoped_session_is_still_gated(home, monkeypatch):
    """Unchanged behaviour, pinned so the rework did not loosen the ordinary case it was already handling correctly."""
    from localm import auth, sessions
    auth.set_api_key(KEY)
    created = auth.create_key("ro", [S.MODELS_READ], allow_privileged=False)
    sid = sessions.create(scopes={S.MODELS_READ},
                          key_hash=auth._hash_key(created["key"]),
                          fs_access="none", owner_key_minted=False)
    assert _resolves(sid)

    monkeypatch.setattr(sessions, "revoke_by_key_hash", lambda *a, **kw: 0)
    auth.revoke_key(created["id"])

    assert not _resolves(sid)
