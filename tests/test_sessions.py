# SPDX-License-Identifier: AGPL-3.0-or-later
"""Opaque server-side browser sessions (localm/sessions.py).

The session store is what decouples "this browser is logged in" from the raw API
key: rolling the key must NOT invalidate live sessions, the id is stored only as a
hash, expiry (absolute + idle) and revocation work, and a corrupt store fails
CLOSED. Each test carries its negative case.
"""

import json
import time

import pytest

from localm import sessions


@pytest.fixture(autouse=True)
def _home(tmp_path, monkeypatch):
    # sessions.sessions_file() resolves via config.home_dir(), which reads
    # LOCALM_HOME at call time - point it at a throwaway dir. Also reset the
    # module-level mtime cache so tests never see each other's writes.
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path))
    monkeypatch.setattr(sessions, "_CACHE", {"mtime": None, "records": None})
    monkeypatch.setattr(sessions, "_last_used_writes", {})
    return tmp_path


def test_create_returns_opaque_id_not_stored_plaintext(_home):
    sid = sessions.create(scopes={"admin"}, key_hash="KH", fs_access="host")
    assert sid and len(sid) >= 20
    # The store must persist only a HASH of the id, never the id itself.
    raw = sessions.sessions_file().read_text(encoding="utf-8")
    assert sid not in raw
    assert sessions._hash(sid) in raw


def test_lookup_roundtrips_the_snapshot(_home):
    sid = sessions.create(scopes={"chat", "models:read"}, key_hash="KH",
                          fs_access="none")
    rec = sessions.lookup(sid)
    assert rec is not None
    assert set(rec["scopes"]) == {"chat", "models:read"}
    assert rec["key_hash"] == "KH"
    assert rec["fs_access"] == "none"


def test_lookup_unknown_and_blank_return_none(_home):
    sessions.create(scopes={"admin"}, key_hash="KH")
    assert sessions.lookup("nope") is None
    assert sessions.lookup("") is None
    assert sessions.lookup(None) is None


def test_revoke_removes_the_session(_home):
    sid = sessions.create(scopes={"admin"}, key_hash="KH")
    assert sessions.lookup(sid) is not None
    assert sessions.revoke(sid) is True
    assert sessions.lookup(sid) is None
    assert sessions.revoke(sid) is False        # already gone


def test_revoke_by_key_hash_drops_that_keys_sessions_only(_home):
    a = sessions.create(scopes={"chat"}, key_hash="KH-A")
    b = sessions.create(scopes={"chat"}, key_hash="KH-A")   # same key, another device
    c = sessions.create(scopes={"admin"}, key_hash="KH-B")  # a different key
    assert sessions.revoke_by_key_hash("KH-A") == 2
    assert sessions.lookup(a) is None
    assert sessions.lookup(b) is None
    assert sessions.lookup(c) is not None                   # untouched
    assert sessions.revoke_by_key_hash("KH-NONE") == 0
    assert sessions.revoke_by_key_hash("") == 0


def test_revoke_all_signs_out_every_device(_home):
    a = sessions.create(scopes={"admin"}, key_hash="KH")
    b = sessions.create(scopes={"chat"}, key_hash="KH2")
    assert sessions.revoke_all() == 2
    assert sessions.lookup(a) is None
    assert sessions.lookup(b) is None


def test_absolute_expiry(_home):
    sid = sessions.create(scopes={"admin"}, key_hash="KH", ttl=-1)  # already past
    assert sessions.lookup(sid) is None


def test_idle_expiry(_home):
    # A long absolute TTL but a zero idle window: unused, it is immediately idle-out.
    sid = sessions.create(scopes={"admin"}, key_hash="KH", ttl=10_000, idle_ttl=-1)
    assert sessions.lookup(sid) is None


def test_sweep_prunes_only_expired(_home, monkeypatch):
    good = sessions.create(scopes={"admin"}, key_hash="KH", ttl=10_000)
    dead = sessions.create(scopes={"chat"}, key_hash="KH2", ttl=10_000)
    # Expire `dead` on disk (create() prunes already-expired rows at mint time, so
    # age an existing one instead), then sweep.
    data = json.loads(sessions.sessions_file().read_text(encoding="utf-8"))
    for r in data:
        if r["key_hash"] == "KH2":
            r["expires"] = time.time() - 1
    sessions.sessions_file().write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr(sessions, "_CACHE", {"mtime": None, "records": None})
    removed = sessions.sweep()
    assert removed == 1
    assert sessions.lookup(good) is not None
    assert sessions.lookup(dead) is None


def test_corrupt_store_fails_closed(_home, monkeypatch):
    sid = sessions.create(scopes={"admin"}, key_hash="KH")
    # Corrupt the file after a successful create; a valid session is then
    # REFUSED (fail closed), not accepted and not treated as an empty store.
    sessions.sessions_file().write_text("{ this is not json", encoding="utf-8")
    monkeypatch.setattr(sessions, "_CACHE", {"mtime": None, "records": None})
    assert sessions.lookup(sid) is None


def test_missing_store_is_simply_empty(_home):
    # A never-created store is absent -> no sessions, no error (fresh install).
    assert not sessions.sessions_file().exists()
    assert sessions.lookup("anything") is None
