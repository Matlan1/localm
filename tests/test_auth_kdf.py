# SPDX-License-Identifier: AGPL-3.0-or-later
"""The owner key is USER-CHOOSABLE, so its persisted digest gets a real KDF.

``auth._hash_key`` does not only ever see ``secrets.token_urlsafe(32)``: that
holds for named KEYSTORE keys, but ``localm key set KEY`` persists a key the user
provides, and ``LOCALM_API_KEY`` / a hand-edited ``auth.key`` bypass
``set_api_key`` entirely, so they are not even length- or charset-checked.

The harm is not that the digest authenticates (the owner key is verified by a
PLAINTEXT constant-time compare against ``auth.key``). It is that the digest is
PERSISTED - ``sessions.json`` ``key_hash``, ``jobs.json`` ``owner`` - where one
fast unsalted hash is an offline brute-force oracle for the plaintext, which does
authenticate. So these tests assert on what reaches DISK, not merely on what the
function returns.
"""

import hashlib
import json
import os
import subprocess

import pytest


@pytest.fixture
def auth(tmp_path, monkeypatch):
    """localm.auth with a throwaway data dir and a clean auth environment.

    Mirrors tests/test_auth.py's fixture, plus a reset of the process-level
    derivation memo: that cache is module state and would otherwise carry a
    derivation from one test into the next."""
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path))
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.delenv("LOCALM_REQUIRE_AUTH", raising=False)
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", tmp_path)
    monkeypatch.setattr(cfg, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", tmp_path / "registry.json")
    import localm.auth as a
    a._forget_cached_digests()
    yield a
    a._forget_cached_digests()


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _kdf_records(auth) -> list:
    return json.loads(auth.owner_kdf_file().read_text(encoding="utf-8"))["records"]


def _counting_derive(auth, monkeypatch) -> list:
    """Patch the KDF and return a list that receives one entry per invocation."""
    calls: list = []
    real = auth._scrypt_derive

    def counted(key, salt, n, r, p, dklen):
        calls.append(key)
        return real(key, salt, n, r, p, dklen)

    monkeypatch.setattr(auth, "_scrypt_derive", counted)
    return calls


# --------------------------------------------------------------------------- #
#  (a) a key set through set_api_key is stored under a KDF, not a bare sha256  #
# --------------------------------------------------------------------------- #

def test_set_api_key_stores_a_kdf_record_not_a_bare_sha256(auth):
    """The exact thing the alert is about: what lands on disk for a chosen key."""
    key = "correct-horse"            # a plausible human-chosen owner key
    auth.set_api_key(key)

    # It still authenticates.
    from localm import scopes as S
    assert auth.verify(key) == {S.ADMIN}
    assert auth.get_api_key() == key

    records = _kdf_records(auth)
    assert len(records) == 1
    rec = records[0]
    assert rec["alg"] == "scrypt"
    # Salt AND parameters are stored.
    for field in ("salt", "n", "r", "p", "dklen", "digest"):
        assert field in rec, f"{field} missing from the stored record"
    assert len(bytes.fromhex(rec["salt"])) >= 16

    bare = _sha256(key)
    assert rec["digest"] != bare
    # The bare digest appears nowhere in the artefact.
    assert bare not in auth.owner_kdf_file().read_text(encoding="utf-8")
    # And the identity the rest of the system persists is the derived one.
    assert auth._hash_key(key) == rec["digest"] != bare


def test_the_persisted_session_identity_is_not_a_bare_sha256(auth):
    """End-to-end version of (a): what a real login writes into sessions.json.

    sessions.json is one of the two files that actually holds this digest, so
    asserting on _hash_key alone would not prove the artefact changed."""
    from localm import sessions
    key = "correct-horse"
    auth.set_api_key(key)
    sessions.create(scopes=["admin"], key_hash=auth._hash_key(key),
                    fs_access="host")
    on_disk = sessions.sessions_file().read_text(encoding="utf-8")
    assert _sha256(key) not in on_disk


def test_generated_keystore_tokens_stay_on_the_cheap_path(auth):
    """256 bits of CSPRNG output is not brute-forceable at any hash speed, so a
    generated token keeps the fast digest - marked EXPLICITLY on the record
    rather than inferred from the key's shape."""
    rec = auth.create_key("ci", ["chat"])
    stored = json.loads(auth.keystore_file().read_text(encoding="utf-8"))[0]
    assert stored["alg"] == "sha256"
    assert stored["hash"] == _sha256(rec["key"])
    assert auth.verify(rec["key"]) == {"chat"}


def test_a_generated_token_never_runs_the_kdf(auth, monkeypatch):
    """The cheap path must stay cheap: verifying a scoped key must not pay a
    memory-hard derivation, or the hot path becomes a DoS lever."""
    rec = auth.create_key("ci", ["chat"])
    calls = _counting_derive(auth, monkeypatch)
    for _ in range(5):
        assert auth.verify(rec["key"]) == {"chat"}
        auth._hash_key(rec["key"])
    assert calls == []


# --------------------------------------------------------------------------- #
#  (b) an old-format record still verifies, and is upgraded in place           #
# --------------------------------------------------------------------------- #

def test_legacy_keystore_record_verifies_and_is_upgraded_in_place(auth):
    """An install predating the marker must keep working AND get migrated."""
    token = auth.generate_key()
    legacy = [{
        "id": "legacy01", "name": "old", "hash": _sha256(token),
        "scopes": ["chat"], "created": 1.0, "expires": None,
        "fs_access": "none",
        # no "alg": the legacy on-disk shape
    }]
    auth.keystore_file().write_text(json.dumps(legacy, indent=2),
                                    encoding="utf-8")
    before = auth.keystore_file().read_text(encoding="utf-8")
    assert "alg" not in before

    # It still authenticates.
    assert auth.verify(token) == {"chat"}

    after = auth.keystore_file().read_text(encoding="utf-8")
    assert after != before, "the legacy record was not upgraded in place"
    upgraded = json.loads(after)[0]
    assert upgraded["alg"] == "sha256"
    assert upgraded["hash"] == _sha256(token)      # the digest itself is unchanged
    # ...and it still verifies after the rewrite.
    assert auth.verify(token) == {"chat"}
    assert auth.fs_access_for(token) == "none"


def test_an_install_that_never_reauthenticates_is_not_touched(auth):
    """Migration is triggered by a successful verify, never by mere presence, so
    an untouched install cannot be broken by the upgrade."""
    token = auth.generate_key()
    legacy = [{"id": "legacy01", "name": "old", "hash": _sha256(token),
               "scopes": ["chat"], "created": 1.0, "expires": None,
               "fs_access": "none"}]
    raw = json.dumps(legacy, indent=2)
    auth.keystore_file().write_text(raw, encoding="utf-8")
    assert auth.verify("some-other-key") is None
    assert auth.keystore_file().read_text(encoding="utf-8") == raw


def test_legacy_owner_sessions_are_relinked_to_the_derived_identity(auth):
    """The owner key's identity moves from the unsalted digest to the derived
    one. A session minted BEFORE that must follow, or a job created from that
    cookie stops being recognised as the same principal presenting the key as a
    bearer (the parity sessions.create promises)."""
    from localm import sessions
    key = "correct-horse"
    # A pre-upgrade world: auth.key on disk, no KDF record, a session stamped
    # with the legacy digest.
    auth.key_file().write_text(key + "\n", encoding="utf-8")
    sid = sessions.create(scopes=["admin"], key_hash=_sha256(key),
                          fs_access="host")
    assert sessions.lookup(sid)["key_hash"] == _sha256(key)

    derived = auth._hash_key(key)          # first derivation -> triggers migration

    assert derived != _sha256(key)
    assert sessions.lookup(sid)["key_hash"] == derived
    assert sessions.lookup(sid)["scopes"] == ["admin"]     # still valid, not revoked
    assert _sha256(key) not in sessions.sessions_file().read_text(encoding="utf-8")


def test_a_job_stamped_with_the_legacy_owner_digest_stays_the_owners(auth,
                                                                    monkeypatch):
    """jobs.json is the other store holding this digest. A scheduled job stamped
    before the upgrade must not silently lose its owner."""
    from localm.plugins.builtin.jobs.runner import _shell_still_authorized
    key = "correct-horse"
    auth.set_api_key(key)

    class _Job:
        owner = _sha256(key)               # stamped before the upgrade
        owner_is_owner_key = False
        id = "j1"

    job = _Job()
    monkeypatch.setattr(
        "localm.plugins.builtin.jobs.runner._remember_owner_key_job",
        lambda j: None)
    assert _shell_still_authorized(job) is True

    # The same path refuses a digest that is neither the legacy nor the derived
    # identity.
    class _Foreign:
        owner = _sha256("somebody-elses-key")
        owner_is_owner_key = False
        id = "j2"

    assert _shell_still_authorized(_Foreign()) is False


# --------------------------------------------------------------------------- #
#  (c) the KDF runs ONCE per key per process, not per request                  #
# --------------------------------------------------------------------------- #

def test_kdf_runs_once_across_many_verifies(auth, monkeypatch):
    """The whole reason the KDF is allowed on this path at all."""
    key = "correct-horse"
    auth.set_api_key(key)
    auth._forget_cached_digests()          # simulate a fresh process
    calls = _counting_derive(auth, monkeypatch)

    from localm import scopes as S
    for _ in range(10):
        assert auth.verify(key) == {S.ADMIN}
        auth._hash_key(key)

    assert len(calls) == 1, f"KDF ran {len(calls)} times across 10 verifies"


def test_kdf_runs_once_on_the_real_request_path(auth, monkeypatch):
    """The same claim through the function the server actually calls per request,
    rather than through _hash_key directly."""
    from localm.inference.http_server import _principal_from_token
    from localm import scopes as S
    key = "correct-horse"
    auth.set_api_key(key)
    auth._forget_cached_digests()
    calls = _counting_derive(auth, monkeypatch)

    seen = set()
    for _ in range(10):
        held, key_hash, fs, rag_roots = _principal_from_token(key, "header")
        assert S.ADMIN in held and fs == "host"
        seen.add(key_hash)

    assert len(calls) == 1, f"KDF ran {len(calls)} times across 10 requests"
    assert len(seen) == 1, "the derived identity was not stable across requests"
    assert seen.pop() != _sha256(key)


def test_the_derived_identity_survives_a_restart(auth):
    """A fresh process must derive the SAME value, or every stored identity
    (sessions, jobs) would break on restart. This is what the persisted salt is
    for, and a per-call random salt would fail here."""
    key = "correct-horse"
    auth.set_api_key(key)
    first = auth._hash_key(key)
    auth._forget_cached_digests()          # a "restart": memo gone, disk kept
    assert auth._hash_key(key) == first


# --------------------------------------------------------------------------- #
#  (c2) set_api_key's OWN call to the KDF is NOT memoised (unlike verify()),   #
#       so its cost is bounded by _OWNER_KDF_KEEP, not by "once per process"   #
# --------------------------------------------------------------------------- #

def test_owner_kdf_keep_is_small_enough_to_stay_fast(auth):
    """Pins the actual fix: the constant that bounds how many full scrypt
    derivations a single set_api_key call can burn re-verifying stale,
    guaranteed-not-to-match records before minting a new one. A wall-clock
    assertion would be flaky on a busy shared box; this constant IS the bound,
    so assert it directly."""
    assert auth._OWNER_KDF_KEEP <= 4, (
        f"_OWNER_KDF_KEEP={auth._OWNER_KDF_KEEP} lets a single set_api_key "
        "call burn that many extra full scrypt derivations against kept "
        "stale records before minting - keep it small (see the constant's "
        "own comment in auth.py)")


def test_set_api_key_scan_cost_is_bounded_by_owner_kdf_keep(auth, monkeypatch):
    """The mechanism the constant above bounds, pinned directly by COUNTING
    scrypt calls (deterministic - no wall clock): a single set_api_key call
    for a genuinely new key must never run the KDF more than once per KEPT
    record plus once to mint, regardless of how many historical records the
    install has accumulated in total."""
    # Saturate the kept-records list with distinct keys first, then measure ONE
    # MORE call, where every kept record is a guaranteed miss for the new key.
    for i in range(auth._OWNER_KDF_KEEP + 5):
        auth.set_api_key(f"distinct-key-{i}-aaaaaaaa")
    calls = _counting_derive(auth, monkeypatch)

    auth.set_api_key("one-more-new-key-bbbbbbbb")

    assert len(calls) <= auth._OWNER_KDF_KEEP + 1, (
        f"a single set_api_key call ran the KDF {len(calls)} times; expected "
        f"at most {auth._OWNER_KDF_KEEP + 1} (one check per kept record, all "
        "misses, plus one mint) - _owner_kdf_record_for is scanning more than "
        "the kept records")


# --------------------------------------------------------------------------- #
#  (d) a wrong key still fails once the cache is warm                          #
# --------------------------------------------------------------------------- #

def test_wrong_key_fails_after_the_cache_is_warm(auth):
    key = "correct-horse"
    auth.set_api_key(key)
    warm = auth._hash_key(key)             # warm the memo
    assert warm == auth._hash_key(key)

    assert auth.verify("correct-horsf") is None
    assert auth.verify("wrong-key-entirely") is None
    assert auth.verify(key[:-1]) is None
    # A wrong key must not collide with the owner's identity either.
    assert auth._hash_key("correct-horsf") != warm


def test_a_wrong_key_cannot_grow_or_poison_the_cache(auth):
    """The memo holds derived results only, so spraying tokens neither pays the
    KDF nor grows anything - an unbounded per-request cache would be a leak."""
    key = "correct-horse"
    auth.set_api_key(key)
    auth._hash_key(key)
    size = len(auth._digest_cache)
    for i in range(200):
        assert auth.verify(f"junk-token-{i}") is None
        auth._hash_key(f"junk-token-{i}")
    assert len(auth._digest_cache) == size


def test_the_cache_is_bounded(auth):
    """Even by its own (owner-key-only) entries."""
    for i in range(auth._DIGEST_CACHE_MAX + 20):
        auth._cache_put(f"k{i}", f"v{i}")
    assert len(auth._digest_cache) == auth._DIGEST_CACHE_MAX


def test_the_cache_never_holds_the_plaintext(auth):
    key = "correct-horse"
    auth.set_api_key(key)
    auth._hash_key(key)
    assert auth._digest_cache, "nothing was memoised, so this proves nothing"
    for ck, value in auth._digest_cache.items():
        assert key not in ck
        assert key not in value


# --------------------------------------------------------------------------- #
#  (e) the two paths that bypass set_api_key entirely                          #
# --------------------------------------------------------------------------- #

def test_env_var_owner_key_works_end_to_end_and_is_derived(auth, monkeypatch):
    """LOCALM_API_KEY never goes through set_api_key, so it is not length- or
    charset-checked - it is the sharpest version of the user-chosen case."""
    from localm.inference.http_server import _principal_from_token
    from localm import scopes as S
    key = "hunter2"                        # shorter than MIN_KEY_LEN
    monkeypatch.setenv("LOCALM_API_KEY", key)

    assert auth.get_api_key() == key
    assert auth.verify(key) == {S.ADMIN}
    held, key_hash, fs, rag_roots = _principal_from_token(key, "header")
    assert S.ADMIN in held and fs == "host"
    assert key_hash != _sha256(key)
    assert _kdf_records(auth)[0]["alg"] == "scrypt"


def test_hand_written_auth_key_works_end_to_end_and_is_derived(auth):
    """A hand-edited auth.key is the other bypass. Written with a trailing
    newline, as an editor would leave it."""
    from localm.inference.http_server import _principal_from_token
    from localm import scopes as S
    key = "my-lan-box"
    auth.key_file().write_text(key + "\n", encoding="utf-8")

    assert auth.get_api_key() == key
    assert auth.verify(key) == {S.ADMIN}
    held, key_hash, fs, rag_roots = _principal_from_token(key, "header")
    assert S.ADMIN in held and fs == "host"
    assert key_hash != _sha256(key)
    assert _kdf_records(auth)[0]["alg"] == "scrypt"


def test_set_api_key_did_not_get_stricter(auth):
    """Guard on the fix's own blast radius: choosing your own owner key is
    documented behaviour (docs/cli.md), so this change must not have quietly
    removed it by raising MIN_KEY_LEN or tightening the charset."""
    assert auth.MIN_KEY_LEN == 8
    auth.set_api_key("abcd1234")           # exactly MIN_KEY_LEN, human-chosen
    assert auth.get_api_key() == "abcd1234"
    auth.set_api_key("with-dash_and_underscore")
    assert auth.get_api_key() == "with-dash_and_underscore"
    with pytest.raises(ValueError):
        auth.set_api_key("short")


def test_rotating_the_owner_key_changes_the_identity_at_once(auth):
    """The memo must not serve a stale identity after a roll."""
    auth.set_api_key("first-key-aa")
    first = auth._hash_key("first-key-aa")
    auth.set_api_key("second-key-bb")
    second = auth._hash_key("second-key-bb")
    assert second != first
    # ...and the OLD key's record is still on file, so setting it again derives
    # the same identity.
    auth.set_api_key("first-key-aa")
    assert auth._hash_key("first-key-aa") == first


def test_clearing_the_key_removes_the_derivation_records(auth):
    """A clear that leaves credential artefacts behind is not a clear."""
    auth.set_api_key("correct-horse")
    assert auth.owner_kdf_file().exists()
    auth.clear_api_key()
    assert not auth.owner_kdf_file().exists()


def test_a_corrupt_kdf_file_does_not_lock_the_owner_out(auth):
    """The file is a derivation aid, not a credential: nothing authenticates from
    it. Failing closed here would break a working install and buy nothing."""
    from localm import scopes as S
    key = "correct-horse"
    auth.set_api_key(key)
    auth.owner_kdf_file().write_text("{not json", encoding="utf-8")
    auth._forget_cached_digests()
    assert auth.verify(key) == {S.ADMIN}
    assert auth._hash_key(key) != _sha256(key)


def test_an_unknown_hash_alg_refuses_to_match(auth):
    """Fail closed on a record written by a NEWER localm: guessing would let a
    strong record be matched by a weak comparison."""
    token = auth.generate_key()
    auth.keystore_file().write_text(json.dumps([{
        "id": "future01", "name": "future", "hash": _sha256(token),
        "alg": "argon2id-v99", "scopes": ["chat"], "created": 1.0,
        "expires": None, "fs_access": "none"}]), encoding="utf-8")
    assert auth.verify(token) is None


# --------------------------------------------------------------------------- #
#  A failed migration is surfaced and never locks the owner out                #
# --------------------------------------------------------------------------- #
#  Every test below asserts both halves: still authenticated, and the failure
#  was surfaced.

def _captured_warnings(module, monkeypatch) -> list:
    seen: list = []
    monkeypatch.setattr(module.logger, "warning",
                        lambda msg, *a, **k: seen.append(str(msg) % a if a else str(msg)))
    return seen


def test_an_unpersistable_kdf_record_warns_and_does_not_lock_the_owner_out(
        auth, monkeypatch):
    """A read-only data dir must not become an authentication failure."""
    from localm import scopes as S
    key = "correct-horse"
    auth.key_file().write_text(key + "\n", encoding="utf-8")

    def boom(records):
        raise OSError("data dir is read-only")

    monkeypatch.setattr(auth, "_save_owner_kdf", boom)
    warnings = _captured_warnings(auth, monkeypatch)

    derived = auth._hash_key(key)
    assert derived != _sha256(key)              # still derived, not degraded
    assert auth.verify(key) == {S.ADMIN}        # and NOT denied
    assert warnings, "a failed persist was silent"
    assert any("derivation record" in w for w in warnings), warnings


def test_a_failed_session_relink_warns_and_does_not_lock_the_owner_out(
        auth, monkeypatch):
    """The re-link is best-effort; losing it costs a re-login, never access."""
    from localm import scopes as S, sessions
    key = "correct-horse"
    auth.key_file().write_text(key + "\n", encoding="utf-8")
    sid = sessions.create(scopes=["admin"], key_hash=_sha256(key),
                          fs_access="host")

    def boom(old, new):
        raise RuntimeError("store is wedged")

    monkeypatch.setattr(sessions, "relink_key_hash", boom)
    warnings = _captured_warnings(auth, monkeypatch)

    assert auth._hash_key(key) != _sha256(key)
    assert auth.verify(key) == {S.ADMIN}
    assert sessions.lookup(sid) is not None     # the session was NOT dropped
    assert warnings, "a failed re-link was silent"
    assert any("re-link" in w for w in warnings), warnings


def test_a_legacy_record_that_cannot_be_marked_still_verifies(auth, monkeypatch):
    """The keystore marker upgrade is an annotation, not a gate: if it cannot be
    written the key must keep working rather than start failing."""
    token = auth.generate_key()
    auth.keystore_file().write_text(json.dumps([{
        "id": "legacy01", "name": "old", "hash": _sha256(token),
        "scopes": ["chat"], "created": 1.0, "expires": None,
        "fs_access": "none"}]), encoding="utf-8")

    def boom(records):
        raise OSError("keystore is read-only")

    monkeypatch.setattr(auth, "_save_keystore", boom)
    assert auth.verify(token) == {"chat"}
    assert auth.verify(token) == {"chat"}       # and again, not a one-shot


def test_migration_never_silently_downgrades_to_the_weak_digest(auth,
                                                                monkeypatch):
    """The failure that would matter most: falling back to the bare sha256 when
    the KDF path has trouble would restore the exact weakness this fixes."""
    key = "correct-horse"
    auth.key_file().write_text(key + "\n", encoding="utf-8")

    def boom(records):
        raise OSError("nope")

    monkeypatch.setattr(auth, "_save_owner_kdf", boom)
    _captured_warnings(auth, monkeypatch)
    assert auth._hash_key(key) != _sha256(key)


# --------------------------------------------------------------------------- #
#  (f) the ACL half: sessions.json is restricted exactly like auth.key         #
# --------------------------------------------------------------------------- #

def _acl_fingerprint(path):
    """A comparable description of *path*'s permissions on this platform."""
    if os.name == "posix":
        return oct(os.stat(path).st_mode & 0o777)
    out = subprocess.run(["icacls", str(path)], capture_output=True,
                         check=False).stdout.decode("utf-8", "replace")
    # Drop the leading path token on each line and the trailing summary, so two
    # different files with identical ACLs compare equal.
    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    return sorted(ln.replace(str(path), "").strip()
                  for ln in lines if "Successfully processed" not in ln)


def test_sessions_restrict_perms_matches_auth_restrict_perms(auth, tmp_path):
    """Same implementation, same observable result - not merely 'both were
    called'."""
    from localm import sessions
    a_file = tmp_path / "auth-side.txt"
    s_file = tmp_path / "sessions-side.txt"
    a_file.write_text("x", encoding="utf-8")
    s_file.write_text("x", encoding="utf-8")

    assert auth._restrict_perms(a_file) == sessions._restrict_perms(s_file)
    assert _acl_fingerprint(a_file) == _acl_fingerprint(s_file)


def test_sessions_restrict_perms_is_not_posix_only(auth):
    """The actual regression guard. A fires-control is built in: on POSIX the
    file must really be 0600, and on Windows the ACL must really differ from an
    untouched sibling - so this cannot pass by doing nothing."""
    from localm import sessions
    key = "correct-horse"
    auth.set_api_key(key)
    sessions.create(scopes=["admin"], key_hash=auth._hash_key(key),
                    fs_access="host")
    store = sessions.sessions_file()
    assert store.exists()

    untouched = store.parent / "untouched.json"
    untouched.write_text("[]", encoding="utf-8")

    if os.name == "posix":
        # Pin the sibling's mode rather than inheriting the environment's umask.
        os.chmod(untouched, 0o644)
        assert oct(os.stat(store).st_mode & 0o777) == "0o600"
        assert oct(os.stat(untouched).st_mode & 0o777) == "0o644"
    else:
        # The control is the RETURN VALUE plus the contract of
        # `/inheritance:r /grant:r <user>:F`: after it the file has no
        # INHERITED entry. Ambient ACL layout is not asserted on.
        assert sessions._restrict_perms(store) is True
        fingerprint = _acl_fingerprint(store)
        assert not any("(I)" in ace for ace in fingerprint), fingerprint
    # On both platforms: sessions.json (the key digest) is treated exactly like
    # auth.key (the plaintext).
    assert _acl_fingerprint(store) == _acl_fingerprint(auth.key_file())


def test_the_kdf_record_file_is_restricted_too(auth):
    """It is a new file in the data dir holding credential-derived material, so
    it must not be the fourth file that misses the treatment."""
    auth.set_api_key("correct-horse")
    assert _acl_fingerprint(auth.owner_kdf_file()) == \
        _acl_fingerprint(auth.key_file())
