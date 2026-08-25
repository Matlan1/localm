# SPDX-License-Identifier: AGPL-3.0-or-later
"""Single shared API-key authentication for localm's HTTP surface."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Callable
from typing import Optional

from localm.debuglog import logger

ENV_VAR = "LOCALM_API_KEY"
REQUIRE_ENV_VAR = "LOCALM_REQUIRE_AUTH"
_TRUTHY = ("1", "true", "yes", "on")

# Minimum length for an owner key to authenticate a network bind. Enforced at set
# time and at the network-bind gate (cli._exposed_bind_warning), so a key from the
# env var or a hand-edited auth.key cannot be served to the LAN either.
MIN_KEY_LEN = 8

# Characters an owner key may contain, enforced at set time. Exactly the alphabet
# generate_key() emits, and a strict subset of RFC 7235 token68, so a conforming
# key is safe in an Authorization header. Explicit ASCII classes rather than a
# word class, which would also match non-ASCII letters and digits.
_KEY_CHARSET = re.compile(r"\A[A-Za-z0-9_-]+\Z")


def ct_equal(presented: Optional[str], expected: Optional[str]) -> bool:
    """Constant-time compare of two secrets."""
    # A falsy operand means "no credential presented" / "no secret configured";
    # neither is secret-dependent, so short-circuiting leaks nothing.
    if not presented or not expected:
        return False
    return hmac.compare_digest(presented.encode("utf-8", "surrogatepass"),
                               expected.encode("utf-8", "surrogatepass"))


def key_file() -> Path:
    """Path to the persisted API key, inside the resolved localm data dir."""
    from localm.config import home_dir
    return home_dir() / "auth.key"


def generate_key(nbytes: int = 32) -> str:
    """Return a fresh, URL-safe random key."""
    return secrets.token_urlsafe(nbytes)


# Transient read retry for the owner key file: a concurrent atomic replace, an
# antivirus scanner or an indexer can hold auth.key open briefly on Windows and
# make read_text raise PermissionError. A persistent failure still returns None,
# and any_key_configured() then keeps auth in effect rather than opening.
_KEY_READ_RETRIES = 8
_KEY_READ_BACKOFF = 0.01       # seconds; escalates linearly to the cap
_KEY_READ_BACKOFF_CAP = 0.05


# The three genuinely distinct states auth.key can be in. Keep them distinct:
# collapsing "unreadable" into "absent" silently opens a keyed server, and
# collapsing "holds nothing" into "holds a key" locks its owner out (REG-579).
_KEY_ABSENT = "absent"
_KEY_UNREADABLE = "unreadable"
_KEY_OK = "ok"


def _read_owner_key_file():
    """Read auth.key once and report WHICH state it is in: ``(status, text, err)``."""
    path = key_file()
    for attempt in range(_KEY_READ_RETRIES):
        try:
            return _KEY_OK, path.read_text(encoding="utf-8-sig"), None
        except FileNotFoundError:
            return _KEY_ABSENT, None, None     # genuinely absent -> open by design
        except PermissionError as e:
            # Transient class on Windows. Retry briefly; a persistent failure
            # falls through to unreadable (and the caller's warning).
            if attempt < _KEY_READ_RETRIES - 1:
                time.sleep(min(_KEY_READ_BACKOFF * (attempt + 1),
                               _KEY_READ_BACKOFF_CAP))
                continue
            return _KEY_UNREADABLE, None, e
        except (OSError, ValueError) as e:
            # Not the transient sharing-violation class (a real IO error, a
            # directory in its place, undecodable bytes): do not spin the budget.
            return _KEY_UNREADABLE, None, e
    return _KEY_UNREADABLE, None, None


def _key_text_or_none(text: str) -> Optional[str]:
    """The key *text* holds, or None when it holds none."""
    return text.strip().strip("\x00").strip() or None


def _read_key_file() -> Optional[str]:
    """The persisted owner key, or None when the file is absent or persistently unreadable."""
    status, text, err = _read_owner_key_file()
    if status == _KEY_UNREADABLE:
        logger.warning("owner key file %s exists but is unreadable (%s); "
                       "treating auth as IN EFFECT (fail closed) until it can "
                       "be read - fix its permissions", key_file(), err)
        return None
    if status == _KEY_ABSENT:
        return None
    return _key_text_or_none(text)


def get_api_key() -> Optional[str]:
    """The active key, or None for open mode."""
    env = os.environ.get(ENV_VAR)
    if env and env.strip():
        return env.strip()
    return _read_key_file()


def resolve_bearer_token(instance_token: Optional[str] = None) -> Optional[str]:
    """The bearer credential value (if any) a self-call or management client should present: the owner key (env, else the persisted ``auth.key``) if one is configured, else *instance_token* in OPEN (keyless) mode, else None."""
    key = get_api_key()
    if key:
        return key
    return instance_token or None


def resolve_bearer_headers(instance_token: Optional[str] = None) -> dict:
    """The ``Authorization`` header (if any) a self-call or management client should send - see ``resolve_bearer_token`` for the precedence."""
    token = resolve_bearer_token(instance_token)
    if token:
        return {"Authorization": f"Bearer {token}"}
    return {}


def set_api_key(key: Optional[str]) -> None:
    """Persist *key* to ``auth.key`` (atomic write, owner-only perms)."""
    if not key or not key.strip():
        clear_api_key()
        return
    key = key.strip()
    if len(key) < MIN_KEY_LEN:
        raise ValueError(
            f"API key must be at least {MIN_KEY_LEN} characters long.")
    if not _KEY_CHARSET.match(key):
        # A key is carried in an HTTP Authorization header. Clients send UTF-8
        # while RFC 7230 obs-text decodes latin-1, so a non-ASCII key arrives as
        # mojibake and fails to match; spaces and control characters break or
        # inject into the header. This constrains what may be CHOSEN here; it does
        # not make verify() safe, which is total on its own.
        raise ValueError(
            "API key must use only letters, numbers, '-' and '_' (the characters "
            "'localm key generate' produces). Spaces, punctuation, and non-English "
            "letters cannot be sent reliably in an HTTP Authorization header, so "
            "most clients would fail to authenticate with such a key.")
    from localm.config import ensure_dirs
    ensure_dirs()
    path = key_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_private(path, key.strip() + "\n")
    # The key changed, so any memoised derivation for the OLD one is stale.
    _forget_cached_digests()
    # Derived at set time rather than on the per-request verify path. _owner_digest
    # re-verifies every kept record before minting a new one, so a caller that sets
    # the key repeatedly pays one derivation per kept record; _OWNER_KDF_KEEP bounds
    # that. Best-effort: a failure costs a derivation on first use, not access.
    try:
        _owner_digest(key)
    except Exception as e:
        logger.warning("could not pre-derive the owner key identity (%s); it "
                       "will be derived on first use instead", e)


def clear_api_key() -> list[dict[str, str]]:
    """Remove the persisted key (open mode)."""
    failures: list[dict[str, str]] = []
    try:
        key_file().unlink(missing_ok=True)
    except OSError as e:
        # Surface, do not silence: this is a security step (removing the persisted
        # key). If the file cannot be deleted the key STILL GRANTS access, so a
        # silent pass would imply a clear that did not happen (rule 5). Warn loudly.
        logger.warning("could not remove the API key file %s (%s); the key may "
                       "still be active until it is deleted by hand", key_file(), e)
        failures.append({"what": "the API key file", "path": str(key_file()),
                         "error": f"{type(e).__name__}: {e}"})
    try:
        keystore_file().unlink(missing_ok=True)
    except OSError as e:
        # Same: a leftover keystore means scoped keys remain valid. Do not let a
        # failed delete look like a successful clear.
        logger.warning("could not remove the keystore %s (%s); scoped keys may "
                       "still be active until it is deleted by hand",
                       keystore_file(), e)
        failures.append({"what": "the keystore (scoped device keys)",
                         "path": str(keystore_file()),
                         "error": f"{type(e).__name__}: {e}"})
    # The derivation records describe credentials that no longer exist. They hold
    # no plaintext, but a stale salt would re-link a future identical key to the
    # cleared install's identities.
    try:
        owner_kdf_file().unlink(missing_ok=True)
    except OSError as e:
        logger.warning("could not remove the owner key derivation records %s "
                       "(%s); delete it by hand", owner_kdf_file(), e)
        failures.append({"what": "the owner key derivation records",
                         "path": str(owner_kdf_file()),
                         "error": f"{type(e).__name__}: {e}"})
    _forget_cached_digests()
    return failures


def regenerate_key(nbytes: int = 32) -> str:
    """Generate a new random key, persist it, and return it (for display)."""
    key = generate_key(nbytes)
    set_api_key(key)
    return key


def require_auth_enabled() -> bool:
    """True when the server must refuse requests if no key is configured."""
    if os.environ.get(REQUIRE_ENV_VAR, "").strip().lower() in _TRUTHY:
        return True
    try:
        from localm.config import load_config
        return bool(load_config().get("require_auth", False))
    except Exception:
        # Fail closed: an admin who set require_auth: true must not drop to open
        # mode on a read glitch. load_config() only raises here for something
        # severe; ordinary corrupt config.json is absorbed by its own .bak fallback.
        # LOCALM_REQUIRE_AUTH is checked before this try and is config-independent.
        logger.warning(
            "config unreadable; cannot confirm require_auth - treating as "
            "required (fail closed) until it can be read")
        return True


def _restrict_perms(path: Path) -> bool:
    """Best-effort: restrict the key file to the current user."""
    from localm.config import restrict_file_perms
    return restrict_file_perms(path)


def _atomic_write_private(path: Path, text: str) -> None:
    """Write *text* to *path* atomically, owner-restricted from the moment the bytes first exist on disk."""
    from localm.config import atomic_write_private
    atomic_write_private(path, text)


# --------------------------------------------------------------------------- #
#  Scoped keystore (auth.json) - named keys with explicit scopes              #
# --------------------------------------------------------------------------- #
# The owner key (env LOCALM_API_KEY or auth.key) is implicitly ADMIN. auth.json
# holds additional named keys, each limited to a set of scopes. Only a hash of
# each key is stored; the plaintext is shown once at creation.


# Serializes the read-modify-write of the keystore: create_key and revoke_key both
# load the record list, mutate it and write it back, so without this the last
# writer clobbers the other's change.
_KEYSTORE_LOCK = threading.Lock()


def keystore_file() -> Path:
    """Path to the scoped-key store (inside the localm data dir)."""
    from localm.config import home_dir
    return home_dir() / "auth.json"


# --------------------------------------------------------------------------- #
#  Key digests: a KDF for the user-choosable owner key, fast for generated ones #
# --------------------------------------------------------------------------- #
# Two kinds of secret are digested here:
#
#   * NAMED KEYSTORE KEYS are always secrets.token_urlsafe(32), so 256 bits of
#     CSPRNG output. They stay on the cheap path, marked explicitly on the record
#     rather than inferred from the key's shape.
#   * THE OWNER KEY can be user-chosen and human-memorable, and its digest is
#     persisted (sessions.json key_hash, jobs.json owner), so it gets a salted
#     scrypt derivation.
#
# The digest is also a stable PRINCIPAL IDENTIFIER, recomputed later and compared
# with ==, so it must be deterministic for a given key. The salt is therefore
# persisted per key, and the derivation is memoised per process so the KDF never
# runs more than once per key on the verify path.
_SCRYPT_N = 2 ** 14            # RFC 7914 interactive-login cost
_SCRYPT_R = 8
_SCRYPT_P = 1
_SCRYPT_DKLEN = 32
# 128*N*r = 16 MiB is the actual working set; ask for headroom explicitly rather
# than relying on OpenSSL's default cap, which has varied across versions and
# fails the derivation outright when it is too low.
_SCRYPT_MAXMEM = 64 * 1024 * 1024

# Marker recorded on a stored record naming which construction produced its
# digest, so the cheap path is a declared property rather than a guess from the
# key's shape. A record with no marker is a generated keystore token, reads as
# _ALG_FAST, and is upgraded in place on its next successful verify.
_ALG_FAST = "sha256"
_ALG_KDF = "scrypt"

# How many owner-key verifier records to keep. An install legitimately sees more
# than one owner key over time, and the same key must always derive the same
# digest or job ownership and session identity flip. The cap also bounds how many
# full scrypt derivations one set_api_key call can burn: _owner_kdf_record_for
# re-verifies every kept record before minting a new one.
_OWNER_KDF_KEEP = 3


def owner_kdf_file() -> Path:
    """Path to the owner-key KDF verifier records (inside the localm data dir)."""
    from localm.config import home_dir
    return home_dir() / "auth.kdf.json"


def _fast_digest(key: str) -> str:
    """Unsalted sha256 of a GENERATED 256-bit token."""
    return hashlib.sha256(key.encode("utf-8", "surrogatepass")).hexdigest()


def _legacy_owner_identity(key: str) -> str:
    """The digest the owner key USED to be identified by, before the KDF landed."""
    return _fast_digest(key)


def _memo_key(key: str) -> str:
    """Lookup handle for the in-memory memo: the presented secret's fast digest."""
    return _fast_digest(key)


def _scrypt_derive(key: str, salt: bytes, n: int, r: int, p: int,
                   dklen: int) -> str:
    """Salted scrypt of *key* -> hex."""
    return hashlib.scrypt(key.encode("utf-8", "surrogatepass"), salt=salt,
                          n=n, r=r, p=p, dklen=dklen,
                          maxmem=_SCRYPT_MAXMEM).hex()


# Memoises the expensive derivation only; fast-path digests are never inserted,
# so a caller spraying random bearer tokens cannot grow this. Bounded LRU. Keyed
# on the presented secret's fast digest plus the data home, since one process can
# serve more than one LOCALM_HOME and each has its own salt.
_DIGEST_CACHE_MAX = 64
_digest_cache: "OrderedDict[str, str]" = OrderedDict()
_DIGEST_CACHE_LOCK = threading.Lock()

# Serialises MINTING an owner-key record: without it two concurrent requests with
# a cold memo both mint a fresh salt and both write, stamping two identities.
# Distinct from _DIGEST_CACHE_LOCK, which only guards the dict, because this is
# held across the whole read-derive-write. Taken BEFORE any sessions lock, never
# after. RLock because _hash_key holds it and then calls _owner_digest, which
# takes it again.
_OWNER_KDF_LOCK = threading.RLock()


def _cache_scope() -> str:
    """Scope component for a cache entry: which data home the salt came from."""
    try:
        return str(owner_kdf_file())
    except Exception:
        return ""


def _cache_get(ck: str) -> Optional[str]:
    with _DIGEST_CACHE_LOCK:
        hit = _digest_cache.get(ck)
        if hit is not None:
            _digest_cache.move_to_end(ck)
        return hit


def _cache_put(ck: str, digest: str) -> None:
    with _DIGEST_CACHE_LOCK:
        _digest_cache[ck] = digest
        _digest_cache.move_to_end(ck)
        while len(_digest_cache) > _DIGEST_CACHE_MAX:
            _digest_cache.popitem(last=False)


def _forget_cached_digests() -> None:
    """Drop the memoised derivations."""
    with _DIGEST_CACHE_LOCK:
        _digest_cache.clear()


def _load_owner_kdf() -> list:
    """The owner-key verifier records, or [] when there are none."""
    path = owner_kdf_file()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return []
    except (OSError, ValueError) as e:
        logger.warning("owner key KDF file %s is unreadable or corrupt (%s); "
                       "minting a fresh derivation record. Sessions and jobs "
                       "owned by the owner key are re-linked automatically on "
                       "the next successful verify", path, e)
        return []
    if not isinstance(data, dict):
        return []
    recs = data.get("records")
    return [r for r in recs if isinstance(r, dict)] if isinstance(recs, list) else []


def _save_owner_kdf(records: list) -> None:
    from localm.config import ensure_dirs
    ensure_dirs()
    path = owner_kdf_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    # Restricts the TEMP file, which already holds the whole payload, before the
    # rename, so a crash between the two cannot leave an unrestricted copy.
    # _atomic_write_private also creates the temp already restricted.
    _atomic_write_private(path, json.dumps({"v": 1, "records": records},
                                           indent=2))


def _owner_kdf_record_for(key: str, records: list) -> Optional[dict]:
    """The existing verifier record matching *key*, or None."""
    for r in records:
        if r.get("alg") != _ALG_KDF:
            continue
        try:
            salt = bytes.fromhex(str(r.get("salt", "")))
            digest = _scrypt_derive(key, salt, int(r["n"]), int(r["r"]),
                                    int(r["p"]), int(r["dklen"]))
        except (ValueError, KeyError, TypeError) as e:
            # A hand-edited or truncated row must not break the whole lookup.
            logger.debug("skipping an unusable owner KDF record (%s)", e)
            continue
        if ct_equal(digest, str(r.get("digest", ""))):
            return r
    return None


def _owner_digest(key: str) -> str:
    """The KDF-derived principal identity for the OWNER key *key*."""
    with _OWNER_KDF_LOCK:
        return _owner_digest_locked(key)


def _owner_digest_locked(key: str) -> str:
    records = _load_owner_kdf()
    existing = _owner_kdf_record_for(key, records)
    if existing is not None:
        return str(existing["digest"])
    salt = secrets.token_bytes(16)
    rec = {
        "alg": _ALG_KDF, "salt": salt.hex(), "n": _SCRYPT_N, "r": _SCRYPT_R,
        "p": _SCRYPT_P, "dklen": _SCRYPT_DKLEN, "created": time.time(),
        "digest": _scrypt_derive(key, salt, _SCRYPT_N, _SCRYPT_R, _SCRYPT_P,
                                 _SCRYPT_DKLEN),
    }
    records.append(rec)
    try:
        _save_owner_kdf(records[-_OWNER_KDF_KEEP:])
    except OSError as e:
        # Not fatal: the digest is correct for this process, so auth keeps working.
        # It is not persisted, so the next process mints a different salt and any
        # identity stored under this one stops matching.
        logger.warning("could not persist the owner key derivation record %s "
                       "(%s); sessions and jobs stamped in this process may not "
                       "be recognised after a restart", owner_kdf_file(), e)
    _migrate_legacy_owner_identity(_legacy_owner_identity(key), rec["digest"])
    return str(rec["digest"])


def _migrate_legacy_owner_identity(legacy: str, current: str) -> None:
    """Re-link anything recorded under the owner key's OLD unsalted digest."""
    if not legacy or not current or legacy == current:
        return
    try:
        from localm import sessions
        n = sessions.relink_key_hash(legacy, current)
        if n:
            logger.debug("re-linked %d session(s) to the owner key's derived "
                         "identity", n)
    except Exception as e:
        logger.warning("could not re-link existing sessions to the owner key's "
                       "new derived identity (%s); those sessions stay valid, "
                       "but a job created from one may not be recognised as the "
                       "owner's until the next sign-in", e)


def _is_owner_key(key: str) -> bool:
    """True when *key* is the owner key currently in effect."""
    return ct_equal(key, get_api_key())


def _hash_key(key: str) -> str:
    """The stable identity digest for *key*."""
    ck = _memo_key(key) + "@" + _cache_scope()
    hit = _cache_get(ck)
    if hit is not None:
        return hit
    if not _is_owner_key(key):
        # A generated keystore token, or a token that matches nothing at all
        # (every wrong guess lands here). Cheap, and never cached, so an attacker
        # spraying tokens neither pays nor grows anything.
        return _fast_digest(key)
    with _OWNER_KDF_LOCK:
        # Re-check under the lock: a concurrent request may have minted the
        # record (and warmed the memo) while we waited, and deriving again here
        # would mint a SECOND salt for the same key.
        hit = _cache_get(ck)
        if hit is not None:
            return hit
        derived = _owner_digest(key)
        _cache_put(ck, derived)
    return derived


def _record_digest_for(record: dict, key: str, fast: Callable[[], str]) -> Optional[str]:
    """The digest *key* would produce under *record*'s DECLARED construction, or None when the record cannot be evaluated."""
    alg = record.get("alg") or _ALG_FAST
    if alg == _ALG_FAST:
        return fast()
    if alg == _ALG_KDF:
        try:
            return _scrypt_derive(key, bytes.fromhex(str(record.get("salt", ""))),
                                  int(record["n"]), int(record["r"]),
                                  int(record["p"]), int(record["dklen"]))
        except (ValueError, KeyError, TypeError) as e:
            logger.debug("keystore record %s has unusable KDF parameters (%s); "
                         "it cannot match", record.get("id"), e)
            return None
    logger.warning("keystore record %s declares an unknown hash algorithm %r; "
                   "refusing to match it. A key written by a NEWER localm cannot "
                   "be verified by this one - upgrade rather than editing the "
                   "keystore by hand", record.get("id"), alg)
    return None


def _find_keystore_record(key: str, records: list) -> Optional[dict]:
    """The keystore record *key* authenticates against, or None."""
    _fast_memo: list = []

    def fast() -> str:
        if not _fast_memo:
            _fast_memo.append(_fast_digest(key))
        return _fast_memo[0]

    for r in records:
        cand = _record_digest_for(r, key, fast)
        if cand is None:
            continue
        # ct_equal, not compare_digest: both sides are normally hexdigests, but a
        # hand-edited or corrupted keystore row could hold a non-ASCII "hash" and
        # must fail to match, not 500 every request that reaches this loop.
        if ct_equal(str(r.get("hash", "")), cand):
            return r
    return None


def _mark_record_alg(key_id: Optional[str], alg: str) -> None:
    """Stamp the construction marker onto a legacy record, in place."""
    if not key_id:
        return
    try:
        with _KEYSTORE_LOCK:
            records = _load_keystore()
            for r in records:
                if r.get("id") == key_id and not r.get("alg"):
                    r["alg"] = alg
                    _save_keystore(records)
                    return
    except Exception as e:
        logger.debug("could not stamp the hash-alg marker on key %s (%s); it "
                     "still verifies, the upgrade is retried next time",
                     key_id, e)


def _load_keystore() -> list:
    try:
        data = json.loads(keystore_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return data if isinstance(data, list) else []


def _save_keystore(records: list) -> None:
    from localm.config import ensure_dirs
    ensure_dirs()
    path = keystore_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_private(path, json.dumps(records, indent=2))


# Filesystem-access level a credential may reach on the server host: a graded
# dial (host > shared > none), not a scope, so it can be set below host even on
# an owner's own device. "none" is no host filesystem at all; "host" is the whole
# server filesystem.
#
# "shared" is reserved scaffolding and is NOT enforced anywhere yet:
# require_fs_host() grants only "host", so a "shared" key currently reaches no
# more host filesystem than "none". It is kept out of the `localm key create
# --fs-access` choices until the confinement exists.
FS_ACCESS_LEVELS = ("none", "shared", "host")


def norm_fs_access(level: Optional[str]) -> str:
    """Coerce *level* to a known FS-access level; unknown/blank -> the safe default 'none' (least privilege)."""
    lv = (level or "none").strip().lower()
    return lv if lv in FS_ACCESS_LEVELS else "none"


def fs_access_for(token: str, default: str = "none") -> str:
    """The stored host-filesystem-access level for the key behind *token*, or *default* if the key is unknown or has no level recorded (a legacy key minted before this attribute existed defaults to the safe 'none')."""
    if not token or not token.strip():
        return default
    rec = _find_keystore_record(token.strip(), _load_keystore())
    if rec is None:
        return default
    return norm_fs_access(rec.get("fs_access", default))


def norm_rag_roots(roots) -> list:
    """Coerce *roots* to a clean, order-preserving, de-duplicated list of folder-path strings for a key's per-key RAG-indexing allowlist; anything that is not a non-empty string is dropped."""
    if not roots or not isinstance(roots, (list, tuple)):
        return []
    out: list = []
    seen: set = set()
    for r in roots:
        if not isinstance(r, str):
            continue
        r = r.strip()
        if r and r not in seen:
            seen.add(r)
            out.append(r)
    return out


def rag_roots_for(token: str, default: Optional[list] = None) -> list:
    """The stored per-key RAG-indexing folder allowlist for the key behind *token*, or *default* (``[]`` if not given) if the key is unknown or has no list recorded."""
    if default is None:
        default = []
    if not token or not token.strip():
        return default
    rec = _find_keystore_record(token.strip(), _load_keystore())
    if rec is None:
        return default
    return norm_rag_roots(rec.get("rag_roots", default))


def list_keys() -> list:
    """Public metadata for every named key (never the hash)."""
    return [
        {"id": r.get("id"), "name": r.get("name", ""),
         "scopes": r.get("scopes", []), "created": r.get("created"),
         "expires": r.get("expires"), "last_used": r.get("last_used"),
         "fs_access": norm_fs_access(r.get("fs_access", "none")),
         "rag_roots": norm_rag_roots(r.get("rag_roots", []))}
        for r in _load_keystore()
    ]


def create_key(name: str, scope_list, *, allow_privileged: bool = False,
               expires: Optional[float] = None, fs_access: str = "none",
               rag_roots: Optional[list] = None) -> dict:
    """Mint a named key with *scope_list*, persist its hash, and return a record INCLUDING the plaintext key once - the caller must surface it now, it cannot be recovered."""
    from localm import scopes as S
    clean = S.normalize(scope_list)
    bad = [s for s in clean if not S.is_valid_scope(s)]
    if bad:
        raise ValueError(f"Unknown scope(s): {', '.join(bad)}")
    if not allow_privileged:
        privileged = [s for s in clean if s in S.PRIVILEGED_SCOPES]
        if privileged:
            raise PermissionError(
                "Refusing to grant privileged scope(s): "
                f"{', '.join(privileged)}. Only the owner key may mint keys "
                "with these capabilities."
            )
    key = generate_key()
    record = {
        "id": secrets.token_hex(6),
        "name": (name or "").strip() or "key",
        # _fast_digest, not _hash_key: this key was just generated by
        # generate_key(), so it is 256 bits of CSPRNG output and the cheap digest
        # is sound. The "alg" marker records that on the row, so the verify path
        # reads it as a declared property rather than inferring it from the key's
        # shape.
        "hash": _fast_digest(key),
        "alg": _ALG_FAST,
        "scopes": clean,
        "created": time.time(),
        "expires": float(expires) if expires is not None else None,
        "fs_access": norm_fs_access(fs_access),
        "rag_roots": norm_rag_roots(rag_roots),
    }
    with _KEYSTORE_LOCK:
        records = _load_keystore()
        records.append(record)
        _save_keystore(records)
    return {"id": record["id"], "name": record["name"],
            "scopes": record["scopes"], "expires": record["expires"],
            "fs_access": record["fs_access"], "rag_roots": record["rag_roots"],
            "key": key}


def revoke_key(key_id: str) -> bool:
    """Delete a named key by id."""
    with _KEYSTORE_LOCK:
        records = _load_keystore()
        target = next((r for r in records if r.get("id") == key_id), None)
        remaining = [r for r in records if r.get("id") != key_id]
        if len(remaining) == len(records):
            return False
        _save_keystore(remaining)
    if target and target.get("hash"):
        try:
            from localm import sessions
            if sessions.revoke_by_key_hash(target["hash"]) is None:
                # WARNING, not debug: an ADMIN-scoped session is exempt from the
                # cookie path's per-request key re-check, so for an admin-scoped
                # device key this cleanup is the enforcement and a failed write
                # leaves its cookie working. The key itself is revoked either way.
                logger.warning(
                    "key %s was revoked, but its browser sessions could not be "
                    "dropped; an admin-scoped session for it may still be "
                    "usable until the session store is writable again", key_id)
        except Exception as e:
            logger.debug("session cleanup after key revoke failed (non-fatal): %s", e)
    return True


def key_hash_live(key_hash: Optional[str]) -> bool:
    """True when a keystore key with this sha256 *key_hash* still exists AND has not expired."""
    if not key_hash:
        return False
    now = time.time()
    for r in _load_keystore():
        if r.get("hash") == key_hash:
            exp = r.get("expires")
            return not (exp is not None and now > float(exp))
    return False


def scopes_for_key_hash(key_hash: Optional[str]) -> Optional[set]:
    """The scopes a LIVE keystore key with this sha256 *key_hash* grants, or None."""
    if not key_hash:
        return None
    now = time.time()
    # _load_keystore() returns [] on OSError/ValueError, so a transient unreadable
    # auth.json makes this return None, which callers must read as DENY. That is
    # why the return is Optional rather than a bare set.
    for r in _load_keystore():
        if r.get("hash") == key_hash:
            exp = r.get("expires")
            if exp is not None and now > float(exp):
                return None
            return set(r.get("scopes", []))
    return None


def _keystore_configured() -> bool:
    """True when the scoped keystore should count as 'auth in effect'."""
    path = keystore_file()
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return False                       # absent -> no scoped keys
    except OSError:
        return True                        # exists but unreadable -> fail closed
    try:
        data = json.loads(raw)
    except ValueError:
        return True                        # exists but corrupt -> fail closed
    return bool(data) if isinstance(data, list) else True


# One-shot latch for the empty-auth.key notice (see _owner_key_present). This runs
# on EVERY request via any_key_configured() -> require_auth, so an unthrottled
# warning would put one line per request in the log for a persistent state.
_empty_owner_key_warned = False


def _owner_key_present() -> bool:
    """True when an owner key is in effect: the LOCALM_API_KEY env var is set, OR the auth.key file exists AND is not readably empty."""
    global _empty_owner_key_warned
    env = os.environ.get(ENV_VAR)
    if env and env.strip():
        return True
    status, text, _err = _read_owner_key_file()
    if status == _KEY_ABSENT:
        return False                       # genuinely absent -> open by design
    if status == _KEY_UNREADABLE:
        # Present but unreadable or undecodable: whether a key exists cannot be
        # determined, so fail closed. _read_key_file warns on the value path.
        return True
    if _key_text_or_none(text) is not None:
        # Re-arm the notice: a server that later drops KEYED -> OPEN (the file is
        # truncated while running) must say so again, or the second downgrade
        # would be the silent one.
        _empty_owner_key_warned = False
        return True
    if not _empty_owner_key_warned:
        _empty_owner_key_warned = True
        logger.warning(
            "owner key file %s exists but holds no key; treating it as NO key, "
            "so the server runs in OPEN (keyless) mode - set a key (the "
            "launcher, `localm key set`, or LOCALM_API_KEY) or delete the file",
            key_file())
    return False


def any_key_configured() -> bool:
    """True when auth is in effect: an owner key (env, or an auth.key file holding one - see _owner_key_present) OR a configured scoped keystore."""
    return _owner_key_present() or _keystore_configured()


# Throttle for last-used stamping: verify() runs on every request, so a key's
# last_used is stamped at most once per this many seconds per process, tracked in
# memory so the hot path stays lock-free until a write is due.
_LAST_USED_THROTTLE_S = 300
_last_used_writes: dict = {}
_LAST_USED_LOCK = threading.Lock()


def _touch_last_used(key_hash: str) -> None:
    """Best-effort: stamp a just-verified key's last_used (throttled)."""
    if not key_hash:
        return
    now = time.monotonic()
    with _LAST_USED_LOCK:
        # Dict absence, not a 0.0 sentinel, for "never stamped this process":
        # time.monotonic()'s epoch is platform-defined and is seconds-since-boot on
        # Linux, so `now - 0.0` would wrongly throttle the first stamp on a
        # freshly-booted machine.
        prev = _last_used_writes.get(key_hash)
        if prev is not None and now - prev < _LAST_USED_THROTTLE_S:
            return
        _last_used_writes[key_hash] = now
    try:
        with _KEYSTORE_LOCK:
            records = _load_keystore()
            for r in records:
                # Plain == is fine here (not constant-time): the key is ALREADY
                # verified; this only locates its row to stamp, not authenticating.
                if r.get("hash") == key_hash:
                    r["last_used"] = time.time()
                    _save_keystore(records)
                    break
    except Exception as e:
        # Best-effort: a usage stamp must never break auth. Surface the reason at
        # debug level (RULE 5: do not mute silently) rather than a bare pass.
        logger.debug("last_used stamp failed (non-fatal): %s", e)


def verify(presented: Optional[str]) -> Optional[set]:
    """Resolve a presented bearer token to the set of scopes it grants, or None if it matches nothing."""
    if not presented or not presented.strip():
        return None
    from localm import scopes as S
    presented = presented.strip()
    owner = get_api_key()
    if ct_equal(presented, owner):
        return {S.ADMIN}
    rec = _find_keystore_record(presented, _load_keystore())
    if rec is None:
        return None
    exp = rec.get("expires")
    if exp is not None and time.time() > float(exp):
        return None               # matched a real key, but it has expired
    if not rec.get("alg"):
        # Legacy row: it verified on the cheap path, which is what it has always
        # been. Record that now so the construction is declared rather than
        # assumed on every later verify.
        _mark_record_alg(rec.get("id"), _ALG_FAST)
    _touch_last_used(str(rec.get("hash", "")))
    return set(rec.get("scopes", []))
