# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Single shared API-key authentication for localm's HTTP surface.

localm has no accounts or login - "auth" is one shared secret (the API key) that
a client presents as ``Authorization: Bearer <key>`` to reach protected
endpoints. This module is the single source of truth for that key.

Key resolution precedence (first hit wins):

    1. LOCALM_API_KEY environment variable (non-empty)  - transient override
    2. <data dir>/auth.key file                         - persistent; written by
       the launcher's Auth card or ``localm key`` and read by every entry point
    3. none                                             - open mode (no key)

When no key resolves the server runs open (local/dev) UNLESS auth is *required*
(``LOCALM_REQUIRE_AUTH`` env var, or config ``"require_auth": true``), in which
case protected endpoints refuse every request until a key is configured.

The empty string is treated as "no key" everywhere (so ``LOCALM_API_KEY=""`` and
an empty file both mean open mode, not a usable empty key).
"""

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
# time and at the network-bind gate (cli._exposed_bind_warning).
MIN_KEY_LEN = 8

# Characters an owner key may contain, enforced at set time. The alphabet
# generate_key() emits, and a strict subset of RFC 7235 token68.
_KEY_CHARSET = re.compile(r"\A[A-Za-z0-9_-]+\Z")


def ct_equal(presented: Optional[str], expected: Optional[str]) -> bool:
    """Constant-time compare of two secrets. Safe for ANY input, never raises.

    Use this for every secret comparison rather than calling hmac.compare_digest()
    on str, which raises TypeError if EITHER operand is a non-ASCII str. Both
    operands are encoded with ``surrogatepass``, so a lone surrogate cannot raise
    either.

    A falsy operand ("no credential presented" / "no secret configured") returns
    False.
    """
    if not presented or not expected:
        return False
    return hmac.compare_digest(presented.encode("utf-8", "surrogatepass"),
                               expected.encode("utf-8", "surrogatepass"))


def key_file() -> Path:
    """Path to the persisted API key, inside the resolved localm data dir.

    Resolved at call time, so it honours LOCALM_HOME and portable installs."""
    from localm.config import home_dir
    return home_dir() / "auth.key"


def generate_key(nbytes: int = 32) -> str:
    """Return a fresh, URL-safe random key. Not persisted - the caller decides
    whether to store it (see ``set_api_key`` / ``regenerate_key``)."""
    return secrets.token_urlsafe(nbytes)


# Transient read retry for the owner key file. A persistent failure still returns
# None, and any_key_configured() then keeps auth in effect rather than opening.
_KEY_READ_RETRIES = 8
_KEY_READ_BACKOFF = 0.01       # seconds; escalates linearly to the cap
_KEY_READ_BACKOFF_CAP = 0.05


# The three distinct states auth.key can be in.
_KEY_ABSENT = "absent"
_KEY_UNREADABLE = "unreadable"
_KEY_OK = "ok"


def _read_owner_key_file():
    """Read auth.key once and report WHICH state it is in: ``(status, text, err)``.

      (_KEY_ABSENT, None, None)     no file -> no owner key, open by design
      (_KEY_UNREADABLE, None, err)  it exists but cannot be read or decoded: we
                                    cannot TELL whether a key exists -> callers
                                    fail CLOSED
      (_KEY_OK, text, None)         we read it; *text* is what it holds (maybe "")

    THE single place that decides what auth.key contains: both the value path
    (get_api_key) and the in-effect path (any_key_configured) go through it.

    A transient Windows sharing violation is ridden out with a bounded retry. The
    file is read as ``utf-8-sig``, so a leading BOM is stripped from *text*."""
    path = key_file()
    for attempt in range(_KEY_READ_RETRIES):
        try:
            return _KEY_OK, path.read_text(encoding="utf-8-sig"), None
        except FileNotFoundError:
            return _KEY_ABSENT, None, None     # genuinely absent -> open by design
        except PermissionError as e:
            # Transient class on Windows: retry briefly, then report unreadable.
            if attempt < _KEY_READ_RETRIES - 1:
                time.sleep(min(_KEY_READ_BACKOFF * (attempt + 1),
                               _KEY_READ_BACKOFF_CAP))
                continue
            return _KEY_UNREADABLE, None, e
        except (OSError, ValueError) as e:
            # Not the transient sharing-violation class: do not spin the budget.
            return _KEY_UNREADABLE, None, e
    return _KEY_UNREADABLE, None, None


def _key_text_or_none(text: str) -> Optional[str]:
    """The key *text* holds, or None when it holds none.

    Whitespace and NUL bytes are stripped; text made only of those means no key,
    exactly like an empty file."""
    return text.strip().strip("\x00").strip() or None


def _read_key_file() -> Optional[str]:
    """The persisted owner key, or None when the file is absent or persistently
    unreadable. A persistent unreadable file returns None but is separately
    treated as auth-in-effect by any_key_configured() (fail closed), and the read
    failure is logged as a warning rather than equated with "no key"."""
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
    """The active key, or None for open mode. Env var (non-empty) wins over the
    persisted file. Empty / whitespace-only values count as no key."""
    env = os.environ.get(ENV_VAR)
    if env and env.strip():
        return env.strip()
    return _read_key_file()


def resolve_bearer_token(instance_token: Optional[str] = None) -> Optional[str]:
    """The bearer credential value (if any) a self-call or management client
    should present: the owner key (env, else the persisted ``auth.key``) if
    one is configured, else *instance_token* in OPEN (keyless) mode, else
    None.

    *instance_token* is used ONLY when no key is configured.
    ``_enforce_request``'s key check (every gated route once a key IS
    configured) accepts a real key only and has no notion of instance
    tokens, so presenting one to a protected-mode server always 401s.

    THE single place this precedence is decided. ``resolve_bearer_headers``
    below is a thin wrapper for callers that want a ready-to-use headers
    dict; callers that build their own request should call this directly.
    """
    key = get_api_key()
    if key:
        return key
    return instance_token or None


def resolve_bearer_headers(instance_token: Optional[str] = None) -> dict:
    """The ``Authorization`` header (if any) a self-call or management client
    should send - see ``resolve_bearer_token`` for the precedence. Returns a
    plain ``dict`` (empty when neither credential is available) rather than
    mutating a caller-supplied one.
    """
    token = resolve_bearer_token(instance_token)
    if token:
        return {"Authorization": f"Bearer {token}"}
    return {}


def set_api_key(key: Optional[str]) -> None:
    """Persist *key* to ``auth.key`` (atomic write, owner-only perms). An empty
    or None *key* clears it, returning the server to open mode.

    Raises ValueError if *key* is shorter than MIN_KEY_LEN or uses characters
    outside _KEY_CHARSET. Both are guards on what a user may CHOOSE here, not a
    promise about what verify() will see: the LOCALM_API_KEY env var and a
    hand-edited auth.key both bypass this function entirely, and verify() stays
    liberal and never raises on whatever it is handed."""
    if not key or not key.strip():
        clear_api_key()
        return
    key = key.strip()
    if len(key) < MIN_KEY_LEN:
        raise ValueError(
            f"API key must be at least {MIN_KEY_LEN} characters long.")
    if not _KEY_CHARSET.match(key):
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
    # Drops any memoised derivation for the previous key.
    _forget_cached_digests()
    # Pre-derive the owner identity at set time. Best-effort: a failure costs a
    # derivation on first use, not access.
    try:
        _owner_digest(key)
    except Exception as e:
        logger.warning("could not pre-derive the owner key identity (%s); it "
                       "will be derived on first use instead", e)


def clear_api_key() -> list[dict[str, str]]:
    """Remove the persisted key (open mode). A leftover env var still applies.

    RETURNS the list of things that could NOT be removed. An EMPTY list means the
    clear genuinely completed; a non-empty one means credentials SURVIVE and the
    caller must not report success. Each entry is::

        {"what":  "the API key file",        # path-free label, safe anywhere
         "path":  "<absolute path>",         # LOCAL surfaces only
         "error": "PermissionError: ..."}    # LOCAL surfaces only

    **A caller on a NETWORK surface must expose only ``what``.** ``path`` is an
    absolute filesystem path and ``error`` is raw OS exception text; the local
    CLI prints all three, a network response must not."""
    failures: list[dict[str, str]] = []
    try:
        key_file().unlink(missing_ok=True)
    except OSError as e:
        logger.warning("could not remove the API key file %s (%s); the key may "
                       "still be active until it is deleted by hand", key_file(), e)
        failures.append({"what": "the API key file", "path": str(key_file()),
                         "error": f"{type(e).__name__}: {e}"})
    try:
        keystore_file().unlink(missing_ok=True)
    except OSError as e:
        logger.warning("could not remove the keystore %s (%s); scoped keys may "
                       "still be active until it is deleted by hand",
                       keystore_file(), e)
        failures.append({"what": "the keystore (scoped device keys)",
                         "path": str(keystore_file()),
                         "error": f"{type(e).__name__}: {e}"})
    # The derivation records describe credentials that no longer exist.
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
    """True when the server must refuse requests if no key is configured.

    Enabled via the ``LOCALM_REQUIRE_AUTH`` env var or config
    ``"require_auth": true``. Default false keeps loopback installs keyless.

    On a config-read failure this resolves to True (fail closed)."""
    if os.environ.get(REQUIRE_ENV_VAR, "").strip().lower() in _TRUTHY:
        return True
    try:
        from localm.config import load_config
        return bool(load_config().get("require_auth", False))
    except Exception:
        logger.warning(
            "config unreadable; cannot confirm require_auth - treating as "
            "required (fail closed) until it can be read")
        return True


def _restrict_perms(path: Path) -> bool:
    """Best-effort: restrict the key file to the current user. No-op on failure
    or unsupported platforms. Returns True when the tightening is believed to
    have happened.

    Delegates to ``config.restrict_file_perms``. The bool is PASSED THROUGH so a
    caller doing the atomic temp+replace dance can restrict the temp file and
    retry on the destination only when the first attempt failed."""
    from localm.config import restrict_file_perms
    return restrict_file_perms(path)


def _atomic_write_private(path: Path, text: str) -> None:
    """Write *text* to *path* atomically, owner-restricted from the moment the
    bytes first exist on disk.

    Delegates to ``config.atomic_write_private``; the bool that returns is
    dropped, since reporting a failed tightening is ``restrict_file_perms``'s
    own job."""
    from localm.config import atomic_write_private
    atomic_write_private(path, text)


# --------------------------------------------------------------------------- #
#  Scoped keystore (auth.json) - named keys with explicit scopes              #
# --------------------------------------------------------------------------- #
# The owner key (env LOCALM_API_KEY or auth.key) is implicitly ADMIN. auth.json
# holds additional named keys, each limited to a set of scopes. Only a hash of
# each key is stored; the plaintext is shown once at creation.


# Serializes the read-modify-write of the keystore (create_key, revoke_key).
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
#   * NAMED KEYSTORE KEYS are always secrets.token_urlsafe(32). They stay on the
#     cheap path, marked explicitly on the record.
#   * THE OWNER KEY can be user-chosen, and its digest is persisted
#     (sessions.json key_hash, jobs.json owner), so it gets a salted scrypt
#     derivation.
#
# The digest is also a stable PRINCIPAL IDENTIFIER, recomputed later and compared
# with ==, so it must be deterministic for a given key. The salt is persisted per
# key, and the derivation is memoised per process.
_SCRYPT_N = 2 ** 14            # RFC 7914 interactive-login cost
_SCRYPT_R = 8
_SCRYPT_P = 1
_SCRYPT_DKLEN = 32
# 128*N*r = 16 MiB is the actual working set; this asks for headroom explicitly
# rather than relying on OpenSSL's default cap.
_SCRYPT_MAXMEM = 64 * 1024 * 1024

# Marker recorded on a stored record naming which construction produced its
# digest. A record with no marker is a generated keystore token, reads as
# _ALG_FAST, and is upgraded in place on its next successful verify.
_ALG_FAST = "sha256"
_ALG_KDF = "scrypt"

# How many owner-key verifier records to keep. Also bounds how many full scrypt
# derivations one set_api_key call can burn, since _owner_kdf_record_for
# re-verifies every kept record before minting a new one.
_OWNER_KDF_KEEP = 3


def owner_kdf_file() -> Path:
    """Path to the owner-key KDF verifier records (inside the localm data dir)."""
    from localm.config import home_dir
    return home_dir() / "auth.kdf.json"


def _fast_digest(key: str) -> str:
    """Unsalted sha256 of a GENERATED 256-bit token. Never call this on a secret
    a human may have chosen - that is what the KDF path exists for.

    Encodes with ``surrogatepass``, so a lone surrogate cannot raise;
    byte-identical to ``encode("utf-8")`` for every key that encodes at all."""
    return hashlib.sha256(key.encode("utf-8", "surrogatepass")).hexdigest()


def _legacy_owner_identity(key: str) -> str:
    """The digest the owner key USED to be identified by, before the KDF landed.

    Confined to the MIGRATION path and never persisted: it FINDS rows still
    carrying the old identity so they can be rewritten to the derived one (and,
    in the jobs plugin, recognises a job stamped before the upgrade so it is not
    orphaned). Nothing is stored under it and nothing authenticates from it.

    A separate function from _fast_digest despite the identical body, so the two
    uses cannot be confused."""
    return _fast_digest(key)


def _memo_key(key: str) -> str:
    """Lookup handle for the in-memory memo: the presented secret's fast digest.

    Never persisted and never leaves the process; the digest that IS written to
    disk for the owner key is the scrypt derivation."""
    return _fast_digest(key)


def _scrypt_derive(key: str, salt: bytes, n: int, r: int, p: int,
                   dklen: int) -> str:
    """Salted scrypt of *key* -> hex."""
    return hashlib.scrypt(key.encode("utf-8", "surrogatepass"), salt=salt,
                          n=n, r=r, p=p, dklen=dklen,
                          maxmem=_SCRYPT_MAXMEM).hex()


# Memoises the expensive derivation only; fast-path digests are never inserted.
# Bounded LRU, keyed on the presented secret's fast digest plus the data home
# (one process can serve more than one LOCALM_HOME, each with its own salt).
_DIGEST_CACHE_MAX = 64
_digest_cache: "OrderedDict[str, str]" = OrderedDict()
_DIGEST_CACHE_LOCK = threading.Lock()

# Serialises MINTING an owner-key record, held across the whole
# read-derive-write. Distinct from _DIGEST_CACHE_LOCK, which only guards the
# dict. Taken BEFORE any sessions lock, never after. RLock: _hash_key holds it
# and then calls _owner_digest, which takes it again.
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
    """Drop the memoised derivations. Called whenever the owner key changes."""
    with _DIGEST_CACHE_LOCK:
        _digest_cache.clear()


def _load_owner_kdf() -> list:
    """The owner-key verifier records, or [] when there are none.

    A present-but-unreadable or corrupt file logs a warning and returns [], so a
    fresh record is minted. The file holds no plaintext and nothing AUTHENTICATES
    from it: the owner key is verified by plaintext ct_equal against auth.key."""
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
    # _atomic_write_private creates the temp file already restricted.
    _atomic_write_private(path, json.dumps({"v": 1, "records": records},
                                           indent=2))


def _owner_kdf_record_for(key: str, records: list) -> Optional[dict]:
    """The existing verifier record matching *key*, or None.

    Verifies with EACH record's OWN stored parameters, so a key derived under
    older cost parameters still matches. Only ever called for a key already known
    to be the owner key.

    NOT "once per process": that only holds on the verify() path, where _hash_key
    memoises the result. set_api_key calls this again on EVERY set, uncached, so
    a caller that sets the key repeatedly pays a full scrypt derivation per kept
    record, per call. _OWNER_KDF_KEEP bounds that cost."""
    for r in records:
        if r.get("alg") != _ALG_KDF:
            continue
        try:
            salt = bytes.fromhex(str(r.get("salt", "")))
            digest = _scrypt_derive(key, salt, int(r["n"]), int(r["r"]),
                                    int(r["p"]), int(r["dklen"]))
        except (ValueError, KeyError, TypeError) as e:
            logger.debug("skipping an unusable owner KDF record (%s)", e)
            continue
        if ct_equal(digest, str(r.get("digest", ""))):
            return r
    return None


def _owner_digest(key: str) -> str:
    """The KDF-derived principal identity for the OWNER key *key*.

    Reuses the persisted record for this key when there is one, so the value is
    stable forever; otherwise mints one (fresh salt, current cost parameters) and
    migrates any identity previously recorded under the legacy unsalted digest.

    Holds _OWNER_KDF_LOCK across the whole read-derive-write, so two callers
    cannot mint two different salts for the same key. Reentrant, so _hash_key may
    hold it already."""
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
        logger.warning("could not persist the owner key derivation record %s "
                       "(%s); sessions and jobs stamped in this process may not "
                       "be recognised after a restart", owner_kdf_file(), e)
    _migrate_legacy_owner_identity(_legacy_owner_identity(key), rec["digest"])
    return str(rec["digest"])


def _migrate_legacy_owner_identity(legacy: str, current: str) -> None:
    """Re-link anything recorded under the owner key's OLD unsalted digest.

    Runs once, when a key's KDF record is first minted. Best-effort: a failure
    here costs a re-login, never access, so it is logged rather than raised.

    jobs.json is deliberately NOT rewritten here; that side accepts the legacy
    digest and stamps the job as owner-owned the first time it matches (see
    jobs.runner)."""
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
    """True when *key* is the owner key currently in effect. An exact identity
    check against the live value, NOT a guess from the key's length or
    alphabet."""
    return ct_equal(key, get_api_key())


def _hash_key(key: str) -> str:
    """The stable identity digest for *key*.

    The owner key (user-choosable, possibly human-memorable) gets a salted scrypt
    derivation; a generated keystore token gets the cheap unsalted digest. The
    expensive path is memoised per process, so it costs one derivation per key
    rather than one per request."""
    ck = _memo_key(key) + "@" + _cache_scope()
    hit = _cache_get(ck)
    if hit is not None:
        return hit
    if not _is_owner_key(key):
        # A generated keystore token, or a token that matches nothing at all.
        # Cheap, and never cached.
        return _fast_digest(key)
    with _OWNER_KDF_LOCK:
        # Re-check under the lock: a concurrent request may have minted the
        # record and warmed the memo while we waited.
        hit = _cache_get(ck)
        if hit is not None:
            return hit
        derived = _owner_digest(key)
        _cache_put(ck, derived)
    return derived


def _record_digest_for(record: dict, key: str, fast: Callable[[], str]) -> Optional[str]:
    """The digest *key* would produce under *record*'s DECLARED construction, or
    None when the record cannot be evaluated.

    *fast* is a CALLABLE, not a string, so the cheap digest is computed only on
    the branch that actually returns one.

    A record with no "alg" is treated as a generated token on the cheap path. An
    UNKNOWN alg returns None (refuse to match) rather than falling back to the
    cheap digest."""
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
    """The keystore record *key* authenticates against, or None.

    The cheap digest is computed LAZILY, on first use, and only when a record
    actually declares the fast construction. Memoised, so a keystore holding many
    generated-token rows still hashes once per lookup rather than once per
    row."""
    _fast_memo: list = []

    def fast() -> str:
        if not _fast_memo:
            _fast_memo.append(_fast_digest(key))
        return _fast_memo[0]

    for r in records:
        cand = _record_digest_for(r, key, fast)
        if cand is None:
            continue
        # ct_equal, not compare_digest: a corrupted row holding a non-ASCII
        # "hash" must fail to match rather than raise.
        if ct_equal(str(r.get("hash", "")), cand):
            return r
    return None


def _mark_record_alg(key_id: Optional[str], alg: str) -> None:
    """Stamp the construction marker onto a legacy record, in place.

    Best-effort: a read-only store must not break authentication."""
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
# dial (host > shared > none), not a scope. "none" is no host filesystem at all;
# "host" is the whole server filesystem.
#
# "shared" is reserved scaffolding and is NOT enforced anywhere yet:
# require_fs_host() grants only "host", so a "shared" key currently reaches no
# more host filesystem than "none". It is kept out of the `localm key create
# --fs-access` choices until the confinement exists.
FS_ACCESS_LEVELS = ("none", "shared", "host")


def norm_fs_access(level: Optional[str]) -> str:
    """Coerce *level* to a known FS-access level; unknown/blank -> the safe
    default 'none' (least privilege)."""
    lv = (level or "none").strip().lower()
    return lv if lv in FS_ACCESS_LEVELS else "none"


def fs_access_for(token: str, default: str = "none") -> str:
    """The stored host-filesystem-access level for the key behind *token*, or
    *default* if the key is unknown or has no level recorded (a legacy key minted
    before this attribute existed defaults to the safe 'none')."""
    if not token or not token.strip():
        return default
    rec = _find_keystore_record(token.strip(), _load_keystore())
    if rec is None:
        return default
    return norm_fs_access(rec.get("fs_access", default))


def norm_rag_roots(roots) -> list:
    """Coerce *roots* to a clean, order-preserving, de-duplicated list of
    folder-path strings for a key's per-key RAG-indexing allowlist; anything
    that is not a non-empty string is dropped. None/blank -> [] (no per-key
    restriction - the key falls back to the global ``rag_allowed_roots`` policy,
    see rag/store.py's ``confine_index_path``).

    *roots* must be a list/tuple; a bare string is REJECTED (returns []) rather
    than being iterated character by character."""
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
    """The stored per-key RAG-indexing folder allowlist for the key behind
    *token*, or *default* (``[]`` if not given) if the key is unknown or has no
    list recorded. An empty result means NO per-key restriction: the key falls
    back to the global policy."""
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
    """Mint a named key with *scope_list*, persist its hash, and return a record
    INCLUDING the plaintext key once - the caller must surface it now, it cannot
    be recovered. Raises ValueError on an unknown scope.

    *fs_access* is the host-filesystem reach this key grants ("none" | "shared" |
    "host"); it defaults to "none". The owner/ADMIN key always resolves to "host"
    regardless of this field (see effective_fs_access).

    *rag_roots* is an optional per-key RAG-indexing folder allowlist. Empty/None
    (the default) applies no per-key restriction, so the key falls back to the
    global ``rag_allowed_roots`` policy (rag/store.py's ``confine_index_path``).
    A non-empty list instead CONFINES the key to exactly those folders: the home
    directory, the working directory and the global allowed-roots list are not
    implied on top of it. The owner/ADMIN key is never confined by this field
    regardless of what is stored (see effective_rag_roots).

    PRIVILEGED_SCOPES (admin / keys:admin / plugins:admin / config:write /
    coder:full) are refused with PermissionError unless *allow_privileged* is
    True. Callers must only set that for an owner/ADMIN principal.

    *expires* is an optional epoch-seconds deadline after which verify() rejects
    the key; None (default) never expires."""
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
        # generate_key(). The "alg" marker records that on the row.
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
    """Delete a named key by id. Returns True if it existed.

    Also drops any browser SESSIONS minted from that key, so revoking a key cuts
    off a paired device immediately instead of leaving its cookie session valid
    until expiry. The cookie auth path additionally re-validates a scoped
    session's key on every request via key_hash_live."""
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
                # An ADMIN-scoped session is exempt from the cookie path's
                # per-request key re-check, so a failed write here leaves its
                # cookie working. The key itself is revoked either way.
                logger.warning(
                    "key %s was revoked, but its browser sessions could not be "
                    "dropped; an admin-scoped session for it may still be "
                    "usable until the session store is writable again", key_id)
        except Exception as e:
            logger.debug("session cleanup after key revoke failed (non-fatal): %s", e)
    return True


def key_hash_live(key_hash: Optional[str]) -> bool:
    """True when a keystore key with this sha256 *key_hash* still exists AND has not
    expired. The cookie path checks this every request, so revoking or expiring the
    underlying key also invalidates the session. Owner/ADMIN sessions are NOT gated
    on this: the owner key is not in the keystore, and an owner session is decoupled
    from the key VALUE."""
    if not key_hash:
        return False
    now = time.time()
    for r in _load_keystore():
        if r.get("hash") == key_hash:
            exp = r.get("expires")
            return not (exp is not None and now > float(exp))
    return False


def scopes_for_key_hash(key_hash: Optional[str]) -> Optional[set]:
    """The scopes a LIVE keystore key with this sha256 *key_hash* grants, or None.

    The by-hash sibling of ``verify()``: same liveness rules (a missing or expired
    record grants nothing), but keyed on the stored digest rather than a presented
    secret.

    Returns None (not an empty set) when nothing matches, so "no such key" stays
    distinguishable from "a key that grants nothing". A caller making a privilege
    decision must treat None as DENY.

    Does NOT resolve the owner key: it is not a keystore entry, so a caller that
    cares about the owner must compare against ``get_api_key()`` itself. Nor is it
    a substitute for ``key_hash_live``; it adds the scope question on top."""
    if not key_hash:
        return None
    now = time.time()
    # _load_keystore() returns [] on OSError/ValueError, so an unreadable
    # auth.json makes this return None, which callers must read as DENY.
    for r in _load_keystore():
        if r.get("hash") == key_hash:
            exp = r.get("expires")
            if exp is not None and now > float(exp):
                return None
            return set(r.get("scopes", []))
    return None


def _keystore_configured() -> bool:
    """True when the scoped keystore should count as 'auth in effect'.

    A present-but-UNPARSEABLE or unreadable auth.json counts as configured, so a
    transient corruption fails CLOSED. A genuinely absent or empty (``[]``)
    keystore is NOT configured, so a fresh or cleared install runs open.
    """
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


# One-shot latch for the empty-auth.key notice (see _owner_key_present).
_empty_owner_key_warned = False


def _owner_key_present() -> bool:
    """True when an owner key is in effect: the LOCALM_API_KEY env var is set, OR
    the auth.key file exists AND is not readably empty.

    A present-but-UNREADABLE file counts as present so auth stays IN EFFECT (fail
    CLOSED). Only a genuinely ABSENT file, or one we can READ and see holds no
    key, is "no owner key" -> open by design.

    localm itself never writes an empty auth.key (set_api_key('') unlinks
    instead), so one is an anomaly and is warned about once per state change.

    Distinct from get_api_key(), which returns the key VALUE (or None when it
    cannot be read): when the file is present but unreadable this returns True
    (auth in effect) while verify() matches nothing, so every request is rejected
    (401 / locked) rather than served open."""
    global _empty_owner_key_warned
    env = os.environ.get(ENV_VAR)
    if env and env.strip():
        return True
    status, text, _err = _read_owner_key_file()
    if status == _KEY_ABSENT:
        return False                       # genuinely absent -> open by design
    if status == _KEY_UNREADABLE:
        # Present but unreadable or undecodable: fail closed.
        return True
    if _key_text_or_none(text) is not None:
        # Re-arm the notice, so a later KEYED -> OPEN drop warns again.
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
    """True when auth is in effect: an owner key (env, or an auth.key file holding
    one - see _owner_key_present) OR a configured scoped keystore. When this is
    False the server runs open (unless require_auth_enabled()). A corrupt/
    unreadable keystore, and a present-but-unreadable owner key, both count as
    configured so they fail CLOSED. A credential file we CAN read and which holds
    no key (an empty auth.key, an empty ``[]`` keystore) means no key."""
    return _owner_key_present() or _keystore_configured()


# Throttle for last-used stamping: a key's last_used is stamped at most once per
# this many seconds per process, tracked in memory.
_LAST_USED_THROTTLE_S = 300
_last_used_writes: dict = {}
_LAST_USED_LOCK = threading.Lock()


def _touch_last_used(key_hash: str) -> None:
    """Best-effort: stamp a just-verified key's last_used (throttled). Never raises -
    a usage timestamp is non-essential and must not break verification."""
    if not key_hash:
        return
    now = time.monotonic()
    with _LAST_USED_LOCK:
        # Dict absence, not a 0.0 sentinel, marks "never stamped this process":
        # time.monotonic()'s epoch is platform-defined.
        prev = _last_used_writes.get(key_hash)
        if prev is not None and now - prev < _LAST_USED_THROTTLE_S:
            return
        _last_used_writes[key_hash] = now
    try:
        with _KEYSTORE_LOCK:
            records = _load_keystore()
            for r in records:
                # Plain ==, not constant-time: the key is ALREADY verified; this
                # only locates its row to stamp.
                if r.get("hash") == key_hash:
                    r["last_used"] = time.time()
                    _save_keystore(records)
                    break
    except Exception as e:
        # Best-effort: a usage stamp must never break auth.
        logger.debug("last_used stamp failed (non-fatal): %s", e)


def verify(presented: Optional[str]) -> Optional[set]:
    """Resolve a presented bearer token to the set of scopes it grants, or None
    if it matches nothing. The owner key grants ADMIN (every scope)."""
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
        # Legacy row: it verified on the cheap path. Record that now.
        _mark_record_alg(rec.get("id"), _ALG_FAST)
    _touch_last_used(str(rec.get("hash", "")))
    return set(rec.get("scopes", []))
