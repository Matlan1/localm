# SPDX-License-Identifier: AGPL-3.0-or-later
"""Opaque server-side browser sessions for localm's GUI (session decoupled from key).

On login (or the loopback auto-seed / one-time launch grant) the server mints a
random opaque session id, records a HASH of it (never the id) plus a snapshot of
what that login may do, and hands the browser only the opaque id as its HttpOnly
cookie. Rolling the owner key does not touch sessions, the key never lives in a
cookie, and each session can expire or be revoked on its own.

Mirrors auth.py's keystore discipline: a home-dir JSON file, atomic writes,
owner-only perms, sha256 of the secret stored and compared constant-time, and a lock
serialising the read-modify-write. A present-but-unreadable/corrupt store fails
CLOSED: it verifies no session.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import threading
import time
from pathlib import Path
from typing import Optional

from localm.auth import ct_equal
from localm.debuglog import logger

# Absolute lifetime matches the browser cookie cap (~400 days). Idle lifetime
# prunes a session unused for that long. On loopback a fresh session is re-minted
# transparently, so idle expiry only asks a LAN client to re-pair.
_ABS_TTL_S = 400 * 24 * 3600
_IDLE_TTL_S = 30 * 24 * 3600

# Serialises the read-modify-write of the store (create/revoke/last_used).
_LOCK = threading.Lock()

# mtime-keyed cache for the hot lookup path; invalidated whenever the file is
# written.
_CACHE: dict = {"mtime": None, "records": None}

# Per-process throttle: a session's last_used is stamped at most once per this
# many seconds.
_LAST_USED_THROTTLE_S = 300
_last_used_writes: dict = {}


def sessions_file() -> Path:
    """Path to the session store, inside the resolved localm data dir."""
    from localm.config import home_dir
    return home_dir() / "sessions.json"


def _hash(sid: str) -> str:
    # surrogatepass: see auth.ct_equal - a plain utf-8 encode raises on a lone
    # surrogate. Byte-identical for every sid that encodes at all.
    return hashlib.sha256(sid.encode("utf-8", "surrogatepass")).hexdigest()


def _load() -> list:
    """Load the store as a list of records, using the mtime cache. A missing store
    is an empty list; a present-but-corrupt/unreadable one raises so callers can
    fail CLOSED (lookup treats a load failure as 'no valid session')."""
    path = sessions_file()
    try:
        st = path.stat()
    except FileNotFoundError:
        return []
    except OSError:
        # Exists but unreadable: do NOT pretend it is empty (that would drop every
        # session silently). Surface it; lookup() fails closed on the exception.
        raise
    key = (st.st_mtime_ns, st.st_size)
    if _CACHE["mtime"] == key and _CACHE["records"] is not None:
        return _CACHE["records"]
    data = json.loads(path.read_text(encoding="utf-8"))  # ValueError on corrupt
    records = data if isinstance(data, list) else []
    _CACHE["mtime"] = key
    _CACHE["records"] = records
    return records


def _save(records: list) -> None:
    from localm.config import ensure_dirs
    ensure_dirs()
    path = sessions_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    # The temp file is restricted rather than the destination: it already holds
    # the full digest payload, and os.replace carries its ACL onto the
    # destination. atomic_write_private also creates the temp at 0600.
    from localm.config import atomic_write_private
    atomic_write_private(path, json.dumps(records, indent=2))
    # Refresh the cache to the just-written content so the next lookup is warm and
    # never reads a torn view.
    try:
        st = path.stat()
        _CACHE["mtime"] = (st.st_mtime_ns, st.st_size)
        _CACHE["records"] = records
    except OSError:
        _CACHE["mtime"] = None
        _CACHE["records"] = None


def _restrict_perms(path: Path) -> bool:
    """Owner-only perms where the OS supports it (best-effort, never fatal).
    Returns True when the tightening is believed to have happened.

    Delegates to ``config.restrict_file_perms``, which covers Windows as well
    as POSIX."""
    from localm.config import restrict_file_perms
    return restrict_file_perms(path)


def _expired(rec: dict, now: float) -> bool:
    """True when *rec* is past its absolute deadline or its idle window."""
    exp = rec.get("expires")
    if exp is not None and now > float(exp):
        return True
    last = rec.get("last_used") or rec.get("issued") or 0
    idle = rec.get("idle_ttl", _IDLE_TTL_S)
    return now > float(last) + float(idle)


def create(*, scopes, key_hash: Optional[str], fs_access: str = "none",
           rag_roots: Optional[list] = None,
           label: str = "", ttl: float = _ABS_TTL_S,
           idle_ttl: float = _IDLE_TTL_S,
           owner_key_minted: bool = False) -> str:
    """Mint a new session, persist it, and return the opaque session id (the value
    the caller sets as the HttpOnly cookie). Stores only a HASH of the id.

    *scopes* is the scope set this login grants (snapshot from auth.verify), so the
    session stays valid across a later owner-key roll. *key_hash* is the sha256 of
    the key that minted it (so principal_id() over a cookie matches the same key
    presented as a bearer, for job ownership). *fs_access* is the host-filesystem
    reach for this session. *rag_roots* is this session's per-key RAG-indexing
    folder allowlist snapshot, same shape and same reason as *fs_access* - taken
    now so it survives a later owner-key roll rather than being re-derived from a
    key hash that may no longer resolve. Raises if the store cannot be persisted
    (a security step that fails must not look like success).

    *owner_key_minted* records WHAT KIND of credential minted this session: True
    only when the presented credential WAS the owner key itself (auth._is_owner_key,
    a constant-time plaintext compare against auth.get_api_key()), as opposed to a
    minted, revocable keystore key. Every caller must pass a freshly computed
    POSITIVE proof; the False default is the fail-closed answer, so a mint site that
    forgets it, or a record persisted before this field existed, resolves to "not
    the owner".

    The flag is recorded at MINT time. *key_hash* is a snapshot of the key VALUE at
    login and an owner key ROLL leaves sessions alive (cli/keys.py), so after a roll
    the frozen hash matches neither the new owner key nor any keystore entry -
    exactly what a REVOKED scoped key looks like - and the two are indistinguishable
    from then on.

    It grants no authority on its own: it is an ATTRIBUTE OF THE SESSION, so it is
    exactly as revocable as the session carrying it (revoke / revoke_all / absolute
    and idle expiry all destroy it), and the sessions it can appear on already carry
    the owner's ADMIN scope snapshot."""
    now = time.time()
    sid = secrets.token_urlsafe(32)
    rec = {
        "id_hash": _hash(sid),
        "scopes": sorted(scopes) if scopes else [],
        "key_hash": key_hash,
        "fs_access": fs_access or "none",
        "rag_roots": list(rag_roots) if rag_roots else [],
        "owner_key_minted": bool(owner_key_minted),
        "label": (label or "")[:120],
        "issued": now,
        "last_used": now,
        "expires": now + float(ttl),
        "idle_ttl": float(idle_ttl),
    }
    with _LOCK:
        records = _load()
        records.append(rec)
        # Opportunistically drop expired rows so the store cannot grow without
        # bound as sessions come and go.
        records = [r for r in records if not _expired(r, now)] or [rec]
        _save(records)
    return sid


def lookup(sid: Optional[str]) -> Optional[dict]:
    """Resolve a presented session id to a PUBLIC copy of its record
    (scopes/key_hash/fs_access/...), or None if it matches nothing or has expired.
    Bumps last_used (throttled). Fails CLOSED: if the store cannot be read, no
    session verifies."""
    if not sid or not sid.strip():
        return None
    sid = sid.strip()
    h = _hash(sid)
    now = time.time()
    try:
        records = _load()
    except (OSError, ValueError) as e:
        # Corrupt/unreadable store: refuse every session, and surface the
        # failure rather than swallowing it.
        logger.warning("session store unreadable (%s); refusing cookie sessions "
                       "until it is repaired", e)
        return None
    for r in records:
        # ct_equal, not compare_digest: a corrupted row could hold a non-ASCII
        # id_hash, which must fail to match rather than raise.
        rh = r.get("id_hash", "")
        if ct_equal(rh, h):
            if _expired(r, now):
                return None
            _touch_last_used(h)
            return {"scopes": list(r.get("scopes", [])),
                    "key_hash": r.get("key_hash"),
                    "fs_access": r.get("fs_access", "none"),
                    "rag_roots": list(r.get("rag_roots", []) or []),
                    # `is True`, not bool(): create() writes a real bool, so any
                    # other value is a record predating this field or a corrupted
                    # store, and must not read as the owner stamp.
                    "owner_key_minted": r.get("owner_key_minted") is True,
                    "label": r.get("label", "")}
    return None


def _touch_last_used(id_hash: str) -> None:
    """Best-effort throttled last_used stamp so a session used within its idle
    window does not expire. Never raises: a usage stamp must not break auth."""
    if not id_hash:
        return
    now = time.monotonic()
    prev = _last_used_writes.get(id_hash)
    if prev is not None and now - prev < _LAST_USED_THROTTLE_S:
        return
    _last_used_writes[id_hash] = now
    try:
        with _LOCK:
            records = _load()
            changed = False
            for r in records:
                if r.get("id_hash") == id_hash:
                    r["last_used"] = time.time()
                    changed = True
                    break
            if changed:
                _save(records)
    except Exception as e:
        logger.debug("session last_used stamp failed (non-fatal): %s", e)


# Path-free label for a network surface when revocation could not complete. The
# store path carries the account name and must not appear in a response.
REVOKE_FAILURE_LABEL = "browser sessions (some devices may still be signed in)"


def revoke(sid: Optional[str]) -> Optional[bool]:
    """Delete the session behind *sid* (real, server-side logout).

    Returns True if it existed, False if it did not, and **None when the store
    could not be written**. A failed revocation is NOT the same as "there was
    nothing to revoke". None is falsy, so ``if revoke(...)`` keeps the ordinary
    meaning; only a caller that asks ``is None`` learns the difference.

    Still never raises: logout must not 500. It reports instead of throwing."""
    if not sid or not sid.strip():
        return False               # nothing presented: a real answer, not a failure
    h = _hash(sid.strip())
    try:
        with _LOCK:
            records = _load()
            remaining = [r for r in records if r.get("id_hash") != h]
            if len(remaining) == len(records):
                return False
            _save(remaining)
        return True
    except (OSError, ValueError) as e:
        logger.warning("could not revoke session (%s); the store may be unreadable; "
                       "the session id remains valid on the server", e)
        return None


def revoke_by_key_hash(key_hash: Optional[str]) -> Optional[int]:
    """Delete every session minted from the key with this hash, so revoking a scoped
    key also drops the sessions it authorized. Returns the count removed, or **None
    when the store could not be written** (see ``revoke`` for why that is a distinct
    answer rather than 0). Never raises on a missing/unreadable store.

    A failure here is not always contained by the per-request re-check: the cookie
    path re-validates a session's owning key against the live keystore on every
    request, which covers a scoped key, but an ADMIN-scoped session is exempt from
    that check so an owner-key roll cannot sign the owner out. If this cleanup fails
    for an ADMIN-scoped DEVICE key, its cookie keeps working."""
    if not key_hash:
        return 0
    try:
        with _LOCK:
            records = _load()
            keep = [r for r in records if r.get("key_hash") != key_hash]
            removed = len(records) - len(keep)
            if removed:
                _save(keep)
            return removed
    except (OSError, ValueError) as e:
        logger.warning("could not revoke sessions for a key (%s)", e)
        return None


def relink_key_hash(old_hash: Optional[str], new_hash: Optional[str]) -> int:
    """Repoint every session recorded under *old_hash* to *new_hash*. Returns the
    count changed. Never raises on a missing/unreadable store.

    Used once when the owner key's identity moves from the legacy unsalted digest
    to its salted KDF derivation. A session minted before that upgrade would
    otherwise keep reporting the OLD identity, so a job created from that cookie
    would stop being recognised as the same principal presenting the key as a
    bearer - the job-ownership parity ``create``'s docstring promises. The session
    stays valid and the user is never signed out.

    This changes an IDENTIFIER, never a credential: ``id_hash`` (what actually
    authenticates a cookie) is untouched, and an ADMIN session is exempt from
    ``key_hash_live`` anyway, so a session's validity does not depend on it."""
    if not old_hash or not new_hash or old_hash == new_hash:
        return 0
    try:
        with _LOCK:
            records = _load()
            changed = 0
            for r in records:
                if r.get("key_hash") == old_hash:
                    r["key_hash"] = new_hash
                    changed += 1
            if changed:
                _save(records)
            return changed
    except (OSError, ValueError) as e:
        # Surfaced by the caller, which explains the user-visible consequence.
        logger.warning("could not re-link sessions to a new key identity (%s)", e)
        return 0


def remember_owner_key_minted(sid: Optional[str]) -> bool:
    """Back-fill ``owner_key_minted`` on a session PROVEN to be the owner's.

    Best-effort, and never raises: the caller has already established the fact by
    comparing the recorded ``key_hash`` against the live owner key, so this run is
    correct either way and the next request re-proves it identically. Returns True
    when the record was actually updated.

    A session minted BEFORE the field existed carries no stamp, so it is
    recognised as the owner's only while the owner key still has the value it was
    minted with. After a roll the recorded hash matches neither the new owner key
    nor any keystore entry, so without this back-fill the session would be treated
    as a revocable keystore key's and signed out. Jobs' ``_remember_owner_key_job``
    does the same for a job row.

    Writes at most once per session: after this the stamp short-circuits the
    value comparison, so the auth path does not keep re-writing the store."""
    if not sid or not sid.strip():
        return False
    h = _hash(sid.strip())
    try:
        with _LOCK:
            records = _load()
            for r in records:
                if r.get("id_hash") == h:
                    if r.get("owner_key_minted") is True:
                        return False            # already recorded: nothing to do
                    r["owner_key_minted"] = True
                    _save(records)
                    return True
            return False
    except Exception as e:
        # Not silenced: if this keeps failing the session stays exposed to being
        # signed out on the owner's next key roll, and that has to be findable.
        logger.debug("could not record the owner-key stamp on a session (%s); "
                     "it will be re-derived on the next request", e)
        return False


def revoke_all() -> Optional[int]:
    """Delete EVERY session (log out all devices). Used by the explicit 'clear owner
    key' / 'sign out everywhere' flows: ``localm key clear``, ``localm key recover``,
    and ``POST /api/auth/key/clear``.

    Returns the count removed, or **None when the store could not be written**.
    Returning 0 for a FAILED write would be indistinguishable from 0 for "there
    were no sessions", and a caller would then report a completed sign-out while
    every session was still live.

    None is falsy, so ``if revoke_all():`` keeps the ordinary meaning (do not claim
    devices were signed out); a caller must ask ``is None`` to distinguish a failure
    from an empty store."""
    try:
        with _LOCK:
            records = _load()
            n = len(records)
            if n:
                _save([])
            return n
    except (OSError, ValueError) as e:
        logger.warning("could not clear sessions (%s); sessions that were live "
                       "REMAIN live and the sign-out did not happen", e)
        return None


def sweep() -> int:
    """Prune expired sessions (called on startup). Returns the count removed."""
    now = time.time()
    try:
        with _LOCK:
            records = _load()
            keep = [r for r in records if not _expired(r, now)]
            removed = len(records) - len(keep)
            if removed:
                _save(keep)
            return removed
    except (OSError, ValueError):
        # A corrupt store cannot be swept safely; leave it for lookup() to fail
        # closed on. Not fatal to startup.
        return 0
