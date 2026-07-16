# SPDX-License-Identifier: AGPL-3.0-or-later
"""Opaque server-side browser sessions for localm's GUI (decouple session <-> key).

The browser used to authenticate with a cookie whose VALUE WAS THE RAW API KEY, so
rolling the key invalidated every live session and the durable secret sat in cookie
jars for ~400 days. Instead, on login (or the loopback auto-seed / one-time launch
grant) the server mints a random opaque session id, records a HASH of it (never the
id) plus a snapshot of what that login may do, and hands the browser only the opaque
id as its HttpOnly cookie. Rolling the owner key no longer touches sessions, the key
never lives in a cookie, and each session can expire or be revoked on its own.

Mirrors auth.py's keystore discipline: a home-dir JSON file, atomic writes,
owner-only perms, sha256 of the secret stored and compared constant-time, and a lock
serialising the read-modify-write. A present-but-unreadable/corrupt store fails
CLOSED (verifies no session) rather than silently authorising nothing-or-everything.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import threading
import time
from pathlib import Path
from typing import Optional

from localm.auth import ct_equal
from localm.debuglog import logger

# Absolute lifetime: matches the browser cookie cap (~400 days) so a session that
# the browser still remembers is still honoured by the server (SEAMLESS: stay
# signed in across a browser/PWA/server restart). Idle lifetime: a session unused
# for this long is pruned, so an abandoned device does not stay valid forever. On
# loopback a fresh session is re-minted transparently (auto-seed / launch grant),
# so idle expiry is invisible there; it only asks a LAN/phone client to re-pair.
_ABS_TTL_S = 400 * 24 * 3600
_IDLE_TTL_S = 30 * 24 * 3600

# Serialises the read-modify-write of the store (create/revoke/last_used), exactly
# as auth._KEYSTORE_LOCK does for the keystore, so two concurrent writers cannot
# lose each other's change.
_LOCK = threading.Lock()

# mtime-keyed cache so the hot path (lookup runs on EVERY cookie-authed request)
# does not re-parse the file each time; invalidated automatically whenever the file
# is written (create/revoke/last_used bump its mtime).
_CACHE: dict = {"mtime": None, "records": None}

# last_used write throttle (per process): lookup runs constantly, so stamp a
# session's last_used at most once per this many seconds instead of rewriting the
# store on every request.
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
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(records, indent=2), encoding="utf-8")
    os.replace(tmp, path)          # atomic on Windows + POSIX (same dir)
    _restrict_perms(path)
    # Refresh the cache to the just-written content so the next lookup is warm and
    # never reads a torn view.
    try:
        st = path.stat()
        _CACHE["mtime"] = (st.st_mtime_ns, st.st_size)
        _CACHE["records"] = records
    except OSError:
        _CACHE["mtime"] = None
        _CACHE["records"] = None


def _restrict_perms(path: Path) -> None:
    """Owner-only perms where the OS supports it (POSIX chmod; best-effort). The
    data dir is already user-scoped, so this is defense-in-depth, never fatal."""
    try:
        if os.name == "posix":
            os.chmod(path, 0o600)
    except OSError:
        # A usage/permissions nicety on a filesystem without per-file perms must
        # never break session persistence; the home dir scoping is the real guard.
        pass


def _expired(rec: dict, now: float) -> bool:
    """True when *rec* is past its absolute deadline or its idle window."""
    exp = rec.get("expires")
    if exp is not None and now > float(exp):
        return True
    last = rec.get("last_used") or rec.get("issued") or 0
    idle = rec.get("idle_ttl", _IDLE_TTL_S)
    return now > float(last) + float(idle)


def create(*, scopes, key_hash: Optional[str], fs_access: str = "none",
           label: str = "", ttl: float = _ABS_TTL_S,
           idle_ttl: float = _IDLE_TTL_S) -> str:
    """Mint a new session, persist it, and return the opaque session id (the value
    the caller sets as the HttpOnly cookie). Stores only a HASH of the id.

    *scopes* is the scope set this login grants (snapshot from auth.verify), so the
    session stays valid across a later owner-key roll. *key_hash* is the sha256 of
    the key that minted it (so principal_id() over a cookie matches the same key
    presented as a bearer, for job ownership). *fs_access* is the host-filesystem
    reach for this session. Raises if the store cannot be persisted (a security
    step that fails must not look like success)."""
    now = time.time()
    sid = secrets.token_urlsafe(32)
    rec = {
        "id_hash": _hash(sid),
        "scopes": sorted(scopes) if scopes else [],
        "key_hash": key_hash,
        "fs_access": fs_access or "none",
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
        # Corrupt/unreadable store: refuse every session rather than silently
        # authorising or de-authorising. Surface it (rule 5); do not swallow.
        logger.warning("session store unreadable (%s); refusing cookie sessions "
                       "until it is repaired", e)
        return None
    for r in records:
        # ct_equal, not compare_digest: both sides are normally hexdigests, but a
        # hand-edited or corrupted store row could hold a non-ASCII "id_hash" and
        # must simply fail to match, not 500 every cookie lookup that reaches it.
        rh = r.get("id_hash", "")
        if ct_equal(rh, h):
            if _expired(r, now):
                return None
            _touch_last_used(h)
            return {"scopes": list(r.get("scopes", [])),
                    "key_hash": r.get("key_hash"),
                    "fs_access": r.get("fs_access", "none"),
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


def revoke(sid: Optional[str]) -> bool:
    """Delete the session behind *sid* (real, server-side logout). Returns True if
    it existed. Never raises on a missing/unreadable store (logout must not error)."""
    if not sid or not sid.strip():
        return False
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
        logger.warning("could not revoke session (%s); the store may be unreadable", e)
        return False


def revoke_by_key_hash(key_hash: Optional[str]) -> int:
    """Delete every session minted from the key with this hash, so revoking a scoped
    key also drops the sessions it authorized. Returns the count removed. Never
    raises on a missing/unreadable store."""
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
        return 0


def revoke_all() -> int:
    """Delete EVERY session (log out all devices). Returns the count removed. Used
    by the explicit 'clear owner key' / 'sign out everywhere' flows."""
    try:
        with _LOCK:
            records = _load()
            n = len(records)
            if n:
                _save([])
            return n
    except (OSError, ValueError) as e:
        logger.warning("could not clear sessions (%s)", e)
        return 0


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
