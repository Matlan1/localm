# SPDX-License-Identifier: AGPL-3.0-or-later
"""Opaque server-side browser sessions for localm's GUI (decouple session <-> key)."""

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
    """Load the store as a list of records, using the mtime cache."""
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
    # Restrict the TEMP file, not the destination: it already holds the full
    # digest payload, so a crash between the write and the rename would
    # otherwise leave an unrestricted copy behind. Doing it there also means ONE
    # icacls spawn instead of two, which matters because this runs on the
    # asyncio event loop (session_login, create_key_ep, _gui_index). MEASURED:
    # os.replace carries the source's ACL onto the destination, so the
    # restriction survives the rename; atomic_write_private retries the
    # destination for the case where it did not happen at all.
    #
    # The shared writer also CREATES the temp at 0600 via os.open rather than
    # write_text, which closes the residual POSIX window where the digest
    # payload existed at the umask default between the create and the chmod.
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
    """Owner-only perms where the OS supports it (best-effort, never fatal)."""
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
    """Mint a new session, persist it, and return the opaque session id (the value the caller sets as the HttpOnly cookie)."""
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
    """Resolve a presented session id to a PUBLIC copy of its record (scopes/key_hash/fs_access/...), or None if it matches nothing or has expired."""
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
                    "rag_roots": list(r.get("rag_roots", []) or []),
                    # `is True`, not bool(): create() only ever writes a real
                    # bool, so anything else in the file is a record written
                    # before this field existed, or a hand-edited/corrupted store.
                    # A truthy string must NOT read as the owner stamp, and an
                    # absent field must read as False - a session cannot acquire
                    # this retroactively, only by proving it at mint time.
                    "owner_key_minted": r.get("owner_key_minted") is True,
                    "label": r.get("label", "")}
    return None


def _touch_last_used(id_hash: str) -> None:
    """Best-effort throttled last_used stamp so a session used within its idle window does not expire."""
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


# The path-free label a caller may put on a NETWORK surface when revocation could
# not complete. Mirrors auth.clear_api_key's ``what`` vocabulary, and lives here so
# both the HTTP route and the CLI name the same thing: the store PATH carries the
# account name (rule 2) and must never ride out on a response.
REVOKE_FAILURE_LABEL = "browser sessions (some devices may still be signed in)"


def revoke(sid: Optional[str]) -> Optional[bool]:
    """Delete the session behind *sid* (real, server-side logout)."""
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
    """Delete every session minted from the key with this hash, so revoking a scoped key also drops the sessions it authorized."""
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
    """Repoint every session recorded under *old_hash* to *new_hash*."""
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
    """Back-fill ``owner_key_minted`` on a session PROVEN to be the owner's."""
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
    """Delete EVERY session (log out all devices)."""
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
    """Prune expired sessions (called on startup)."""
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
