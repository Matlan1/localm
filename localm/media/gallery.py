# SPDX-License-Identifier: AGPL-3.0-or-later
"""Per-key ownership for the generated-media galleries (image/music/video).

Each media plugin's file/delete/move/rename/history routes serve artifacts off a
flat directory on disk with no per-request auth check of their own. This module
gives them the same stamp-at-creation + check-at-access pattern jobs uses
(`localm.plugins.builtin.jobs.plug`'s `job_owner_ok` / `owned_job` in
`localm.inference.http_server`), applied to filesystem artifacts instead of
JobStore records.

The owner index lives OUTSIDE the served gallery directory (a sibling file under
the data dir), so it is never reachable through the gallery's own confined
file-serve route and never appears in a directory listing / history glob.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Optional

from fastapi import HTTPException, Request

from localm.storekit import atomic_write

_LOCK = threading.Lock()

_READ_ATTEMPTS = 5
_READ_RETRY_DELAY = 0.02

# Ownership changes not yet written to the on-disk index, keyed by the index
# file's path, then by artifact name. Each value is an owner, a _RenamedFrom, or
# _FORGOTTEN. Guarded by _LOCK. Applied on top of every read and written out by
# the next successful index write.
_PENDING: dict[str, dict[str, object]] = {}


class _RenamedFrom:
    """A held rename: the artifact takes the on-disk owner of *old*."""

    def __init__(self, old: str) -> None:
        self.old = old


_FORGOTTEN = object()


class GalleryIndexUnreadable(Exception):
    """The on-disk owner index EXISTS but could not be read or parsed (corrupt,
    truncated, locked, or a permission error), so ownership is INDETERMINATE.

    Callers must FAIL CLOSED - deny a non-owner - rather than treat every
    artifact as unowned/open. Distinct from a genuinely ABSENT index (the benign
    open/untracked case): collapsing "corrupt" into "empty dict" would be a
    fail-OPEN of an authorization gate."""


def _index_path(media_kind: str) -> Path:
    from localm.config import home_dir
    return home_dir() / "gallery_index" / f"{media_kind}.json"


def _read_index(media_kind: str) -> dict:
    """The owner map for *media_kind*. Returns {} ONLY when the index is
    genuinely absent (the benign open/untracked case). When the file EXISTS but
    cannot be read or parsed, raises GalleryIndexUnreadable instead of returning
    {}. The callers on the auth path translate the exception into a deny."""
    p = _index_path(media_kind)
    if not p.is_file():
        return {}
    try:
        data = json.loads(_read_text_retrying(p))
    except (OSError, ValueError, RecursionError) as e:
        from localm.debuglog import logger
        logger.warning(
            "gallery ownership index %s exists but is unreadable (%s); access "
            "checks deny non-owner keys %s media until it is repaired", p, e,
            media_kind)
        raise GalleryIndexUnreadable(f"{p}: {e}") from e
    if not isinstance(data, dict):
        # A file that parses to a list/null/scalar is still "exists but is not
        # a usable owner map", so it fails closed exactly like a truncated one.
        from localm.debuglog import logger
        logger.warning(
            "gallery ownership index %s parsed as %s, not an object; access "
            "checks deny non-owner keys %s media until it is repaired",
            p, type(data).__name__, media_kind)
        raise GalleryIndexUnreadable(f"{p}: not a JSON object")
    return data


def _read_text_retrying(p: Path) -> str:
    """``p.read_text()``, retried on ``PermissionError`` up to _READ_ATTEMPTS
    times, the same bound ``storekit.atomic_write`` gives the write side.
    Raises the last error when every attempt fails."""
    for _attempt in range(_READ_ATTEMPTS - 1):
        try:
            return p.read_text(encoding="utf-8")
        except PermissionError:
            time.sleep(_READ_RETRY_DELAY)
    return p.read_text(encoding="utf-8")


def _pending(media_kind: str) -> dict:
    """The held ownership changes for *media_kind* (the live dict). Caller must
    hold _LOCK."""
    return _PENDING.setdefault(str(_index_path(media_kind)), {})


def _apply_pending(idx: dict, pending: dict) -> dict:
    """A copy of the on-disk map *idx* with the held changes in *pending*
    applied: an owner sets the entry, a _RenamedFrom moves the old name's
    on-disk owner to the new name, _FORGOTTEN removes the entry."""
    out = dict(idx)
    for name, change in pending.items():
        if change is _FORGOTTEN:
            out.pop(name, None)
        elif isinstance(change, _RenamedFrom):
            owner = idx.get(change.old)
            out.pop(change.old, None)
            if owner is not None:
                out[name] = owner
        else:
            out[name] = change
    return out


def _read_effective(media_kind: str) -> dict:
    """The on-disk owner map with the held changes applied. Caller must hold
    _LOCK. Raises GalleryIndexUnreadable like _read_index."""
    return _apply_pending(_read_index(media_kind), _pending(media_kind))


def _write_with_pending(media_kind: str, idx: dict) -> None:
    """Write *idx* with the held changes applied, then clear them. Caller must
    hold _LOCK. Raises what _write_index raises, keeping the held changes."""
    pending = _pending(media_kind)
    _write_index(media_kind, _apply_pending(idx, pending))
    pending.clear()


def _unwritten_warning(name: str) -> str:
    """The user-facing warning for an artifact whose owner is held in memory
    only."""
    return (f"WARNING: the owner of {name} could not be recorded on disk. It stays "
            "private to your key while localm runs and is recorded with the next "
            "gallery change; if localm restarts first, every key can see it.")


def _log_held(media_kind: str, what: str, name: str, err: Exception) -> None:
    from localm.debuglog import logger
    logger.warning(
        "gallery: the %s of %s %s could not be written to the ownership index "
        "(%s); it is applied in memory and written with the next gallery change, "
        "and a restart before then loses it", what, media_kind, name, err)


def _write_index(media_kind: str, data: dict) -> None:
    """Persist the owner map ATOMICALLY (unique temp file + replace), so no
    half-written index the next read would have to reject is ever left behind.

    Delegates to ``storekit.atomic_write``, the shared kernel helper rag's and
    memory's stores also use. It rides out a transient external handle on the
    index at the swap instant (an AV mid-scan, the Search Indexer, a backup
    agent, the user's file browser), which a bare os.replace answers with
    PermissionError, and it cleans up its own temp file.
    """
    p = _index_path(media_kind)
    p.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(p, json.dumps(data))


def stamp_owner(media_kind: str, name: str, owner: Optional[str]) -> Optional[str]:
    """Record *owner* (a `principal_id()`, or None in open mode) for a newly
    generated artifact *name*. A None owner writes no entry - `owner_of()`
    already defaults an untracked name to None, so open-mode installs never
    grow an index at all.

    Returns None when the stamp reached the on-disk index. When the index
    cannot be read (after the bounded retry) or written, the stamp is held in
    memory instead: every ownership check in this process applies it, and the
    next successful index write persists it. A corrupt index is never
    overwritten. The return value is then a user-facing warning that the
    caller must show with the generation result, because a restart before the
    stamp is written leaves the artifact with no recorded owner. Never raises
    for an index read or write failure."""
    if owner is None:
        return None
    with _LOCK:
        pending = _pending(media_kind)
        pending.pop(name, None)
        try:
            idx = _read_index(media_kind)
            idx[name] = owner
            _write_with_pending(media_kind, idx)
        except (GalleryIndexUnreadable, OSError) as e:
            pending[name] = owner
            _log_held(media_kind, "owner", name, e)
            return _unwritten_warning(name)
        return None


def held_stamp_warning(media_kind: str, name: str) -> Optional[str]:
    """The `stamp_owner` warning for *name* while its owner is held in memory
    only, else None."""
    with _LOCK:
        change = _pending(media_kind).get(name)
    if change is None or change is _FORGOTTEN:
        return None
    return _unwritten_warning(name)


def push_stamp_warning(job, warning: Optional[str]) -> None:
    """Show a `stamp_owner` warning as a line in *job*'s output; a no-op for
    None."""
    if warning:
        job.push({"type": "line", "text": warning})


def owner_of(media_kind: str, name: str) -> Optional[str]:
    """The recorded owner for *name*, or None when untracked (open mode, or a
    file that never went through `stamp_owner` - a legacy/manually-placed file).
    None is UNRESTRICTED, mirroring jobs' "no recorded owner" semantics. Reads
    under `_LOCK` so it never observes a concurrent writer's half-updated map.
    Includes changes held in memory after a failed index write (see
    `stamp_owner`); never writes. Raises GalleryIndexUnreadable when the index
    exists but cannot be read - `require_owner` turns that into a deny (fail
    closed)."""
    with _LOCK:
        return _read_effective(media_kind).get(name)


def rename_owner(media_kind: str, old_name: str, new_name: str) -> Optional[str]:
    """Carry an owner entry across a rename; a no-op when *old_name* was
    untracked (nothing to carry). Writes every held change out with it.

    Returns None when the index reflects the rename. When the index cannot be
    read or written, the rename is held in memory (applied by every ownership
    check, written with the next successful index write; a corrupt index is
    never overwritten) and the return value is the user-facing warning to show
    with the result. Never raises for an index read or write failure."""
    with _LOCK:
        pending = _pending(media_kind)
        if old_name in pending:
            pending[new_name] = pending.pop(old_name)
        else:
            pending.pop(new_name, None)
        try:
            idx = _read_index(media_kind)
        except GalleryIndexUnreadable as e:
            if new_name not in pending:
                pending[new_name] = _RenamedFrom(old_name)
            _log_held(media_kind, "rename to", new_name, e)
            return _unwritten_warning(new_name)
        owner = idx.pop(old_name, None)
        if owner is not None:
            idx[new_name] = owner
        if owner is None and not pending:
            return None
        try:
            _write_with_pending(media_kind, idx)
        except OSError as e:
            if owner is not None:
                pending[new_name] = owner
                pending[old_name] = _FORGOTTEN
            _log_held(media_kind, "rename to", new_name, e)
            return _unwritten_warning(new_name) if new_name in pending else None
        return None


def forget_owner(media_kind: str, name: str) -> None:
    """Drop the owner entry for *name* (delete, or move OUT of the gallery),
    including a change held in memory, and write every other held change out
    with it. When the index cannot be read or written, the removal is held in
    memory instead (a corrupt index is never overwritten). Never raises for an
    index read or write failure."""
    with _LOCK:
        pending = _pending(media_kind)
        pending.pop(name, None)
        try:
            idx = _read_index(media_kind)
            if idx.pop(name, None) is None and not pending:
                return
            _write_with_pending(media_kind, idx)
        except (GalleryIndexUnreadable, OSError) as e:
            pending[name] = _FORGOTTEN
            _log_held(media_kind, "removal", name, e)


def _privileged(request: Request) -> bool:
    """True for a caller that may still reach media when ownership CANNOT be
    determined: the open-mode loopback owner (no key configured anywhere) or an
    ADMIN/owner key. A scoped non-owner key is NOT privileged and is denied -
    fail closed."""
    from localm.inference.http_server import caller_is_operator
    return caller_is_operator(request)


def require_owner(media_kind: str):
    """FastAPI dependency factory: gate a route on ownership of the gallery
    artifact named by its ``name`` path param. Depends()-injectable (use as
    ``dependencies=[Depends(gallery.require_owner("image"))]``). Raises the SAME
    404 a missing artifact would (never 403), so a foreign key cannot even
    confirm another principal's media exists. Mirrors jobs' `owned_job` /
    http_server's `require_owner` factory pattern.

    If the owner index is unreadable (GalleryIndexUnreadable), ownership cannot
    be proven, so a non-privileged caller gets that same 404 (fail closed); a
    privileged caller (open-mode owner / ADMIN) still passes."""
    from localm.inference.http_server import require_owner as _require_owner

    def _resolve(request: Request, name: str):
        try:
            owner = owner_of(media_kind, name)
        except GalleryIndexUnreadable:
            if not _privileged(request):
                # Same 404 as a missing/foreign artifact; the real cause was
                # already logged by _read_index, so suppress the chain here.
                raise HTTPException(404, f"No such {media_kind}: {name}") from None
            owner = None                     # privileged: treat as unowned/allowed
        return name, owner, f"No such {media_kind}: {name}"
    return _require_owner(_resolve)


def owned_names(request: Request, media_kind: str, names: list[str]) -> list[str]:
    """Filter already-listed *names* down to the ones the caller may see - same
    rule as `require_owner`, applied to a whole history listing in one index read.

    An unreadable index fails closed: a scoped key sees NOTHING (not everything)
    until it is repaired, while a privileged caller still sees the full list."""
    from localm.inference.http_server import job_owner_ok
    try:
        with _LOCK:
            idx = _read_effective(media_kind)
    except GalleryIndexUnreadable:
        return list(names) if _privileged(request) else []
    return [n for n in names if job_owner_ok(request, idx.get(n))]
