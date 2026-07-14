# SPDX-License-Identifier: AGPL-3.0-or-later
"""Per-key ownership for the generated-media galleries (image/music/video).

Each media plugin's file/delete/move/rename/history routes serve artifacts off a
flat directory on disk with no per-request auth check of their own - the
generation JOB is owner-stamped (`localm.plugins.builtin.jobs.plug`'s pattern:
`job_owner_ok` / `owned_job` in `localm.inference.http_server`), but the
resulting FILE never was. This module gives the media plugins the same
stamp-at-creation + check-at-access pattern jobs already has, applied to
filesystem artifacts instead of JobStore records.

The owner index lives OUTSIDE the served gallery directory (a sibling file under
the data dir), so it is never reachable through the gallery's own confined
file-serve route and never appears in a directory listing / history glob.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from pathlib import Path
from typing import Optional

from fastapi import HTTPException, Request

_LOCK = threading.Lock()


class GalleryIndexUnreadable(Exception):
    """The on-disk owner index EXISTS but could not be read or parsed (corrupt,
    truncated, locked, or a permission error), so ownership is INDETERMINATE.

    Callers must FAIL CLOSED - deny a non-owner - rather than treat every
    artifact as unowned/open. This is deliberately distinct from a genuinely
    ABSENT index (the benign open/untracked case): collapsing "corrupt" into
    "empty dict" is a fail-OPEN of an authorization gate, exactly the
    missing-vs-corrupt conflation AGENTS.md rule 5 forbids."""


def _index_path(media_kind: str) -> Path:
    from localm.config import home_dir
    return home_dir() / "gallery_index" / f"{media_kind}.json"


def _read_index(media_kind: str) -> dict:
    """The owner map for *media_kind*. Returns {} ONLY when the index is
    genuinely absent (the benign open/untracked case). When the file EXISTS but
    cannot be read or parsed, raise GalleryIndexUnreadable instead of returning
    {}: an empty map makes owner_of() report EVERY artifact as unowned, which
    job_owner_ok() treats as unrestricted - a silent fail-open of the ownership
    gate. The callers on the auth path translate the exception into a deny."""
    p = _index_path(media_kind)
    if not p.is_file():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        from localm.debuglog import logger
        logger.warning(
            "gallery ownership index %s exists but is unreadable (%s); failing "
            "closed - denying non-owner access to %s media until it is repaired, "
            "rather than treating every artifact as unowned", p, e, media_kind)
        raise GalleryIndexUnreadable(str(p)) from e
    return data if isinstance(data, dict) else {}


def _write_index(media_kind: str, data: dict) -> None:
    """Persist the owner map ATOMICALLY (unique temp file + os.replace, the same
    crash- and concurrency-safe pattern jobs' store and comfy_patches already
    use). A plain in-place write could leave a half-written index that the next
    read would have to reject; the atomic swap removes that routine corruption
    trigger so the fail-closed read path stays a rare last resort."""
    p = _index_path(media_kind)
    p.parent.mkdir(parents=True, exist_ok=True)
    # Unique temp name (pid + thread + uuid) so concurrent writers never share a
    # temp path; os.replace is atomic, so the last writer wins cleanly.
    tmp = p.with_name(
        f"{p.name}.tmp.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}")
    tmp.write_text(json.dumps(data), encoding="utf-8")
    os.replace(tmp, p)


def stamp_owner(media_kind: str, name: str, owner: Optional[str]) -> None:
    """Record *owner* (a `principal_id()`, or None in open mode) for a newly
    generated artifact *name*. A None owner writes no entry - `owner_of()`
    already defaults an untracked name to None, so open-mode installs never
    grow an index at all."""
    if owner is None:
        return
    with _LOCK:
        try:
            idx = _read_index(media_kind)
        except GalleryIndexUnreadable:
            # Never clobber a corrupt index with a fresh one-entry map - that
            # would silently drop every prior owner record. Skip the stamp
            # (already warned in _read_index): while the index stays unreadable
            # the read path denies non-owners anyway, so nothing is exposed then.
            # Known residual: if the index is later REPAIRED, this one file has no
            # recorded owner and so reverts to the open/untracked default (the
            # same rule legacy/hand-placed files follow) - a strictly smaller
            # exposure than the pre-fix bug that exposed EVERY file, and
            # unavoidable here since the file is already on disk (failing the
            # generation would not un-write it). Atomic _write_index keeps an
            # unreadable index rare to begin with.
            return
        idx[name] = owner
        _write_index(media_kind, idx)


def owner_of(media_kind: str, name: str) -> Optional[str]:
    """The recorded owner for *name*, or None when untracked (open mode, or a
    file that never went through `stamp_owner` - a legacy/manually-placed file).
    None is UNRESTRICTED, mirroring jobs' "no recorded owner" semantics. Reads
    under `_LOCK` so it never observes a concurrent writer's half-updated map.
    Raises GalleryIndexUnreadable when the index exists but cannot be read -
    `require_owner` turns that into a deny (fail closed)."""
    with _LOCK:
        return _read_index(media_kind).get(name)


def rename_owner(media_kind: str, old_name: str, new_name: str) -> None:
    """Carry an owner entry across a rename; a no-op when *old_name* was
    untracked (nothing to carry)."""
    with _LOCK:
        try:
            idx = _read_index(media_kind)
        except GalleryIndexUnreadable:
            return          # do not clobber a corrupt index (warned in _read_index)
        owner = idx.pop(old_name, None)
        if owner is not None:
            idx[new_name] = owner
            _write_index(media_kind, idx)


def forget_owner(media_kind: str, name: str) -> None:
    """Drop the owner entry for *name* (delete, or move OUT of the gallery)."""
    with _LOCK:
        try:
            idx = _read_index(media_kind)
        except GalleryIndexUnreadable:
            return          # do not clobber a corrupt index (warned in _read_index)
        if idx.pop(name, None) is not None:
            _write_index(media_kind, idx)


def _privileged(request: Request) -> bool:
    """True for a caller that may still reach media when ownership CANNOT be
    determined: the open-mode loopback owner (no key configured anywhere) or an
    ADMIN/owner key. For them there is no cross-owner boundary to protect, so an
    unreadable index must not lock them out of their own gallery (they are the
    ones who repair it). A scoped non-owner key is NOT privileged and is denied -
    fail closed. Mirrors job_owner_ok's admin bypass, plus the open mode where
    caller_scopes() is None yet the loopback owner is fully trusted."""
    from localm.auth import any_key_configured
    if not any_key_configured():
        return True                          # open/dev mode = loopback owner
    from localm import scopes
    from localm.inference.http_server import caller_scopes
    held = caller_scopes(request)
    return held is not None and scopes.ADMIN in held


def require_owner(media_kind: str):
    """FastAPI dependency factory: gate a route on ownership of the gallery
    artifact named by its ``name`` path param. Depends()-injectable (use as
    ``dependencies=[Depends(gallery.require_owner("image"))]``), so a new
    per-owner media route cannot omit the check by construction (design-audit
    LM-DA-020). Raises the SAME 404 a missing artifact would (never 403), so a
    foreign key cannot even confirm another principal's media exists. Mirrors
    jobs' `owned_job` / http_server's `require_owner` factory pattern.

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


def owned_names(request: Request, media_kind: str, names: "list[str]") -> "list[str]":
    """Filter already-listed *names* down to the ones the caller may see - same
    rule as `require_owner`, applied to a whole history listing in one index read.

    An unreadable index fails closed: a scoped key sees NOTHING (not everything)
    until it is repaired, while a privileged caller still sees the full list."""
    from localm.inference.http_server import job_owner_ok
    try:
        with _LOCK:
            idx = _read_index(media_kind)
    except GalleryIndexUnreadable:
        return list(names) if _privileged(request) else []
    return [n for n in names if job_owner_ok(request, idx.get(n))]
