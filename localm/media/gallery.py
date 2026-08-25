# SPDX-License-Identifier: AGPL-3.0-or-later
"""Per-key ownership for the generated-media galleries (image/music/video)."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Optional

from fastapi import HTTPException, Request

from localm.storekit import atomic_write

_LOCK = threading.Lock()


class GalleryIndexUnreadable(Exception):
    """The on-disk owner index EXISTS but could not be read or parsed (corrupt, truncated, locked, or a permission error), so ownership is INDETERMINATE."""


def _index_path(media_kind: str) -> Path:
    from localm.config import home_dir
    return home_dir() / "gallery_index" / f"{media_kind}.json"


def _read_index(media_kind: str) -> dict:
    """The owner map for *media_kind*."""
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
    if not isinstance(data, dict):
        # Whether json.loads() SUCCEEDED is an implementation detail, not the
        # security property. A file that parses to a list/null/scalar is still
        # "exists but is not a usable owner map", so it fails closed exactly
        # like a truncated one: returning {} here would make owner_of() report
        # every artifact unowned -> job_owner_ok() -> unrestricted, which is
        # the very fail-open this function exists to prevent.
        from localm.debuglog import logger
        logger.warning(
            "gallery ownership index %s parsed as %s, not an object; failing "
            "closed - denying non-owner access to %s media until it is repaired",
            p, type(data).__name__, media_kind)
        raise GalleryIndexUnreadable(str(p))
    return data


def _write_index(media_kind: str, data: dict) -> None:
    """Persist the owner map ATOMICALLY (unique temp file + replace, the same crash- and concurrency-safe pattern jobs' store and comfy_patches already use)."""
    p = _index_path(media_kind)
    p.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(p, json.dumps(data))


def stamp_owner(media_kind: str, name: str, owner: Optional[str]) -> None:
    """Record *owner* (a `principal_id()`, or None in open mode) for a newly generated artifact *name*."""
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
    """The recorded owner for *name*, or None when untracked (open mode, or a file that never went through `stamp_owner` - a legacy/manually-placed file)."""
    with _LOCK:
        return _read_index(media_kind).get(name)


def rename_owner(media_kind: str, old_name: str, new_name: str) -> None:
    """Carry an owner entry across a rename; a no-op when *old_name* was untracked (nothing to carry)."""
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
    """True for a caller that may still reach media when ownership CANNOT be determined: the open-mode loopback owner (no key configured anywhere) or an ADMIN/owner key."""
    from localm.auth import any_key_configured
    if not any_key_configured():
        return True                          # open/dev mode = loopback owner
    from localm import scopes
    from localm.inference.http_server import caller_scopes
    held = caller_scopes(request)
    return held is not None and scopes.ADMIN in held


def require_owner(media_kind: str):
    """FastAPI dependency factory: gate a route on ownership of the gallery artifact named by its ``name`` path param."""
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
    """Filter already-listed *names* down to the ones the caller may see - same rule as `require_owner`, applied to a whole history listing in one index read."""
    from localm.inference.http_server import job_owner_ok
    try:
        with _LOCK:
            idx = _read_index(media_kind)
    except GalleryIndexUnreadable:
        return list(names) if _privileged(request) else []
    return [n for n in names if job_owner_ok(request, idx.get(n))]
