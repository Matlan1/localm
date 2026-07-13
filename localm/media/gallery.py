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
import threading
from pathlib import Path
from typing import Optional

from fastapi import Request

_LOCK = threading.Lock()


def _index_path(media_kind: str) -> Path:
    from localm.config import home_dir
    return home_dir() / "gallery_index" / f"{media_kind}.json"


def _read_index(media_kind: str) -> dict:
    p = _index_path(media_kind)
    if not p.is_file():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_index(media_kind: str, data: dict) -> None:
    p = _index_path(media_kind)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data), encoding="utf-8")


def stamp_owner(media_kind: str, name: str, owner: Optional[str]) -> None:
    """Record *owner* (a `principal_id()`, or None in open mode) for a newly
    generated artifact *name*. A None owner writes no entry - `owner_of()`
    already defaults an untracked name to None, so open-mode installs never
    grow an index at all."""
    if owner is None:
        return
    with _LOCK:
        idx = _read_index(media_kind)
        idx[name] = owner
        _write_index(media_kind, idx)


def owner_of(media_kind: str, name: str) -> Optional[str]:
    """The recorded owner for *name*, or None when untracked (open mode, or a
    file that never went through `stamp_owner` - a legacy/manually-placed file).
    None is UNRESTRICTED, mirroring jobs' "no recorded owner" semantics."""
    return _read_index(media_kind).get(name)


def rename_owner(media_kind: str, old_name: str, new_name: str) -> None:
    """Carry an owner entry across a rename; a no-op when *old_name* was
    untracked (nothing to carry)."""
    with _LOCK:
        idx = _read_index(media_kind)
        owner = idx.pop(old_name, None)
        if owner is not None:
            idx[new_name] = owner
            _write_index(media_kind, idx)


def forget_owner(media_kind: str, name: str) -> None:
    """Drop the owner entry for *name* (delete, or move OUT of the gallery)."""
    with _LOCK:
        idx = _read_index(media_kind)
        if idx.pop(name, None) is not None:
            _write_index(media_kind, idx)


def require_owner(media_kind: str):
    """FastAPI dependency factory: gate a route on ownership of the gallery
    artifact named by its ``name`` path param. Depends()-injectable (use as
    ``dependencies=[Depends(gallery.require_owner("image"))]``), so a new
    per-owner media route cannot omit the check by construction (design-audit
    LM-DA-020). Raises the SAME 404 a missing artifact would (never 403), so a
    foreign key cannot even confirm another principal's media exists. Mirrors
    jobs' `owned_job` / http_server's `require_owner` factory pattern."""
    from localm.inference.http_server import require_owner as _require_owner

    def _resolve(name: str):
        return name, owner_of(media_kind, name), f"No such {media_kind}: {name}"
    return _require_owner(_resolve)


def owned_names(request: Request, media_kind: str, names: "list[str]") -> "list[str]":
    """Filter already-listed *names* down to the ones the caller may see - same
    rule as `require_owner`, applied to a whole history listing in one index read."""
    from localm.inference.http_server import job_owner_ok
    idx = _read_index(media_kind)
    return [n for n in names if job_owner_ok(request, idx.get(n))]
