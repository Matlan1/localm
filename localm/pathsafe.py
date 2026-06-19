# SPDX-License-Identifier: AGPL-3.0-or-later
"""Filesystem path-confinement helpers shared by the kernel GUI and plugins.

These guarantee a user-supplied name stays directly inside a known base
directory. They are backend-agnostic (no media/ComfyUI knowledge) - the kernel's
session-log browser and the media plugins (image/music/video) all use them, so
they live here rather than being duplicated per caller.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import HTTPException


def confined_name(base: Path, name: str) -> Path:
    """Resolve *name* and guarantee it stays directly inside *base*, without
    requiring it to exist (rename targets, new files).

    Blocklisting separators is not enough on Windows: a drive-relative name like
    ``C:evil`` joins to ``C:evil`` (outside *base*), and an absolute name
    replaces the join entirely. We verify the *resolved* path's parent is *base*
    and the basename is unchanged, which also rejects ``..``, nested subpaths,
    and ``con``/device names."""
    if name != Path(name).name or name in ("", ".", ".."):
        raise HTTPException(400, "Invalid file name")
    try:
        resolved = (base / name).resolve()
    except (OSError, ValueError):
        raise HTTPException(400, "Invalid file name")
    if resolved.parent != base.resolve() or resolved.name != name:
        raise HTTPException(400, "Invalid file name")
    return resolved


def confined_file(base: Path, name: str, kind: str) -> Path:
    """confined_name plus an existence check - for files being read."""
    resolved = confined_name(base, name)
    if not resolved.is_file():
        raise HTTPException(404, f"No such {kind}")
    return resolved
