# SPDX-License-Identifier: AGPL-3.0-or-later
"""Filesystem path-confinement helpers shared by the kernel GUI and plugins."""

from __future__ import annotations

import ntpath
import os
from pathlib import Path

from fastapi import HTTPException

# Characters Windows refuses in a filename, plus the C0 control range.
# '/' and '\' are excluded: the confinement functions below handle separators
# themselves, per platform.
WINDOWS_RESERVED_NAME_CHARS = frozenset('<>:"|?*') | frozenset(chr(c) for c in range(32))


def confined_name(base: Path, name: str) -> Path:
    """Resolve *name* and guarantee it stays directly inside *base*, without requiring it to exist (rename targets, new files)."""
    if name != Path(name).name or name in ("", ".", ".."):
        raise HTTPException(400, "Invalid file name")
    if set(name) & WINDOWS_RESERVED_NAME_CHARS:
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


def confined_under(base: Path, relpath: str) -> Path:
    """Join *relpath* under *base* and guarantee the result stays strictly inside it, raising ``ValueError`` when it would not."""
    norm = (relpath or "").strip().replace("\\", "/")
    if not norm:
        raise ValueError("empty path")
    if norm.startswith("/"):
        raise ValueError(f"absolute path not allowed: {relpath!r}")
    if len(norm) >= 2 and norm[1] == ":":
        raise ValueError(f"drive-qualified path not allowed: {relpath!r}")
    parts = [p for p in norm.split("/") if p not in ("", ".")]
    if not parts:
        raise ValueError(f"path resolves to no name: {relpath!r}")
    if ".." in parts:
        raise ValueError(f"traversal component not allowed: {relpath!r}")
    # Reject a drive-qualified component at any position, not just the first.
    for p in parts:
        if len(p) >= 2 and p[1] == ":":
            raise ValueError(f"drive-qualified path component not allowed: {relpath!r}")
    # Reject a reserved character in any component.
    for p in parts:
        if set(p) & WINDOWS_RESERVED_NAME_CHARS:
            raise ValueError(f"reserved character in path component not allowed: {relpath!r}")
    try:
        resolved = base.joinpath(*parts).resolve()
        base_resolved = base.resolve()
    except (OSError, ValueError) as e:
        raise ValueError(f"path could not be resolved: {relpath!r} ({e})")
    # Strictly below base: base itself is not a valid target.
    if base_resolved not in resolved.parents:
        raise ValueError(f"path escapes {base_resolved}: {relpath!r}")
    # Reject a component whose resolved name differs from the requested one,
    # at every nesting level.
    node = resolved
    for part in reversed(parts):
        if node.name != part:
            raise ValueError(
                f"path component resolved to a different name than requested, "
                f"possibly a short-name alias: {relpath!r}")
        node = node.parent
    return resolved


def confined_absolute_or_under(base: Path, raw: str) -> Path:
    """Resolve *raw* and guarantee it stays inside *base*, raising ``ValueError`` when it would not."""
    s = (raw or "").strip()
    if not s or s in (".", ".."):
        raise ValueError(f"invalid path: {raw!r}")
    if is_unc_or_device_path(s):
        raise ValueError(f"UNC and device paths are not allowed: {raw!r}")
    p = Path(s)
    parts = p.parts[1:] if p.is_absolute() else p.parts
    if not parts:
        raise ValueError(f"path resolves to no name: {raw!r}")
    for part in parts:
        if set(part) & WINDOWS_RESERVED_NAME_CHARS:
            raise ValueError(f"reserved character in path component not allowed: {raw!r}")
    joined = p if p.is_absolute() else base / p
    try:
        resolved = joined.resolve()
        base_resolved = base.resolve()
    except (OSError, ValueError) as e:
        raise ValueError(f"path could not be resolved: {raw!r} ({e})")
    if resolved == base_resolved or base_resolved not in resolved.parents:
        raise ValueError(f"path escapes {base_resolved}: {raw!r}")
    depth = len(resolved.relative_to(base_resolved).parts)
    node = resolved
    for part in reversed(parts[-depth:]):
        if node.name != part:
            raise ValueError(
                f"path component resolved to a different name than requested, "
                f"possibly a short-name alias: {raw!r}")
        node = node.parent
    return resolved


def is_unc_or_device_path(raw: str) -> bool:
    """True if *raw* is Windows UNC or device-namespace syntax: ``\\\\host\\share``, ``\\\\.\\PhysicalDrive0``, ``\\\\?\\C:\\``, or the ``//host/share`` spelling."""
    if raw[:2] in ("\\\\", "//", "\\/", "/\\"):
        return True
    # Any non-empty drive that is not the "X:" form is a UNC or device root.
    drive = ntpath.splitdrive(raw)[0]
    return bool(drive) and not (len(drive) == 2 and drive[1] == ":")


# Win32 GetDriveTypeW return code for a drive mapped to a network share.
_DRIVE_REMOTE = 4


def is_mapped_network_drive(raw: str) -> bool:
    """True if *raw* names an ordinary Windows drive letter (``Z:\\...``) that is actually MAPPED to a network share (``net use Z: \\\\host\\share``), per a real ``GetDriveTypeW`` call."""
    if os.name != "nt":
        return False
    drive = ntpath.splitdrive(raw)[0]
    if not (len(drive) == 2 and drive[1] == ":"):
        return False
    import ctypes
    try:
        return ctypes.windll.kernel32.GetDriveTypeW(drive + "\\") == _DRIVE_REMOTE
    except (OSError, AttributeError, ValueError):
        return False


def reject_unsafe_path_string(raw: str, *, require_absolute: bool = False,
                               reject_network_drives: bool = False) -> None:
    """Reject a caller-supplied path string LEXICALLY, before any filesystem call."""
    s = raw or ""
    # Backslash-led UNC and device forms are refused on every platform.
    if s[:2] in ("\\\\", "\\/"):
        raise ValueError("UNC and device paths are not allowed")
    # Slash-led UNC and device forms are refused on Windows only.
    if os.name == "nt" and is_unc_or_device_path(s):
        raise ValueError("UNC and device paths are not allowed")
    if reject_network_drives and is_mapped_network_drive(s):
        raise ValueError("network drives are not allowed")
    if require_absolute and not os.path.isabs(s):
        raise ValueError("path must be absolute")
