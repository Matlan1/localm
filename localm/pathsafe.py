# SPDX-License-Identifier: AGPL-3.0-or-later
"""Filesystem path-confinement helpers shared by the kernel GUI and plugins.

These guarantee a user-supplied name stays directly inside a known base
directory. They are backend-agnostic (no media/ComfyUI knowledge) - the kernel's
session-log browser and the media plugins (image/music/video) all use them, so
they live here rather than being duplicated per caller.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath, PureWindowsPath

from fastapi import HTTPException


def confined_name(base: Path, name: str) -> Path:
    """Resolve *name* and guarantee it stays directly inside *base*, without
    requiring it to exist (rename targets, new files).

    Blocklisting separators is not enough on Windows: a drive-relative name like
    ``C:evil`` joins to ``C:evil`` (outside *base*), and an absolute name
    replaces the join entirely. We verify the *resolved* path's parent is *base*
    and the basename is unchanged, which also rejects ``..`` and nested subpaths.

    Windows reserved device names (``con``, ``nul``, ``com1`` ...) are NOT
    specially rejected: they pass as ordinary basenames and resolve directly
    inside *base*, so confinement still holds for them. Blocking device names is
    deliberately not done here - it is not required for confinement and would
    reject otherwise-legitimate names."""
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


def is_unc_or_device_path(raw: str) -> bool:
    """True when *raw* names a Windows UNC share (``\\\\server\\share``, or its
    forward-slash form ``//server/share``) or a device path (``\\\\.\\PhysicalDrive0``,
    ``\\\\?\\C:\\...``). Purely textual: makes NO filesystem call.

    Being syscall-free is the entire point. On Windows the *first* touch of a UNC
    path is what costs - ``Path.resolve()`` calls ``ntpath.realpath`` ->
    ``_getfinalpathname`` (a ``CreateFileW``), and a UNC target on an unroutable
    host blocks in the SMB redirector for minutes before failing, while a
    *reachable* attacker share makes Windows auto-authenticate and surrender the
    host's net-NTLMv2 credential. So a caller-supplied path string must be
    screened by this BEFORE any stat/resolve/exists, not after.

    Every UNC and device form shares one property - two leading separators - so
    one check covers them all. Deliberately NOT gated on ``os.name == "nt"``: the
    inputs this screens are remote- or lower-privilege-supplied, a leading ``//``
    is never a legitimate value there on any platform, and an unconditional rule
    means the Linux CI run exercises the same branch Windows does."""
    return str(raw).strip().replace("/", "\\").startswith("\\\\")


def confined_under(base: Path, relpath: str) -> Path:
    """Join *relpath* under *base*, guaranteeing the result stays inside it.

    The non-HTTP, nesting-tolerant sibling of :func:`confined_name`, for path
    components that arrive from remote data (a HuggingFace file listing, a
    ComfyUI reply, an update manifest) rather than from a route parameter:

    * it PERMITS nested subpaths (``foo/bar.safetensors``), which a real remote
      file listing uses and ``confined_name`` rejects; and
    * it raises ``ValueError``, not ``HTTPException``, so CLI code, the
      downloader and the updater can use it without importing fastapi semantics.

    Rejects an empty component, a UNC/device path, an absolute or root-anchored
    path, a drive-qualified path (``C:/x`` AND the drive-RELATIVE ``C:x``, which
    pathlib silently lets replace the base on Windows), and any ``..`` segment.
    The component analysis runs under BOTH path flavours regardless of the host
    OS, so ``C:/Windows/win.ini`` is rejected on Linux too - a remote-supplied
    name must be judged by what it would mean on the worst platform, not on
    whichever one happens to be running the check."""
    if not isinstance(relpath, str) or not relpath.strip():
        raise ValueError("empty path component")
    if is_unc_or_device_path(relpath):
        raise ValueError(f"UNC or device path is not allowed: {relpath!r}")
    for flavour in (PureWindowsPath, PurePosixPath):
        pure = flavour(relpath)
        if pure.is_absolute() or pure.drive or pure.root:
            raise ValueError(f"absolute or drive-qualified path: {relpath!r}")
        if any(part == ".." for part in pure.parts):
            raise ValueError(f"'..' is not allowed in a path component: {relpath!r}")
    target = base / relpath
    # is_relative_to is LEXICAL - it does not resolve '..', which is exactly the
    # case that defeats it. That is sound HERE, and only here, because every '..'
    # segment was rejected above under both flavours, so there is nothing left for
    # a resolve to normalise away. Kept as a belt-and-braces assert that the join
    # really did stay inside base; it is deliberately syscall-free (see
    # is_unc_or_device_path on why a stat must not happen before the screen).
    if not target.is_relative_to(base):
        raise ValueError(f"path escapes {base}: {relpath!r}")
    return target
