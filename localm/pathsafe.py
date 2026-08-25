# SPDX-License-Identifier: AGPL-3.0-or-later
"""Filesystem path-confinement helpers shared by the kernel GUI and plugins."""

from __future__ import annotations

import ntpath
import os
from pathlib import Path

from fastapi import HTTPException

# Characters Windows itself refuses to let a real filename contain
# (docs.microsoft.com/windows/win32/fileio/naming-a-file - "Naming
# Conventions"), plus the C0 control range - MINUS '/' and '\\', deliberately:
# both confined_name and confined_under already treat separators correctly
# and PLATFORM-APPROPRIATELY on their own (confined_name via its
# name != Path(name).name check, which - unlike a blanket rejection - accepts
# a literal backslash as an ORDINARY basename character on POSIX, matching
# PurePosixPath's own behavior and this module's own tested contract, see
# test_pathsafe_confined_name.py's test_backslash_is_an_ordinary_basename_on_posix;
# confined_under via its explicit split on '/' after normalizing '\\' to it).
# Including them here would silently override that already-correct,
# already-tested platform split with a blanket, POSIX-incorrect rejection -
# a regression this constant must not introduce while closing an unrelated gap.
#
# ':' is the character that matters most of what remains: it does not fail
# file creation at all - it opens an NTFS Alternate Data Stream instead, so
# 'somefile.exe:hidden.gguf' both passes a naive "no separators" check AND
# lands INSIDE base (the confinement check below genuinely holds - this is
# not CWE-22/path-injection), while writing its content into a stream hidden
# from a normal directory listing behind an innocuous, apparently-empty
# sibling 'somefile.exe'. Live-confirmed against this exact module:
# confined_name(base, "somefile.exe:hidden.gguf") was accepted and
# successfully wrote a hidden stream before this constant existed. Near-same
# set as localm/model_manager/gguf.py's _safe_models_filename uses (that
# module's own _WINDOWS_RESERVED_CHARS, which #1068 hardened first) - that
# one also includes '/' and '\\', safe there only because that validator's
# OWN separate Path(filename).name check runs first and unconditionally
# rejects any separator before this set is even consulted, so the overlap is
# redundant rather than reachable; duplicated here (not imported) rather
# than shared, since pathsafe is the lower-level, backend-agnostic module
# gguf.py (or anything else) may depend on, never the reverse. Exported (no
# leading underscore) so a caller needing a bare character check without a
# base_dir - e.g. a filename that will be folded into a synthesized,
# already-unique on-disk name rather than resolved directly - can reuse the
# SAME set instead of maintaining its own copy that can drift.
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
    # PER-COMPONENT, not just position 1 of the whole string. The early check
    # above only sees a drive on the FIRST component, so ``a/C:evil`` slipped
    # past it - and that is not a harmless miss. pathlib's joinpath treats a
    # drive-relative component against a SAME-DRIVE base by silently DROPPING
    # the drive: base.joinpath("a", "C:evil") -> base/a/evil (measured on 3.12).
    # The result stays strictly under base, so the resolve-based containment
    # check below cannot see it - it is not an escape, it is a SILENT RENAME.
    # For the ComfyUI delete call site that means unlinking a real file that is
    # NOT the one ComfyUI named, while reporting containment succeeded, which is
    # exactly the failure AGENTS.md rule 5 forbids. It also reproduces only when
    # base is on the same drive as the injected letter, so it would present as
    # "works on my D: install, deletes the wrong file on a C: one".
    # Reported by the WS2 lane against this exact call site.
    for p in parts:
        if len(p) >= 2 and p[1] == ":":
            raise ValueError(f"drive-qualified path component not allowed: {relpath!r}")
    # Reserved characters (WINDOWS_RESERVED_NAME_CHARS - see that constant's
    # docstring above confined_name), checked per component now that
    # separators have already been consumed by the split above so neither
    # '/' nor '\\' can appear WITHIN a component here. ':' is the one with a
    # live consequence beyond position 1 (already covered above): NTFS opens
    # an Alternate Data Stream for it rather than failing the write, so e.g.
    # 'sub/somefile.exe:hidden.gguf' stayed confined (this check exists
    # precisely because that DOES pass containment - it is not CWE-22) while
    # writing invisibly behind an apparently-empty sibling file. Live-
    # confirmed against this exact function before this check existed.
    for p in parts:
        if set(p) & WINDOWS_RESERVED_NAME_CHARS:
            raise ValueError(f"reserved character in path component not allowed: {relpath!r}")
    try:
        resolved = base.joinpath(*parts).resolve()
        base_resolved = base.resolve()
    except (OSError, ValueError) as e:
        raise ValueError(f"path could not be resolved: {relpath!r} ({e})")
    # Strictly BELOW base: base itself is not a valid target (an empty filename
    # must not resolve to "delete the output directory").
    if base_resolved not in resolved.parents:
        raise ValueError(f"path escapes {base_resolved}: {relpath!r}")
    # NAME PRESERVATION, per component, walking back up from the resolved leaf.
    # confined_name has always had this (resolved.name != name) as a side effect
    # of its single-component check; confined_under never got the equivalent,
    # even though its own docstring names "the ComfyUI delete call site" as a
    # consumer - an 8.3 short name resolving to a pre-existing, differently-named
    # sibling stays strictly under base (this function's only check until now),
    # so containment held while the identity silently changed. Live-confirmed
    # against this exact function: confined_under(base, "LONGMO~1.GGU") returned
    # a DIFFERENT real file's resolved path with no error, before this loop
    # existed. Checked at every nesting level, not just the last component - an
    # aliased INTERMEDIATE directory (`subfolder` in ComfyUI's own reply) would
    # otherwise still resolve strictly under base while silently descending into
    # the wrong subdirectory.
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
    # Authoritative backstop: any non-empty drive that is NOT the "X:" form is a
    # UNC or device root (``\\host\share``, ``\\?\C:``, ``\\.\PhysicalDrive0``).
    drive = ntpath.splitdrive(raw)[0]
    return bool(drive) and not (len(drive) == 2 and drive[1] == ":")


# Win32 GetDriveTypeW return code for a drive mapped to a network share
# (docs.microsoft.com/windows/win32/api/fileapi/nf-fileapi-getdrivetypew).
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
    # BACKSLASH-LED UNC/device forms are refused on every platform: a leading
    # ``\\`` or ``\/`` is not a meaningful prefix for a local path on POSIX
    # either, so there is nothing legitimate to protect there.
    if s[:2] in ("\\\\", "\\/"):
        raise ValueError("UNC and device paths are not allowed")
    # SLASH-LED forms (``//host/share``, ``/\host/share``) are refused on Windows
    # ONLY, via the full predicate so device namespaces and mixed separators are
    # covered rather than re-enumerated here. On POSIX a leading ``//`` is a legal
    # prefix equivalent to ``/``, and this function's input is a path the USER
    # named, so refusing it there would break a legitimate local folder.
    if os.name == "nt" and is_unc_or_device_path(s):
        raise ValueError("UNC and device paths are not allowed")
    if reject_network_drives and is_mapped_network_drive(s):
        raise ValueError("network drives are not allowed")
    if require_absolute and not os.path.isabs(s):
        raise ValueError("path must be absolute")
