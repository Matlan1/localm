# SPDX-License-Identifier: AGPL-3.0-or-later
"""Filesystem path-confinement helpers shared by the kernel GUI and plugins.

These guarantee a user-supplied name stays directly inside a known base
directory. They are backend-agnostic (no media/ComfyUI knowledge): the kernel's
session-log browser and the media plugins (image/music/video) all use them.

Four distinct shapes:

* ``confined_name`` - one flat basename, HTTP call sites. Raises HTTPException.
* ``confined_under`` - a possibly NESTED relative path, non-HTTP call sites
  (the media clients, CLI/updater code). Raises ValueError. Rejects an
  absolute *relpath* as an escape attempt.
* ``confined_absolute_or_under`` - the same nested-path confinement, but for
  callers where an absolute path landing inside *base* is a legitimate input
  rather than an escape attempt (an LLM tool call naming a file by its full
  path). Raises ValueError.
* ``reject_unsafe_path_string`` - a LEXICAL pre-check for a caller-supplied
  ABSOLUTE path that is about to be handed to the filesystem. It performs no
  syscall itself, so it can run before the first one.
"""

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
    """Resolve *name* and guarantee it stays directly inside *base*, without
    requiring it to exist (rename targets, new files).

    Blocklisting separators is not enough on Windows: a drive-relative name like
    ``C:evil`` joins to ``C:evil`` (outside *base*), and an absolute name
    replaces the join entirely. This verifies the *resolved* path's parent is
    *base* and the basename is unchanged, which also rejects ``..`` and nested
    subpaths AND, as a side effect, an OS-level alias substitution (an NTFS 8.3
    short name resolving to a pre-existing, differently-named file: the resolved
    name differs from the requested one, so this check catches it without any
    dedicated alias-detection logic). It is narrower than
    model_manager/gguf.py's ``_safe_models_filename``, which additionally
    accepts a case-only variant of an existing name; this function does not.

    Reserved characters (``WINDOWS_RESERVED_NAME_CHARS`` above) are rejected
    LEXICALLY, before any filesystem call - ':' is the one with a live
    consequence (NTFS Alternate Data Streams; see that constant's docstring).

    Windows reserved device names (``con``, ``nul``, ``com1`` ...) are NOT
    specially rejected: they pass as ordinary basenames and resolve directly
    inside *base*, so confinement still holds for them."""
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
    """Join *relpath* under *base* and guarantee the result stays strictly inside
    it, raising ``ValueError`` when it would not.

    Two differences from :func:`confined_name`:

    * NESTING IS ALLOWED. ``sub/dir/file.png`` is fine, because ComfyUI's own
      ``subfolder`` field legitimately nests an output one level down. Only
      escape is rejected, not depth.
    * It raises ``ValueError``, not ``fastapi.HTTPException``, so non-HTTP call
      sites (the media clients, CLI and updater code) can use it without
      turning a filesystem decision into an HTTP one.

    The lexical rejections run BEFORE any filesystem call, so a hostile
    component never reaches a syscall. Backslash is treated as a separator on
    every platform, matching ``_apply_update._unsafe_member``: a POSIX filename
    may legally contain one, but splitting it can only ever confine further.

    Rejected: an empty result, an absolute component (which would REPLACE the
    base entirely under pathlib's join), a drive-qualified component (``C:evil``
    joins to ``C:evil``, outside *base*, and is absolute on Windows only - so it
    is rejected on every platform for uniform behavior), any ``..`` component,
    a reserved character (``WINDOWS_RESERVED_NAME_CHARS`` - ':' opens an NTFS
    Alternate Data Stream rather than failing the write), anything whose
    RESOLVED location is not strictly below *base* (which is what catches a
    symlink inside *base* pointing out of it), and - per component, not just
    the last one - a resolved name that does not match the requested one (an
    OS-level alias substitution: an NTFS 8.3 short name resolving to a
    pre-existing, differently-named sibling stays strictly under *base*, so
    containment alone would not catch it; the same check
    :func:`confined_name` makes, generalised to every nesting level since a
    subfolder component can be aliased too).

    CONTRACT: *base* itself must ALREADY be trusted before it reaches this
    function. Only *relpath* is lexically validated here - *base* is resolved
    directly (see below), so a caller that hands this an unvalidated,
    attacker-influenced *base* (a UNC path dials SMB before the containment
    check below can refuse it) has to guard *base* itself first; this
    function cannot do it for them without widening its own contract.
    """
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
    """Resolve *raw* and guarantee it stays inside *base*, raising
    ``ValueError`` when it would not.

    Unlike :func:`confined_under`, an ABSOLUTE *raw* is ACCEPTED rather than
    rejected as an escape attempt, as long as it resolves inside *base* - for
    callers where an absolute path naming a file already inside the confined
    root is a legitimate, expected input (an LLM coding-agent tool call
    naming a file by its full path; an MCP ``output_path``). A RELATIVE *raw*
    is joined onto *base* first, exactly like ``confined_under`` (nesting
    permitted, only escape rejected).

    Rejected, in order, before any filesystem call the string would drive: an
    empty or collapsing result, a UNC or device path (judged unconditionally
    via :func:`is_unc_or_device_path`, since the syscall that would resolve
    one dials SMB and can hang for minutes, and *raw* here is caller-supplied
    rather than typed by the user into a folder picker), and a reserved
    character (``WINDOWS_RESERVED_NAME_CHARS``). After resolving: anything
    outside *base*, and - per component, walking back from the resolved leaf,
    exactly like ``confined_under`` - a resolved name that does not match the
    requested one (an OS-level short-name alias; see that function's
    docstring for why containment alone cannot catch it). For an absolute
    *raw* only the components AFTER its anchor are user-supplied content to
    check - the anchor/drive is not - and only however many of them actually
    land inside *base* after resolution are compared (an absolute *raw* may
    repeat *base*'s own already-trusted prefix verbatim)."""
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
    """True if *raw* is Windows UNC or device-namespace syntax: ``\\\\host\\share``,
    ``\\\\.\\PhysicalDrive0``, ``\\\\?\\C:\\``, or the ``//host/share`` spelling.

    Judged by WINDOWS rules on EVERY host, and NOT gated on ``os.name``. This is
    a question about what the string MEANS, not about where it is being
    evaluated: a name that arrives from a remote source (a HuggingFace
    ``rfilename``, a ComfyUI filename) is judged by what it would do on the worst
    platform, not the running one. It also means a Linux run exercises exactly
    the branch Windows does.

    This is the PREDICATE. Whether a given call site should REFUSE such a path is
    a policy question with a different answer per route - see
    :func:`reject_unsafe_path_string`, which is the policy for a LOCAL path the
    user themselves named. Callers handling REMOTE-supplied values should use this
    predicate directly and refuse unconditionally.

    MIXED SEPARATORS COUNT. Windows treats ``\\`` and ``/`` interchangeably in the
    UNC prefix, so ``\\/host\\share`` and ``/\\host/share`` are UNC to the OS even
    though neither starts with a doubled separator of one kind. Both
    ``PureWindowsPath(...).drive`` and ``ntpath.splitdrive`` report a UNC drive
    for them, while a bare ``startswith("\\\\\\\\")`` or ``startswith("//")``
    test returns False for both spellings.

    ``ntpath`` is used rather than ``os.path``: it is the Windows
    implementation on every host, which is what "judge by Windows rules
    everywhere" requires, and it is authoritative for spellings not enumerated
    here."""
    if raw[:2] in ("\\\\", "//", "\\/", "/\\"):
        return True
    # Any non-empty drive that is not the "X:" form is a UNC or device root.
    drive = ntpath.splitdrive(raw)[0]
    return bool(drive) and not (len(drive) == 2 and drive[1] == ":")


# Win32 GetDriveTypeW return code for a drive mapped to a network share.
_DRIVE_REMOTE = 4


def is_mapped_network_drive(raw: str) -> bool:
    """True if *raw* names an ordinary Windows drive letter (``Z:\\...``) that
    is actually MAPPED to a network share (``net use Z: \\\\host\\share``),
    per a real ``GetDriveTypeW`` call.

    This is NOT a syntax predicate like :func:`is_unc_or_device_path` - a
    mapped drive is syntactically an ordinary "X:" local path, and that
    function correctly returns False for it - so telling the two apart
    requires asking the RUNNING MACHINE what "Z:" actually is. It therefore
    only means anything for a path being evaluated on the host that mapped the
    drive, and callers combine the two predicates rather than expecting either
    to cover both cases.

    Windows-only: on any other platform, and for *raw* with no drive-letter
    prefix at all (a UNC path, a relative path, a POSIX path), this always
    returns False without any Win32 call - :func:`is_unc_or_device_path`
    already owns the UNC/device shape.

    Never raises: a Win32 call failure, or a drive letter that does not exist
    on this machine, both return False."""
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
    """Reject a caller-supplied path string LEXICALLY, before any filesystem call.

    The syscall itself is the vulnerability. ``Path.resolve`` on Windows calls
    ``_getfinalpathname`` (CreateFileW) plus a stat, so a UNC path pointed at an
    attacker's SMB server makes an outbound connection that Windows
    AUTO-AUTHENTICATES, surrendering the host's net-NTLMv2 credential - and it
    completes before any allowlist check downstream can reject the path. It also
    blocks: a UNC path to a non-routable address stalls for minutes before
    failing. Inside an ``async def`` that is the whole event loop, per request.
    So the check is pure string work and performs no syscall of its own.

    ``\\\\``-prefixed input is rejected on EVERY platform, not just Windows: it
    covers UNC (``\\\\host\\share``) and the device namespaces (``\\\\.\\PhysicalDrive0``,
    ``\\\\?\\C:\\``), and a leading backslash pair is not a meaningful prefix for
    any local path a picker or a config value would carry on POSIX either.

    ``//``-prefixed input is rejected on Windows ONLY, where it is an equivalent
    spelling of UNC. On POSIX a leading ``//`` is a legal path prefix equivalent
    to ``/`` (POSIX leaves exactly-two-slashes implementation-defined), and this
    function's input is a path the USER named (a folder picker, a configured
    directory), not remote data.

    That platform split is the POLICY for a local path, not the syntax rule. A
    caller handling a REMOTE-supplied value must not inherit it - use
    :func:`is_unc_or_device_path` directly and refuse unconditionally, or
    :func:`confined_under`, which rejects any leading ``/`` on every platform.

    *reject_network_drives*, OFF by default (an ordinary drive letter like
    ``Q:\\ordinary`` is accepted unless a caller opts in): when True, also refuse
    *raw* if it names a Windows drive letter that
    :func:`is_mapped_network_drive` reports is actually mapped to a network
    share. This is a POLICY choice, not the SMB-dial-and-hang safety property
    the rest of this function guards - a mapped drive is already connected
    and does not carry the same stall/credential risk a fresh UNC dial does -
    so it is opt-in per call, driven by the caller's own read of the
    ``allow_network_drives`` config setting, never decided here.

    Raises ``ValueError`` (callers translate it to their own error shape).
    """
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
