# SPDX-License-Identifier: AGPL-3.0-or-later
"""Filesystem path-confinement helpers shared by the kernel GUI and plugins.

These guarantee a user-supplied name stays directly inside a known base
directory. They are backend-agnostic (no media/ComfyUI knowledge) - the kernel's
session-log browser and the media plugins (image/music/video) all use them, so
they live here rather than being duplicated per caller.

Four shapes, deliberately distinct:

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
    """Resolve *name* and guarantee it stays directly inside *base*, without
    requiring it to exist (rename targets, new files).

    Blocklisting separators is not enough on Windows: a drive-relative name like
    ``C:evil`` joins to ``C:evil`` (outside *base*), and an absolute name
    replaces the join entirely. We verify the *resolved* path's parent is *base*
    and the basename is unchanged, which also rejects ``..`` and nested subpaths
    AND, as a side effect, an OS-level alias substitution (an NTFS 8.3 short
    name resolving to a pre-existing, differently-named file: the resolved
    name differs from the requested one, so this check catches it without any
    dedicated alias-detection logic - narrower than
    model_manager/gguf.py's ``_safe_models_filename``, which additionally
    accepts a case-only variant of an existing name; this function does not,
    which is a minor strictness difference, not a security gap).

    Reserved characters (``WINDOWS_RESERVED_NAME_CHARS`` above) are rejected
    LEXICALLY, before any filesystem call - ':' is the one with a live
    consequence (NTFS Alternate Data Streams; see that constant's docstring).

    Windows reserved device names (``con``, ``nul``, ``com1`` ...) are NOT
    specially rejected: they pass as ordinary basenames and resolve directly
    inside *base*, so confinement still holds for them. Blocking device names is
    deliberately not done here - it is not required for confinement and would
    reject otherwise-legitimate names."""
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

    Two deliberate differences from :func:`confined_name`, both required by the
    callers this exists for:

    * NESTING IS ALLOWED. ``sub/dir/file.png`` is fine, because ComfyUI's own
      ``subfolder`` field legitimately nests an output one level down. Only
      escape is rejected, not depth.
    * It raises ``ValueError``, not ``fastapi.HTTPException``, so non-HTTP call
      sites (the media clients, CLI and updater code) can use it without
      pretending a filesystem decision is an HTTP one.

    The lexical rejections run BEFORE any filesystem call, so a hostile
    component never reaches a syscall. Backslash is treated as a separator on
    every platform, matching ``_apply_update._unsafe_member``: a POSIX filename
    may legally contain one, but splitting it can only ever confine further, and
    a uniform rule beats a platform-conditional one.

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
    containment alone would not catch it; matches :func:`confined_name`'s own
    ``resolved.name != name`` check, generalised to every nesting level since a
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
    """Resolve *raw* and guarantee it stays inside *base*, raising
    ``ValueError`` when it would not.

    Unlike :func:`confined_under`, an ABSOLUTE *raw* is ACCEPTED rather than
    rejected as an escape attempt, as long as it resolves inside *base* - for
    callers where an absolute path naming a file already inside the confined
    root is a legitimate, expected input (an LLM coding-agent tool call
    naming a file by its full path; an MCP ``output_path``), not merely
    tolerated the way :func:`confined_name`/:func:`confined_under`'s callers
    treat one. A RELATIVE *raw* is joined onto *base* first, exactly like
    ``confined_under`` (nesting permitted, only escape rejected).

    Rejected, in order, before any filesystem call the string would drive: an
    empty or collapsing result, a UNC or device path (the syscall that would
    resolve one dials SMB and can hang for minutes - same reasoning as
    :func:`reject_unsafe_path_string`, judged unconditionally via
    :func:`is_unc_or_device_path` since *raw* here is caller-supplied, not
    something the user themselves typed into a folder picker), and a reserved
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

    Judged by WINDOWS rules on EVERY host, deliberately, and NOT gated on
    ``os.name``. This is a question about what the string MEANS, not about where
    it is being evaluated: a name that arrives from a remote source (a HuggingFace
    ``rfilename``, a ComfyUI filename) should be judged by what it would do on the
    worst platform, not the running one. It also means a Linux CI run exercises
    exactly the branch Windows does, instead of the rule going untested until it
    reaches a Windows box.

    This is the PREDICATE. Whether a given call site should REFUSE such a path is
    a policy question with a different answer per route - see
    :func:`reject_unsafe_path_string`, which is the policy for a LOCAL path the
    user themselves named. Callers handling REMOTE-supplied values should use this
    predicate directly and refuse unconditionally.

    MIXED SEPARATORS COUNT. Windows treats ``\\`` and ``/`` interchangeably in the
    UNC prefix, so ``\\/host\\share`` and ``/\\host/share`` are UNC to the OS even
    though neither starts with a doubled separator of one kind. Both
    ``PureWindowsPath(...).drive`` and ``ntpath.splitdrive`` report a UNC drive
    for them (measured). An earlier revision tested only ``startswith("\\\\\\\\")``
    or ``startswith("//")`` and returned False for both spellings - a live bypass
    in precisely the position this predicate exists to guard, since its documented
    use is remote-supplied values. Reported by the WS7 and WS2 lanes.

    ``ntpath`` is used rather than ``os.path`` on purpose: it is the Windows
    implementation on every host, which is what "judge by Windows rules
    everywhere" requires, and it is authoritative for spellings not enumerated
    here."""
    if raw[:2] in ("\\\\", "//", "\\/", "/\\"):
        return True
    # Authoritative backstop: any non-empty drive that is NOT the "X:" form is a
    # UNC or device root (``\\host\share``, ``\\?\C:``, ``\\.\PhysicalDrive0``).
    drive = ntpath.splitdrive(raw)[0]
    return bool(drive) and not (len(drive) == 2 and drive[1] == ":")


def reject_unsafe_path_string(raw: str, *, require_absolute: bool = False) -> None:
    """Reject a caller-supplied path string LEXICALLY, before any filesystem call.

    This exists because the syscall itself is the vulnerability. ``Path.resolve``
    on Windows calls ``_getfinalpathname`` (CreateFileW) plus a stat, so a UNC
    path pointed at an attacker's SMB server makes an outbound connection that
    Windows AUTO-AUTHENTICATES, surrendering the host's net-NTLMv2 credential -
    and it completes before any allowlist check downstream can reject the path.
    It also blocks: measured on a Windows box, a UNC path to a non-routable
    RFC5737 address stalled 271 seconds before WinError 64, and an unresolvable
    UNC hostname 9.9 seconds. Inside an ``async def`` that is the whole event
    loop, per request. So the check has to be pure string work, and it is.

    ``\\\\``-prefixed input is rejected on EVERY platform, not just Windows: it
    covers UNC (``\\\\host\\share``) and the device namespaces (``\\\\.\\PhysicalDrive0``,
    ``\\\\?\\C:\\``), and a leading backslash pair is not a meaningful prefix for
    any local path a picker or a config value would carry on POSIX either. A
    platform-conditional security check is one that rots.

    ``//``-prefixed input is rejected on Windows ONLY, where it is an equivalent
    spelling of UNC. On POSIX a leading ``//`` is a legal path prefix equivalent
    to ``/`` (POSIX leaves exactly-two-slashes implementation-defined), so
    rejecting it here would break a legitimate local folder for no security gain:
    this function's input is a path the USER named (a folder picker, a configured
    directory), not remote data.

    That platform split is the POLICY for a local path, not the syntax rule. A
    caller handling a REMOTE-supplied value must not inherit it - use
    :func:`is_unc_or_device_path` directly and refuse unconditionally, or
    :func:`confined_under`, which rejects any leading ``/`` on every platform.

    Raises ``ValueError`` (callers translate it to their own error shape).
    """
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
    if require_absolute and not os.path.isabs(s):
        raise ValueError("path must be absolute")
