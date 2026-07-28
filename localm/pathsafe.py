# SPDX-License-Identifier: AGPL-3.0-or-later
"""Filesystem path-confinement helpers shared by the kernel GUI and plugins.

These guarantee a user-supplied name stays directly inside a known base
directory. They are backend-agnostic (no media/ComfyUI knowledge) - the kernel's
session-log browser and the media plugins (image/music/video) all use them, so
they live here rather than being duplicated per caller.

Three shapes, deliberately distinct:

* ``confined_name`` - one flat basename, HTTP call sites. Raises HTTPException.
* ``confined_under`` - a possibly NESTED relative path, non-HTTP call sites
  (the media clients, CLI/updater code). Raises ValueError.
* ``reject_unsafe_path_string`` - a LEXICAL pre-check for a caller-supplied
  ABSOLUTE path that is about to be handed to the filesystem. It performs no
  syscall itself, so it can run before the first one.
"""

from __future__ import annotations

import os
from pathlib import Path

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
    and anything whose RESOLVED location is not strictly below *base* (which is
    what catches a symlink inside *base* pointing out of it).
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
    try:
        resolved = base.joinpath(*parts).resolve()
        base_resolved = base.resolve()
    except (OSError, ValueError) as e:
        raise ValueError(f"path could not be resolved: {relpath!r} ({e})")
    # Strictly BELOW base: base itself is not a valid target (an empty filename
    # must not resolve to "delete the output directory").
    if base_resolved not in resolved.parents:
        raise ValueError(f"path escapes {base_resolved}: {relpath!r}")
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
    predicate directly and refuse unconditionally."""
    return raw.startswith("\\\\") or raw.startswith("//")


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
    if s.startswith("\\\\") or (os.name == "nt" and s.startswith("//")):
        raise ValueError("UNC and device paths are not allowed")
    if require_absolute and not os.path.isabs(s):
        raise ValueError("path must be absolute")
