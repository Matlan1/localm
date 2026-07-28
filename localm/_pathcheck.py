# SPDX-License-Identifier: AGPL-3.0-or-later
"""Syscall-free screens for a path component that came from somewhere untrusted.

TEMPORARY MODULE - MERGE INTO ``localm/pathsafe.py`` AND DELETE THIS FILE.
Tracked as issue #843, which also records the two properties that must survive
the move (both were found by measurement, not review).

These two helpers belong next to ``pathsafe.confined_name``. They live here only
because ``localm/pathsafe.py`` is being rewritten concurrently by another change
(the shared path-helper / UNC-stall work) and adding a second author to that file
mid-flight risks the worst possible outcome for a security primitive: two
divergent copies, each looking authoritative. When the pathsafe versions land,
move these there, update the two call sites named below, and delete this module.
Until then, do NOT add a third copy anywhere.

Call sites, so the collapse is mechanical:
  * ``localm/model_manager/pull.py`` -> ``_snapshot_is_complete`` (confined_under)
  * ``localm/plugins/gui/routes/models.py`` -> ``_spec_names_a_host_path``
    (is_unc_or_device_path)

Why this is not simply ``pathsafe.confined_name``: that helper raises
``fastapi.HTTPException`` (wrong for a CLI/downloader caller) and forbids nested
subpaths (a real HuggingFace file listing legitimately contains
``subdir/model-00001-of-2.safetensors``). Both differences are load-bearing.

There is deliberately no fastapi import here, so a CLI import path does not pull
in the web stack to validate a filename.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath, PureWindowsPath


def is_unc_or_device_path(raw: str) -> bool:
    r"""True when *raw* names a Windows UNC share (``\\server\share``, or its
    forward-slash form ``//server/share``) or a device path (``\\.\PhysicalDrive0``,
    ``\\?\C:\...``). Purely textual: makes NO filesystem call.

    Being syscall-free is the entire point. On Windows the FIRST touch of a UNC
    path is what costs - ``Path.resolve()`` calls ``ntpath.realpath`` ->
    ``_getfinalpathname`` (a ``CreateFileW``), and a UNC target on an unroutable
    host blocks in the SMB redirector for minutes before failing, while a
    *reachable* attacker share makes Windows auto-authenticate and surrender the
    host's net-NTLMv2 credential. So a caller-supplied path string must be
    screened by this BEFORE any stat/resolve/exists, never after.

    Every UNC and device form shares one property - two leading separators - so
    one normalise-then-test covers them all, including the MIXED spellings
    (``\/host\share``, ``/\host/share``) that a two-prefix check misses. Windows
    treats ``\`` and ``/`` interchangeably in the prefix; a raw
    ``startswith("\\\\") or startswith("//")`` does not.

    Deliberately NOT gated on ``os.name == "nt"``: the inputs this screens are
    remote- or lower-privilege-supplied, a leading ``//`` is never a legitimate
    value there on any platform, and an unconditional rule means the Linux CI run
    exercises the same branch Windows does.

    DELIBERATELY NO ``.strip()`` - and this is the subtle half. Stripping first
    looks harmless and is an OVER-MATCH: for ``"  \\\\host\\share\\x"`` Windows
    keeps the leading whitespace and COLLAPSES the doubled separator, so both
    ``PureWindowsPath.drive`` and ``ntpath.splitdrive`` report no drive at all -
    it is an ordinary RELATIVE path under a directory literally named ``"  "``.
    Windows does not strip whitespace to reveal a UNC prefix, so neither may this.

    A predicate can be wrong in two directions: UNDER-matching (the bypass, which
    the normalisation above fixes) and OVER-matching (this). Only the OS parser
    says which side you are on, so the test corpus carries false-positive traps
    next to the hostile cases. Refusing legitimate input is not the safe
    direction: a guard that rejects real paths gets loosened later to compensate,
    and that is how the property it was protecting gets lost.

    A caller that legitimately wants to trim USER-typed input does so itself,
    where the intent is visible (see _spec_names_a_host_path in
    plugins/gui/routes/models.py).
    """
    return str(raw).replace("/", "\\").startswith("\\\\")


def confined_under(base: Path, relpath: str) -> Path:
    """Join *relpath* under *base*, guaranteeing the result stays inside it.

    The non-HTTP, nesting-tolerant sibling of ``pathsafe.confined_name``, for path
    components that arrive from remote data (a HuggingFace file listing, a ComfyUI
    reply, an update manifest) rather than from a route parameter:

    * it PERMITS nested subpaths, which a real remote file listing uses and
      ``confined_name`` rejects; and
    * it raises ``ValueError``, not ``HTTPException``, so CLI code, the downloader
      and the updater can use it without importing fastapi semantics.

    Rejects an empty component, a UNC/device path, an absolute or root-anchored
    path, a drive-qualified path (both ``C:/x`` and the drive-RELATIVE ``C:x``,
    which pathlib silently lets replace the base on Windows and which
    ``is_absolute()`` alone does NOT catch), and any ``..`` segment.

    The component analysis runs under BOTH path flavours regardless of the host
    OS, so a Windows-shaped escape is rejected on Linux too. A remote-supplied
    name has to be judged by what it would mean on the worst platform, not on
    whichever one happens to be running the check - otherwise the Linux CI run
    silently exercises a weaker rule than the box the user is on.
    """
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
    # A drive letter on ANY component, not just the first. pathlib only parses a
    # drive at position 0, so "a/C:evil" passes every check above - and then
    # joinpath() treats that component as drive-relative. MEASURED: with a base on
    # C:, Path(base).joinpath("a", "C:evil") yields <base>/a/EVIL - the "C:" is
    # silently dropped and the caller gets back a path naming a DIFFERENT FILE
    # than it asked for. That is not an escape (it stays under base), which is
    # exactly why a containment-only check cannot see it; it is a silent rename,
    # and at a delete call site it means unlinking the wrong file. With a base on
    # a DIFFERENT drive the same component escapes instead. Reject the shape.
    for part in str(relpath).replace("\\", "/").split("/"):
        if len(part) >= 2 and part[1] == ":":
            raise ValueError(f"drive-qualified path component: {relpath!r}")
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
