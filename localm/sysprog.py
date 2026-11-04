# SPDX-License-Identifier: AGPL-3.0-or-later
"""Full paths of the programs the operating system ships, found without PATH.

A process started with a bare program name is looked up on PATH (and, on
Windows, in the current directory first), so what runs depends on how the app
was started. :func:`find` names the program by its full path instead: the
Windows system directory for the tools Windows ships, the POSIX default search
path (``confstr("CS_PATH")``) and the ``sbin`` folders for the standard
utilities, and only then the absolute folders on PATH, for a tool a vendor
installs somewhere else. The current directory is never searched.

Stdlib only: :mod:`localm.hwdetect` and :mod:`localm.uv_tool` import this
before the venv holds any of localm's dependencies.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

IS_WINDOWS = sys.platform == "win32"

# Where a Windows tool lives, relative to the system directory (System32).
_WINDOWS_RELATIVE = {
    "powershell": ("WindowsPowerShell", "v1.0", "powershell.exe"),
    "wmic": ("wbem", "WMIC.exe"),
}


def windows_system_dir() -> Optional[str]:
    """The Windows system directory, as Windows reports it; None elsewhere or
    when Windows does not answer."""
    if sys.platform != "win32":
        return None
    import ctypes
    buf = ctypes.create_unicode_buffer(520)
    n = ctypes.windll.kernel32.GetSystemDirectoryW(buf, len(buf))
    return buf.value if 0 < n < len(buf) else None


def posix_system_dirs() -> list:
    """The POSIX default search path, then ``/usr/sbin`` and ``/sbin``."""
    try:
        default = os.confstr("CS_PATH")
    except (AttributeError, ValueError, OSError):
        default = None
    dirs = [d for d in (default or "/bin:/usr/bin").split(":") if d]
    return dirs + [d for d in ("/usr/sbin", "/sbin") if d not in dirs]


def system_candidates(name: str) -> list:
    """The full paths :func:`find` tries for *name* before PATH."""
    if IS_WINDOWS:
        sysdir = windows_system_dir()
        if not sysdir:
            return []
        rel = _WINDOWS_RELATIVE.get(name.lower())
        if rel is None:
            rel = (name if name.lower().endswith(".exe") else name + ".exe",)
        return [os.path.join(sysdir, *rel)]
    return [os.path.join(d, name) for d in posix_system_dirs()]


def _runnable(path: str) -> bool:
    return os.path.isfile(path) and (IS_WINDOWS or os.access(path, os.X_OK))


def _path_names(name: str) -> list:
    """The file names *name* is found under on PATH: on Windows *name* itself
    when it ends in a ``PATHEXT`` extension, else *name* with each of them."""
    if not IS_WINDOWS:
        return [name]
    exts = [e for e in os.environ.get("PATHEXT", ".COM;.EXE;.BAT;.CMD").split(os.pathsep) if e]
    if any(name.lower().endswith(e.lower()) for e in exts):
        return [name]
    return [name + e for e in exts]


def path_candidates(name: str) -> list:
    """The full paths :func:`find` tries for *name* on PATH, in order. Empty,
    relative and ``.`` entries of PATH are skipped, so the current directory is
    never searched; a *name* holding a folder is not looked up on PATH."""
    if not name or os.path.basename(name) != name or (IS_WINDOWS and "/" in name):
        return []
    names = _path_names(name)
    out = []
    for folder in os.environ.get("PATH", "").split(os.pathsep):
        if not folder or not os.path.isabs(folder):
            continue
        out += [os.path.join(folder, n) for n in names]
    return out


def find_on_path(name: str) -> Optional[str]:
    """Full path of the program *name* in the absolute folders on PATH, or None."""
    for candidate in path_candidates(name):
        if _runnable(candidate):
            return candidate
    return None


def find(name: str) -> Optional[str]:
    """Full path of the program *name*: the system folders first, then the
    absolute folders on PATH. None when it is in neither."""
    for candidate in system_candidates(name):
        if _runnable(candidate):
            return candidate
    return find_on_path(name)
