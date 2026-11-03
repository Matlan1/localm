# SPDX-License-Identifier: AGPL-3.0-or-later
"""Full paths of the programs the operating system ships, found without PATH.

A process started with a bare program name is looked up on PATH (and, on
Windows, in the current directory first), so what runs depends on how the app
was started. :func:`find` names the program by its full path instead: the
Windows system directory for the tools Windows ships, the POSIX default search
path (``confstr("CS_PATH")``) and the ``sbin`` folders for the standard
utilities, and only then PATH, for a tool a vendor installs somewhere else.

Stdlib only: :mod:`localm.hwdetect` and :mod:`localm.uv_tool` import this
before the venv holds any of localm's dependencies.
"""

from __future__ import annotations

import os
import shutil
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


def find(name: str) -> Optional[str]:
    """Full path of the program *name*: the system folders first, then PATH.
    None when it is in neither."""
    for candidate in system_candidates(name):
        if _runnable(candidate):
            return candidate
    on_path = shutil.which(name)
    return os.path.abspath(on_path) if on_path else None
