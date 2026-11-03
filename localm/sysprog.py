# SPDX-License-Identifier: AGPL-3.0-or-later
"""Full paths of the programs the operating system ships, found without PATH.

A process started with a bare program name is looked up on PATH (and, on
Windows, in the current directory first), so what runs depends on how the app
was started. :func:`find` names the program by its full path instead: the
Windows system directory for the tools Windows ships, the POSIX default search
path (``confstr("CS_PATH")``) and the ``sbin`` folders for the standard
utilities, the GPU driver's own folders for its tools, and only then, in a
source checkout or for a GPU driver tool, PATH. An install setup made never
takes any other program from PATH.

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


# GPU driver tools, which a GPU driver installs outside the system folders.
_DRIVER_TOOLS = ("nvidia-smi", "rocm-smi", "rocminfo")


def driver_candidates(name: str) -> list:
    """The fixed folders the GPU driver installs the driver tool *name* in, as
    full paths (empty for any other program)."""
    if name.lower() not in _DRIVER_TOOLS:
        return []
    if IS_WINDOWS:
        pf = os.environ.get("ProgramW6432") or os.environ.get("ProgramFiles")
        if not pf:
            return []
        exe = name + ".exe"
        return [os.path.join(pf, "NVIDIA Corporation", "NVSMI", exe),  # hygiene-ok: vendor install location
                os.path.join(pf, "AMD", "ROCm", "bin", exe)]  # hygiene-ok: vendor install location
    rocm = os.environ.get("ROCM_PATH") or "/opt/rocm"  # hygiene-ok: vendor install location
    return [os.path.join(rocm, "bin", name),
            os.path.join("/opt/rocm", "bin", name)]  # hygiene-ok: vendor install location


def setup_install() -> bool:
    """Whether this localm is an install setup made (an install record beside the
    package), which uses only what it brought and the operating system's own
    programs."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.isfile(os.path.join(root, ".localm-install.json"))


def _runnable(path: str) -> bool:
    return os.path.isfile(path) and (IS_WINDOWS or os.access(path, os.X_OK))


def find_system(name: str) -> Optional[str]:
    """Full path of the program *name* in the system folders
    (:func:`system_candidates`), or in a GPU driver's fixed folders
    (:func:`driver_candidates`); None when it is in none of them. PATH is never
    searched."""
    for candidate in system_candidates(name) + driver_candidates(name):
        if _runnable(candidate):
            return candidate
    return None


def require_system(name: str) -> str:
    """:func:`find_system`, or FileNotFoundError naming *name*."""
    path = find_system(name)
    if path is None:
        raise FileNotFoundError(2, f"{name} is not in the system folders", name)
    return path


def find(name: str) -> Optional[str]:
    """Full path of the program *name*: :func:`find_system`, then, in a source
    checkout (not :func:`setup_install`) or for a GPU driver tool, PATH. None
    when it is in none of them."""
    found = find_system(name)
    if found or (setup_install() and name.lower() not in _DRIVER_TOOLS):
        return found
    on_path = shutil.which(name)
    return os.path.abspath(on_path) if on_path else None
