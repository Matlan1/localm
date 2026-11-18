# SPDX-License-Identifier: AGPL-3.0-or-later
"""Which Python runs a project's pytest: the project's, never localm's own.

localm's interpreter is its own private environment; the project's tests need
the PROJECT's packages, so detection and ``run_tests`` both ask this module for
the interpreter. One resolver, so what the verify oracle checks and what the
model's own tool runs cannot drift apart.

Order:

1. A virtual environment inside the project (``.venv``, ``venv``, ``env``,
   ``.env``) that has pytest installed. Found by looking at its files, never by
   starting it: detection runs on session start and on every gated turn, and a
   project-owned interpreter must not execute anything before the user has
   approved a run.
2. ``python3`` / ``python`` on PATH with pytest installed, asked in isolated
   mode (``-I``) so nothing from the project directory is imported. An
   interpreter that lives inside the project directory is skipped: a file the
   project can plant is not "the interpreter the project uses".
3. localm's own interpreter, only when pytest happens to be installed there (a
   development checkout).
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

PROJECT_VENV_NAMES = (".venv", "venv", "env", ".env")

_PROBE_TIMEOUT_S = 15
_probe_cache: dict[tuple[str, float], bool] = {}


@dataclass(frozen=True)
class PythonLookup:
    """The interpreter to run pytest with, or why there is none."""
    python: Optional[str]
    reason: str = ""


def _venv_python(venv: Path) -> Optional[Path]:
    """The interpreter of *venv* on this platform's layout, or None."""
    sub = ("Scripts", "python.exe") if os.name == "nt" else ("bin", "python")
    exe = venv / sub[0] / sub[1]
    if not (venv / "pyvenv.cfg").is_file() or not exe.is_file():
        return None
    return exe


def _site_packages(venv: Path) -> list[Path]:
    """The site-packages folders of *venv* (Windows ``Lib``, POSIX ``lib``)."""
    found: list[Path] = []
    win = venv / "Lib" / "site-packages"
    if win.is_dir():
        found.append(win)
    for lib in ("lib", "lib64"):
        base = venv / lib
        if base.is_dir():
            found.extend(p for p in sorted(base.glob("python*/site-packages"))
                         if p.is_dir())
    return found


def _venv_has_pytest(venv: Path) -> bool:
    return any((sp / "pytest" / "__init__.py").is_file()
               for sp in _site_packages(venv))


def _inside(path: Path, root: Path) -> bool:
    try:
        Path(os.path.realpath(path)).relative_to(os.path.realpath(root))
    except ValueError:
        return False
    return True


def _probe_has_pytest(exe: str) -> bool:
    """Ask the interpreter *exe* whether pytest imports, in isolated mode."""
    try:
        key = (exe, os.stat(exe).st_mtime)
    except OSError:
        return False
    if key in _probe_cache:
        return _probe_cache[key]
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        done = subprocess.run(
            [exe, "-I", "-c", "import pytest"], stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=_PROBE_TIMEOUT_S, creationflags=flags, check=False)
        ok = done.returncode == 0
    except (OSError, subprocess.SubprocessError):
        ok = False
    _probe_cache[key] = ok
    return ok


def _path_interpreters(cwd: Path) -> list[str]:
    seen: list[str] = []
    for name in ("python3", "python"):
        found = shutil.which(name)
        if not found:
            continue
        found = os.path.abspath(found)
        if _inside(Path(found), cwd) or found in seen:
            continue
        seen.append(found)
    return seen


def _own_interpreter_has_pytest() -> bool:
    try:
        return importlib.util.find_spec("pytest") is not None
    except Exception:                                          # noqa: BLE001
        return False


def _install_hint(venv_name: str) -> str:
    sub = "Scripts\\python" if os.name == "nt" else "bin/python"
    return "%s -m pip install pytest" % (Path(venv_name) / sub)


def find_project_python(cwd: Path) -> PythonLookup:
    """The interpreter that runs *cwd*'s pytest, or a message saying what to fix."""
    cwd = Path(cwd)
    bare: Optional[str] = None
    for name in PROJECT_VENV_NAMES:
        venv = cwd / name
        exe = _venv_python(venv)
        if exe is None:
            continue
        if _venv_has_pytest(venv):
            return PythonLookup(str(exe))
        bare = bare or name
    for exe in _path_interpreters(cwd):
        if _probe_has_pytest(exe):
            return PythonLookup(exe)
    if _own_interpreter_has_pytest():
        return PythonLookup(sys.executable)
    if bare is not None:
        reason = ("The project's virtual environment (%s) has no pytest. "
                  "Install it there: %s" % (bare, _install_hint(bare)))
    else:
        reason = ("No Python with pytest was found for this project: it has "
                  "no virtual environment (.venv, venv, env) with pytest "
                  "installed and python/python3 on PATH has none either. "
                  "Create one and install pytest, for example: "
                  "python -m venv .venv && %s" % _install_hint(".venv"))
    return PythonLookup(None, reason)
