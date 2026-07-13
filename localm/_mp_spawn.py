# SPDX-License-Identifier: AGPL-3.0-or-later
"""Windows multiprocessing-spawn fix for the branded LocaLM.exe launcher (#617).

CPython's multiprocessing has a Windows-only optimization (bpo-35797):
whenever it detects the running interpreter differs from ``sys._base_executable``
(its own definition of "running inside a venv"), a spawned child is launched via
``sys._base_executable`` instead of ``sys.executable`` - see
``multiprocessing/popen_spawn_win32.py``'s ``WINENV`` check. CPython computes
``sys._base_executable`` as ``<base_prefix>/<basename of the running executable>``;
it does not look up what the base install's binary is actually named.

localm's branded launcher (``localm make-launcher`` -> ``<venv>/localm-app/
LocaLM.exe``, see applaunch.py) is a COPY of the base interpreter renamed to
LocaLM.exe. Running under that renamed copy, ``sys._base_executable`` becomes
``<base_prefix>/LocaLM.exe`` - a file that does not exist (the base install's
real file is named python.exe/python3) - so every
``multiprocessing.get_context("spawn")`` child (a GGUF model load, the voice/STT
worker) fails with ``FileNotFoundError: [WinError 2] The system cannot find the
file specified``. The GGUF loader (gguf.py) reports that as a misleading "Native
llama runtime failed to load" error that has nothing to do with the actual
llama.cpp runtime - confirmed live via GitHub issue #617.

Fix: point multiprocessing at the venv's own (never renamed) python.exe/python3
launcher instead of trusting ``sys.executable``/``sys._base_executable`` as-is.
That file's basename always matches a real file in the base install directory,
so CPython's own substitution resolves correctly one hop later. Calling this when
NOT running under a renamed launcher is a harmless no-op (it just repoints at the
interpreter that is already running).
"""

from __future__ import annotations

import multiprocessing
import sys
from pathlib import Path


def ensure_spawn_uses_venv_python() -> None:
    """Make ``multiprocessing.get_context("spawn")`` children spawn via the venv's
    own interpreter rather than trusting a possibly-renamed ``sys.executable`` (see
    module docstring). Windows-only; a no-op elsewhere. Best-effort: leaves
    multiprocessing's default untouched if the expected venv layout is not found -
    this must never block a normal launch, branded or not."""
    if sys.platform != "win32":
        return
    venv_python = Path(sys.prefix) / "Scripts" / "python.exe"
    if venv_python.is_file():
        multiprocessing.set_executable(str(venv_python))
