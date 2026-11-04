# SPDX-License-Identifier: AGPL-3.0-or-later
"""A program file for a test that needs one to exist.

A text file named ``*.exe`` is not a program. When a child process reaches it
(a script that runs ``uv`` or ``localm``), cmd.exe hands it to the shell, which
shows Windows' "This app can't run on your PC" dialog on the desktop.
``tests._no_window`` refuses to start a child that could reach one. A test that
needs a ``uv.exe`` or ``localm.exe`` to exist calls ``write_exe_stub``.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


def write_exe_stub(path) -> Path:
    """Create *path* (and its folders) as a real program.

    On Windows the file is a copy of the interpreter running the tests, which
    has a valid program header. Elsewhere it is an empty file, since only
    Windows has the problem. Returns *path* as a ``Path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        shutil.copyfile(sys.executable, path)
    else:
        path.write_bytes(b"")
    return path
