# SPDX-License-Identifier: AGPL-3.0-or-later
"""Reading ``localm-home.cfg``, the one-line file in a clone that names the
clone's data folder.

Stdlib only and free of ``localm`` imports, so the standalone scripts
(``scripts/rollback_update.py``, ``scripts/report_issue.py``) load it by file
path; ``localm.config`` and ``localm.install_manifest`` import it."""

from __future__ import annotations

import os
import sys
from pathlib import Path

#: The file's name in the clone root.
NAME = "localm-home.cfg"


def read_line(path) -> str:
    """The first non-blank line of the file at *path*, stripped: read as UTF-8
    (with or without a BOM), or on Windows, when that fails, in the console
    (OEM) code page that cmd.exe's ``echo`` writes. "" when the file holds no
    such line. Raises OSError, ValueError or LookupError when it cannot be
    read or decoded."""
    raw = Path(path).read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        if sys.platform != "win32":
            raise
        import ctypes
        text = raw.decode(f"cp{ctypes.windll.kernel32.GetOEMCP()}")
    lines = text.strip().splitlines()
    return lines[0].strip() if lines else ""


def data_dir(line: str, root) -> Path:
    """The data folder a non-empty *line* names: ``~`` expanded, and a
    relative path taken relative to the clone at *root*."""
    home = Path(os.path.expanduser(line))
    return home if home.is_absolute() else Path(root) / home
