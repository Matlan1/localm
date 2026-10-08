# SPDX-License-Identifier: AGPL-3.0-or-later
"""Standard-stream setup shared by the CLI entry points."""

import sys


def force_utf8_stdio() -> None:
    """Reconfigure stdout and stderr to UTF-8 with replacement on Windows.

    A stream that is absent (``sys.stdout``/``sys.stderr`` is None under
    pythonw.exe, which is how the launcher runs) or cannot be reconfigured is
    left as it is. No-op off Windows."""
    if sys.platform != "win32":
        return
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")
