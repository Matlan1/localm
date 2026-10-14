# SPDX-License-Identifier: AGPL-3.0-or-later
"""Console output and the maintainer contact address used across the bug reporter.
"""

from __future__ import annotations

import sys

from localm.console import SafeConsole as Console


console = Console(highlight=False)


# The maintainer contact address bug reports are addressed to.
MAINTAINER_EMAIL = "theilige@gmail.com"


def stdin_is_interactive() -> bool:
    """True when stdin is a terminal a person can answer at: ``isatty()`` is
    True and, on Windows, stdin is also a console (``GetConsoleMode``
    succeeds), which the NUL device is not although it reports ``isatty()``.
    False when either check fails or cannot run. Never raises."""
    try:
        if not (sys.stdin and sys.stdin.isatty()):
            return False
        if sys.platform != "win32":
            return True
        import ctypes
        import msvcrt
        handle = msvcrt.get_osfhandle(sys.stdin.fileno())
        mode = ctypes.c_uint32()
        return bool(ctypes.windll.kernel32.GetConsoleMode(handle, ctypes.byref(mode)))
    except Exception:
        return False
