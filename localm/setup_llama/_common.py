# SPDX-License-Identifier: AGPL-3.0-or-later
"""Console output and the stdin helper shared by ``localm setup-llama``.
"""

from __future__ import annotations

import sys

from rich.console import Console

console = Console(highlight=False)


def _flush_stdin() -> None:
    """Discard any input the OS/terminal buffered while we were NOT actually
    waiting on it (e.g. a stray Enter pressed while a driver probe or a
    multi-hundred-MB download was running). Without this, that buffered
    keystroke is silently consumed the instant the NEXT ``click.confirm()``
    prompt appears - answering a question the user never actually read, rather
    than the one they meant to answer (or none at all). Call this immediately
    before every interactive prompt in the setup flow.

    Best-effort and silent on failure: a piped/non-tty stdin (tests, CI, a
    non-interactive install) has nothing to flush and isatty() already guards
    that; any other failure just leaves stray input in place - the pre-fix
    behaviour - which is not a regression, so there is nothing worth surfacing."""
    if not sys.stdin.isatty():
        return
    try:
        if sys.platform == "win32":
            import msvcrt
            while msvcrt.kbhit():
                msvcrt.getch()
        else:
            import termios
            termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
    except Exception:
        pass
