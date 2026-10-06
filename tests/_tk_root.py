# SPDX-License-Identifier: AGPL-3.0-or-later
"""Build a real Tk window in a test.

Skips only when there is no display. A Tcl library read failure (init.tcl or
tk.tcl) is retried a few times and then fails the test, so it cannot pass as a
missing display.
"""

from __future__ import annotations

import time

import pytest

ATTEMPTS = 5
RETRY_DELAY = 0.05


def is_no_display(error) -> bool:
    return "display" in str(error).lower()


def build_tk_root(factory, *, attempts=ATTEMPTS, delay=RETRY_DELAY,
                  sleep=time.sleep):
    """Return ``factory()``, retrying a failed Tk build up to *attempts* times.

    *factory* is ``tkinter.Tk`` or any callable that builds a Tk window."""
    tkinter = pytest.importorskip("tkinter")
    last = None
    for attempt in range(attempts):
        try:
            return factory()
        except tkinter.TclError as e:
            if is_no_display(e):
                pytest.skip(f"no display: {e}")
            last = e
            if attempt + 1 < attempts:
                sleep(delay)
    pytest.fail(f"Tk could not be built after {attempts} attempts, and the "
                f"error is not a missing display: {last}", pytrace=False)
