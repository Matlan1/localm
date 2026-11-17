# SPDX-License-Identifier: AGPL-3.0-or-later
"""The check behind the autouse fixture ``_no_home_bound_work_left_running``
(tests/conftest.py): tracked background work (``localm.config.home_bound``)
that a test leaves running."""

from __future__ import annotations

import threading
import time

# Seconds a test's teardown waits for its tracked background work to finish.
WAIT_SECONDS = 10.0


def unfinished_home_bound_runs(before, timeout: float = WAIT_SECONDS) -> list:
    """The tracked ``home_bound`` calls in progress now that are not in *before*
    and have not finished *timeout* seconds from now.

    *before* is a collection of ``HomeBoundRun`` objects (a snapshot of
    ``config.home_bound_runs()``); calls in it are ignored, and so is a call
    running on the current thread. Waits at most *timeout* seconds in all."""
    from localm import config
    me = threading.current_thread()
    deadline = time.monotonic() + timeout
    left = []
    for run in config.home_bound_runs():
        if run in before or run.thread is me:
            continue
        if not run.done.wait(max(0.0, deadline - time.monotonic())):
            left.append(run)
    return left


def describe(runs) -> str:
    """One line per run: its name and its thread's name."""
    return "\n".join(f"  {run.name} (thread {run.thread.name})" for run in runs)
