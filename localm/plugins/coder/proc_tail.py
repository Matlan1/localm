# SPDX-License-Identifier: AGPL-3.0-or-later
"""A bounded tail of a child process's stderr, drained on a daemon thread."""

from __future__ import annotations

import collections
import subprocess
import threading


class StderrTail:
    """Drains a Popen's stderr on a daemon thread into a bounded ring of the
    last *maxlines* lines. An unread stderr pipe can deadlock a child that
    fills it, so the draining must never stop; a ring keeps the memory cost
    fixed no matter how chatty the child is."""

    def __init__(self, proc: subprocess.Popen, maxlines: int = 20) -> None:
        self._lines: "collections.deque[str]" = collections.deque(maxlen=maxlines)
        self._lock = threading.Lock()
        threading.Thread(target=self._drain, args=(proc,), daemon=True).start()

    def _drain(self, proc: subprocess.Popen) -> None:
        if proc.stderr is None:
            return
        for raw in proc.stderr:
            line = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
            line = line.rstrip("\n")
            if not line:
                continue
            with self._lock:
                self._lines.append(line)

    def tail(self) -> str:
        """Captured lines, oldest first, one per line. "" if nothing was
        captured (including if nothing was ever written)."""
        with self._lock:
            return "\n".join(self._lines)
