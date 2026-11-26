# SPDX-License-Identifier: AGPL-3.0-or-later
"""The applier's side of the small update window (:mod:`.window_tk`, a program of its
own): :class:`Window` starts it and talks to it. Standard library only, so that the
program can load this file for the words both sides use."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Callable, Optional

#: The program that shows the window. It reads the words below from this file.
PROGRAM = Path(__file__).with_name("window_tk.py")
#: Seconds the program has to say whether it shows the window.
SHOW_S = 20.0
#: Seconds the outcome stays in the window before it closes itself.
END_S = 5.0
#: What the program prints: the two answers to a question, and that it shows the window.
AGAIN, CANCEL, SHOWN = "again", "cancel", "shown"


# ---------------------------------------------------------- the applier's side

class Window:
    """The window of one update, shown by the program :data:`PROGRAM` in its own
    process. Each method returns at once except :meth:`ask`, which waits for the
    user."""

    def __init__(self, proc: subprocess.Popen) -> None:
        self.proc = proc
        self.gone = False

    @classmethod
    def start(cls, say: Callable[[str], None], program=None,
              show_s: float = SHOW_S) -> Optional[Window]:
        """Start the window; None, with the reason given to *say*, when it cannot be
        shown here (no display, no Tk), and the update then goes on without it."""
        program = Path(program or PROGRAM)
        flags: dict = ({"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)}
                       if os.name == "nt" else {})
        try:
            proc = subprocess.Popen(
                [sys.executable, "-I", "-S", str(program)], stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                encoding="utf-8", errors="replace", **flags)
        except OSError as e:
            say(f"The update window cannot be shown ({e}); the update goes on without it.")
            return None
        assert proc.stdout is not None
        stdout = proc.stdout
        first: list = []
        reader = threading.Thread(target=lambda: first.append(stdout.readline()), daemon=True)
        reader.start()
        reader.join(show_s)
        said = first[0].strip() if first else ""
        if said == SHOWN:
            return cls(proc)
        try:
            why = str(json.loads(said).get("error"))
        except (ValueError, AttributeError):
            why = said or f"it did not open within {show_s:.0f} s"
        say(f"The update window cannot be shown ({why}); the update goes on without it.")
        _stop(proc)
        return None

    def _send(self, message: dict) -> bool:
        if self.gone:
            return False
        try:
            assert self.proc.stdin is not None
            self.proc.stdin.write(json.dumps(message) + "\n")
            self.proc.stdin.flush()
            return True
        except (OSError, ValueError):
            self.gone = True
            return False

    def say(self, text: str) -> None:
        self._send({"say": text})

    def ask(self, text: str) -> bool:
        """Ask *text* in the window: True for Try again. Cancel when the window is
        gone."""
        if not self._send({"ask": text}):
            return False
        assert self.proc.stdout is not None
        try:
            answer = self.proc.stdout.readline().strip()
        except (OSError, ValueError):
            answer = ""
        if answer not in (AGAIN, CANCEL):
            self.gone = True
        return answer == AGAIN

    def end(self, text: str, wait_s: float = END_S + 10.0) -> None:
        """Show the outcome *text*; the window closes itself :data:`END_S` seconds later.
        Waits at most *wait_s* seconds for that."""
        self._send({"end": text})
        try:
            if self.proc.stdin is not None:
                self.proc.stdin.close()
        except OSError:
            pass
        try:
            self.proc.wait(timeout=wait_s)
        except subprocess.TimeoutExpired:
            _stop(self.proc)
        self.gone = True


def _stop(proc: subprocess.Popen) -> None:
    try:
        proc.kill()
    except OSError:
        pass
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
