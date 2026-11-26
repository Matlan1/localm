# SPDX-License-Identifier: AGPL-3.0-or-later
"""The applier's side of the small update window (:mod:`.window_tk`, a program of its
own): :class:`Window` starts it and talks to it. Standard library only, so that the
program can load this file for the words both sides use."""

from __future__ import annotations

import json
import os
import queue
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
#: Seconds a question waits for its answer; then it is answered Cancel.
ASK_S = 30 * 60.0
#: What the program prints: the two answers to a question, and that it shows the window.
AGAIN, CANCEL, SHOWN = "again", "cancel", "shown"


# ---------------------------------------------------------- the applier's side

class Window:
    """The window of one update, shown by the program :data:`PROGRAM` in its own
    process. Each method returns at once except :meth:`ask`, which waits for the
    user at most :data:`ASK_S` seconds. *say* is the update's log: what the window
    could not do, and that it is gone, are said there."""

    def __init__(self, proc: subprocess.Popen, say: Callable[[str], None]) -> None:
        self.proc, self._say = proc, say
        self.gone = False
        #: The lines the program printed after ``shown``; None once its output ended.
        self._out: queue.SimpleQueue = queue.SimpleQueue()
        threading.Thread(target=self._read, daemon=True).start()

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
            return cls(proc, say)
        try:
            why = str(json.loads(said).get("error"))
        except (ValueError, AttributeError, RecursionError):
            why = said or f"it did not open within {show_s:.0f} s"
        say(f"The update window cannot be shown ({why}); the update goes on without it.")
        _stop(proc)
        return None

    def _read(self) -> None:
        """Pass the program's answers on to :meth:`ask` and say its notes."""
        assert self.proc.stdout is not None
        try:
            for line in self.proc.stdout:
                line = line.strip()
                if line in (AGAIN, CANCEL):
                    self._out.put(line)
                    continue
                try:
                    note = json.loads(line).get("note")
                except (ValueError, AttributeError, RecursionError):
                    note = None
                if isinstance(note, str):
                    self._say(note)
        except (OSError, ValueError):
            pass
        self._out.put(None)

    def _lost(self, how: str) -> None:
        """Say that the window is gone, *how*, once, with the program's exit status."""
        if not self.gone:
            self.gone = True
            try:
                status = self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                status = None
            self._say(f"The update window {how} (exit status {status}); the update goes on "
                      f"without it.")

    def _send(self, message: dict) -> bool:
        if self.gone:
            return False
        try:
            assert self.proc.stdin is not None
            self.proc.stdin.write(json.dumps(message) + "\n")
            self.proc.stdin.flush()
            return True
        except (OSError, ValueError):
            self._lost("closed")
            return False

    def say(self, text: str) -> None:
        self._send({"say": text})

    def ask(self, text: str, closing: Optional[str] = CANCEL,
            limit_s: Optional[float] = None) -> bool:
        """Ask *text* in the window: True for Try again. Closing the window answers
        *closing* (``again``, ``cancel``, or None: the window stays and asks on). Cancel
        when the window is gone or nobody answers within *limit_s* (:data:`ASK_S`)
        seconds; both are said in the log."""
        while True:   # an answer to an earlier question that came too late
            try:
                if self._out.get_nowait() is None:
                    self._lost("closed")
            except queue.Empty:
                break
        limit_s = ASK_S if limit_s is None else limit_s
        if not self._send({"ask": text, "closing": closing}):
            return False
        try:
            answer = self._out.get(timeout=limit_s)
        except queue.Empty:
            took = f"{limit_s / 60:.0f} minutes" if limit_s >= 120 else f"{limit_s:.0f} s"
            why = f"Nobody answered within {took}: Cancel."
            self._say(f"The update window asked: {text} {why}")
            self._send({"unask": why})
            return False
        if answer is None:
            self._lost("closed while it asked")
            return False
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
            self._say(f"The update window did not close within {wait_s:.0f} s; it is ended.")
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
