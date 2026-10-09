# SPDX-License-Identifier: AGPL-3.0-or-later
"""Stop sequences for generated text: validation of a request's ``stop`` value
and the filter that cuts a text stream at the first match.

Pure Python with no imports from the rest of the package, so the request models
and the streaming code can both use it."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

MAX_STOP_SEQUENCES = 16
MAX_STOP_LENGTH = 1024


def normalize_stop(value: Any) -> list[str] | None:
    """A request's ``stop`` as a list of non-empty strings, or ``None`` for none.

    Accepts one string or a list of strings; empty strings are dropped. Raises
    ``ValueError`` for any other type, more than ``MAX_STOP_SEQUENCES``
    sequences, or a sequence longer than ``MAX_STOP_LENGTH`` characters."""
    if value is None:
        return None
    if isinstance(value, str):
        items: list[Any] = [value]
    elif isinstance(value, (list, tuple)):
        items = list(value)
    else:
        raise ValueError("stop must be a string or a list of strings")
    if not all(isinstance(s, str) for s in items):
        raise ValueError("stop must be a string or a list of strings")
    seqs = [s for s in items if s]
    if len(seqs) > MAX_STOP_SEQUENCES:
        raise ValueError(f"stop takes at most {MAX_STOP_SEQUENCES} sequences")
    if any(len(s) > MAX_STOP_LENGTH for s in seqs):
        raise ValueError(f"a stop sequence is at most {MAX_STOP_LENGTH} characters")
    return seqs or None


class StopFilter:
    """Cuts a text stream at the first stop sequence.

    ``feed`` returns the text that is safe to emit: everything before a stop
    sequence, and everything except a tail that could still become one.
    ``hit`` turns true once a stop sequence has been seen; nothing is emitted
    after that. ``flush`` releases the held tail when the stream ends without
    a hit."""

    def __init__(self, stops: Iterable[str]) -> None:
        self._stops = [s for s in stops if s]
        self._buf = ""
        self.hit = False

    def feed(self, text: str) -> str:
        if self.hit:
            return ""
        if not self._stops:
            return text
        self._buf += text
        cut = min((i for i in (self._buf.find(s) for s in self._stops) if i >= 0),
                  default=-1)
        if cut >= 0:
            out, self._buf, self.hit = self._buf[:cut], "", True
            return out
        hold = 0
        for s in self._stops:
            for k in range(min(len(s) - 1, len(self._buf)), 0, -1):
                if self._buf.endswith(s[:k]):
                    hold = max(hold, k)
                    break
        out = self._buf[:len(self._buf) - hold]
        self._buf = self._buf[len(self._buf) - hold:]
        return out

    def flush(self) -> str:
        out, self._buf = ("", "") if self.hit else (self._buf, "")
        return out


def apply_stop(text: str, stops: Iterable[str]) -> tuple[str, bool]:
    """*text* cut at the first stop sequence, and whether one was found."""
    flt = StopFilter(stops)
    out = flt.feed(text)
    if flt.hit:
        return out, True
    return out + flt.flush(), False
