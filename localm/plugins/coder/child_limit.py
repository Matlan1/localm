# SPDX-License-Identifier: AGPL-3.0-or-later
"""One global gate on how many child agents may run at once."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional

# The practical resident-model ceiling on a single consumer GPU. Deliberately a
# design constraint, not a tunable to grow: raising it does not buy parallelism the
# hardware can deliver, it buys VRAM thrash.
MAX_CONCURRENT_CHILDREN = 2


@dataclass(frozen=True)
class Token:
    """Opaque proof that a slot is held."""
    id: int
    kind: str
    label: str


@dataclass(frozen=True)
class Holder:
    """A currently-running child, for reporting who is using the budget."""
    kind: str
    label: str
    started_at: float          # epoch seconds, for display

    def age_s(self, now: Optional[float] = None) -> float:
        return (time.time() if now is None else now) - self.started_at


class ChildLimit:
    """The gate itself."""

    def __init__(self, max_children: int = MAX_CONCURRENT_CHILDREN) -> None:
        self._max = max_children
        self._lock = threading.Lock()
        self._holders: dict[int, Holder] = {}
        self._next_id = 1

    def try_acquire(self, kind: str, label: str) -> Optional[Token]:
        """Take a slot, or return None immediately if the budget is full."""
        with self._lock:
            if len(self._holders) >= self._max:
                return None
            token = Token(id=self._next_id, kind=kind, label=label)
            self._next_id += 1
            self._holders[token.id] = Holder(kind=kind, label=label,
                                             started_at=time.time())
            return token

    def release(self, token: Optional[Token]) -> None:
        """Give a slot back."""
        if token is None:
            return
        with self._lock:
            self._holders.pop(token.id, None)

    def holders(self) -> list[Holder]:
        """Snapshot of the running children, so a rejection can NAME them."""
        with self._lock:
            return list(self._holders.values())

    def available(self) -> int:
        with self._lock:
            return max(0, self._max - len(self._holders))

    @property
    def max_children(self) -> int:
        return self._max


# The process-wide budget every child-spawning path draws from.
_GATE = ChildLimit()


def try_acquire(kind: str, label: str) -> Optional[Token]:
    """Take a child slot, or None if the budget is full."""
    return _GATE.try_acquire(kind, label)


def release(token: Optional[Token]) -> None:
    """Return a child slot."""
    _GATE.release(token)


def holders() -> list[Holder]:
    """The children currently holding the budget."""
    return _GATE.holders()


def available() -> int:
    """How many child slots are free right now."""
    return _GATE.available()


def describe_holders() -> str:
    """One-line description of who holds the budget, for a rejection message."""
    current = holders()
    if not current:
        return "no child agents are running"
    return "; ".join(
        f"{h.kind} '{h.label}' (running {h.age_s():.0f}s)" for h in current
    )


def _reset_for_tests() -> None:
    """Clear the process-wide gate."""
    global _GATE
    _GATE = ChildLimit()
