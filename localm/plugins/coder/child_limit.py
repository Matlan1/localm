# SPDX-License-Identifier: AGPL-3.0-or-later
"""One global gate on how many child agents may run at once.

ONE BUDGET, SHARED BY EVERY CHILD-SPAWNING PATH
-----------------------------------------------
Worktree-isolated parallel dispatch (``tools/parallel.py``) and background
sub-agent jobs both draw from this single budget rather than each holding a
private cap. A full 2-child parallel dispatch therefore SATURATES it, and a
background job requested while that runs is rejected.

NO BLOCKING ACQUIRE
-------------------
Only ``try_acquire`` exists and it never blocks: "no free slot" is returned as
None, never queued. A caller that wants to wait does so explicitly in its own
code.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional

# The practical resident-model ceiling on a single consumer GPU. A design
# constraint, not a tunable.
MAX_CONCURRENT_CHILDREN = 2


@dataclass(frozen=True)
class Token:
    """Opaque proof that a slot is held. Pass back to ``release``."""
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
    """The gate itself. A module-level singleton backs the functions below;
    tests construct their own instance so they never race the real one."""

    def __init__(self, max_children: int = MAX_CONCURRENT_CHILDREN) -> None:
        self._max = max_children
        self._lock = threading.Lock()
        self._holders: dict[int, Holder] = {}
        self._next_id = 1

    def try_acquire(self, kind: str, label: str) -> Optional[Token]:
        """Take a slot, or return None immediately if the budget is full.

        Never blocks. The capacity check and the insert happen under one lock
        hold, so two near-simultaneous spawns can never both admit on the same
        free slot.
        """
        with self._lock:
            if len(self._holders) >= self._max:
                return None
            token = Token(id=self._next_id, kind=kind, label=label)
            self._next_id += 1
            self._holders[token.id] = Holder(kind=kind, label=label,
                                             started_at=time.time())
            return token

    def release(self, token: Optional[Token]) -> None:
        """Give a slot back. Idempotent and safe from any thread.

        A double release, a release of an unknown or stale token, and a None
        token are all no-ops: none corrupts the budget and none raises.
        """
        if token is None:
            return
        with self._lock:
            self._holders.pop(token.id, None)

    def holders(self) -> list[Holder]:
        """Snapshot of the running children."""
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
    """Take a child slot, or None if the budget is full. Never blocks."""
    return _GATE.try_acquire(kind, label)


def release(token: Optional[Token]) -> None:
    """Return a child slot. Idempotent; tolerates None and stale tokens."""
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
    """Clear the process-wide gate. TEST ONLY - never call from product code."""
    global _GATE
    _GATE = ChildLimit()
