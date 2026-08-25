# SPDX-License-Identifier: AGPL-3.0-or-later
"""One global gate on how many child agents may run at once.

WHY THIS IS SHARED, AND WHY IT IS GLOBAL RATHER THAN PER-MECHANISM
------------------------------------------------------------------
Two different features spawn concurrent children: worktree-isolated parallel
dispatch (``tools/parallel.py``) and, separately, background sub-agent jobs. If
each enforced its own private cap of 2, a model could hold 2 of one plus 2 of the
other and run 4 children at once. Both features would be individually correct and
jointly wrong, and neither feature's tests would catch it.

The ceiling is a property of the BOX, not of the dispatch mechanism: the practical
resident-model limit on a single consumer GPU is about two. A per-mechanism cap
therefore enforces the wrong noun. So there is exactly one budget here and every
child-spawning path draws from it.

The deliberate consequence: a full 2-child parallel dispatch SATURATES the budget,
so a background job requested while it runs is rejected. That is honest - the
hardware genuinely cannot do more - and rejecting loudly beats admitting four
children and thrashing VRAM.

WHY THERE IS NO BLOCKING ACQUIRE
--------------------------------
Only ``try_acquire`` exists, and it never blocks. A blocking
``Semaphore(2).acquire()`` would turn "no free slot" into an invisible wait, which
is exactly the silent queue a caller must not have: a background-spawn tool that
blocks has defeated its own purpose, and a caller that wanted to report "rejected,
these two are running" instead hangs. Not exposing a blocking acquire at all means
neither caller can reintroduce that failure by accident. A caller that genuinely
wants to wait must do so explicitly, in its own code, where the waiting is visible.
"""

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
        hold: doing them separately would let two near-simultaneous spawns both
        observe the same single free slot and both admit, which is the whole
        failure this gate exists to prevent.
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

        Idempotent on purpose: the caller releases in a ``finally``, and an error
        path may well have released already. A double release must not corrupt the
        budget (a decrement-based counter would drift negative and silently widen
        the cap), and a release of an unknown/stale token must not raise inside
        somebody's cleanup handler. Tolerating None keeps caller cleanup free of
        null-checks when the acquire itself failed.
        """
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
