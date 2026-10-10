# SPDX-License-Identifier: AGPL-3.0-or-later
"""Admission to one loaded model on the server's event loop.

:class:`InferenceGate` admits up to ``capacity`` generations at once (``async
with gate``), or one load, unload or eviction alone (``async with
gate.exclusive()``). Waiters are served first come, first served: a waiting
exclusive holder is not overtaken by generations that arrive after it.

With capacity 1 it behaves as ``asyncio.Semaphore(1)``: one holder at a time,
shared or exclusive. Used only from the event loop thread.
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
from typing import AsyncIterator

_SHARED = "shared"
_EXCLUSIVE = "exclusive"


class InferenceGate:
    """Shared/exclusive admission with a settable shared capacity (>= 1)."""

    def __init__(self, capacity: int = 1) -> None:
        self._capacity = max(1, int(capacity))
        self._shared = 0
        self._exclusive = False
        # Pending acquisitions in arrival order, as (future, kind). Named like
        # asyncio.Semaphore's, which the queue-depth metric reads.
        self._waiters: collections.deque[tuple[asyncio.Future, str]] = collections.deque()

    @property
    def capacity(self) -> int:
        return self._capacity

    @capacity.setter
    def capacity(self, value: int) -> None:
        """Change how many generations may run at once; raising it admits
        waiting generations at once. Lowering it below the running count lets
        those finish and admits no more until the count is under it."""
        self._capacity = max(1, int(value))
        self._wake()

    @property
    def active(self) -> int:
        """Generations holding the gate right now."""
        return self._shared

    def locked(self) -> bool:
        """True when a generation arriving now would have to wait."""
        return (self._exclusive or self._shared >= self._capacity
                or bool(self._waiters))

    def _can(self, kind: str) -> bool:
        if self._exclusive:
            return False
        if kind == _EXCLUSIVE:
            return self._shared == 0
        return self._shared < self._capacity

    def _take(self, kind: str) -> None:
        if kind == _EXCLUSIVE:
            self._exclusive = True
        else:
            self._shared += 1

    def _give_back(self, kind: str) -> None:
        if kind == _EXCLUSIVE:
            self._exclusive = False
        else:
            self._shared -= 1
        self._wake()

    def _wake(self) -> None:
        """Grant waiters from the front of the queue while the front one fits."""
        while self._waiters:
            fut, kind = self._waiters[0]
            if fut.done():
                self._waiters.popleft()
                continue
            if not self._can(kind):
                return
            self._waiters.popleft()
            self._take(kind)
            fut.set_result(None)

    async def _acquire(self, kind: str) -> None:
        if not self._waiters and self._can(kind):
            self._take(kind)
            return
        fut = asyncio.get_running_loop().create_future()
        entry = (fut, kind)
        self._waiters.append(entry)
        try:
            await fut
        except asyncio.CancelledError:
            if fut.done() and not fut.cancelled():
                self._give_back(kind)
            else:
                with contextlib.suppress(ValueError):
                    self._waiters.remove(entry)
                self._wake()
            raise

    async def acquire(self) -> bool:
        """Take a generation place, waiting for one. Returns True."""
        await self._acquire(_SHARED)
        return True

    def release(self) -> None:
        """Give back a generation place."""
        if self._shared <= 0:
            raise ValueError("InferenceGate released more often than acquired")
        self._give_back(_SHARED)

    async def __aenter__(self) -> None:
        await self._acquire(_SHARED)

    async def __aexit__(self, *exc) -> None:
        self.release()

    @contextlib.asynccontextmanager
    async def exclusive(self) -> AsyncIterator[None]:
        """Hold the gate alone: waits until no generation holds it."""
        await self._acquire(_EXCLUSIVE)
        try:
            yield
        finally:
            self._give_back(_EXCLUSIVE)


def exclusively(gate) -> contextlib.AbstractAsyncContextManager:
    """The async context manager that holds *gate* alone: its ``exclusive()``
    for an :class:`InferenceGate`, the object itself for anything else (an
    ``asyncio.Semaphore(1)`` is already exclusive)."""
    if isinstance(gate, InferenceGate):
        return gate.exclusive()
    return gate


def set_capacity(gate, slots: int) -> None:
    """Set *gate*'s shared capacity to *slots* when it is an
    :class:`InferenceGate`; anything else is left as it is."""
    if isinstance(gate, InferenceGate) and gate.capacity != slots:
        gate.capacity = slots
