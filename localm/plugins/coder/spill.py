# SPDX-License-Identifier: AGPL-3.0-or-later
"""In-memory store for tool output too large to show the model in full.

A tool result over the size cap keeps its head and tail in the conversation;
the whole text is kept here under a short id ("o1", "o2", ...) that the
``read_output`` tool reads back by line range or pattern. Nothing is written to
disk, so this holds in every session mode, privacy mode included.

The store is bound to the running tool call through a context variable that the
agent sets around each tool function, so the truncation helper every tool uses
can file the full text without each tool knowing about the store.
"""

from __future__ import annotations

import contextlib
import contextvars
import threading
from dataclasses import dataclass

MAX_ENTRIES = 24
MAX_TOTAL_CHARS = 16_000_000


@dataclass
class SpilledOutput:
    id: str
    tool: str
    text: str
    untrusted: bool

    @property
    def line_count(self) -> int:
        return self.text.count("\n") + (0 if self.text.endswith("\n") or not self.text else 1)


class SpillStore:
    """Bounded, thread-safe map of spill id to full output. The oldest entries
    are dropped first once MAX_ENTRIES or MAX_TOTAL_CHARS is exceeded."""

    def __init__(self, max_entries: int = MAX_ENTRIES,
                 max_total_chars: int = MAX_TOTAL_CHARS) -> None:
        self._lock = threading.Lock()
        self._entries: dict[str, SpilledOutput] = {}
        self._seq = 0
        self._max_entries = max_entries
        self._max_total = max_total_chars

    def add(self, text: str, tool: str, untrusted: bool) -> str:
        with self._lock:
            self._seq += 1
            sid = f"o{self._seq}"
            self._entries[sid] = SpilledOutput(sid, tool, text, untrusted)
            self._evict_locked()
            return sid

    def get(self, sid: str) -> SpilledOutput | None:
        with self._lock:
            return self._entries.get(sid)

    def ids(self) -> list[str]:
        with self._lock:
            return list(self._entries)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def _evict_locked(self) -> None:
        total = sum(len(e.text) for e in self._entries.values())
        while self._entries and (len(self._entries) > self._max_entries
                                 or total > self._max_total):
            oldest = next(iter(self._entries))
            if len(self._entries) == 1:
                break
            total -= len(self._entries.pop(oldest).text)


_active: contextvars.ContextVar[tuple[SpillStore, str, bool] | None] = (
    contextvars.ContextVar("coder_spill_target", default=None))


@contextlib.contextmanager
def spilling_to(store: SpillStore | None, tool: str, untrusted: bool):
    """Within the block, oversized output from tool *tool* is filed in *store*."""
    token = _active.set((store, tool, untrusted) if store is not None else None)
    try:
        yield
    finally:
        _active.reset(token)


def active_store() -> SpillStore | None:
    target = _active.get()
    return target[0] if target else None


def spill(text: str) -> str | None:
    """File *text* in the active store and return its id, or None when no tool
    call is spilling (a direct call from code or a test)."""
    target = _active.get()
    if target is None:
        return None
    store, tool, untrusted = target
    return store.add(text, tool, untrusted)
