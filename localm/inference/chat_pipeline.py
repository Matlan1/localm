# SPDX-License-Identifier: AGPL-3.0-or-later
"""Chat-pipeline hooks: an ordered inlet / stream / outlet chain that loaded plugins use to intercept and transform a chat turn server-side, in the kernel ``/v1/chat/completions`` path."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Callable, Optional

from localm.debuglog import logger as _log

#: The valid hook phases, in pipeline order.
PHASES = ("inlet", "stream", "outlet")


@dataclass
class ChatHookContext:
    """Per-request context handed to every hook in one chat turn."""
    model_id: str
    stream: bool
    request_id: str
    state: dict = field(default_factory=dict)
    principal: Optional[str] = None
    scopes: tuple = ()


@dataclass(order=True)
class _Entry:
    """One registered hook."""
    priority: int
    seq: int
    plugin: str = field(compare=False, default="")
    fn: Optional[Callable] = field(compare=False, default=None)


class ChatPipeline:
    """Registry of inlet/stream/outlet hooks plus the runners that apply them."""

    def __init__(self) -> None:
        self._hooks: dict[str, list[_Entry]] = {p: [] for p in PHASES}
        self._seq = 0

    def add_hook(self, phase: str, fn: Callable, *,
                 priority: int = 0, plugin: str = "") -> None:
        if phase not in self._hooks:
            raise ValueError(
                f"unknown chat-hook phase {phase!r}; expected one of {PHASES}")
        if not callable(fn):
            raise TypeError("chat hook must be callable")
        self._seq += 1
        bucket = self._hooks[phase]
        bucket.append(_Entry(priority=priority, seq=self._seq,
                             plugin=plugin, fn=fn))
        bucket.sort()                       # stable by (priority, seq)

    def remove_plugin(self, name: str) -> None:
        """Drop every hook a plugin registered, across all phases."""
        for bucket in self._hooks.values():
            bucket[:] = [e for e in bucket if e.plugin != name]

    def has(self, phase: str) -> bool:
        """True when at least one hook is registered for *phase* (fast early-out so a no-hook turn pays nothing)."""
        return bool(self._hooks.get(phase))

    async def run_inlet(self, messages: list, ctx: ChatHookContext) -> list:
        for entry in list(self._hooks["inlet"]):
            try:
                result = entry.fn(messages, ctx)
                if inspect.isawaitable(result):
                    result = await result
                if result is not None:
                    messages = result
            except Exception:
                _log.exception("chat inlet hook from %r failed; skipping",
                               entry.plugin)
        return messages

    def run_stream(self, token: str, ctx: ChatHookContext) -> str:
        for entry in list(self._hooks["stream"]):
            try:
                result = entry.fn(token, ctx)
                if inspect.isawaitable(result):
                    # Stream hooks run on the per-token hot path and must be
                    # synchronous; close the orphaned coroutine and skip it.
                    result.close()
                    _log.warning(
                        "chat stream hook from %r is async; stream hooks must "
                        "be synchronous - skipping", entry.plugin)
                    continue
                if result is not None:
                    token = result
            except Exception:
                _log.exception("chat stream hook from %r failed; skipping",
                               entry.plugin)
        return token

    async def run_outlet(self, text: str, messages: list,
                         ctx: ChatHookContext) -> str:
        for entry in list(self._hooks["outlet"]):
            try:
                result = entry.fn(text, messages, ctx)
                if inspect.isawaitable(result):
                    result = await result
                if result is not None:
                    text = result
            except Exception:
                _log.exception("chat outlet hook from %r failed; skipping",
                               entry.plugin)
        return text
