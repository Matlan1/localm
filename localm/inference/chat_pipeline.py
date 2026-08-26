# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Chat-pipeline hooks: an ordered inlet / stream / outlet chain that loaded
plugins use to intercept and transform a chat turn server-side, in the kernel
``/v1/chat/completions`` path.

It is the convergence primitive behind cross-ecosystem chat extensions (see
``docs/plugin-interop.md``): Open WebUI "Filter" functions (inlet/stream/outlet)
and oobabooga input/output text modifiers both map onto this one chain.

Scope and layering:
  - The chain runs for EVERY client of ``/v1/chat/completions`` (the GUI, raw
    API callers, and the coder agent pointed at localm). It does NOT run for the
    in-process ``localm run`` REPL, which calls the engine directly.
  - It runs at the kernel HTTP boundary, downstream of ``Engine.chat_stream``'s
    marker scrubbing (``localm/textnorm.py``), so a hook sees clean
    content text, not model-internal control markers. A stream hook therefore
    receives a scrubbed text piece (possibly several characters), not a raw
    model token.
  - It is independent of localm's client-side context injection (RAG, memory
    and web are assembled in the SPA before the request is sent).

Phases (a hook is any callable):
  - ``inlet(messages, ctx) -> messages``    pre-inference; transform the message
    list. May be sync or async. Returning None keeps the prior list (so a hook
    may mutate in place and return nothing).
  - ``stream(token, ctx) -> token``         per streamed text piece. SYNC only -
    this is the per-token hot path; an async stream hook is skipped with a
    warning. Returning None keeps the prior piece.
  - ``outlet(text, messages, ctx) -> text`` post-inference; transform the reply
    text. May be sync or async. Returning None keeps the prior text.

Ordering: hooks run in ascending ``priority`` (lower first); ties keep
registration order. Failure isolation: a hook that raises is logged and skipped,
and the turn continues with the value from before that hook, so one bad plugin
can never break chat. Teardown: the engine removes a plugin's hooks when it is
disabled or uninstalled (``PluginHost.unmount``), so the chain tracks the active
plugin set with no restart.

Streaming note: in a streaming turn the inlet and stream hooks act on what the
client receives, but the outlet runs only after all chunks have been sent, so it
cannot retract them. There it serves record/side-effect purposes (audit,
transcript); use a stream hook to rewrite streamed output live.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Callable, Optional

from localm.debuglog import logger as _log

#: The valid hook phases, in pipeline order.
PHASES = ("inlet", "stream", "outlet")


@dataclass
class ChatHookContext:
    """Per-request context handed to every hook in one chat turn.

    ``state`` is a mutable scratch dict shared across the inlet/stream/outlet of
    a single request, so an inlet can stash data an outlet later reads (mirrors
    how an Open WebUI Filter threads state through the request body).
    ``principal`` / ``scopes`` are reserved for future per-user gating (Open
    WebUI's ``__user__``) and are unset in open mode.
    """
    model_id: str
    stream: bool
    request_id: str
    state: dict = field(default_factory=dict)
    principal: Optional[str] = None
    scopes: tuple = ()


@dataclass(order=True)
class _Entry:
    """One registered hook. Ordered by (priority, seq) so a stable sort keeps
    registration order within a priority."""
    priority: int
    seq: int
    plugin: str = field(compare=False, default="")
    fn: Optional[Callable] = field(compare=False, default=None)


class ChatPipeline:
    """Registry of inlet/stream/outlet hooks plus the runners that apply them.

    One instance lives on ``app.state.chat_pipeline`` for the life of the server.
    Plugins add hooks through ``PluginHost.register_chat_hook`` (which calls
    ``add_hook``) and the engine drops them via ``remove_plugin`` on unload.
    """

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
        """True when at least one hook is registered for *phase* (fast early-out
        so a no-hook turn pays nothing)."""
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
