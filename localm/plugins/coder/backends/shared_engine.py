# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Coder backend over an inference Engine that somebody else owns.

Wraps an already-constructed Engine (an EngineCache resident in the MCP
server) so the coder Agent drives the model in-process. The engine's lifecycle
stays with its owner: this backend never loads or unloads it. One lock per
engine serialises generations, so parallel sub-agents sharing the backend take
turns on the one model, and every generation pins the engine
(``active_requests``) while it runs so the residency policy never evicts a
model that is mid-generation, whichever thread is driving it.

A thinking model's ``<think>`` scratchpad is split off before the text leaves
this backend: ``chat`` returns the answer alone and latches ``last_reasoning``,
``chat_stream`` routes reasoning pieces to ``on_reasoning`` and yields only the
answer, the same contract the HTTP backend honours through the server's
``reasoning_content`` split.
"""

from __future__ import annotations

import threading
import weakref
from typing import Callable, Iterator, Optional

from localm.inference.residency import try_pin_engine, unpin_engine
from localm.textnorm import ThinkSplitter, split_think

from .base import BaseLLMBackend
from .http import CoderServerError
from .local_engine import _ENGINE_GEN_KWARGS

# Everything Engine.chat_stream accepts, including the lazy-grammar pair the
# coder sends with its tool-call grammar on every turn.
_GEN_KWARGS = frozenset(_ENGINE_GEN_KWARGS | {"grammar_lazy", "grammar_triggers"})

# LOCK ORDER: engine_lock(engine) is the OUTERMOST lock of the process. A
# generation under it takes engine._LOAD_LOCK (auto-reload) and, below that,
# embedder._LOCK; residency._PIN_LOCK is a leaf. Never acquire engine_lock while
# holding any of those three.
_LOCKS: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()
_LOCKS_GUARD = threading.Lock()
# The one lock every engine that cannot be a WeakKeyDictionary key shares, so
# serialisation is never silently switched off for such an engine.
_UNKEYABLE_ENGINE_LOCK = threading.Lock()

ENGINE_GONE_MESSAGE = (
    "The model this session was running on is no longer loaded: it was "
    "unloaded by its owner while the session was between generations. The "
    "session cannot continue on it."
)


def engine_lock(engine) -> threading.Lock:
    """The generation lock for *engine*, one per engine object, created on
    first use. An engine that cannot be weakly referenced or hashed shares one
    module-wide lock with every other such engine."""
    with _LOCKS_GUARD:
        try:
            lock = _LOCKS.get(engine)
        except TypeError:
            return _UNKEYABLE_ENGINE_LOCK
        if lock is None:
            lock = threading.Lock()
            _LOCKS[engine] = lock
        return lock


class SharedEngineBackend(BaseLLMBackend):
    """A BaseLLMBackend over an Engine the caller already holds loaded.

    ``chat``/``chat_stream`` hold the engine's generation lock and pin the
    engine for the whole call. ``last_usage`` and ``last_reasoning`` are
    per-thread, so concurrent sub-agents sharing one backend each read their
    own call's figures; usage is counted with the engine's own tokenizer while
    the generation lock is still held. ``unload`` is deliberately absent: the
    owner decides residency. ``still_resident`` is asked, atomically with the
    pin, before every generation; when it answers False the call is refused
    with ENGINE_GONE_MESSAGE instead of letting the engine reload itself
    outside its owner's gate. ``cancel`` refuses every later generation and
    aborts the one in flight. A sub-agent cannot ask for a different model on
    this backend."""

    native_tools = False
    supports_native_tools = False
    supports_model_override = False

    def __init__(self, engine, model_id: str, *,
                 lock: Optional[threading.Lock] = None,
                 still_resident: Optional[Callable[[], bool]] = None) -> None:
        self._engine = engine
        self._model_id = model_id
        self._lock = lock if lock is not None else engine_lock(engine)
        self._still_resident = still_resident
        self._local = threading.local()
        self._cancel_reason: Optional[str] = None
        self.supports_grammar = bool(getattr(engine, "supports_grammar", False))

    @property
    def engine(self):
        return self._engine

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def last_usage(self) -> dict:
        """Token usage of this thread's most recent call: prompt, completion,
        total. Reasoning tokens are counted in the completion."""
        return dict(getattr(self._local, "usage", {}))

    @property
    def last_reasoning(self) -> str:
        """The reasoning text of this thread's most recent call, with the
        think tags removed; empty when the model produced none."""
        return getattr(self._local, "reasoning", "")

    @property
    def cancelled(self) -> bool:
        return self._cancel_reason is not None

    def cancel(self, reason: str = "cancelled") -> None:
        """Refuse every later generation and abort the one in flight: the
        engine's stream is closed at its next piece, which cancels the native
        generation."""
        self._cancel_reason = reason or "cancelled"

    def _gen(self, kwargs: dict) -> dict:
        """The engine kwargs for one call. A grammar is validated up front, the
        way the server's chat route does, and a refusal is raised as
        CoderServerError carrying the backend's own message so the agent's
        grammar fallback sees the same text it would from the server."""
        gen = {k: v for k, v in kwargs.items()
               if k in _GEN_KWARGS and v is not None}
        grammar = gen.get("grammar")
        if not grammar:
            gen.pop("grammar_lazy", None)
            gen.pop("grammar_triggers", None)
            return gen
        lazy = bool(gen.get("grammar_lazy"))
        if lazy and not gen.get("grammar_triggers"):
            from localm.inference.backends.base import GRAMMAR_LAZY_NO_TRIGGERS_MESSAGE
            raise CoderServerError(GRAMMAR_LAZY_NO_TRIGGERS_MESSAGE)
        from localm.inference.backends.base import (
            GrammarUnsupportedError, InvalidGrammarError)
        try:
            self._engine.validate_grammar(grammar, lazy=lazy)
        except GrammarUnsupportedError as e:
            raise CoderServerError(str(e)) from e
        except InvalidGrammarError as e:
            raise CoderServerError(f"Invalid grammar: {e}") from e
        return gen

    def _check_cancelled(self) -> None:
        if self._cancel_reason is not None:
            raise CoderServerError(
                f"generation refused: this run was cancelled ({self._cancel_reason})")

    def _claim(self) -> None:
        """Pin the engine for one generation, provided the owner still holds it
        resident; the residency check and the pin are one operation."""
        if not try_pin_engine(self._engine, check=self._still_resident):
            raise CoderServerError(ENGINE_GONE_MESSAGE)

    def _record_usage(self, messages: list[dict], text: str) -> None:
        try:
            prompt = int(self._engine.count_messages_tokens(messages))
            completion = int(self._engine.count_tokens(text)) if text else 0
        except Exception as e:
            from localm.debuglog import logger
            logger.debug("shared engine backend: token count unavailable: %s", e)
            self._local.usage = {}
            return
        self._local.usage = {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        }

    def chat(self, messages: list[dict], **kwargs) -> str:
        self._local.usage = {}
        self._local.reasoning = ""
        self._check_cancelled()
        gen = self._gen(kwargs)
        parts: list[str] = []
        with self._lock:
            self._check_cancelled()
            self._claim()
            try:
                for piece in self._engine.chat_stream(messages, **gen):
                    parts.append(piece)
                    if self._cancel_reason is not None:
                        break
            finally:
                try:
                    self._record_usage(messages, "".join(parts))
                finally:
                    unpin_engine(self._engine)
        answer, reasoning = split_think("".join(parts))
        self._local.reasoning = reasoning
        return answer

    def chat_stream(self, messages: list[dict],
                    on_reasoning: Optional[Callable[[str], None]] = None,
                    **kwargs) -> Iterator[str]:
        self._local.usage = {}
        self._local.reasoning = ""
        self._check_cancelled()
        gen = self._gen(kwargs)
        parts: list[str] = []
        reasoning_parts: list[str] = []
        splitter = ThinkSplitter()

        def _split(piece: str) -> str:
            content, reasoning = splitter.feed(piece)
            if reasoning:
                reasoning_parts.append(reasoning)
                if on_reasoning is not None:
                    on_reasoning(reasoning)
            return content

        with self._lock:
            self._check_cancelled()
            self._claim()
            try:
                for piece in self._engine.chat_stream(messages, **gen):
                    parts.append(piece)
                    content = _split(piece)
                    if content:
                        yield content
                    if self._cancel_reason is not None:
                        break
                content, reasoning = splitter.flush()
                if reasoning:
                    reasoning_parts.append(reasoning)
                    if on_reasoning is not None:
                        on_reasoning(reasoning)
                if content:
                    yield content
            finally:
                try:
                    self._record_usage(messages, "".join(parts))
                    self._local.reasoning = "".join(reasoning_parts)
                finally:
                    unpin_engine(self._engine)

    def context_capacity(self) -> Optional[int]:
        """The engine's resolved context window, or None when unknown."""
        try:
            cap = self._engine.context_capacity()
        except Exception:
            return None
        return cap if isinstance(cap, int) and cap > 0 else None
