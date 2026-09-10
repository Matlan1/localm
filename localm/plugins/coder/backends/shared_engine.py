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
"""

from __future__ import annotations

import threading
import weakref
from typing import Callable, Iterator, Optional

from localm.inference.residency import pin_engine, unpin_engine

from .base import BaseLLMBackend
from .http import CoderServerError
from .local_engine import _ENGINE_GEN_KWARGS

# Everything Engine.chat_stream accepts, including the lazy-grammar pair the
# coder sends with its tool-call grammar on every turn.
_GEN_KWARGS = frozenset(_ENGINE_GEN_KWARGS | {"grammar_lazy", "grammar_triggers"})

_LOCKS: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()
_LOCKS_GUARD = threading.Lock()

ENGINE_GONE_MESSAGE = (
    "The model this session was running on is no longer loaded: it was "
    "unloaded by its owner while the session was between generations. The "
    "session cannot continue on it."
)


def engine_lock(engine) -> threading.Lock:
    """The generation lock for *engine*, one per engine object, created on
    first use. An engine that cannot be weakly referenced gets a fresh lock
    per call."""
    with _LOCKS_GUARD:
        try:
            lock = _LOCKS.get(engine)
        except TypeError:
            return threading.Lock()
        if lock is None:
            lock = threading.Lock()
            _LOCKS[engine] = lock
        return lock


class SharedEngineBackend(BaseLLMBackend):
    """A BaseLLMBackend over an Engine the caller already holds loaded.

    ``chat``/``chat_stream`` hold the engine's generation lock and pin the
    engine for the whole call. ``last_usage`` is counted with the engine's own
    tokenizer after each call. ``unload`` is deliberately absent: the owner
    decides residency. ``still_resident`` is asked before every generation;
    when it answers False the call is refused with ENGINE_GONE_MESSAGE
    instead of letting the engine reload itself outside its owner's gate.
    A sub-agent cannot ask for a different model on this backend."""

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
        self._last_usage: dict = {}
        self.supports_grammar = bool(getattr(engine, "supports_grammar", False))

    @property
    def engine(self):
        return self._engine

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def last_usage(self) -> dict:
        """Token usage of the most recent call: prompt, completion, total."""
        return dict(self._last_usage)

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

    def _check_resident(self) -> None:
        if self._still_resident is not None and not self._still_resident():
            raise CoderServerError(ENGINE_GONE_MESSAGE)

    def _record_usage(self, messages: list[dict], text: str) -> None:
        try:
            prompt = int(self._engine.count_messages_tokens(messages))
            completion = int(self._engine.count_tokens(text)) if text else 0
        except Exception as e:
            from localm.debuglog import logger
            logger.debug("shared engine backend: token count unavailable: %s", e)
            self._last_usage = {}
            return
        self._last_usage = {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        }

    def chat(self, messages: list[dict], **kwargs) -> str:
        self._last_usage = {}
        gen = self._gen(kwargs)
        with self._lock:
            self._check_resident()
            pin_engine(self._engine)
            try:
                text = "".join(self._engine.chat_stream(messages, **gen))
            finally:
                unpin_engine(self._engine)
        self._record_usage(messages, text)
        return text

    def chat_stream(self, messages: list[dict],
                    on_reasoning: Optional[Callable[[str], None]] = None,
                    **kwargs) -> Iterator[str]:
        self._last_usage = {}
        gen = self._gen(kwargs)
        parts: list[str] = []
        with self._lock:
            self._check_resident()
            pin_engine(self._engine)
            try:
                for piece in self._engine.chat_stream(messages, **gen):
                    parts.append(piece)
                    yield piece
            finally:
                unpin_engine(self._engine)
        self._record_usage(messages, "".join(parts))

    def context_capacity(self) -> Optional[int]:
        """The engine's resolved context window, or None when unknown."""
        try:
            cap = self._engine.context_capacity()
        except Exception:
            return None
        return cap if isinstance(cap, int) and cap > 0 else None
