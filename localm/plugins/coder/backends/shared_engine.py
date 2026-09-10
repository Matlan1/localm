# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Coder backend over an inference Engine that somebody else owns.

Wraps an already-constructed Engine (an EngineCache resident in the MCP
server) so the coder Agent drives the model in-process. The engine's lifecycle
stays with its owner: this backend never loads or unloads it. One lock per
engine serialises generations, so parallel sub-agents sharing the backend take
turns on the one model instead of interleaving requests.
"""

from __future__ import annotations

import threading
import weakref
from typing import Callable, Iterator, Optional

from .base import BaseLLMBackend
from .local_engine import _ENGINE_GEN_KWARGS

_LOCKS: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()
_LOCKS_GUARD = threading.Lock()


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

    ``chat``/``chat_stream`` hold the engine's generation lock for the whole
    call. ``last_usage`` is counted with the engine's own tokenizer after each
    call. ``unload`` is deliberately absent: the owner decides residency."""

    native_tools = False
    supports_native_tools = False

    def __init__(self, engine, model_id: str) -> None:
        self._engine = engine
        self._model_id = model_id
        self._lock = engine_lock(engine)
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

    @staticmethod
    def _gen(kwargs: dict) -> dict:
        return {k: v for k, v in kwargs.items()
                if k in _ENGINE_GEN_KWARGS and v is not None}

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
        with self._lock:
            text = "".join(self._engine.chat_stream(messages, **self._gen(kwargs)))
        self._record_usage(messages, text)
        return text

    def chat_stream(self, messages: list[dict],
                    on_reasoning: Optional[Callable[[str], None]] = None,
                    **kwargs) -> Iterator[str]:
        self._last_usage = {}
        parts: list[str] = []
        with self._lock:
            for piece in self._engine.chat_stream(messages, **self._gen(kwargs)):
                parts.append(piece)
                yield piece
        self._record_usage(messages, "".join(parts))

    def context_capacity(self) -> Optional[int]:
        """The engine's resolved context window, or None when unknown."""
        try:
            cap = self._engine.context_capacity()
        except Exception:
            return None
        return cap if isinstance(cap, int) and cap > 0 else None
