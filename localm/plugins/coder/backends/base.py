# SPDX-License-Identifier: AGPL-3.0-or-later
"""Abstract LLM backend."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Iterator, Optional


class ModelSwitchUnsupported(RuntimeError):
    """The backend has no set_model, so it cannot be repointed at another
    model in place."""


class BaseLLMBackend(ABC):
    """
    Minimal interface that all backends must satisfy.

    The agent loop only calls ``chat()`` (for tool-use turns) and
    ``chat_stream()`` (for the final response shown to the user).
    """

    # Subclasses pointing at a local GBNF-capable server set this to True
    supports_grammar: bool = False

    # Whether this backend sends the OpenAI-compatible ``tools`` / ``tool_choice``
    # request fields to its server. Defaults to True: the only consumer is the
    # warning that fires when a caller ASKED for native tools and will not get
    # them, so a backend that has never declared an answer does not manufacture
    # that warning. A backend that does not send them overrides this to False -
    # see ``HTTPBackend.supports_native_tools``, which does that for localm's own
    # server.
    supports_native_tools: bool = True

    @abstractmethod
    def chat(self, messages: list[dict], **kwargs) -> str:
        """Send messages, return the complete response string."""

    @abstractmethod
    def chat_stream(self, messages: list[dict],
                    on_reasoning: Optional[Callable[[str], None]] = None,
                    on_status: Optional[Callable[[str, Optional[str]], None]] = None,
                    **kwargs) -> Iterator[str]:
        """Send messages, yield VISIBLE text pieces as they arrive.

        ``on_status`` is an OPTIONAL side channel called with
        ``(text, code)`` for each status the model or server reports before
        or between pieces (loading the model, processing the prompt, waiting
        for another request): ``text`` is the English status and ``code`` its
        stable id from ``localm.inference.protocol.STATUS_CODE_BY_TEXT``, or
        None. A backend without statuses ignores it.

        ``on_reasoning`` is an OPTIONAL side channel: a backend that can split a
        thinking model's reasoning from its answer (e.g. the OpenAI-compatible
        ``reasoning_content`` delta field) calls it with each reasoning piece as
        it streams, instead of mixing that text into the yielded content. A
        backend without a reasoning channel simply ignores the callback - the
        agent loop never requires it to fire. Never yield reasoning inline (e.g.
        wrapped in ``<think>`` tags): callers of this Iterator have no splitter
        downstream, so an inlined tag would leak into the visible answer, the
        audit log, and conversation history verbatim."""

    @property
    def model_id(self) -> str:
        return "(unknown)"
