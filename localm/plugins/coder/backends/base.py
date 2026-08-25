# SPDX-License-Identifier: AGPL-3.0-or-later
"""Abstract LLM backend."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Iterator, Optional


class BaseLLMBackend(ABC):
    """Minimal interface that all backends must satisfy."""

    # Subclasses pointing at a local GBNF-capable server set this to True
    supports_grammar: bool = False

    # Whether the SERVER this backend talks to implements the OpenAI-compatible
    # ``tools`` / ``tool_choice`` request fields. The default is True on purpose,
    # and it is a "do not cry wolf" default rather than an optimistic one: the
    # only consumer is the warning that fires when a caller ASKED for native
    # tools and will not get them, so a backend that has never thought about the
    # question must not manufacture that warning. A backend that KNOWS its server
    # cannot honour the fields overrides this to False - see
    # ``HTTPBackend.supports_native_tools``, which does exactly that for localm's
    # own server (measured: ``ChatRequest`` declares no such fields and pydantic
    # drops them).
    supports_native_tools: bool = True

    @abstractmethod
    def chat(self, messages: list[dict], **kwargs) -> str:
        """Send messages, return the complete response string."""

    @abstractmethod
    def chat_stream(self, messages: list[dict],
                    on_reasoning: Optional[Callable[[str], None]] = None,
                    **kwargs) -> Iterator[str]:
        """Send messages, yield VISIBLE text pieces as they arrive."""

    @property
    def model_id(self) -> str:
        return "(unknown)"
