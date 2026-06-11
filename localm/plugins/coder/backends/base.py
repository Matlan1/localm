"""Abstract LLM backend."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator


class BaseLLMBackend(ABC):
    """
    Minimal interface that all backends must satisfy.

    The agent loop only calls ``chat()`` (for tool-use turns) and
    ``chat_stream()`` (for the final response shown to the user).
    """

    # Subclasses pointing at a local GBNF-capable server set this to True
    supports_grammar: bool = False

    @abstractmethod
    def chat(self, messages: list[dict], **kwargs) -> str:
        """Send messages, return the complete response string."""

    @abstractmethod
    def chat_stream(self, messages: list[dict], **kwargs) -> Iterator[str]:
        """Send messages, yield text pieces as they arrive."""

    @property
    def model_id(self) -> str:
        return "(unknown)"
