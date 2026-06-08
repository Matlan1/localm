"""Abstract backend interface shared by HF and GGUF backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator, List


class BaseBackend(ABC):
    """Loaded model that can stream chat completions."""

    @abstractmethod
    def load(self) -> None:
        """Load the model into memory (possibly onto GPU)."""

    @abstractmethod
    def unload(self) -> None:
        """Free GPU/CPU memory."""

    @abstractmethod
    def chat_stream(
        self,
        messages: List[dict],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 40,
        repeat_penalty: float = 1.1,
    ) -> Iterator[str]:
        """Yield text tokens one at a time."""

    @property
    @abstractmethod
    def loaded(self) -> bool:
        """True once load() has completed successfully."""
