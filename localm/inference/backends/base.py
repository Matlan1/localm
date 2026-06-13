"""Abstract backend interface shared by HF and GGUF backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator, List, Optional


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
        grammar: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> Iterator[str]:
        """
        Yield text tokens one at a time.

        Parameters
        ----------
        grammar:
            Optional GBNF grammar string.  When provided (GGUF backend only),
            the sampler masks tokens that would violate the grammar at the
            current parse position.  Use ``localm.inference.gbnf`` for
            pre-built grammars.  Ignored by backends that do not support it.
        seed:
            RNG seed for reproducible generation.  GGUF: passed to the sampler.
            HF: sets ``torch.manual_seed`` before generating.
        """

    @property
    @abstractmethod
    def loaded(self) -> bool:
        """True once load() has completed successfully."""

    def count_tokens(self, text: str) -> int:
        """
        Return the number of tokens in *text* as tokenised by this model.

        The base implementation uses a chars-÷-4 heuristic when the backend
        has not overridden this method (e.g. subprocess fallback or when the
        model is not yet loaded).  Concrete backends should override this with
        their actual tokenizer for precise counts.
        """
        return max(1, len(text) // 4)

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Return embedding vectors for a list of texts.

        Raises ``NotImplementedError`` by default - not all models support
        embedding.  For quality embeddings, use a dedicated embedding model
        (nomic-embed, bge-*, e5-*) rather than a chat/instruct model.
        """
        raise NotImplementedError(
            "This backend does not support embedding.  "
            "Load a dedicated embedding model (e.g. nomic-embed-text)."
        )
