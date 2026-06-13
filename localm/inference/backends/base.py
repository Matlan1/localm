"""Abstract backend interface shared by HF and GGUF backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator, List, Optional


class UnsupportedInputError(ValueError):
    """Raised when a backend is handed input it cannot process - for example an
    image attached to a text-only model - instead of silently dropping it.

    Silently discarding an image is the worst failure mode: the model answers
    confidently about a picture it never received. Backends raise this so the
    caller can report the problem instead.
    """


# Shown to the user when an image is attached to a model that cannot see images.
# Accurate whether the active model is GGUF (always text-only) or a text-only
# HuggingFace checkpoint.
IMAGE_UNSUPPORTED_MESSAGE = (
    "This model cannot accept image input, so the attached image would be "
    "ignored. To chat about images, load a vision-capable HuggingFace-format "
    "model (for example a Gemma 3 vision or Qwen2-VL checkpoint). The built-in "
    "GGUF backend is text-only: the --mmproj / image path is not implemented."
)


def messages_contain_image(messages: List[dict]) -> bool:
    """True if any message carries an ``image_url`` content part.

    Operates on the plain-dict OpenAI message shape used between the server and
    the backends, so it is safe to call before a model is loaded.
    """
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    return True
    return False


class BaseBackend(ABC):
    """Loaded model that can stream chat completions."""

    # Whether this backend class can ever handle images (warrants loading the
    # model to find out for sure). GGUF is text-only; HF may be multimodal.
    can_be_multimodal: bool = False

    @property
    def supports_images(self) -> bool:
        """True when this backend, in its current state, can actually see
        images. Default False; multimodal backends override this."""
        return False

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
