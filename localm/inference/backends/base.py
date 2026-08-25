# SPDX-License-Identifier: AGPL-3.0-or-later
"""Abstract backend interface shared by HF and GGUF backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator, List, Optional


class UnsupportedInputError(ValueError):
    """Raised when a backend is handed input it cannot process - for example an image attached to a text-only model - instead of silently dropping it."""


class VisionInputError(UnsupportedInputError):
    """Raised when a vision-capable model could not process THIS image or prompt, as opposed to not supporting images at all."""


class ImageDecodeUnavailable(UnsupportedInputError):
    """Raised when an image cannot be decoded because Pillow is not installed, as opposed to the image or the model being at fault."""


class InvalidGrammarError(ValueError):
    """Raised when a GBNF grammar string cannot be parsed by the native grammar engine, so the request can be rejected with a clean 400 up front."""


class TriggerValidatorUnavailableError(InvalidGrammarError):
    """Raised when a lazy-grammar trigger pattern could not be CHECKED - the validator's probe pool was saturated, or its daemon could not be spawned or reached - as opposed to the pattern having been checked and found unsafe."""


class GrammarUnsupportedError(ValueError):
    """Raised when a grammar was requested but this backend cannot constrain generation with one at all, so the request is refused instead of answered with unconstrained text the caller believes is grammar-conformant."""


class EmbedBatchTooLargeError(ValueError):
    """Raised when an ``/v1/embeddings`` request against an HF-backed model exceeds the configured per-request text-count or character-count cap (see ``HFBackend.embed``, ``hf_embed_max_texts``, ``hf_embed_max_chars``), so the request can be rejected with a clean, fast 413 up front."""


class ContextCapacityExceededError(ValueError):
    """Raised when a prompt or conversation exceeds the model's maximum context capacity or leaves insufficient room for generation under the configured ceiling."""


class ModelLoadCancelled(Exception):
    """Raised by ``load()`` when an in-flight model load was deliberately aborted because a newer model selection superseded it (preemptive model switching)."""


# Shown to the user when an image is attached to a model that cannot see images.
# Accurate whether the active model is GGUF (always text-only) or a text-only
# HuggingFace checkpoint.
IMAGE_UNSUPPORTED_MESSAGE = (
    "This model cannot accept image input, so the attached image would be "
    "ignored. To chat about images, load a vision-capable HuggingFace-format "
    "model, or ensure you have a vision projector (.mmproj) loaded alongside "
    "your GGUF model."
)


# Shown to the user when a grammar is requested of a backend that cannot apply
# one. Names both routes out, because which one applies depends on the install:
# a GGUF model always has native grammar support, while an HF model needs the
# optional extra that ships xgrammar.
GRAMMAR_UNSUPPORTED_MESSAGE = (
    "This model cannot constrain generation to a grammar, so the requested "
    "grammar would be ignored and the reply would not match it. Use a "
    "GGUF-format model (grammar support is built in), or install the grammar "
    "extra for HuggingFace-format models with: pip install 'localm[grammar]'"
)


# Shown when a LAZY grammar is requested of a backend that can constrain generation
# but cannot do it lazily. Deliberately distinct from GRAMMAR_UNSUPPORTED_MESSAGE
# above: that one says "this model cannot constrain generation at all" and sends the
# reader to install the grammar extra, which is the wrong advice here - the extra may
# already be installed and plain (non-lazy) grammar may work perfectly. Naming the
# lazy mode specifically is what lets the caller pick the recovery that actually
# applies, which is the same wrong-thing-to-fix reasoning that kept
# GrammarUnsupportedError separate from InvalidGrammarError.
GRAMMAR_LAZY_UNSUPPORTED_MESSAGE = (
    "This model cannot apply a LAZY grammar (one that leaves generation "
    "unconstrained until a trigger pattern matches, then enforces the grammar "
    "from there), so the requested grammar would be ignored and the reply would "
    "not match it. Use a GGUF-format model, whose native sampler implements lazy "
    "grammars, or resend without grammar_lazy to constrain the whole reply."
)


# Shown when a LAZY grammar is requested with NO trigger patterns. Deliberately a
# THIRD message rather than a reuse of GRAMMAR_LAZY_UNSUPPORTED_MESSAGE above,
# because the two need OPPOSITE recoveries and one of them is not the caller's to
# make: this one is fixed by supplying grammar_triggers, that one only by dropping
# grammar_lazy or changing model. Reusing the other string would also be actively
# harmful, not merely imprecise - ``coder/agent/context.py`` latches
# ``_lazy_grammar_confirmed_unsupported`` on a SUBSTRING match against it, so a
# caller that simply forgot its triggers would permanently disable trigger-gated
# tool-call sampling for the rest of that session, blaming the backend for its own
# omission. Kept mutually non-containing with both other messages (asserted in
# tests/test_lazy_grammar.py) so no substring overlap can re-open that route.
GRAMMAR_LAZY_NO_TRIGGERS_MESSAGE = (
    "A lazy grammar was requested with no trigger patterns, so nothing could ever "
    "switch the grammar on and the reply would not match it. Send grammar_triggers "
    "alongside grammar_lazy, or resend without grammar_lazy to constrain the whole "
    "reply."
)


def messages_contain_image(messages: List[dict]) -> bool:
    """True if any message carries an ``image_url`` content part."""
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
        """True when this backend, in its current state, can actually see images."""
        return False

    @property
    def supports_grammar(self) -> bool:
        """True when this backend, in its current state, can actually constrain generation to a GBNF grammar."""
        return False

    @property
    def supports_mtp(self) -> bool:
        """True when this backend has active Multi-Token Prediction (MTP) heads loaded for speculative drafting."""
        return False

    def validate_grammar(self, grammar: Optional[str], *, lazy: bool = False) -> None:
        """Check *grammar* against this backend before generation starts."""
        if not grammar:
            return
        if not self.supports_grammar:
            raise GrammarUnsupportedError(GRAMMAR_UNSUPPORTED_MESSAGE)

    @abstractmethod
    def load(self) -> None:
        """Load the model into memory (possibly onto GPU)."""

    def set_load_cancel(self, event) -> None:
        """Install a ``threading.Event`` that, when set during ``load()``, aborts the load mid-flight (raising :class:`ModelLoadCancelled`)."""

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
        """Yield text tokens one at a time."""

    @property
    @abstractmethod
    def loaded(self) -> bool:
        """True once load() has completed successfully."""

    def count_tokens(self, text: str) -> int:
        """Return the number of tokens in *text* as tokenised by this model."""
        return max(1, len(text) // 4)

    def count_messages_tokens(self, messages: List[dict]) -> int:
        """Return the estimated number of tokens in a list of structured messages, including chat template formatting."""
        text = " ".join(
            m.get("content") if isinstance(m.get("content"), str)
            else " ".join(p.get("text", "") for p in (m.get("content") or [])
                          if p.get("type") == "text")
            for m in messages
        )
        return self.count_tokens(text)

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Return embedding vectors for a list of texts."""
        raise NotImplementedError(
            "This backend does not support embedding.  "
            "Load a dedicated embedding model (e.g. nomic-embed-text)."
        )
