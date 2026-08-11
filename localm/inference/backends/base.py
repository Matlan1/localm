# SPDX-License-Identifier: AGPL-3.0-or-later
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


class VisionInputError(UnsupportedInputError):
    """Raised when a vision-capable model could not process THIS image or prompt,
    as opposed to not supporting images at all.

    A subclass so every existing ``except UnsupportedInputError`` handler (the
    CLI's vision guidance, the RAG plugin, http_engine) keeps working unchanged.

    Why it must not be a bare ``RuntimeError``: the GGUF worker's dispatch loop
    deliberately lets an escaping exception KILL the process, because the
    contract there is that an escaping exception means a native fault which left
    the model in an unknown state (see ``_runner.py``'s chat_stream branch). Every
    failure ``mtmd.eval_into`` reports is the opposite of that - a native call
    that RETURNED NORMALLY with a status code localm itself checked - so the model
    is unharmed and the request should fail, not the worker. Before this existed,
    one unprocessable image tore down the worker process, evicted the model from
    VRAM and logged a CRITICAL crash with a full traceback.
    """


class ImageDecodeUnavailable(UnsupportedInputError):
    """Raised when an image cannot be decoded because Pillow is not installed,
    as opposed to the image or the model being at fault.

    A sibling of :class:`VisionInputError` rather than the same class, because
    the CAUSE is different and the user-facing fix is different: nothing is
    wrong with the picture, the build simply has no image decoder. It shares the
    :class:`UnsupportedInputError` parent for the recovery semantics documented
    there - the GGUF worker reports it as a per-request error and keeps serving,
    instead of treating an escaping exception as a native fault and tearing the
    process down.

    That distinction is the whole point: a missing pure-Python dependency used
    to surface as "Native inference fault (worker exit 1) ... see the debug log
    for the native stack trace", which is wrong in every part. There was no
    native fault, no native stack trace, and the model was unharmed.
    """


class InvalidGrammarError(ValueError):
    """Raised when a GBNF grammar string cannot be parsed by the native grammar
    engine, so the request can be rejected with a clean 400 up front.

    Why this exists: ``llama_sampler_init_grammar`` returns NULL for a malformed
    grammar; adding that NULL sampler to the chain NULL-derefs at sample time (a
    native access violation). The GGUF backend used to CATCH that fault and latch
    a persistent ``_grammar_unsupported`` flag, silently stripping grammar from
    EVERY later request (valid ones too) until the model reloaded - a single bad
    grammar poisoned the whole feature for all clients. Surfacing the parse failure
    as this typed error (instead of a native crash) lets the request path reject a
    bad grammar cleanly and keeps a per-request user error from disabling the
    feature globally.
    """


class GrammarUnsupportedError(ValueError):
    """Raised when a grammar was requested but this backend cannot constrain
    generation with one at all, so the request is refused instead of answered
    with unconstrained text the caller believes is grammar-conformant.

    Deliberately NOT an :class:`InvalidGrammarError`: the grammar is fine, the
    BACKEND is the limitation. Reporting this as "Invalid grammar" would send the
    caller to fix a grammar that has nothing wrong with it - the same
    wrong-thing-to-fix failure already documented at the worker-fault arm in
    ``routes/chat.py``.

    Deliberately NOT an :class:`UnsupportedInputError` either, even though the
    name fits: ``cli/chat.py``'s ``except UnsupportedInputError`` arm DISCARDS the
    exception message and prints vision-capability guidance in its place (see the
    ordering comment there). Inheriting from it would mean a grammar refusal could
    surface as advice about picking a vision model. A sibling of
    :class:`InvalidGrammarError` under ``ValueError`` keeps the recovery semantics
    honest at every existing handler.

    Why this exists: ``Engine.validate_grammar`` used to probe the backend with
    ``getattr(backend, "validate_grammar", None)`` and no-op when it was absent.
    Only the GGUF backend defined it, so a grammar sent to an HF-backed model was
    never checked AND never applied when the optional ``[grammar]`` extra is
    missing - the worker logs a warning and generates unconstrained. The client
    got a normal 200 full of text that satisfies no grammar, with no signal that
    the constraint had been dropped (AGENTS.md rule 5).
    """


class EmbedBatchTooLargeError(ValueError):
    """Raised when an ``/v1/embeddings`` request against an HF-backed model
    exceeds the configured per-request text-count or character-count cap
    (see ``HFBackend.embed``, ``hf_embed_max_texts``, ``hf_embed_max_chars``),
    so the request can be rejected with a clean, fast 413 up front.

    Why this exists: ``HFBackend.embed()`` loops over texts one at a time
    with no batching (or, for a sentence-transformer model, batches with no
    truncation at all), against a model that is always loaded full
    bf16/fp32. An oversized batch has no native bound of its own - only the
    generous ``hf_embed_timeout_s`` - so without this check it would run for
    however long that allows instead of failing fast.
    """


class ModelLoadCancelled(Exception):
    """Raised by ``load()`` when an in-flight model load was deliberately aborted
    because a newer model selection superseded it (preemptive model switching).

    This is NOT a failure: the user picked a different model while this one was
    still loading, so the native load is stopped mid-flight to avoid wasting time
    finishing a model nobody wants. Callers distinguish it from a real load error
    and report "superseded", not an error.
    """


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

    @property
    def supports_grammar(self) -> bool:
        """True when this backend, in its current state, can actually constrain
        generation to a GBNF grammar.

        Default False, and the default matters: a capability answers DENY when
        nobody declared it. The previous arrangement had no declaration at all -
        callers probed for a ``validate_grammar`` METHOD and read its absence as
        "nothing to validate", which is a different question and answers it in the
        dangerous direction. A backend that cannot constrain generation then
        produced unconstrained output that the caller had no way to distinguish
        from a grammar-conformant answer.

        A new backend inherits the safe answer for free. To offer grammar, set
        this True; overriding :meth:`validate_grammar` on top of that is optional
        and only buys an UP-FRONT rejection of a malformed grammar.
        """
        return False

    def validate_grammar(self, grammar: Optional[str]) -> None:
        """Check *grammar* against this backend before generation starts.

        Base behaviour, which every backend inherits unless it overrides:

        - no grammar requested: nothing to do.
        - grammar requested and :attr:`supports_grammar` is False: raise
          :class:`GrammarUnsupportedError`, so the request is refused with a
          reason rather than silently answered with unconstrained text.
        - grammar requested and the backend declares support but cannot
          pre-parse it: no-op. Deferring to generation time is a real, chosen
          behaviour (see ``GgufBackend.validate_grammar``'s busy-worker branch),
          not a silent drop - the constraint IS applied, it is only the early
          rejection of a malformed string that is skipped.

        This is a concrete method rather than an optional attribute on purpose.
        An absent method cannot say "I cannot do this"; it can only be missing,
        and every caller has to guess what missing meant.
        """
        if not grammar:
            return
        if not self.supports_grammar:
            raise GrammarUnsupportedError(GRAMMAR_UNSUPPORTED_MESSAGE)

    @abstractmethod
    def load(self) -> None:
        """Load the model into memory (possibly onto GPU).

        May raise :class:`ModelLoadCancelled` if a cancel event installed via
        :meth:`set_load_cancel` is set during the load (preemptive switching).
        """

    def set_load_cancel(self, event) -> None:
        """Install a ``threading.Event`` that, when set during ``load()``, aborts
        the load mid-flight (raising :class:`ModelLoadCancelled`).

        Best-effort: the default is a no-op, so a backend that cannot abort a
        partial load simply runs to completion. The GGUF backend honours it via
        llama.cpp's native load progress callback. ``None`` clears it.
        """

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
            Optional GBNF/EBNF grammar string.  When provided, the sampler masks
            tokens that would violate the grammar at the current parse position,
            so output is structurally valid by construction.  The GGUF backend
            uses llama.cpp's native grammar sampler; the HF backend uses xgrammar
            (the optional ``[grammar]`` extra) and falls back to unconstrained
            generation if it is not installed.  Use ``localm.inference.gbnf`` for
            pre-built grammars.
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

    def count_messages_tokens(self, messages: List[dict]) -> int:
        """
        Return the estimated number of tokens in a list of structured messages,
        including chat template formatting.  Subclasses should override this
        with their actual tokenizer and chat template for precise counts.
        """
        text = " ".join(
            m.get("content") if isinstance(m.get("content"), str)
            else " ".join(p.get("text", "") for p in (m.get("content") or [])
                          if p.get("type") == "text")
            for m in messages
        )
        return self.count_tokens(text)

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
