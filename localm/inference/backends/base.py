# SPDX-License-Identifier: AGPL-3.0-or-later
"""Abstract backend interface shared by HF and GGUF backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator, List, Optional


class UnsupportedInputError(ValueError):
    """Raised when a backend is handed input it cannot process - for example an
    image attached to a text-only model - instead of silently dropping it.

    A backend raises this so the caller can report the problem; a silently
    discarded image leaves the model answering about a picture it never
    received.
    """


class VisionInputError(UnsupportedInputError):
    """Raised when a vision-capable model could not process THIS image or prompt,
    as opposed to not supporting images at all.

    A subclass, so every existing ``except UnsupportedInputError`` handler (the
    CLI's vision guidance, the RAG plugin, http_engine) catches it too.

    It must not be a bare ``RuntimeError``: the GGUF worker's dispatch loop lets
    an escaping exception KILL the process, because an escaping exception there
    means a native fault that left the model in an unknown state (see
    ``_runner.py``'s chat_stream branch). Every failure ``mtmd.eval_into``
    reports is the opposite of that - a native call that RETURNED NORMALLY with a
    status code localm itself checked - so the model is unharmed and the request
    should fail, not the worker.
    """


class ImageDecodeUnavailable(UnsupportedInputError):
    """Raised when an image cannot be decoded because Pillow is not installed,
    as opposed to the image or the model being at fault.

    A sibling of :class:`VisionInputError` rather than the same class: the CAUSE
    is different and so is the user-facing fix - nothing is wrong with the
    picture, the build simply has no image decoder. It shares the
    :class:`UnsupportedInputError` parent for the recovery semantics documented
    there - the GGUF worker reports it as a per-request error and keeps serving,
    instead of treating an escaping exception as a native fault and tearing the
    process down.
    """


class InvalidGrammarError(ValueError):
    """Raised when a GBNF grammar string cannot be parsed by the native grammar
    engine, so the request can be rejected with a clean 400 up front.

    ``llama_sampler_init_grammar`` returns NULL for a malformed grammar, and
    adding that NULL sampler to the chain NULL-derefs at sample time (a native
    access violation). Surfacing the parse failure as this typed error keeps a
    malformed grammar from reaching the sampler at all, and keeps a per-request
    user error from disabling the feature for every other client.
    """


class TriggerValidatorUnavailableError(InvalidGrammarError):
    """Raised when a lazy-grammar trigger pattern could not be CHECKED - the
    validator's probe pool was saturated, or its daemon could not be spawned or
    reached - as opposed to the pattern having been checked and found unsafe.

    ``gbnf.validate_trigger_patterns`` caches a verdict per pattern for the whole
    process lifetime, so this outcome is NOT cached: recording "I could not ask"
    against a pattern would reject that pattern until restart for a reason that
    had nothing to do with it.

    The request is still refused either way - a pattern that was not PROVEN safe
    must never reach the native sampler (see gbnf.py's module comment for what
    an unvalidated pattern can do there). What changes is the status: this maps
    to 503, because a saturated validator is a condition on THIS side of the
    wire that a retry can clear, while a genuine ``InvalidGrammarError`` is a
    400 the caller must fix.

    A SUBCLASS of ``InvalidGrammarError``, so every existing
    ``except InvalidGrammarError`` arm keeps catching it and no call site can
    silently start letting it escape as an opaque 500. Sites that want the
    sharper answer catch this first, and ``_BACKEND_ERROR_STATUS`` lists it
    BEFORE its parent - that table's order is load-bearing and has its own test.
    """


class GrammarUnsupportedError(ValueError):
    """Raised when a grammar was requested but this backend cannot constrain
    generation with one at all, so the request is refused instead of answered
    with unconstrained text the caller believes is grammar-conformant.

    NOT an :class:`InvalidGrammarError`: the grammar is fine, the BACKEND is the
    limitation, and reporting this as "Invalid grammar" would send the caller to
    fix a grammar that has nothing wrong with it.

    NOT an :class:`UnsupportedInputError` either, even though the name fits:
    ``cli/chat.py``'s ``except UnsupportedInputError`` arm DISCARDS the exception
    message and prints vision-capability guidance in its place (see the ordering
    comment there), so inheriting from it would make a grammar refusal surface as
    advice about picking a vision model. It is a sibling of
    :class:`InvalidGrammarError` under ``ValueError``, which keeps the recovery
    semantics honest at every existing handler.
    """


class EmbedBatchTooLargeError(ValueError):
    """Raised when an ``/v1/embeddings`` request against an HF-backed model
    exceeds the configured per-request text-count or character-count cap
    (see ``HFBackend.embed``, ``hf_embed_max_texts``, ``hf_embed_max_chars``),
    so the request can be rejected with a clean, fast 413 up front.

    ``HFBackend.embed()`` loops over texts one at a time with no batching (or,
    for a sentence-transformer model, batches with no truncation at all),
    against a model that is always loaded full bf16/fp32. An oversized batch has
    no native bound of its own - only the generous ``hf_embed_timeout_s``.
    """


class ContextCapacityExceededError(ValueError):
    """Raised when a prompt or conversation exceeds the model's maximum context
    capacity or leaves insufficient room for generation under the configured ceiling.

    A ValueError subclass, carried across IPC as a typed error: the GGUF and HF
    worker dispatch loops treat an uncaught exception as an unrecoverable native
    fault that left the model in an unknown state, which kills the worker process
    and evicts the loaded model from memory/VRAM. An oversized prompt is checked
    in pure Python before native generation begins, so the loaded model is
    unharmed and the worker keeps serving other requests without a reload.
    """


class ModelLoadCancelled(Exception):
    """Raised by ``load()`` when an in-flight model load was aborted because a
    newer model selection superseded it (preemptive model switching).

    This is NOT a failure: the user picked a different model while this one was
    still loading, so the native load is stopped mid-flight. Callers distinguish
    it from a real load error and report "superseded", not an error.
    """


# Shown when an image is attached to a model that cannot see images.
IMAGE_UNSUPPORTED_MESSAGE = (
    "This model cannot accept image input, so the attached image would be "
    "ignored. To chat about images, load a vision-capable HuggingFace-format "
    "model, or ensure you have a vision projector (.mmproj) loaded alongside "
    "your GGUF model."
)


# Shown when a grammar is requested of a backend that cannot apply one. Names
# both routes out: a GGUF model has native grammar support, an HF model needs the
# optional extra that ships xgrammar.
GRAMMAR_UNSUPPORTED_MESSAGE = (
    "This model cannot constrain generation to a grammar, so the requested "
    "grammar would be ignored and the reply would not match it. Use a "
    "GGUF-format model (grammar support is built in), or install the grammar "
    "extra for HuggingFace-format models with: pip install 'localm[grammar]'"
)


# Shown when a LAZY grammar is requested of a backend that can constrain
# generation but cannot do it lazily. A distinct string from
# GRAMMAR_UNSUPPORTED_MESSAGE above, which names a different recovery.
GRAMMAR_LAZY_UNSUPPORTED_MESSAGE = (
    "This model cannot apply a LAZY grammar (one that leaves generation "
    "unconstrained until a trigger pattern matches, then enforces the grammar "
    "from there), so the requested grammar would be ignored and the reply would "
    "not match it. Use a GGUF-format model, whose native sampler implements lazy "
    "grammars, or resend without grammar_lazy to constrain the whole reply."
)


# Shown when a LAZY grammar is requested with NO trigger patterns. A third,
# distinct string: coder/agent/context.py latches
# _lazy_grammar_confirmed_unsupported on a SUBSTRING match against
# GRAMMAR_LAZY_UNSUPPORTED_MESSAGE, so all three messages must stay mutually
# non-containing.
GRAMMAR_LAZY_NO_TRIGGERS_MESSAGE = (
    "A lazy grammar was requested with no trigger patterns, so nothing could ever "
    "switch the grammar on and the reply would not match it. Send grammar_triggers "
    "alongside grammar_lazy, or resend without grammar_lazy to constrain the whole "
    "reply."
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
        nobody declared it, so a new backend inherits the safe answer. A backend
        that cannot constrain generation must not produce unconstrained output
        the caller has no way to distinguish from a grammar-conformant answer.

        To offer grammar, set this True; overriding :meth:`validate_grammar` on
        top of that is optional and only buys an UP-FRONT rejection of a
        malformed grammar.
        """
        return False

    @property
    def supports_mtp(self) -> bool:
        """True when this backend has active Multi-Token Prediction (MTP) heads
        loaded for speculative drafting. Default False."""
        return False

    def validate_grammar(self, grammar: Optional[str], *, lazy: bool = False) -> None:
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

        A concrete method rather than an optional attribute: an absent method
        cannot say "I cannot do this", it can only be missing, and every caller
        then has to guess what missing meant.

        *lazy* says the caller asked for LAZY semantics (unconstrained until a
        trigger pattern matches, grammar enforced from there). The base does NOT
        consult a lazy capability flag, and there is no ``supports_lazy_grammar``
        alongside :attr:`supports_grammar`: the GGUF backend cannot probe its own
        lazy support cheaply or safely. ``_api.has_lazy_grammar()`` RAISES
        RuntimeError rather than returning False when no runtime is provisioned,
        and when one IS provisioned it answers only by loading llama.dll into the
        calling process - in the server parent that is the doomed combination the
        whole spawn-worker isolation exists to avoid (see
        ``_loader.native_lib_loaded``). Only a backend that can PROVE it cannot
        do lazy overrides this and refuses - see ``HFBackend.validate_grammar``,
        whose answer rests on a static fact about xgrammar rather than on a probe.

        Staying silent HERE costs earliness, not honesty. GGUF's sampler build
        RAISES :class:`GrammarUnsupportedError` when it cannot apply the lazy
        grammar, instead of dropping it and generating unconstrained text (see
        ``llamacpp/llama.py``'s ``_build_sampler``), and that type survives the
        worker IPC as a tagged envelope, so the caller still gets a 400 naming
        the real problem - one request later than an up-front check would have,
        and never a reply that quietly does not match the grammar.
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
        (nomic-embed, bge-*, e5-*); a chat/instruct model makes poor ones.
        """
        raise NotImplementedError(
            "This backend does not support embedding.  "
            "Load a dedicated embedding model (e.g. nomic-embed-text)."
        )
