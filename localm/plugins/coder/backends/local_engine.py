# SPDX-License-Identifier: AGPL-3.0-or-later
"""
In-process coder backend backed by the inference Engine.

Used for a CPU-RESIDENT reviewer model: a small local model loaded in the coder's
OWN process, separate from the localm server's GPU model. That gives a genuinely
HETEROGENEOUS review (a different model than the one that wrote the code) that is
also fully LOCAL and PRIVATE - no cloud, no off-machine traffic - and on CPU
(``n_gpu_layers=0`` / ``device="cpu"``) so it never touches the server's GPU VRAM
or evicts the main model. The trade-off is latency: a CPU load plus slow CPU
inference, paid only when the (opt-in) reviewer runs.

The model is loaded lazily on first use; a load failure surfaces through the
caller, which for the reviewer is fail-open (a broken reviewer never blocks).
"""

from __future__ import annotations

from typing import Iterator

from .base import BaseLLMBackend

# Generation kwargs the inference Engine.chat_stream accepts; anything else the
# coder passes (it forwards arbitrary gen_kwargs) is dropped so the call cannot
# raise a TypeError.
_ENGINE_GEN_KWARGS = frozenset({
    "max_tokens", "temperature", "top_p", "top_k", "repeat_penalty", "grammar", "seed",
})


class LocalEngineBackend(BaseLLMBackend):
    """A BaseLLMBackend that runs an in-process inference Engine (e.g. a small GGUF
    on CPU). Construction validates the model path but does NOT load weights; the
    first chat() loads them."""

    def __init__(self, model_path: str, *, device: str = "cpu",
                 n_gpu_layers: int = 0, display_name=None) -> None:
        from localm.inference.engine import Engine, model_display_name
        # Engine.__init__ -> create_backend validates the path is a GGUF/HF model
        # (raises ValueError otherwise) but does not load it.
        self._engine = Engine(model_path, device=device,
                              n_gpu_layers=n_gpu_layers, display_name=display_name)
        self._model_id = display_name or model_display_name(model_path)
        self._loaded = False
        # supports_grammar (BaseLLMBackend) gates whether callers (context.py's
        # tool-call forcing and JSON-summary compaction) trust this backend to
        # actually enforce a GBNF grammar it is handed, rather than silently
        # generating unconstrained.
        #
        # Deferred to the Engine's own capability property (engine.py's
        # supports_grammar -> backend.supports_grammar) instead of hardcoding
        # "only GgufBackend" - that used to be correct because HFBackend's
        # grammar support was a DELIBERATE, ROUTINE soft-degrade (the worker
        # silently dropped the grammar and warned whenever the optional
        # `[grammar]` extra was missing or failed to compile), so a caller
        # handed grammar=... had no reliable signal it was actually applied.
        # #1215 changed that: HFBackend.supports_grammar (hf.py) is now True
        # only when xgrammar is actually importable, and the worker drops
        # only a LAZY grammar - a FORCED one (the non-lazy form this flag
        # gates for rung 2 of the no-tool-call escalation ladder) still goes
        # through the grammar processor and is honoured. Reading the
        # underlying backend's own honest answer means an HF-backed reviewer
        # model with the extra installed gets forcing too, instead of being
        # permanently denied it by a class check that predates that fix.
        # GgufBackend still reports True unconditionally (llama.cpp applies a
        # GBNF grammar natively, no optional dependency involved), so this
        # changes nothing for the common case.
        self.supports_grammar = self._engine.supports_grammar

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self._engine.load()
            self._loaded = True

    @staticmethod
    def _gen(kwargs: dict) -> dict:
        return {k: v for k, v in kwargs.items()
                if k in _ENGINE_GEN_KWARGS and v is not None}

    def chat(self, messages: list[dict], **kwargs) -> str:
        self._ensure_loaded()
        return "".join(self._engine.chat_stream(messages, **self._gen(kwargs)))

    def chat_stream(self, messages: list[dict], **kwargs) -> Iterator[str]:
        self._ensure_loaded()
        yield from self._engine.chat_stream(messages, **self._gen(kwargs))

    @property
    def model_id(self) -> str:
        return self._model_id

    def unload(self) -> None:
        """Free the model (CPU RAM). Best-effort."""
        try:
            self._engine.unload()
        finally:
            self._loaded = False
