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
        # enforce a GBNF grammar it is handed rather than generating
        # unconstrained. Deferred to the Engine's own capability property
        # (engine.py's supports_grammar -> backend.supports_grammar), so an
        # HF-backed reviewer model with xgrammar installed reports True and a
        # GgufBackend reports True unconditionally.
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
