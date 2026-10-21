# SPDX-License-Identifier: AGPL-3.0-or-later
"""Parent-side proxy for an EXL2 / EXL3 model.

Never imports torch or an ExLlama runtime: the model lives in an isolated child
process (core's ``_hf_runner.HFRunner`` hosting ``worker.ExllamaWorker``), so a
hang or a native fault there is contained the same way as for the HF backend.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Callable, Iterator, Optional

from localm.console import console
from localm.debuglog import logger
from localm.inference.backends._hf_runner import HFRunner, RunnerBusy
from localm.inference.backends.base import (
    AUDIO_UNSUPPORTED_MESSAGE,
    BaseBackend,
    GrammarUnsupportedError,
    ModelRefusedError,
    UnsupportedInputError,
    image_unsupported_message,
    messages_contain_audio,
    messages_contain_image,
)
from localm.inference.backends.hf import HFBackend

WORKER_FILE = Path(__file__).resolve().with_name("worker.py")
WORKER_CLASS = "ExllamaWorker"

_LABELS = {"exl2": "EXL2", "exl3": "EXL3"}
_VENDOR_NAMES = {"amd": "AMD", "intel": "Intel", "apple": "Apple", "nvidia": "NVIDIA"}
_RUNTIMES = {"exl2": "exllamav2", "exl3": "exllamav3"}

# Tokens per ExLlama cache page; the cache size is rounded up to a whole page.
CACHE_PAGE_TOKENS = 256
_FP16_BYTES = 2

GRAMMAR_REFUSAL = ("Grammar-constrained output is not available for EXL2 or EXL3 "
                   "models in localm yet; send the request without a grammar.")

_ALTERNATIVE = "Use a GGUF of the same model, or the original Hugging Face weights."


def _gpu_refusal(fmt: str) -> Optional[str]:
    """The sentence refusing a load on a machine whose GPUs include no NVIDIA
    one, or None when an NVIDIA GPU was found or the GPU probe could not run
    (the worker then checks CUDA itself)."""
    from localm.hwdetect import detect
    found = detect()
    if "nvidia" in found.vendors or found.gpu_state == "unknown":
        return None
    names = [str(_VENDOR_NAMES.get(v, v)) for v in found.vendors]
    if not names:
        have = "no GPU"
    elif len(names) == 1:
        have = f"an {names[0]} GPU" if names[0][0] in "AEIOU" else f"a {names[0]} GPU"
    else:
        have = f"{', '.join(names[:-1])} and {names[-1]} GPUs"
    return (f"This model is quantized with {_LABELS.get(fmt, fmt)}, which localm runs with "
            f"ExLlama on an NVIDIA GPU (CUDA) only, and this machine has {have}. "
            f"{_ALTERNATIVE}")


def _platform_refusal(fmt: str) -> Optional[str]:
    """The sentence refusing a load on an operating system the plugin does not
    run on (anything but Linux), or None."""
    if sys.platform.startswith("linux"):
        return None
    return (f"This model is quantized with {_LABELS.get(fmt, fmt)}; localm's ExLlama "
            f"plugin runs it on Linux with an NVIDIA GPU only. {_ALTERNATIVE}")


def _runtime_refusal(fmt: str) -> Optional[str]:
    """The sentence refusing a load when a module the worker needs is not
    installed, or None."""
    missing = [m for m in ("torch", "transformers", _RUNTIMES.get(fmt, "exllamav3"))
               if importlib.util.find_spec(m) is None]
    if not missing:
        return None
    return (f"The ExLlama plugin cannot load this model: {', '.join(missing)} "
            "is not installed. Install the plugin's dependencies with: "
            "localm plugin install-deps exllama")


def _read_config(model_path: str) -> dict:
    try:
        data = json.loads((Path(model_path) / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, RecursionError):
        return {}
    return data if isinstance(data, dict) else {}


def _text_config(config: dict) -> dict:
    nested = config.get("text_config")
    return nested if isinstance(nested, dict) else config


def _positive_int(value) -> Optional[int]:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def cache_tokens_for(model_path: str, n_ctx: Optional[int]) -> int:
    """Tokens the ExLlama cache holds, a whole number of
    :data:`CACHE_PAGE_TOKENS` pages: *n_ctx* (else the ``n_ctx`` setting) rounded
    up to a page, or, when that exceeds the model's ``max_position_embeddings``,
    that limit rounded down to a page. Never less than one page."""
    wanted = _positive_int(n_ctx)
    if wanted is None:
        from localm.config import load_config
        wanted = _positive_int(load_config().get("n_ctx")) or 4096
    pages = -(-wanted // CACHE_PAGE_TOKENS)
    limit = _positive_int(_text_config(_read_config(model_path)).get("max_position_embeddings"))
    if limit is not None:
        pages = min(pages, limit // CACHE_PAGE_TOKENS)
    return max(1, pages) * CACHE_PAGE_TOKENS


def kv_cache_bytes(model_path: str, tokens: int) -> Optional[int]:
    """FP16 key/value cache size for *tokens* tokens, from ``config.json``
    (``num_hidden_layers``, ``num_key_value_heads`` or ``num_attention_heads``,
    ``head_dim`` or ``hidden_size / num_attention_heads``); None when those
    cannot be read."""
    cfg = _text_config(_read_config(model_path))
    layers = _positive_int(cfg.get("num_hidden_layers"))
    heads = _positive_int(cfg.get("num_attention_heads"))
    kv_heads = _positive_int(cfg.get("num_key_value_heads")) or heads
    head_dim = _positive_int(cfg.get("head_dim"))
    hidden = _positive_int(cfg.get("hidden_size"))
    if head_dim is None and hidden is not None and heads is not None:
        head_dim = hidden // heads or None
    if layers is None or kv_heads is None or head_dim is None:
        return None
    return tokens * layers * kv_heads * head_dim * 2 * _FP16_BYTES


class ExllamaBackend(BaseBackend):
    """An EXL2 / EXL3 model folder, loaded in an isolated ExLlama worker."""

    def __init__(self, model_path: str, device: Optional[str] = None,
                 n_ctx: Optional[int] = None, fmt: Optional[str] = None) -> None:
        self.model_path = str(Path(model_path).resolve())
        self._fmt = fmt or "exl3"
        self._device = device
        self._n_ctx = n_ctx
        self._runner: Optional[HFRunner] = None
        self._loaded = False
        self.effective_ctx_max: Optional[int] = None
        self.n_ctx_max: Optional[int] = None
        self.last_finish_reason = "stop"

    @property
    def loaded(self) -> bool:
        """True while the worker that loaded the model is alive."""
        if not self._loaded:
            return False
        runner = self._runner
        return runner is not None and runner.is_alive()

    @property
    def streams_waiting(self) -> int:
        """Generations queued behind one already running; 0 when unloaded."""
        runner = self._runner
        return runner.streams_waiting if runner is not None else 0

    def unsupported_sampling(self, options: dict) -> list:
        """None: both ExLlama samplers apply min_p and the presence and
        frequency penalties."""
        return []

    def full_offload_vram_bytes(self) -> Optional[int]:
        """VRAM the load needs: the weights on disk plus the FP16 cache for
        :func:`cache_tokens_for` tokens; None when the cache size cannot be
        computed from ``config.json``."""
        from localm.inference.residency import model_footprint_bytes
        kv = kv_cache_bytes(self.model_path, cache_tokens_for(self.model_path, self._n_ctx))
        if kv is None:
            return None
        return model_footprint_bytes(self.model_path) + kv

    def load(self) -> None:
        """Start the worker and load the model. Raises ModelRefusedError, before
        any child is started, on a machine without an NVIDIA GPU, on an
        operating system other than Linux, or when the runtime is not
        installed; RuntimeError when the worker fails or crashes."""
        for refusal in (_gpu_refusal, _platform_refusal, _runtime_refusal):
            sentence = refusal(self._fmt)
            if sentence is not None:
                raise ModelRefusedError(sentence)
        from localm.inference.hf_shard_index_safety import validate_shard_index
        from localm.inference.hf_tokenizer_safety import validate_tokenizer_json
        validate_tokenizer_json(self.model_path)
        validate_shard_index(self.model_path)

        cache_tokens = cache_tokens_for(self.model_path, self._n_ctx)
        self._runner = HFRunner(worker_target=(str(WORKER_FILE), WORKER_CLASS),
                                process_name="exllama-worker")
        meta = self._runner.spawn_and_load(
            {"model_path": self.model_path, "fmt": self._fmt,
             "cache_tokens": cache_tokens, "device": self._device},
            timeout=HFBackend._load_timeout_seconds())
        capacity = meta.get("context_capacity") or cache_tokens
        self.effective_ctx_max = capacity
        self.n_ctx_max = capacity
        self._loaded = True
        from rich.markup import escape
        for note in meta.get("load_notes") or []:
            if isinstance(note, str) and note.strip():
                console.print(f"[dim]  {escape(note.strip())}[/dim]")
        device = meta.get("device") or "?"
        console.print(f"[green]✓[/green] Model loaded with ExLlama (device: {escape(str(device))})")

    def unload(self) -> None:
        """Stop the worker, freeing its VRAM. A no-op when none is running."""
        if self._runner is not None:
            try:
                self._runner.shutdown()
            except Exception as e:
                logger.debug("exllama worker shutdown failed (%s); its process may "
                             "not be fully torn down", type(e).__name__)
        self._runner = None
        self._loaded = False

    def count_tokens(self, text: str) -> int:
        """Exact count from the worker's tokenizer, or chars/4 while the worker is
        busy streaming or not loaded."""
        if self.loaded and self._runner is not None:
            try:
                return self._runner.count_tokens(text)
            except RunnerBusy:
                logger.debug("exllama count_tokens: worker busy; using the chars/4 estimate")
        return max(1, len(text) // 4)

    def count_messages_tokens(self, messages: list[dict]) -> int:
        """Exact count of *messages* rendered with the chat template, or the base
        estimate while the worker is busy or not loaded."""
        if self.loaded and self._runner is not None:
            try:
                return self._runner.count_messages_tokens(messages)
            except RunnerBusy:
                logger.debug("exllama count_messages_tokens: worker busy; using an estimate")
        return super().count_messages_tokens(messages)

    def chat_stream(
        self,
        messages: list[dict],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 40,
        repeat_penalty: float = 1.1,
        grammar: Optional[str] = None,
        grammar_lazy: bool = False,
        grammar_triggers: Optional[list[str]] = None,
        seed: Optional[int] = None,
        on_status: Optional[Callable[[str], None]] = None,
        thinking: Optional[bool] = None,
        min_p: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        cache_prompt: bool = True,
    ) -> Iterator[str]:
        """Stream the reply from the worker. Refuses images and grammars before
        the worker is asked; ``cache_prompt`` changes nothing."""
        if messages_contain_image(messages):
            raise UnsupportedInputError(image_unsupported_message(None))
        if messages_contain_audio(messages):
            raise UnsupportedInputError(AUDIO_UNSUPPORTED_MESSAGE)
        if grammar:
            raise GrammarUnsupportedError(GRAMMAR_REFUSAL)
        if self._runner is None or not self.loaded:
            raise RuntimeError("Model not loaded - call load() first")
        self.last_finish_reason = "stop"
        extra: dict = {"thinking": thinking} if thinking is not None else {}
        for key, value in (("min_p", min_p), ("presence_penalty", presence_penalty),
                           ("frequency_penalty", frequency_penalty)):
            if value is not None:
                extra[key] = value
        yield from self._runner.chat_stream(
            first_chunk_timeout=HFBackend._first_token_timeout_seconds(),
            on_status=on_status,
            **extra,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
            seed=seed,
        )
        done = getattr(self._runner, "last_done", None) or {}
        self.last_finish_reason = done.get("finish_reason", "stop")
