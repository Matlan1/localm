# SPDX-License-Identifier: AGPL-3.0-or-later
"""ExLlama worker: runs ONLY inside the isolated child process core's
``_hf_runner`` spawns for ``backend.ExllamaBackend``.

Imported from its file under a private module name, so it uses absolute
imports only. EXL3 folders load through ExLlamaV3, EXL2 folders through
ExLlamaV2. The prompt is rendered and tokenised with the folder's Hugging Face
tokenizer and chat template, the same path the HF backend uses.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Callable, Iterator, Optional

from localm.debuglog import logger

NO_CUDA_MESSAGE = (
    "ExLlama needs an NVIDIA GPU with CUDA, and the PyTorch in this environment "
    "{why}. Use a GGUF of the same model, or the original Hugging Face weights.")


def _stop_token_ids(model_path: str, tokenizer) -> list[int]:
    """End-of-sequence ids: the tokenizer's, plus every ``eos_token_id`` in
    ``config.json`` and ``generation_config.json`` (an int or a list)."""
    ids: set[int] = set()
    eos = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos, int):
        ids.add(eos)
    for name in ("config.json", "generation_config.json"):
        try:
            data = json.loads((Path(model_path) / name).read_text(encoding="utf-8"))
        except (OSError, ValueError, RecursionError):
            continue
        if not isinstance(data, dict):
            continue
        for source in (data, data.get("text_config")):
            value = source.get("eos_token_id") if isinstance(source, dict) else None
            for item in (value if isinstance(value, list) else [value]):
                if isinstance(item, int) and not isinstance(item, bool):
                    ids.add(item)
    return sorted(ids)


def _template_messages(messages: list[dict]) -> list[dict]:
    """*messages* for the chat template: list content keeps its text parts;
    any other part raises UnsupportedInputError."""
    from localm.inference.backends.base import UnsupportedInputError
    out = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            parts = []
            for part in content:
                if part.get("type", "text") != "text":
                    raise UnsupportedInputError(
                        "EXL2 and EXL3 models in localm take text only; this "
                        f"message carries a {part.get('type')!r} part.")
                parts.append({"type": "text", "text": part.get("text", "")})
            out.append({**msg, "content": parts})
        else:
            out.append(msg)
    return out


class _V3Runtime:
    """ExLlamaV3: EXL3 weights."""

    def load(self, model_path: str, cache_tokens: int, device: Optional[str]) -> None:
        from exllamav3 import Cache, Config, Generator, Model, Tokenizer
        self.config = Config.from_directory(model_path)
        self.model = Model.from_config(self.config)
        self.cache = Cache(self.model, max_num_tokens=cache_tokens)
        if device:
            self.model.load(device=device)
        else:
            self.model.load()
        self.tokenizer = Tokenizer.from_config(self.config)
        self.generator = Generator(model=self.model, cache=self.cache, tokenizer=self.tokenizer)

    def job(self, input_ids, max_new_tokens: int, sampling: dict, seed, stop: list):
        from exllamav3 import Job
        from exllamav3.generator.sampler import ComboSampler
        sampler = ComboSampler(
            rep_p=sampling["repeat_penalty"], pres_p=sampling["presence_penalty"],
            freq_p=sampling["frequency_penalty"], temperature=sampling["temperature"],
            min_p=sampling["min_p"], top_k=sampling["top_k"], top_p=sampling["top_p"])
        return Job(input_ids=input_ids, max_new_tokens=max_new_tokens, sampler=sampler,
                   seed=seed, stop_conditions=stop)

    def unload(self) -> None:
        model = getattr(self, "model", None)
        if model is not None:
            model.unload()
        self.generator = self.cache = self.model = None


class _V2Runtime:
    """ExLlamaV2: EXL2 weights, split across every visible GPU."""

    def load(self, model_path: str, cache_tokens: int, device: Optional[str]) -> None:
        from exllamav2 import ExLlamaV2, ExLlamaV2Cache, ExLlamaV2Config, ExLlamaV2Tokenizer
        from exllamav2.generator import ExLlamaV2DynamicGenerator
        self.config = ExLlamaV2Config(model_path)
        self.config.arch_compat_overrides()
        self.config.max_seq_len = cache_tokens
        self.model = ExLlamaV2(self.config)
        self.cache = ExLlamaV2Cache(self.model, max_seq_len=cache_tokens, lazy=True)
        self.model.load_autosplit(self.cache)
        self.tokenizer = ExLlamaV2Tokenizer(self.config)
        self.generator = ExLlamaV2DynamicGenerator(
            model=self.model, cache=self.cache, tokenizer=self.tokenizer)

    def job(self, input_ids, max_new_tokens: int, sampling: dict, seed, stop: list):
        from exllamav2.generator import ExLlamaV2DynamicJob, ExLlamaV2Sampler
        settings = ExLlamaV2Sampler.Settings()
        settings.temperature = sampling["temperature"]
        settings.top_k = sampling["top_k"]
        settings.top_p = sampling["top_p"]
        settings.min_p = sampling["min_p"]
        settings.token_repetition_penalty = sampling["repeat_penalty"]
        settings.token_presence_penalty = sampling["presence_penalty"]
        settings.token_frequency_penalty = sampling["frequency_penalty"]
        return ExLlamaV2DynamicJob(input_ids=input_ids, max_new_tokens=max_new_tokens,
                                   gen_settings=settings, seed=seed, stop_conditions=stop)

    def unload(self) -> None:
        model = getattr(self, "model", None)
        if model is not None:
            model.unload()
        self.generator = self.cache = self.model = None


_RUNTIMES = {"exl3": _V3Runtime, "exl2": _V2Runtime}


class ExllamaWorker:
    """One EXL2 / EXL3 model on the GPU, driven by ``_hf_runner``'s dispatch
    loop. Not thread-safe; the runner serialises every call."""

    supports_images = False
    processor_error: Optional[str] = None
    can_embed = False

    def __init__(self, model_path: str, fmt: str, cache_tokens: int,
                 device: Optional[str] = None) -> None:
        if fmt not in _RUNTIMES:
            raise ValueError(f"unknown ExLlama format {fmt!r}")
        self.model_path = model_path
        self.fmt = fmt
        self.cache_tokens = int(cache_tokens)
        self.device = device
        self.context_capacity: Optional[int] = None
        self.resolved_device: Optional[str] = None
        self.load_notes: list[str] = []
        self.last_finish_reason = "stop"
        self._runtime = None
        self._tokenizer = None
        self._stop_ids: list[int] = []

    def load(self) -> None:
        """Check for a CUDA device, then load the tokenizer and the weights.
        Raises ModelRefusedError when PyTorch has no CUDA device (a CPU or ROCm
        build, or no NVIDIA driver)."""
        import torch

        from localm.inference.backends.base import ModelRefusedError
        if getattr(torch.version, "hip", None):
            raise ModelRefusedError(NO_CUDA_MESSAGE.format(why="is a ROCm build"))
        if not torch.cuda.is_available():
            raise ModelRefusedError(NO_CUDA_MESSAGE.format(why="sees no CUDA device"))
        from transformers import AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=False)
        self._stop_ids = _stop_token_ids(self.model_path, self._tokenizer)
        runtime = _RUNTIMES[self.fmt]()
        runtime.load(self.model_path, self.cache_tokens, self.device)
        self._runtime = runtime
        self.context_capacity = self.cache_tokens
        count = torch.cuda.device_count()
        names = [torch.cuda.get_device_name(i) for i in range(count)]
        self.resolved_device = self.device or (
            f"cuda ({', '.join(names)})" if names else "cuda")
        if self.device and self.fmt == "exl2":
            self.load_notes.append(
                f"device {self.device} was not applied: ExLlamaV2 splits EXL2 "
                "models across every visible GPU")

    def unload(self) -> None:
        """Free the model and the cache."""
        runtime, self._runtime = self._runtime, None
        if runtime is not None:
            runtime.unload()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception as e:
            logger.debug("exllama worker: empty_cache failed: %s", e)

    @property
    def loaded(self) -> bool:
        return self._runtime is not None

    def _loaded_tokenizer(self):
        """The loaded tokenizer. Raises RuntimeError before :meth:`load`."""
        if self._tokenizer is None:
            raise RuntimeError("Model not loaded - call load() first")
        return self._tokenizer

    def _render(self, template_messages: list[dict], thinking: Optional[bool]):
        """``(text, prompt, template_kwargs)`` for *template_messages*: the
        template's rendering and the prompt to tokenise."""
        from localm.inference.backends._hf_worker import _require_chat_template
        tokenizer = self._loaded_tokenizer()
        _require_chat_template(tokenizer)
        template_kwargs = {"enable_thinking": False} if thinking is False else {}
        text = tokenizer.apply_chat_template(
            template_messages, tokenize=False, add_generation_prompt=True, **template_kwargs)
        prompt = text
        if thinking is False:
            from localm.inference.backends.base import no_think_prompt
            tmpl = getattr(tokenizer, "chat_template", None)
            prompt = no_think_prompt(text, tmpl if isinstance(tmpl, str) else None)
        return text, prompt, template_kwargs

    def count_tokens(self, text: str) -> int:
        """Token count of *text* with the model's tokenizer."""
        return max(1, len(self._loaded_tokenizer().encode(text, add_special_tokens=False)))

    def count_messages_tokens(self, messages: list[dict]) -> int:
        """Token count of *messages* rendered with the chat template."""
        template_messages = _template_messages(messages)
        text, prompt, kwargs = self._render(template_messages, None)
        from localm.inference.backends._hf_worker import _tokenize_prompt
        ids = _tokenize_prompt(self._tokenizer, template_messages, text, "cpu",
                               template_kwargs=kwargs, prompt=prompt)["input_ids"]
        return int(ids.shape[-1])

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError("EXL2 and EXL3 models do not produce embeddings in localm")

    def chat_stream(
        self,
        messages: list[dict],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 40,
        repeat_penalty: float = 1.1,
        seed: Optional[int] = None,
        cancel_event: Optional[threading.Event] = None,
        on_status: Optional[Callable[[str], None]] = None,
        thinking: Optional[bool] = None,
        min_p: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        grammar: Optional[str] = None,
        **_unused,
    ) -> Iterator[str]:
        """Yield the reply's text pieces. Sets :attr:`last_finish_reason` to
        "length" when ``max_tokens`` ran out and "stop" otherwise, a cancel
        through *cancel_event* included. Raises ContextCapacityExceededError
        when the prompt leaves no room in the cache."""
        from localm.inference.backends._hf_worker import (
            _resolve_max_new_tokens,
            _tokenize_prompt,
        )
        from localm.inference.backends.base import (
            ContextCapacityExceededError,
            GrammarUnsupportedError,
        )
        if grammar:
            raise GrammarUnsupportedError(
                "Grammar-constrained output is not available for EXL2 or EXL3 models.")
        if self._runtime is None:
            raise RuntimeError("Model not loaded - call load() first")
        self.last_finish_reason = "stop"
        template_messages = _template_messages(messages)
        text, prompt, kwargs = self._render(template_messages, thinking)
        input_ids = _tokenize_prompt(self._tokenizer, template_messages, text, "cpu",
                                     template_kwargs=kwargs, prompt=prompt)["input_ids"]
        n_prompt = int(input_ids.shape[-1])
        room = self.cache_tokens - n_prompt
        if room < 1:
            raise ContextCapacityExceededError(
                f"The prompt is {n_prompt} tokens and this model's context holds "
                f"{self.cache_tokens}, which leaves no room for a reply. Shorten the "
                "conversation or raise the context size.")
        max_new = min(_resolve_max_new_tokens(max_tokens, self.cache_tokens, n_prompt), room)
        sampling = {
            "temperature": float(temperature), "top_p": float(top_p), "top_k": int(top_k),
            "min_p": float(min_p or 0.0), "repeat_penalty": float(repeat_penalty or 1.0),
            "presence_penalty": float(presence_penalty or 0.0),
            "frequency_penalty": float(frequency_penalty or 0.0),
        }
        if on_status is not None:
            try:
                from localm.inference.protocol import PROCESSING_PROMPT_STATUS
                on_status(PROCESSING_PROMPT_STATUS)
            except Exception:
                logger.debug("exllama worker: on_status raised (ignored)", exc_info=True)
        runtime = self._runtime
        job = runtime.job(input_ids, max_new, sampling, seed, self._stop_ids)
        generator = runtime.generator
        generator.enqueue(job)
        finished = False
        try:
            while generator.num_remaining_jobs():
                if cancel_event is not None and cancel_event.is_set():
                    break
                for result in generator.iterate():
                    if result.get("stage") != "streaming":
                        continue
                    piece = result.get("text")
                    if piece:
                        yield piece
                    if result.get("eos"):
                        finished = True
                        if result.get("eos_reason") == "max_new_tokens":
                            self.last_finish_reason = "length"
        finally:
            if not finished:
                generator.cancel(job)
