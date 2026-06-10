"""HuggingFace Transformers backend — supports text-only and multimodal models.

Tested with:
  - Gemma4UnifiedForConditionalGeneration (text + image + audio)
  - AutoModelForCausalLM (text only)
  - Any model with apply_chat_template support

GPU: uses torch.cuda (which maps to ROCm on AMD systems with PyTorch+ROCm).
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Iterator, List, Optional

from rich.console import Console

from .base import BaseBackend

console = Console()


def _require_torch():
    try:
        import torch
        return torch
    except ImportError:
        console.print(
            "[red]torch not installed.[/red]\n"
            "Install for AMD GPU: [bold]uv pip install -e '.[gpu]'[/bold]"
        )
        raise


def _require_transformers():
    try:
        import transformers
        return transformers
    except ImportError:
        console.print(
            "[red]transformers not installed.[/red]\n"
            "Install: [bold]uv pip install -e '.[gpu]'[/bold]"
        )
        raise


class HFBackend(BaseBackend):
    """
    Loads any HuggingFace-format model directory.

    Multimodal detection is automatic: if the model directory ships a processor
    that handles images/audio, multimodal content in messages is handled.
    If the model only has a tokenizer, image/audio parts are silently dropped
    and only text is passed to the model.
    """

    def __init__(self, model_path: str, device: Optional[str] = None) -> None:
        self.model_path = str(Path(model_path).resolve())
        self._device = device      # None = auto-detect
        self._model = None
        self._processor = None     # AutoProcessor (multimodal)
        self._tokenizer = None     # AutoTokenizer fallback
        self._is_multimodal = False
        self._loaded = False

    # ------------------------------------------------------------------ #
    #  Load / unload                                                       #
    # ------------------------------------------------------------------ #

    def load(self) -> None:
        torch = _require_torch()
        tr = _require_transformers()

        device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if device == "cuda" else torch.float32

        console.print(f"[dim]  device   : {device}[/dim]")
        if device == "cuda":
            console.print(f"[dim]  gpu      : {torch.cuda.get_device_name(0)}[/dim]")
        console.print(f"[dim]  dtype    : {dtype}[/dim]")

        # --- Processor / tokenizer ---
        console.print("[dim]  loading processor…[/dim]")
        try:
            self._processor = tr.AutoProcessor.from_pretrained(
                self.model_path, trust_remote_code=True
            )
            # A processor that wraps only a tokenizer is not "multimodal"
            has_image = hasattr(self._processor, "image_processor")
            has_audio = hasattr(self._processor, "feature_extractor") or hasattr(
                self._processor, "audio_processor"
            )
            self._is_multimodal = has_image or has_audio
            self._tokenizer = getattr(self._processor, "tokenizer", self._processor)
        except Exception:
            # Fall back to plain tokenizer
            self._processor = None
            self._tokenizer = tr.AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=True
            )

        # --- Model ---
        console.print("[dim]  loading weights…[/dim]")
        load_kwargs = {
            "device_map": "auto" if device == "cuda" else "cpu",
            "torch_dtype": dtype,
            "trust_remote_code": True,
        }

        # Try classes in order: conditional generation (multimodal/seq2seq),
        # then causal LM (text-only), then generic fallback.
        for cls_name in (
            "AutoModelForConditionalGeneration",
            "AutoModelForCausalLM",
            "AutoModel",
        ):
            try:
                cls = getattr(tr, cls_name)
                self._model = cls.from_pretrained(self.model_path, **load_kwargs)
                console.print(f"[dim]  class    : {cls_name}[/dim]")
                break
            except (ValueError, OSError, RuntimeError):
                continue

        if self._model is None:
            raise RuntimeError(f"Could not load model from {self.model_path}")

        self._loaded = True
        mm_note = " (multimodal)" if self._is_multimodal else ""
        console.print(f"[green]✓[/green] Model loaded{mm_note}")

    def unload(self) -> None:
        import gc
        self._model = None
        self._processor = None
        self._tokenizer = None
        self._loaded = False
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    @property
    def loaded(self) -> bool:
        return self._loaded

    # ------------------------------------------------------------------ #
    #  Inference                                                           #
    # ------------------------------------------------------------------ #

    def chat_stream(
        self,
        messages: List[dict],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 40,
        repeat_penalty: float = 1.1,
        grammar: Optional[str] = None,   # accepted but ignored — HF has no GBNF sampler
    ) -> Iterator[str]:
        from transformers import TextIteratorStreamer

        tokenizer = self._tokenizer
        model = self._model

        # --- Extract and decode media, rebuild messages for the chat template ---
        images = []
        audios = []
        template_messages = []

        for msg in messages:
            if isinstance(msg.get("content"), list):
                parts = []
                for part in msg["content"]:
                    ptype = part.get("type", "text")
                    if ptype == "text":
                        parts.append({"type": "text", "text": part["text"]})
                    elif ptype == "image_url" and self._is_multimodal:
                        from localm.inference.media import decode_image_url
                        img = decode_image_url(part["image_url"]["url"])
                        images.append(img)
                        parts.append({"type": "image"})
                    elif ptype == "input_audio" and self._is_multimodal:
                        from localm.inference.media import decode_audio
                        audio, sr = decode_audio(
                            part["input_audio"]["data"],
                            part["input_audio"].get("format", "wav"),
                        )
                        audios.append((audio, sr))
                        parts.append({"type": "audio"})
                    # else: drop unsupported media on text-only models
                template_messages.append({"role": msg["role"], "content": parts})
            else:
                template_messages.append(msg)

        # --- Tokenize / process ---
        if self._processor and (images or audios):
            # Full multimodal path
            text = self._processor.apply_chat_template(
                template_messages, tokenize=False, add_generation_prompt=True
            )
            process_kwargs = {"text": text, "return_tensors": "pt"}
            if images:
                process_kwargs["images"] = images
            if audios:
                process_kwargs["audios"] = audios
            inputs = self._processor(**process_kwargs).to(model.device)
        else:
            # Text-only path (even if processor exists, no media was provided)
            text = tokenizer.apply_chat_template(
                template_messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(text, return_tensors="pt").to(model.device)

        input_len = inputs["input_ids"].shape[-1]

        # --- Streaming generation ---
        streamer = TextIteratorStreamer(
            tokenizer, skip_special_tokens=True, skip_prompt=True
        )

        gen_kwargs: dict = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": max_tokens,
            "repetition_penalty": repeat_penalty,
        }
        if temperature > 0:
            gen_kwargs.update(
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
        else:
            gen_kwargs["do_sample"] = False

        thread = threading.Thread(
            target=model.generate, kwargs=gen_kwargs, daemon=True
        )
        thread.start()

        for token_text in streamer:
            yield token_text

        thread.join()
