# SPDX-License-Identifier: AGPL-3.0-or-later
"""HuggingFace Transformers backend - supports text-only and multimodal models.

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

from localm.debuglog import logger

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


def _cuda_device_map(torch, config: Optional[dict] = None) -> dict:
    """Build the ``device_map`` (+ optional ``max_memory``) load kwargs for a
    CUDA load, honouring ``gpu_split_indices`` / ``main_gpu_index`` the same
    way the GGUF backend's native params do (see
    ``discover.apply_gpu_split`` / ``discover.apply_main_gpu``) - closing a
    real gap where this backend used to hardcode ``device_map="auto"``
    regardless of either setting, so "Main GPU = 1" silently did nothing for
    an HF (transformers) load:

    - 2+ valid ``gpu_split_indices`` -> ``"auto"`` sharded ONLY across those
      devices. Any GPU id absent from ``max_memory`` is excluded from
      accelerate's auto-shard - the standard technique for restricting
      ``device_map="auto"`` to a device subset.
    - no split, but a valid ``main_gpu_index`` -> pin the WHOLE model onto
      that one device (``device_map={"": idx}``) instead of auto-sharding
      across every visible card.
    - neither configured -> ``"auto"`` across every visible device, exactly
      today's default (existing installs see no behavior change).
    """
    from localm.config import load_config
    from localm.discover import resolve_gpu_split, resolve_main_gpu_index
    cfg = config if config is not None else load_config()

    pairs = resolve_gpu_split(cfg.get("gpu_split_indices"), cfg.get("gpu_split_ratios"))
    if len(pairs) >= 2:
        headroom = int(0.5e9)   # leave a little free per device, like the GGUF backend
        max_memory: dict = {}
        for idx, _ratio in pairs:
            try:
                free, _total = torch.cuda.mem_get_info(idx)
            except Exception:
                continue   # one device failing to report never blocks the rest
            max_memory[idx] = max(0, int(free) - headroom)
        if len(max_memory) >= 2:
            return {"device_map": "auto", "max_memory": max_memory}
        logger.warning(
            "gpu_split_indices is configured but free VRAM could not be read "
            "for enough devices (only %s usable); falling back to the "
            "default device_map", sorted(max_memory))

    if cfg.get("main_gpu_index") is not None:
        idx = resolve_main_gpu_index(cfg.get("main_gpu_index"))
        return {"device_map": {"": idx}}

    return {"device_map": "auto"}


def _auto_device(torch, override: Optional[str] = None) -> str:
    """Pick the HF inference device: an explicit *override*, else the best available
    GPU, else CPU. CUDA (which also covers AMD ROCm via PyTorch) is preferred, then
    Intel XPU (torch.xpu) so an Intel Arc/Xe GPU is used instead of silently falling
    back to CPU. torch.xpu is absent on older PyTorch, hence the getattr guard.
    Pure + torch-injected so it is testable without a GPU."""
    if override:
        return override
    if torch.cuda.is_available():
        return "cuda"
    xpu = getattr(torch, "xpu", None)
    if xpu is not None and xpu.is_available():
        return "xpu"
    return "cpu"


class _SafeGrammarProcessor:
    """Wrap an xgrammar HF LogitsProcessor so a RUNTIME failure during generation
    (for example xgrammar needing Triton, which is not available on Windows)
    degrades to unconstrained decoding instead of raising inside the generate()
    thread - which crashes the thread and hangs the HTTP request indefinitely.

    The grammar compiles fine, so the build-time soft-degrade cannot catch this;
    the failure only surfaces on the first token's logits call. We catch it there,
    warn once, and pass logits through unchanged for the rest of the generation.
    """

    def __init__(self, inner):
        self._inner = inner
        self._failed = False

    def __call__(self, input_ids, scores):
        if self._failed:
            return scores
        try:
            return self._inner(input_ids, scores)
        except Exception as e:
            console.print(
                f"[yellow]grammar constraint disabled mid-generation "
                f"({type(e).__name__}: {e}); continuing without constraint.[/yellow]"
            )
            self._failed = True
            return scores


def _grammar_processor(grammar: Optional[str], tokenizer, model):
    """Build an xgrammar LogitsProcessor that masks any token which would violate
    *grammar* at the current parse position (so output is structurally valid by
    construction, not by post-hoc repair).

    *grammar* is a GBNF/EBNF string with a ``root`` rule - see
    ``localm.inference.gbnf`` for ready-made JSON / tool-call grammars.

    Returns a one-element ``LogitsProcessorList`` or ``None``. ``None`` means
    "generate unconstrained": either no grammar was requested, or xgrammar is
    not installed / could not compile the grammar - in which case we warn and
    proceed rather than fail the request (soft-degrade). A FRESH processor is
    built per call because the underlying grammar matcher is stateful.
    """
    if not grammar:
        return None
    try:
        import xgrammar as xgr
        from xgrammar.contrib.hf import LogitsProcessor
        from transformers import LogitsProcessorList
    except ImportError:
        console.print(
            "[yellow]A grammar was requested but the [grammar] extra is not "
            "installed (pip install 'localm[gpu,grammar]'); generating without "
            "constraint.[/yellow]"
        )
        return None
    try:
        vocab = getattr(getattr(model, "config", None), "vocab_size", None)
        info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=vocab)
        compiled = xgr.GrammarCompiler(info).compile_grammar(grammar)
        return LogitsProcessorList([_SafeGrammarProcessor(LogitsProcessor(compiled))])
    except Exception as e:   # malformed grammar, tokenizer mismatch, etc.
        console.print(
            f"[yellow]grammar ignored ({type(e).__name__}: {e}); generating "
            f"without constraint.[/yellow]"
        )
        return None


class HFBackend(BaseBackend):
    """
    Loads any HuggingFace-format model directory.

    Multimodal detection is automatic: if the model directory ships a processor
    that handles images/audio, multimodal content in messages is handled.
    If the model only has a tokenizer, image/audio parts are silently dropped
    and only text is passed to the model.
    """

    # An HF checkpoint may ship an image processor; whether this instance can
    # actually see images is only known after load() (see supports_images).
    can_be_multimodal = True

    def __init__(self, model_path: str, device: Optional[str] = None) -> None:
        self.model_path = str(Path(model_path).resolve())
        self._device = device      # None = auto-detect
        self._model = None
        self._processor = None     # AutoProcessor (multimodal)
        self._tokenizer = None     # AutoTokenizer fallback
        self._is_multimodal = False
        self._loaded = False

    @property
    def supports_images(self) -> bool:
        """True once a multimodal processor has been detected at load time."""
        return self._is_multimodal

    # ------------------------------------------------------------------ #
    #  Load / unload                                                       #
    # ------------------------------------------------------------------ #

    def load(self) -> None:
        torch = _require_torch()
        tr = _require_transformers()

        device = _auto_device(torch, self._device)
        dtype = torch.bfloat16 if device in ("cuda", "xpu") else torch.float32
        device_map_kwargs = (_cuda_device_map(torch) if device == "cuda"
                              else {"device_map": "cpu"})

        console.print(f"[dim]  device   : {device}[/dim]")
        if device == "cuda":
            dm = device_map_kwargs["device_map"]
            if isinstance(dm, dict):   # pinned to one explicit device: {"": idx}
                idx = dm[""]
                console.print(
                    f"[dim]  gpu      : {torch.cuda.get_device_name(idx)} "
                    f"(pinned, device {idx})[/dim]")
            elif "max_memory" in device_map_kwargs:   # split across a device subset
                names = ", ".join(
                    f"{i}:{torch.cuda.get_device_name(i)}"
                    for i in sorted(device_map_kwargs["max_memory"]))
                console.print(f"[dim]  gpu      : split across {names}[/dim]")
            else:
                console.print(f"[dim]  gpu      : {torch.cuda.get_device_name(0)}[/dim]")
        elif device == "xpu":
            try:
                console.print(f"[dim]  gpu      : {torch.xpu.get_device_name(0)}[/dim]")
            except Exception:
                pass
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
        except Exception as e:
            # Fall back to plain tokenizer. This is intentional for text-only
            # models (no processor to load), but a logged failure here may mean
            # a genuine multimodal model lost its image/audio capability, so
            # surface it rather than swallowing it silently.
            logger.warning(
                "processor load failed (%s: %s); falling back to text-only tokenizer",
                type(e).__name__, e,
            )
            self._processor = None
            self._tokenizer = tr.AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=True
            )

        # --- Model ---
        console.print("[dim]  loading weights…[/dim]")
        load_kwargs = {
            **device_map_kwargs,
            "torch_dtype": dtype,
            "trust_remote_code": True,
        }

        # Try Auto classes in order: multimodal (vision/audio + text), then
        # encoder-decoder, then causal LM (text-only), then generic fallback.
        # getattr-with-default skips a class that this transformers version does
        # not expose (the names drift between major releases) instead of raising.
        errors: list[str] = []
        for cls_name in (
            "AutoModelForImageTextToText",   # modern multimodal, transformers 5+
            "AutoModelForSeq2SeqLM",         # encoder-decoder
            "AutoModelForCausalLM",          # text-only decoder
            "AutoModel",                     # generic fallback
        ):
            cls = getattr(tr, cls_name, None)
            if cls is None:
                continue
            try:
                self._model = cls.from_pretrained(self.model_path, **load_kwargs)
                console.print(f"[dim]  class    : {cls_name}[/dim]")
                break
            except (ValueError, OSError, RuntimeError, KeyError) as e:
                # Record why each class was rejected so the final error names
                # the actual failures instead of a bare "could not load".
                errors.append(f"{cls_name}: {type(e).__name__}: {e}")
                continue

        if self._model is None:
            detail = "; tried: " + "; ".join(errors) if errors else ""
            raise RuntimeError(f"Could not load model from {self.model_path}{detail}")

        if device == "xpu":
            # The model loaded on CPU (device_map "cpu" above); move it to the Intel
            # GPU explicitly. device_map="auto" is unreliable on consumer Arc (many
            # parts do not implement the free-memory query accelerate needs), so we
            # place the whole model with .to("xpu") rather than auto-sharding it.
            # A single-device "cpu" map attaches no accelerate hook, so this .to()
            # moves the whole model (verified vs accelerate dispatch_model + the HF
            # Intel-Arc guide). PENDING real-Arc verification (dev box is AMD): that
            # this move, and the model.device-based input placement in chat_stream/
            # embed, actually run on the Intel GPU end to end.
            try:
                self._model = self._model.to("xpu")
            except Exception as e:
                raise RuntimeError(
                    f"loaded the model but could not place it on the Intel GPU (xpu): "
                    f"{e}. Check the Intel GPU driver and that torch was installed from "
                    "the xpu wheel index.") from e

        self._loaded = True
        mm_note = " (multimodal)" if self._is_multimodal else ""

        # VRAM usage after load
        if device == "cuda":
            try:
                for i in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(i) / 1024**3
                    reserved  = torch.cuda.memory_reserved(i)  / 1024**3
                    console.print(
                        f"[dim]  vram     : {allocated:.2f} GB allocated / "
                        f"{reserved:.2f} GB reserved (device {i})[/dim]"
                    )
            except Exception as e:
                # VRAM readout is cosmetic; a failure here must not fail the
                # load, but surface it under --debug so a broken stat is visible.
                logger.debug("could not read VRAM after load (%s)", type(e).__name__)
        elif device == "xpu":
            try:
                allocated = torch.xpu.memory_allocated() / 1024**3
                reserved  = torch.xpu.memory_reserved()  / 1024**3
                console.print(
                    f"[dim]  vram     : {allocated:.2f} GB allocated / "
                    f"{reserved:.2f} GB reserved (xpu)[/dim]"
                )
            except Exception as e:
                # Some consumer Arc parts do not implement the memory query; cosmetic.
                logger.debug("could not read XPU VRAM after load (%s)", type(e).__name__)

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
        except Exception as e:
            # Best-effort cache release; log under --debug so a failed reclaim
            # (cache may not be cleared) is discoverable without failing unload.
            logger.debug("empty_cache failed (%s); cache may not be cleared", type(e).__name__)
        try:
            import torch
            xpu = getattr(torch, "xpu", None)
            if xpu is not None and xpu.is_available():
                xpu.empty_cache()
        except Exception as e:
            logger.debug("xpu empty_cache failed (%s); cache may not be cleared", type(e).__name__)

    @property
    def loaded(self) -> bool:
        return self._loaded

    # ------------------------------------------------------------------ #
    #  Tokenisation                                                        #
    # ------------------------------------------------------------------ #

    def count_tokens(self, text: str) -> int:
        """Return exact token count using the loaded HF tokenizer."""
        if self._tokenizer is not None:
            try:
                ids = self._tokenizer.encode(text, add_special_tokens=False)
                return max(1, len(ids))
            except Exception as e:
                # Surface the failure under --debug: the return below is then a
                # chars/4 ESTIMATE, not an exact count, and context-budgeting
                # downstream is trusting an approximation.
                logger.debug(
                    "tokenizer.encode failed (%s); using heuristic estimate",
                    type(e).__name__,
                )
        return max(1, len(text) // 4)

    def count_messages_tokens(self, messages: List[dict]) -> int:
        """Return exact token count of the structured messages formatted with the
        HF tokenizer/processor's chat template."""
        if self._tokenizer is not None:
            try:
                template_messages = []
                for msg in messages:
                    content = msg.get("content")
                    if isinstance(content, list):
                        parts = []
                        for part in content:
                            ptype = part.get("type", "text")
                            if ptype == "text":
                                parts.append({"type": "text", "text": part.get("text", "")})
                            elif ptype == "image_url" and self._is_multimodal:
                                parts.append({"type": "image"})
                            elif ptype == "input_audio" and self._is_multimodal:
                                parts.append({"type": "audio"})
                        template_messages.append({"role": msg.get("role", "user"), "content": parts})
                    else:
                        template_messages.append(msg)

                if self._processor and self._is_multimodal:
                    text = self._processor.apply_chat_template(
                        template_messages, tokenize=False, add_generation_prompt=True
                    )
                else:
                    text = self._tokenizer.apply_chat_template(
                        template_messages, tokenize=False, add_generation_prompt=True
                    )
                return len(self._tokenizer.encode(text, add_special_tokens=False))
            except Exception:
                pass
        return super().count_messages_tokens(messages)

    # ------------------------------------------------------------------ #
    #  Embeddings                                                          #
    # ------------------------------------------------------------------ #

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Return embedding vectors via mean-pooling of the last hidden states.

        Works for any AutoModelForCausalLM or AutoModel that outputs hidden
        states.  For dedicated sentence-transformer models that expose
        `.encode()`, that method is preferred.
        """
        import torch
        tokenizer = self._tokenizer
        model = self._model

        if tokenizer is None or model is None:
            raise RuntimeError("Model not loaded - call load() first")

        # Sentence-transformer style models (e.g. nomic-embed, bge)
        if hasattr(model, "encode"):
            vecs = model.encode(texts, convert_to_tensor=False)
            return [v.tolist() for v in vecs]

        embeddings: list[list[float]] = []
        model.train(False)
        with torch.no_grad():
            for text in texts:
                enc = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                ).to(model.device)
                out = model(**enc, output_hidden_states=True)
                # Mean-pool the last hidden state over non-padding tokens
                hidden = out.hidden_states[-1]          # (1, seq, dim)
                mask   = enc["attention_mask"].unsqueeze(-1).float()
                vec    = (hidden * mask).sum(1) / mask.sum(1)
                embeddings.append(vec[0].cpu().tolist())
        return embeddings

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
        grammar: Optional[str] = None,   # GBNF/EBNF; masks output via xgrammar ([grammar] extra)
        grammar_lazy: bool = False,
        grammar_triggers: Optional[List[str]] = None,
        seed: Optional[int] = None,
    ) -> Iterator[str]:
        # xgrammar has no trigger/lazy mode: a lazy request must not silently
        # become a STRICT constraint (a strict grammar stalls thinking models),
        # so drop the grammar with a trace and generate unconstrained - the
        # same soft-degrade contract as everywhere else on this backend.
        if grammar and grammar_lazy:
            from localm.debuglog import logger as _dbg
            _dbg.debug("lazy grammar is not supported on the HF backend; "
                       "generating unconstrained")
            grammar = None
        # Refuse images on a text-only checkpoint instead of silently dropping
        # them (a processor-less model would otherwise ignore the picture and
        # answer from the text alone). Checked before importing transformers so
        # it fails fast and clearly.
        if not self._is_multimodal:
            from .base import (
                IMAGE_UNSUPPORTED_MESSAGE,
                UnsupportedInputError,
                messages_contain_image,
            )
            if messages_contain_image(messages):
                raise UnsupportedInputError(IMAGE_UNSUPPORTED_MESSAGE)

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
            # add_special_tokens=False: the template already emitted the model's
            # BOS, so re-tokenizing with the default would prepend a SECOND one
            # (see the text-path note below). Standard processors forward this to
            # their tokenizer.
            process_kwargs = {"text": text, "return_tensors": "pt",
                              "add_special_tokens": False}
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
            # add_special_tokens=False: the chat template already emits the
            # model's BOS (Gemma <bos>, Llama-3 <|begin_of_text|>, Mistral <s>),
            # so re-tokenizing with the tokenizer default would prepend a SECOND
            # BOS and degrade coherence. This matches what apply_chat_template(
            # tokenize=True) does internally; templates that emit no BOS
            # (ChatML/Qwen) are for models that take no standalone BOS, so
            # suppressing it here is correct for them too.
            inputs = tokenizer(
                text, return_tensors="pt", add_special_tokens=False
            ).to(model.device)

        # --- Streaming generation ---
        streamer = TextIteratorStreamer(
            tokenizer, skip_special_tokens=True, skip_prompt=True
        )

        if seed is not None:
            import torch as _torch
            _torch.manual_seed(seed)

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

        # Grammar-constrained decoding (optional [grammar] extra). When a grammar
        # is supplied, xgrammar masks tokens that would break it; sampling/greedy
        # then picks only from the still-legal tokens. Soft-degrades to
        # unconstrained generation if xgrammar is absent or the grammar is bad.
        lp = _grammar_processor(grammar, tokenizer, model)
        if lp is not None:
            gen_kwargs["logits_processor"] = lp

        thread = threading.Thread(
            target=model.generate, kwargs=gen_kwargs, daemon=True
        )
        thread.start()

        for token_text in streamer:
            yield token_text

        thread.join()
