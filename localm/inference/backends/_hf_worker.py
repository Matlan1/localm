# SPDX-License-Identifier: AGPL-3.0-or-later
"""HuggingFace Transformers native worker - supports text-only and multimodal
models. Runs ONLY inside the isolated child process spawned by
``_hf_runner.py`` (see that module's docstring): every native call here
(tokenizer regex, `model.generate()`, a torch forward pass) is
uninterruptible from Python, so the process boundary is what makes a hang or
a native abort containable without taking the server down with it - the same
arrangement as ``backends/llamacpp/_worker.py``'s ``GgufWorker`` and
``embedder.py``'s ``GGUFEmbedder``.

This class does NOT inherit ``BaseBackend``; only the parent-side proxy in
``hf.py`` is handed to a caller expecting that public contract (mirroring
``GgufWorker``, likewise a plain class alongside its ``VramSizingMixin``).

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

from localm.debuglog import logger


def _require_torch():
    try:
        import torch
        return torch
    except ImportError:
        # logger.error, not console.print: this runs inside the isolated
        # child process (see the module docstring), whose stdout is not the
        # server's own console. The raise below carries this message back to
        # the parent as the RPC error envelope.
        logger.error("torch not installed. Install for AMD GPU: "
                     "uv pip install -e '.[gpu]'")
        raise


def _trust_remote_code_enabled() -> bool:
    """Whether transformers may import and execute a model directory's own .py.

    Config-driven and DEFAULT OFF (config key ``hf_trust_remote_code``,
    owner-only to set). When on, loading a model directory executes whatever
    Python that directory ships, via its ``auto_map``.

    A second copy of this function lives in the PARENT proxy (``hf.py``), used
    by ``_check_custom_code_allowed`` there: the REFUSAL decision runs in the
    parent, before a child is ever spawned, while this class needs the same
    boolean for the ``trust_remote_code=`` kwarg every ``from_pretrained`` call
    below takes. Two independent copies, not one cross-import.
    """
    try:
        from localm.config import load_config
        return bool(load_config().get("hf_trust_remote_code", False))
    except Exception as e:
        # Fail CLOSED: an unreadable config is never read as permission to
        # execute a model's bundled code. Logged rather than silent.
        logger.debug("hf: could not read hf_trust_remote_code, assuming off: %s", e)
        return False


def _require_transformers():
    try:
        import transformers
        from localm.inference.backends.awq import register_native_awq_quantizer
        register_native_awq_quantizer()
        return transformers
    except ImportError:
        # logger.error, not console.print - see _require_torch's identical note.
        logger.error("transformers not installed. Install: "
                     "uv pip install -e '.[gpu]'")
        raise


def _cuda_device_map(torch, config: Optional[dict] = None) -> dict:
    """Build the ``device_map`` (+ optional ``max_memory``) load kwargs for a
    CUDA load, honouring ``gpu_split_indices`` / ``main_gpu_index`` the same
    way the GGUF backend's native params do (see
    ``discover.apply_gpu_split`` / ``discover.apply_main_gpu``):

    - 2+ valid ``gpu_split_indices`` -> ``"auto"`` sharded ONLY across those
      devices. Any GPU id absent from ``max_memory`` is excluded from
      accelerate's auto-shard.
    - no split, but a valid ``main_gpu_index`` -> ``"auto"`` confined to that
      ONE device by the same technique (its id is the only GPU in
      ``max_memory``).
    - neither configured -> ``"auto"`` across every visible device.

    Every ``max_memory`` built here also carries a ``"cpu"`` budget: passing a
    max_memory at all suppresses the one accelerate would otherwise build for
    itself (``accelerate.utils.get_max_memory()`` populates a "cpu" entry from
    ``psutil.virtual_memory().available`` ONLY when the caller passes no dict).
    Without it, weights that do not fit the chosen GPU(s) are mapped to "disk",
    which needs an ``offload_folder`` and errors without one. With it, the
    overflow spills to CPU exactly as plain "auto" does.
    """
    from localm.config import load_config
    from localm.discover import resolve_gpu_split, resolve_main_gpu_index
    cfg = config if config is not None else load_config()

    headroom = int(0.5e9)   # leave a little free per device, like the GGUF backend

    def _free_minus_headroom(idx: int) -> Optional[int]:
        try:
            free, _total = torch.cuda.mem_get_info(idx)
        except Exception:
            return None
        return max(0, int(free) - headroom)

    def _cpu_budget() -> int:
        """Mirrors accelerate's own default for the "cpu" entry."""
        import psutil
        return int(psutil.virtual_memory().available)

    pairs = resolve_gpu_split(cfg.get("gpu_split_indices"), cfg.get("gpu_split_ratios"))
    if len(pairs) >= 2:
        max_memory: dict = {}
        for idx, _ratio in pairs:
            budget = _free_minus_headroom(idx)
            if budget is None:
                continue   # one device failing to report never blocks the rest
            max_memory[idx] = budget
        if len(max_memory) >= 2:
            max_memory["cpu"] = _cpu_budget()
            return {"device_map": "auto", "max_memory": max_memory}
        logger.warning(
            "gpu_split_indices is configured but free VRAM could not be read "
            "for enough devices (only %s usable); falling back to the "
            "default device_map", sorted(max_memory))

    if cfg.get("main_gpu_index") is not None:
        idx = resolve_main_gpu_index(cfg.get("main_gpu_index"))
        budget = _free_minus_headroom(idx)
        if budget is None:
            # No readable free VRAM means no budget to build, so there is no way
            # to express "this device, with CPU overflow". The explicit selection
            # is honoured by pinning, and reported: this is the one path with no
            # CPU fallback, so an oversized model here still OOMs.
            logger.warning(
                "main_gpu_index=%s is configured but its free VRAM could not be "
                "read; pinning the whole model to that device. A model larger "
                "than its free VRAM will fail to load rather than offloading to "
                "CPU.", idx)
            return {"device_map": {"": idx}}
        return {"device_map": "auto",
                "max_memory": {idx: budget, "cpu": _cpu_budget()}}

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
    the failure only surfaces on the first token's logits call. It is caught
    there, warned about once, and logits pass through unchanged for the rest of
    the generation.
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
            # logger.warning, not console.print: this runs inside the isolated
            # child process (see this module's docstring), whose stdout is not
            # guaranteed to share the parent's encoding. The debug log is UTF-8
            # regardless.
            logger.warning(
                "grammar constraint disabled mid-generation (%s: %s); "
                "continuing without constraint", type(e).__name__, e)
            self._failed = True
            return scores


def _eos_token_ids(model, tokenizer) -> set:
    """The end-of-sequence token id(s) transformers' own default stopping
    criteria would halt generation on for *model* - see ``EosTokenCriteria``
    / ``_get_stopping_criteria`` in transformers/generation/utils.py, which
    reads ``generation_config.eos_token_id`` (an int, a list, or unset).
    Reads the SAME source of truth rather than re-deriving it. Falls back to
    the tokenizer's own ``eos_token_id`` for a checkpoint whose
    generation_config leaves it unset."""
    raw = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if raw is None:
        raw = getattr(tokenizer, "eos_token_id", None)
    if raw is None:
        return set()
    if isinstance(raw, (list, tuple, set)):
        return {int(x) for x in raw}
    return {int(raw)}


def _resolve_max_new_tokens(max_tokens: int, context_capacity: Optional[int],
                             n_prompt: Optional[int]) -> int:
    """Translate ``max_tokens`` into the ``max_new_tokens`` transformers'
    ``generate()`` will accept.

    ``max_tokens<=0`` is this codebase's "unlimited" sentinel (see the GGUF
    backend's ``LlamaCpp._fit_generation_budget``), but ``generate()`` itself
    raises ``ValueError`` for a non-positive ``max_new_tokens`` - there is no
    equivalent sentinel on the transformers side. A positive ``max_tokens``
    passes through unchanged. Unlimited resolves to the room left in the
    context window (``context_capacity - n_prompt``, floored at 1) when both
    are known, else a fixed fallback (``DEFAULT_CONFIG["max_tokens"]``).
    """
    if max_tokens > 0:
        return max_tokens
    if context_capacity and n_prompt is not None:
        return max(1, context_capacity - n_prompt)
    from localm.config import DEFAULT_CONFIG
    return DEFAULT_CONFIG["max_tokens"]


class _FinishReasonObserver:
    """Installed as one of ``model.generate()``'s ``stopping_criteria`` to
    record WHY generation ended, without influencing the decision itself -
    the actual stop is still made by transformers' own built-in
    ``EosTokenCriteria``/``MaxLengthCriteria`` (``__call__`` below always
    returns "never stop"). Mirrors the native GGUF worker's EOG-vs-budget
    distinction (``backends/llamacpp/llama.py``'s ``_generate``: an
    end-of-generation token always wins over the length budget - "length"
    is reported only when the budget ran out with no EOG ever produced).

    In transformers' ``_sample`` loop (generation/utils.py) a new token is
    appended to ``input_ids`` via ``torch.cat`` BEFORE ``stopping_criteria``
    is called, so ``input_ids`` already includes it on every call: the delta
    from the length seen on the FIRST call is exactly the count of new tokens
    generated so far, for both decoder-only and encoder-decoder ``input_ids``
    conventions, with no need to know the prompt length in advance.
    """

    def __init__(self, eos_token_ids: set) -> None:
        self._eos_token_ids = eos_token_ids
        self._baseline_len: Optional[int] = None
        self.generated = 0
        self.ended_on_eos = False

    def __call__(self, input_ids, scores, **kwargs):
        import torch
        seq_len = input_ids.shape[-1]
        if self._baseline_len is None:
            self._baseline_len = seq_len - 1
        self.generated = seq_len - self._baseline_len
        self.ended_on_eos = int(input_ids[0, -1]) in self._eos_token_ids
        # Never itself vote to stop - a torch.BoolTensor of shape
        # (batch_size,), matching StoppingCriteriaList's `is_done |
        # criteria(...)` contract (see MaxLengthCriteria/EosTokenCriteria's
        # own return shape in transformers/generation/stopping_criteria.py).
        return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)


def _grammar_processor(grammar: Optional[str], tokenizer, model):
    """Build an xgrammar LogitsProcessor that masks any token which would violate
    *grammar* at the current parse position (so output is structurally valid by
    construction, not by post-hoc repair).

    *grammar* is a GBNF/EBNF string with a ``root`` rule - see
    ``localm.inference.gbnf`` for ready-made JSON / tool-call grammars.

    Returns a one-element ``LogitsProcessorList``, or ``None`` when no grammar was
    requested at all. It NEVER returns ``None`` to mean "I gave up on the
    grammar": a grammar that cannot be applied RAISES.

    Both raises are marshalled back as CLEAN refusals by ``_hf_runner``'s tagged
    error envelope, so neither kills the worker or leaves the model in an unknown
    state. A FRESH processor is built per call because the matcher is stateful.
    """
    from .base import (
        GRAMMAR_UNSUPPORTED_MESSAGE,
        GrammarUnsupportedError,
        InvalidGrammarError,
    )
    if not grammar:
        return None
    try:
        import xgrammar as xgr
        from xgrammar.contrib.hf import LogitsProcessor
        from transformers import LogitsProcessorList
    except ImportError:
        # Normally unreachable: HFBackend.supports_grammar is False without the
        # extra, so the up-front check in the chat routes already refused this
        # request. Reaching it anyway means the parent's check was bypassed or
        # the install changed under a live worker, and the same error that check
        # would have raised is raised here.
        raise GrammarUnsupportedError(GRAMMAR_UNSUPPORTED_MESSAGE)
    try:
        vocab = getattr(getattr(model, "config", None), "vocab_size", None)
        info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=vocab)
        compiled = xgr.GrammarCompiler(info).compile_grammar(grammar)
        return LogitsProcessorList([_SafeGrammarProcessor(LogitsProcessor(compiled))])
    except Exception as e:   # malformed grammar, tokenizer mismatch, etc.
        # Mirrors the GGUF backend: llama.py raises InvalidGrammarError when the
        # native GBNF parser rejects a grammar (a NULL sampler), which the route
        # turns into a 400 naming the grammar as the thing to fix. A grammar
        # xgrammar cannot compile gets the same answer.
        raise InvalidGrammarError(
            f"grammar could not be compiled ({type(e).__name__}: {e})") from e


class _CancelCriteria:
    """Duck-typed transformers StoppingCriteria that polls a shared
    threading.Event for a real, cooperative mid-stream cancel - the same idea
    as GgufBackend's ctrl_q relay into llama.cpp's native progress-callback
    hook, built on the hook transformers actually exposes instead.

    transformers' decode loop (generation/utils.py) calls every
    StoppingCriteria in ``stopping_criteria=`` once per generated token, right
    after that token is pushed onto the streamer
    (``unfinished_sequences = unfinished_sequences & ~stopping_criteria(...)``),
    so setting *cancel_event* stops generation within one extra token rather
    than waiting for max_new_tokens or a full process kill. See
    _hf_runner.py's module docstring for how the event gets set from a
    parent-process disconnect.

    Does NOT subclass ``transformers.StoppingCriteria``, which would force
    ``from transformers import StoppingCriteria`` at MODULE IMPORT time - the
    eager transformers/torch import this file avoids everywhere else (see
    ``_declared_generative``'s docstring). ``StoppingCriteriaList.__call__``
    and ``_merge_criteria_processor_list`` call/compare criteria purely by duck
    typing (``criteria(input_ids, scores, **kwargs)`` / ``type(custom) is
    type(default)``); the only ``isinstance(..., StoppingCriteria)`` in the
    generate() path selects a warning message's wording for a type COLLISION
    with a built-in criterion, which this class's unique type never triggers.
    Mirrors ``_SafeGrammarProcessor`` above, which duck-types
    ``LogitsProcessor``.

    __call__ must return a torch.BoolTensor of shape (batch_size,) -
    StoppingCriteriaList.__call__ ORs every criterion's result together
    (``is_done = is_done | criteria(...)``), so a plain Python bool would
    break under that ``|``.
    """

    def __init__(self, cancel_event: threading.Event):
        self._cancel_event = cancel_event

    def __call__(self, input_ids, scores, **kwargs):
        import torch
        stop = self._cancel_event.is_set()
        return torch.full((input_ids.shape[0],), stop, dtype=torch.bool,
                          device=input_ids.device)


# transformers' naming convention for a GENERATIVE task head. A checkpoint whose
# declared architecture ends in one of these generates text; anything else (the
# bare ``*Model`` encoders: BertModel, XLMRobertaModel, NomicBertModel,
# T5EncoderModel, DistilBertModel, MPNetModel) is an encoder that embeds.
# Matched by name rather than by importing transformers - see
# HFWorker._declared_generative.
_GENERATIVE_ARCH_SUFFIXES = (
    "ForCausalLM",
    "LMHeadModel",
    "ForConditionalGeneration",
    "ForSeq2SeqLM",
    "ForImageTextToText",
    "ForVision2Seq",
)


class HFWorker:
    """
    Loads any HuggingFace-format model directory. Runs only inside the
    isolated child process - see this module's docstring.

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
        # The RESOLVED device ("cuda"/"xpu"/"cpu"), set once load() picks
        # one - None beforehand ("auto" was requested and not decided yet).
        # Reported back to the parent proxy for its post-load status line
        # (the child cannot print it directly - see load()'s own note).
        self.resolved_device: Optional[str] = None
        # Maximum context capacity extracted from model config at load time.
        self.context_capacity: Optional[int] = None
        # Why the most recent chat_stream() call ended - "stop" (EOS) or
        # "length" (max_tokens exhausted first). Mirrors GgufWorker's
        # identical attribute; recomputed for real by chat_stream() below.
        self.last_finish_reason = "stop"

    @property
    def supports_images(self) -> bool:
        """True once a multimodal processor has been detected at load time."""
        return self._is_multimodal

    # ------------------------------------------------------------------ #
    #  Load / unload                                                       #
    # ------------------------------------------------------------------ #

    def load(self) -> None:
        # The custom-code refusal (_check_custom_code_allowed) and the
        # tokenizer.json ReDoS gate (validate_tokenizer_json) both run in the
        # PARENT, BEFORE this class is ever constructed - see hf.py's load().
        # This method only needs the boolean for the trust_remote_code= kwarg
        # below.
        trust_remote_code = _trust_remote_code_enabled()
        torch = _require_torch()
        tr = _require_transformers()

        device = _auto_device(torch, self._device)
        self.resolved_device = device
        dtype = torch.bfloat16 if device in ("cuda", "xpu") else torch.float32
        device_map_kwargs = (_cuda_device_map(torch) if device == "cuda"
                              else {"device_map": "cpu"})

        # logger.debug throughout load(), never console.print: this runs
        # inside the isolated child process (see this module's docstring),
        # whose stdout inherits whatever codepage the spawn gave it rather
        # than the parent's own console configuration. The debug log is UTF-8
        # regardless. User-facing status ("Model loaded", the final
        # device/vram summary) is printed by the PARENT proxy (hf.py's
        # HFBackend.load()) from this method's returned/cached metadata,
        # mirroring GgufBackend's own load().
        logger.debug("hf load: device=%s", device)
        if device == "cuda":
            dm = device_map_kwargs["device_map"]
            if isinstance(dm, dict):   # pinned to one explicit device: {"": idx}
                idx = dm[""]
                logger.debug("hf load: gpu=%s (pinned, device %d)",
                            torch.cuda.get_device_name(idx), idx)
            elif "max_memory" in device_map_kwargs:
                # Only the int keys are GPUs - max_memory also carries a "cpu"
                # overflow budget (see _cuda_device_map), which is neither a
                # device to name nor sortable against an int.
                gpus = sorted(i for i in device_map_kwargs["max_memory"]
                              if isinstance(i, int))
                names = ", ".join(f"{i}:{torch.cuda.get_device_name(i)}" for i in gpus)
                where = f"split across {names}" if len(gpus) > 1 else names
                logger.debug("hf load: gpu=%s (overflow to CPU if needed)", where)
            else:
                logger.debug("hf load: gpu=%s", torch.cuda.get_device_name(0))
        elif device == "xpu":
            try:
                logger.debug("hf load: gpu=%s", torch.xpu.get_device_name(0))
            except Exception:
                pass
        logger.debug("hf load: dtype=%s", dtype)

        # --- Processor / tokenizer ---
        logger.debug("hf load: loading processor")
        try:
            self._processor = tr.AutoProcessor.from_pretrained(
                self.model_path, trust_remote_code=trust_remote_code
            )
            # A processor that wraps only a tokenizer is not "multimodal"
            has_image = hasattr(self._processor, "image_processor")
            has_audio = hasattr(self._processor, "feature_extractor") or hasattr(
                self._processor, "audio_processor"
            )
            self._is_multimodal = has_image or has_audio
            self._tokenizer = getattr(self._processor, "tokenizer", self._processor)
        except Exception as e:
            # Fall back to plain tokenizer. Expected for text-only models (no
            # processor to load), but a logged failure here may mean a genuine
            # multimodal model lost its image/audio capability, so it is
            # surfaced rather than swallowed.
            logger.warning(
                "processor load failed (%s: %s); falling back to text-only tokenizer",
                type(e).__name__, e,
            )
            self._processor = None
            self._tokenizer = tr.AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=trust_remote_code
            )

        # --- Model ---
        logger.debug("hf load: loading weights")
        load_kwargs = {
            **device_map_kwargs,
            "torch_dtype": dtype,
            "trust_remote_code": trust_remote_code,
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
                logger.debug("hf load: class=%s", cls_name)
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
            # parts do not implement the free-memory query accelerate needs), so the
            # whole model is placed with .to("xpu") rather than auto-sharded.
            # A single-device "cpu" map attaches no accelerate hook, so this .to()
            # moves the whole model.
            try:
                self._model = self._model.to("xpu")
            except Exception as e:
                raise RuntimeError(
                    f"loaded the model but could not place it on the Intel GPU (xpu): "
                    f"{e}. Check the Intel GPU driver and that torch was installed from "
                    "the xpu wheel index.") from e

        self._loaded = True

        config = getattr(self._model, "config", None)
        self.context_capacity = None
        if config is not None:
            max_pos = (
                getattr(config, "max_position_embeddings", None) or
                getattr(config, "seq_length", None) or
                getattr(config, "max_sequence_length", None) or
                getattr(config, "n_positions", None)
            )
            if isinstance(max_pos, int) and max_pos > 0:
                self.context_capacity = max_pos

        # VRAM usage after load - debug log only (see the note at the top of
        # this method). The final "Model loaded" user-facing line (with its
        # device/vram summary) is printed by the PARENT proxy after this method
        # returns successfully - see hf.py.
        if device == "cuda":
            try:
                for i in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(i) / 1024**3
                    reserved  = torch.cuda.memory_reserved(i)  / 1024**3
                    logger.debug("hf load: vram device %d = %.2f GB allocated / "
                                "%.2f GB reserved", i, allocated, reserved)
            except Exception as e:
                # VRAM readout is cosmetic; a failure here must not fail the
                # load, but surface it under --debug so a broken stat is visible.
                logger.debug("could not read VRAM after load (%s)", type(e).__name__)
        elif device == "xpu":
            try:
                allocated = torch.xpu.memory_allocated() / 1024**3
                reserved  = torch.xpu.memory_reserved()  / 1024**3
                logger.debug("hf load: vram (xpu) = %.2f GB allocated / "
                            "%.2f GB reserved", allocated, reserved)
            except Exception as e:
                # Some consumer Arc parts do not implement the memory query; cosmetic.
                logger.debug("could not read XPU VRAM after load (%s)", type(e).__name__)

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
            except Exception as e:
                # Surface under --debug (mirroring count_tokens above): the
                # super() return below is then a heuristic ESTIMATE, not an
                # exact count, so context-budgeting downstream is trusting an
                # approximation. Never a silent pass.
                logger.debug(
                    "chat-template token count failed (%s); using heuristic estimate",
                    type(e).__name__,
                )
        # No BaseBackend to super() into here (see module docstring) - inline
        # the identical fallback BaseBackend.count_messages_tokens used.
        text = " ".join(
            m.get("content") if isinstance(m.get("content"), str)
            else " ".join(p.get("text", "") for p in (m.get("content") or [])
                          if p.get("type") == "text")
            for m in messages
        )
        return self.count_tokens(text)

    # ------------------------------------------------------------------ #
    #  Embeddings                                                          #
    # ------------------------------------------------------------------ #

    def _declared_generative(self) -> Optional[bool]:
        """Whether the CHECKPOINT declares a generative architecture, or None when
        it declares none.

        The checkpoint's own ``config.architectures`` is the signal, NOT the
        class ``load()`` happened to pick. ``load()`` tries
        ``AutoModelForCausalLM`` BEFORE ``AutoModel``, and transformers registers
        the pure-encoder families in ``MODEL_FOR_CAUSAL_LM_MAPPING_NAMES``
        (bert -> BertLMHeadModel, roberta -> RobertaForCausalLM, xlm-roberta,
        electra), so an embedding checkpoint declaring ``["BertModel"]``
        (bge-small, all-MiniLM, e5, bge-m3) loads as ``BertLMHeadModel`` and
        answers ``can_generate()`` True while being an encoder that embeds.

        Matched on transformers' task-head NAMING convention rather than by
        resolving the class: resolving would mean importing transformers here,
        and that import pulls in torch, whose ROCm init
        (``rocm_sdk.preload_libraries``) fails with ``OSError: [WinError 127]``
        in any process that already loaded the bundled llama.dll.
        """
        archs = getattr(getattr(self._model, "config", None), "architectures", None)
        if not archs:
            return None
        return any(str(a).endswith(_GENERATIVE_ARCH_SUFFIXES) for a in archs)

    @property
    def can_embed(self) -> bool:
        """True only when the LOADED model is a GENUINE embedding model.

        Unlike GgufBackend's fixed ``can_embed = False``, an HF checkpoint may be
        either: a sentence-transformer or a plain encoder (a real embedder, which
        ``embed()`` below serves well), or a chat decoder (which it does not).

        Mean-pooling a chat decoder's last hidden states returns healthy,
        non-zero, plausible-looking vectors that nevertheless cannot separate
        related from unrelated text: a decoder's max UNRELATED cosine can exceed
        its min RELATED cosine, so no threshold splits the two. That is decoder
        anisotropy (no contrastive training objective), not a pooling artifact -
        LAST-token pooling scores worse still.

        Unloaded -> True ("unknown, load to find out"): the capability is only
        knowable once the weights are in, and answering False here would stop
        routes/chat.py from ever loading a genuine HF embedding model. The
        parent-side proxy re-checks this once, right after load, and caches it
        (see hf.py); Engine.embed cannot re-check it live, since this class runs
        in a child process.
        """
        model = self._model
        if model is None:
            return True                      # unknown until loaded; the load decides
        if hasattr(model, "encode"):
            return True                      # sentence-transformer: purpose-built
        declared = self._declared_generative()
        if declared is not None:
            return not declared              # the checkpoint's own word (see above)
        # Nothing declared and nothing resolvable: fall back to what the loaded
        # class says about itself. Weaker (it is the signal the encoder families
        # defeat above), but with no declared architecture it is the only
        # evidence there is.
        can_generate = getattr(model, "can_generate", None)
        if not callable(can_generate):
            # Not a transformers PreTrainedModel (or a version without the API):
            # absence of proof is not proof of an embedder, so the dedicated
            # embedder is preferred over pooling what may be a chat decoder.
            logger.debug(
                "HF model %s declares no architecture and exposes no "
                "can_generate(); treating it as NOT an embedding model",
                type(model).__name__)
            return False
        try:
            return not can_generate()
        except Exception as e:
            logger.debug(
                "HF model %s: can_generate() raised (%s: %s); treating it as NOT "
                "an embedding model", type(model).__name__, type(e).__name__, e)
            return False

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Return embedding vectors via mean-pooling of the last hidden states.

        Works for any AutoModel-style ENCODER that outputs hidden states; for
        dedicated sentence-transformer models that expose `.encode()`, that
        method is preferred. Callers must gate on ``can_embed`` above: this is
        NOT a valid embedding path for a chat decoder.
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
        cancel_event: Optional[threading.Event] = None,
    ) -> Iterator[str]:
        # xgrammar has no trigger/lazy mode, and a lazy request must not silently
        # become a STRICT constraint either (a strict grammar stalls thinking
        # models). Both remaining options - drop it, or enforce it strictly - give
        # the caller something other than what it asked for, so this REFUSES.
        #
        # Normally unreachable: HFBackend refuses lazy in validate_grammar (a
        # clean 400 before either response is committed) AND again in chat_stream
        # before the runner is touched. This is the third line of defence, for a
        # caller driving HFWorker directly; _hf_runner marshals it back as a clean
        # refusal rather than a worker-killing fault.
        if grammar and grammar_lazy:
            from .base import GRAMMAR_LAZY_UNSUPPORTED_MESSAGE, GrammarUnsupportedError
            raise GrammarUnsupportedError(GRAMMAR_LAZY_UNSUPPORTED_MESSAGE)
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

        from transformers import (
            StoppingCriteriaList,
            TextIteratorStreamer,
        )

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

        # Check prompt token length against context capacity before generate
        input_ids = inputs.get("input_ids")
        n_prompt = None
        if input_ids is not None and self.context_capacity:
            n_prompt = int(input_ids.shape[-1])
            if n_prompt > self.context_capacity:
                from .base import ContextCapacityExceededError
                raise ContextCapacityExceededError(
                    f"Conversation ({n_prompt} tokens) has outgrown the maximum "
                    f"context window (context_capacity={self.context_capacity})."
                )

        effective_max_new_tokens = _resolve_max_new_tokens(
            max_tokens, self.context_capacity, n_prompt)

        # --- Streaming generation ---
        streamer = TextIteratorStreamer(
            tokenizer, skip_special_tokens=True, skip_prompt=True
        )

        if seed is not None:
            import torch as _torch
            _torch.manual_seed(seed)

        self.last_finish_reason = "stop"
        finish_observer = _FinishReasonObserver(_eos_token_ids(model, tokenizer))

        gen_kwargs: dict = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": effective_max_new_tokens,
            "repetition_penalty": repeat_penalty,
            "stopping_criteria": StoppingCriteriaList([finish_observer]),
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

        # Cooperative mid-stream cancel (optional): a disconnect relayed from
        # the parent sets *cancel_event*, and _CancelCriteria makes
        # generate()'s own decode loop see it and stop within one extra
        # token - see _hf_runner.py's module docstring for the full
        # mechanism. cancel_event=None (the default, e.g. any direct or
        # standalone caller) leaves generation unconstrained.
        # APPENDED to gen_kwargs["stopping_criteria"] (already carrying
        # finish_observer, added unconditionally above) rather than replacing
        # it: StoppingCriteriaList evaluates every criterion it holds
        # (StoppingCriteriaList.__call__ ORs them together), so both the
        # finish-reason observer and the cancel check run on every decode
        # step.
        if cancel_event is not None:
            gen_kwargs["stopping_criteria"].append(_CancelCriteria(cancel_event))

        thread = threading.Thread(
            target=model.generate, kwargs=gen_kwargs, daemon=True
        )
        thread.start()

        for token_text in streamer:
            yield token_text

        thread.join()
        # EOS wins over the length budget whenever both are true at once
        # (mirrors llama.py's _generate - see _FinishReasonObserver above);
        # "length" only when the budget ran out with no EOS ever produced.
        if finish_observer.ended_on_eos:
            self.last_finish_reason = "stop"
        elif finish_observer.generated >= effective_max_new_tokens:
            self.last_finish_reason = "length"
        else:
            self.last_finish_reason = "stop"
