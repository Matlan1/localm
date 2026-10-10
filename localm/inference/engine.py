# SPDX-License-Identifier: AGPL-3.0-or-later
"""Model factory - picks the right backend and exposes a unified inference API."""

from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path
from typing import Callable, Iterator, Optional

from rich.markup import escape

from localm.config import load_config
from localm.console import console
from localm.debuglog import logger
from localm.inference.backends.base import LOADING_MODEL_STATUS, BaseBackend
from localm.inference.mmap_setting import describe_mmap, resolve_use_mmap
from localm.textnorm import scrub_stream


# Process-global model-load lock: only one model is ever loading at a time,
# across every Engine in the process. Inference is NOT held here, only the load.
# RLock, so a re-entrant load on the same thread cannot deadlock.
#
# LOCK ORDER with localm.inference.embedder._LOCK: _LOAD_LOCK is the OUTER lock
# of the pair. The load path under this lock calls embedder status reads that
# take embedder._LOCK (backend ctx sizing -> loaded_path), so code holding
# embedder._LOCK must NEVER acquire _LOAD_LOCK or start a load.
_LOAD_LOCK = threading.RLock()


# Files that mark a directory as a real HF model. A config.json alone is not
# enough; weights or a tokenizer must sit next to it.
_HF_WEIGHT_GLOBS = ("*.safetensors", "*.bin", "*.pt", "*.pth")
_HF_TOKENIZER_FILES = (
    "tokenizer.json", "tokenizer.model", "tokenizer_config.json",
    "vocab.json", "sentencepiece.bpe.model",
)


def _has_hf_model_artifacts(p: Path) -> bool:
    """True when an HF directory holds real model files, not just a config.json."""
    if (p / "adapter_config.json").exists():   # LoRA / adapter directory
        return True
    if any(next(p.glob(pat), None) is not None for pat in _HF_WEIGHT_GLOBS):
        return True
    return any((p / t).exists() for t in _HF_TOKENIZER_FILES)


def _is_hf_dir(path: str) -> bool:
    """True when path is a directory that looks like a HuggingFace model."""
    p = Path(path)
    return (
        p.is_dir()
        and (p / "config.json").exists()
        and _has_hf_model_artifacts(p)
    )


def _is_gguf(path: str) -> bool:
    p = Path(path)
    # Standard GGUF extension OR Ollama blob (sha256-<digest>, no extension)
    return p.suffix.lower() == ".gguf" or (p.is_file() and p.name.startswith("sha256-"))


def _resolve_vram_overhead_bytes(cfg: dict) -> int:
    """``vram_overhead_mb`` (MB) from config, in bytes, or the built-in default on
    a missing/unparseable value. Normal writes (PATCH /v1/config, ``localm
    config``) already enforce a valid int via settings_schema.validate_update, but
    a hand-edited config.json is not type-checked on load (config.py's
    ``load_config`` just merges the stored dict), so a present-but-unparseable
    value is a real misconfiguration rather than the benign missing case and is
    surfaced under --debug instead of bricking every later load."""
    from localm.vram import VRAM_OVERHEAD_BYTES
    raw = cfg.get("vram_overhead_mb")
    if raw is None:
        return VRAM_OVERHEAD_BYTES
    try:
        return int(raw) * 1024 ** 2
    except (TypeError, ValueError):
        from localm.debuglog import logger as _dbg
        _dbg.warning("vram_overhead_mb is set but not a valid number (%r); "
                     "using the default %.1f GB", raw, VRAM_OVERHEAD_BYTES / 1024 ** 3)
        return VRAM_OVERHEAD_BYTES


def _resolve_mtp_draft_tokens(cfg: dict, override: Optional[int]) -> int:
    """Draft tokens per MTP step: *override* when given, else the
    ``mtp_draft_tokens`` config key, else the default; clamped to 1..max.

    A hand-edited value that is not a number is logged under --debug and the
    default is used."""
    from localm.inference.backends.llamacpp.llama import (
        MTP_DRAFT_TOKENS_DEFAULT,
        MTP_DRAFT_TOKENS_MAX,
    )
    raw = override if override is not None else cfg.get("mtp_draft_tokens")
    if raw is None:
        return MTP_DRAFT_TOKENS_DEFAULT
    try:
        value = int(raw)
    except (TypeError, ValueError):
        from localm.debuglog import logger as _dbg
        _dbg.warning("mtp_draft_tokens is set but not a valid number (%r); "
                     "using the default %d", raw, MTP_DRAFT_TOKENS_DEFAULT)
        return MTP_DRAFT_TOKENS_DEFAULT
    return max(1, min(value, MTP_DRAFT_TOKENS_MAX))


def resolve_spec_source(cfg: dict, mtp_enabled: Optional[bool] = None,
                        spec_source: Optional[str] = None) -> str:
    """The draft source a load uses, one of ``off``, ``mtp``, ``ngram``.

    In order: *spec_source* when given; then *mtp_enabled* when given (True
    means ``mtp``, False ``off``); then the ``spec_source`` config key; then the
    ``mtp_enabled`` config key. A config value that is not a source name is
    logged under --debug and the ``mtp_enabled`` key decides. An invalid
    *spec_source* argument raises ValueError."""
    from localm.inference.backends.llamacpp._drafting import SPEC_MTP, SPEC_OFF
    from localm.inference.backends.llamacpp._drafting import resolve_spec_source as _resolve
    if spec_source is not None and spec_source != "":
        return _resolve(spec_source, False)
    if mtp_enabled is not None:
        return SPEC_MTP if mtp_enabled else SPEC_OFF
    cfg_mtp = bool(cfg.get("mtp_enabled", False))
    try:
        return _resolve(cfg.get("spec_source"), cfg_mtp)
    except ValueError:
        from localm.debuglog import logger as _dbg
        _dbg.warning("spec_source is set but not a valid source (%r); "
                     "following mtp_enabled", cfg.get("spec_source"))
        return _resolve(None, cfg_mtp)


def _resolve_spec_draft_tokens(cfg: dict, override: Optional[int]) -> Optional[int]:
    """Draft tokens per n-gram step: *override* when given, else the
    ``spec_draft_tokens`` config key, else None (the source's default). A value
    that is not a number is logged under --debug and None is used."""
    raw = override if override is not None else cfg.get("spec_draft_tokens")
    if raw is None or raw == "":
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        from localm.debuglog import logger as _dbg
        _dbg.warning("spec_draft_tokens is set but not a valid number (%r); "
                     "using the default", raw)
        return None


def resolve_spec_draft_model(cfg: dict, override: Optional[str] = None) -> Optional[str]:
    """The draft GGUF path for the draft source: *override* when given, else
    the ``spec_draft_model`` config key, as a registered model name or a path
    (``get_operator_model_info``). None when unset; a name that resolves to
    nothing is returned as given, so the load reports the draft model missing."""
    raw = override if override is not None else cfg.get("spec_draft_model")
    name = str(raw or "").strip()
    if not name:
        return None
    from localm.model_manager.registry import get_operator_model_info
    info = get_operator_model_info(name)
    return str(info[0]) if info is not None else name


def create_backend(
    model_path: str,
    *,
    mmproj_path: Optional[str] = None,
    n_ctx: Optional[int] = None,
    n_gpu_layers: Optional[int] = None,
    device: Optional[str] = None,
    mtp_enabled: Optional[bool] = None,
    mtp_draft_tokens: Optional[int] = None,
    spec_source: Optional[str] = None,
    spec_draft_tokens: Optional[int] = None,
    spec_draft_model: Optional[str] = None,
) -> BaseBackend:
    """
    Return the appropriate backend for the given model path, without loading it.

    model_path:   HF model directory  →  HFBackend
                  *.gguf file         →  GgufBackend

    mtp_enabled:  None reads the ``mtp_enabled`` config key; True or False
                  overrides it for this backend only, leaving the stored
                  setting untouched.
    mtp_draft_tokens: None reads the ``mtp_draft_tokens`` config key; an int
                  overrides it the same way.
    spec_source:  None resolves the draft source from *mtp_enabled* and the
                  config (see resolve_spec_source); a source name overrides
                  both for this backend only.
    spec_draft_tokens: None reads the ``spec_draft_tokens`` config key; an int
                  overrides it the same way.
    spec_draft_model: None reads the ``spec_draft_model`` config key; a name
                  or path overrides it the same way.
    """
    cfg = load_config()

    if _is_hf_dir(model_path):
        from localm.inference.backends.hf import HFBackend
        return HFBackend(model_path, device=device)

    if _is_gguf(model_path):
        from localm.inference.backends.gguf import GgufBackend
        source = resolve_spec_source(cfg, mtp_enabled, spec_source)
        return GgufBackend(
            model_path,
            mmproj_path=mmproj_path,
            n_ctx=n_ctx or cfg["n_ctx"],
            n_gpu_layers=n_gpu_layers if n_gpu_layers is not None else cfg["n_gpu_layers"],
            n_ctx_max=cfg.get("n_ctx_max", 16384),
            n_ctx_grow=cfg.get("n_ctx_grow", 4096),
            ctx_auto=bool(cfg.get("ctx_auto", False)),
            n_gpu_layers_auto=bool(cfg.get("n_gpu_layers_auto", True)),
            n_cpu_moe=int(cfg.get("n_cpu_moe", 0) or 0),
            use_mmap=resolve_use_mmap(cfg),
            mtp_enabled=source == "mtp",
            spec_source=source,
            spec_draft_tokens=_resolve_spec_draft_tokens(cfg, spec_draft_tokens),
            spec_draft_model=(resolve_spec_draft_model(cfg, spec_draft_model)
                              if source == "draft" else None),
            mtp_draft_tokens=_resolve_mtp_draft_tokens(cfg, mtp_draft_tokens),
            vram_overhead_bytes=_resolve_vram_overhead_bytes(cfg),
        )

    raise ValueError(
        f"Cannot determine backend for: {model_path}\n"
        "Expected a HuggingFace model directory (contains config.json) "
        "or a .gguf file."
    )


# Shape a model's self-declared name must match to be echoed back to API/GUI
# callers, as an allowlist. Anything else (control characters, newlines, markup,
# a whole absolute path, a paragraph) falls back to the directory name.
_SANE_DISPLAY_NAME = re.compile(r"^[\w.-]+(?:/[\w.-]+)*$")
_MAX_DISPLAY_NAME = 96


def _sane_display_name(value) -> Optional[str]:
    """*value* if it is a plausible model name, else None."""
    if not isinstance(value, str):
        return None
    v = value.strip()
    if not v or len(v) > _MAX_DISPLAY_NAME:
        return None
    return v if _SANE_DISPLAY_NAME.match(v) else None


def model_display_name(model_path: str) -> str:
    """Human-readable model name from path.

    For an HF directory this prefers the model's own ``_name_or_path``, but only
    after checking its shape: the directory may be one an untrusted caller named,
    and this string is handed straight back to API and GUI callers. A value that
    does not look like a model name is dropped in favour of the directory name.
    """
    p = Path(model_path)
    if p.is_dir():
        cfg_file = p / "config.json"
        if cfg_file.exists():
            try:
                cfg = json.loads(cfg_file.read_text())
                declared = _sane_display_name(cfg.get("_name_or_path"))
                if declared:
                    return declared
                if cfg.get("_name_or_path"):
                    logger.debug(
                        "model_display_name: ignoring implausible _name_or_path in "
                        "%s; using the directory name instead", cfg_file)
            except Exception as exc:
                # config.json is optional for the display name only; load()
                # validates it later. Falls back to the directory name, and logs
                # the corrupt config so it is discoverable under --debug.
                logger.debug("model_display_name: unreadable config.json at %s: %s", cfg_file, exc)
        return p.name
    return p.stem


def describe_gpu_placement(placement: dict) -> str:
    """One phrase naming an ``Engine.gpu_placement`` dict's placement:
    ``"<offloaded>/<total> layers on the GPU"``, followed by ``", the rest on
    the CPU"`` when fewer than all landed there and by ``", routed experts of
    <n> layers in system RAM"`` when it carries ``moe_cpu_layers``."""
    offloaded = placement.get("gpu_layers_offloaded")
    total = placement.get("gpu_layers_total")
    text = f"{offloaded}/{total} layers on the GPU"
    if isinstance(offloaded, int) and isinstance(total, int) and offloaded < total:
        text += ", the rest on the CPU"
    moe = placement.get("moe_cpu_layers")
    if isinstance(moe, int) and moe > 0:
        text += f", routed experts of {moe} layers in system RAM"
    return text


class Engine:
    """
    High-level wrapper: loads a backend and streams chat completions.

    Usage:
        engine = Engine(model_path)
        engine.load()
        for tok in engine.chat_stream(messages):
            print(tok, end="", flush=True)
        engine.unload()
    """

    def __init__(
        self,
        model_path: str,
        *,
        mmproj_path: Optional[str] = None,
        n_ctx: Optional[int] = None,
        n_gpu_layers: Optional[int] = None,
        device: Optional[str] = None,
        display_name: Optional[str] = None,
        mtp_enabled: Optional[bool] = None,
        mtp_draft_tokens: Optional[int] = None,
        spec_source: Optional[str] = None,
        spec_draft_tokens: Optional[int] = None,
        spec_draft_model: Optional[str] = None,
    ) -> None:
        self.model_path = model_path
        self.display_name = display_name or model_display_name(model_path)
        # Every Engine shares the one process-global load lock, so loads
        # serialise across the server, jobs and embeds, not just within a single
        # Engine instance.
        self._load_lock = _LOAD_LOCK
        self._backend = create_backend(
            model_path,
            mmproj_path=mmproj_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            device=device,
            mtp_enabled=mtp_enabled,
            mtp_draft_tokens=mtp_draft_tokens,
            spec_source=spec_source,
            spec_draft_tokens=spec_draft_tokens,
            spec_draft_model=spec_draft_model,
        )
        self.active_requests = 0
        # Set by http_server.switch_engine after a load placed partly on the
        # CPU: a PlacementHeal naming what held the VRAM, else None.
        self.placement_heal = None
        # Set True by an unload/eviction path for the duration of the native
        # free, so get_engine()/switch_engine()'s fast paths refuse to hand this
        # engine back, and so refuse to let a request pin it, while it is being
        # torn down. A request pins lock-free, after the active_requests==0 check
        # has already passed, so active_requests alone does not close that
        # window.
        self.unloading = False

    @property
    def loaded(self) -> bool:
        return self._backend.loaded

    def set_load_cancel(self, event) -> None:
        """Install a cancel event on the backend so an in-flight load() can be
        aborted mid-flight when a newer model selection supersedes it (preemptive
        switching). Best-effort: backends that cannot abort a partial load ignore
        it. ``None`` clears it."""
        self._backend.set_load_cancel(event)

    def load(self) -> None:
        if not hasattr(self, "_load_lock"):
            self._load_lock = _LOAD_LOCK
        with self._load_lock:
            if self._backend.loaded:
                return
            backend_type = type(self._backend).__name__.replace("Backend", "")
            console.print(
                f"Loading [bold cyan]{self.display_name}[/bold cyan] "
                f"[dim](backend: {backend_type})[/dim]"
            )
            self._attach_registered_adapters()
            for path, scale in getattr(self._backend, "adapters", None) or ():
                console.print(f"[dim]  with LoRA adapter {escape(Path(path).name)} "
                              f"(scale {scale:g})[/dim]")
            self._backend.load()

    def _attach_registered_adapters(self) -> None:
        """Give a GGUF backend the LoRA adapters currently attached to this
        engine's registered model, so a load applies the attachments as they are
        now. Applies only when ``display_name`` is a registered model whose file
        is this engine's ``model_path``. Raises
        :class:`~localm.inference.backends.base.AdapterLoadError` when an
        attached adapter's registry entry is unusable."""
        from localm.inference.backends.gguf import GgufBackend
        backend = self._backend
        if not isinstance(backend, GgufBackend):
            return
        from localm.model_manager import AdapterError, get_model_adapters, get_model_info
        from .backends.base import AdapterLoadError
        try:
            info = get_model_info(self.display_name)
            if info is None or Path(str(info[0])).resolve() != Path(self.model_path).resolve():
                return
            attached = get_model_adapters(self.display_name)
        except AdapterError as exc:
            raise AdapterLoadError(str(exc)) from exc
        backend.adapters = [(os.path.abspath(p), float(s)) for p, s in attached]

    @property
    def applied_adapters(self) -> list:
        """The LoRA adapters the loaded model runs with, as ``{"name", "scale"}``
        (the adapter file's name, never its full path); empty when none is
        applied or the backend does not report any."""
        applied = getattr(self._backend, "applied_adapters", None)
        if not isinstance(applied, (list, tuple)):
            return []
        return [{"name": Path(str(a.get("path", ""))).name, "scale": a.get("scale")}
                for a in applied if isinstance(a, dict)]

    def unload(self) -> None:
        self._backend.unload()

    @property
    def effective_ctx_max(self):
        """Resolved context ceiling of the last load (VRAM-derived when
        ctx_auto is on), or None when unknown / not loaded yet."""
        return getattr(self._backend, "effective_ctx_max", None)

    @property
    def reply_reserve(self) -> Optional[int]:
        """Tokens of the context capacity the loaded model's reply always takes
        (a diffusion model's reply canvas), or None when the reply grows into
        whatever room is left."""
        return getattr(self._backend, "reply_reserve", None)

    @property
    def encoder_decoder(self) -> bool:
        """True when the loaded model encodes its whole prompt in one pass
        before decoding (a T5 GGUF); its context capacity is then that pass's
        prompt limit."""
        return bool(getattr(self._backend, "encoder_decoder", False))

    @property
    def gpu_placement(self) -> Optional[dict]:
        """Where the last load's transformer layers actually ended up: GPU vs
        CPU. ``{"gpu_layers_offloaded": N, "gpu_layers_total": M, "degraded":
        bool}`` when the backend can report it (GgufBackend, once a load has
        completed and the model's true layer count is known), else None -
        e.g. before any load, or for a backend that places layers itself
        without a layer-count knob (HF's device_map="auto").

        ``moe_cpu_layers`` is present when the load kept the routed experts of
        that many layers in system RAM (a Mixture-of-Experts model with
        n_cpu_moe, configured or chosen by auto sizing).

        ``degraded`` is True whenever fewer than the full layer count landed
        on the GPU or any layer's experts stayed in system RAM, whatever the
        reason (VRAM-constrained auto-sizing, or an explicit partial
        n_gpu_layers or n_cpu_moe): a caller of /v1/models/load has no
        visibility into the server's own config either way, so this is
        reported unconditionally rather than only for the auto-sized case, and
        a load response can tell a full GPU load from a silent CPU fallback."""
        offloaded = getattr(self._backend, "gpu_layers_offloaded", None)
        total = getattr(self._backend, "gpu_layers_total", None)
        if offloaded is None or not total:
            return None
        moe = getattr(self._backend, "moe_cpu_layers", 0)
        moe = moe if isinstance(moe, int) and moe > 0 else 0
        placement = {
            "gpu_layers_offloaded": offloaded,
            "gpu_layers_total": total,
            "degraded": offloaded < total or moe > 0,
        }
        if moe:
            placement["moe_cpu_layers"] = moe
        return placement

    def full_offload_vram_bytes(self) -> Optional[int]:
        """Free VRAM the next load needs for every layer to go on the GPU, as
        the backend sizes it (``full_offload_vram_bytes`` of a GGUF backend),
        or None for a backend that does not size its GPU layers, or when the
        backend cannot answer. Blocking: reads the model file and may take a
        GPU reading."""
        size = getattr(self._backend, "full_offload_vram_bytes", None)
        return size() if callable(size) else None

    @property
    def gpu_sizing(self) -> Optional[dict]:
        """How the last load chose its GPU layer count (the backend's
        ``last_gpu_sizing``: ``mode``, ``layers``, ``n_ctx`` and, for an
        auto-sized load, the free/total/model/KV/overhead bytes it was sized
        against), or None for a backend that does not size layers or before
        any load."""
        sizing = getattr(self._backend, "last_gpu_sizing", None)
        return dict(sizing) if isinstance(sizing, dict) else None

    @property
    def mmap_state(self) -> Optional[dict]:
        """What the last load did with memory-mapping, or None when the backend
        did not report it: ``use_mmap`` (the setting: auto, on or off), ``mmap``
        (bool, whether the model was memory-mapped) and ``mmap_from_disk`` (bool,
        True when ``auto`` mapped it because the model may not fit in available
        RAM). ``mmap_note`` is the one-line description
        (``mmap_setting.describe_mmap``), present only when there is something
        to say."""
        effective = getattr(self._backend, "effective_use_mmap", None)
        if effective is None:
            return None
        setting = getattr(self._backend, "use_mmap", "auto")
        forced = (setting == "auto" and bool(effective) and bool(
            getattr(self._backend, "mmap_forced_by_ram", False)))
        state = {"use_mmap": setting, "mmap": bool(effective),
                 "mmap_from_disk": forced}
        note = describe_mmap(setting, bool(effective), forced)
        if note:
            state["mmap_note"] = note
        return state

    @property
    def last_finish_reason(self) -> str:
        """Why the most recent generation ended: "stop" (model finished) or
        "length" (the max_tokens budget ran out). Backends that cannot tell
        report "stop"."""
        return getattr(self._backend, "last_finish_reason", "stop")

    @property
    def supports_images(self) -> bool:
        """True when the active backend can actually see image input. For HF
        this is only accurate once the model is loaded (see can_be_multimodal)."""
        return getattr(self._backend, "supports_images", False)

    @property
    def can_be_multimodal(self) -> bool:
        """True when the backend class could support images, so it is worth
        loading the model to find out. False for text-only backends (GGUF)."""
        return getattr(self._backend, "can_be_multimodal", False)

    @property
    def supports_mtp(self) -> bool:
        """True when the loaded model has active Multi-Token Prediction (MTP) heads."""
        return getattr(self._backend, "supports_mtp", False)

    def mtp_usage(self) -> Optional[dict]:
        """MTP figures for the reply that just finished (see GgufBackend.last_mtp_usage),
        or None when the backend reports none."""
        usage = getattr(self._backend, "last_mtp_usage", None)
        return usage if isinstance(usage, dict) else None

    def draft_model_on_gpu(self) -> Optional[bool]:
        """Where the loaded backend placed its draft model: True on the GPU,
        False on the CPU, None when the backend has no draft placement."""
        placed = getattr(self._backend, "draft_model_on_gpu", None)
        return placed if isinstance(placed, bool) else None

    def draft_step_costs(self) -> Optional[dict]:
        """The step costs the loaded backend measured for its n-gram or
        draft-model source (``StepCosts.report()``) plus ``observed_ms``, the
        corrected step milliseconds of each draft length seen so far ({} when
        none), or None when it measured none."""
        rep = getattr(self._backend, "last_speculation", None)
        if not isinstance(rep, dict):
            return None
        costs = rep.get("costs")
        if not isinstance(costs, dict):
            return None
        observed = rep.get("observed_ms")
        return {**costs, "observed_ms": observed if isinstance(observed, dict) else {}}

    def speculation_usage(self) -> Optional[dict]:
        """Speculative-drafting figures for the reply that just finished, for
        any draft source (see GgufBackend.last_speculation_usage), or None when
        the backend reports none."""
        usage = getattr(self._backend, "last_speculation_usage", None)
        return usage if isinstance(usage, dict) else None

    def count_tokens(self, text: str) -> int:
        """
        Return the number of tokens in *text* using the loaded backend's
        tokenizer.  Falls back to a chars-÷-4 heuristic when the model is
        not yet loaded.
        """
        return self._backend.count_tokens(text)

    def count_messages_tokens(self, messages: list[dict]) -> int:
        """
        Return the number of tokens in a list of structured messages,
        including chat template formatting.
        """
        return self._backend.count_messages_tokens(messages)

    def context_capacity(self) -> Optional[int]:
        """Maximum token capacity of the loaded model's context window.

        Prefers the RESOLVED ceiling from the last load (VRAM-derived under
        ctx_auto), then the configured ceiling, then the base window. Returns None
        only when nothing is loaded or resolvable."""
        b = self._backend
        eff = getattr(b, "effective_ctx_max", None)
        if isinstance(eff, int) and eff > 0:
            return eff
        for attr in ("n_ctx_max", "_n_ctx_max", "n_ctx", "_n_ctx"):
            v = getattr(b, attr, None)
            if isinstance(v, int) and v > 0:
                return v
        llm = getattr(b, "_llm", None)
        try:
            n = llm.n_ctx() if llm is not None else None
            return int(n) if n else None
        except Exception:
            return None

    def _embed_via_dedicated(self, texts: list[str]) -> list[list[float]]:
        """Embed with the small DEDICATED on-device embedding model
        (:mod:`localm.inference.embedder`), or raise with the one command that
        fixes it. Raises rather than returning the chat model's own vectors: RAG
        catches this and degrades to lexical-only BM25 with a warning, which beats
        blending unusable vectors into its 50/50 lexical+vector score."""
        from localm.inference.embedder import embed_texts
        vecs = embed_texts(list(texts))
        if vecs is not None:
            return vecs
        raise NotImplementedError(
            "No embedding model available. Run 'localm setup-embeddings' (or "
            "set net_mode=allow) to enable semantic search; memory and RAG use "
            "lexical BM25 until then.")

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return embedding vectors for a list of texts.

        A backend that can genuinely embed its own loaded model (a HuggingFace
        encoder / sentence-transformer) is used directly. Anything that cannot is
        served by a small DEDICATED on-device embedding model
        (:mod:`localm.inference.embedder`). Two backends cannot, for different
        reasons and at different times:

        - the bundled GGUF chat backend NEVER can (``can_embed = False``, a fixed
          class attribute: the ctypes binding exposes no create_embedding), which
          is known up front, so the large chat model is not loaded just to fail;
        - the HF backend can only tell once the weights are LOADED, because it
          depends on what the checkpoint is: an encoder can embed, a chat decoder
          cannot (see ``HFBackend.can_embed``).

        So the capability is checked TWICE, before AND after the load. An HF chat
        decoder reports the "unknown" True while unloaded, and checking only
        before the load would send it straight to ``backend.embed()``, returning
        mean-pooled chat vectors that cannot separate related from unrelated text
        to both /v1/embeddings and RAG. Raises ``NotImplementedError`` only when
        no embedding path is available at all.
        """
        if getattr(self._backend, "can_embed", True) is False:
            return self._embed_via_dedicated(texts)
        if not self._backend.loaded:
            with _LOAD_LOCK:
                if not self._backend.loaded:
                    self._backend.load()
        if getattr(self._backend, "can_embed", True) is False:
            # Only knowable now (HF): the loaded checkpoint is a chat decoder,
            # so the dedicated embedder is substituted, and the substitution is
            # logged rather than left invisible.
            logger.debug(
                "%s is not an embedding model; embedding via the dedicated "
                "on-device embedder instead", self.display_name)
            return self._embed_via_dedicated(texts)
        return self._backend.embed(texts)

    @property
    def supports_grammar(self) -> bool:
        """True when the active backend can actually constrain generation to a
        grammar. See ``BaseBackend.supports_grammar`` for why the default denies."""
        return getattr(self._backend, "supports_grammar", False)

    def unsupported_sampling(self, names) -> list:
        """The sampling options in *names* (``min_p``, ``presence_penalty``,
        ``frequency_penalty``) the active backend cannot apply."""
        return self._backend.unsupported_sampling(names)

    def validate_grammar(self, grammar: Optional[str], *, lazy: bool = False) -> None:
        """Up-front grammar validation, delegated to the backend.

        Raises :class:`GrammarUnsupportedError` when the backend cannot apply a
        grammar at all - or, with *lazy*, cannot apply it LAZILY - and
        :class:`InvalidGrammarError` when it can but this grammar will not parse.
        Either way the request path turns it into a clean 4xx instead of
        generating text that silently ignores the constraint.

        *lazy* is forwarded rather than interpreted here: only the backend knows
        whether it can honour trigger-gated enforcement, and only some backends
        can answer that honestly at all (see ``BaseBackend.validate_grammar``).

        Called unconditionally, never probed with ``getattr``: the method lives on
        ``BaseBackend`` and denies by default, so a backend with no grammar
        support refuses rather than skipping validation."""
        self._backend.validate_grammar(grammar, lazy=lazy)

    def chat_stream(
        self,
        messages: list[dict],
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repeat_penalty: Optional[float] = None,
        grammar: Optional[str] = None,
        grammar_lazy: bool = False,
        grammar_triggers: Optional[list[str]] = None,
        seed: Optional[int] = None,
        on_status: Optional[Callable[[str], None]] = None,
        thinking: Optional[bool] = None,
        min_p: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
    ) -> Iterator[str]:
        """Stream the reply to *messages*. ``thinking=False`` asks a reasoning
        model to answer without its reasoning channel; ``None`` leaves the
        model's default. *min_p*, *presence_penalty* and *frequency_penalty* are
        passed to the backend only when set; check :meth:`unsupported_sampling`
        first. The other parameters default to the config values."""
        # Auto-reload if the model was unloaded. Holds the process-global load
        # lock so a reload cannot race another load onto the GPU, and
        # double-checks inside the lock so a model another thread just brought
        # back is not reloaded.
        if not self._backend.loaded:
            with _LOAD_LOCK:
                if not self._backend.loaded:
                    if on_status is not None:
                        on_status(LOADING_MODEL_STATUS)
                    console.print(
                        f"[dim]Reloading [bold]{self.display_name}[/bold]…[/dim]"
                    )
                    self._attach_registered_adapters()
                    self._backend.load()

        cfg = load_config()
        extra: dict = {"thinking": thinking} if thinking is not None else {}
        for key, value in (("min_p", min_p), ("presence_penalty", presence_penalty),
                           ("frequency_penalty", frequency_penalty)):
            if value is not None:
                extra[key] = value
        # Normalise model-internal control markers (harmony and Gemma channel
        # tags, and similar) once here, so every backend inherits it. The GGUF
        # backend also scrubs internally and scrub_stream is idempotent; the HF
        # backend relies on this pass alone.
        return scrub_stream(self._backend.chat_stream(
            messages,
            max_tokens=max_tokens if max_tokens is not None else cfg["max_tokens"],
            temperature=temperature if temperature is not None else cfg["temperature"],
            top_p=top_p if top_p is not None else cfg["top_p"],
            top_k=top_k if top_k is not None else cfg["top_k"],
            repeat_penalty=repeat_penalty if repeat_penalty is not None else cfg["repeat_penalty"],
            grammar=grammar,
            grammar_lazy=grammar_lazy,
            grammar_triggers=grammar_triggers,
            seed=seed,
            on_status=on_status,
            **extra,
        ))

    def __enter__(self) -> Engine:
        self.load()
        return self

    def __exit__(self, *_) -> None:
        self.unload()
