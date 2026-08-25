# SPDX-License-Identifier: AGPL-3.0-or-later
"""In-app model discovery: search HuggingFace for GGUF models and judge, per quantization, whether a file fits this machine's VRAM."""

from __future__ import annotations

import ctypes
import inspect
import re
import threading
import time
from collections.abc import Sequence
from typing import Optional

from localm.debuglog import logger

HF_API = "https://huggingface.co"
_TIMEOUT = 20

# HuggingFace library-tag filter for each discoverable model format. GGUF repos
# carry the "gguf" tag; transformers-native (safetensors / pytorch) repos carry
# the "transformers" library tag, which is exactly the set localm's HF backend
# loads. Both are real HF /api/models filter values, so classification comes from
# WHICH query a repo answered, not from parsing per-result tag fields the list
# response may omit.
_FORMAT_FILTER = {"gguf": "gguf", "hf": "transformers"}

# Type-scoped search narrows on TWO orthogonal, independently-selectable axes -
# model TYPE (llm/embedding/diffusion/lora/vae/text-encoder/unknown) and file
# FORMAT (gguf vs safetensors). Both are surfaced as explicit checkboxes in the
# GUI (a type is never inferred silently from the active tab). Every filter value
# below is LIVE-VERIFIED against the real HF /api/models API (not assumed): the
# list endpoint accepts expand[]=pipeline_tag,library_name,tags directly (no
# per-repo fetch to classify a result), and both `pipeline_tag=` and repeated
# `filter=` (ANDed) work as query params.
#
# The "hf" (non-gguf / safetensors) side of the format axis narrows PER TYPE
# where HF exposes a reliable signal:
_HF_TYPE_FILTER = {
    "llm": {"filter": "transformers"},
    "embedding": {"pipeline_tag": "feature-extraction"},
    "diffusion-unet": {"pipeline_tag": "text-to-image"},
    "lora": {"filter": "peft"},
}
# vae / text-encoder / unknown carry NO reliable type signal on HF - the single
# most-used real-world repo for each has none at all (stabilityai/sd-vae-ft-mse
# has no "vae" tag or pipeline_tag; comfyanonymous/flux_text_encoders has no tag
# beyond a license marker). A hard TYPE filter would systematically exclude
# exactly the repos users search for. But the FORMAT axis is still reliable for
# them: filter=safetensors returns the canonical diffusers VAEs/encoders, and
# filter=gguf the GGUF ones (both verified live). So these types narrow by format
# only; their type comes from the result badge (classify_hf_metadata) and, at
# pull time, from the checkbox the user searched under.
_HF_TYPE_FILTER_DEFAULT = {"filter": "safetensors"}
_ALL_SEARCHABLE_TYPES = frozenset(
    {"llm", "embedding", "diffusion-unet", "lora", "vae", "text-encoder", "unknown"})
# Sentinel model_type for the "all types selected" broad search: the widest
# reliable format filter (gguf / safetensors), badged, so nothing is excluded.
_ANY_TYPE = "__any__"

# Single-sourced from localm.vram, the same overhead (KV cache + compute buffers)
# and weight safety factor GgufBackend._check_vram / sysstats.estimate_vram use,
# so the fit badge and the loader agree on "does it fit". Kept as module-level
# names so the value has one home while fit_label still reads a local constant.
from localm.vram import VRAM_OVERHEAD_BYTES as _OVERHEAD_BYTES
from localm.vram import VRAM_WEIGHT_FACTOR as _WEIGHT_FACTOR

# Single-sourced from model_manager.gguf, the same verified llama.cpp
# encoder/embedding architecture allowlist gguf_embedding_signal() uses on a
# freshly-downloaded file's own header - so a search-time badge and a
# post-download registration never disagree about the SAME architecture value.
# No cycle: gguf.py (and the model_manager package it lives in) never imports
# discover, only the reverse (verified live before wiring this in).
from localm.model_manager.gguf import _GGUF_EMBEDDING_ARCHITECTURES

# Quantization label inside a GGUF filename, e.g. Q4_K_M, Q8_0, IQ4_XS,
# Q6_K, F16, BF16, MXFP4_MOE, TQ1_0. Matched case-insensitively on word-ish
# boundaries.
_QUANT_RE = re.compile(
    r"(?i)(?<![A-Z0-9])(IQ\d+_[A-Z0-9]+|Q\d+_K(?:_[SML])?|Q\d+_\d+|TQ[12]_0"
    r"|MXFP4(?:_MOE)?|BF16|F16|F32|FP16|FP32)(?![A-Z0-9])")

# Split GGUF naming: model-00001-of-00003.gguf
_SPLIT_RE = re.compile(r"^(?P<stem>.+)-(?P<part>\d{5})-of-(?P<total>\d{5})\.gguf$",
                       re.IGNORECASE)


class DiscoverError(Exception):
    """Discovery failed - network off, HF unreachable, or repo unusable."""


def _ensure_online() -> None:
    from localm.netpolicy import network_mode
    if network_mode() == "off":
        raise DiscoverError(
            "Network access is disabled (net_mode=off). Enable it with: "
            "localm config net_mode ask")


def _get(url: str, params: Optional[dict] = None) -> object:
    """Policy-checked GET returning parsed JSON."""
    import json as _json
    import urllib.parse

    from localm import netpolicy
    # doseq=True so a list-valued param (expand[]=safetensors&expand[]=downloads
    # ...) encodes as repeated keys, which is how the HF models API takes expand.
    full = url + ("?" + urllib.parse.urlencode(params, doseq=True) if params else "")
    try:
        _final, _ctype, body = netpolicy.safe_fetch_bytes(
            full, max_bytes=32 * 1024 * 1024, timeout=int(_TIMEOUT))
        return _json.loads(body.decode("utf-8"))
    except Exception as e:
        raise DiscoverError(f"HuggingFace request failed: {e}")


# Tag-set fallback for LLM / embedding-ness, consulted alongside pipeline_tag
# (never instead of it): a HF-repacked GGUF-only upload routinely carries no
# pipeline_tag at all (that field belongs to the ORIGINAL checkpoint's model
# card, which a pure-GGUF quantizer repo often never fills in) while still
# carrying the base model's standard HF tags. Exact tokens only, same
# substring-safety rule as the tag checks above.
_LLM_TAGS = frozenset({"conversational", "text-generation", "text2text-generation"})
_EMBEDDING_TAGS = frozenset({"feature-extraction", "sentence-similarity"})


def classify_hf_metadata(pipeline_tag: Optional[str], library_name: Optional[str],
                          tags, architecture: Optional[str] = None) -> str:
    """Classify a model_manager.registry MODEL_TYPES value from HARD HF metadata (pipeline_tag, library_name, exact tag tokens, GGUF architecture) - no network, pure function."""
    tag = pipeline_tag
    library = (library_name or "").strip().lower()
    # Exact, lowercased tag tokens - a set so membership is equality, not
    # substring containment.
    tagset = {str(t).strip().lower() for t in (tags or []) if isinstance(t, str)}
    arch = str(architecture).strip().lower() if architecture else ""

    if "vae" in tagset:
        return "vae"
    if "lora" in tagset or library == "peft":
        return "lora"
    if {"text-encoder", "clip"} & tagset:
        return "text-encoder"
    if arch in _GGUF_EMBEDDING_ARCHITECTURES:
        return "embedding"
    # Media / diffusion signal, checked after the exact tag tokens above (a
    # LoRA/VAE repo commonly also carries its base model's diffusion pipeline_tag).
    if tag in ("text-to-image", "image-to-image", "text-to-audio", "audio-to-audio"):
        return "diffusion-unet"
    if tag in ("feature-extraction", "sentence-similarity") or tagset & _EMBEDDING_TAGS:
        return "embedding"
    # image-text-to-text: a vision-language model is still an LLM (a chat model
    # that additionally accepts image input), never a diffusion pipeline - HF
    # uses this pipeline_tag for VLM chat checkpoints (e.g. a Qwen-VL GGUF).
    if (tag in ("text-generation", "text2text-generation", "conversational",
                "image-text-to-text")
            or tagset & _LLM_TAGS):
        return "llm"
    return "unknown"


def _hf_pipeline_tag_to_type(repo_id: str) -> str:
    """Classify a HuggingFace repo's model type by fetching its metadata and running it through classify_hf_metadata()."""
    try:
        data = _get(f"{HF_API}/api/models/{repo_id}", {"full": "false"})
        if isinstance(data, dict):
            return classify_hf_metadata(
                data.get("pipeline_tag"), data.get("library_name"), data.get("tags", []))
    except Exception as e:
        logger.debug("HF pipeline tag query failed for %s: %s", repo_id, e)
        return "unknown"
    return "unknown"


def hf_param_bytes(safetensors: Optional[dict]) -> Optional[int]:
    """Estimated GPU weight footprint in bytes for an HF model, from its safetensors param metadata (the ``safetensors`` expand field of the HF models API: ``{'total': <param count>, 'parameters': {...}}``)."""
    if not isinstance(safetensors, dict):
        return None
    total = safetensors.get("total")
    if not isinstance(total, int) or isinstance(total, bool) or total <= 0:
        return None
    return total * 2


# Name-based MoE fallback for when the header signal (architecture containing
# "moe") is absent or - per the Mixtral counter-example below - the header lies
# by omission: TheBloke/Mixtral-8x7B-v0.1-GGUF (a real MoE model, live-verified)
# reports gguf.architecture == "llama", an older GGUF conversion predating
# llama.cpp's dedicated mixtral arch tag. Matches the two common MoE naming
# conventions: "8x7B"/"8x22B" style (Mixtral) and "A3B"/"A22B" active-param
# style (Qwen3's MoE line), plus a bare "moe" token. Inherently incomplete (a
# DeepSeek-style repo carries neither convention in its name) - that is why
# this is only ever the FALLBACK behind the header signal, and why callers must
# label a match from this pattern as inferred, never as confirmed.
_MOE_NAME_RE = re.compile(r"(?i)\bmoe\b|\b\d+x\d+b\b|\ba\d+b\b")


def _moe_signal(architecture: Optional[str], repo_id: str) -> Optional[str]:
    """MoE-ness for a search-result row: ``'confirmed'`` (the model's own ``architecture`` string says so - reliable, see the module note above), ``'likely'`` (name pattern only - a guess, must be labelled as such in the GUI), or ``None`` (no evidence either way)."""
    if architecture and "moe" in str(architecture).lower():
        return "confirmed"
    if _MOE_NAME_RE.search(repo_id):
        return "likely"
    return None


def _param_count(row_fmt: str, gguf_meta: object, safetensors_meta: object) -> Optional[int]:
    """Total parameter count for a classified row, or None when unavailable."""
    src = gguf_meta if row_fmt == "gguf" else safetensors_meta
    if not isinstance(src, dict):
        return None
    total = src.get("total")
    if not isinstance(total, int) or isinstance(total, bool) or total <= 0:
        return None
    return total


def _rows_from_items(data: object, limit: int, *, fmt: Optional[str],
                      classify: bool) -> list[dict]:
    """Build result rows from a raw HF /api/models list response."""
    if not isinstance(data, list):
        raise DiscoverError("Unexpected response from HuggingFace search")
    out = []
    for item in data[:limit]:
        repo = item.get("id") or item.get("modelId")
        if not repo:
            continue
        raw_tags = item.get("tags") or []
        row_fmt = fmt if fmt else ("gguf" if "gguf" in raw_tags else "hf")
        row = {
            "id": repo,
            "downloads": item.get("downloads", 0),
            "likes": item.get("likes", 0),
            "updated": item.get("lastModified", ""),
            "formats": [row_fmt],
        }
        if row_fmt == "hf":
            # bf16 weight footprint from the param count, or None when HF has no
            # safetensors metadata (the row then shows "size unknown").
            row["size_bytes"] = hf_param_bytes(item.get("safetensors"))
        if classify:
            # gguf.architecture (gguf-format results) wins when present - the
            # model's own header; config.model_type (hf-format results, when
            # HF has a config.json) is the fallback so both format branches
            # get an architecture-based classification attempt. isinstance-
            # guarded like hf_param_bytes' safetensors check above: a
            # malformed/adversarial API response returning a truthy non-dict
            # for either expand field must degrade this ONE row's signal to
            # None, not crash the whole hf_search() call for every row.
            gguf_meta = item.get("gguf")
            config_meta = item.get("config")
            architecture = (
                (gguf_meta.get("architecture") if isinstance(gguf_meta, dict) else None)
                or (config_meta.get("model_type") if isinstance(config_meta, dict) else None))
            row["detected_type"] = classify_hf_metadata(
                item.get("pipeline_tag"), item.get("library_name"), raw_tags,
                architecture)
            # Display-only "what is this model" fields - see _moe_signal /
            # _param_count docstrings for the reliability contract each carries.
            row["architecture"] = architecture or None
            row["moe"] = _moe_signal(architecture, repo)
            row["param_count"] = _param_count(
                row_fmt, gguf_meta, item.get("safetensors"))
        out.append(row)
    return out


def _type_fmt_filter(model_type: Optional[str], fmt: str) -> dict:
    """HF query narrowing (a params fragment: ``filter=`` and/or ``pipeline_tag=``) for one (model_type, format) pair."""
    if model_type is None:
        return {"filter": _FORMAT_FILTER[fmt]}
    if fmt == "gguf":
        if model_type == "diffusion-unet":
            return {"filter": ["gguf", "diffusers"]}
        return {"filter": "gguf"}
    if model_type == _ANY_TYPE:
        return {"filter": "safetensors"}
    return _HF_TYPE_FILTER.get(model_type, _HF_TYPE_FILTER_DEFAULT)


def _run_query(query: str, limit: int, fmt: str, model_type: Optional[str],
                classify: bool) -> list[dict]:
    """One HF /api/models query for a single (format, type), rows tagged *fmt*."""
    params: dict = {"sort": "downloads", "direction": "-1", "limit": str(limit)}
    if query.strip():
        params["search"] = query.strip()
    params.update(_type_fmt_filter(model_type, fmt))
    expand: list[str] = []
    if fmt == "hf":
        # Expand the safetensors param metadata so each result carries a param
        # count we can turn into a VRAM fit estimate inline (no per-repo tree
        # fetch). `expand` is restrictive - it drops the default stat fields - so
        # re-request downloads/likes/lastModified alongside it.
        expand += ["safetensors", "downloads", "likes", "lastModified"]
    elif classify:
        # gguf never requests safetensors, so without this the stats vanish
        # too: `expand` is restrictive (drops every default field once ANY
        # field is requested), and classify below always requests at least
        # pipeline_tag - measured live: a classified gguf query that requested
        # only pipeline_tag/library_name/tags silently dropped downloads AND
        # likes from every row (both default-present with no expand at all).
        expand += ["downloads", "likes", "lastModified"]
    if classify:
        expand += ["pipeline_tag", "library_name", "tags", "config"]
        if fmt == "gguf":
            # The real llama.cpp architecture, read from the GGUF header
            # itself (see classify_hf_metadata) - only meaningful for
            # gguf-format results; an hf/safetensors repo has no gguf
            # metadata to expand.
            expand += ["gguf"]
    if expand:
        params["expand[]"] = expand
    data = _get(f"{HF_API}/api/models", params)
    return _rows_from_items(data, limit, fmt=fmt, classify=classify)


def _spec_key(model_type: Optional[str], fmt: str):
    """Hashable identity of the HF request a (type, fmt) pair resolves to, so two selected types that produce the SAME query (e.g. vae + text-encoder both -> filter=safetensors on the hf side) fire ONE call, not two."""
    frag = _type_fmt_filter(model_type, fmt)
    return (fmt, tuple(sorted(
        (k, tuple(v) if isinstance(v, list) else v) for k, v in frag.items())))


def hf_search(query: str = "", limit: int = 20, formats: Sequence[str] = ("gguf",),
              model_type: Optional[str] = None,
              model_types: Optional[Sequence[str]] = None) -> list[dict]:
    """Search HF for model repos."""
    _ensure_online()
    limit = max(1, min(int(limit), 50))

    # Resolve the requested type set. model_types (GUI) wins; else the singular
    # model_type (back-compat); else None = legacy broad search (no type scoping,
    # no classify) so the CLI/MCP path is untouched.
    if model_types is not None:
        types: Optional[list[str]] = [t for t in model_types if t in _ALL_SEARCHABLE_TYPES]
        if not types:
            raise DiscoverError("No valid model type requested for search.")
    elif model_type is not None:
        if model_type not in _ALL_SEARCHABLE_TYPES:
            raise DiscoverError(f"Unknown model type for search: {model_type}")
        types = [model_type]
    else:
        types = None

    fmts = [f for f in formats if f in _FORMAT_FILTER]
    if not fmts:
        raise DiscoverError(
            "No valid model format requested (choose gguf and/or hf).")

    classify = types is not None
    if types is None:
        query_types: list[Optional[str]] = [None]
    elif set(types) == _ALL_SEARCHABLE_TYPES:
        # Everything selected: the widest reliable format filter, not N*fmts
        # near-duplicate calls. Still classified, so results are badged.
        query_types = [_ANY_TYPE]
    else:
        query_types = list(types)

    # Build the (type, fmt) query list, collapsing pairs that resolve to the
    # SAME HF request so a multi-type selection never fires duplicate calls.
    seen_specs: set = set()
    query_specs: list[tuple] = []
    for mt in query_types:
        for fmt in fmts:
            key = _spec_key(mt, fmt)
            if key in seen_specs:
                continue
            seen_specs.add(key)
            query_specs.append((mt, fmt))

    # One list per query (each already download-sorted by the API), de-duped by
    # repo id: a repo surfacing under several queries stays in the FIRST it
    # appeared in and unions its formats there.
    seen: dict[str, dict] = {}
    per_query: list[list[dict]] = []
    for mt, fmt in query_specs:
        lst: list[dict] = []
        for item in _run_query(query, limit, fmt, mt, classify):
            existing = seen.get(item["id"])
            if existing:
                for f in item["formats"]:
                    if f not in existing["formats"]:
                        existing["formats"].append(f)
                # A repo entering via a gguf query carried no size; keep the hf
                # query's size estimate if that pass has one.
                if existing.get("size_bytes") is None and item.get("size_bytes") is not None:
                    existing["size_bytes"] = item["size_bytes"]
            else:
                seen[item["id"]] = item
                lst.append(item)
        per_query.append(lst)

    # Round-robin interleave by per-query rank, then trim to `limit`. A plain
    # merge-then-sort-by-downloads would let the highest-download query (HF repos
    # routinely dwarf GGUF repacks) crowd the others out of the top `limit`
    # entirely, so a "show GGUF" toggle could return zero GGUF. Interleaving keeps
    # every enabled query visible while still leading each with its most popular.
    out: list[dict] = []
    rank = 0
    while len(out) < limit and any(rank < len(lst) for lst in per_query):
        for lst in per_query:
            if rank < len(lst):
                out.append(lst[rank])
                if len(out) >= limit:
                    break
        rank += 1
    return out


def hf_backend_available() -> bool:
    """True when the HF/transformers runtime can actually RUN a model here: both torch and transformers are importable."""
    import importlib.util
    try:
        return bool(importlib.util.find_spec("torch")
                    and importlib.util.find_spec("transformers"))
    except (ImportError, ValueError):
        # find_spec can raise on a half-installed namespace package; treat an
        # unresolvable probe as "not available" rather than crash discovery.
        return False


def hf_gguf_files(repo: str) -> list[dict]:
    """List the GGUF files of *repo* with size and quant label."""
    _ensure_online()
    repo = repo.strip().strip("/")
    if not re.match(r"^[\w.-]+/[\w.-]+$", repo):
        raise DiscoverError(f"Not a HuggingFace repo id: {repo}")
    tree = _get(f"{HF_API}/api/models/{repo}/tree/main")
    if not isinstance(tree, list):
        raise DiscoverError(f"Unexpected tree response for {repo}")

    singles: list[dict] = []
    groups: dict[tuple, dict] = {}
    for entry in tree:
        path = entry.get("path", "")
        if not path.lower().endswith(".gguf"):
            continue
        size = entry.get("size") or (entry.get("lfs") or {}).get("size") or 0
        m = _SPLIT_RE.match(path)
        if m:
            key = (m.group("stem").lower(), m.group("total"))
            g = groups.setdefault(key, {
                "file": None, "size_bytes": 0, "n_parts": 0,
                "quant": _quant_of(m.group("stem")),
            })
            g["size_bytes"] += size
            g["n_parts"] += 1
            if m.group("part") == "00001":
                g["file"] = path
        else:
            singles.append({
                "file": path,
                "quant": _quant_of(path),
                "size_bytes": size,
                "n_parts": 1,
            })

    files = singles + [g for g in groups.values() if g["file"]]
    if not files:
        raise DiscoverError(
            f"{repo} has no GGUF files. It may be a transformers-format "
            f"repo - pull it whole with:  localm pull {repo}")
    files.sort(key=lambda f: f["size_bytes"])
    return files


def _quant_of(name: str) -> str:
    """The single quant label for *name*, preferring an MXFP4/MXFP4_MOE match over any earlier one in the string."""
    matches = _QUANT_RE.findall(name)
    if not matches:
        return ""
    for m in matches:
        if m.upper().startswith("MXFP4"):
            return m.upper()
    return matches[0].upper()


# ---- GPU probe safety: a hardware probe must never block its caller -------- #
# _list_gpus_probe() calls the GPU driver: torch.cuda.mem_get_info (which, on a
# torch ROCm build, calls into HIP) has NO timeout, and nvidia-smi is a
# subprocess. A busy or wedged driver call would block the CALLER for as long as
# the driver takes. The public list_gpus() below makes the probe safe by
# construction: it runs on a helper thread with a hard DEADLINE; if it overruns,
# the caller gets the last-known-good reading (or []) and moves on. A wedged
# NATIVE call cannot be interrupted from Python, so that one helper thread is
# abandoned; the in-flight guard means at most ONE such thread ever exists, and
# the overrun is surfaced at debug level (AGENTS.md rule 5), never silently
# eaten.
#
# WHAT THE DEADLINE IS FOR (and what it is NOT for). PR #541 diagnosed GUI
# routes running this probe inline on the server's single asyncio loop, which
# froze the whole WebUI while a probe was busy - and fixed that by OFFLOADING
# every server call site to an executor. As of that PR no production caller
# probes ON THE EVENT LOOP (re-verified 2026-07-17: the GUI routes all
# run_in_executor, and the GPU-registry heartbeat's probe - it DOES probe, via
# resolve_main_gpu_index -> list_gpus every ~20s when main_gpu_index >= 1 - is
# likewise executor-offloaded, see http_server's heartbeat loop), so the
# deadline does NOT protect the loop; it only bounds how long one worker
# thread (or a blocking CLI call) waits on a wedged driver before degrading.
#
# That is why the default is COLD-INIT-TOLERANT. The first torch.cuda / HIP
# call of a process initializes the ROCm/CUDA driver: measured 2.6-3.1s on a
# warm system, 4.63s observed on a genuinely cold driver, ~6.5s historically.
# The original 4.0s default sat INSIDE that range, so a legitimate cold init
# "timed out" - and a timeout is served as [] / a frozen last-known-good, which
# a bare-list caller cannot tell apart from "no GPU at all". That one thin
# margin manufactured a whole bug class ("no torch / no GPU" misreports #581,
# a silently skipped pre-load VRAM gate #722). The deadline must sit ABOVE any
# legitimate cold init; 15.0 is the value blocking callers have used since
# #581. The cost on a truly wedged driver is one worker thread parked for 15s
# ONCE - the in-flight guard hands every concurrent caller an instant BUSY,
# and after the overrun the last-known-good path takes over - so nothing
# user-facing ever freezes for it.
#
# NOTE - deliberately NO freshness/TTL cache: every call re-probes. A TTL cache
# would hand a STALE "free" reading to callers that need a live one, most
# critically switch_engine's eviction loop, whose wait_for_vram_release polls
# free-VRAM to confirm a native free has landed before re-checking (AUDIT-MED-11);
# a stale value there would defeat that guard and over-evict. The last-known-good
# value is kept ONLY as the wedge fallback, never to short-circuit a live probe.
_GPU_PROBE_DEADLINE = 15.0    # seconds a probe may block its caller; must exceed
                              # a legitimate COLD driver init (see above)
# Historical alias, kept for the call sites and tests that opt into it by name
# (doctor, `localm gpus`, switch_engine). #541 split a short 4.0s "server" cap
# from this longer blocking-caller deadline; the short cap guarded an event loop
# that (per the same PR) no longer runs probes, while turning every cold driver
# init into a timeout->[]->"no GPU" misreport. Unified 2026-07-17.
_GPU_PROBE_CLI_DEADLINE = _GPU_PROBE_DEADLINE

# Outcome of a probe, surfaced by list_gpus(..., return_status=True) so a caller
# can tell a slow / timed-out probe apart from a genuine "nothing here" reading
# and not misattribute the former (AGENTS.md rule 5). A user-facing "no GPU"
# message MUST branch on this.
GPU_PROBE_OK = "ok"            # a fresh probe completed - an empty list means genuinely none
GPU_PROBE_TIMEOUT = "timeout"  # probe exceeded the deadline (cold driver init / wedge); INCONCLUSIVE
GPU_PROBE_BUSY = "busy"        # another probe is inflight, or the probe thread could not start
# The probe completed WITHIN its deadline (unlike TIMEOUT) but no source could
# conclusively rule out a GPU: the isolated torch enumeration could not be asked
# this round (latched-unavailable, or wedged on this attempt - see
# _torch_gpus_isolated_once) and nvidia-smi, the only other source, also found
# nothing - which proves nothing on an AMD/Intel box nvidia-smi cannot see at
# all. An empty list under this status is INCONCLUSIVE, same as TIMEOUT, and
# for the same reason a caller must not retry-with-a-longer-deadline expecting
# it to help: nvidia-smi's answer will not change no matter how long you wait.
GPU_PROBE_INCONCLUSIVE = "inconclusive"

_gpu_probe_lock = threading.Lock()
_gpu_last_good: Optional[list] = None    # last SUCCESSFUL probe; served on a wedge
_gpu_probe_inflight = False
# Published TOGETHER with _gpu_probe_inflight (under _gpu_probe_lock) when a probe
# thread is started, and cleared when it lands, so a PATIENT off-loop caller can
# JOIN the running probe instead of being handed an instant GPU_PROBE_BUSY. The
# default guard behaviour is still to refuse to pile a second probe on a driver
# already being probed (BUSY) - that instant answer is what keeps the WebUI
# responsive on a permanent wedge. Joining is strictly opt-in (list_gpus'
# wait_for_inflight), for a caller already OFF the event loop that can afford to
# wait out a cold driver init: it is the ONLY thing that lets a generous deadline
# actually help on a cold box, because there the first probe (typically the
# /api/stats heartbeat's) holds the in-flight slot for the whole cold init, so
# every other caller in that window would otherwise short-circuit on BUSY
# without ever probing - the exact 0.0000s no-op a RETRY at any deadline hits.
_gpu_probe_done: Optional[threading.Event] = None
_gpu_probe_result: Optional[dict] = None
# Bumped by _reset_gpu_probe_cache() to ORPHAN any probe thread still in flight.
# An abandoned probe (see the DEADLINE note above) is by definition still running
# and will write its reading whenever the native call finally returns - which can
# be long after the reset. Clearing the globals alone cannot prevent that write,
# so a stale thread's result is fenced out by epoch instead of raced against.
_gpu_probe_epoch = 0


def last_known_gpus() -> list:
    """The most recent SUCCESSFUL :func:`list_gpus` reading, WITHOUT probing."""
    return list(_gpu_last_good or [])


def _reset_gpu_probe_cache() -> None:
    """Test hook: drop the last-known-good GPU reading + in-flight flag, and INVALIDATE any probe still in flight so it cannot bleed into the next test."""
    global _gpu_last_good, _gpu_probe_inflight, _gpu_probe_epoch
    global _gpu_probe_done, _gpu_probe_result, _isolated_torch_unavailable
    global _isolated_torch_broken_warned
    with _gpu_probe_lock:
        _gpu_last_good = None
        _gpu_probe_inflight = False
        # Cleared with the rest of the probe state: a test (or a caller) resetting
        # the cache must get a clean slate, or one test's simulated spawn failure
        # would silently disable the torch path for every later test in the worker.
        _isolated_torch_unavailable = False
        _isolated_torch_broken_warned = False
        # Unpublish the join handles too: after a reset the slot reads free, so no
        # caller should join a probe from the epoch just retired. An abandoned
        # thread still holding its own local done/result is unaffected (it sets its
        # local event and, epoch-mismatched, will not touch these globals again).
        _gpu_probe_done = None
        _gpu_probe_result = None
        _gpu_probe_epoch += 1


def list_gpus(*, deadline: float = _GPU_PROBE_DEADLINE, return_status: bool = False,
              wait_for_inflight: bool = False):
    """Every GPU device visible right now: ``[{'index', 'name', 'total', 'free'}, ...]``, or ``[]`` when nothing is measurable."""
    gpus, status = _list_gpus_with_status(deadline, wait_for_inflight)
    return (gpus, status) if return_status else gpus


def _list_gpus_with_status(deadline: float, wait_for_inflight: bool = False) -> tuple:
    """The real probe driver behind :func:`list_gpus`, returning ``(gpus, status)`` where status is one of :data:`GPU_PROBE_OK` / :data:`GPU_PROBE_TIMEOUT` / :data:`GPU_PROBE_BUSY` / :data:`GPU_PROBE_INCONCLUSIVE`."""
    global _gpu_last_good, _gpu_probe_inflight, _gpu_probe_done, _gpu_probe_result
    global _probe_deadline_at   # published with the slot for #697's cold-budget check
    join_done = None
    join_result = None
    with _gpu_probe_lock:
        if _gpu_probe_inflight:
            if wait_for_inflight and _gpu_probe_done is not None:
                # A patient off-loop caller. Rather than pile a second probe on a
                # driver already being probed (or be handed an instant BUSY for a
                # reading that is on its way), JOIN the in-flight probe: wait on ITS
                # completion event, bounded by our own deadline. Both handles are
                # captured HERE, under the same lock that observed the in-flight
                # slot, so a concurrent completion cannot null them between this
                # check and the wait below.
                join_done = _gpu_probe_done
                join_result = _gpu_probe_result
            else:
                # Default: never pile on. Hand back the last-known-good reading so
                # this caller stays free. No fresh reading was taken, so the status
                # is BUSY (not a clean OK); [] is the safe "unknown" answer when
                # nothing has succeeded yet. This instant answer is what keeps the
                # event loop responsive on a permanent wedge.
                served = list(_gpu_last_good) if _gpu_last_good is not None else []
                return served, GPU_PROBE_BUSY
        else:
            _gpu_probe_inflight = True
            # Published under the SAME lock that claims the in-flight slot (only one
            # probe is ever in flight, so there is only one deadline to describe):
            # #697's probe body reads it to decide whether it can afford a cold
            # device-global VRAM source without overrunning this deadline. See
            # _apply_device_global_free.
            _probe_deadline_at = time.monotonic() + deadline
            # Captured under the SAME lock that claims the in-flight slot: an
            # unlocked read here could pair this probe with an epoch a concurrent
            # reset has already retired, which is the exact race the epoch exists to
            # close.
            my_epoch = _gpu_probe_epoch
            # Created and PUBLISHED under the lock, atomically with the in-flight
            # slot, so any joiner that sees inflight=True also sees these handles
            # (never a half-published state). Cleared by _run when the probe lands.
            result: dict = {}
            done = threading.Event()
            _gpu_probe_done = done
            _gpu_probe_result = result

    # JOIN path: we did not start a probe; wait on the one already running.
    if join_done is not None:
        if join_done.wait(deadline):
            # The joined probe landed. Its result carries a "value" key iff its
            # thread actually ran to completion; the key is ABSENT only when the
            # starting caller could not spawn the thread and woke joiners via
            # done.set() so they would not hang - that is a BUSY, not a fresh OK.
            if "value" in join_result:
                v = join_result["value"]
                status = (GPU_PROBE_OK if join_result.get("conclusive", True)
                         else GPU_PROBE_INCONCLUSIVE)
                return (list(v) if v is not None else []), status
            with _gpu_probe_lock:
                served = list(_gpu_last_good) if _gpu_last_good is not None else []
            return served, GPU_PROBE_BUSY
        # Our own deadline expired while waiting on the in-flight probe: same
        # outcome as starting one that overran - the driver is stuck, serve
        # last-known-good and report TIMEOUT (never mistaken for "no GPU"). We
        # spawned nothing, so this never piled onto the wedge.
        logger.debug("list_gpus: waited %.1fs on an in-flight GPU probe that did "
                     "not complete; returning last-known GPU info", deadline)
        with _gpu_probe_lock:
            served = list(_gpu_last_good) if _gpu_last_good is not None else []
            return served, GPU_PROBE_TIMEOUT

    # START path: we own the in-flight slot.
    def _run() -> None:
        global _gpu_last_good, _gpu_probe_inflight, _gpu_probe_done, _gpu_probe_result
        global _probe_deadline_at
        value = None
        try:
            value = _list_gpus_probe()
        except Exception as e:   # the probe swallows its own errors; belt-and-braces
            logger.debug("list_gpus: probe raised unexpectedly: %s", e)
        with _gpu_probe_lock:
            # CONCLUSIVENESS (GPU_PROBE_INCONCLUSIVE): propagates the distinction
            # _isolated_torch_unavailable already makes rather than inventing a new
            # one - that latch is set ONLY when the isolated torch enumeration
            # proved it could not answer this round (never for an honest "torch
            # answered, zero devices" - see test_a_real_empty_answer_does_NOT_latch),
            # so reading it here is exact, not a heuristic. Gated on an EMPTY value:
            # a non-empty reading came from a source that DID answer (nvidia-smi
            # found real hardware, or torch answered before the latch engaged) and
            # is conclusive regardless of the latch - the actual sm_120 case this
            # isolation exists for is NVIDIA, where nvidia-smi still answers while
            # torch is latched-unavailable. Read under this same lock (not a
            # separate one) because it is the same global _torch_gpus_isolated_once
            # mutates, and this probe thread is the only writer while it runs.
            conclusive = not (not value and _isolated_torch_unavailable)
            if _gpu_probe_epoch != my_epoch:
                # A reset retired this probe while it ran: its reading describes a
                # state the owner has explicitly dropped, and the in-flight slot is
                # no longer ours to clear (a later probe may already own it). Drop
                # BOTH writes rather than corrupt the current epoch's state.
                # Surfaced, not silenced (AGENTS.md rule 5): debug is the right
                # altitude because this is the deliberate consequence of a reset,
                # not a fault.
                logger.debug("list_gpus: discarding probe result from retired "
                             "epoch %s (current %s)", my_epoch, _gpu_probe_epoch)
            else:
                if value is not None:
                    _gpu_last_good = value
                _gpu_probe_inflight = False
                # Unpublish alongside the in-flight slot: a NEW caller must start a
                # fresh probe, not join one that has already landed. A joiner that
                # already captured its local handle is unaffected - it waits on that
                # same `done`, which is set unconditionally just below.
                _gpu_probe_done = None
                _gpu_probe_result = None
                # Cleared with the in-flight slot too (#697): the budget describes
                # THIS probe and nothing else. Leaving it set would hand a later
                # reader an expired deadline, which reads as "no budget left" and
                # would skip a cold source that in fact had all the time in the world.
                _probe_deadline_at = None
        # Deliberately OUTSIDE the epoch gate and unconditional: a caller still
        # inside its deadline - the starter OR any joiner - is waiting on `done`,
        # and withholding it would make it wait out the full deadline and report a
        # COMPLETED probe as a TIMEOUT, manufacturing the very "no GPU"/inconclusive
        # lie the status contract above exists to prevent.
        result["value"] = value
        result["conclusive"] = conclusive
        done.set()

    try:
        threading.Thread(target=_run, name="localm-gpu-probe", daemon=True).start()
    except Exception as e:
        # Could not spawn the probe thread (e.g. OS thread exhaustion). Reset the
        # in-flight guard so a LATER call can retry (never leave it stuck True with
        # no thread to clear it), surface it at debug (rule 5), and degrade to the
        # last-known-good reading rather than propagating a 500 to the caller. No
        # fresh reading was taken -> BUSY. Epoch-gated for the same reason as the
        # clear in _run: if a reset retired us, the slot is no longer ours and may
        # already belong to a newer probe we must not clear.
        with _gpu_probe_lock:
            if _gpu_probe_epoch == my_epoch:
                _gpu_probe_inflight = False
                _gpu_probe_done = None
                _gpu_probe_result = None
        # Wake any caller that joined between our publish above and this failure, so
        # it does not wait out its full deadline on a probe that will never run.
        # `result` has no "value" key (the thread never set it), which the join
        # path reads as BUSY - the honest status here.
        done.set()
        logger.debug("list_gpus: could not start probe thread: %s", e)
        served = list(_gpu_last_good) if _gpu_last_good is not None else []
        return served, GPU_PROBE_BUSY
    if done.wait(deadline):
        # Fresh probe finished in time: return ITS result (never a cached one).
        v = result.get("value")
        status = (GPU_PROBE_OK if result.get("conclusive", True)
                 else GPU_PROBE_INCONCLUSIVE)
        return (list(v) if v is not None else []), status
    # Deadline exceeded: the driver call is stuck in native code and cannot be
    # cancelled. Serve the last-known-good value and let the abandoned thread
    # finish (or never); _gpu_probe_inflight stays True until it does, so a wedge
    # spawns no further threads. Surfaced, not silenced (rule 5). The status is
    # TIMEOUT so a caller does not mistake an inconclusive probe for "no GPU".
    logger.debug("list_gpus: GPU probe exceeded %.1fs deadline (driver call stuck); "
                 "returning last-known GPU info so the caller does not block", deadline)
    with _gpu_probe_lock:
        served = list(_gpu_last_good) if _gpu_last_good is not None else []
        return served, GPU_PROBE_TIMEOUT


def native_hip_runtime_resident() -> bool:
    """True when llama.cpp's bundled HIP-linked runtime is resident IN THIS process on Windows: the native lib has been loaded (``_loader.load_lib``) and the resolved runtime ships a HIP ggml backend (same shipped-DLL-set authority as :func:`_native_backend_has_vulkan`)."""
    import sys
    if sys.platform != "win32":
        return False
    try:
        from localm.inference.backends.llamacpp import _loader
        if not _loader.native_lib_loaded():
            return False
        d = _loader.runtime_binary_dir()
        return d is not None and any(
            "hip" in p.name.lower() for p in d.glob(_loader._ggml_glob()))
    except Exception as e:
        logger.debug("native-HIP-resident check failed (%s); answering False",
                     type(e).__name__)
        return False


def _torch_gpu_probe_known_doomed() -> bool:
    """True when :func:`_list_gpus_probe`'s ``import torch`` attempt below is KNOWN, ahead of time, to fail in this exact process state - so the probe skips it at the root instead of triggering the failure and catching the aftermath."""
    import sys
    if "torch" in sys.modules:
        # A resident torch (imported for real before the runtime loaded, or a
        # test's injected stand-in) makes `import torch` a plain cache hit: no
        # rocm_sdk preload runs, so the conflict cannot occur and the working
        # enumeration must be kept. On the doomed combo torch can never BE
        # resident - the faulted module is evicted on every attempt - so this
        # never defuses the real guard.
        return False
    try:
        if not native_hip_runtime_resident():
            return False
        import importlib.util
        if importlib.util.find_spec("rocm_sdk") is None:
            return False
    except Exception as e:
        logger.debug("list_gpus: torch-conflict detector failed (%s); "
                     "proceeding with the normal torch attempt", type(e).__name__)
        return False
    logger.debug(
        "list_gpus: skipping the torch GPU probe: the bundled HIP llama.cpp "
        "runtime is already loaded in this process and a ROCm (rocm_sdk) torch "
        "is installed, so `import torch` here is a known-doomed DLL-identity "
        "conflict (STATUS_ENTRYPOINT_NOT_FOUND; see "
        "_torch_gpu_probe_known_doomed's docstring); using the non-torch sources")
    return True


# How long the out-of-process torch enumeration may take before it is abandoned
# and the probe falls through to nvidia-smi.
#
# Sized to fit INSIDE _GPU_PROBE_DEADLINE (15.0s) together with the nvidia-smi
# fallback's own timeout=5, and to sit above a legitimate cold driver init
# (measured ~6.5s, see _reset_gpu_probe_cache's docstring; a cold torch import on
# a healthy box is ~2.7s). Getting this wrong in the generous direction is the
# subtle failure: a ceiling ABOVE the caller's deadline means a box whose torch
# wedges never reaches the fallback within the caller's window at all, so every
# probe costs the full deadline and reports TIMEOUT while nvidia-smi, which could
# have answered in milliseconds, is never consulted.
_ISOLATED_TORCH_PROBE_TIMEOUT = 10.0

# Latched True once the out-of-process torch enumeration proves it CANNOT answer
# on this box (spawn failure, timeout, unusable reply). Read/written under
# _gpu_probe_lock, cleared by _reset_gpu_probe_cache.
#
# WHY (this is not a reading cache - see the no-TTL note above, which still
# holds): without it, a box whose torch import wedges pays the full timeout on
# EVERY probe forever, because each probe starts the attempt again from scratch.
# That is the sm_120 case in the report this fix came from, where the import does
# not merely take long, it does not finish. What is remembered here is
# "torch cannot be asked here", never a VRAM number, so nothing stale can reach
# switch_engine's eviction loop - the property AUDIT-MED-11 protects.
_isolated_torch_unavailable = False

# Latched once the isolated probe has been reported BROKEN. Separate from
# _isolated_torch_unavailable because that one disables a capability while this one
# only suppresses a repeated log line: broken isolation deliberately keeps retrying
# (it still enumerates, in-process), so the warning would otherwise repeat on every
# probe - roughly every 2.5s under the live VRAM meter.
_isolated_torch_broken_warned = False


def isolated_torch_unavailable() -> bool:
    """True once the isolated probe has PROVEN, in a child process, that torch cannot finish enumerating on this box (see :func:`_torch_gpus_isolated_once`, which sets the latch this reads)."""
    with _gpu_probe_lock:
        return _isolated_torch_unavailable


# Field evidence (a real user's debug log): the child probe's stderr routinely
# starts with a long virtualenv install-path prefix from Python's own
# warnings.warn() formatting (``<path>:<line>: <Category>: <message>``) - on
# its own, longer than the 200-char cap this used to be truncated to. So the
# message body, the part that is actually actionable (e.g. "The following
# list of GPU architectures compatible with this version of PyTorch..."),
# never survived: measured, the fragment "The following list" appeared with
# nothing after it dozens of times across one session. Raised generously
# rather than truncated from either a fixed front or back: this stderr can
# carry either a warnings.warn() message (the point comes AFTER the
# file:line: Category: prefix) or an uncaught exception's traceback (the
# point is the LAST line), and a truncation direction that helps one shape
# reliably guts the other. A blind cut is also a rule-5 violation regardless
# of direction, so any truncation that still happens beyond this generous a
# limit is marked, never silent.
_CHILD_STDERR_LOG_CAP = 2000


def _capped_stderr(text: str, limit: int = _CHILD_STDERR_LOG_CAP) -> str:
    """*text* (child-probe stderr), capped to *limit* chars for a log line."""
    if len(text) <= limit:
        return text
    return text[:limit] + f"... [truncated, {len(text) - limit} more chars]"


class _IsolatedTorchWedged(Exception):
    """The out-of-process torch probe ran but did not finish in time, i.e. TORCH ITSELF is wedging on this box (the sm_120 case)."""


def _torch_is_resident() -> bool:
    """True when torch is ALREADY imported in this process, so enumerating here is a free ``sys.modules`` cache hit that takes no OS loader lock."""
    import sys
    return "torch" in sys.modules


def _torch_gpus_resident() -> list:
    """torch's device list, read IN THIS PROCESS."""
    import torch
    if not torch.cuda.is_available():
        return []
    out = []
    for i in range(torch.cuda.device_count()):
        try:
            free, total = torch.cuda.mem_get_info(i)
        except Exception:
            continue   # one device failing to report never hides the rest
        try:
            name = torch.cuda.get_device_name(i)
        except Exception:
            name = f"GPU {i}"
        out.append({"index": i, "name": name,
                    "total": int(total), "free": int(free)})
    return out


def _torch_gpus_isolated() -> "Optional[list]":
    """torch's device list read from a CHILD process, for the case where torch is not yet resident and importing it HERE would take the Windows OS loader lock and block thread creation process-wide, stalling the event loop (issue #833; full mechanism and measurements in :mod:`localm._torch_gpu_probe`)."""
    import json
    import subprocess
    from localm._mp_spawn import interpreter_for_localm_children
    try:
        proc = subprocess.run(
            [interpreter_for_localm_children(), "-u", "-m",
             "localm._torch_gpu_probe"],
            capture_output=True, text=True,
            timeout=_ISOLATED_TORCH_PROBE_TIMEOUT)
    except subprocess.TimeoutExpired:
        # Surfaced, not silenced (rule 5): this is the wedged-driver case, and a
        # silent [] here is indistinguishable from "this box has no GPU".
        logger.debug("list_gpus: out-of-process torch probe did not answer "
                     "within %.1fs; falling through to nvidia-smi",
                     _ISOLATED_TORCH_PROBE_TIMEOUT)
        raise _IsolatedTorchWedged() from None
    except Exception as e:
        logger.debug("list_gpus: could not spawn the out-of-process torch probe "
                     "(%s); falling through to nvidia-smi", type(e).__name__)
        return None
    err = (proc.stderr or "").strip()
    raw = (proc.stdout or "").strip()
    if not raw:
        # The child ALWAYS prints one line, "[]" included on its own failure path,
        # so empty stdout means it died before printing (killed, hard crash, a
        # native fault taking the process down). That is COULD NOT ASK, not
        # "torch sees no device" - collapsing the two would report "no GPU" on a
        # box whose GPU torch can see perfectly well.
        logger.debug("list_gpus: out-of-process torch probe printed nothing "
                     "(rc=%s)%s; treating as unavailable, not as 'no device'",
                     proc.returncode,
                     f"; child said: {_capped_stderr(err)}" if err else "")
        return None
    try:
        devices = json.loads(raw)
        if not isinstance(devices, list) or not all(
                isinstance(d, dict) and isinstance(d.get("index"), int)
                and isinstance(d.get("total"), int)
                and isinstance(d.get("free"), int)
                for d in devices):
            raise ValueError("torch probe reply has the wrong shape")
    except Exception as e:
        logger.debug("list_gpus: out-of-process torch probe reply unusable "
                     "(%s)%s; falling through to nvidia-smi", e,
                     f"; child said: {_capped_stderr(err)}" if err else "")
        return None
    if err:
        # The child prints its own failure cause here before answering []. That
        # is the reason the reading is missing, so it must not die with the
        # discarded stream.
        logger.debug("list_gpus: out-of-process torch probe reported: %s",
                     _capped_stderr(err))
    return devices


def _torch_gpus_isolated_once() -> list:
    """:func:`_torch_gpus_isolated`, but never retried on a box that has already proven it cannot answer."""
    global _isolated_torch_unavailable
    with _gpu_probe_lock:
        if _isolated_torch_unavailable:
            return []
    try:
        devices = _torch_gpus_isolated()
    except _IsolatedTorchWedged:
        with _gpu_probe_lock:
            if not _isolated_torch_unavailable:
                _isolated_torch_unavailable = True
                # Said once, not once per probe: the per-attempt reason is already
                # logged by _torch_gpus_isolated, and this line explains why those
                # stop appearing rather than leaving the silence unexplained.
                logger.debug(
                    "list_gpus: torch did not finish enumerating within %.1fs in "
                    "an isolated probe; skipping it for the rest of this process "
                    "and using the non-torch sources",
                    _ISOLATED_TORCH_PROBE_TIMEOUT)
        return []
    if devices is None:
        global _isolated_torch_broken_warned
        with _gpu_probe_lock:
            first = not _isolated_torch_broken_warned
            _isolated_torch_broken_warned = True
        if first:
            # ONCE per process, not once per probe. The live VRAM meter polls
            # /api/stats every 2.5s and each poll drives a probe, so an
            # unconditional warning here would emit ~24 lines a minute for the
            # life of the server - a real defect of its own, and the kind of
            # noise that trains people to ignore the log. The condition is
            # permanent-ish and identical every time, so repeating it adds no
            # information; later occurrences stay at debug.
            logger.warning(
                "list_gpus: could not run the isolated GPU probe, falling back "
                "to importing torch in this process. GPU detection still works; "
                "on Windows this import can briefly stall the server (see "
                "localm._torch_gpu_probe). Please report this - the isolated "
                "probe is meant to work everywhere.")
        else:
            logger.debug("list_gpus: isolated probe still unavailable; using "
                         "the in-process torch import again")
        return _torch_gpus_resident()
    return devices


def _list_gpus_probe() -> list:
    """The actual (blocking) GPU driver probe."""
    if not _torch_gpu_probe_known_doomed():
        try:
            out = _torch_gpus_resident() if _torch_is_resident() \
                else _torch_gpus_isolated_once()
            if out:
                _apply_device_global_free(out)
                return out
        except Exception:
            pass

    try:
        import subprocess
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        if proc.returncode == 0 and proc.stdout.strip():
            out = []
            for line in proc.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 4:
                    continue
                idx_s, name, total_mb, free_mb = parts[0], parts[1], parts[2], parts[3]
                try:
                    out.append({
                        "index": int(idx_s), "name": name,
                        "total": int(total_mb) * 1024 ** 2,
                        "free": int(free_mb) * 1024 ** 2,
                        # nvidia-smi's memory.free is the whole board's, across
                        # every process (that is what it exists to report), so
                        # unlike the torch path above it needs no correction.
                        "free_scope": FREE_SCOPE_DEVICE,
                    })
                except ValueError:
                    continue   # a malformed line never hides the rest
            if out:
                return out
    except Exception:
        pass
    return []


# How much of the world a GPU entry's "free" actually accounts for. A caller that
# presents free VRAM as CURRENT FACT (a "will it fit" refusal, a freed-bytes report)
# must know the difference; a caller that only wants a fit CEILING ("total") does not.
FREE_SCOPE_DEVICE = "device"    # every process's VRAM is counted - the number is the board's
FREE_SCOPE_PROCESS = "process"  # ONLY this process's own allocations are counted (see below)

# Probe budget below which a COLD (not-yet-opened) device-global source is skipped
# rather than risk overrunning the probe deadline. The cold open is MEASURED at
# ~750ms; this is that with margin, since overrunning costs the caller its free
# reading entirely. See _apply_device_global_free.
_CORRECTION_COLD_BUDGET_S = 1.5

# When the in-flight probe's deadline expires (monotonic), or None outside a probe.
# Set by _list_gpus_with_status under the same lock that claims the in-flight slot,
# so the probe body can tell how much of its budget is left before spending ~750ms
# on a cold source. Safe as a module global precisely because _gpu_probe_inflight
# serialises probes: only ever one in flight to describe.
_probe_deadline_at = None


def _apply_device_global_free(gpus: list) -> None:
    """Correct each entry's ``free`` to a DEVICE-GLOBAL figure where this platform's driver query is not one already, and tag every entry with ``free_scope`` so a caller can tell a whole-board number from a process-local one."""
    import sys
    if sys.platform != "win32":
        for g in gpus:
            g["free_scope"] = FREE_SCOPE_DEVICE
        return

    # The scope to use when a device-global correction is NOT available for an entry
    # (source cold-skipped, unmappable, or failed). Tag PROCESS only where the raw
    # reading is KNOWN blind (Windows + an AMD ROCm/HIP torch build); elsewhere on
    # Windows the raw cudaMemGetInfo is device-global by documentation (NVIDIA), so
    # tagging it PROCESS would assert a blindness never measured and raise a spurious
    # uncertainty flag on a number that is actually fine. Computed defensively up
    # front so it is defined on every path below, including the import-failure except.
    try:
        from localm.gpu_usage import raw_reading_is_process_scoped
        uncorrected_scope = (FREE_SCOPE_PROCESS if raw_reading_is_process_scoped()
                             else FREE_SCOPE_DEVICE)
    except Exception:
        # gpu_usage unimportable is a real bug, not an environment condition, but it
        # must not crash a probe. Conservative default: DEVICE - never assert a
        # blindness we cannot confirm.
        uncorrected_scope = FREE_SCOPE_DEVICE

    try:
        from localm.gpu_usage import device_global_used_bytes, source_is_warm
        # This runs INSIDE the deadline-bounded probe, so it spends the SAME budget
        # the driver call already spent. Opening the source costs ~750ms ONCE per
        # process (a driver init); a warm read costs ~0.02ms. Measured under the
        # old 4.0s default, that cold 750ms pushed cold probes from a comfortable
        # 2.9-3.5s to 3.6-4.0s and started timing them out - and a timeout costs
        # the caller its free reading ENTIRELY (list_gpus serves [] and vram_info
        # falls to the registry tier, which has no "free" at all). A correct
        # number is not worth trading for no number, so a COLD source is skipped
        # when the remaining budget is too thin to absorb it; the reading is then
        # tagged with the uncorrected scope instead of silently uncorrected. The
        # cold-tolerant default deadline has room for it on the first go, so this
        # guard now matters only to callers that pass a deliberately short
        # deadline; a warm source is free and always runs.
        if not source_is_warm():
            remaining = None
            if _probe_deadline_at is not None:
                remaining = _probe_deadline_at - time.monotonic()
            if remaining is not None and remaining < _CORRECTION_COLD_BUDGET_S:
                logger.debug(
                    "list_gpus: %.2fs left of the probe budget is too thin for a "
                    "cold device-global source (~%.1fs); leaving this reading "
                    "%s rather than risking a timeout that would return no free VRAM "
                    "at all", remaining, _CORRECTION_COLD_BUDGET_S, uncorrected_scope)
                for g in gpus:
                    g["free_scope"] = uncorrected_scope
                return
        used = device_global_used_bytes(gpus)
    except Exception as e:
        # Surfaced, not silenced: the entries below are then tagged with the
        # uncorrected scope (PROCESS only where the raw reading is known blind), so a
        # real blindness is reported without over-claiming one where it is not.
        logger.debug("list_gpus: device-global VRAM source failed: %s", e)
        used = {}
    for g in gpus:
        u = used.get(g.get("index"))
        if u is None:
            g["free_scope"] = uncorrected_scope
            continue
        total = int(g["total"])
        # Clamp: the used figure and `total` come from different sources (the driver's
        # total vs the adapter's dedicated usage), so their difference can land just
        # outside [0, total] without either being wrong enough to matter.
        g["free"] = max(0, min(total, total - int(u)))
        g["free_scope"] = FREE_SCOPE_DEVICE


def _native_backend_has_vulkan() -> bool:
    """True when the currently-resolved native runtime directory ships the Vulkan ggml backend (a ``ggml-vulkan.*`` file) - i.e. the active install is the ``vulkan`` build."""
    try:
        from localm.inference.backends.llamacpp._loader import (
            runtime_binary_dir, _ggml_glob,
        )
        d = runtime_binary_dir()
        if d is None:
            return False
        return any("vulkan" in p.name.lower() for p in d.glob(_ggml_glob()))
    except Exception:
        return False


def _llama_visible_devices(devices: list) -> list:
    """The subset of a native non-CPU device inventory that llama.cpp will actually place layers on, RENUMBERED into the index space ``mp.main_gpu`` and ``mp.tensor_split`` consume - i.e. what a configured ``main_gpu_index`` / ``gpu_split_indices`` has to be expressed in to name the card the user meant."""
    from localm.inference.backends.llamacpp._loader import GGML_DEV_TYPE_GPU
    gpus = [d for d in devices
            if isinstance(d, dict) and d.get("type") == GGML_DEV_TYPE_GPU]
    if not gpus:
        return list(devices)
    return [{**d, "index": i} for i, d in enumerate(gpus)]


def native_gpu_devices() -> Optional[list]:
    """Selector-shaped devices from the ACTIVE native runtime's OWN registry, read crash-isolated (the probe daemon - ``_loader.gpu_devices_isolated``): ``[{'index', 'name', 'total'?, 'free'?}, ...]``, or ``None`` when the daemon/registry cannot answer this call."""
    from localm.inference.backends.llamacpp import _loader
    raw = _loader.gpu_devices_isolated()
    if raw is None:
        return None
    out = []
    for d in raw:
        try:
            entry = {"index": int(d["index"])}
        except (KeyError, TypeError, ValueError):
            continue   # defensive: gpu_devices_isolated already shape-checked
        name = str(d.get("description") or "").strip() or str(d.get("name") or "").strip()
        entry["name"] = name or f"device {entry['index']}"
        for key in ("total", "free"):
            v = d.get(key)
            if isinstance(v, int) and v > 0:
                entry[key] = v
        # ggml_backend_dev_type, passed through so a caller can tell a DISCRETE
        # GPU from an integrated one or an accelerator. llama.cpp treats them
        # very differently when placing layers - it skips ACCEL entirely and
        # uses an iGPU ONLY when no discrete GPU exists - so a caller summing
        # capacity across "the devices this load will spread over" must be able
        # to filter. Absent when the probe did not report one; a caller that
        # NEEDS the distinction must then decline rather than assume.
        t = d.get("type")
        if isinstance(t, int):
            entry["type"] = t
        out.append(entry)
    return _llama_visible_devices(out)


def resolve_main_gpu_index(configured, *, gpus: Optional[list] = None) -> int:
    """The GPU device index to actually use, given the user's ``main_gpu_index`` config value."""
    if configured is None:
        return 0
    try:
        idx = int(configured)
    except (TypeError, ValueError):
        logger.warning("main_gpu_index=%r is not a valid integer; using device 0",
                       configured)
        return 0
    if idx < 0:
        logger.warning("main_gpu_index=%d is negative; using device 0", idx)
        return 0
    if idx > _MAX_GPU_SPLIT_INDEX:
        logger.warning(
            "main_gpu_index=%d is above the sanity ceiling (%d); using device 0",
            idx, _MAX_GPU_SPLIT_INDEX)
        return 0
    if idx == 0:
        return 0   # the native default anyway - no need to enumerate devices
    if gpus is None:
        gpus = list_gpus()
    # Check membership by the "index" field, NOT list position: a device that
    # fails to report (list_gpus() skips it rather than hide the rest) leaves a
    # gap, so "idx < len(gpus)" alone could wrongly wave through an idx that
    # does not actually correspond to any detected device. Skipped entirely
    # when the active backend is vulkan (GPU-SPLIT-VKINDEX): list_gpus() is
    # blind to Vulkan-only devices, so a non-empty result here does not mean
    # it is authoritative for THIS backend's index space.
    if gpus and not _native_backend_has_vulkan() and not any(
            g.get("index") == idx for g in gpus):
        logger.warning(
            "main_gpu_index=%d does not match any of the %d GPU(s) detected "
            "right now; falling back to device 0", idx, len(gpus))
        return 0
    return idx


def apply_main_gpu(mp, *, config: Optional[dict] = None) -> None:
    """Set ``mp.main_gpu`` from the configured ``main_gpu_index``, validated via :func:`resolve_main_gpu_index`."""
    from localm.config import load_config
    cfg = config if config is not None else load_config()
    configured = cfg.get("main_gpu_index")
    if configured is None:
        return
    mp.main_gpu = resolve_main_gpu_index(configured)


# llama.cpp's LLAMA_SPLIT_MODE_LAYER (see llamacpp/_structs.py / _abi.py's
# split_mode notes: 0=NONE/single-GPU, 1=LAYER, 2=ROW, 3=TENSOR). LAYER splits
# whole layers across devices proportional to tensor_split - the right default
# for "spread a too-big model over N cards" (as opposed to ROW/TENSOR, which
# split individual tensors and generally need fast inter-GPU interconnect to
# be worthwhile).
_LLAMA_SPLIT_MODE_LAYER = 1

# Fallback tensor_split array capacity when llama_max_devices() cannot be
# probed (an older build without the symbol). Matches LLAMA_MAX_DEVICES from
# the pre-dynamic-backend-registry era this build's own _structs.py docstring
# says it predates. tensor_split is a raw `const float*` with no length of its
# own, so under-allocating would be a genuine out-of-bounds read - this is a
# best-effort safety net, not a verified value (no multi-GPU hardware or
# provisioned native runtime was available to confirm it against the actual
# bundled build; see apply_gpu_split).
_TENSOR_SPLIT_FALLBACK_CAPACITY = 16

# Sanity ceiling for a gpu_split_indices entry - no real machine has anywhere
# near this many GPU devices, so an index above it is a config error, never a
# legitimate one. Bounds the ctypes tensor_split allocation apply_gpu_split
# eventually drives: without this, [0, 500000] would attempt a 500,001-element
# allocation before the native loader is ever invoked. settings_schema.py's
# MAX_GPU_SPLIT_INDEX applies the same value at config WRITE time; this is the
# independent check at READ time, so a hand-edited config.json that bypasses
# schema validation entirely is still bounded here.
#
# Also used by resolve_main_gpu_index below to bound a single main_gpu_index:
# the same sanity reasoning applies to one device index as to a list of them,
# and both values reach the identical ctypes.c_int32 main_gpu field.
_MAX_GPU_SPLIT_INDEX = 127


def resolve_gpu_split(configured_indices, configured_ratios=None, *,
                       gpus: Optional[list] = None) -> list:
    """Validate a configured multi-GPU split (``gpu_split_indices`` / ``gpu_split_ratios``) against the devices ``list_gpus()`` (or the injected *gpus*, for tests) currently sees, returning ``[(index, ratio), ...]`` ready to write into ``tensor_split``."""
    if not configured_indices:
        return []
    try:
        raw_indices = [int(i) for i in configured_indices]
    except (TypeError, ValueError):
        logger.warning(
            "gpu_split_indices=%r is not a list of integers; ignoring the "
            "split (single-GPU behavior)", configured_indices)
        return []
    if any(i < 0 for i in raw_indices):
        logger.warning(
            "gpu_split_indices=%r contains a negative index; ignoring the "
            "split (single-GPU behavior)", configured_indices)
        return []
    if any(i > _MAX_GPU_SPLIT_INDEX for i in raw_indices):
        logger.warning(
            "gpu_split_indices=%r contains an index above the sanity ceiling "
            "(%d); ignoring the split (single-GPU behavior)",
            configured_indices, _MAX_GPU_SPLIT_INDEX)
        return []

    seen: set = set()
    deduped: list = []
    for i in raw_indices:
        if i not in seen:
            seen.add(i)
            deduped.append(i)

    if gpus is None:
        gpus = list_gpus()
    if gpus and not _native_backend_has_vulkan():
        known = {g.get("index") for g in gpus}
        valid = [i for i in deduped if i in known]
        dropped = [i for i in deduped if i not in known]
        if dropped:
            logger.warning(
                "gpu_split_indices contains %d device(s) not currently "
                "detected (%s); dropping them from the split",
                len(dropped), dropped)
    else:
        # Detection unmeasurable (no torch, no nvidia-smi) OR the active
        # native backend is vulkan (GPU-SPLIT-VKINDEX - list_gpus() cannot see
        # Vulkan-only devices, so a non-empty result here would not be
        # authoritative for this backend's index space): same documented
        # boundary as resolve_main_gpu_index - cannot cross-check either way,
        # so the configured indices pass through rather than discarding an
        # explicit user choice we have no way to disprove.
        valid = deduped

    if len(valid) < 2:
        return []

    ratios: Optional[list] = None
    if configured_ratios:
        try:
            raw_ratios = [float(r) for r in configured_ratios]
        except (TypeError, ValueError):
            raw_ratios = None
        if raw_ratios is not None and any(r <= 0 for r in raw_ratios):
            raw_ratios = None
        if raw_ratios is not None and len(raw_ratios) == len(raw_indices):
            # Re-pair by ORIGINAL position so a ratio still lines up with its
            # index even when another index was dropped/de-duped above.
            by_index = dict(zip(raw_indices, raw_ratios))
            ratios = [by_index[i] for i in valid]
        else:
            logger.warning(
                "gpu_split_ratios (%d entries) does not match gpu_split_indices "
                "(%d entries); falling back to an equal split",
                len(configured_ratios), len(raw_indices))

    if ratios is None:
        ratios = [1.0] * len(valid)

    return list(zip(valid, ratios))


def resolve_auto_split_ratios(config: Optional[dict] = None, *,
                              gpus: Optional[list] = None,
                              wait_for_inflight: bool = False) -> Optional[list]:
    """Free-VRAM-proportional split ratios for the configured ``gpu_split_indices``, or ``None`` when automatic distribution does not apply - the parent-side decision behind 'query free vram from each card, compare and distribute' (the auto-split feature request)."""
    from localm.config import load_config
    cfg = config if config is not None else load_config()
    indices = cfg.get("gpu_split_indices")
    if not indices or cfg.get("gpu_split_ratios"):
        return None
    try:
        idx_list = [int(i) for i in indices]
    except (TypeError, ValueError):
        # resolve_gpu_split itself warns and drops the split for this case -
        # there will be no split to distribute, so stay silent here.
        return None
    if len(idx_list) < 2:
        return None

    def _fallback(reason: str) -> None:
        logger.info(
            "auto GPU split: cannot distribute by free VRAM (%s); "
            "falling back to the equal split", reason)
        return None

    frees: list = []
    if _native_backend_has_vulkan():
        # GPU-SPLIT-VKINDEX: the configured indices live in ggml-vulkan's own
        # index space, so only the native registry's reading can be paired
        # with them; a list_gpus() (torch-space) reading here would compute
        # shares for the WRONG cards.
        devices = native_gpu_devices()
        if devices is None:
            return _fallback("the native device registry did not answer")
        by_index = {d.get("index"): d for d in devices}
        for i in idx_list:
            d = by_index.get(i)
            if not isinstance(d, dict):
                # ABSENT, which is a different problem from UNMEASURABLE and
                # needs different words. These devices are llama.cpp's own
                # list (integrated GPUs and accelerators already removed, the
                # rest renumbered - see _llama_visible_devices), so a
                # configured index can legitimately point past the end:
                # typically a split saved before that filtering existed, on a
                # box whose raw registry had more entries than the loader
                # keeps. Calling that "reported no free-VRAM figure" sends a
                # reader hunting a driver fault instead of a stale setting.
                return _fallback(
                    f"device {i} is not one of the {len(devices)} device(s) "
                    "this load will actually use")
            free = d.get("free")
            if not isinstance(free, int):
                return _fallback(
                    f"device {i} reported no free-VRAM figure")
            frees.append(free)
    else:
        if gpus is None:
            # wait_for_inflight (load-path callers pass True): a concurrent
            # probe (the GUI's 2.5s stats heartbeat) holding the slot would
            # otherwise hand this an instant BUSY + stale reading, silently
            # degrading the load to the equal split on exactly the asymmetric
            # box auto exists for. Joining is safe: every probing caller here
            # is off the event loop (executor / CLI thread).
            gpus, status = _list_gpus_reading(wait_for_inflight=wait_for_inflight)
            if status != GPU_PROBE_OK:
                return _fallback(
                    f"no fresh per-device VRAM reading (probe status {status})")
        by_index = {g.get("index"): g for g in gpus}
        for i in idx_list:
            g = by_index.get(i)
            free = g.get("free") if isinstance(g, dict) else None
            if not isinstance(free, int):
                return _fallback(
                    f"device {i} is not detected or reported no free VRAM")
            # Device-global or nothing: see the TRUSTWORTHINESS section of
            # this function's docstring for why a PROPORTIONAL split cannot
            # accept a PROCESS-scoped (or untagged) reading the way
            # gpu_split_shortfall's refuse-only gate does. Real list_gpus()
            # output always carries this tag (_apply_device_global_free sets
            # it on every entry, every platform) - a missing tag here means a
            # synthetic/test double, not a production reading, and is
            # rejected the same as an explicitly PROCESS-scoped one rather
            # than silently assumed safe.
            if g.get("free_scope") != FREE_SCOPE_DEVICE:
                return _fallback(
                    f"device {i}'s free-VRAM reading is not device-global "
                    f"(free_scope={g.get('free_scope')!r})")
            frees.append(free)

    floored = [max(f, 1) for f in frees]
    total = sum(floored)
    ratios = [f / total for f in floored]
    logger.info(
        "auto GPU split: distributing by free VRAM - %s",
        ", ".join(
            f"device {i}: {r * 100:.0f}% ({f / 1024 ** 3:.1f} GB free)"
            for i, r, f in zip(idx_list, ratios, frees)))
    return ratios


def _tensor_split_capacity(min_len: int) -> int:
    """Float-slot count to allocate for ``tensor_split``: the native loader's own answer when available (authoritative - see the capacity comment above), else the documented fallback."""
    try:
        from localm.inference.backends.llamacpp import _api
        if _api.has_max_devices():
            return max(_api.llama_max_devices(), min_len)
    except Exception as e:
        logger.debug(
            "llama_max_devices() probe failed (%s); using the fallback "
            "tensor_split capacity", type(e).__name__)
    return max(_TENSOR_SPLIT_FALLBACK_CAPACITY, min_len)


def apply_gpu_split(mp, *, config: Optional[dict] = None,
                    ratios_override: Optional[list] = None):
    """Set ``mp.split_mode``/``mp.tensor_split`` from the configured ``gpu_split_indices``/``gpu_split_ratios``, validated via :func:`resolve_gpu_split`."""
    from localm.config import load_config
    cfg = config if config is not None else load_config()
    ratios = ratios_override if ratios_override else cfg.get("gpu_split_ratios")
    pairs = resolve_gpu_split(cfg.get("gpu_split_indices"), ratios)
    if len(pairs) < 2:
        return None

    capacity = _tensor_split_capacity(max(idx for idx, _ in pairs) + 1)
    arr = (ctypes.c_float * capacity)()
    for idx, ratio in pairs:
        arr[idx] = ratio

    mp.tensor_split = ctypes.cast(arr, ctypes.c_void_p)
    mp.split_mode = _LLAMA_SPLIT_MODE_LAYER

    split_indices = {idx for idx, _ in pairs}
    if mp.main_gpu not in split_indices:
        new_main = pairs[0][0]
        logger.warning(
            "main_gpu_index=%d is not one of the configured gpu_split_indices "
            "%s; using device %d (the first split device) as the primary "
            "instead", mp.main_gpu, sorted(split_indices), new_main)
        mp.main_gpu = new_main

    return arr


def _list_gpus_kw(*, deadline: Optional[float] = None, return_status: bool = False,
                  wait_for_inflight: bool = False):
    """Call :func:`list_gpus` passing ONLY the kwargs the caller actually asked for."""
    kw = {}
    if deadline is not None:
        kw["deadline"] = deadline
    if return_status:
        kw["return_status"] = True
    if wait_for_inflight:
        kw["wait_for_inflight"] = True
    return list_gpus(**kw)


def vram_info(*, return_status: bool = False, deadline: Optional[float] = None,
              wait_for_inflight: bool = False):
    """{'total': bytes, 'free'?: bytes} for the CONFIGURED main GPU device (see main_gpu_index / resolve_main_gpu_index), or the largest GPU when none is configured, or {} when not measurable."""
    from localm.config import load_config
    if return_status:
        gpus, status = _list_gpus_kw(deadline=deadline, return_status=True,
                                     wait_for_inflight=wait_for_inflight)
    else:
        gpus = _list_gpus_kw(deadline=deadline, wait_for_inflight=wait_for_inflight)
        status = None   # unused: _ret() never reads it when return_status=False

    def _ret(info: dict):
        return (info, status) if return_status else info

    if gpus:
        configured = load_config().get("main_gpu_index")
        idx = resolve_main_gpu_index(configured, gpus=gpus)
        # Look up by the "index" field, not list position (list_gpus() can
        # have a gap when one device fails to report - see
        # resolve_main_gpu_index); gpus[0] is a defensive fallback that should
        # not be reachable since resolve_main_gpu_index already validated idx.
        g = next((x for x in gpus if x.get("index") == idx), gpus[0])
        out = {"total": g["total"]}
        if g.get("free") is not None:
            out["free"] = g["free"]
            # Travels WITH the number it describes: a caller presenting free VRAM as
            # current fact must be able to tell a whole-board figure from a
            # process-local one (see _apply_device_global_free). Absent when free is.
            if g.get("free_scope") is not None:
                out["free_scope"] = g["free_scope"]
        return _ret(out)

    import sys
    if sys.platform == "win32":
        try:
            import winreg
            best = 0
            best_desc = ""
            base = (r"SYSTEM\CurrentControlSet\Control\Class"
                    r"\{4d36e968-e325-11ce-bfc1-08002be10318}")
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, base) as root:
                i = 0
                while True:
                    try:
                        sub = winreg.EnumKey(root, i)
                    except OSError:
                        break
                    i += 1
                    if not sub.isdigit():
                        continue
                    try:
                        with winreg.OpenKey(root, sub) as key:
                            val, _typ = winreg.QueryValueEx(
                                key, "HardwareInformation.qwMemorySize")
                            if isinstance(val, int) and val > best:
                                best = val   # largest adapter wins (skip iGPU)
                                # The adapter's human name lives in the SAME key; it
                                # is what lets the device-global lookup below authorise
                                # an AMD single-adapter pairing by vendor (see
                                # gpu_usage._gpu_is_amd). Absent on odd drivers -> "".
                                try:
                                    desc, _dt = winreg.QueryValueEx(key, "DriverDesc")
                                    best_desc = str(desc or "")
                                except OSError:
                                    best_desc = ""
                    except OSError as e:
                        # Unexpected (vs the EnumKey end-of-list break above):
                        # access denied or a removed key. Surface under --debug
                        # so incomplete VRAM detection is diagnosable; the silent
                        # fallback is deliberate (a note beats crashing fit badges).
                        logger.debug("vram_info: registry subkey %s unreadable: %s",
                                     sub, e)
                        continue
            if best:
                out = {"total": int(best)}
                # The registry gives total but NO free. Torch-less builds land here
                # for EVERY VRAM query (list_gpus() is empty - no torch to enumerate,
                # and nvidia-smi is NVIDIA-only), so without this the meter and every
                # fit/admission gate see total-only forever on a GGUF-only install.
                # Recover a DEVICE-GLOBAL free from the ADL/PDH usage source, which
                # works torch-less and in-process (ADL for AMD, PDH's WDDM counter as
                # the vendor-neutral fallback - the same source _apply_device_global_free
                # uses). It maps ONLY when unambiguous (exactly one AMD adapter for an
                # AMD-named GPU, or exactly one WDDM instance); a non-AMD or multi-
                # adapter box declines and we keep total-only rather than guess a
                # pairing. The synthetic index 0 never feeds GPU SELECTION (this tier
                # is single-adapter by design, see the docstring) - it only carries the
                # name so the AMD pairing can be authorised.
                #
                # ONLY when the probe COMPLETED empty (torch-less: it returns [] fast,
                # status OK) - never when it TIMED OUT, was BUSY, or was INCONCLUSIVE.
                # A timeout means the driver is wedged/cold and the box is
                # unmeasurable; the pre-load gate deliberately treats that as "skip
                # the VRAM check", and surfacing an independent ADL number there
                # would silently turn a skipped gate into an enforcing one (and could
                # act on a reading taken while the driver is in a bad state).
                # INCONCLUSIVE (the isolated torch probe could not be asked and
                # nvidia-smi also found nothing) gets the same conservative treatment:
                # gpu_usage.device_global_used_bytes' ADL/PDH mapping itself partially
                # depends on torch's pci_bus_id as one of its strategies, so it is not
                # proven independent of the same trouble - the honest degrade is
                # total-only, exactly as for TIMEOUT/BUSY, not a fresh claim of
                # certainty this call has not earned. status is None only when the
                # caller did not ask for it (return_status=False fit-badges), which
                # never gates on any of these.
                if status not in (GPU_PROBE_TIMEOUT, GPU_PROBE_BUSY, GPU_PROBE_INCONCLUSIVE):
                    try:
                        from localm import gpu_usage
                        entry = {"index": 0, "name": best_desc, "total": int(best)}
                        u = gpu_usage.device_global_used_bytes([entry]).get(0)
                        if u is not None:
                            out["free"] = max(0, min(int(best), int(best) - int(u)))
                            out["free_scope"] = FREE_SCOPE_DEVICE
                    except Exception as e:
                        # Best-effort enrichment: total-only is the honest fallback, so
                        # a failed lookup degrades to it rather than losing the total.
                        logger.debug("vram_info: device-global free lookup failed: %s", e)
                return _ret(out)
        except Exception:
            pass
    return _ret({})


def vram_capacity(config: Optional[dict] = None, *, return_status: bool = False,
                  deadline: Optional[float] = None, wait_for_inflight: bool = False,
                  combined_only: bool = False):
    """{'total': bytes, 'free'?: bytes} to weigh a model's fit against - the right ceiling for any 'will this model fit' decision (a pre-load refusal gate, a fit badge, a VRAM-estimate readout)."""
    from localm.config import load_config
    cfg = config if config is not None else load_config()

    def _vi():
        # Forward ONLY the opt-in kwargs the caller supplied, so the call stays
        # byte-identical to a bare vram_info() for the no-kwarg vram_info() doubles
        # (~11 test modules patch it; same reason as _list_gpus_kw).
        kw = {}
        if return_status:
            kw["return_status"] = True
        if deadline is not None:
            kw["deadline"] = deadline
        if wait_for_inflight:
            kw["wait_for_inflight"] = True
        return vram_info(**kw)

    # Cheap short-circuit for the common (no split configured) case, mirroring
    # resolve_gpu_split's own early return - skips a real hardware probe
    # (list_gpus() -> torch/nvidia-smi) on every request for the vast majority
    # of single-GPU installs that never configured a split. Under combined_only
    # this is a conclusive, probe-free "no combined figure exists" - reported
    # with GPU_PROBE_OK, same as gpu_split_shortfall's no-split return: a
    # deterministic routing answer, not an inconclusive reading.
    if not cfg.get("gpu_split_indices"):
        if combined_only:
            return ({}, GPU_PROBE_OK) if return_status else {}
        return _vi()

    if return_status:
        gpus, status = _list_gpus_kw(deadline=deadline, return_status=True,
                                     wait_for_inflight=wait_for_inflight)
    else:
        gpus = _list_gpus_kw(deadline=deadline, wait_for_inflight=wait_for_inflight)
        status = None

    def _ret(info: dict):
        return (info, status) if return_status else info

    pairs = resolve_gpu_split(
        cfg.get("gpu_split_indices"), cfg.get("gpu_split_ratios"), gpus=gpus)
    by_index = {g.get("index"): g for g in gpus}
    split_gpus = [by_index[idx] for idx, _ in pairs if idx in by_index]
    if len(split_gpus) < 2:
        # Split configured but degraded to a single detected device (the other
        # vanished / was never present). Same forwarding as the no-split path, so a
        # cold init on THIS path also completes / joins rather than timing out.
        # Under combined_only there is nothing honest to sum - return {} with the
        # probe's REAL status (a TIMEOUT/BUSY here may be why the split looks
        # degraded, and the caller must be able to tell).
        if combined_only:
            return _ret({})
        return _vi()

    out = {"total": sum(g["total"] for g in split_gpus)}
    if combined_only:
        # See the docstring: lets the caller require a genuine 2+-device sum
        # (a plain test double's dict, lacking this key, then reads honestly
        # as "not a combined figure"). Gated so the classic shape stays
        # byte-identical for every existing caller and test.
        out["devices"] = len(split_gpus)
    frees = [g.get("free") for g in split_gpus]
    if all(f is not None for f in frees):
        out["free"] = sum(frees)
        # All-or-nothing, mirroring the "free" key above: a sum is only a whole-board
        # figure if EVERY device in it is. One process-scoped device makes the whole
        # sum process-scoped, because that device's other-process VRAM is missing
        # from it. Absent entirely when NO device reported a scope: that means
        # UNKNOWN, and labelling it "process" would assert a blindness we have not
        # measured (as wrong as asserting the number is fact) while also breaking the
        # plain-dict contract every existing caller and test double relies on.
        scopes = [g.get("free_scope") for g in split_gpus if g.get("free_scope")]
        if scopes:
            out["free_scope"] = (FREE_SCOPE_DEVICE
                                 if all(s == FREE_SCOPE_DEVICE for s in scopes)
                                 else FREE_SCOPE_PROCESS)
    return _ret(out)


def implicit_split_capacity(config: Optional[dict] = None, *,
                            wait_for_inflight: bool = False) -> dict:
    """``{'free', 'total', 'devices'}`` summed across every GPU device llama.cpp's DEFAULT layer split will spread a load over, or ``{}`` when no implicit split applies or it is not measurable."""
    from localm.config import load_config
    cfg = config if config is not None else load_config()
    if cfg.get("gpu_split_indices"):
        return {}
    if _native_backend_has_vulkan():
        devices = native_gpu_devices()
        if not devices:
            return {}
        # DISCRETE GPUs ONLY, and this is a load-safety filter, not tidiness.
        # llama.cpp's device list SKIPS accelerators outright and appends
        # integrated GPUs only when no discrete GPU was found. So a box with a
        # discrete card AND an iGPU - an ordinary laptop, or any desktop CPU
        # with integrated graphics - must not have the iGPU's memory summed
        # into a budget llama.cpp then places entirely on the discrete card.
        # That over-budgets, which is the direction that OOMs rather than
        # merely wasting memory.
        #
        # SINCE 2026-08-12 :func:`native_gpu_devices` ALREADY APPLIES EXACTLY
        # THIS FILTER (see _llama_visible_devices), so on a real reading this
        # pass is now a no-op and is kept as defence in depth rather than as
        # the thing standing between an iGPU and the budget. It still earns
        # its place: ~5 test modules inject device lists by patching
        # native_gpu_devices directly, which bypasses that derivation
        # entirely, and a sum is the one consumer where a stray iGPU is
        # actively unsafe rather than merely mis-numbered. Do not read its
        # presence as evidence the upstream list is unfiltered.
        #
        # Filter to GGML_DEV_TYPE_GPU rather than excluding the others by
        # value: the enum has GROWN (IGPU was inserted ahead of ACCEL, so the
        # numeric value of ACCEL differs between builds we may ship), and this
        # module cannot know which llama.cpp is provisioned. CPU=0 and GPU=1
        # have been stable throughout, so an allowlist is version-independent
        # where a denylist is not. A device whose type the probe did not report
        # is not assumed to be discrete - it fails the filter and, if that
        # leaves fewer than 2, the single-device reading stands.
        #
        # The iGPU-only box needs no special case: llama.cpp keeps at most ONE
        # integrated GPU, so it can never reach the 2+ devices this requires.
        from localm.inference.backends.llamacpp._loader import GGML_DEV_TYPE_GPU
        devices = [d for d in devices
                   if isinstance(d, dict) and d.get("type") == GGML_DEV_TYPE_GPU]
    else:
        devices, status = _list_gpus_kw(return_status=True,
                                        wait_for_inflight=wait_for_inflight)
        if status != GPU_PROBE_OK or not devices:
            return {}
    if len(devices) < 2:
        return {}
    frees, totals = [], []
    for d in devices:
        free = d.get("free") if isinstance(d, dict) else None
        total = d.get("total") if isinstance(d, dict) else None
        if not isinstance(free, int) or not isinstance(total, int):
            return {}
        frees.append(free)
        totals.append(total)
    out = {"free": sum(frees), "total": sum(totals), "devices": len(devices)}
    # SURFACE THE DECISION (rule 5), same contract and same level as
    # resolve_auto_split_ratios' own "auto GPU split: distributing by free VRAM"
    # line, and for the same reason: the always-on ring buffer is INFO+, so a bug
    # report about a wrongly-sized load shows WHICH budget was used and which
    # per-device readings produced it. Until this line existed the success path
    # was silent - only the DECLINE path logged - so a capture could not tell a
    # load budgeted against the whole board from one budgeted against a single
    # card, which is precisely the defect this function was added to fix.
    #
    # INFO is affordable here because this runs per LOAD, not per poll: the
    # callers are the backend's load-time preflights (_check_vram,
    # _auto_gpu_layers, _auto_ctx_max), and the GUI's polling routes reach
    # sysstats.estimate_vram instead, which borrows only the pure
    # _bytes_per_token helper and never this. A single user-initiated load emits
    # this at most three times, not the per-poll flood that a 2.5s heartbeat
    # would make of it.
    logger.info(
        "implicit GPU split: sizing against %d devices by free VRAM - %s "
        "(combined %.1f GB free / %.1f GB total)",
        out["devices"],
        ", ".join(f"device {d.get('index')}: {f / 1024 ** 3:.1f} GB free"
                  for d, f in zip(devices, frees)),
        out["free"] / 1024 ** 3, out["total"] / 1024 ** 3)
    return out


def split_device_count(config: Optional[dict] = None) -> int:
    """How many DETECTED devices the configured gpu_split resolves to - the DETECTED/labelling signal, NOT a load-safety gate."""
    from localm.config import load_config
    cfg = config if config is not None else load_config()
    split = cfg.get("gpu_split_indices")
    if not split:
        return 0
    gpus = list_gpus()
    pairs = resolve_gpu_split(split, cfg.get("gpu_split_ratios"), gpus=gpus)
    by_index = {g.get("index") for g in gpus}
    return len([idx for idx, _ in pairs if idx in by_index])


def applied_split_device_count(config: Optional[dict] = None) -> int:
    """How many devices the loader will ACTUALLY tensor_split across for a GGUF/llama.cpp load - the loader-truth counterpart to :func:`split_device_count`'s DETECTED/labelling count."""
    from localm.config import load_config
    cfg = config if config is not None else load_config()
    if not cfg.get("gpu_split_indices"):
        return 0
    return len(resolve_gpu_split(
        cfg.get("gpu_split_indices"), cfg.get("gpu_split_ratios")))


def _list_gpus_reading(deadline: Optional[float] = None, *,
                       wait_for_inflight: bool = False) -> tuple:
    """``(gpus, status)`` from :func:`list_gpus`, tolerant of a test double patched in as a plain no-kwarg callable - the historical bare-list contract that the ~28 test modules stubbing ``list_gpus`` rely on."""
    try:
        params = inspect.signature(list_gpus).parameters
        accepts = "return_status" in params
        accepts_wfi = ("wait_for_inflight" in params or any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()))
    except (TypeError, ValueError):
        accepts = False
        accepts_wfi = False
    if accepts:
        kw = {"return_status": True}
        if deadline is not None:
            kw["deadline"] = deadline
        if wait_for_inflight and accepts_wfi:
            kw["wait_for_inflight"] = True
        return list_gpus(**kw)
    return list_gpus(), GPU_PROBE_OK


def gpu_split_shortfall(vram_required: int, config: Optional[dict] = None,
                        *, return_status: bool = False,
                        deadline: Optional[float] = None,
                        return_shares_adaptive: bool = False):
    """``[{'index', 'needed', 'free'}, ...]`` for every configured split device whose free VRAM, read from a FRESH probe this call (:data:`GPU_PROBE_OK`), cannot cover its proportional share of *vram_required*."""
    from localm.config import load_config
    cfg = config if config is not None else load_config()

    def _ret(shortfall, status, shares_adaptive=False):
        out = [shortfall]
        if return_status:
            out.append(status)
        if return_shares_adaptive:
            out.append(shares_adaptive)
        return out[0] if len(out) == 1 else tuple(out)

    if not cfg.get("gpu_split_indices"):
        # No split configured: a conclusive answer that needs no hardware probe.
        return _ret([], GPU_PROBE_OK)
    if _native_backend_has_vulkan():
        # GPU-SPLIT-VKINDEX honest-unknown: on the vulkan build the configured
        # split indices live in ggml-vulkan's own index space at load time, which
        # list_gpus() (torch.cuda / nvidia-smi) cannot see or order - torch index
        # N is NOT ggml-vulkan index N (resolve_preferred_device documents exactly
        # this hazard). A per-device share check here would measure the WRONG
        # cards: a silent no-op when torch sees nothing, a wrong refusal/pass on a
        # mixed box. We cannot honestly check per-device fit on this backend, so we
        # do not - but we SURFACE the skip rather than present a check that never
        # ran as "passed" (rule 5, do-not-hide-problems). INFO not debug: the
        # always-on ring buffer is INFO+, so a debug line would never reach a bug
        # report - the same reason vram.py's media_split_notice gives for not
        # burying a user-configured-split shortfall at debug. Not WARNING: the skip
        # is benign whenever the model fits (the common case), so WARNING would cry
        # wolf every load. The GGUF load is subprocess-isolated, so an oversized
        # model still fails as a catchable error, not a lost check - that isolation
        # is the real backstop this defers to.
        logger.info(
            "gpu_split_shortfall: skipping the per-device split VRAM preflight on "
            "the vulkan backend - the configured split indices are in ggml-vulkan's "
            "index space, which list_gpus() cannot map to a card, so no per-device "
            "check can name the right device (GPU-SPLIT-VKINDEX); relying on the "
            "subprocess-isolated loader to catch an oversized load instead.")
        # Conclusive skip with no probe, so it mirrors the no-split return above and
        # reports GPU_PROBE_OK - NOT a non-OK "stale probe" status: nothing was
        # probed, and (like the no-split branch) this is a deterministic routing
        # decision, not an inconclusive reading. A future return_status consumer that
        # needs to tell "checked-clear" from "vulkan-skip" apart would want a distinct
        # status; flagged for the probe-status owner rather than overloaded here.
        return _ret([], GPU_PROBE_OK)

    gpus, status = _list_gpus_reading(deadline)
    if status != GPU_PROBE_OK:
        # No FRESH reading this call: list_gpus served a frozen last-known-good value
        # (or []) after a probe TIMEOUT/BUSY. This gate's whole contract is a LIVE
        # per-device check, so it neither quotes that stale "free" as a current figure
        # (AGENTS.md rule 5) NOR refuses on it: a non-OK probe can be a healthy box
        # whose driver is merely busy/contended (or a caller-shortened deadline on a
        # cold init - see _GPU_PROBE_DEADLINE), so refusing would break working
        # setups on a routine slow probe. The check could not
        # run this call -> admit best-effort, surfaced via debug + the returned status,
        # never a silent success. The GGUF/embedder load runs in an isolated worker
        # whose native abort is contained to that child (PR #606) - the backstop a
        # best-effort admit relies on.
        logger.debug("gpu_split_shortfall: probe status=%s (no fresh per-device VRAM "
                     "reading); admitting split load best-effort, per-device fit "
                     "unverified this call", status)
        return _ret([], status)

    # Judge each device by the share the loader will ACTUALLY give it. With
    # ratios unset the loader gets the auto free-VRAM-proportional split
    # (resolve_auto_split_ratios, computed here from THIS SAME fresh reading,
    # so gate and shares come from one snapshot) - under which a device's
    # share is needed_i = R * free_i / total_free <= free_i whenever the
    # aggregate R fits, making the asymmetric-occupancy refusal structurally
    # impossible; a non-empty result then means the COMBINED estimate is
    # short. Pinned ratios keep the historical static-share math (the loader
    # will not adapt for them). Auto declining (a device's free unmeasurable,
    # a configured index not detected) falls back to the historical
    # equal-split math unchanged - and shares_adaptive stays False there, so
    # a refuse-vs-defer caller (switch_engine) can tell a genuine adaptive
    # all-short result from the pre-feature static-share hazard (see the
    # docstring: the invariant above holds ONLY for adaptive shares).
    cfg_ratios = cfg.get("gpu_split_ratios")
    shares_adaptive = False
    if not cfg_ratios:
        auto_ratios = resolve_auto_split_ratios(cfg, gpus=gpus)
        shares_adaptive = auto_ratios is not None
        cfg_ratios = auto_ratios
    pairs = resolve_gpu_split(
        cfg.get("gpu_split_indices"), cfg_ratios, gpus=gpus)
    if len(pairs) < 2:
        return _ret([], status)
    by_index = {g.get("index"): g for g in gpus}
    total_ratio = sum(ratio for _, ratio in pairs)
    if total_ratio <= 0:
        return _ret([], status)
    shortfall = []
    for idx, ratio in pairs:
        g = by_index.get(idx)
        if g is None or g.get("free") is None:
            # Structural guard for a malformed/absent device dict ONLY - NOT the
            # "probe could not run" handler (that is the status != OK branch above).
            # Under GPU_PROBE_OK, list_gpus emits an int "free" for every device and
            # DROPS a non-reporting one (see :493/:525), so this branch is dead in
            # production; it only stops a None-free entry (e.g. test-injected) from
            # crashing the loop.
            continue
        # NOTE: deliberately NOT gated on g["free_scope"] - do not "fix" that (see the
        # blindness paragraph in the docstring; tried in PR #710 and reverted).
        needed = int(vram_required * (ratio / total_ratio))
        free = g["free"]
        if free < needed:
            shortfall.append({"index": idx, "needed": needed, "free": free})
    return _ret(shortfall, status, shares_adaptive)


def _device_choice_configured(cfg: dict) -> bool:
    """True when the user actually chose a device: a GPU split, or a Main GPU."""
    return bool(cfg.get("gpu_split_indices")) or cfg.get("main_gpu_index") is not None


def resolve_preferred_device(config: Optional[dict] = None, *,
                            gpus: Optional[list] = None) -> Optional[int]:
    """The device a media workload should DEFAULT to, with every OTHER card left VISIBLE. ``None`` when nothing is configured, or when no torch-visible device can be named honestly."""
    from localm.config import load_config
    cfg = config if config is not None else load_config()
    split = cfg.get("gpu_split_indices")
    main = cfg.get("main_gpu_index")
    if not _device_choice_configured(cfg):
        return None                 # nothing configured: do not invent a device
    devices = gpus if gpus is not None else list_gpus()
    by_index = {g.get("index"): g for g in devices}
    if not by_index:
        # No torch-visible device at all, so any index we named would be a guess in an
        # index space we cannot check. Let ComfyUI choose; do not pretend to know.
        return None
    if split:
        pairs = resolve_gpu_split(split, cfg.get("gpu_split_ratios"), gpus=devices)
        if len(pairs) >= 2:
            visible = [idx for idx, _ in pairs if idx in by_index]
            measured = [(idx, by_index[idx]["free"]) for idx in visible
                        if by_index[idx].get("free") is not None]
            if measured:
                return max(measured, key=lambda t: t[1])[0]
            if visible:
                # Split devices ARE torch-visible but none reports free VRAM, so the
                # capacity-informed choice cannot be made. Lead with the first visible
                # one and SAY SO (rule 5): it may not be the emptiest card. Warned, not
                # raised - refusing would break a working setup over a probe that is
                # allowed to be unmeasurable.
                logger.warning(
                    "gpu_split is configured but no split device reports free VRAM, so "
                    "the best card cannot be chosen for media; defaulting to device %d, "
                    "which may have less free VRAM than its peers.", visible[0])
                return visible[0]
            # NOT ONE configured split device is torch-visible. resolve_gpu_split()
            # passes indices through UNVALIDATED on the vulkan llama.cpp build
            # (GPU-SPLIT-VKINDEX), so these are very likely ggml-vulkan indices, which
            # mean something else entirely to torch. Naming one would point ComfyUI at
            # the wrong card. Say so and let ComfyUI default.
            logger.warning(
                "gpu_split %r resolves to no torch-visible device, so media cannot name "
                "one: those indices are not in torch's index space (a Vulkan-only "
                "llama.cpp split does this). Leaving the device to ComfyUI's default.",
                split)
            return None
        # A split was configured but did not resolve to 2+ detected devices.
        # resolve_gpu_split() has already WARNED about the dropped indices; fall
        # through to the main-index answer below rather than guess a device.
    if main is None:
        return None
    idx = resolve_main_gpu_index(main, gpus=devices)
    return idx if idx in by_index else None


def visible_device_order(config: Optional[dict] = None, *,
                         gpus: Optional[list] = None) -> Optional[list]:
    """Every torch-visible device index with the PREFERRED one FIRST, or ``None`` when no device should be named."""
    from localm.config import load_config
    cfg = config if config is not None else load_config()
    if not _device_choice_configured(cfg):
        # Nothing configured, so the answer is None either way - take it from config and
        # do NOT probe. This gate mirrors resolve_preferred_device's own (both call the
        # shared helper): without it, every ComfyUI spawn on an unconfigured box pays for
        # a driver probe to learn what config already knew (#688 regression). Callers put
        # this on the launch path (comfy_client.comfy_child_env), so the probe is not free.
        return None
    devices = gpus if gpus is not None else list_gpus()
    chosen = resolve_preferred_device(cfg, gpus=devices)
    if chosen is None:
        return None
    rest = sorted(g.get("index") for g in devices
                  if g.get("index") is not None and g.get("index") != chosen)
    return [chosen] + rest


def comfy_gpu_option(device_index: int, config: Optional[dict] = None, *,
                     gpus: Optional[list] = None) -> Optional[str]:
    """The ``gpu:N`` string ComfyUI will understand for OUR *device_index*, or ``None`` when it cannot be named honestly."""
    order = visible_device_order(config, gpus=gpus)
    if not order or device_index not in order:
        return None
    return f"gpu:{order.index(device_index)}"


def plan_media_placement(config: Optional[dict] = None, *,
                         gpu_options: Optional[list] = None) -> Optional[dict]:
    """Assign media components to cards for a box ComfyUI sees as 2+ GPUs, or ``None`` to keep the single-card floor."""
    _ = config  # reserved for a future size/identity-aware policy; v1 is positional
    gpu = [o for o in (gpu_options or [])
           if isinstance(o, str) and o.startswith("gpu:")]
    if len(gpu) < 2:
        return None
    second = gpu[1]
    return {"model": None, "clip": second, "vae": second}


def fit_label(size_bytes: int, total_vram: Optional[int]) -> str:
    """Capacity badge for one file, against a single-GPU (or combined-split) VRAM ceiling: 'fits' / 'tight' / 'too-big', or '' when VRAM is unknown. 'tight' means it should load with little headroom (small context, nothing else on the GPU); 'too-big' still runs, with some layers offloaded to system RAM (sl..."""
    if not total_vram or not size_bytes:
        return ""
    need = size_bytes * _WEIGHT_FACTOR + _OVERHEAD_BYTES
    if need <= total_vram * 0.85:
        return "fits"
    if need <= total_vram:
        return "tight"
    return "too-big"
