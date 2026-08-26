# SPDX-License-Identifier: AGPL-3.0-or-later
"""
In-app model discovery: search HuggingFace for GGUF models and judge,
per quantization, whether a file fits this machine's VRAM.

Discovery is not routed through the net_allow/net_deny domain rules, but
``net_mode = off`` still blocks it.

"Fits your VRAM" badges compare against TOTAL VRAM, not currently-free VRAM.
The estimate mirrors the GGUF backend's preflight: weights + ~1.5 GB overhead
for KV cache and compute buffers.
"""

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

# HuggingFace library-tag filter per discoverable model format: GGUF repos carry
# the gguf tag, transformers-native repos the transformers library tag.
_FORMAT_FILTER = {"gguf": "gguf", "hf": "transformers"}

# Per-model-type HF query narrowing for the non-gguf (safetensors) format axis.
_HF_TYPE_FILTER = {
    "llm": {"filter": "transformers"},
    "embedding": {"pipeline_tag": "feature-extraction"},
    "diffusion-unet": {"pipeline_tag": "text-to-image"},
    "lora": {"filter": "peft"},
}
# vae, text-encoder and unknown narrow by FORMAT only; their type comes from the
# result badge and, at pull time, from the checkbox the user searched under.
_HF_TYPE_FILTER_DEFAULT = {"filter": "safetensors"}
_ALL_SEARCHABLE_TYPES = frozenset(
    {"llm", "embedding", "diffusion-unet", "lora", "vae", "text-encoder", "unknown"})
# Sentinel model_type for the all-types-selected broad search: the widest
# reliable format filter (gguf / safetensors), badged.
_ANY_TYPE = "__any__"

# Overhead (KV cache + compute buffers) and weight safety factor from
# localm.vram, as module-level names so fit_label reads a local constant.
from localm.vram import VRAM_OVERHEAD_BYTES as _OVERHEAD_BYTES
from localm.vram import VRAM_WEIGHT_FACTOR as _WEIGHT_FACTOR

# The llama.cpp encoder/embedding architecture allowlist from
# model_manager.gguf. No cycle: gguf.py never imports discover.
from localm.model_manager.gguf import _GGUF_EMBEDDING_ARCHITECTURES

# Quantization label inside a GGUF filename (Q4_K_M, Q8_0, IQ4_XS, F16, BF16,
# MXFP4_MOE, TQ1_0), matched case-insensitively on word-ish boundaries.
_QUANT_RE = re.compile(
    r"(?i)(?<![A-Z0-9])(IQ\d+_[A-Z0-9]+|Q\d+_K(?:_[SML])?|Q\d+_\d+|TQ[12]_0"
    r"|MXFP4(?:_MOE)?|BF16|F16|F32|FP16|FP32)(?![A-Z0-9])")

# Split GGUF naming: model-00001-of-00003.gguf
_SPLIT_RE = re.compile(r"^(?P<stem>.+)-(?P<part>\d{5})-of-(?P<total>\d{5})\.gguf$",
                       re.IGNORECASE)


class DiscoverError(Exception):
    """Discovery failed - network off, HF unreachable, or repo unusable.
    Messages are safe to show in the GUI."""


def _ensure_online() -> None:
    from localm.netpolicy import network_mode
    if network_mode() == "off":
        raise DiscoverError(
            "Network access is disabled (net_mode=off). Enable it with: "
            "localm config net_mode ask")


def _get(url: str, params: Optional[dict] = None) -> object:
    """Policy-checked GET returning parsed JSON.

    Routes through ``netpolicy.safe_fetch_bytes``, so the request is pinned to
    the validated IP and EVERY redirect hop is re-checked against the network
    policy.

    Raises DiscoverError when the request or the JSON decode fails."""
    import json as _json
    import urllib.parse

    from localm import netpolicy
    # doseq=True encodes a list-valued param as repeated keys, which is how the
    # HF models API takes expand.
    full = url + ("?" + urllib.parse.urlencode(params, doseq=True) if params else "")
    try:
        _final, _ctype, body = netpolicy.safe_fetch_bytes(
            full, max_bytes=32 * 1024 * 1024, timeout=int(_TIMEOUT))
        return _json.loads(body.decode("utf-8"))
    except Exception as e:
        raise DiscoverError(f"HuggingFace request failed: {e}")


# Exact tag tokens consulted alongside pipeline_tag for LLM and embedding-ness.
_LLM_TAGS = frozenset({"conversational", "text-generation", "text2text-generation"})
_EMBEDDING_TAGS = frozenset({"feature-extraction", "sentence-similarity"})


def classify_hf_metadata(pipeline_tag: Optional[str], library_name: Optional[str],
                          tags, architecture: Optional[str] = None) -> str:
    """Classify a model_manager.registry MODEL_TYPES value from HARD HF
    metadata (pipeline_tag, library_name, exact tag tokens, GGUF
    architecture). No network, pure function.

    ``architecture`` is the repo's ``gguf.architecture`` (or, for a non-GGUF
    result, ``config.model_type``) expand field - the model's OWN declared
    architecture. Optional, defaults to None.

    Matching is EXACT, never substring: a tag that merely CONTAINS 'vae' /
    'lora' / 'clip' must not be misclassified.

    Order matters:

    - The exact-tag checks (vae/lora/text-encoder) run BEFORE every other
      check, so a repo carrying both a diffusion-flavored pipeline_tag and an
      exact 'lora'/'vae' tag classifies by the tag.
    - ``architecture`` is checked next, before pipeline_tag/tagset, against
      ``localm.model_manager.gguf._GGUF_EMBEDDING_ARCHITECTURES``. Exact
      allowlist membership, so an architecture string that fails to match
      falls through to the pipeline_tag/tagset checks below. Positive-embedding
      only: there is no matching list of architectures meaning llm.

    Returns the 'unknown' sentinel, not a silent 'llm', when no hard signal
    resolves."""
    tag = pipeline_tag
    library = (library_name or "").strip().lower()
    # Exact, lowercased tag tokens; membership is equality, not substring.
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
    # Media / diffusion signal, checked after the exact tag tokens above.
    if tag in ("text-to-image", "image-to-image", "text-to-audio", "audio-to-audio"):
        return "diffusion-unet"
    if tag in ("feature-extraction", "sentence-similarity") or tagset & _EMBEDDING_TAGS:
        return "embedding"
    # image-text-to-text (a vision-language chat checkpoint) classifies as llm,
    # never as a diffusion pipeline.
    if (tag in ("text-generation", "text2text-generation", "conversational",
                "image-text-to-text")
            or tagset & _LLM_TAGS):
        return "llm"
    return "unknown"


def _hf_pipeline_tag_to_type(repo_id: str) -> str:
    """Classify a HuggingFace repo's model type by fetching its metadata and
    running it through classify_hf_metadata(). Returns 'unknown' - not a silent
    'llm' - on a failed or offline query."""
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
    """Estimated GPU weight footprint in bytes for an HF model, from its
    safetensors param metadata (the ``safetensors`` expand field of the HF
    models API: ``{"total": <param count>, "parameters": {...}}``).

    The HF backend loads in bf16 on GPU with no on-load quantization, so the
    footprint is ``total_params * 2`` regardless of the STORED dtype. This is
    the weight size only; fit_label() adds KV-cache / compute overhead.

    Returns None when the repo has no usable param count, so the GUI can show
    "size unknown" rather than a guessed badge."""
    if not isinstance(safetensors, dict):
        return None
    total = safetensors.get("total")
    if not isinstance(total, int) or isinstance(total, bool) or total <= 0:
        return None
    return total * 2


# Name-based MoE fallback for when the header architecture signal is absent or
# wrong. Matches 8x7B style, A3B active-param style, and a bare moe token. Only
# ever the fallback behind the header signal; callers must label a match as
# inferred.
_MOE_NAME_RE = re.compile(r"(?i)\bmoe\b|\b\d+x\d+b\b|\ba\d+b\b")


def _moe_signal(architecture: Optional[str], repo_id: str) -> Optional[str]:
    """MoE-ness for a search-result row: ``"confirmed"`` (the model's own
    ``architecture`` string says so), ``"likely"`` (name pattern only - a
    guess, which the GUI must label as such), or ``None`` (no evidence either
    way). Never returns a "dense" verdict."""
    if architecture and "moe" in str(architecture).lower():
        return "confirmed"
    if _MOE_NAME_RE.search(repo_id):
        return "likely"
    return None


def _param_count(row_fmt: str, gguf_meta: object, safetensors_meta: object) -> Optional[int]:
    """Total parameter count for a classified row, or None when unavailable.

    Reads ``gguf.total`` for gguf-format rows and ``safetensors.total`` for
    hf-format rows. A malformed or adversarial expand field degrades this row's
    count to None rather than raising."""
    src = gguf_meta if row_fmt == "gguf" else safetensors_meta
    if not isinstance(src, dict):
        return None
    total = src.get("total")
    if not isinstance(total, int) or isinstance(total, bool) or total <= 0:
        return None
    return total


def _rows_from_items(data: object, limit: int, *, fmt: Optional[str],
                      classify: bool) -> list[dict]:
    """Build result rows from a raw HF /api/models list response.

    ``fmt`` given: every row is tagged with that one format. ``fmt=None``: the
    format is derived from the item's OWN raw tags, where the Hub-assigned
    "gguf" tag marks a repo containing .gguf files.

    ``classify``: attach a ``detected_type`` (a localm.model_manager.registry
    MODEL_TYPES value, or "unknown") from the item's pipeline_tag /
    library_name / tags fields, for DISPLAY ONLY - never used to exclude a
    result. Also attaches ``architecture`` (the raw
    gguf.architecture/config.model_type string), ``moe``
    ("confirmed"/"likely"/None) and ``param_count``. All four are omitted
    entirely when False.

    Raises DiscoverError when *data* is not a list."""
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
            # safetensors metadata.
            row["size_bytes"] = hf_param_bytes(item.get("safetensors"))
        if classify:
            # gguf.architecture when present, else config.model_type.
            # isinstance-guarded, so a malformed response degrades this row's
            # signal to None.
            gguf_meta = item.get("gguf")
            config_meta = item.get("config")
            architecture = (
                (gguf_meta.get("architecture") if isinstance(gguf_meta, dict) else None)
                or (config_meta.get("model_type") if isinstance(config_meta, dict) else None))
            row["detected_type"] = classify_hf_metadata(
                item.get("pipeline_tag"), item.get("library_name"), raw_tags,
                architecture)
            # Display-only what-is-this-model fields.
            row["architecture"] = architecture or None
            row["moe"] = _moe_signal(architecture, repo)
            row["param_count"] = _param_count(
                row_fmt, gguf_meta, item.get("safetensors"))
        out.append(row)
    return out


def _type_fmt_filter(model_type: Optional[str], fmt: str) -> dict:
    """HF query narrowing (a params fragment: ``filter=`` and/or
    ``pipeline_tag=``) for one (model_type, format) pair.

    ``model_type is None`` is the LEGACY path (CLI ``localm search`` / MCP
    ``search_models``): the plain per-format library tag, gguf -> "gguf",
    hf -> "transformers".

    Otherwise the gguf side is the type-independent "gguf" Hub tag (diffusion
    additionally ANDs "diffusers"), and the hf (safetensors) side narrows per
    type where HF exposes a signal, else the plain "safetensors" format tag."""
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
    """One HF /api/models query for a single (format, type), rows tagged *fmt*.

    ``classify`` requests the pipeline_tag/library_name/tags expand fields and
    attaches a ``detected_type`` badge to each row (display only, never used to
    exclude a result)."""
    params: dict = {"sort": "downloads", "direction": "-1", "limit": str(limit)}
    if query.strip():
        params["search"] = query.strip()
    params.update(_type_fmt_filter(model_type, fmt))
    expand: list[str] = []
    if fmt == "hf":
        # Expand the safetensors param metadata so each result carries a param
        # count for an inline VRAM fit estimate. expand drops the default stat
        # fields, so re-request downloads/likes/lastModified alongside it.
        expand += ["safetensors", "downloads", "likes", "lastModified"]
    elif classify:
        # expand drops the default stat fields once any field is requested, and
        # classify below always requests at least pipeline_tag.
        expand += ["downloads", "likes", "lastModified"]
    if classify:
        expand += ["pipeline_tag", "library_name", "tags", "config"]
        if fmt == "gguf":
            # The llama.cpp architecture from the GGUF header itself; only
            # meaningful for gguf-format results.
            expand += ["gguf"]
    if expand:
        params["expand[]"] = expand
    data = _get(f"{HF_API}/api/models", params)
    return _rows_from_items(data, limit, fmt=fmt, classify=classify)


def _spec_key(model_type: Optional[str], fmt: str):
    """Hashable identity of the HF request a (type, fmt) pair resolves to, so
    two selected types that produce the SAME query (e.g. vae + text-encoder
    both -> filter=safetensors on the hf side) fire ONE call, not two. A
    result's badge comes from its own metadata, not the query's type."""
    frag = _type_fmt_filter(model_type, fmt)
    return (fmt, tuple(sorted(
        (k, tuple(v) if isinstance(v, list) else v) for k, v in frag.items())))


def hf_search(query: str = "", limit: int = 20, formats: Sequence[str] = ("gguf",),
              model_type: Optional[str] = None,
              model_types: Optional[Sequence[str]] = None) -> list[dict]:
    """Search HF for model repos. Empty query = most downloaded.

    Two independent axes:

    - *formats*: a subset of {"gguf", "hf"} ("hf" == the non-gguf / safetensors
      world). One HF query runs per requested format.
    - *model_types*: which registry types to search for (a subset of
      _ALL_SEARCHABLE_TYPES). Each is narrowed server-side where HF exposes a
      signal, and every result is badged with its detected type. When ALL
      searchable types are selected it collapses to the widest reliable format
      filter (2 queries), not a fan-out. *model_type* (singular) is the alias
      for a single-element *model_types*.

    Results across every (type, format) query are merged de-duped by repo id
    and round-robin interleaved so no single query crowds the others out of
    *limit*.

    Returns [{id, downloads, likes, updated, formats, size_bytes?,
    detected_type?}]. ``detected_type`` is present only when a type was
    requested (display only, never used to exclude).

    Raises DiscoverError when the network is off, or when no valid format or
    model type was requested."""
    _ensure_online()
    limit = max(1, min(int(limit), 50))

    # model_types (GUI) wins; else the singular model_type; else None = legacy
    # broad search, no type scoping and no classify.
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
        # Everything selected: the widest reliable format filter, still
        # classified so results are badged.
        query_types = [_ANY_TYPE]
    else:
        query_types = list(types)

    # Build the (type, fmt) query list, collapsing pairs that resolve to the
    # SAME HF request.
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

    # Round-robin interleave by per-query rank, then trim to limit.
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
    """True when the HF/transformers runtime can actually RUN a model here:
    both torch and transformers are importable. Uses importlib.util.find_spec,
    so nothing heavy is imported.

    When False, an HF (transformers-format) model can still be DOWNLOADED via
    pull; it simply cannot be loaded until the ``.[gpu]`` extra (torch +
    transformers) is installed. The GUI surfaces that and does NOT block the
    download."""
    import importlib.util
    try:
        return bool(importlib.util.find_spec("torch")
                    and importlib.util.find_spec("transformers"))
    except (ImportError, ValueError):
        # find_spec can raise on a half-installed namespace package; treat an
        # unresolvable probe as not available.
        return False


def hf_gguf_files(repo: str) -> list[dict]:
    """
    List the GGUF files of *repo* with size and quant label. Split files
    (``-00001-of-0000N``) are grouped into one logical entry whose ``file``
    is the first part (what ``localm pull repo:file`` expects) and whose
    size is the sum of all parts. Sorted smallest-first.
    """
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
    """The single quant label for *name*, preferring an MXFP4/MXFP4_MOE match
    over any earlier one in the string, so a mixed-precision MoE export that
    names the non-expert tensor precision FIRST (e.g.
    '...-bf16_MXFP4_MOE.gguf') still reports the expert quantization. Empty
    string when there is no match."""
    matches = _QUANT_RE.findall(name)
    if not matches:
        return ""
    for m in matches:
        if m.upper().startswith("MXFP4"):
            return m.upper()
    return matches[0].upper()


# ---- GPU probe safety: a hardware probe must never block its caller -------- #
# list_gpus() runs _list_gpus_probe() on a helper thread with a hard deadline; on
# overrun the caller gets the last-known-good reading (or []) and that thread is
# abandoned. At most one probe is ever in flight, and the overrun is surfaced at
# debug level. Every call re-probes: there is no freshness/TTL cache, and the
# last-known-good value is the wedge fallback only.
_GPU_PROBE_DEADLINE = 15.0    # seconds a probe may block its caller
# Historical alias for the call sites and tests that opt into it by name
# (doctor, localm gpus, switch_engine).
_GPU_PROBE_CLI_DEADLINE = _GPU_PROBE_DEADLINE

# Outcome of a probe, surfaced by list_gpus(..., return_status=True). A
# user-facing no-GPU message MUST branch on this.
GPU_PROBE_OK = "ok"            # a fresh probe completed - an empty list means genuinely none
GPU_PROBE_TIMEOUT = "timeout"  # probe exceeded the deadline (cold driver init / wedge); INCONCLUSIVE
GPU_PROBE_BUSY = "busy"        # another probe is inflight, or the probe thread could not start
# The probe completed within its deadline, but the isolated torch enumeration
# could not be asked this round and nvidia-smi also found nothing. An empty list
# under this status is inconclusive, and a longer deadline cannot help.
GPU_PROBE_INCONCLUSIVE = "inconclusive"

_gpu_probe_lock = threading.Lock()
_gpu_last_good: Optional[list] = None    # last SUCCESSFUL probe; served on a wedge
_gpu_probe_inflight = False
# Published together with _gpu_probe_inflight (under _gpu_probe_lock) when a
# probe thread is started, and cleared when it lands, so a caller passing
# wait_for_inflight can JOIN the running probe instead of getting GPU_PROBE_BUSY.
# The default is still to refuse a second probe on a driver already being probed.
_gpu_probe_done: Optional[threading.Event] = None
_gpu_probe_result: Optional[dict] = None
# Bumped by _reset_gpu_probe_cache() to ORPHAN any probe thread still in flight.
# A stale thread's result is fenced out by epoch rather than raced against.
_gpu_probe_epoch = 0


def last_known_gpus() -> list:
    """The most recent SUCCESSFUL :func:`list_gpus` reading, WITHOUT probing.

    For a caller that has JUST driven a probe (e.g. via :func:`vram_capacity`)
    and wants the per-device detail behind the number it already has.

    Returns ``[]`` when no probe has ever succeeded - never a fabricated or
    partial reading, and never a fresh probe.

    NOT a substitute for ``list_gpus`` when the reading must be current: the
    value here is as fresh as whatever last probed, and nothing about it says
    when.
    """
    return list(_gpu_last_good or [])


def _reset_gpu_probe_cache() -> None:
    """Test hook: drop the last-known-good GPU reading and the in-flight flag,
    and INVALIDATE any probe still in flight so it cannot bleed into the next
    test.

    Clearing the globals is not enough on its own: an overrunning probe is
    abandoned rather than cancelled, so that thread outlives this reset and
    would otherwise write its reading into _gpu_last_good afterwards. Bumping
    the epoch makes that late write a no-op (see _run)."""
    global _gpu_last_good, _gpu_probe_inflight, _gpu_probe_epoch
    global _gpu_probe_done, _gpu_probe_result, _isolated_torch_unavailable
    global _isolated_torch_broken_warned
    with _gpu_probe_lock:
        _gpu_last_good = None
        _gpu_probe_inflight = False
        # Cleared with the rest of the probe state.
        _isolated_torch_unavailable = False
        _isolated_torch_broken_warned = False
        # Unpublish the join handles too, so no caller joins a probe from the
        # epoch just retired.
        _gpu_probe_done = None
        _gpu_probe_result = None
        _gpu_probe_epoch += 1


def list_gpus(*, deadline: float = _GPU_PROBE_DEADLINE, return_status: bool = False,
              wait_for_inflight: bool = False):
    """Every GPU device visible right now: ``[{"index", "name", "total",
    "free"}, ...]``, or ``[]`` when nothing is measurable.

    The real driver probe (:func:`_list_gpus_probe`) runs on a helper thread
    with a hard ``deadline``-second timeout, so this call NEVER blocks its
    caller for longer than ``deadline`` even if the GPU driver wedges. Every
    call re-probes; there is no TTL cache, so a live "free" reading is never
    stale. On an overrun the last-known-good value (or ``[]``) is returned and
    the stuck probe thread is abandoned. The default ``deadline`` is generous
    enough to wait out a legitimate COLD driver init
    (:data:`_GPU_PROBE_CLI_DEADLINE` is an alias of it); override it only in
    tests, or where a caller wants a faster degraded answer.

    When ``return_status`` is True, returns ``(gpus, status)`` where ``status``
    is:

    - :data:`GPU_PROBE_OK` - a fresh probe completed, so an empty ``gpus``
      means genuinely no measurable GPU.
    - :data:`GPU_PROBE_TIMEOUT` - the probe exceeded ``deadline``, typically a
      cold ROCm/CUDA driver init that has not finished, so an empty ``gpus`` is
      INCONCLUSIVE and a retry with a longer deadline may succeed.
    - :data:`GPU_PROBE_BUSY` - another probe is already inflight, or the probe
      thread could not start; no fresh reading was taken.
    - :data:`GPU_PROBE_INCONCLUSIVE` - the probe completed, but the isolated
      torch enumeration could not be asked this round and nvidia-smi, the only
      other source, also found nothing; unlike TIMEOUT, a longer deadline will
      not help.

    A caller that renders a user-facing "no GPU" message MUST branch on this.
    ``return_status`` defaults to False, which returns the bare list.

    Tries torch first (CUDA/ROCm - torch's ROCm build aliases torch.cuda.* to
    HIP, so an AMD card enumerates through the same API) since it also gives a
    device name; falls back to a name-aware ``nvidia-smi`` listing of ALL
    devices for the GGUF-only install that has no torch.

    ``wait_for_inflight`` (default False) changes ONLY what happens when a
    probe is already in flight: instead of returning :data:`GPU_PROBE_BUSY` at
    once with the last-known-good reading, this call JOINS the running probe
    and waits on its completion, bounded by its own ``deadline``. It never
    spawns a second probe. Set it ONLY together with a long ``deadline`` and
    ONLY off the event loop: a joining wait can block the caller for up to
    ``deadline`` seconds.

    Does NOT fall back to the Windows display-adapter registry: that tier (see
    vram_info()) reports one aggregate "largest adapter" number with no
    per-device identity, so it cannot support GPU *selection*."""
    gpus, status = _list_gpus_with_status(deadline, wait_for_inflight)
    return (gpus, status) if return_status else gpus


def _list_gpus_with_status(deadline: float, wait_for_inflight: bool = False) -> tuple:
    """The real probe driver behind :func:`list_gpus`, returning
    ``(gpus, status)`` where status is one of :data:`GPU_PROBE_OK` /
    :data:`GPU_PROBE_TIMEOUT` / :data:`GPU_PROBE_BUSY` /
    :data:`GPU_PROBE_INCONCLUSIVE`. ``wait_for_inflight``: see
    :func:`list_gpus` - a patient off-loop caller JOINS a probe already in
    flight, bounded by ``deadline``, rather than short-circuiting on BUSY."""
    global _gpu_last_good, _gpu_probe_inflight, _gpu_probe_done, _gpu_probe_result
    global _probe_deadline_at   # published with the slot for the cold-budget check
    join_done = None
    join_result = None
    with _gpu_probe_lock:
        if _gpu_probe_inflight:
            if wait_for_inflight and _gpu_probe_done is not None:
                # JOIN the in-flight probe: wait on ITS completion event, bounded
                # by our own deadline. Both handles are captured here, under the
                # same lock that observed the in-flight slot.
                join_done = _gpu_probe_done
                join_result = _gpu_probe_result
            else:
                # Hand back the last-known-good reading. No fresh reading was
                # taken, so the status is BUSY; [] when nothing has succeeded yet.
                served = list(_gpu_last_good) if _gpu_last_good is not None else []
                return served, GPU_PROBE_BUSY
        else:
            _gpu_probe_inflight = True
            # Published under the SAME lock that claims the in-flight slot. The
            # probe body reads it to decide whether it can afford a cold
            # device-global VRAM source. See _apply_device_global_free.
            _probe_deadline_at = time.monotonic() + deadline
            # Captured under the SAME lock that claims the in-flight slot.
            my_epoch = _gpu_probe_epoch
            # Created and published under the lock, atomically with the in-flight
            # slot. Cleared by _run when the probe lands.
            result: dict = {}
            done = threading.Event()
            _gpu_probe_done = done
            _gpu_probe_result = result

    # JOIN path: we did not start a probe; wait on the one already running.
    if join_done is not None:
        if join_done.wait(deadline):
            # The joined probe landed. Its result carries a value key only when
            # its thread ran to completion; an absent key is a BUSY, not an OK.
            if "value" in join_result:
                v = join_result["value"]
                status = (GPU_PROBE_OK if join_result.get("conclusive", True)
                         else GPU_PROBE_INCONCLUSIVE)
                return (list(v) if v is not None else []), status
            with _gpu_probe_lock:
                served = list(_gpu_last_good) if _gpu_last_good is not None else []
            return served, GPU_PROBE_BUSY
        # Our own deadline expired while waiting on the in-flight probe: serve
        # last-known-good and report TIMEOUT. Nothing was spawned.
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
            # Conclusive unless the value is EMPTY and the isolated torch
            # enumeration latched as unable to answer this round. Read under the
            # same lock that guards _torch_gpus_isolated_once.
            conclusive = not (not value and _isolated_torch_unavailable)
            if _gpu_probe_epoch != my_epoch:
                # A reset retired this probe while it ran: both writes are
                # dropped, at debug rather than silently, and the in-flight slot
                # is left alone.
                logger.debug("list_gpus: discarding probe result from retired "
                             "epoch %s (current %s)", my_epoch, _gpu_probe_epoch)
            else:
                if value is not None:
                    _gpu_last_good = value
                _gpu_probe_inflight = False
                # Unpublish alongside the in-flight slot, so a NEW caller starts a
                # fresh probe rather than joining one that has already landed.
                _gpu_probe_done = None
                _gpu_probe_result = None
                # Cleared with the in-flight slot: the budget describes THIS probe
                # and nothing else.
                _probe_deadline_at = None
        # Outside the epoch gate and unconditional: the starter and any joiner
        # are waiting on done.
        result["value"] = value
        result["conclusive"] = conclusive
        done.set()

    try:
        threading.Thread(target=_run, name="localm-gpu-probe", daemon=True).start()
    except Exception as e:
        # Could not spawn the probe thread. Resets the in-flight guard so a later
        # call can retry, surfaces it at debug, and degrades to the
        # last-known-good reading with status BUSY. Epoch-gated like the clear in
        # _run.
        with _gpu_probe_lock:
            if _gpu_probe_epoch == my_epoch:
                _gpu_probe_inflight = False
                _gpu_probe_done = None
                _gpu_probe_result = None
        # Wake any caller that joined between the publish above and this failure.
        # result has no value key, which the join path reads as BUSY.
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
    # cancelled. Serves the last-known-good value and lets the abandoned thread
    # finish; _gpu_probe_inflight stays True until it does. Status is TIMEOUT.
    logger.debug("list_gpus: GPU probe exceeded %.1fs deadline (driver call stuck); "
                 "returning last-known GPU info so the caller does not block", deadline)
    with _gpu_probe_lock:
        served = list(_gpu_last_good) if _gpu_last_good is not None else []
        return served, GPU_PROBE_TIMEOUT


def native_hip_runtime_resident() -> bool:
    """True when llama.cpp's bundled HIP-linked runtime is resident IN THIS
    process on Windows: the native lib has been loaded (``_loader.load_lib``)
    and the resolved runtime ships a HIP ggml backend.

    Two callers read it, each adding its own narrowing:

    - :func:`_torch_gpu_probe_known_doomed`: a FRESH ``import torch`` here
      collides with the resident HIP DLLs (it adds the torch-absence and
      ``rocm_sdk`` conditions on top).
    - ``gpu_usage.raw_reading_is_process_scoped``: the raw free-VRAM readings
      this process can take are HIP-sourced, and the HIP runtime's reading on
      Windows is blind to other processes, so blindness can be answered even
      where torch cannot be consulted at all.

    Fails closed (False) when the check itself errors; both callers treat False
    as "no special handling". The glob re-resolves ``runtime_binary_dir()`` at
    check time."""
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
    """True when :func:`_list_gpus_probe`'s ``import torch`` attempt below is
    KNOWN, ahead of time, to fail in this exact process state - so the probe
    skips it at the root instead of triggering the failure and catching the
    aftermath.

    All three conditions must hold:

    - torch is not already resident in ``sys.modules``. A resident torch
      re-imports as a free cache hit: no preload runs, nothing can fault, and
      its working enumeration is kept.
    - :func:`native_hip_runtime_resident` - Windows, the native lib loaded, and
      the resolved runtime ships a HIP ggml backend. The conflict is Windows
      OS-loader same-name resolution against resident HIP DLLs, so a fresh
      process, or a vulkan/cpu/cuda build, leaves nothing to collide with.
    - ``rocm_sdk`` is importable: the failing preload belongs to the
      ROCm-for-Windows torch. Necessary, not sufficient - firing with a
      non-ROCm torch loses nothing material, since a CPU torch enumerates no
      CUDA devices and a CUDA torch's devices are what the nvidia-smi fallback
      reports anyway.

    Fails OPEN: if the detector itself errors, the probe proceeds with its
    normal torch attempt, which catches its own failures. The skip is surfaced
    at debug level."""
    import sys
    if "torch" in sys.modules:
        # A resident torch makes import torch a plain cache hit: no rocm_sdk
        # preload runs, so the conflict cannot occur.
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
# and the probe falls through to nvidia-smi. Must fit inside _GPU_PROBE_DEADLINE
# together with the nvidia-smi fallback's own timeout=5, and sit above a
# legitimate cold driver init.
_ISOLATED_TORCH_PROBE_TIMEOUT = 10.0

# Latched True once the out-of-process torch enumeration proves it CANNOT answer
# on this box (spawn failure, timeout, unusable reply). Read and written under
# _gpu_probe_lock, cleared by _reset_gpu_probe_cache. Records only that torch
# cannot be asked here, never a VRAM number.
_isolated_torch_unavailable = False

# Latched once the isolated probe has been reported BROKEN. Suppresses a repeated
# log line only; it disables no capability.
_isolated_torch_broken_warned = False


def isolated_torch_unavailable() -> bool:
    """True once the isolated probe has PROVEN, in a child process, that torch
    cannot finish enumerating on this box (see
    :func:`_torch_gpus_isolated_once`, which sets the latch this reads).

    Public because the latch binds callers outside this module: retrying that
    import IN-PROCESS would reproduce the multi-minute startup hang the
    isolation exists to prevent, so any other caller about to ``import torch``
    on a hot path must skip it too. In particular
    ``_sizing.VramSizingMixin._free_total_vram_bytes``, which sits on the
    model-LOAD path.

    False means "not proven unavailable", NOT "torch works" - the probe may
    simply never have run. It is a reason to SKIP an attempt, never evidence
    that an attempt will succeed, so a caller still needs its own bound.
    """
    with _gpu_probe_lock:
        return _isolated_torch_unavailable


# Cap on how much of the child probe's stderr is kept. Any truncation beyond it
# is marked, never silent.
_CHILD_STDERR_LOG_CAP = 2000


def _capped_stderr(text: str, limit: int = _CHILD_STDERR_LOG_CAP) -> str:
    """*text* (child-probe stderr), capped to *limit* chars for a log line.
    Marks the cut explicitly when it actually truncates - a silently cut
    diagnostic is the same rule-5 shape as swallowing it outright."""
    if len(text) <= limit:
        return text
    return text[:limit] + f"... [truncated, {len(text) - limit} more chars]"


class _IsolatedTorchWedged(Exception):
    """The out-of-process torch probe ran but did not finish in time, i.e.
    TORCH ITSELF is wedging on this box. Distinct from the child mechanism
    being broken, and the distinction decides the fallback: this one must never
    fall back to an in-process import."""


def _torch_is_resident() -> bool:
    """True when torch is ALREADY imported in this process, so enumerating here
    is a free ``sys.modules`` cache hit that takes no OS loader lock."""
    import sys
    return "torch" in sys.modules


def _torch_gpus_resident() -> list:
    """torch's device list, read IN THIS PROCESS. Only safe to call when
    :func:`_torch_is_resident` - see :mod:`localm._torch_gpu_probe` for why a
    COLD import here freezes every thread in the process."""
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
    """torch's device list read from a CHILD process, for the case where torch
    is not yet resident and importing it HERE would take the Windows OS loader
    lock and block thread creation process-wide, stalling the event loop (full
    mechanism in :mod:`localm._torch_gpu_probe`).

    Returns the device list (possibly ``[]``, a real answer meaning torch sees
    no CUDA/HIP device), or ``None`` when the child COULD NOT ANSWER at all -
    spawn failure, timeout, or an unusable reply. The caller falls through to
    nvidia-smi either way, but only ``None`` latches
    :data:`_isolated_torch_unavailable`, so a box where torch simply has no
    device is not mistaken for one where torch cannot be asked. The child
    inherits this process's environment, so ``CUDA_VISIBLE_DEVICES`` selects
    and orders devices identically and the TORCH index space
    :func:`list_gpus` promises is preserved.

    Spawned via ``interpreter_for_localm_children()``, NOT bare
    ``sys.executable``: inside a Windows multiprocessing-spawn worker the
    latter is the BASE interpreter, whose children get no venv context and so
    cannot import torch or localm at all."""
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
        # Surfaced, not silenced: a silent [] here is indistinguishable from
        # this box has no GPU.
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
        # The child ALWAYS prints one line, [] included on its own failure path,
        # so empty stdout means it died before printing. That is COULD NOT ASK,
        # not torch sees no device.
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
        # The child prints its own failure cause here before answering [].
        logger.debug("list_gpus: out-of-process torch probe reported: %s",
                     _capped_stderr(err))
    return devices


def _torch_gpus_isolated_once() -> list:
    """:func:`_torch_gpus_isolated`, but never retried on a box that has
    already proven it cannot answer. Returns the device list, or ``[]`` so the
    caller falls through to nvidia-smi.

    The latch keeps a wedged torch from costing the FULL timeout on every
    probe, since `list_gpus` re-probes on every call.

    Two failures that look alike and are not treated alike:

    - TORCH WEDGES (timeout). Isolation worked and told us torch cannot finish
      here. Latch, and never retry in-process.
    - ISOLATION IS BROKEN (cannot spawn, unusable reply). Nothing was learned
      about torch, and falling straight through to nvidia-smi would turn "we
      could not look" into a confident "no GPU" on every AMD and Intel box. So
      degrade to the IN-PROCESS import and say plainly at WARNING that the
      isolation was lost and the stall risk is back.

    Once latched, this still returns [] and the probe still falls through to
    nvidia-smi. :func:`_list_gpus_with_status` reads
    ``_isolated_torch_unavailable`` once this call returns and reports
    :data:`GPU_PROBE_INCONCLUSIVE` instead of :data:`GPU_PROBE_OK` exactly when
    the reading came back empty AND this latch is set - never for a non-empty
    reading.

    The return type is a bare ``list``, never ``None``; the status channel is a
    separate, additive path through module state, not a change to this
    signature.

    Out of scope: :func:`_torch_gpu_probe_known_doomed` skips the torch attempt
    ENTIRELY on its narrower doomed combination without touching this latch, so
    that skip is not detected as inconclusive here either."""
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
                # Said once, not once per probe; marks where the per-attempt
                # reasons stop appearing.
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
            # Once per process, not once per probe. Later occurrences stay at
            # debug.
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
    """The actual (blocking) GPU driver probe. Call :func:`list_gpus`, not this -
    this one has no timeout and can wedge on a busy/broken driver."""
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
                        # every process, so it needs no correction.
                        "free_scope": FREE_SCOPE_DEVICE,
                    })
                except ValueError:
                    continue   # a malformed line never hides the rest
            if out:
                return out
    except Exception:
        pass
    return []


# How much of the world a GPU entry's free actually accounts for.
FREE_SCOPE_DEVICE = "device"    # every process's VRAM is counted - the number is the board's
FREE_SCOPE_PROCESS = "process"  # ONLY this process's own allocations are counted (see below)

# Probe budget below which a COLD (not-yet-opened) device-global source is
# skipped. See _apply_device_global_free.
_CORRECTION_COLD_BUDGET_S = 1.5

# When the in-flight probe's deadline expires (monotonic), or None outside a
# probe. Set by _list_gpus_with_status under the same lock that claims the
# in-flight slot; _gpu_probe_inflight serialises probes, so there is only ever
# one in flight to describe.
_probe_deadline_at = None


def _apply_device_global_free(gpus: list) -> None:
    """Correct each entry's ``free`` to a DEVICE-GLOBAL figure where this
    platform's driver query is not one already, and tag every entry with
    ``free_scope`` so a caller can tell a whole-board number from a
    process-local one. Mutates *gpus*.

    On Windows with an AMD ROCm/HIP torch build,
    ``torch.cuda.mem_get_info`` reports ``total - the calling process's own
    allocations`` and is blind to every other process. Every GGUF load is
    out-of-process, so the model's own VRAM is always in another process from
    the server measuring it, as is a game or a ComfyUI.

    On Linux, and on NVIDIA, the driver query is device-global by
    documentation, so nothing is corrected there and the reading is tagged
    :data:`FREE_SCOPE_DEVICE` unchanged.

    When no better source can answer on Windows, the entry keeps the driver's
    number but is tagged :data:`FREE_SCOPE_PROCESS` rather than passing a
    known-process-local figure off as the board's."""
    import sys
    if sys.platform != "win32":
        for g in gpus:
            g["free_scope"] = FREE_SCOPE_DEVICE
        return

    # The scope used when a device-global correction is NOT available for an
    # entry (source cold-skipped, unmappable, or failed). Tags PROCESS only where
    # the raw reading is known blind (Windows + an AMD ROCm/HIP torch build).
    # Computed up front so it is defined on every path below.
    try:
        from localm.gpu_usage import raw_reading_is_process_scoped
        uncorrected_scope = (FREE_SCOPE_PROCESS if raw_reading_is_process_scoped()
                             else FREE_SCOPE_DEVICE)
    except Exception:
        # gpu_usage unimportable: default to DEVICE.
        uncorrected_scope = FREE_SCOPE_DEVICE

    try:
        from localm.gpu_usage import device_global_used_bytes, source_is_warm
        # Runs inside the deadline-bounded probe. Opening the source costs a
        # driver init once per process; a warm read is effectively free. A COLD
        # source is skipped when the remaining budget is too thin, and the reading
        # is then tagged with the uncorrected scope. A warm source always runs.
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
        # uncorrected scope.
        logger.debug("list_gpus: device-global VRAM source failed: %s", e)
        used = {}
    for g in gpus:
        u = used.get(g.get("index"))
        if u is None:
            g["free_scope"] = uncorrected_scope
            continue
        total = int(g["total"])
        # Clamp: the used figure and total come from different sources, so their
        # difference can land just outside [0, total].
        g["free"] = max(0, min(total, total - int(u)))
        g["free_scope"] = FREE_SCOPE_DEVICE


def _native_backend_has_vulkan() -> bool:
    """True when the currently-resolved native runtime directory ships the
    Vulkan ggml backend (a ``ggml-vulkan.*`` file), i.e. the active install is
    the ``vulkan`` build.

    ``list_gpus()`` enumerates ONLY via torch.cuda (CUDA, or HIP under a
    ROCm-build torch) or nvidia-smi; it never calls the Vulkan loader, so it is
    structurally blind to any device only visible through Vulkan. On the vulkan
    build the REAL device selection at load time happens inside
    ggml-vulkan/llama.dll's own enumeration, a different index space
    list_gpus() cannot see or validate against.

    Checks the actual shipped DLL/SO set, NOT the ``.localm-backend``
    provisioning marker: that marker can be absent (a ``--from`` build, an
    install predating it) or generic (``"custom"`` for a ``--url``/``--sha256``
    provision), while the real file set is always authoritative for which
    backend will actually be loaded."""
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
    """The subset of a native non-CPU device inventory that llama.cpp will
    actually place layers on, RENUMBERED into the index space ``mp.main_gpu``
    and ``mp.tensor_split`` consume - i.e. what a configured ``main_gpu_index``
    / ``gpu_split_indices`` has to be expressed in to name the card the user
    meant.

    ``_loader.native_device_inventory`` numbers EVERY non-CPU device in raw
    ``ggml_backend_dev_get`` order. llama.cpp's own ``model->devices``
    (``llama_prepare_model_devices``) is instead built as:

        RPC-backed devices, hoisted to the FRONT
        + GPU-type devices in registry order, deduplicated by device_id
        + at most ONE integrated GPU, and ONLY when no discrete GPU was found
        CPU and ACCEL devices are SKIPPED; META aborts fatally

    So on a box with a discrete card beside integrated graphics the inventory
    carries a device llama.cpp's list does not, and if the iGPU enumerates
    first every index is off by one.

    RPC hoisting and device_id dedup cannot arise here, so neither is emulated:
    ``ggml_backend_rpc_add_server`` is never called anywhere in this project,
    and ggml-vulkan already dedups one physical GPU seen under two drivers by
    ``deviceUUID``/``deviceLUID`` before it reaches the registry.

    Allowlists ``GPU`` rather than excluding the others by value: the enum has
    GROWN (IGPU was inserted AHEAD of ACCEL, so the value 2 means ACCEL on an
    older runtime and INTEGRATED GPU on a newer one) while ``CPU`` 0 and
    ``GPU`` 1 have held. A device whose type the probe did not report fails the
    filter rather than being assumed discrete.

    WHEN NO GPU-TYPE DEVICE IS PRESENT THE LIST IS RETURNED UNCHANGED: an
    iGPU-only box has llama.cpp fall back to its single integrated GPU as
    device 0, which is exactly what this inventory already numbers 0.
    Identifying an IGPU device positively would need the unstable enum value
    above, so this branch declines to guess."""
    from localm.inference.backends.llamacpp._loader import GGML_DEV_TYPE_GPU
    gpus = [d for d in devices
            if isinstance(d, dict) and d.get("type") == GGML_DEV_TYPE_GPU]
    if not gpus:
        return list(devices)
    return [{**d, "index": i} for i, d in enumerate(gpus)]


def native_gpu_devices() -> Optional[list]:
    """Selector-shaped devices from the ACTIVE native runtime's OWN registry,
    read crash-isolated (the probe daemon - ``_loader.gpu_devices_isolated``):
    ``[{"index", "name", "total"?, "free"?}, ...]``, or ``None`` when the
    daemon/registry cannot answer this call. An empty list is a real answer
    (the runtime registers no non-CPU device).

    The ``index`` values are the index space a configured
    ``gpu_split_indices`` / ``main_gpu_index`` actually means at load time, and
    on the ``vulkan`` build the only source that can express it at all
    (:func:`list_gpus` is structurally blind to it). That is NOT simply the
    registry's own numbering: llama.cpp drops integrated GPUs whenever a
    discrete card exists and skips accelerators outright, so the raw inventory
    from ``_loader.native_device_inventory`` is passed through
    :func:`_llama_visible_devices` first, which keeps the devices the loader
    will really use and renumbers them into the space it indexes. The GUI
    selectors write these numbers into config,
    :func:`resolve_auto_split_ratios` pairs a configured index back to a device
    by it, and :func:`implicit_split_capacity` sums over the set.

    This is the enumeration source for the GUI's split/main-GPU SELECTORS on
    that build. NOT merged into :func:`list_gpus`, whose torch/nvidia-smi index
    space feeds the torch-side reads (:func:`vram_capacity`'s per-device sums,
    :func:`gpu_split_shortfall`).

    ``name`` prefers the registry's human description ("AMD Radeon RX 6900
    XT...") over the backend's terse name ("Vulkan0"). ``total``/``free`` are
    included only when the registry reported positive bytes."""
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
        # GPU from an integrated one or an accelerator. Absent when the probe did
        # not report one.
        t = d.get("type")
        if isinstance(t, int):
            entry["type"] = t
        out.append(entry)
    return _llama_visible_devices(out)


def resolve_main_gpu_index(configured, *, gpus: Optional[list] = None) -> int:
    """The GPU device index to actually use, given the user's
    ``main_gpu_index`` config value.

    None (not configured) resolves to device 0, with no detection work done at
    all. An explicitly configured index is validated against the devices
    ``list_gpus()`` (or the injected *gpus*, for tests) currently sees; an
    index that matches none of them is surfaced as a WARNING and swapped for
    device 0 rather than trusted blindly.

    An index above ``_MAX_GPU_SPLIT_INDEX`` is rejected unconditionally, the
    same sanity ceiling :func:`resolve_gpu_split` applies - checked BEFORE any
    device-membership branching, so it still applies when detection is
    unmeasurable or skipped.

    When detection itself is unmeasurable (``list_gpus()`` returns nothing) OR
    the active native backend is ``vulkan`` (see
    :func:`_native_backend_has_vulkan`), the configured index cannot be
    cross-checked against a reliable, backend-matching device list, and is
    passed through unchecked apart from the ceiling above."""
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
    # Check membership by the index field, NOT list position: a device that fails
    # to report leaves a gap. Skipped entirely when the active backend is vulkan,
    # where list_gpus() is blind to Vulkan-only devices.
    if gpus and not _native_backend_has_vulkan() and not any(
            g.get("index") == idx for g in gpus):
        logger.warning(
            "main_gpu_index=%d does not match any of the %d GPU(s) detected "
            "right now; falling back to device 0", idx, len(gpus))
        return 0
    return idx


def apply_main_gpu(mp, *, config: Optional[dict] = None) -> None:
    """Set ``mp.main_gpu`` from the configured ``main_gpu_index``, validated via
    :func:`resolve_main_gpu_index`. Leaves the native default (0, set by
    ``llama_model_default_params()``) untouched when unset. Shared by the
    llama.cpp chat backend and the embedder, so both native-load call sites
    honour the same selection with the same fallback and warning behaviour."""
    from localm.config import load_config
    cfg = config if config is not None else load_config()
    configured = cfg.get("main_gpu_index")
    if configured is None:
        return
    mp.main_gpu = resolve_main_gpu_index(configured)


# llama.cpp's LLAMA_SPLIT_MODE_LAYER (0=NONE/single-GPU, 1=LAYER, 2=ROW,
# 3=TENSOR). LAYER splits whole layers across devices proportional to
# tensor_split; ROW and TENSOR split individual tensors instead.
_LLAMA_SPLIT_MODE_LAYER = 1

# Fallback tensor_split array capacity when llama_max_devices() cannot be probed.
# tensor_split is a raw const float* with no length of its own.
_TENSOR_SPLIT_FALLBACK_CAPACITY = 16

# Sanity ceiling for a gpu_split_indices entry, bounding the ctypes tensor_split
# allocation apply_gpu_split drives. settings_schema.py's MAX_GPU_SPLIT_INDEX
# applies the same value at config WRITE time; this is the READ-time check. Also
# bounds a single main_gpu_index in resolve_main_gpu_index: both values reach the
# identical ctypes.c_int32 main_gpu field.
_MAX_GPU_SPLIT_INDEX = 127


def resolve_gpu_split(configured_indices, configured_ratios=None, *,
                       gpus: Optional[list] = None) -> list:
    """Validate a configured multi-GPU split (``gpu_split_indices`` /
    ``gpu_split_ratios``) against the devices ``list_gpus()`` (or the injected
    *gpus*, for tests) currently sees, returning ``[(index, ratio), ...]``
    ready to write into ``tensor_split``.

    An index that does not match a currently-detected device is dropped with a
    WARNING, so a stale config referencing a since-removed GPU degrades to
    single-GPU. Duplicate indices keep their first occurrence. Fewer than 2
    valid indices after validation means "no split" (returns ``[]``); the
    single-GPU path driven by ``apply_main_gpu`` is unaffected. This validation
    is SKIPPED, and the indices pass through unchecked, when the active native
    backend is ``vulkan`` - see :func:`_native_backend_has_vulkan`, since
    ``list_gpus()`` cannot see Vulkan-only devices and a non-empty result there
    is not authoritative.

    ``configured_ratios``, when given, must be the SAME LENGTH as
    ``configured_indices`` (before validation) to be honoured; a length
    mismatch is WARNED and falls back to an equal split across the surviving
    indices. ``None``, or any non-positive entry, also means an equal split.
    llama.cpp treats tensor_split entries as relative proportions, not values
    that must sum to 1.
    """
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
        # Detection unmeasurable (no torch, no nvidia-smi) OR the active native
        # backend is vulkan, where list_gpus() cannot see Vulkan-only devices: no
        # cross-check is possible, so the configured indices pass through.
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
            # index even when another index was dropped or de-duped above.
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
    """Free-VRAM-proportional split ratios for the configured
    ``gpu_split_indices``, or ``None`` when automatic distribution does not
    apply.

    Returns a list of positive floats aligned 1:1 BY POSITION with
    ``cfg["gpu_split_indices"]`` (the exact shape a configured
    ``gpu_split_ratios`` would have, so :func:`resolve_gpu_split`'s
    re-pair-by-original-position logic applies unchanged), normalized to sum
    1.0 and proportional to each device's CURRENT free VRAM. Callers pin the
    result into the isolated load worker via
    ``apply_gpu_split(ratios_override=...)``; the worker itself never probes.

    Returns ``None``, so the caller keeps the config-driven equal split, in
    every case where auto would be dishonest or unwanted:

    - Fewer than 2 configured indices, or non-integer indices: no split will be
      applied at all. Answered from config alone, with NO hardware probe.
    - ``gpu_split_ratios`` is explicitly configured: an explicit choice is
      never silently overridden.
    - Per-device free VRAM is not measurable for EVERY configured device
      (all-or-nothing, mirroring ``vram_capacity``'s "free" key).
    - The probe did not complete fresh this call (non-``GPU_PROBE_OK``).
    - (``list_gpus()`` path only) any configured device's reading is not
      device-global (``free_scope != FREE_SCOPE_DEVICE``).

    On the ``vulkan`` build the reading comes from :func:`native_gpu_devices`,
    the only per-device source in ggml-vulkan's index space, which is the space
    ``tensor_split`` actually consumes. Everywhere else the reading is
    ``list_gpus()``'s, reusing the caller-injected *gpus* snapshot when given.

    The ``list_gpus()`` branch REQUIRES every configured device's
    ``free_scope`` to be :data:`FREE_SCOPE_DEVICE`: a reading equally blind on
    every device makes an empty card and a nearly-full one look equally free,
    which can steer too much of a real split onto the full one. The vulkan
    branch is left ungated on scope, since :func:`native_gpu_devices` carries
    no such tag. Freshness needs no separate check on that branch either - its
    contract is fresh-or-``None`` with no last-known-good caching, so a
    non-``None`` reply is this call's own live round-trip to the probe daemon.

    A device reporting 0 bytes free keeps a tiny positive share (1-byte floor)
    instead of a 0.0 ratio, which ``resolve_gpu_split`` would discard along
    with the whole ratio list.

    The successful distribution, and a fallback on a configured-but-
    unmeasurable split, are logged at INFO."""
    from localm.config import load_config
    cfg = config if config is not None else load_config()
    indices = cfg.get("gpu_split_indices")
    if not indices or cfg.get("gpu_split_ratios"):
        return None
    try:
        idx_list = [int(i) for i in indices]
    except (TypeError, ValueError):
        # resolve_gpu_split itself warns and drops the split for this case.
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
        # The configured indices live in ggml-vulkan's own index space, so only
        # the native registry's reading can be paired with them.
        devices = native_gpu_devices()
        if devices is None:
            return _fallback("the native device registry did not answer")
        by_index = {d.get("index"): d for d in devices}
        for i in idx_list:
            d = by_index.get(i)
            if not isinstance(d, dict):
                # ABSENT, which is a different problem from UNMEASURABLE. These
                # devices are llama.cpp's own list (integrated GPUs and
                # accelerators removed, the rest renumbered), so a configured
                # index can legitimately point past the end.
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
            # wait_for_inflight (load-path callers pass True): joins a concurrent
            # probe rather than taking an instant BUSY plus a stale reading. Every
            # probing caller on this path is off the event loop.
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
            # Device-global or nothing: a PROPORTIONAL split cannot accept a
            # PROCESS-scoped (or untagged) reading. Real list_gpus() output always
            # carries this tag, so a missing tag is rejected the same as an
            # explicitly PROCESS-scoped one.
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
    """Float-slot count to allocate for ``tensor_split``: the native loader's
    own answer when available, else the documented fallback. Never smaller than
    *min_len* (the caller's highest configured device index + 1)."""
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
    """Set ``mp.split_mode``/``mp.tensor_split`` from the configured
    ``gpu_split_indices``/``gpu_split_ratios``, validated via
    :func:`resolve_gpu_split`. Leaves ``split_mode``/``tensor_split`` at their
    native defaults when fewer than 2 valid devices are configured. Shared by
    the llama.cpp chat backend and the embedder, same as ``apply_main_gpu``.

    THOSE NATIVE DEFAULTS ARE NOT A SINGLE-GPU LOAD.
    ``llama_model_default_params()`` sets ``split_mode =
    LLAMA_SPLIT_MODE_LAYER`` with ``tensor_split = NULL``, and llama.cpp
    confines a load to ``main_gpu`` only under ``LLAMA_SPLIT_MODE_NONE``, which
    nothing here ever sets. So leaving the defaults alone yields an IMPLICIT
    layer split across every registered GPU, distributed by each device's free
    memory. Anything sizing or budgeting a load must account for that: see
    :func:`implicit_split_capacity`.

    ``ratios_override`` (when non-empty) replaces the config's
    ``gpu_split_ratios`` for THIS load: it carries the PARENT's already-
    resolved effective ratios (:func:`resolve_auto_split_ratios`) into the
    isolated worker, which must not probe for them itself. It takes precedence
    over a config value read here. Validated by the exact same
    :func:`resolve_gpu_split` path as a configured value, so a malformed
    override degrades to the equal split with a WARNING rather than crashing.
    ``None`` or empty keeps the config-driven behaviour.

    Returns the ctypes float array backing ``mp.tensor_split``, or ``None``
    when no split was applied. THE CALLER MUST keep this referenced until after
    the ``llama_load_model_from_file()`` call that consumes *mp*: llama.cpp
    copies ``tensor_split``'s contents at load time and does not hold the
    pointer, so the buffer only needs to survive that one call."""
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
    """Call :func:`list_gpus` passing ONLY the kwargs the caller actually asked
    for.

    Many test modules patch list_gpus() with a zero-arg double
    (``lambda: gpus``), which its documented bare-list contract entitles them
    to. Forwarding ``deadline=None`` unconditionally would hand those doubles a
    kwarg they never agreed to accept and raise TypeError.
    ``wait_for_inflight`` is forwarded the same way, only when True."""
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
    """{"total": bytes, "free"?: bytes} for the CONFIGURED main GPU device (see
    main_gpu_index / resolve_main_gpu_index), or the largest GPU when none is
    configured, or {} when not measurable. Tries torch (CUDA/ROCm) then
    nvidia-smi (both via list_gpus()), then the Windows display-adapter
    registry, which is all the GGUF-only install without torch has (total is
    all fit_label needs).

    When ``return_status`` is True, returns ``(info, status)`` where ``status``
    is list_gpus()'s own GPU_PROBE_OK / GPU_PROBE_TIMEOUT / GPU_PROBE_BUSY /
    GPU_PROBE_INCONCLUSIVE. A caller that will present a specific number as
    CURRENT FACT (not just a fit ceiling) must check this rather than trust a
    timed-out probe's stale last-known-good fallback. Defaults to False, which
    returns the plain dict and makes a plain, no-kwarg list_gpus() call.

    ``deadline`` overrides list_gpus()'s default probe deadline. None keeps
    list_gpus()'s own default and keeps the call byte-identical for every
    existing caller.

    ``wait_for_inflight``: when a probe is already running, JOIN it and wait on
    its result up to ``deadline`` instead of being handed an instant
    last-known-good/BUSY. Only safe for a caller OFF the event loop."""
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
        # Look up by the index field, not list position (list_gpus() can have a
        # gap when one device fails to report); gpus[0] is a defensive fallback.
        g = next((x for x in gpus if x.get("index") == idx), gpus[0])
        out = {"total": g["total"]}
        if g.get("free") is not None:
            out["free"] = g["free"]
            # Travels WITH the number it describes, so a caller can tell a
            # whole-board figure from a process-local one. Absent when free is.
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
                                # The adapter's human name lives in the SAME key,
                                # and is what lets the device-global lookup below
                                # authorise an AMD single-adapter pairing by
                                # vendor. Absent on odd drivers -> empty string.
                                try:
                                    desc, _dt = winreg.QueryValueEx(key, "DriverDesc")
                                    best_desc = str(desc or "")
                                except OSError:
                                    best_desc = ""
                    except OSError as e:
                        # Access denied or a removed key. Surfaced under --debug
                        # so incomplete VRAM detection is diagnosable; the
                        # fallback is a note rather than a crash.
                        logger.debug("vram_info: registry subkey %s unreadable: %s",
                                     sub, e)
                        continue
            if best:
                out = {"total": int(best)}
                # The registry gives total but NO free. Recovers a DEVICE-GLOBAL
                # free from the ADL/PDH usage source, which works torch-less and
                # in-process (ADL for AMD, PDH's WDDM counter as the
                # vendor-neutral fallback). It maps ONLY when unambiguous (exactly
                # one AMD adapter for an AMD-named GPU, or exactly one WDDM
                # instance); otherwise total-only stands. The synthetic index 0
                # never feeds GPU SELECTION - it only carries the name so the AMD
                # pairing can be authorised.
                #
                # Runs ONLY when the probe COMPLETED empty (status OK), never on
                # TIMEOUT, BUSY or INCONCLUSIVE. status is None only when the
                # caller did not ask for it (return_status=False fit badges).
                if status not in (GPU_PROBE_TIMEOUT, GPU_PROBE_BUSY, GPU_PROBE_INCONCLUSIVE):
                    try:
                        from localm import gpu_usage
                        entry = {"index": 0, "name": best_desc, "total": int(best)}
                        u = gpu_usage.device_global_used_bytes([entry]).get(0)
                        if u is not None:
                            out["free"] = max(0, min(int(best), int(best) - int(u)))
                            out["free_scope"] = FREE_SCOPE_DEVICE
                    except Exception as e:
                        # Best-effort enrichment: a failed lookup degrades to
                        # total-only rather than losing the total.
                        logger.debug("vram_info: device-global free lookup failed: %s", e)
                return _ret(out)
        except Exception:
            pass
    return _ret({})


def vram_capacity(config: Optional[dict] = None, *, return_status: bool = False,
                  deadline: Optional[float] = None, wait_for_inflight: bool = False,
                  combined_only: bool = False):
    """{"total": bytes, "free"?: bytes} to weigh a model's fit against - the
    right ceiling for any "will this model fit" decision (a pre-load refusal
    gate, a fit badge, a VRAM-estimate readout).

    ``vram_info()`` alone is single-GPU and is the wrong ceiling once a
    multi-GPU ``gpu_split_indices`` is configured: a model too big for the
    single main GPU but that fits COMBINED across the configured split devices
    must not be refused or badged "too-big".

    Sums ``total``/``free`` across every device in :func:`resolve_gpu_split`'s
    validated split (via :func:`list_gpus`) when 2+ valid devices are
    configured. ``free`` is included only when EVERY split device reports a
    measurable free value, mirroring vram_info()'s own all-or-nothing "free"
    key. Falls back to :func:`vram_info` untouched (the single main-GPU number)
    whenever fewer than 2 valid split devices are configured or GPU detection
    is unmeasurable.

    ``return_status``: see :func:`vram_info` - propagated through both the
    single-GPU short-circuit and the split-summed path, so a caller weighing
    whether to trust a specific number as CURRENT fact can tell a fresh reading
    from a timed-out or stale one. Made ONLY when a caller opts in.

    ``deadline`` / ``wait_for_inflight``: see :func:`vram_info` - forwarded
    through ALL paths (the no-split short-circuit, the split-summed path, and
    the degrade-to-single-device fallback). Defaults keep list_gpus()'s own
    cold-init-tolerant deadline and no-join.

    ``combined_only``: return the summed figure or NOTHING (``{}``), never the
    single-device :func:`vram_info` fallback. For a caller budgeting a load
    that WILL be tensor-split across the configured devices, that fallback
    would silently substitute one device's capacity for the split's. Under
    ``combined_only`` the summed dict also carries ``"devices"`` (how many
    detected split devices were summed), so the caller can require a genuine
    2+-device sum; ``{}`` means no honest combined figure this call (no split
    configured, the split degraded to fewer than 2 detected devices, or -
    visible via ``return_status`` - a non-OK probe served stale data). The
    classic default shape is unchanged; ``"devices"`` is added ONLY under
    ``combined_only``.
    """
    from localm.config import load_config
    cfg = config if config is not None else load_config()

    def _vi():
        # Forward ONLY the opt-in kwargs the caller supplied, so the call stays
        # byte-identical to a bare vram_info() for the no-kwarg doubles.
        kw = {}
        if return_status:
            kw["return_status"] = True
        if deadline is not None:
            kw["deadline"] = deadline
        if wait_for_inflight:
            kw["wait_for_inflight"] = True
        return vram_info(**kw)

    # Short-circuit for the common no-split-configured case, skipping a hardware
    # probe. Under combined_only this is a conclusive, probe-free no-combined-
    # figure-exists answer, reported with GPU_PROBE_OK.
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
        # Split configured but degraded to a single detected device. Same
        # forwarding as the no-split path. Under combined_only, returns {} with
        # the probe's REAL status.
        if combined_only:
            return _ret({})
        return _vi()

    out = {"total": sum(g["total"] for g in split_gpus)}
    if combined_only:
        # Lets the caller require a genuine 2+-device sum: a plain dict lacking
        # this key reads as not a combined figure.
        out["devices"] = len(split_gpus)
    frees = [g.get("free") for g in split_gpus]
    if all(f is not None for f in frees):
        out["free"] = sum(frees)
        # All-or-nothing, mirroring the free key above: a sum is a whole-board
        # figure only if EVERY device in it is. Absent entirely when NO device
        # reported a scope, which means UNKNOWN.
        scopes = [g.get("free_scope") for g in split_gpus if g.get("free_scope")]
        if scopes:
            out["free_scope"] = (FREE_SCOPE_DEVICE
                                 if all(s == FREE_SCOPE_DEVICE for s in scopes)
                                 else FREE_SCOPE_PROCESS)
    return _ret(out)


def implicit_split_capacity(config: Optional[dict] = None, *,
                            wait_for_inflight: bool = False) -> dict:
    """``{"free", "total", "devices"}`` summed across every GPU device
    llama.cpp's DEFAULT layer split will spread a load over, or ``{}`` when no
    implicit split applies or it is not measurable.

    With no ``gpu_split_indices`` configured, :func:`apply_gpu_split` leaves
    ``split_mode``/``tensor_split`` at ``llama_model_default_params()``'s own
    values, and those are ``LLAMA_SPLIT_MODE_LAYER`` with ``tensor_split ==
    NULL``, NOT a single-GPU load. llama.cpp's "remove all except the main GPU"
    narrowing (``llama_prepare_model_devices``) is gated on
    ``LLAMA_SPLIT_MODE_NONE`` alone, which localm never sets, so ``main_gpu``
    does not confine the load. The device list is every registered discrete GPU
    (deduped by device id; integrated GPUs only when no discrete one exists). A
    ``NULL`` ``tensor_split`` then takes llama.cpp's default split by free
    memory: ``splits[i] = free_i``, normalized, with each layer assigned by
    ``upper_bound`` over the cumulative fractions, and the per-layer KV cache
    follows its layer's device.

    Because the weighting is by FREE MEMORY, device *i* receives the fraction
    ``free_i / SUM(free)`` of the offloaded layers, so a budget of
    ``SUM(free)`` places exactly ``free_i`` on device *i*: every card is filled
    to its own free memory and no further, which is what makes a HETEROGENEOUS
    set safe.

    Callers must still charge overhead PER DEVICE (each one carries its own
    compute buffers) - see ``_sizing.VramSizingMixin._split_overhead_bytes``.

    Separate from :func:`vram_capacity`, which answers for a CONFIGURED split
    and feeds the admission gate. Answers ``{}``, i.e. "no implicit combined
    figure - use the single-device reading", in every case where a sum would be
    dishonest:

    - A ``gpu_split_indices`` IS configured: an explicit ``tensor_split`` is
      written and :func:`vram_capacity` already owns that case. Answered from
      config alone, with NO hardware probe.
    - Fewer than 2 devices are detected.
    - Any device does not report BOTH ``free`` and ``total`` (all-or-nothing,
      mirroring :func:`vram_capacity`'s own "free" key).
    - (``list_gpus()`` path only) the probe did not complete fresh this call.

    On the ``vulkan`` build the reading comes from :func:`native_gpu_devices`,
    because that is the device space the layers are actually placed in;
    :func:`list_gpus` speaks torch's space and is structurally blind there.

    Never raises: a combined reading is an upgrade over the single-device one,
    and failing to fetch it must never break a load that worked without it."""
    from localm.config import load_config
    cfg = config if config is not None else load_config()
    if cfg.get("gpu_split_indices"):
        return {}
    if _native_backend_has_vulkan():
        devices = native_gpu_devices()
        if not devices:
            return {}
        # DISCRETE GPUs ONLY. llama.cpp's device list SKIPS accelerators outright
        # and appends integrated GPUs only when no discrete GPU was found.
        # native_gpu_devices already applies this same filter, so on a real
        # reading this pass is a no-op; it still runs for callers that patch
        # native_gpu_devices directly.
        #
        # Filters to GGML_DEV_TYPE_GPU rather than excluding the others by value,
        # since the enum has grown while CPU=0 and GPU=1 stayed stable. A device
        # whose type the probe did not report fails the filter, and if that leaves
        # fewer than 2 the single-device reading stands.
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
    # Logs WHICH budget was used and which per-device readings produced it, at
    # INFO so it reaches the always-on ring buffer. Runs per LOAD (the backend's
    # load-time preflights), not per poll.
    logger.info(
        "implicit GPU split: sizing against %d devices by free VRAM - %s "
        "(combined %.1f GB free / %.1f GB total)",
        out["devices"],
        ", ".join(f"device {d.get('index')}: {f / 1024 ** 3:.1f} GB free"
                  for d, f in zip(devices, frees)),
        out["free"] / 1024 ** 3, out["total"] / 1024 ** 3)
    return out


def split_device_count(config: Optional[dict] = None) -> int:
    """How many DETECTED devices the configured gpu_split resolves to - the
    DETECTED/labelling signal, NOT a load-safety gate.

    This is the exact signal ``vram_capacity()`` uses to decide whether its
    total is COMBINED across a split (>= 2) or the single main GPU (< 2): the
    same ``resolve_gpu_split`` plus detected-device re-filter. Callers that
    LABEL a VRAM number ("combined across N GPUs" vs "your main GPU's") must
    gate on this, not on the raw ``gpu_split_indices`` length, since a stale or
    typo'd index, or a GGUF-only box, leaves a 2-entry split resolving to one
    device.

    Do NOT use this to decide whether the loader will ACTUALLY apply a
    multi-device split: on the ``vulkan`` build the real split devices live in
    ggml-vulkan's own index space, which ``list_gpus()`` is structurally blind
    to, so the detected re-filter here COLLAPSES a live, working 2-way vulkan
    split to < 2. Use :func:`applied_split_device_count` for that question - it
    mirrors :func:`apply_gpu_split`'s own gate and does not apply the detected
    re-filter.

    Returns 0 when no split is configured (the common single-GPU path, with no
    hardware probe); otherwise the count of valid split devices (0/1 =
    effectively single, 2+ = combined)."""
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
    """How many devices the loader will ACTUALLY tensor_split across for a
    GGUF/llama.cpp load - the loader-truth counterpart to
    :func:`split_device_count`'s DETECTED/labelling count.

    Mirrors :func:`apply_gpu_split`'s own gate
    (``len(resolve_gpu_split(...)) < 2`` -> no split), so it answers "will a
    multi-device split be applied at load time", NOT "can we MEASURE that
    split's combined VRAM". The two counts differ on exactly one axis: the
    detected-device re-filter that :func:`split_device_count` /
    :func:`vram_capacity` apply against :func:`list_gpus` AFTER
    ``resolve_gpu_split``. On the ``vulkan`` build, where
    ``resolve_gpu_split`` passes the configured indices through UNVALIDATED in
    ggml-vulkan's own index space, this returns 2 while
    :func:`split_device_count` collapses to < 2. On a non-vulkan box with a
    detected device list the two are IDENTICAL.

    Deliberately does NOT pass ``gpus=`` (so ``resolve_gpu_split`` calls
    ``list_gpus()`` itself) and does NOT re-filter the result, exactly as
    :func:`apply_gpu_split` does.

    Returns 0 when no split is configured (the common path, no hardware probe);
    otherwise the count ``resolve_gpu_split`` yields. Domain is
    {0} U {2, 3, ...}: a single surviving index collapses to 0, same as
    ``apply_gpu_split`` leaving the native single-GPU default untouched."""
    from localm.config import load_config
    cfg = config if config is not None else load_config()
    if not cfg.get("gpu_split_indices"):
        return 0
    return len(resolve_gpu_split(
        cfg.get("gpu_split_indices"), cfg.get("gpu_split_ratios")))


def _list_gpus_reading(deadline: Optional[float] = None, *,
                       wait_for_inflight: bool = False) -> tuple:
    """``(gpus, status)`` from :func:`list_gpus`, tolerant of a test double
    patched in as a plain no-kwarg callable - the historical bare-list
    contract. A double whose signature does not accept ``return_status`` is
    called bare and its reading treated as :data:`GPU_PROBE_OK`, so only a real
    status-capable probe can ever report itself stale or busy here.
    Signature-inspected rather than a blanket ``except TypeError``, so a
    genuine ``TypeError`` raised INSIDE ``list_gpus`` is never mistaken for a
    rejected kwarg and swallowed. In production ``list_gpus`` always accepts
    ``return_status``, so the bare branch is a test-only affordance.

    *deadline* is forwarded to ``list_gpus`` only when given (None leaves its
    default cap untouched), so an OFF-event-loop caller can spend a longer
    budget on a cold driver init.

    *wait_for_inflight* (off-loop callers only - see :func:`list_gpus`) JOINS a
    probe another caller already holds instead of taking an instant BUSY plus a
    stale reading. Forwarded only when the callable's signature can accept it
    (a named parameter or ``**kwargs``)."""
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
    """``[{"index", "needed", "free"}, ...]`` for every configured split device
    whose free VRAM, read from a FRESH probe this call (:data:`GPU_PROBE_OK`),
    cannot cover its proportional share of *vram_required*. Empty when no split
    is configured, fewer than two split devices resolve, every device has live
    headroom, OR the live per-device check could not run this call. A shortfall
    entry is emitted ONLY under a fresh ``GPU_PROBE_OK`` reading, so every
    ``free`` in the result is a current measurement a caller may quote to the
    user as fact.

    ``vram_capacity()`` is an AGGREGATE check. With a PINNED
    ``gpu_split_ratios``, ``apply_gpu_split()`` divides a model by that static
    per-config ratio with NO live per-device capacity awareness, so a model too
    big for one device's actual share can still pass the aggregate check and
    reach llama.cpp's native loader with too little room on that device - not
    always a catchable Python exception, since the native loader can hard-abort
    the WORKER process rather than return NULL. Callers should treat a
    non-empty result on a pinned-ratio split as a hard refusal for a
    GGUF-backend load, not merely a warning.

    With ratios UNSET the parent pins :func:`resolve_auto_split_ratios`'s
    free-VRAM-proportional shares into the load, and this gate computes its
    per-device shares with the SAME auto ratios from its own fresh reading.
    Under those adaptive shares a device's proportional share fits its free
    whenever the aggregate fits, so a non-empty result means the COMBINED
    estimate is short. Auto can DECLINE (a configured index not currently
    detected, a device without a free reading) and fall back to the equal-share
    math, where that invariant does NOT hold. A caller deciding refuse-vs-defer
    MUST know which math produced the result: ``return_shares_adaptive=True``
    appends that fact - ``True`` only when live auto ratios were actually used
    for the shares, ``False`` for pinned ratios, the equal fallback, and every
    early return. Appended AFTER ``status`` when both opt-ins are set:
    ``(shortfall, status, shares_adaptive)``; alone:
    ``(shortfall, shares_adaptive)``. The bare-call shape is untouched.

    Probe freshness: ``list_gpus()`` is deadline-bounded and on a TIMEOUT/BUSY
    serves a FROZEN last-known-good reading. This gate does NOT compute a
    shortfall from a stale reading and does NOT refuse on one; on a non-OK
    probe it returns ``[]`` (best-effort admit, logged at debug), relying on
    the isolated worker's contained abort as the backstop. An empty bare-list
    result therefore cannot be told apart from "verified all-clear": a caller
    that must distinguish "checked, clear" from "could not check" MUST pass
    ``return_status=True`` to receive ``(shortfall, status)`` carrying
    :data:`GPU_PROBE_OK` / :data:`GPU_PROBE_TIMEOUT` / :data:`GPU_PROBE_BUSY`.

    Completeness (the blindness axis) is NOT gated on here. ``list_gpus`` tags
    each device :data:`FREE_SCOPE_DEVICE` (the board's number) or
    :data:`FREE_SCOPE_PROCESS` (counts ONLY this process's own allocations). A
    PROCESS-scoped reading OVER-states free, so in the REFUSE direction this
    gate governs, ignoring the tag is SOUND: if even the over-stated ``free``
    is short, the real free is shorter still. Do NOT omit a PROCESS-scoped
    device from the check - that trades a sound refusal for a permit, and the
    load then reaches llama.cpp too small and dies in the worker instead of
    returning a clean 503. The blindness that DOES bite is the PERMIT
    direction, and it is not detectable from the reading itself; a permit-side
    caution belongs with the aggregate gate that owns eviction.

    Only meaningful for the GGUF/llama.cpp load path - callers must gate on
    that themselves (e.g. via ``inference.engine._is_gguf``).

    Takes no headroom margin of its own: a device with EXACTLY enough free for
    its proportional share passes. A caller wanting the same safety margin the
    aggregate ``vram_capacity()`` check demands must add it to *vram_required*
    before calling.

    With ``return_status=True`` returns ``(shortfall, status)``; otherwise the
    bare ``shortfall`` list.

    *deadline* is forwarded to the underlying ``list_gpus`` probe (None leaves
    its default, which is cold-init tolerant). An on-loop caller must not probe
    inline at all - every server call site offloads via ``run_in_executor``.
    """
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
        # On the vulkan build the configured split indices live in ggml-vulkan's
        # own index space at load time, which list_gpus() cannot see or order, so
        # the per-device share check is skipped and the skip is logged at INFO
        # rather than presented as a check that passed.
        logger.info(
            "gpu_split_shortfall: skipping the per-device split VRAM preflight on "
            "the vulkan backend - the configured split indices are in ggml-vulkan's "
            "index space, which list_gpus() cannot map to a card, so no per-device "
            "check can name the right device (GPU-SPLIT-VKINDEX); relying on the "
            "subprocess-isolated loader to catch an oversized load instead.")
        # Conclusive skip with no probe, so it mirrors the no-split return above
        # and reports GPU_PROBE_OK.
        return _ret([], GPU_PROBE_OK)

    gpus, status = _list_gpus_reading(deadline)
    if status != GPU_PROBE_OK:
        # No FRESH reading this call: list_gpus served a frozen last-known-good
        # value (or []) after a probe TIMEOUT/BUSY. The live per-device check
        # could not run, so this admits best-effort, surfaced via debug and the
        # returned status, never a silent success.
        logger.debug("gpu_split_shortfall: probe status=%s (no fresh per-device VRAM "
                     "reading); admitting split load best-effort, per-device fit "
                     "unverified this call", status)
        return _ret([], status)

    # Judges each device by the share the loader will ACTUALLY give it. With
    # ratios unset the loader gets the auto free-VRAM-proportional split
    # (resolve_auto_split_ratios, computed from THIS SAME fresh reading), so a
    # non-empty result means the COMBINED estimate is short. Pinned ratios use the
    # static-share math instead. When auto declines, the equal-split math is used
    # and shares_adaptive stays False.
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
            # Structural guard for a malformed or absent device dict ONLY, not the
            # probe-could-not-run handler above. Under GPU_PROBE_OK, list_gpus
            # emits an int free for every device and DROPS a non-reporting one, so
            # this only stops a None-free entry from crashing the loop.
            continue
        # Not gated on g["free_scope"].
        needed = int(vram_required * (ratio / total_ratio))
        free = g["free"]
        if free < needed:
            shortfall.append({"index": idx, "needed": needed, "free": free})
    return _ret(shortfall, status, shares_adaptive)


def _device_choice_configured(cfg: dict) -> bool:
    """True when the user actually chose a device: a GPU split, or a Main GPU.

    ONE definition of "nothing configured", shared by
    :func:`resolve_preferred_device` and :func:`visible_device_order`. Both
    answer ``None`` in that case, and neither may probe the driver to find that
    out: the answer comes from config alone.
    """
    return bool(cfg.get("gpu_split_indices")) or cfg.get("main_gpu_index") is not None


def resolve_preferred_device(config: Optional[dict] = None, *,
                            gpus: Optional[list] = None) -> Optional[int]:
    """The device a media workload should DEFAULT to, with every OTHER card
    left VISIBLE. ``None`` when nothing is configured, or when no torch-visible
    device can be named honestly.

    NEVER use this to MASK the other cards away. ComfyUI core ships
    per-component GPU PLACEMENT - ``SelectModelDevice`` / ``SelectCLIPDevice``
    / ``SelectVAEDevice`` (``comfy_extras/nodes_multigpu.py``), which call
    ``deepclone_multigpu`` to rehome a component onto another card with
    independent weights. Masking to one device (ComfyUI's ``--cuda-device``, or
    a bare ``CUDA_VISIBLE_DEVICES=N``) deletes the other cards from torch's
    view and turns every one of those nodes into a silent no-op. Prefer
    ComfyUI's ``--default-device``, which reorders rather than masks, or
    :func:`visible_device_order` for an install we cannot pass argv to.

    The predicate is PREFERENCE, not exclusivity: which card should lead, not
    which card is the only one. It is NOT :func:`resolve_main_gpu_index`, which
    answers IDENTITY and resolves an unset value to device 0. On a configured
    split this is a CAPACITY-informed choice: the split device with the MOST
    live free VRAM.

    The answer is always a TORCH device index, because media runs on torch
    (ComfyUI) and :func:`list_gpus` enumerates via torch.cuda. A device is
    returned only when it is genuinely torch-visible; otherwise ``None``, and
    ComfyUI keeps its own default. :func:`resolve_gpu_split`'s Vulkan
    pass-through indices are in ggml-vulkan's index space and must never leak
    here.
    """
    from localm.config import load_config
    cfg = config if config is not None else load_config()
    split = cfg.get("gpu_split_indices")
    main = cfg.get("main_gpu_index")
    if not _device_choice_configured(cfg):
        return None                 # nothing configured: do not invent a device
    devices = gpus if gpus is not None else list_gpus()
    by_index = {g.get("index"): g for g in devices}
    if not by_index:
        # No torch-visible device at all, so any index named would be a guess in
        # an index space we cannot check. Let ComfyUI choose.
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
                # Split devices ARE torch-visible but none reports free VRAM. Lead
                # with the first visible one and SAY SO: it may not be the emptiest
                # card. Warned, not raised.
                logger.warning(
                    "gpu_split is configured but no split device reports free VRAM, so "
                    "the best card cannot be chosen for media; defaulting to device %d, "
                    "which may have less free VRAM than its peers.", visible[0])
                return visible[0]
            # NOT ONE configured split device is torch-visible; on the vulkan
            # build these are likely ggml-vulkan indices, which mean something
            # else to torch. Say so and let ComfyUI default.
            logger.warning(
                "gpu_split %r resolves to no torch-visible device, so media cannot name "
                "one: those indices are not in torch's index space (a Vulkan-only "
                "llama.cpp split does this). Leaving the device to ComfyUI's default.",
                split)
            return None
        # A split was configured but did not resolve to 2+ detected devices;
        # resolve_gpu_split() has already WARNED about the dropped indices. Fall
        # through to the main-index answer below.
    if main is None:
        return None
    idx = resolve_main_gpu_index(main, gpus=devices)
    return idx if idx in by_index else None


def visible_device_order(config: Optional[dict] = None, *,
                         gpus: Optional[list] = None) -> Optional[list]:
    """Every torch-visible device index with the PREFERRED one FIRST, or
    ``None`` when no device should be named.

    For a ComfyUI localm cannot pass argv to, where the child env is the only
    lever. Mirrors ComfyUI's own ``--default-device``, which REORDERS
    ``CUDA_VISIBLE_DEVICES``/``HIP_VISIBLE_DEVICES`` so the chosen device leads
    and leaves the rest visible, rather than ``--cuda-device``, which masks
    them away and silently disables core's ``Select*Device`` placement nodes.

    Every index is torch-visible by construction
    (:func:`resolve_preferred_device` and :func:`list_gpus` share torch's index
    space), so this never emits a Vulkan-space id.

    After this reorder the preferred card becomes torch index 0, so a
    workflow's ``gpu:N`` refers to the REORDERED position, not to localm's own
    ``list_gpus`` index. Anything emitting ``gpu:N`` into a workflow has to map
    through this order, not around it.
    """
    from localm.config import load_config
    cfg = config if config is not None else load_config()
    if not _device_choice_configured(cfg):
        # Nothing configured, so the answer is None either way: take it from
        # config and do NOT probe.
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
    """The ``gpu:N`` string ComfyUI will understand for OUR *device_index*, or
    ``None`` when it cannot be named honestly.

    Three coordinate systems meet here, and an off-by-one puts a component on
    the wrong card and STILL RENDERS:

    1. localm's own device index (``list_gpus()`` -> ``torch.cuda`` enumeration
       of the UNMASKED box). This is what ``gpu_split_indices`` and
       ``main_gpu_index`` mean.
    2. The VISIBLE ORDER imposed by :func:`visible_device_order`, written into
       ``CUDA_VISIBLE_DEVICES``/``HIP_VISIBLE_DEVICES`` either by ComfyUI's own
       ``--default-device`` or by us for a ComfyUI we cannot pass argv to.
    3. ComfyUI's ``gpu:N`` widget value, which is a POSITION, not a device id:
       ``get_gpu_device_options`` emits
       ``gpu:{i} for i in range(len(get_all_torch_devices()))``, and
       ``get_all_torch_devices`` enumerates torch AFTER the mask/reorder has
       applied.

    So ``gpu:N`` means "the Nth entry of the visible order", derivable because
    localm is the one that imposes that order. Masking collapses the order to
    one entry and ``get_gpu_device_options`` then emits no ``gpu:N`` at all (it
    gates on ``len(devices) > 1``), so every placement node silently no-ops.

    Returns ``None`` when no order is established (nothing configured, or no
    torch-visible device) or when *device_index* is not in it, rather than
    guessing a position. Callers must treat None as "do not emit a device for
    this component".
    """
    order = visible_device_order(config, gpus=gpus)
    if not order or device_index not in order:
        return None
    return f"gpu:{order.index(device_index)}"


def plan_media_placement(config: Optional[dict] = None, *,
                         gpu_options: Optional[list] = None) -> Optional[dict]:
    """Assign media components to cards for a box ComfyUI sees as 2+ GPUs, or
    ``None`` to keep the single-card floor. Pure: no I/O, no probe of its own.

    *gpu_options* is the LIVE ``gpu:N`` list read from the running ComfyUI's
    ``/object_info`` device combo
    (:func:`localm.media.comfy_client.probe_placement_capability`). It is
    authoritative about how many cards ComfyUI actually enumerates, and it is
    ComfyUI's OWN index space, so a POSITIONAL policy over it needs no
    localm-index translation and never consults ``split_device_count``.

    Policy: keep the big model on the preferred card (the first visible
    position, where ``--default-device`` already put the most-free card;
    ``"model": None`` means no injection, so the GGUF UNet is never moved off
    its loader default), and offload the smaller CLIP text-encoder and VAE to
    the SECOND visible card. No free-VRAM read is taken.

    Returns ``None`` when fewer than two ``gpu:N`` options exist (the
    single-card floor). Placement is capability-driven: it does not require a
    configured chat ``gpu_split`` - two visible cards is enough.
    """
    _ = config  # reserved for a future size/identity-aware policy; v1 is positional
    gpu = [o for o in (gpu_options or [])
           if isinstance(o, str) and o.startswith("gpu:")]
    if len(gpu) < 2:
        return None
    second = gpu[1]
    return {"model": None, "clip": second, "vae": second}


def fit_label(size_bytes: int, total_vram: Optional[int]) -> str:
    """
    Capacity badge for one file, against a single-GPU (or combined-split) VRAM
    ceiling: "fits" / "tight" / "too-big", or "" when VRAM is unknown. "tight"
    means it should load with little headroom (small context, nothing else on
    the GPU); "too-big" still runs, with some layers offloaded to system RAM
    (slower). That partial offload is delivered automatically: with
    n_gpu_layers_auto on (the default) the loader sizes how many layers fit
    from free VRAM at load time, so a "too-big" model loads instead of being
    refused.

    The need estimate here (weights * safety factor + fixed overhead) is
    context-agnostic and carries no explicit KV term, so it is a weights-fit
    signal rather than a guarantee for an unusually large -c/n_ctx. The
    loader's own preflight remains the authority.
    """
    if not total_vram or not size_bytes:
        return ""
    need = size_bytes * _WEIGHT_FACTOR + _OVERHEAD_BYTES
    if need <= total_vram * 0.85:
        return "fits"
    if need <= total_vram:
        return "tight"
    return "too-big"
