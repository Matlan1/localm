# SPDX-License-Identifier: AGPL-3.0-or-later
"""
In-app model discovery: search HuggingFace for GGUF models and judge,
per quantization, whether a file fits this machine's VRAM.

Discovery is a user-initiated prelude to ``localm pull`` and sits in the same
policy category (explicit user action - see docs/network.md): it is not
routed through the net_allow/net_deny domain rules, but ``net_mode = off``
still blocks it by default - ``net_allow_model_downloads`` exempts it, the
same override an explicit ``localm pull`` respects.

"Fits your VRAM" badges compare against TOTAL VRAM, not currently-free VRAM:
the active chat model occupies the GPU while you browse, and it will be
unloaded before the new one loads. The estimate mirrors the GGUF backend's
preflight: weights + ~1.5 GB overhead for KV cache and compute buffers.
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
    """Discovery failed - network off, HF unreachable, or repo unusable.

    *off* is True specifically for the net_mode=off refusal, so a caller that
    shows this to a GUI/API surface (a browser has no CLI to run) can
    substitute its own remedy instead of this message's CLI-flavored one -
    see ``localm/plugins/gui/routes/models.py``'s ``_run_discover``."""

    def __init__(self, message: str, *, off: bool = False):
        super().__init__(message)
        self.off = off


def _ensure_online() -> None:
    from localm.netpolicy import downloads_allowed_when_off, network_mode
    if network_mode() == "off" and not downloads_allowed_when_off():
        raise DiscoverError(
            "Network access is disabled (net_mode=off). Enable it with: "
            "localm config net_mode ask - or allow just downloads: "
            "localm config net_allow_model_downloads true",
            off=True)


def _hf_auth_headers(token: Optional[str]) -> Optional[dict]:
    """{"Authorization": "Bearer <token>"} when *token* is set, else None -
    HF's own REST API convention. Optional throughout: every call site below
    works anonymously when no token is configured."""
    return {"Authorization": f"Bearer {token}"} if token else None


def _get(url: str, params: Optional[dict] = None, *,
         token: Optional[str] = None) -> object:
    """Policy-checked GET returning parsed JSON.

    Routes through ``netpolicy.safe_fetch_bytes`` so the request is pinned to the
    validated IP and EVERY redirect hop is re-checked against the network
    policy: a DNS-rebind of the HF host, or a redirect from the HF API,
    cannot bounce discovery into a loopback / link-local / private address. The
    same protection the model-pull path uses."""
    import json as _json
    import urllib.parse

    from localm import netpolicy
    # doseq=True so a list-valued param (expand[]=safetensors&expand[]=downloads
    # ...) encodes as repeated keys, which is how the HF models API takes expand.
    full = url + ("?" + urllib.parse.urlencode(params, doseq=True) if params else "")
    try:
        _final, _ctype, body = netpolicy.safe_fetch_bytes(
            full, max_bytes=32 * 1024 * 1024, timeout=int(_TIMEOUT),
            allow_when_off=netpolicy.downloads_allowed_when_off(),
            extra_headers=_hf_auth_headers(token))
        return _json.loads(body.decode("utf-8"))
    except netpolicy.NetworkPolicyError as e:
        raise DiscoverError(f"HuggingFace request failed: {e}", off=e.off)
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
    """Classify a model_manager.registry MODEL_TYPES value from HARD HF metadata
    (pipeline_tag, library_name, exact tag tokens, GGUF architecture) - no
    network, pure function.

    ``architecture`` is the repo's ``gguf.architecture`` (or, for a non-GGUF
    result, ``config.model_type``) expand field - the model's OWN declared
    architecture, read from the file/config itself rather than a repo author's
    free-text tags. Optional and defaults to None so every existing 3-arg call
    site (``_hf_pipeline_tag_to_type``, which does not fetch it) is unaffected.

    Matching is EXACT, never substring: a tag that merely CONTAINS 'vae' / 'lora' /
    'clip' (e.g. 'exploration' contains 'lora') must NOT be misclassified.

    Order matters:

    - The exact-tag checks (vae/lora/text-encoder) run BEFORE every other check.
      A repo can carry a diffusion-flavored pipeline_tag (inherited from its base
      model) AND an exact 'lora'/'vae' tag at the same time - e.g. a FLUX LoRA has
      pipeline_tag=text-to-image (from the base checkpoint) and tags including
      'lora'. The tag is the more specific signal and must win, or every
      diffusion LoRA misclassifies as a full diffusion-unet.
    - ``architecture`` is checked next, before pipeline_tag/tagset: it is read
      straight from the model's own header, the single hardest signal this
      function has, so it outranks a pipeline_tag or tag a repo author set (or
      left stale/absent). Checked against the SAME verified embedding-only
      architecture allowlist a post-download GGUF header read uses
      (``localm.model_manager.gguf._GGUF_EMBEDDING_ARCHITECTURES``) - an exact
      allowlist membership test, so an architecture string that fails to match
      (a different naming convention, e.g. a non-GGUF config.model_type) just
      falls through to the pipeline_tag/tagset checks below rather than
      misclassifying anything. POSITIVE-EMBEDDING ONLY - there is no matching
      "these architectures mean llm" list, so a non-embedding architecture
      abstains and falls through to the tag layer.

    KNOWN FAILURE MODE of the architecture check: ``_GGUF_EMBEDDING_ARCHITECTURES``
    was built (and is verified) against llama.cpp's OWN ``LLM_ARCH_NAMES`` strings,
    read by this codebase's local post-download header parse. At search time,
    ``architecture`` instead comes from HF's SERVER-SIDE parse of the same GGUF
    header (the ``gguf.architecture`` expand field) - a different parser reading
    the same bytes. If HF ever normalizes that string differently from
    llama.cpp's own naming (case, punctuation, a renamed architecture), the exact
    match below fails SILENTLY: no exception, no wrong classification - just an
    abstain that falls through to the pipeline_tag/tagset checks, so a real
    embedding model would classify by tag alone instead of by its header
    architecture. Abstaining is a legal outcome, so nothing reports it; "why did
    this classify by tag instead of architecture" is the only symptom. HF's
    parser and llama.cpp's naming can drift apart independently at any time.

    Returns the 'unknown' sentinel - not a silent 'llm' - when no hard signal
    resolves, so an ambiguous result is never guessed into the wrong bucket."""
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
    """Classify a HuggingFace repo's model type by fetching its metadata and
    running it through classify_hf_metadata(). Returns 'unknown' - not a silent
    'llm' - on a failed/offline query, so an ambiguous pull is registered
    honestly and is not auto-loaded as the chat model."""
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
    safetensors param metadata (the ``safetensors`` expand field of the HF models
    API: ``{"total": <param count>, "parameters": {...}}``).

    localm's HF backend loads in bf16 on GPU with no on-load quantization (see
    inference/backends/hf.py), so the footprint is ``total_params * 2`` regardless
    of the STORED dtype - the loader casts to bf16. This is the weight size only;
    fit_label() adds KV-cache / compute overhead, the same way it does for a GGUF
    file size. Returns None when the repo has no usable param count, so the GUI
    shows "size unknown" rather than a guessed badge - an unknown is never
    treated as zero or as a fit."""
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
# DeepSeek-style repo carries neither convention in its name), so it is only
# ever the FALLBACK behind the header signal, and a caller labels a match from
# it as inferred, never as confirmed.
_MOE_NAME_RE = re.compile(r"(?i)\bmoe\b|\b\d+x\d+b\b|\ba\d+b\b")


def _moe_signal(architecture: Optional[str], repo_id: str) -> Optional[str]:
    """MoE-ness for a search-result row: ``"confirmed"`` (the model's own
    ``architecture`` string says so - reliable, see the module note above),
    ``"likely"`` (name pattern only - a guess, must be labelled as such in the
    GUI), or ``None`` (no evidence either way). Never returns a "dense" verdict:
    the absence of a MoE signal does not prove the model is dense."""
    if architecture and "moe" in str(architecture).lower():
        return "confirmed"
    if _MOE_NAME_RE.search(repo_id):
        return "likely"
    return None


def _param_count(row_fmt: str, gguf_meta: object, safetensors_meta: object) -> Optional[int]:
    """Total parameter count for a classified row, or None when unavailable.

    ``gguf.total`` (gguf-format rows) and ``safetensors.total`` (hf-format
    rows, the same field hf_param_bytes() already reads) are both the model's
    total parameter count, not a byte size: three quantizations of one repo
    report the same ``gguf.total`` while their file sizes differ several-fold.
    Same isinstance/positive guard as hf_param_bytes - a malformed or
    adversarial expand field degrades this row's count to None rather than
    crashing the rest of the search."""
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

    ``fmt`` given (bucket A / legacy path): every row is tagged with that one
    format, matching today's behavior exactly. ``fmt=None`` (bucket B/C, no
    format-split query): the format is derived from the item's OWN raw tags -
    the Hub-assigned "gguf" tag is a mechanical marker ("this repo contains
    .gguf files"), reliable independent of semantic classification.

    ``classify``: attach a ``detected_type`` (localm.model_manager.registry
    MODEL_TYPES value, or "unknown") from the item's pipeline_tag/library_name/
    tags fields for DISPLAY ONLY - never a filter on results. Also attaches
    ``architecture`` (the raw gguf.architecture/config.model_type string),
    ``moe`` ("confirmed"/"likely"/None, see _moe_signal), and ``param_count``
    (see _param_count) - all display-only, all omitted entirely when False, so
    a non-type-scoped caller's response shape is the plain one."""
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
    """HF query narrowing (a params fragment: ``filter=`` and/or
    ``pipeline_tag=``) for one (model_type, format) pair.

    ``model_type is None`` is the LEGACY path (CLI ``localm search`` / MCP
    ``search_models``): the plain per-format library tag - gguf -> "gguf",
    hf -> "transformers".

    Otherwise the gguf side is the reliable, type-independent "gguf" Hub tag
    (diffusion additionally ANDs "diffusers" so a gguf search isn't drowned by
    unrelated gguf repos), and the hf (safetensors) side narrows per type where
    HF exposes a reliable signal, else the plain "safetensors" format tag."""
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
                classify: bool, token: Optional[str] = None) -> list[dict]:
    """One HF /api/models query for a single (format, type), rows tagged *fmt*.

    ``classify`` requests the pipeline_tag/library_name/tags expand fields and
    attaches a ``detected_type`` badge to each row (display only, never a filter
    on results). Off for the legacy CLI/MCP path."""
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
        # pipeline_tag, so a classified gguf query asking only for
        # pipeline_tag/library_name/tags would drop downloads AND likes from
        # every row (both default-present with no expand at all).
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
    data = _get(f"{HF_API}/api/models", params, token=token)
    return _rows_from_items(data, limit, fmt=fmt, classify=classify)


def _spec_key(model_type: Optional[str], fmt: str):
    """Hashable identity of the HF request a (type, fmt) pair resolves to, so
    two selected types that produce the SAME query (e.g. vae + text-encoder both
    -> filter=safetensors on the hf side) fire ONE call, not two. Safe because a
    result's badge comes from its own metadata, not the query's type."""
    frag = _type_fmt_filter(model_type, fmt)
    return (fmt, tuple(sorted(
        (k, tuple(v) if isinstance(v, list) else v) for k, v in frag.items())))


def hf_search(query: str = "", limit: int = 20, formats: Sequence[str] = ("gguf",),
              model_type: Optional[str] = None,
              model_types: Optional[Sequence[str]] = None,
              token: Optional[str] = None) -> list[dict]:
    """Search HF for model repos. Empty query = most downloaded.

    Two independent axes, both from explicit GUI controls:

    - *formats*: a subset of {"gguf", "hf"} ("hf" == the non-gguf / safetensors
      world). One HF query runs per requested format.
    - *model_types*: which registry types to search for (a subset of
      _ALL_SEARCHABLE_TYPES). Each is narrowed server-side where HF exposes a
      reliable signal, and every result is badged with its detected type. When
      ALL searchable types are selected it collapses to the widest reliable
      format filter (2 queries), not a fan-out. *model_type* (singular) is the
      back-compat alias for a single-element *model_types*.

    Results across every (type, format) query are merged de-duped by repo id and
    round-robin interleaved so no single query crowds the others out of *limit*.

    Returns [{id, downloads, likes, updated, formats, size_bytes?, detected_type?}].
    ``detected_type`` is present only when a type was requested (display only,
    never a filter on results). With NEITHER *model_types* nor *model_type* (the
    CLI ``localm search`` / MCP ``search_models`` default) the query shape and
    response are the plain, unscoped ones.

    *token*: optional HF API token, sent as an Authorization header. Raises
    rate limits and lets gated repos appear in results; omitted, every query
    runs anonymously exactly as before."""
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
        for item in _run_query(query, limit, fmt, mt, classify, token=token):
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
    """True when the HF/transformers runtime can actually RUN a model here: both
    torch and transformers are importable. Uses importlib.util.find_spec, a cheap
    capability probe with no heavy import side effect.

    When False, an HF (transformers-format) model can still be DOWNLOADED via
    pull - it simply cannot be loaded until the ``.[gpu]`` extra (torch +
    transformers) is installed. The GUI surfaces exactly that, and does NOT block
    the download (a user may only want the files)."""
    import importlib.util
    try:
        return bool(importlib.util.find_spec("torch")
                    and importlib.util.find_spec("transformers"))
    except (ImportError, ValueError):
        # find_spec can raise on a half-installed namespace package; treat an
        # unresolvable probe as "not available" rather than crash discovery.
        return False


def hf_gguf_files(repo: str, token: Optional[str] = None) -> list[dict]:
    """
    List the GGUF files of *repo* with size and quant label. Split files
    (``-00001-of-0000N``) are grouped into one logical entry whose ``file``
    is the first part (what ``localm pull repo:file`` expects) and whose
    size is the sum of all parts. Sorted smallest-first.

    *token*: optional HF API token (see hf_search) - required for a gated
    repo's tree to be visible at all.
    """
    _ensure_online()
    repo = repo.strip().strip("/")
    if not re.match(r"^[\w.-]+/[\w.-]+$", repo):
        raise DiscoverError(f"Not a HuggingFace repo id: {repo}")
    tree = _get(f"{HF_API}/api/models/{repo}/tree/main", token=token)
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
    over any earlier one in the string. A mixed-precision MoE export commonly
    names the non-expert tensor precision FIRST (e.g.
    '...-bf16_MXFP4_MOE.gguf'), which would otherwise win under plain
    re.search - it returns the LEFTMOST match regardless of alternation
    order - misreporting the actual expert quantization as a plain
    unquantized BF16/F16 dtype."""
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
# the overrun is surfaced at debug level, never silently eaten.
#
# WHAT THE DEADLINE BOUNDS. No production caller probes ON THE EVENT LOOP: the
# GUI routes all run_in_executor, and the GPU-registry heartbeat's probe (via
# resolve_main_gpu_index -> list_gpus every ~20s when main_gpu_index >= 1) is
# likewise executor-offloaded. So the deadline does NOT protect the loop; it
# only bounds how long one worker thread (or a blocking CLI call) waits on a
# wedged driver before degrading.
#
# The default is COLD-INIT-TOLERANT. The first torch.cuda / HIP call of a
# process initializes the ROCm/CUDA driver, which takes several seconds, and a
# timeout is served as [] / a frozen last-known-good, which a bare-list caller
# cannot tell apart from "no GPU at all". So the deadline must sit ABOVE any
# legitimate cold init. The cost on a truly wedged driver is one worker thread
# parked for the deadline ONCE - the in-flight guard hands every concurrent
# caller an instant BUSY, and after the overrun the last-known-good path takes
# over - so nothing user-facing ever freezes for it.
#
# NOTE - NO freshness/TTL cache: every call re-probes. A TTL cache would hand a
# STALE "free" reading to callers that need a live one, most critically
# switch_engine's eviction loop, whose wait_for_vram_release polls free-VRAM to
# confirm a native free has landed before re-checking; a stale value there would
# defeat that guard and over-evict. The last-known-good value is kept ONLY as
# the wedge fallback, never to short-circuit a live probe.
_GPU_PROBE_DEADLINE = 15.0    # seconds a probe may block its caller; must exceed
                              # a legitimate COLD driver init (see above)
# Alias, kept for the call sites that opt into it by name (doctor,
# `localm gpus`, switch_engine). Same value as _GPU_PROBE_DEADLINE.
_GPU_PROBE_CLI_DEADLINE = _GPU_PROBE_DEADLINE

# Outcome of a probe, surfaced by list_gpus(..., return_status=True) so a caller
# can tell a slow / timed-out probe apart from a genuine "nothing here" reading
# and not misattribute the former. A user-facing "no GPU" message MUST branch
# on this.
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
# without ever probing.
_gpu_probe_done: Optional[threading.Event] = None
_gpu_probe_result: Optional[dict] = None
# Bumped by _reset_gpu_probe_cache() to ORPHAN any probe thread still in flight.
# An abandoned probe (see the DEADLINE note above) is by definition still running
# and will write its reading whenever the native call finally returns - which can
# be long after the reset. Clearing the globals alone cannot prevent that write,
# so a stale thread's result is fenced out by epoch instead of raced against.
_gpu_probe_epoch = 0


def last_known_gpus() -> list:
    """The most recent SUCCESSFUL :func:`list_gpus` reading, WITHOUT probing.

    ``list_gpus`` has no TTL cache - every call re-probes so a live ``free`` is
    never stale - which makes it the wrong thing to call for a second,
    incidental use right after something else has already probed. A probe spawns
    a torch-importing subprocess and costs seconds.

    This is for exactly that case: a caller that has JUST driven a probe (e.g. via
    :func:`vram_capacity`) and wants the per-device detail behind the number it
    already has. Returns ``[]`` when no probe has ever succeeded - never a
    fabricated or partial reading, and never a fresh probe.

    NOT a substitute for ``list_gpus`` when the reading must be current: the value
    here is as fresh as whatever last probed, and nothing about it says when.
    """
    return list(_gpu_last_good or [])


def _reset_gpu_probe_cache() -> None:
    """Test hook: drop the last-known-good GPU reading + in-flight flag, and
    INVALIDATE any probe still in flight so it cannot bleed into the next test.

    Clearing the globals is not enough on its own: an overrunning probe is
    abandoned, not cancelled (a wedged native call cannot be interrupted from
    Python), so that thread outlives this reset and would otherwise write its
    reading into _gpu_last_good AFTERWARDS: a cold ROCm/CUDA init can overrun
    _GPU_PROBE_DEADLINE, so the abandoned thread lands its write seconds later,
    inside whichever test is running by then. Bumping the epoch makes that late
    write a no-op (see _run), which the clears alone cannot do."""
    global _gpu_last_good, _gpu_probe_inflight, _gpu_probe_epoch
    global _gpu_probe_done, _gpu_probe_result, _isolated_torch_unavailable
    global _isolated_torch_broken_warned, _child_stderr_cap_reported
    with _gpu_probe_lock:
        _gpu_last_good = None
        _gpu_probe_inflight = False
        # Cleared with the rest of the probe state: a test (or a caller) resetting
        # the cache must get a clean slate, or one test's simulated spawn failure
        # would silently disable the torch path for every later test in the worker.
        _isolated_torch_unavailable = False
        _isolated_torch_broken_warned = False
        # The child-stderr latch has the SAME cross-test leak as the line above and
        # therefore belongs in the same reset: without it, one test's simulated
        # probe failure suppresses the stderr relay for every later test in the
        # worker, which reads as "the relay is broken" rather than "it already
        # said this once".
        _child_stderr_seen.clear()
        _child_stderr_cap_reported = False
        # Unpublish the join handles too: after a reset the slot reads free, so no
        # caller should join a probe from the epoch just retired. An abandoned
        # thread still holding its own local done/result is unaffected (it sets its
        # local event and, epoch-mismatched, will not touch these globals again).
        _gpu_probe_done = None
        _gpu_probe_result = None
        _gpu_probe_epoch += 1
    # The resolved source selection has the same cross-test leak as the latches
    # above: without this, one test's resident-HIP or rocm_sdk state would decide
    # the answer for every later test in the worker, and the skip decision would
    # stay announced so a later test could not observe it being made. Taken
    # outside _gpu_probe_lock: _source_selection_lock is a leaf and nothing is
    # ever held while acquiring it.
    global _native_hip_resident, _rocm_sdk_present, _torch_doomed_announced
    with _source_selection_lock:
        _native_hip_resident = None
        _rocm_sdk_present = None
        _torch_doomed_announced = None
    try:
        from localm import gpu_usage
        gpu_usage._reset_source_selection_notices()
    except Exception:
        # gpu_usage unimportable is a real bug, not an environment condition, but
        # a test-state reset must not be the thing that raises for it.
        logger.debug("gpu-probe reset: gpu_usage notice reset unavailable")


def list_gpus(*, deadline: float = _GPU_PROBE_DEADLINE, return_status: bool = False,
              wait_for_inflight: bool = False):
    """Every GPU device visible right now: ``[{"index", "name", "total",
    "free"}, ...]``, or ``[]`` when nothing is measurable.

    Safe by construction: the real driver probe (:func:`_list_gpus_probe`) runs on
    a helper thread with a hard ``deadline``-second timeout, so this call NEVER
    blocks its caller for longer than ``deadline`` even if the GPU driver wedges.
    Every call re-probes (see the module note above: no TTL cache, so a live
    "free" reading is never stale); on an overrun the last-known-good value (or
    ``[]``) is returned and the stuck probe thread is abandoned. The default
    ``deadline`` is generous enough to wait out a legitimate COLD
    driver init rather than misreport it (see the module note above); override it
    only in tests, or where a caller genuinely wants a faster degraded answer
    (:data:`_GPU_PROBE_CLI_DEADLINE` is an alias of the default).

    When ``return_status`` is True, returns ``(gpus, status)`` where ``status`` is
    :data:`GPU_PROBE_OK` (a fresh probe completed - an empty ``gpus`` then means
    genuinely no measurable GPU), :data:`GPU_PROBE_TIMEOUT` (the probe exceeded
    ``deadline`` - typically a cold ROCm/CUDA driver init that has not finished, so
    an empty ``gpus`` is INCONCLUSIVE and a retry with a longer deadline may
    succeed), :data:`GPU_PROBE_BUSY` (another probe is already inflight or the
    probe thread could not start; no fresh reading was taken), or
    :data:`GPU_PROBE_INCONCLUSIVE` (the probe completed, but the isolated torch
    enumeration could not be asked this round and nvidia-smi - the only other
    source - also found nothing; unlike TIMEOUT, a longer deadline will not help,
    since nvidia-smi's answer does not change with time). A caller that renders a
    user-facing "no GPU" message MUST branch on this so a slow cold probe, or an
    inconclusive one, is not misreported as "no torch / no GPU".
    ``return_status`` defaults to False, which is the bare-list contract every
    existing caller relies on.

    Tries torch first (CUDA/ROCm - torch's ROCm build aliases torch.cuda.* to
    HIP under the hood, so an AMD card enumerates through the exact same API,
    no special-casing needed) since it also gives a device name; falls back to
    a name-aware ``nvidia-smi`` listing (ALL devices, not just the first) for
    the GGUF-only install that has no torch.

    ``wait_for_inflight`` (opt-in, default False) changes ONLY what happens when a
    probe is already in flight: instead of returning :data:`GPU_PROBE_BUSY` at once
    with the last-known-good reading, this call JOINS the running probe and waits on
    its completion, bounded by its own ``deadline``. This is what makes a longer
    ``deadline`` actually help on a cold box: there the FIRST probe (typically the
    GUI's /api/stats heartbeat, executor-offloaded) holds the in-flight slot for
    the entire cold ROCm/CUDA init, so a model-load probe arriving in that window
    would otherwise short-circuit on BUSY without ever probing. Set it ONLY
    together with a long ``deadline`` and ONLY off the event loop: like the long
    deadline itself, a joining wait can block the caller up to ``deadline``
    seconds, which must never land on the server's single loop. It never spawns a
    second probe, so it cannot pile onto a wedged driver; a permanent wedge still
    just times the joiner out at its own ``deadline``.

    Does NOT fall back to the Windows display-adapter registry: that tier (see
    vram_info()) can only report one aggregate "largest adapter" number with no
    per-device identity, so it cannot support GPU *selection* - only
    vram_info()'s single-number "total VRAM for fit badges" use case."""
    gpus, status = _list_gpus_with_status(deadline, wait_for_inflight)
    return (gpus, status) if return_status else gpus


def _list_gpus_with_status(deadline: float, wait_for_inflight: bool = False) -> tuple:
    """The real probe driver behind :func:`list_gpus`, returning ``(gpus, status)``
    where status is one of :data:`GPU_PROBE_OK` / :data:`GPU_PROBE_TIMEOUT` /
    :data:`GPU_PROBE_BUSY` / :data:`GPU_PROBE_INCONCLUSIVE`. Split out so
    ``list_gpus`` can expose the status opt-in without duplicating the thread +
    deadline machinery. ``wait_for_inflight``: see
    :func:`list_gpus` - a patient off-loop caller JOINS a probe already in flight
    (bounded by ``deadline``) rather than short-circuiting on BUSY."""
    global _gpu_last_good, _gpu_probe_inflight, _gpu_probe_done, _gpu_probe_result
    global _probe_deadline_at   # published with the slot for the cold-budget check
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
            # the probe body reads it to decide whether it can afford a cold
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
                # Surfaced, not silenced; debug is the right altitude because
                # this is a consequence of a reset, not a fault.
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
                # Cleared with the in-flight slot too: the budget describes
                # THIS probe and nothing else. Leaving it set would hand a later
                # reader an expired deadline, which reads as "no budget left" and
                # would skip a cold source that in fact had all the time in the world.
                _probe_deadline_at = None
        # OUTSIDE the epoch gate and unconditional: a caller still
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
        # no thread to clear it), surface it at debug, and degrade to the
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
    # spawns no further threads. Surfaced, not silenced. The status is
    # TIMEOUT so a caller does not mistake an inconclusive probe for "no GPU".
    logger.debug("list_gpus: GPU probe exceeded %.1fs deadline (driver call stuck); "
                 "returning last-known GPU info so the caller does not block", deadline)
    with _gpu_probe_lock:
        served = list(_gpu_last_good) if _gpu_last_good is not None else []
        return served, GPU_PROBE_TIMEOUT


# Resolved source-selection state, guarded by _source_selection_lock. This lock is
# a LEAF: it is never held across a call out of this block, and no other lock in
# this module or in gpu_usage is acquired while it is held.
#
# _native_hip_resident: the native_hip_runtime_resident() answer once the glob has
# resolved it, or None while the native lib is unloaded and it is still open.
# _rocm_sdk_present: the find_spec("rocm_sdk") answer, or None before it is taken.
# _torch_doomed_announced: the last _torch_gpu_probe_known_doomed() answer written
# to the log, or None before anything has been.
_source_selection_lock = threading.Lock()
_native_hip_resident: "bool | None" = None
_rocm_sdk_present: "bool | None" = None
_torch_doomed_announced: "bool | None" = None


def native_hip_runtime_resident() -> bool:
    """True when llama.cpp's bundled HIP-linked runtime is resident IN THIS
    process on Windows: the native lib has been loaded (``_loader.load_lib``)
    and the resolved runtime ships a HIP ggml backend (same shipped-DLL-set
    authority as :func:`_native_backend_has_vulkan`).

    This is the platform signal for TWO distinct conclusions, each taken by its
    own caller with its own extra narrowing:

    - :func:`_torch_gpu_probe_known_doomed` below: a FRESH ``import torch``
      here collides with the resident HIP DLLs (it adds the torch-absence and
      ``rocm_sdk`` conditions on top).
    - ``gpu_usage.raw_reading_is_process_scoped``: the raw free-VRAM readings
      this process can take are HIP-sourced, and the HIP runtime's reading on
      Windows is the blind one (``ggml_backend_dev_memory`` and torch's
      ``mem_get_info`` are byte-identical and equally blind; see gpu_usage's
      module docstring) - so blindness can be answered truthfully even where
      torch itself cannot be consulted at all (the GGUF worker).

    Fails closed (False) when the check itself errors: both callers treat
    False as "no special handling", today's behavior.

    Once the native lib is loaded, the answer is LATCHED for the life of the
    process (cleared only by :func:`_reset_gpu_probe_cache`), EITHER WAY: a
    resolved True and a resolved False are both kept, so the directory glob and
    the ``runtime_binary_dir()`` resolution behind it run at most once per
    process rather than on every call. The latch relies on this INVARIANT: a
    native lib that has been loaded is never unloaded and the DLL set it shipped
    with does not change, so neither answer can flip once the glob has read it.

    The answer is NOT latched while the lib is unloaded, because it can still
    load later. That path returns False on the ``native_lib_loaded()`` check
    alone and never reaches the glob, so re-deriving it is cheap.

    While the answer is latched it no longer re-resolves
    ``runtime_binary_dir()``, so it can no longer drift from the dir the
    resident lib actually loaded from."""
    global _native_hip_resident
    import sys
    if sys.platform != "win32":
        return False
    with _source_selection_lock:
        if _native_hip_resident is not None:
            return _native_hip_resident
    try:
        from localm.inference.backends.llamacpp import _loader
        if not _loader.native_lib_loaded():
            return False
        d = _loader.runtime_binary_dir()
        resident = d is not None and any(
            "hip" in p.name.lower() for p in d.glob(_loader._ggml_glob()))
    except Exception as e:
        logger.debug("native-HIP-resident check failed (%s); answering False",
                     type(e).__name__)
        return False
    with _source_selection_lock:
        _native_hip_resident = resident
    return resident


def _rocm_sdk_installed() -> bool:
    """Whether ``rocm_sdk`` is importable, resolved once per process.

    The answer is latched (cleared only by :func:`_reset_gpu_probe_cache`) so the
    ``sys.path`` walk ``find_spec`` performs runs at most once, rather than on
    every GPU probe. Latching relies on this INVARIANT: a package is not
    installed or removed part-way through a process."""
    global _rocm_sdk_present
    with _source_selection_lock:
        if _rocm_sdk_present is not None:
            return _rocm_sdk_present
    import importlib.util
    present = importlib.util.find_spec("rocm_sdk") is not None
    with _source_selection_lock:
        _rocm_sdk_present = present
    return present


def _announce_torch_doomed(doomed: bool) -> bool:
    """Record *doomed* as the current torch-consultability answer, and return
    whether it CHANGED what was last announced.

    True means the caller should write its log line: either nothing has been
    announced yet, or the answer has flipped since it was. False means this
    answer is a repeat and the line would say what the log already says.

    A flip in either direction re-arms the announcement, so a decision that
    becomes doomed for a fresh reason is never hidden by an earlier one."""
    global _torch_doomed_announced
    with _source_selection_lock:
        changed = _torch_doomed_announced != doomed
        _torch_doomed_announced = doomed
    return changed


def _torch_gpu_probe_known_doomed() -> bool:
    """True when :func:`_list_gpus_probe`'s ``import torch`` attempt below is
    KNOWN, ahead of time, to fail in this exact process state - so the probe
    skips it at the root instead of triggering the failure and catching the
    aftermath.

    THE DOOMED COMBINATION (root-caused live, and documented with the same
    skip in ``_loader.native_lib_loaded`` / ``_sizing._free_total_vram_bytes``):
    on Windows, once llama.cpp's bundled HIP-linked runtime has been loaded
    into this process (anything that reaches ``_loader.load_lib()`` -
    ``compute_devices()`` / ``has_max_devices()``, a worker, a mixed test
    run), its bundled ROCm/HIP DLLs are resident under the same names a
    ROCm-for-Windows torch resolves during import via its ``rocm_sdk``
    preload. The OS loader hands torch the already-resident, ABI-incompatible
    copies and the import fails with STATUS_ENTRYPOINT_NOT_FOUND (0xc0000139).
    The failure is caught below and the probe degrades to nvidia-smi, but each
    attempt prints a "Windows fatal exception" faulthandler trace to stderr,
    and Python evicts the faulted module from ``sys.modules`` - and list_gpus
    re-probes on every call (see the no-TTL note above), so the doomed import
    re-runs and re-traces for the rest of the process's life. A concurrent
    second import can even take the process down outright
    (``gpu_usage.raw_reading_is_process_scoped``); never starting the doomed
    import removes that trigger as well.

    NARROWER THAN _sizing's blanket ``native_lib_loaded()`` skip: _sizing can
    skip torch outright because its fallback, ``gpu_memory_isolated()``,
    answers exactly as well.
    THIS probe's fallback is nvidia-smi, which cannot see AMD devices, so a
    blanket skip would trade away real, working torch enumeration on every
    setup where torch and a resident native runtime coexist. Each condition
    below narrows the skip to the PROVEN-doomed combination - where the torch
    attempt fails every time, so skipping provably loses nothing - and any
    setup outside it keeps today's behavior, torch attempt included:

    - torch not already resident in ``sys.modules``: a resident torch was
      imported successfully (before the runtime loaded, or on a setup where
      the two coexist) and importing it again is a free cache hit - no
      preload runs, nothing can fault, and its working enumeration is kept.
    - :func:`native_hip_runtime_resident` (Windows + the native lib loaded +
      the resolved runtime ships a HIP ggml backend): the conflict is Windows
      OS-loader same-name resolution against resident HIP DLLs. Nothing
      resident yet means no conflict - a fresh process (the common probe
      context) keeps its torch enumeration - and a vulkan/cpu/cuda build
      leaves no HIP DLLs resident for torch's preload to collide with. If
      the shipped-DLL-set authority ever proves wrong for some exotic build,
      the cost is today's pre-guard noise, never a lost probe.
    - ``rocm_sdk`` is importable: the failing preload belongs to the
      ROCm-for-Windows torch; a CPU/CUDA torch (or no torch at all) never
      runs it. Importability is necessary, not sufficient (the rocm-sdk
      wheels also serve the HIP llama build itself), but firing with a
      non-ROCm torch loses nothing material: a CPU torch enumerates no
      CUDA devices, and a CUDA torch's NVIDIA devices are exactly what the
      nvidia-smi fallback reports anyway.

    Fails OPEN: if the detector itself errors, the probe proceeds with its
    normal torch attempt (which catches its own failures) - detection must
    never break the working path. The skip is surfaced at debug level, not
    silenced, the first time it is decided and again whenever the decision
    changes; an unchanged repeat is not re-logged.

    Both of the inputs that cost anything to evaluate - the resident-HIP glob
    and the ``rocm_sdk`` ``sys.path`` walk - are resolved once per process, so
    a caller polling this on every GPU probe re-runs neither. Only the
    ``sys.modules`` membership test is re-evaluated, which is what lets the
    answer still flip when torch becomes resident."""
    import sys
    if "torch" in sys.modules:
        # A resident torch (imported for real before the runtime loaded, or a
        # test's injected stand-in) makes `import torch` a plain cache hit: no
        # rocm_sdk preload runs, so the conflict cannot occur and the working
        # enumeration must be kept. On the doomed combo torch can never BE
        # resident - the faulted module is evicted on every attempt - so this
        # never defuses the real guard.
        _announce_torch_doomed(False)
        return False
    try:
        if not native_hip_runtime_resident():
            _announce_torch_doomed(False)
            return False
        if not _rocm_sdk_installed():
            _announce_torch_doomed(False)
            return False
    except Exception as e:
        logger.debug("list_gpus: torch-conflict detector failed (%s); "
                     "proceeding with the normal torch attempt", type(e).__name__)
        _announce_torch_doomed(False)
        return False
    if not _announce_torch_doomed(True):
        return True
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
# Must fit INSIDE _GPU_PROBE_DEADLINE together with the nvidia-smi fallback's own
# timeout, and must sit above a legitimate cold driver init.
_ISOLATED_TORCH_PROBE_TIMEOUT = 10.0

# Bound on the pipe drain attempted after killing a wedged isolated-torch-probe
# child. See _torch_gpus_isolated: subprocess.run's own post-kill drain on
# Windows has no timeout of its own and can block forever, so this module
# calls Popen.communicate() directly instead and bounds BOTH attempts itself.
# Kept well under _ISOLATED_TORCH_PROBE_TIMEOUT so the two together still fit
# inside _GPU_PROBE_DEADLINE.
_ISOLATED_TORCH_DRAIN_TIMEOUT = 3.0

# Latched True once the out-of-process torch enumeration proves it CANNOT answer
# on this box (spawn failure, timeout, unusable reply). Read/written under
# _gpu_probe_lock, cleared by _reset_gpu_probe_cache. This is not a reading cache
# (see the no-TTL note above): what is remembered is "torch cannot be asked here",
# never a VRAM number, so nothing stale can reach switch_engine's eviction loop.
_isolated_torch_unavailable = False

# Latched once the isolated probe has been reported BROKEN. Unlike
# _isolated_torch_unavailable, which disables a capability, this only suppresses a
# repeated log line: broken isolation keeps retrying (it still enumerates,
# in-process), so the warning would otherwise repeat on every probe.
_isolated_torch_broken_warned = False


def isolated_torch_unavailable() -> bool:
    """True once the isolated probe has PROVEN, in a child process, that torch
    cannot finish enumerating on this box (see
    :func:`_torch_gpus_isolated_once`, which sets the latch this reads).

    Public because the conclusion is not this module's alone to act on. The
    latch's own contract - "retrying this import IN-PROCESS would reproduce the
    multi-minute startup hang the isolation exists to prevent, so this one must
    never fall back that way" - binds every OTHER caller that was about to
    ``import torch`` on a hot path too, and until this reader existed the only
    way to honour it was to be inside this module.

    Specifically it binds ``_sizing.VramSizingMixin._free_total_vram_bytes``,
    which sits on the model-LOAD path: an unbounded ``import torch`` there, on a
    box that has just proven torch wedges, wedges the whole load.

    False means "not proven unavailable", NOT "torch works" - the probe may
    simply never have run. It is a reason to SKIP an attempt, never evidence
    that an attempt will succeed, so a caller still needs its own bound.
    """
    with _gpu_probe_lock:
        return _isolated_torch_unavailable


# The child probe's stderr routinely starts with a long virtualenv install-path
# prefix from Python's own warnings.warn() formatting
# (``<path>:<line>: <Category>: <message>``), which on its own can exceed a
# short cap and leave the actionable message body (e.g. "The following list of
# GPU architectures compatible with this version of PyTorch...") cut off. The
# cap is generous rather than a truncation from a fixed front or back: this
# stderr can carry either a warnings.warn() message (the point comes AFTER the
# file:line: Category: prefix) or an uncaught exception's traceback (the point
# is the LAST line), so a truncation direction that helps one shape guts the
# other. Any truncation beyond this limit is marked, never silent.
_CHILD_STDERR_LOG_CAP = 2000


def _capped_stderr(text: str, limit: int = _CHILD_STDERR_LOG_CAP) -> str:
    """*text* (child-probe stderr), capped to *limit* chars for a log line.
    Marks the cut explicitly when it actually truncates - a silently cut
    diagnostic is the same rule-5 shape as swallowing it outright."""
    if len(text) <= limit:
        return text
    return text[:limit] + f"... [truncated, {len(text) - limit} more chars]"


# Distinct child-stderr texts already reported this process, and whether the cap
# below has been announced. Guarded by _gpu_probe_lock (probes can overlap).
_CHILD_STDERR_SEEN_CAP = 8
_child_stderr_seen: set[str] = set()
_child_stderr_cap_reported = False


def _child_stderr_once(err: str) -> "str | None":
    """The child's stderr, capped, the FIRST time this exact text is seen this
    process. ``None`` once it is a repeat, so the caller can leave it out.

    LATCHED because ``list_gpus`` re-probes on every call (no TTL, so the live
    "free" reading is never stale) and the GUI's VRAM meter polls it roughly
    every 2.5s, so on a box where the probe keeps failing an unconditional relay
    of the child's whole stderr blob writes it about 24 times a minute for the
    life of the server - the same log flood the
    ``_isolated_torch_broken_warned`` latch further down prevents.

    KEYED ON THE TEXT rather than a plain once-only bool like that neighbour:
    that latch guards a FIXED sentence, while here the message IS the
    diagnostic and a second, DIFFERENT failure carries real information. Keying
    on the text kills the repeat and keeps the change.

    The cap bounds a pathological case (stderr that varies every probe, e.g. one
    carrying a timestamp or an address) rather than a realistic one - a genuinely
    broken box repeats one text. Reaching it is announced rather than silently
    going blind.
    """
    global _child_stderr_cap_reported
    if not err:
        return None
    capped = _capped_stderr(err)
    with _gpu_probe_lock:
        if capped in _child_stderr_seen:
            return None
        if len(_child_stderr_seen) >= _CHILD_STDERR_SEEN_CAP:
            if _child_stderr_cap_reported:
                return None
            _child_stderr_cap_reported = True
            return (f"[{_CHILD_STDERR_SEEN_CAP} distinct probe failures already "
                    "logged this process; further distinct causes suppressed]")
        _child_stderr_seen.add(capped)
    return capped


class _IsolatedTorchWedged(Exception):
    """The out-of-process torch probe ran but did not finish in time, i.e. TORCH
    ITSELF is wedging on this box (the sm_120 case). Distinct from the child
    mechanism being broken, and the distinction decides the fallback: retrying
    this import IN-PROCESS would reproduce the multi-minute startup hang the
    isolation exists to prevent, so this one must never fall back that way."""


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

    Returns the device list (possibly ``[]``, a real answer meaning torch sees no
    CUDA/HIP device), or ``None`` when the child COULD NOT ANSWER at all - spawn
    failure, timeout, or an unusable reply. The caller falls through to nvidia-smi
    either way, but only ``None`` latches :data:`_isolated_torch_unavailable`, so
    a box where torch simply has no device is not mistaken for one where torch
    cannot be asked - "nothing there" and "could not look" never collapse into
    one silent path. The child inherits this process's
    environment, so ``CUDA_VISIBLE_DEVICES`` selects and orders devices
    identically and the TORCH index space :func:`list_gpus` promises is
    preserved.

    Spawned via ``interpreter_for_localm_children()``, NOT bare
    ``sys.executable``: inside a Windows multiprocessing-spawn worker the latter
    is the BASE interpreter, whose children get no venv context and so cannot
    import torch or localm at all (the same trap documented on
    ``_loader._spawn_probe_daemon``)."""
    import json
    import subprocess
    from localm._mp_spawn import interpreter_for_localm_children
    try:
        proc = subprocess.Popen(
            [interpreter_for_localm_children(), "-u", "-m",
             "localm._torch_gpu_probe"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except Exception as e:
        logger.debug("list_gpus: could not spawn the out-of-process torch probe "
                     "(%s); falling through to nvidia-smi", type(e).__name__)
        return None
    try:
        stdout, stderr = proc.communicate(timeout=_ISOLATED_TORCH_PROBE_TIMEOUT)
    except subprocess.TimeoutExpired:
        # Surfaced, not silenced: this is the wedged-driver case, and a silent
        # [] here is indistinguishable from "this box has no GPU".
        proc.kill()
        try:
            # subprocess.run's own Windows kill-path calls communicate() a
            # SECOND time with NO timeout of its own to drain the pipes
            # (verified directly against CPython's subprocess.run source: the
            # comment there reads "communicate() _after_ kill() is required
            # to collect that"). If this probe's child left a grandchild
            # alive holding the inherited pipe handle, that drain never sees
            # EOF and blocks forever - which would wedge the shared
            # single-flight probe lock (_gpu_probe_inflight) permanently for
            # the rest of the process's life, well past this function
            # returning. Bound the drain ourselves so a killed-but-not-fully-
            # reaped child cannot do that; give up on its output rather than
            # risk an unbounded wait.
            stdout, stderr = proc.communicate(timeout=_ISOLATED_TORCH_DRAIN_TIMEOUT)
        except subprocess.TimeoutExpired:
            stdout, stderr = "", ""
        logger.debug("list_gpus: out-of-process torch probe did not answer "
                     "within %.1fs; falling through to nvidia-smi",
                     _ISOLATED_TORCH_PROBE_TIMEOUT)
        raise _IsolatedTorchWedged() from None
    except Exception as e:
        proc.kill()
        logger.debug("list_gpus: could not spawn the out-of-process torch probe "
                     "(%s); falling through to nvidia-smi", type(e).__name__)
        return None
    err = (stderr or "").strip()
    raw = (stdout or "").strip()
    if not raw:
        # The child ALWAYS prints one line, "[]" included on its own failure path,
        # so empty stdout means it died before printing (killed, hard crash, a
        # native fault taking the process down). That is COULD NOT ASK, not
        # "torch sees no device" - collapsing the two would report "no GPU" on a
        # box whose GPU torch can see perfectly well.
        said = _child_stderr_once(err)
        logger.debug("list_gpus: out-of-process torch probe printed nothing "
                     "(rc=%s)%s; treating as unavailable, not as 'no device'",
                     proc.returncode, f"; child said: {said}" if said else "")
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
        said = _child_stderr_once(err)
        logger.debug("list_gpus: out-of-process torch probe reply unusable "
                     "(%s)%s; falling through to nvidia-smi", e,
                     f"; child said: {said}" if said else "")
        return None
    if err:
        # The child prints its own failure cause here before answering [], so it
        # must not die with the discarded stream. Latched: this line carries
        # NOTHING but the stderr, so once it is a repeat there is nothing left
        # worth writing.
        said = _child_stderr_once(err)
        if said:
            logger.debug("list_gpus: out-of-process torch probe reported: %s", said)
    return devices


def _torch_gpus_isolated_once() -> list:
    """:func:`_torch_gpus_isolated`, but never retried on a box that has already
    proven it cannot answer. Returns the device list, or ``[]`` so the caller
    falls through to nvidia-smi.

    The latch is what keeps a wedged torch from costing the FULL timeout on every
    single probe: `list_gpus` re-probes on every call (no TTL, see above), so
    without this a wedged torch pays the full timeout on every probe and never
    reaches the fallback inside the caller's deadline.

    TWO FAILURES THAT LOOK ALIKE AND MUST NOT BE TREATED ALIKE:

    - TORCH WEDGES (timeout). Isolation worked and told us torch cannot finish
      here. Latch, and never retry in-process - that import is precisely the
      multi-minute hang this whole change exists to remove.
    - ISOLATION IS BROKEN (cannot spawn, unusable reply). We learned nothing
      about torch. Falling through to nvidia-smi would SILENTLY LOSE real GPU
      enumeration on any box nvidia-smi cannot see - every AMD and Intel box -
      turning "we could not look" into a confident "no GPU". So degrade to the
      IN-PROCESS import and say plainly at WARNING that the isolation was lost
      and the stall risk is back. A safety net for a genuine runtime failure,
      not the design.

    Once latched, this still returns [] and the probe still falls through to
    nvidia-smi. On a box where nvidia-smi ALSO cannot answer - an AMD or Intel
    card whose torch wedges - an empty reading is "could not determine", not
    "genuinely no GPU", and that distinction is propagated from this same latch:
    :func:`_list_gpus_with_status` reads
    ``_isolated_torch_unavailable`` once this call returns, and reports
    :data:`GPU_PROBE_INCONCLUSIVE` instead of :data:`GPU_PROBE_OK` exactly when
    the reading came back empty AND this latch is set - never for a non-empty
    reading (nvidia-smi finding real hardware, e.g. the sm_120 case this
    isolation exists for, is conclusive regardless of the latch).

    This function's OWN return type is a bare ``list``, never ``None``: every
    caller of :func:`_list_gpus_probe`, and every double that monkeypatches it,
    relies on that contract. The status channel is a separate, additive path
    through module state, not a change to this function's signature.

    OUT OF SCOPE: :func:`_torch_gpu_probe_known_doomed` skips the torch attempt
    ENTIRELY on its narrower doomed combination (Windows + resident HIP runtime
    + rocm_sdk torch) without touching this latch, so that skip is not detected
    as inconclusive here either. Closing it would need :func:`_list_gpus_probe`
    itself to track conclusiveness across every source it tries, not just this
    one."""
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


_TORCH_RESIDENT_READ_TIMEOUT = 3.0


def _torch_gpus_resident_bounded(timeout: float = _TORCH_RESIDENT_READ_TIMEOUT) -> list:
    """Like :func:`_torch_gpus_resident`, but bounded and safe to call from a
    caller with its own deadline.

    ``"torch" in sys.modules`` becomes True the moment ANY thread starts
    importing torch, before that import finishes: CPython inserts a module
    into ``sys.modules`` before running its body, so a concurrent
    ``_torch_is_resident()`` check can read True while another thread is still
    deep inside torch's own (slow, on some ROCm/HIP builds) DLL-loading
    ``__init__.py``. Calling :func:`_torch_gpus_resident` in that window does
    a bare ``import torch``, which blocks on Python's per-module import lock
    for the remaining duration of that other thread's import.

    Runs the read in a background thread with a short bound; on overrun,
    returns ``[]`` so the caller falls through exactly as if torch had not
    been resident at all. Never raises."""
    result: dict = {}
    done = threading.Event()

    def _read() -> None:
        try:
            result["value"] = _torch_gpus_resident()
        except Exception:
            pass
        finally:
            done.set()

    try:
        threading.Thread(target=_read, name="localm-torch-gpus-resident",
                         daemon=True).start()
    except Exception:
        return []
    if not done.wait(timeout):
        return []
    return result.get("value", [])


def _list_gpus_probe() -> list:
    """The actual (blocking) GPU driver probe. Call :func:`list_gpus`, not this -
    this one has no timeout and can wedge on a busy/broken driver."""
    if not _torch_gpu_probe_known_doomed():
        try:
            out = _torch_gpus_resident_bounded() if _torch_is_resident() \
                else _torch_gpus_isolated_once()
            if out:
                _apply_device_global_free(out)
                return out
        except Exception:
            pass

    try:
        import subprocess
        proc = subprocess.Popen(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            nvidia_smi_out, _ = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            # See _torch_gpus_isolated: subprocess.run's own Windows kill-path
            # drains the pipes with a SECOND communicate() call that carries
            # no timeout of its own, which can block forever. Bound the
            # post-kill drain here the same way.
            proc.kill()
            try:
                nvidia_smi_out, _ = proc.communicate(timeout=_ISOLATED_TORCH_DRAIN_TIMEOUT)
            except subprocess.TimeoutExpired:
                nvidia_smi_out = ""
        if proc.returncode == 0 and nvidia_smi_out.strip():
            out = []
            for line in nvidia_smi_out.strip().splitlines():
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
# rather than risk overrunning the probe deadline. The cold open costs about
# 750ms, and this is that with margin, since overrunning costs the caller its
# free reading entirely. See _apply_device_global_free.
_CORRECTION_COLD_BUDGET_S = 1.5

# When the in-flight probe's deadline expires (monotonic), or None outside a probe.
# Set by _list_gpus_with_status under the same lock that claims the in-flight slot,
# so the probe body can tell how much of its budget is left before spending ~750ms
# on a cold source. Safe as a module global precisely because _gpu_probe_inflight
# serialises probes: only ever one in flight to describe.
_probe_deadline_at = None


def _apply_device_global_free(gpus: list) -> None:
    """Correct each entry's ``free`` to a DEVICE-GLOBAL figure where this platform's
    driver query is not one already, and tag every entry with ``free_scope`` so a
    caller can tell a whole-board number from a process-local one. Mutates *gpus*.

    On Windows with an AMD ROCm/HIP torch build, ``torch.cuda.mem_get_info``
    reports ``total - the calling process's own allocations`` and is blind to
    every other process. Not a staleness bug (the probe here is FRESH and still
    wrong), and not llama.cpp-specific: a plain torch tensor in a child process
    is equally invisible. It matters here because every GGUF load is
    out-of-process (backends/gguf.py), so the model's own VRAM is ALWAYS in
    another process from the server measuring it - as is a game or a ComfyUI.

    On Linux, and on NVIDIA, the driver query is device-global BY DOCUMENTATION (CUDA
    specifies *free as "free according to the OS" and warns that another process can
    move it), so nothing is corrected there and the reading is tagged
    :data:`FREE_SCOPE_DEVICE` unchanged.

    When no better source can answer on Windows, the entry keeps the driver's number
    but is tagged :data:`FREE_SCOPE_PROCESS` rather than silently passing a
    known-process-local figure off as the board's. That tag is what makes
    /v1/models/unload say its reading is uncertain instead of asserting a wrong
    one as fact."""
    import sys
    if sys.platform != "win32":
        for g in gpus:
            g["free_scope"] = FREE_SCOPE_DEVICE
        return

    # The scope to use when a device-global correction is NOT available for an entry
    # (source cold-skipped, unmappable, or failed). Tag PROCESS only where the raw
    # reading is KNOWN blind (Windows + an AMD ROCm/HIP torch build); elsewhere on
    # Windows the raw cudaMemGetInfo is device-global by documentation (NVIDIA), so
    # tagging it PROCESS would assert a blindness that does not hold and raise a
    # spurious uncertainty flag. Computed up front so it is defined on every path
    # below, including the import-failure except.
    try:
        from localm.gpu_usage import raw_reading_is_process_scoped
        uncorrected_scope = (FREE_SCOPE_PROCESS if raw_reading_is_process_scoped()
                             else FREE_SCOPE_DEVICE)
    except Exception:
        # gpu_usage unimportable is a real bug, not an environment condition, but
        # it must not crash a probe. Conservative default: DEVICE - never assert a
        # blindness that cannot be confirmed.
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
        # guard matters only to a caller that passes a short deadline; a warm
        # source is free and always runs.
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
    """True when the currently-resolved native runtime directory ships the
    Vulkan ggml backend (a ``ggml-vulkan.*`` file) - i.e. the active install
    is the ``vulkan`` build.

    ``list_gpus()`` (above) enumerates ONLY via torch.cuda (CUDA, or HIP under
    a ROCm-build torch) or nvidia-smi - it never calls the Vulkan loader, so it
    is structurally blind to any device only visible through Vulkan. On the
    vulkan build, the REAL device selection at load time happens entirely
    inside ggml-vulkan/llama.dll's own enumeration, a different index space
    list_gpus() cannot see or validate against: list_gpus() reports a non-empty
    but VULKAN-INCOMPLETE device list rather than an empty one, which drops a
    valid configured split device and a valid configured main_gpu_index alike.
    The two callers below handle "empty" as "unmeasurable, pass through
    unchecked", and this handles "non-empty but for the wrong backend" the same
    way.

    Checks the actual shipped DLL/SO set, NOT the ``.localm-backend``
    provisioning marker (setup_llama.py): the marker can be absent (a
    ``--from`` build, an install predating the marker) or generic (e.g.
    ``"custom"`` for a ``--url``/``--sha256`` provision) - the real file set
    is always authoritative for which backend will actually be loaded."""
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

    THE TWO SEQUENCES ARE NOT THE SAME.
    ``_loader.native_device_inventory`` is a faithful registry
    inventory: it numbers EVERY non-CPU device in raw
    ``ggml_backend_dev_get`` order. Upstream
    ``llama_prepare_model_devices`` instead builds ``model->devices`` as:

        RPC-backed devices, hoisted to the FRONT
        + GPU-type devices in registry order, deduplicated by device_id
        + at most ONE integrated GPU, and ONLY when no discrete GPU was found
        CPU and ACCEL devices are SKIPPED; META aborts fatally

    So on a box with a discrete card beside integrated graphics - an ordinary
    laptop, or any desktop CPU with an iGPU - the inventory carries a device
    llama.cpp's list does not. If the iGPU enumerates first, EVERY index is
    off by one.

    RPC hoisting and device_id dedup cannot arise here, so neither is
    emulated: ``ggml_backend_rpc_add_server`` is never called anywhere in this
    project (so the RPC backend registers zero devices even though its library
    ships and loads), and ggml-vulkan already dedups one physical GPU seen
    under two drivers by ``deviceUUID``/``deviceLUID`` before it reaches the
    registry, on a build that provisions one GPU backend at a time.

    ALLOWLISTS ``GPU`` RATHER THAN EXCLUDING THE OTHERS BY VALUE, as
    ``implicit_split_capacity`` does: the enum has GROWN (IGPU was inserted
    AHEAD of ACCEL, so one value has meant ACCEL on an older runtime and
    INTEGRATED GPU on a newer one) and this module cannot know which llama.cpp
    is provisioned, so an allowlist is version-independent where a denylist is
    not. A device whose type the probe did not report fails the filter rather
    than being assumed discrete.

    WHEN NO GPU-TYPE DEVICE IS PRESENT THE LIST IS RETURNED UNCHANGED. An
    iGPU-only box (a very common laptop) has llama.cpp fall back to its single
    integrated GPU as device 0, which is exactly what this inventory already
    numbers 0, so the two agree and a load works. Returning an empty list there
    would hide a working device behind a "no GPU here" reading. Identifying an
    IGPU device POSITIVELY would need the unstable enum value above."""
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
    ``gpu_split_indices`` / ``main_gpu_index`` actually means at load time -
    on the ``vulkan`` build the only source that can express it at all
    (:func:`list_gpus` is structurally blind to it). That
    is NOT simply the registry's own numbering: llama.cpp drops integrated
    GPUs whenever a discrete card exists and skips accelerators outright, so
    the raw inventory from ``_loader.native_device_inventory`` is passed
    through :func:`_llama_visible_devices` first, which keeps the devices the
    loader will really use and renumbers them into the space it indexes. See
    that helper for the upstream construction and for the
    iGPU-only case. Every consumer here wants
    that same list: the GUI selectors write these numbers into config,
    :func:`resolve_auto_split_ratios` pairs a configured index BACK to a
    device by it, and :func:`implicit_split_capacity` sums over the set.

    This is the enumeration source for the GUI's split/main-GPU SELECTORS on
    that build. NOT merged into :func:`list_gpus`: its torch/nvidia-smi index
    space feeds the torch-side reads (:func:`vram_capacity`'s per-device sums,
    :func:`gpu_split_shortfall`), and the two index spaces must never be
    mixed.

    ``name`` prefers the registry's human description ("AMD Radeon RX 6900
    XT...") over the backend's terse name ("Vulkan0"). ``total``/``free`` are
    included only when the registry reported positive bytes - the GUI drops
    its size suffix for an absent key rather than showing "0.0 GB"."""
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
    """The GPU device index to actually use, given the user's ``main_gpu_index``
    config value.

    None (not configured) resolves to device 0 - today's behaviour - with no
    detection work done at all. An explicitly configured index is validated
    against the devices ``list_gpus()`` (or the injected *gpus*, for tests)
    currently sees: an index that does not match any of them is a real
    problem (silently substituting the wrong GPU, or handing llama.cpp's
    native loader an index past the end of its device array, is worse than
    device 0), so it is surfaced as a WARNING and swapped for device 0 rather
    than trusted blindly.

    An index above ``_MAX_GPU_SPLIT_INDEX`` is rejected unconditionally, the
    same sanity ceiling :func:`resolve_gpu_split` applies to its indices -
    checked BEFORE any device-membership branching below, so it still applies
    when detection is unmeasurable or skipped (see next paragraph).

    When detection itself is unmeasurable (``list_gpus()`` returns nothing -
    no torch, no nvidia-smi) OR the active native backend is ``vulkan``
    (whose real device enumeration list_gpus() cannot see at all - see
    :func:`_native_backend_has_vulkan`), the configured index cannot be
    cross-checked against a reliable, backend-matching device list either
    way; it is passed through unchecked (aside from the ceiling above) rather
    than discarding an explicit user choice we have no way to disprove (the
    same documented boundary as the Windows-registry VRAM fallback)."""
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
    # when the active backend is vulkan: list_gpus() is
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
    """Set ``mp.main_gpu`` from the configured ``main_gpu_index``, validated via
    :func:`resolve_main_gpu_index`. Leaves the native default (0, set by
    ``llama_model_default_params()``) untouched when unset. Shared by the
    llama.cpp chat backend and the embedder so both native-load call sites
    honour the same selection with the same fallback/warning behaviour."""
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
    """Validate a configured multi-GPU split (``gpu_split_indices`` /
    ``gpu_split_ratios``) against the devices ``list_gpus()`` (or the injected
    *gpus*, for tests) currently sees, returning ``[(index, ratio), ...]``
    ready to write into ``tensor_split``.

    Mirrors :func:`resolve_main_gpu_index`'s posture: an index that does not
    match a currently-detected device is dropped with a WARNING rather than
    trusted blindly - a stale config
    referencing a since-removed GPU degrades to single-GPU instead of
    mis-targeting VRAM or crashing a load. Duplicate indices keep their first
    occurrence. Fewer than 2 valid indices after validation means "no split"
    (returns ``[]``) - the single-GPU path driven by ``apply_main_gpu`` is
    unaffected. This validation is SKIPPED (indices pass through unchecked)
    when the active native backend is ``vulkan`` - see
    :func:`_native_backend_has_vulkan`: ``list_gpus()``
    cannot see Vulkan-only devices, so on that backend a non-empty result here
    is not authoritative: acting on it collapses a configured split to
    single-device and replaces the user's ``gpu_split_ratios`` with
    llama.cpp's own unrelated auto-split.

    ``configured_ratios``, when given, must be the SAME LENGTH as
    ``configured_indices`` (before validation) to be honoured - a length
    mismatch is a real misconfiguration (WARNED), not something to silently
    truncate/pad, so it falls back to an equal split across the surviving
    indices. ``None`` (or any non-positive entry) also means an equal split;
    llama.cpp treats tensor_split entries as relative proportions, not values
    that must sum to 1, so "equal" here is simply the same weight per device.
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
        # Detection unmeasurable (no torch, no nvidia-smi) OR the active
        # native backend is vulkan (list_gpus() cannot see Vulkan-only devices,
        # so a non-empty result here would not be authoritative for this
        # backend's index space): same boundary as resolve_main_gpu_index -
        # no cross-check is possible either way, so the configured indices pass
        # through unchanged.
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
    """Free-VRAM-proportional split ratios for the configured
    ``gpu_split_indices``, or ``None`` when automatic distribution does not
    apply - the parent-side decision behind "query free vram from each card,
    compare and distribute".

    Returns a list of positive floats aligned 1:1 BY POSITION with
    ``cfg["gpu_split_indices"]`` (the exact shape a configured
    ``gpu_split_ratios`` would have, so :func:`resolve_gpu_split`'s
    re-pair-by-original-position logic applies unchanged), normalized to sum
    1.0 and proportional to each device's CURRENT free VRAM. Callers pin the
    result into the isolated load worker (``gguf.py`` -> ``GgufWorker`` ->
    ``LlamaCpp``; ``IsolatedEmbedder._reload`` -> ``GGUFEmbedder``) via
    ``apply_gpu_split(ratios_override=...)`` - the worker itself never probes
    (a torch import inside a native-runtime process is the Windows + AMD DLL
    conflict this exists to prevent, and only the parent holds the
    device-global corrected readings anyway).

    ``None`` (caller keeps today's config-driven behavior, i.e. the equal
    split) in every case where auto would be dishonest or unwanted:

    - Fewer than 2 configured indices, or non-integer indices: no split will
      be applied at all (``resolve_gpu_split`` warns/degrades on its own).
      Answered from config alone, with NO hardware probe.
    - ``gpu_split_ratios`` is explicitly configured: the user pinned the
      shares, and an explicit choice is never silently overridden.
    - Per-device free VRAM is not measurable for EVERY configured device
      (all-or-nothing, mirroring ``vram_capacity``'s "free" key): guessing a
      share for a blind device could overload it.
    - The probe did not complete fresh this call (non-``GPU_PROBE_OK``): a
      frozen last-known-good snapshot is never distributed over, matching
      ``gpu_split_shortfall``'s probe-freshness contract.
    - (``list_gpus()`` path only) any configured device's reading is not
      device-global (``free_scope != FREE_SCOPE_DEVICE``) - see the
      TRUSTWORTHINESS section below.

    On the ``vulkan`` build the reading comes from
    :func:`native_gpu_devices` (the isolated probe daemon's view of ggml's
    own registry) - the ONLY per-device source in ggml-vulkan's
    index space, which is the space ``tensor_split`` actually consumes
    (``list_gpus()`` is structurally blind there and
    speaks torch's index space). Everywhere else the reading is
    ``list_gpus()``'s, reusing the caller-injected *gpus* snapshot when given
    (``gpu_split_shortfall`` passes its own fresh ``GPU_PROBE_OK`` reading,
    so gate and shares are computed from ONE snapshot).

    TRUSTWORTHINESS, per branch:

    * Freshness is a non-issue on BOTH branches. The ``list_gpus()`` branch's
      ``GPU_PROBE_OK`` check above is the explicit form of it;
      :func:`native_gpu_devices` needs no such check: its contract (see its own
      and ``_probe_roundtrip``'s docstrings) is fresh-or-``None`` with NO
      last-known-good caching at all - a non-``None`` reply is this call's own
      live round-trip to the probe daemon, so there is no "served stale" state
      to distinguish.
    * Scope DOES differ by branch. ``list_gpus()``'s entries are scope-tagged
      by :func:`_apply_device_global_free`, where Windows plus an AMD ROCm/HIP
      torch build reports free VRAM blind to every other process - so this
      branch REQUIRES every configured device's ``free_scope`` to be
      :data:`FREE_SCOPE_DEVICE` before trusting the proportion, unlike
      ``gpu_split_shortfall``'s refuse-only use of the identical reading.
      :func:`native_gpu_devices` carries no such tag, and nothing establishes
      that ggml-vulkan's own ``ggml_backend_dev_memory`` query is
      cross-process blind, so the vulkan branch is left UNGATED on scope.

    A device reporting 0 bytes free keeps a tiny positive share (1-byte
    floor) instead of a 0.0 ratio: ``resolve_gpu_split`` discards the WHOLE
    ratio list on any entry <= 0, which would silently hand a completely
    full card an EQUAL share.

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
        # The configured indices live in ggml-vulkan's own index space, so only
        # the native registry's reading can be paired with them; a list_gpus()
        # (torch-space) reading here would compute shares for the WRONG cards.
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
                # configured index can legitimately point past the end.
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
    """Float-slot count to allocate for ``tensor_split``: the native loader's
    own answer when available (authoritative - see the capacity comment
    above), else the documented fallback. Never smaller than *min_len* (the
    caller's highest configured device index + 1)."""
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
    ``llama_model_default_params()`` sets
    ``split_mode = LLAMA_SPLIT_MODE_LAYER`` with ``tensor_split = NULL``, and
    llama.cpp confines a load to ``main_gpu`` only under
    ``LLAMA_SPLIT_MODE_NONE`` - which nothing here ever sets. So leaving the
    defaults alone yields an IMPLICIT layer split across every registered GPU,
    distributed by each device's free memory. Anything sizing or budgeting a
    load must account for that: see :func:`implicit_split_capacity`.

    ``ratios_override`` (when non-empty) replaces the config's
    ``gpu_split_ratios`` for THIS load: it carries the PARENT's already-
    resolved effective ratios (:func:`resolve_auto_split_ratios`) into the
    isolated worker, which must not probe for them itself (see that
    function's docstring). It takes precedence over a config value read
    here - the parent's admission gate checked THOSE shares, and a config
    edited between the parent's read and this one must not produce a split
    the gate never saw. Validated by the exact same
    :func:`resolve_gpu_split` path as a configured value (a malformed
    override degrades to the equal split with a WARNING, never a crash).
    ``None``/empty keeps the config-driven behavior byte-identical to before
    the kwarg existed.

    Returns the ctypes float array backing ``mp.tensor_split`` (or ``None``
    when no split was applied) - the CALLER MUST keep this referenced until
    after the ``llama_load_model_from_file()`` call that consumes *mp*:
    llama.cpp copies ``tensor_split``'s contents at load time (it is not held
    as a live pointer afterward), so the buffer only needs to survive that one
    call, not the loaded model's lifetime."""
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
    """Call :func:`list_gpus` passing ONLY the kwargs the caller actually asked for.

    Test doubles patch list_gpus() with a zero-arg callable (``lambda: gpus``),
    which its documented bare-list contract permits.
    Forwarding ``deadline=None`` unconditionally would hand those doubles a kwarg
    they never agreed to accept and raise TypeError in tests with no stake in this
    change. Omitting it keeps the default call byte-identical, so only a caller
    that opts in pays for opting in. ``wait_for_inflight`` is forwarded the same
    way - only when True."""
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
    registry - the GGUF-only install has no torch, and the fit badges must
    still work there (total is all fit_label needs).

    When ``return_status`` is True, returns ``(info, status)`` where ``status``
    is list_gpus()'s own GPU_PROBE_OK/GPU_PROBE_TIMEOUT/GPU_PROBE_BUSY/
    GPU_PROBE_INCONCLUSIVE - a caller that will present a specific number as
    CURRENT FACT (not just a fit
    ceiling) must check this rather than trust a timed-out probe's stale
    last-known-good fallback (the vram_before/after bytes /v1/models/unload
    reports are exactly that case).
    ``return_status`` defaults to False, preserving the plain-dict contract
    (AND the plain, no-kwarg list_gpus() call) every existing caller and test
    double relies on - the status-aware call is made ONLY when a caller opts
    in, never unconditionally.

    ``deadline`` overrides list_gpus()'s default probe deadline (which is already
    cold-init-tolerant - see :data:`_GPU_PROBE_DEADLINE`). None keeps
    list_gpus()'s own default, and
    keeps the call byte-identical for every existing caller. Callers that pass
    :data:`_GPU_PROBE_CLI_DEADLINE` explicitly do so to PIN their cold-init
    tolerance against any future default change, not to get a different value.

    ``wait_for_inflight`` (opt-in): when a probe is already running (e.g. the
    GUI's 2.5s stats heartbeat holds it through a cold init), JOIN it and wait on its
    result up to ``deadline`` instead of being handed an instant last-known-good/BUSY.
    Only safe for a caller OFF the event loop. Forwarded, not defaulted, for the same
    byte-identical-call reason as ``deadline``."""
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
                # unmeasurable; the pre-load gate treats that as "skip the VRAM
                # check", and surfacing an independent ADL number there
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
    """{"total": bytes, "free"?: bytes} to weigh a model's fit against - the
    right ceiling for any "will this model fit" decision (a pre-load refusal
    gate, a fit badge, a VRAM-estimate readout).

    ``vram_info()`` alone is single-GPU by design (see its docstring) and is
    the wrong ceiling once a multi-GPU ``gpu_split_indices`` is configured: a
    model too big for the single main GPU but that fits COMBINED across the
    configured split devices must not be refused or badged "too-big" just
    because the capacity check only ever looked at one device (the bug this
    function fixes - a model refused/mis-badged despite a working split).

    Sums ``total``/``free`` across every device in :func:`resolve_gpu_split`'s
    validated split (via :func:`list_gpus`) when 2+ valid devices are
    configured; ``free`` is included only when EVERY split device reports a
    measurable free value (mirrors vram_info()'s own all-or-nothing "free" key
    - a partially-measurable split must not silently under-count by treating a
    missing device's free as 0). Falls back to :func:`vram_info` untouched
    (single main-GPU number) whenever fewer than 2 valid split devices are
    configured or GPU detection is unmeasurable (registry-fallback tier) -
    resolve_gpu_split already warns and degrades a stale/invalid split to
    single-GPU; this reuses that same validation rather than duplicating it.

    ``return_status``: see :func:`vram_info` - propagated through both the
    single-GPU short-circuit and the split-summed path, so a caller weighing
    whether to trust a specific number as CURRENT fact (not just a fit
    ceiling) can tell a fresh reading from a timed-out/stale one. Made ONLY
    when a caller opts in (never unconditionally), so every existing caller
    and test double that patches vram_info()/list_gpus() with a plain, no-kwarg
    stand-in keeps working exactly as before.

    ``deadline`` / ``wait_for_inflight``: see :func:`vram_info` - forwarded through
    ALL paths below (the no-split short-circuit, the split-summed path, and the
    degrade-to-single-device fallback), so a blocking (non-event-loop) caller gets
    the same longer probe budget and join behaviour whether or not a split is
    configured. Defaults keep list_gpus()'s own cold-init-tolerant deadline and
    no-join.

    ``combined_only`` (opt-in): return the summed figure or NOTHING (``{}``) -
    never the single-device :func:`vram_info` fallback. For a caller budgeting a
    load that WILL be tensor-split across the configured devices (the GGUF
    backend's sizing preflight, ``llamacpp/_sizing.py``), the single main-GPU
    fallback is not a degraded answer but a wrong one: it would silently
    substitute one device's capacity for the split's, exactly the split-blind
    bug that layer exists to avoid, and the caller could not tell the two apart
    from the dict shape alone. With ``combined_only`` the summed dict also
    carries ``"devices"`` (how many detected split devices were summed), so the
    caller can require a genuine 2+-device sum; ``{}`` means "no honest combined
    figure this call" (no split configured, the split degraded to fewer than 2
    detected devices, or - visible via ``return_status`` - a non-OK probe served
    stale data). The classic (default) shape is unchanged; the ``"devices"``
    key is added ONLY under ``combined_only``.
    """
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
    """``{"free", "total", "devices"}`` summed across every GPU device
    llama.cpp's DEFAULT layer split will spread a load over, or ``{}`` when no
    implicit split applies or it is not measurable.

    THE IMPLICIT SPLIT IS REAL, AND IT IS NOT WHAT THIS PROJECT ASSUMED. With
    no ``gpu_split_indices`` configured, :func:`apply_gpu_split` leaves
    ``split_mode``/``tensor_split`` at ``llama_model_default_params()``'s own
    values - and those are ``LLAMA_SPLIT_MODE_LAYER`` with ``tensor_split ==
    NULL``, NOT a single-GPU load. Read from upstream source (llama.cpp's
    ``llama_prepare_model_devices``): the "remove all except the main GPU"
    narrowing is gated on ``LLAMA_SPLIT_MODE_NONE`` alone, which localm never
    sets, so ``main_gpu`` does not confine the load. The device list is every
    registered discrete GPU (deduped by device id; integrated GPUs only when no
    discrete one exists). A ``NULL`` ``tensor_split`` then takes llama.cpp's
    "default split, by free memory": ``splits[i] = free_i``, normalized, with
    each layer assigned by ``upper_bound`` over the cumulative fractions - and
    the per-layer KV cache follows its layer's device.

    A PLAIN SUM IS THE CORRECT BUDGET: because the weighting is by FREE MEMORY,
    device *i* receives the fraction ``free_i / SUM(free)`` of the offloaded
    layers, so a budget of ``SUM(free)`` places exactly ``free_i`` on device
    *i*. Every card is filled to its own free memory and no further, which is
    what makes a HETEROGENEOUS set safe: a 24/24/8 GB board is not treated as
    56 GB of anything-goes, the 8 GB card is handed a proportionally smaller
    share. Under an EVEN split, summing would overcommit the smallest card, so
    this follows from llama.cpp's split policy rather than from summing.

    Callers must still charge overhead PER DEVICE (each one carries its own
    compute buffers) - see ``_sizing.VramSizingMixin._split_overhead_bytes``.

    Separate from :func:`vram_capacity`, which answers for a
    CONFIGURED split and feeds the admission gate: this is the sizing question
    ("how much can this load actually use") and must not silently move a
    refusal threshold. Answers ``{}``, i.e. "no implicit combined figure - use
    the single-device reading", in every case where a sum would be dishonest:

    - A ``gpu_split_indices`` IS configured: an explicit ``tensor_split`` is
      written, the shares are the configured/auto ratios rather than the
      free-memory default, and :func:`vram_capacity` already owns that case.
      Answered from config alone, with NO hardware probe.
    - Fewer than 2 devices are detected: the single-GPU majority, and the case
      where this must cost nothing and change nothing.
    - Any device does not report BOTH ``free`` and ``total`` (all-or-nothing,
      mirroring :func:`vram_capacity`'s own "free" key): a partially-measurable
      board must not under-count by reading a missing device as 0, nor
      over-count by assuming a blind device is empty.
    - (``list_gpus()`` path only) the probe did not complete fresh this call: a
      load is never sized from a frozen last-known-good snapshot, matching
      :func:`gpu_split_shortfall`'s probe-freshness contract.

    On the ``vulkan`` build the reading comes from :func:`native_gpu_devices`
    (the crash-isolated probe daemon's view of ggml's OWN registry), because
    that is the device space the layers are actually placed in;
    :func:`list_gpus` speaks torch's space and is structurally blind there.
    A sum needs the right device SET rather than an index correspondence, and
    the set is taken from the space that receives the layers. Same branch as
    :func:`resolve_auto_split_ratios`.

    Never raises."""
    from localm.config import load_config
    cfg = config if config is not None else load_config()
    if cfg.get("gpu_split_indices"):
        return {}
    if _native_backend_has_vulkan():
        devices = native_gpu_devices()
        if not devices:
            return {}
        # DISCRETE GPUs ONLY. llama.cpp's device list SKIPS accelerators
        # outright and appends integrated GPUs only when no discrete GPU was
        # found, so a box with a discrete card AND an iGPU must not have the
        # iGPU's memory summed into a budget llama.cpp then places entirely on
        # the discrete card.
        #
        # :func:`native_gpu_devices` ALREADY APPLIES EXACTLY THIS FILTER (see
        # _llama_visible_devices), so on a real reading this pass is a no-op. A
        # caller that injects a device list by patching native_gpu_devices
        # directly bypasses that derivation entirely.
        #
        # Filters to GGML_DEV_TYPE_GPU rather than excluding the others by
        # value: the enum has GROWN (IGPU was inserted ahead of ACCEL, so the
        # numeric value of ACCEL differs between builds we may ship), and this
        # module cannot know which llama.cpp is provisioned, so an allowlist is
        # version-independent where a denylist is not. A device whose type the
        # probe did not report is not assumed to be discrete - it fails the
        # filter and, if that leaves fewer than 2, the single-device reading
        # stands.
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
    # Logged at INFO on the SUCCESS path only, matching resolve_auto_split_ratios'
    # own "auto GPU split: distributing by free VRAM" line: WHICH budget was used
    # and which per-device readings produced it. The decline paths above return
    # {} without logging.
    #
    # This runs per LOAD, not per poll: the callers are the backend's load-time
    # preflights (_check_vram, _auto_gpu_layers, _auto_ctx_max), and the GUI's
    # polling routes reach sysstats.estimate_vram instead, which borrows only
    # the pure _bytes_per_token helper and never this.
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

    This is the exact signal ``vram_capacity()`` uses to decide whether its total
    is COMBINED across a split (>= 2) or the single main GPU (< 2): the same
    ``resolve_gpu_split`` + detected-device re-filter. Callers that LABEL a VRAM
    number ("combined across N GPUs" vs "your main GPU's") must gate on this, not
    on the raw ``gpu_split_indices`` length - a stale/typo'd index or a GGUF-only
    box (no ``list_gpus``) leaves a 2-entry split resolving to one device, where
    the number is single-GPU and calling it "combined" would mislabel it.

    Do NOT use this to decide "will the loader ACTUALLY apply a multi-device
    split" (a VRAM preflight, a swap decision, a "your split spans N cards"
    notice): on the ``vulkan`` build the real split devices live in ggml-vulkan's
    own index space, which ``list_gpus()`` (torch.cuda / nvidia-smi) is
    structurally blind to, so the detected re-filter here
    COLLAPSES a live, working 2-way vulkan split to < 2. That is the honest answer
    for a LABEL (``vram_capacity()`` itself cannot sum a split it cannot measure,
    so it too falls back to the single-GPU number, and calling that "combined"
    would lie), but the WRONG answer for a load-safety gate. Use
    :func:`applied_split_device_count` for the "will a split be applied at load
    time" question - it mirrors :func:`apply_gpu_split`'s own gate and does not
    apply the detected re-filter.

    Returns 0 when no split is configured (the common single-GPU path, with no
    hardware probe); otherwise the count of valid split devices (0/1 = effectively
    single, 2+ = combined)."""
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

    Mirrors :func:`apply_gpu_split`'s own gate (``len(resolve_gpu_split(...)) < 2``
    -> no split), so it answers "will a multi-device split be applied at load
    time", NOT "can we MEASURE that split's combined VRAM". The two counts differ
    on exactly one axis: the detected-device re-filter that
    :func:`split_device_count` / :func:`vram_capacity` apply against
    :func:`list_gpus` AFTER ``resolve_gpu_split``. That filter is CORRECT for a
    VRAM LABEL (you cannot honestly call a number "combined across N GPUs" when
    ``list_gpus()`` only measured one device), but WRONG for a load-safety gate on
    the ``vulkan`` build, where ``resolve_gpu_split`` passes the configured indices
    through UNVALIDATED in ggml-vulkan's own index space - a
    real 2-way split ``list_gpus()`` (torch.cuda / nvidia-smi) is structurally
    blind to. There this returns 2 while :func:`split_device_count` collapses to
    < 2. On a NON-vulkan box with a detected device list the two are IDENTICAL
    (``resolve_gpu_split`` already dropped unknown indices, so that later re-filter
    is a proven no-op).

    Deliberately does NOT pass ``gpus=`` (so ``resolve_gpu_split`` calls
    ``list_gpus()`` itself) and does NOT re-filter the result - exactly what
    :func:`apply_gpu_split` does, which is what makes this the loader truth rather
    than a measurability check.

    Returns 0 when no split is configured (the common path, no hardware probe);
    otherwise the count ``resolve_gpu_split`` yields. Domain is {0} U {2, 3, ...}:
    a single surviving index collapses to 0, same as ``apply_gpu_split`` leaving
    the native single-GPU default untouched."""
    from localm.config import load_config
    cfg = config if config is not None else load_config()
    if not cfg.get("gpu_split_indices"):
        return 0
    return len(resolve_gpu_split(
        cfg.get("gpu_split_indices"), cfg.get("gpu_split_ratios")))


def _list_gpus_reading(deadline: Optional[float] = None, *,
                       wait_for_inflight: bool = False) -> tuple:
    """``(gpus, status)`` from :func:`list_gpus`, tolerant of a test double patched
    in as a plain no-kwarg callable - the bare-list contract that test modules
    stubbing ``list_gpus`` rely on. A double whose signature does
    not accept ``return_status`` is called bare and its reading treated as
    :data:`GPU_PROBE_OK`: it models a completed probe, exactly as a bare stub did
    before the status channel existed, so only a REAL status-capable probe can ever
    report itself stale/busy here. Signature-inspected rather than a blanket
    ``except TypeError`` so a genuine ``TypeError`` raised INSIDE ``list_gpus`` is
    never mistaken for a rejected kwarg and swallowed. In
    production ``list_gpus`` always accepts ``return_status``, so the bare branch is
    a test-only affordance, never taken by the real probe.

    *deadline* is forwarded to ``list_gpus`` only when given (None leaves its
    default cap untouched), so an OFF-event-loop caller can spend a longer budget on
    a cold driver init that overruns the short server cap - the only way to get a
    FRESH first-load reading (a timed-out probe cannot be retried: it is abandoned,
    not cancelled, and a retry short-circuits to the frozen last-known-good).

    *wait_for_inflight* (opt-in, off-loop callers only - see :func:`list_gpus`)
    JOINS a probe another caller already holds (e.g. the GUI's 2.5s stats
    heartbeat) instead of taking an instant BUSY + stale reading. Forwarded only
    when the callable's signature can accept it (a named parameter or
    ``**kwargs``), so a status-capable test double without it keeps working."""
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
    cannot cover its proportional share of *vram_required*. Empty when no split is
    configured, fewer than two split devices resolve, every device has live
    headroom, OR the live per-device check could not run this call (see the probe
    freshness contract below). A shortfall entry is emitted ONLY under a fresh
    ``GPU_PROBE_OK`` reading, so every ``free`` in the result is a current
    measurement a caller may quote to the user as fact.

    ``vram_capacity()`` is an AGGREGATE check: it proves total combined free VRAM
    across the split is enough, but with a PINNED ``gpu_split_ratios``,
    ``apply_gpu_split()`` (the GGUF/llama.cpp backend's tensor_split writer)
    divides a model by that static per-config ratio with NO live per-device
    capacity awareness of its own - unlike the HF/transformers backend, whose
    ``device_map="auto"`` is built from live per-device
    ``torch.cuda.mem_get_info()`` free VRAM instead (see ``backends/hf.py``'s
    ``_cuda_device_map``), so it already self-corrects. Without this check, a model
    too big for one device's actual share could still pass the aggregate check (e.g.
    another already-loaded model sits asymmetrically on one split device more than
    another) and reach llama.cpp's native loader with too little room on that device
    - not always a catchable Python exception, since the native loader can hard-abort
    the WORKER process rather than return NULL (that abort is contained to the
    isolated load worker, never the server - see
    ``backends/llamacpp/_runner.py``). Callers should treat a non-empty result on a
    pinned-ratio split as a hard refusal for a GGUF-backend load (see
    ``http_server.switch_engine``), not merely a warning.

    With ratios UNSET the loader itself now adapts: the parent pins
    :func:`resolve_auto_split_ratios`'s free-VRAM-proportional shares into the
    load, and this gate computes its per-device shares with the SAME auto
    ratios (from its own fresh reading, below). When those adaptive shares
    are in effect, the asymmetric-occupancy refusal is structurally
    impossible (a device's proportional share fits its free whenever the
    aggregate fits), so a non-empty result means the COMBINED estimate is
    short - which ``switch_engine`` defers to the backend's split-aware
    sizing instead of hard-refusing, the same posture as the single-GPU path.
    But auto can DECLINE (a configured index not currently
    detected, a device without a free reading) and fall back to the equal-
    share math, where that invariant does NOT hold and a non-empty result is
    exactly the pre-feature per-device hazard - so a caller deciding
    refuse-vs-defer MUST know which math produced the result, not infer it
    from the config shape. ``return_shares_adaptive=True`` appends that
    fact: ``True`` only when live auto ratios were actually used for the
    shares below; ``False`` for pinned ratios, the equal fallback, and every
    early return (no split, vulkan skip, non-OK probe - where the list is
    empty anyway). Appended AFTER ``status`` when both opt-ins are set:
    ``(shortfall, status, shares_adaptive)``; alone:
    ``(shortfall, shares_adaptive)``. The bare-call shape is untouched.

    Probe freshness. ``list_gpus()`` is deadline-bounded: on a
    TIMEOUT/BUSY it serves a FROZEN last-known-good reading. The default deadline
    now waits out a legitimate cold driver init (see ``_GPU_PROBE_DEADLINE``), but
    a wedged/contended driver can still overrun it, and a caller passing a short
    deadline still times out a cold init, so a non-OK status here is possible and
    is handled, not treated as a fault. This gate
    therefore does NOT compute a shortfall from a stale reading and does NOT refuse
    on one (refusing would break every working box's first load); on a non-OK probe
    it returns ``[]`` (best-effort admit, logged at debug), relying on the isolated
    worker's contained abort above as the backstop. An empty bare-list result thus
    cannot, on its own, be told apart from "verified all-clear": a caller that must
    distinguish "checked, clear" from "could not check" MUST pass
    ``return_status=True`` to receive ``(shortfall, status)`` carrying the underlying
    :data:`GPU_PROBE_OK` / :data:`GPU_PROBE_TIMEOUT` / :data:`GPU_PROBE_BUSY`.

    Completeness (the blindness axis) is NOT gated on here, because the
    directions are asymmetric. ``list_gpus`` tags each device :data:`FREE_SCOPE_DEVICE`
    (the board's number) or :data:`FREE_SCOPE_PROCESS` (counts ONLY this process's own
    allocations - blind to every other process; Windows + AMD with no device-global
    source). A PROCESS-scoped reading OVER-states free (``total`` minus only OUR use,
    missing an out-of-process model's VRAM, or another app's), so in the REFUSE
    direction this gate governs, ignoring the tag is SOUND: if even the over-stated
    ``free`` is short, the real free is shorter still, and the refusal is correct. Only
    the quoted figure is imprecise, and it errs by over-stating what is available, so it
    never talks a user out of a load that would in fact fit.

    Omitting a PROCESS-scoped device from the check trades a SOUND refusal for a
    permit, and the load then reaches llama.cpp too small and dies in the worker
    instead of returning a clean 503.

    The blindness that DOES bite is the PERMIT direction - a blind ``free`` can read
    comfortable while the board is genuinely full - and it is not detectable from the
    reading itself, so no per-device tag check here can catch it. A permit-side caution
    (e.g. prefer single-resident on a PROCESS-scoped reading) belongs with the aggregate
    gate that owns eviction, not with this per-device fit check.

    Only meaningful for the GGUF/llama.cpp load path - callers should gate on that
    themselves (e.g. via ``inference.engine._is_gguf``); this function has no way to
    know which backend a given load will use.

    Takes no headroom margin of its own (a device with EXACTLY enough
    free for its proportional share passes) - if a caller wants the same safety
    margin the aggregate ``vram_capacity()`` check demands, add it to *vram_required*
    before calling (e.g. ``vram_required + headroom``), so a per-device share is not
    held to a thinner margin than the aggregate ceiling it composes with.

    With ``return_status=True`` returns ``(shortfall, status)``; otherwise the bare
    ``shortfall`` list (the historical shape every existing caller relies on).

    *deadline* is forwarded to the underlying ``list_gpus`` probe (None leaves its
    default). The default is cold-init-tolerant (see ``_GPU_PROBE_DEADLINE``), so a
    cold driver init completes and yields a FRESH per-device reading instead of
    timing out into the best-effort admit above; :data:`_GPU_PROBE_CLI_DEADLINE` is
    an alias of it kept for the callers that pass it explicitly. The knob remains
    for a caller that wants a shorter wait (it then falls into that admit on a
    cold first load). An on-loop caller must not probe inline at all - every
    server call site offloads via ``run_in_executor``.
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
        # own index space at load time, which list_gpus() (torch.cuda /
        # nvidia-smi) cannot see or order - torch index N is NOT ggml-vulkan
        # index N (resolve_preferred_device documents exactly this hazard). A
        # per-device share check here would measure the WRONG cards: a silent
        # no-op when torch sees nothing, a wrong refusal/pass on a mixed box.
        # Per-device fit cannot be checked honestly on this backend, so it is
        # not - and the skip is SURFACED rather than presented as a check that
        # passed. Logged at INFO, not debug and not WARNING: the skip is benign
        # whenever the model fits. The GGUF load is subprocess-isolated, so an
        # oversized model still fails as a catchable error.
        logger.info(
            "gpu_split_shortfall: skipping the per-device split VRAM preflight on "
            "the vulkan backend - the configured split indices are in ggml-vulkan's "
            "index space, which list_gpus() cannot map to a card, so no per-device "
            "check can name the right device (GPU-SPLIT-VKINDEX); relying on the "
            "subprocess-isolated loader to catch an oversized load instead.")
        # Conclusive skip with no probe, so it mirrors the no-split return above
        # and reports GPU_PROBE_OK - NOT a non-OK "stale probe" status: nothing
        # was probed, and (like the no-split branch) this is a deterministic
        # routing decision, not an inconclusive reading.
        return _ret([], GPU_PROBE_OK)

    gpus, status = _list_gpus_reading(deadline)
    if status != GPU_PROBE_OK:
        # No FRESH reading this call: list_gpus served a frozen last-known-good value
        # (or []) after a probe TIMEOUT/BUSY. This gate's whole contract is a LIVE
        # per-device check, so it neither quotes that stale "free" as a current
        # figure NOR refuses on it: a non-OK probe can be a healthy box
        # whose driver is merely busy/contended (or a caller-shortened deadline on a
        # cold init - see _GPU_PROBE_DEADLINE), so refusing would break working
        # setups on a routine slow probe. The check could not
        # run this call -> admit best-effort, surfaced via debug + the returned status,
        # never a silent success. The GGUF/embedder load runs in an isolated worker
        # whose native abort is contained to that child - the backstop a
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
        # NOTE: NOT gated on g["free_scope"] - see the blindness paragraph in
        # the docstring.
        needed = int(vram_required * (ratio / total_ratio))
        free = g["free"]
        if free < needed:
            shortfall.append({"index": idx, "needed": needed, "free": free})
    return _ret(shortfall, status, shares_adaptive)


def _device_choice_configured(cfg: dict) -> bool:
    """True when the user actually chose a device: a GPU split, or a Main GPU.

    ONE definition of "nothing configured", shared by :func:`resolve_preferred_device`
    and :func:`visible_device_order`. Both answer ``None`` in that case, and - this is
    the load-bearing part - neither may probe the driver to find that out: the answer
    comes from config alone. Keeping the gate in one place is what stops the two from
    drifting: a caller that reaches list_gpus() eagerly, BEFORE delegating to the
    gated resolve_preferred_device, makes an unconfigured box pay for a GPU probe
    (torch init, or the nvidia-smi fallback) to compute the same ``None`` the
    config could have answered for free.
    """
    return bool(cfg.get("gpu_split_indices")) or cfg.get("main_gpu_index") is not None


def resolve_preferred_device(config: Optional[dict] = None, *,
                            gpus: Optional[list] = None) -> Optional[int]:
    """The device a media workload should DEFAULT to, with every OTHER card left
    VISIBLE. ``None`` when nothing is configured, or when no torch-visible device can
    be named honestly.

    NEVER use this to MASK the other cards away: ComfyUI core ships per-component GPU
    PLACEMENT - ``SelectModelDevice``/``SelectCLIPDevice``/``SelectVAEDevice``
    (``comfy_extras/nodes_multigpu.py``, registered at ``nodes.py:2440``), which call
    ``deepclone_multigpu`` to rehome a component onto another card with independent
    weights. Masking to one device (ComfyUI's ``--cuda-device``, or a bare
    ``CUDA_VISIBLE_DEVICES=N``) deletes the other cards from torch's view and turns
    every one of those nodes into a silent no-op. Prefer ComfyUI's ``--default-device``,
    which reorders rather than masks (``main.py:69-76``), or :func:`visible_device_order`
    for an install we cannot pass argv to.

    The predicate here is PREFERENCE, not exclusivity: "which card should lead", not
    "which card is the only one". It is NOT :func:`resolve_main_gpu_index`, which
    answers IDENTITY ("which device is primary") and resolves an unset value to
    device 0 - using that here would silently pick card 0 and ignore the split. On
    a configured split this is a CAPACITY-informed choice: the split device with
    the MOST live free VRAM.

    INDEX SPACE (this is load-bearing): the answer is always a TORCH device index,
    because media runs on torch (ComfyUI), and :func:`list_gpus` enumerates via
    torch.cuda. It must never leak :func:`resolve_gpu_split`'s Vulkan pass-through:
    on the ``vulkan`` llama.cpp build that function returns indices UNVALIDATED,
    in ggml-vulkan's own index space, which torch does not share.
    Handing one of those to ComfyUI as a CUDA/HIP id would name the wrong card. So a
    device is returned only when it is genuinely torch-visible; otherwise ``None``, and
    ComfyUI keeps its own default.
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
                # one and SAY SO: it may not be the emptiest card. Warned, not
                # raised - refusing would break a working setup over a probe that is
                # allowed to be unmeasurable.
                logger.warning(
                    "gpu_split is configured but no split device reports free VRAM, so "
                    "the best card cannot be chosen for media; defaulting to device %d, "
                    "which may have less free VRAM than its peers.", visible[0])
                return visible[0]
            # NOT ONE configured split device is torch-visible. resolve_gpu_split()
            # passes indices through UNVALIDATED on the vulkan llama.cpp build,
            # so these are very likely ggml-vulkan indices, which mean something
            # else entirely to torch. Naming one would point ComfyUI at
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
    """Every torch-visible device index with the PREFERRED one FIRST, or ``None`` when
    no device should be named.

    For a ComfyUI localm cannot pass argv to: the user's OWN install, started by their
    own launcher (possibly ZLUDA-wrapped), where the child env is the only lever. This
    mirrors exactly what ComfyUI's own ``--default-device`` does at ``main.py:69-76`` -
    it REORDERS ``CUDA_VISIBLE_DEVICES``/``HIP_VISIBLE_DEVICES`` so the chosen device
    leads, leaving the rest visible - rather than what ``--cuda-device`` does at
    ``main.py:78-81``, which masks them away and silently disables core's
    ``Select*Device`` placement nodes.

    Every index is torch-visible by construction (:func:`resolve_preferred_device` and
    :func:`list_gpus` share torch's index space), so this never emits a Vulkan-space id.

    NOTE the consequence, which callers must respect: after this reorder the preferred
    card becomes torch index 0, so a workflow's ``gpu:N`` refers to the REORDERED
    position, not to localm's own ``list_gpus`` index. Anything emitting ``gpu:N`` into
    a workflow has to map through this order, not around it.
    """
    from localm.config import load_config
    cfg = config if config is not None else load_config()
    if not _device_choice_configured(cfg):
        # Nothing configured, so the answer is None either way - take it from config and
        # do NOT probe. This gate mirrors resolve_preferred_device's own (both call the
        # shared helper): without it, every ComfyUI spawn on an unconfigured box pays for
        # a driver probe to learn what config already knew. Callers put
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
    """The ``gpu:N`` string ComfyUI will understand for OUR *device_index*, or ``None``
    when it cannot be named honestly.

    THE INDEX-SPACE GATE. Three coordinate systems meet here and an off-by-one puts a
    component on the wrong card and STILL RENDERS - a silent wrong answer, not a crash:

    1. localm's own device index (``list_gpus()`` -> ``torch.cuda`` enumeration of the
       UNMASKED box). This is what ``gpu_split_indices`` and ``main_gpu_index`` mean.
    2. The VISIBLE ORDER we impose (:func:`visible_device_order`), written into
       ``CUDA_VISIBLE_DEVICES``/``HIP_VISIBLE_DEVICES`` either by ComfyUI's own
       ``--default-device`` (``main.py:69-76``) or by us for a ComfyUI we cannot pass
       argv to.
    3. ComfyUI's ``gpu:N`` widget value, which is a POSITION, not a device id:
       ``get_gpu_device_options`` (``model_management.py:246-257``) emits
       ``gpu:{i} for i in range(len(get_all_torch_devices()))``, and
       ``get_all_torch_devices`` enumerates torch AFTER the mask/reorder has applied.

    So ``gpu:N`` means "the Nth entry of the visible order", and the mapping is
    DERIVABLE rather than guessable precisely because localm is the one that imposes
    that order. This is the whole reason we must never mask: masking collapses the
    order to one entry and ``get_gpu_device_options`` then emits no ``gpu:N`` at all
    (it gates on ``len(devices) > 1``), so every placement node silently no-ops.

    Returns ``None`` when no order is established (nothing configured, or no
    torch-visible device) or when *device_index* is not in it, rather than guessing a
    position. Callers must treat None as "do not emit a device for this component".

    VERIFY, DO NOT TRUST, at runtime: this mapping is derived from source
    (``model_management.py:246-257`` read at ComfyUI git 867404b) and is UNPROVEN on a
    real multi-GPU box - this one has a single card, where the order is trivially
    ``[0]``. Before placement is enabled by default, confirm against the live server's
    ``/object_info`` (does ``SelectModelDevice`` actually offer this ``gpu:N``?) and
    ``/system_stats`` (does that position correspond to the card we meant?).
    """
    order = visible_device_order(config, gpus=gpus)
    if not order or device_index not in order:
        return None
    return f"gpu:{order.index(device_index)}"


def plan_media_placement(config: Optional[dict] = None, *,
                         gpu_options: Optional[list] = None) -> Optional[dict]:
    """Assign media components to cards for a box ComfyUI sees as 2+ GPUs, or ``None`` to
    keep the single-card floor. Pure: no I/O, no probe of its own.

    *gpu_options* is the LIVE ``gpu:N`` list read from the running ComfyUI's
    ``/object_info`` device combo (:func:`localm.media.comfy_client.probe_placement_capability`).
    It is authoritative about how many cards ComfyUI actually enumerates, and it is
    ComfyUI's OWN index space, so a POSITIONAL policy over it inherits NONE of the
    localm-index vs ``gpu:N`` translation hazard (:func:`comfy_gpu_option` exists for a
    future identity-based policy) and never consults ``split_device_count`` (whose
    Vulkan soundness hole it therefore does not inherit).

    v1 policy - no free-VRAM read (the live free number is not yet trustworthy, and
    per-component byte sizes do not exist): keep the big model on the preferred card (the
    first visible position, where ``--default-device`` already put the most-free card;
    ``"model": None`` means "no injection", so the GGUF UNet is never moved off its
    loader default and needs no factory patch), and offload the smaller CLIP text-encoder
    and VAE to the SECOND visible card. That is the concrete win (the FLUX T5-XXL encoder
    and the VAE off the compute card free real headroom on card 0) with zero dependency
    on the lying free-VRAM number or on the GGUF factory.

    Returns ``None`` when fewer than two ``gpu:N`` options exist (single-card floor,
    unchanged). Placement is capability-driven: it does not require a configured chat
    ``gpu_split`` - two visible cards is enough for the second one to carry weight. A
    size-aware spread across 3+ cards is a documented follow-up (SPEC-placement.md).
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
    means it should load with little headroom (small context, nothing else on the
    GPU); "too-big" still runs, with some layers offloaded to system RAM (slower).
    That partial offload is delivered automatically: with n_gpu_layers_auto on
    (the default) the loader sizes how many layers fit from free VRAM at load
    (GgufBackend._auto_gpu_layers), so a "too-big" model loads instead of being
    refused, rather than only if the user manually lowers -g.

    The need estimate here (weights * safety factor + fixed overhead) is
    context-agnostic and a touch more conservative on weights than the loader's
    exact weights + real-KV + overhead math (GgufBackend._check_vram),
    so at a normal/default context a "fits" badge is not optimistic. It carries no
    explicit KV term, so it is a weights-fit signal, not a guarantee for an
    unusually large -c/n_ctx (whose KV can exceed the weight slack); the loader's
    own preflight remains the authority. A "tight"/"too-big" model may still load
    via partial offload.
    """
    if not total_vram or not size_bytes:
        return ""
    need = size_bytes * _WEIGHT_FACTOR + _OVERHEAD_BYTES
    if need <= total_vram * 0.85:
        return "fits"
    if need <= total_vram:
        return "tight"
    return "too-big"
