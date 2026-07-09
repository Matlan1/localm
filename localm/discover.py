# SPDX-License-Identifier: AGPL-3.0-or-later
"""
In-app model discovery: search HuggingFace for GGUF models and judge,
per quantization, whether a file fits this machine's VRAM.

Discovery is a user-initiated prelude to ``localm pull`` and sits in the same
policy category (explicit user action - see docs/network.md): it is not
routed through the net_allow/net_deny domain rules, but ``net_mode = off``
still blocks it, so the one kill switch keeps its promise.

"Fits your VRAM" badges compare against TOTAL VRAM, not currently-free VRAM:
the active chat model occupies the GPU while you browse, and it will be
unloaded before the new one loads. The estimate mirrors the GGUF backend's
preflight: weights + ~1.5 GB overhead for KV cache and compute buffers.
"""

from __future__ import annotations

import ctypes
import re
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

# Mirrors GgufBackend._VRAM_OVERHEAD_BYTES (KV cache + compute buffers)
_OVERHEAD_BYTES = int(1.5e9)
# Weights rarely load at exactly file size - small safety factor
_WEIGHT_FACTOR = 1.10

# Quantization label inside a GGUF filename, e.g. Q4_K_M, Q8_0, IQ4_XS,
# Q6_K, F16, BF16. Matched case-insensitively on word-ish boundaries.
_QUANT_RE = re.compile(
    r"(?i)(?<![A-Z0-9])(IQ\d+_[A-Z0-9]+|Q\d+_K(?:_[SML])?|Q\d+_\d+"
    r"|BF16|F16|F32|FP16|FP32)(?![A-Z0-9])")

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

    Routes through ``netpolicy.safe_fetch_bytes`` so the request is pinned to the
    validated IP and EVERY redirect hop is re-checked against the network policy
    (SSRF-REBIND): a DNS-rebind of the HF host, or a redirect from the HF API,
    cannot bounce discovery into a loopback / link-local / private address. This
    is the same protection the model-pull path already uses; a raw ``requests.get``
    here previously bypassed it (an owner-initiated fetch, so low severity, but the
    inconsistency is closed)."""
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


def hf_param_bytes(safetensors: Optional[dict]) -> Optional[int]:
    """Estimated GPU weight footprint in bytes for an HF model, from its
    safetensors param metadata (the ``safetensors`` expand field of the HF models
    API: ``{"total": <param count>, "parameters": {...}}``).

    localm's HF backend loads in bf16 on GPU with no on-load quantization (see
    inference/backends/hf.py), so the footprint is ``total_params * 2`` regardless
    of the STORED dtype - the loader casts to bf16. This is the weight size only;
    fit_label() adds KV-cache / compute overhead, the same way it does for a GGUF
    file size. Returns None when the repo has no usable param count so the GUI can
    show "size unknown" rather than a guessed badge (do-not-hide-problems: an
    unknown is surfaced as unknown, not silently treated as zero/fits)."""
    if not isinstance(safetensors, dict):
        return None
    total = safetensors.get("total")
    if not isinstance(total, int) or isinstance(total, bool) or total <= 0:
        return None
    return total * 2


def _search_one(fmt: str, query: str, limit: int) -> list[dict]:
    """One HF /api/models query for a single format, each item tagged with it."""
    params: dict = {
        "filter": _FORMAT_FILTER[fmt],
        "sort": "downloads",
        "direction": "-1",
        "limit": str(limit),
    }
    if query.strip():
        params["search"] = query.strip()
    if fmt == "hf":
        # Expand the safetensors param metadata so each result carries a param
        # count we can turn into a VRAM fit estimate inline (no per-repo tree
        # fetch). `expand` is restrictive - it drops the default stat fields - so
        # re-request downloads/likes/lastModified alongside it.
        params["expand[]"] = ["safetensors", "downloads", "likes", "lastModified"]
    data = _get(f"{HF_API}/api/models", params)
    if not isinstance(data, list):
        raise DiscoverError("Unexpected response from HuggingFace search")
    out = []
    for item in data[:limit]:
        repo = item.get("id") or item.get("modelId")
        if not repo:
            continue
        row = {
            "id": repo,
            "downloads": item.get("downloads", 0),
            "likes": item.get("likes", 0),
            "updated": item.get("lastModified", ""),
            "formats": [fmt],
        }
        if fmt == "hf":
            # bf16 weight footprint from the param count, or None when HF has no
            # safetensors metadata (the row then shows "size unknown").
            row["size_bytes"] = hf_param_bytes(item.get("safetensors"))
        out.append(row)
    return out


def hf_search(query: str = "", limit: int = 20,
              formats: Sequence[str] = ("gguf",)) -> list[dict]:
    """Search HF for model repos in the requested *formats* (a subset of
    {"gguf", "hf"}). Empty query = most downloaded.

    One HF query runs per requested format (gguf -> the bundled GGUF backend,
    hf -> the transformers backend); the results are merged de-duped by repo id
    (a repo that surfaces under both formats keeps a merged ``formats`` list),
    sorted by downloads, and trimmed to *limit*.

    Returns [{id, downloads, likes, updated, formats}]. Defaults to gguf-only so
    the CLI ``localm search`` is unchanged; the GUI passes both formats
    explicitly from its toggles."""
    _ensure_online()
    limit = max(1, min(int(limit), 50))
    wanted = [f for f in formats if f in _FORMAT_FILTER]
    if not wanted:
        raise DiscoverError(
            "No valid model format requested (choose gguf and/or hf).")
    # One list per format (each already download-sorted by the API), de-duped by
    # repo id: a repo that surfaces under both formats stays in the FIRST list it
    # appeared in and its tags are unioned there.
    seen: dict[str, dict] = {}
    per_format: list[list[dict]] = []
    for fmt in wanted:
        lst: list[dict] = []
        for item in _search_one(fmt, query, limit):
            existing = seen.get(item["id"])
            if existing:
                for f in item["formats"]:
                    if f not in existing["formats"]:
                        existing["formats"].append(f)
                # A repo in both formats enters via the FIRST (gguf) list, which
                # carried no size; keep its HF size estimate if the hf pass has one.
                if existing.get("size_bytes") is None and item.get("size_bytes") is not None:
                    existing["size_bytes"] = item["size_bytes"]
            else:
                seen[item["id"]] = item
                lst.append(item)
        per_format.append(lst)
    # Round-robin interleave by per-format rank, then trim to `limit`. A plain
    # merge-then-sort-by-downloads would let the higher-download format (HF repos
    # routinely dwarf GGUF repacks) crowd the other out of the top `limit`
    # entirely, so a "show GGUF" toggle could return zero GGUF. Interleaving keeps
    # every enabled format visible while still leading each with its most popular.
    out: list[dict] = []
    rank = 0
    while len(out) < limit and any(rank < len(lst) for lst in per_format):
        for lst in per_format:
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
    m = _QUANT_RE.search(name)
    return m.group(1).upper() if m else ""


def list_gpus() -> list:
    """Every GPU device visible right now: ``[{"index", "name", "total",
    "free"}, ...]``, or ``[]`` when nothing is measurable.

    Tries torch first (CUDA/ROCm - torch's ROCm build aliases torch.cuda.* to
    HIP under the hood, so an AMD card enumerates through the exact same API,
    no special-casing needed) since it also gives a device name; falls back to
    a name-aware ``nvidia-smi`` listing (ALL devices, not just the first) for
    the GGUF-only install that has no torch.

    Deliberately does NOT fall back to the Windows display-adapter registry:
    that tier (see vram_info()) can only report one aggregate "largest
    adapter" number with no per-device identity, so it cannot support GPU
    *selection* - only vram_info()'s single-number "total VRAM for fit
    badges" use case. That is a scope boundary, not an oversight."""
    try:
        import torch
        if torch.cuda.is_available():
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
            if out:
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
                    })
                except ValueError:
                    continue   # a malformed line never hides the rest
            if out:
                return out
    except Exception:
        pass
    return []


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
    than trusted blindly (rule 5, do-not-hide-problems).

    When detection itself is unmeasurable (``list_gpus()`` returns nothing -
    no torch, no nvidia-smi), the configured index cannot be cross-checked
    either way; it is passed through unchecked rather than discarding an
    explicit user choice we have no way to disprove (the same documented
    boundary as the Windows-registry VRAM fallback)."""
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
    if idx == 0:
        return 0   # the native default anyway - no need to enumerate devices
    if gpus is None:
        gpus = list_gpus()
    # Check membership by the "index" field, NOT list position: a device that
    # fails to report (list_gpus() skips it rather than hide the rest) leaves a
    # gap, so "idx < len(gpus)" alone could wrongly wave through an idx that
    # does not actually correspond to any detected device.
    if gpus and not any(g.get("index") == idx for g in gpus):
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
_MAX_GPU_SPLIT_INDEX = 127


def resolve_gpu_split(configured_indices, configured_ratios=None, *,
                       gpus: Optional[list] = None) -> list:
    """Validate a configured multi-GPU split (``gpu_split_indices`` /
    ``gpu_split_ratios``) against the devices ``list_gpus()`` (or the injected
    *gpus*, for tests) currently sees, returning ``[(index, ratio), ...]``
    ready to write into ``tensor_split``.

    Mirrors :func:`resolve_main_gpu_index`'s posture: an index that does not
    match a currently-detected device is dropped with a WARNING rather than
    trusted blindly (rule 5, do-not-hide-problems) - a stale config
    referencing a since-removed GPU degrades to single-GPU instead of
    mis-targeting VRAM or crashing a load. Duplicate indices keep their first
    occurrence. Fewer than 2 valid indices after validation means "no split"
    (returns ``[]``) - the single-GPU path driven by ``apply_main_gpu`` is
    unaffected.

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
    if gpus:
        known = {g.get("index") for g in gpus}
        valid = [i for i in deduped if i in known]
        dropped = [i for i in deduped if i not in known]
        if dropped:
            logger.warning(
                "gpu_split_indices contains %d device(s) not currently "
                "detected (%s); dropping them from the split",
                len(dropped), dropped)
    else:
        # Detection unmeasurable (no torch, no nvidia-smi): same documented
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


def apply_gpu_split(mp, *, config: Optional[dict] = None):
    """Set ``mp.split_mode``/``mp.tensor_split`` from the configured
    ``gpu_split_indices``/``gpu_split_ratios``, validated via
    :func:`resolve_gpu_split`. Leaves native defaults (a single active GPU -
    whatever :func:`apply_main_gpu` already set) untouched when fewer than 2
    valid devices are configured. Shared by the llama.cpp chat backend and the
    embedder, same as ``apply_main_gpu``.

    Returns the ctypes float array backing ``mp.tensor_split`` (or ``None``
    when no split was applied) - the CALLER MUST keep this referenced until
    after the ``llama_load_model_from_file()`` call that consumes *mp*:
    llama.cpp copies ``tensor_split``'s contents at load time (it is not held
    as a live pointer afterward), so the buffer only needs to survive that one
    call, not the loaded model's lifetime."""
    from localm.config import load_config
    cfg = config if config is not None else load_config()
    pairs = resolve_gpu_split(cfg.get("gpu_split_indices"), cfg.get("gpu_split_ratios"))
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


def vram_info() -> dict:
    """{"total": bytes, "free"?: bytes} for the CONFIGURED main GPU device (see
    main_gpu_index / resolve_main_gpu_index), or the largest GPU when none is
    configured, or {} when not measurable. Tries torch (CUDA/ROCm) then
    nvidia-smi (both via list_gpus()), then the Windows display-adapter
    registry - the GGUF-only install has no torch, and the fit badges must
    still work there (total is all fit_label needs)."""
    from localm.config import load_config
    gpus = list_gpus()
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
        return out

    import sys
    if sys.platform == "win32":
        try:
            import winreg
            best = 0
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
                    except OSError as e:
                        # Unexpected (vs the EnumKey end-of-list break above):
                        # access denied or a removed key. Surface under --debug
                        # so incomplete VRAM detection is diagnosable; the silent
                        # fallback is deliberate (a note beats crashing fit badges).
                        logger.debug("vram_info: registry subkey %s unreadable: %s",
                                     sub, e)
                        continue
            if best:
                return {"total": int(best)}
        except Exception:
            pass
    return {}


def vram_capacity(config: Optional[dict] = None) -> dict:
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
    single-GPU (rule 5, do-not-hide-problems); this reuses that same
    validation rather than duplicating it.
    """
    from localm.config import load_config
    cfg = config if config is not None else load_config()
    # Cheap short-circuit for the common (no split configured) case, mirroring
    # resolve_gpu_split's own early return - skips a real hardware probe
    # (list_gpus() -> torch/nvidia-smi) on every request for the vast majority
    # of single-GPU installs that never configured a split.
    if not cfg.get("gpu_split_indices"):
        return vram_info()

    gpus = list_gpus()
    pairs = resolve_gpu_split(
        cfg.get("gpu_split_indices"), cfg.get("gpu_split_ratios"), gpus=gpus)
    by_index = {g.get("index"): g for g in gpus}
    split_gpus = [by_index[idx] for idx, _ in pairs if idx in by_index]
    if len(split_gpus) < 2:
        return vram_info()

    out = {"total": sum(g["total"] for g in split_gpus)}
    frees = [g.get("free") for g in split_gpus]
    if all(f is not None for f in frees):
        out["free"] = sum(frees)
    return out


def fit_label(size_bytes: int, total_vram: Optional[int]) -> str:
    """
    Capacity badge for one file: "fits" / "tight" / "too-big", or "" when
    VRAM is unknown. "tight" means it should load with little headroom
    (small context, nothing else on the GPU); "too-big" still runs with
    partial offload to system RAM, just slower.
    """
    if not total_vram or not size_bytes:
        return ""
    need = size_bytes * _WEIGHT_FACTOR + _OVERHEAD_BYTES
    if need <= total_vram * 0.85:
        return "fits"
    if need <= total_vram:
        return "tight"
    return "too-big"
