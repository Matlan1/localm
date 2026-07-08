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
    full = url + ("?" + urllib.parse.urlencode(params) if params else "")
    try:
        _final, _ctype, body = netpolicy.safe_fetch_bytes(
            full, max_bytes=32 * 1024 * 1024, timeout=int(_TIMEOUT))
        return _json.loads(body.decode("utf-8"))
    except Exception as e:
        raise DiscoverError(f"HuggingFace request failed: {e}")


def _search_one(fmt: str, query: str, limit: int) -> list[dict]:
    """One HF /api/models query for a single format, each item tagged with it."""
    params = {
        "filter": _FORMAT_FILTER[fmt],
        "sort": "downloads",
        "direction": "-1",
        "limit": str(limit),
    }
    if query.strip():
        params["search"] = query.strip()
    data = _get(f"{HF_API}/api/models", params)
    if not isinstance(data, list):
        raise DiscoverError("Unexpected response from HuggingFace search")
    out = []
    for item in data[:limit]:
        repo = item.get("id") or item.get("modelId")
        if not repo:
            continue
        out.append({
            "id": repo,
            "downloads": item.get("downloads", 0),
            "likes": item.get("likes", 0),
            "updated": item.get("lastModified", ""),
            "formats": [fmt],
        })
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
