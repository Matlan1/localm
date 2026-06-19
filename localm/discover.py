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
from typing import Optional

HF_API = "https://huggingface.co"
_TIMEOUT = 20

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
    import requests
    try:
        r = requests.get(url, params=params, timeout=_TIMEOUT,
                         headers={"User-Agent": "localm/0.1 (model discovery)"})
        r.raise_for_status()
        return r.json()
    except Exception as e:
        raise DiscoverError(f"HuggingFace request failed: {e}")


def hf_search(query: str = "", limit: int = 20) -> list[dict]:
    """Search HF for GGUF model repos. Empty query = most downloaded.
    Returns [{id, downloads, likes, updated}] sorted by downloads."""
    _ensure_online()
    limit = max(1, min(int(limit), 50))
    params = {
        "filter": "gguf",
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
        })
    return out


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


def vram_info() -> dict:
    """{"total": bytes, "free"?: bytes} for the largest GPU, or {} when not
    measurable. Tries torch (CUDA/ROCm), then nvidia-smi, then the Windows
    display-adapter registry - the GGUF-only install has no torch, and the
    fit badges must still work there (total is all fit_label needs)."""
    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info(0)
            return {"free": int(free), "total": int(total)}
    except Exception:
        pass

    try:
        import subprocess
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        if proc.returncode == 0 and proc.stdout.strip():
            total_mb, free_mb = proc.stdout.strip().splitlines()[0].split(",")
            return {"total": int(total_mb) * 1024 ** 2,
                    "free": int(free_mb) * 1024 ** 2}
    except Exception:
        pass

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
                    except OSError:
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
