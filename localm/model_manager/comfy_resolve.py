# SPDX-License-Identifier: AGPL-3.0-or-later
"""Download sources for ComfyUI workflow model files.

``curated_comfy_download`` answers from the curated table in ``registry``
alone. ``lookup_comfy_download`` also searches HuggingFace for a repository
file whose name is exactly the missing filename, and ``cached_comfy_download``
returns an earlier search's result without any network access.

A HuggingFace match is offered only for a file whose format cannot run code
on load (``SAFE_EXTENSIONS``), whose workflow slot maps to a known ComfyUI
models folder (``comfy_slot_folder``), and whose name is specific enough to
identify one model (``is_specific_model_name``). Gated, private and disabled
repositories are skipped. A match in a ``COMFY_ORG`` repository wins; among
the rest the most-downloaded repository wins.
"""

from __future__ import annotations

import re
import threading
import time
import urllib.parse
from dataclasses import dataclass
from typing import Optional

from ..debuglog import logger
from .registry import resolve_comfy_model_source

ORIGIN_CURATED = "curated"
ORIGIN_HUGGINGFACE = "huggingface"

LOOKUP_FOUND = "found"
LOOKUP_NOT_FOUND = "not_found"
LOOKUP_UNSUPPORTED = "unsupported"
LOOKUP_OFFLINE = "offline"
LOOKUP_FAILED = "failed"

REASON_FORMAT = "format"
REASON_FOLDER = "folder"
REASON_NAME = "name"

SAFE_EXTENSIONS = (".safetensors", ".sft", ".gguf")

# Stems that name a file's role rather than a model, shared by many unrelated
# repositories.
_GENERIC_STEMS = frozenset({
    "model", "models", "pytorch_model", "diffusion_pytorch_model",
    "consolidated", "adapter_model", "text_encoder", "text_encoder_2",
    "weights", "checkpoint", "final", "latest", "model_fp16", "model_fp32",
    "model.fp16", "diffusion_model", "transformer", "encoder", "decoder",
})
_MIN_SPECIFIC_STEM = 5

COMFY_ORG = "Comfy-Org"
_COMFY_ORG_LIMIT = 100
_COMFY_ORG_TTL = 1800.0
_SEARCH_LIMIT = 20
_MAX_PREFIX_QUERIES = 2
# Most HuggingFace requests one lookup makes: the COMFY_ORG listing, every
# query from _search_queries, and the size read.
MAX_HF_REQUESTS = 1 + (1 + _MAX_PREFIX_QUERIES + 1) + 1
_MIN_TOKEN_QUERY = 5
_EXPAND = ["siblings", "downloads", "gated", "private", "disabled"]
_REPO_ID_RE = re.compile(r"\A[A-Za-z0-9][\w.-]*/[A-Za-z0-9][\w.-]*\Z")
_PATH_PART_RE = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9._+-]*\Z")
_SEPARATOR_RE = re.compile(r"[\s_.\-]+")

_INPUT_FOLDERS = {
    "ckpt_name": "checkpoints",
    "unet_name": "unet",
    "vae_name": "vae",
    "lora_name": "loras",
    "control_net_name": "controlnet",
    "style_model_name": "style_models",
}
_CLIP_INPUT_RE = re.compile(r"\Aclip_name\d*\Z")

_FOLDER_MODEL_TYPES = {
    "checkpoints": "diffusion-unet",
    "unet": "diffusion-unet",
    "diffusion_models": "diffusion-unet",
    "clip": "text-encoder",
    "text_encoders": "text-encoder",
    "vae": "vae",
    "loras": "lora",
}

_FOUND_TTL = 1800.0
_MISS_TTL = 300.0
_FAILED_TTL = 60.0
_CACHE_MAX = 128
_cache: dict[str, tuple[float, "ComfyLookup"]] = {}
_cache_lock = threading.Lock()
_org_rows: Optional[tuple[float, list]] = None


@dataclass(frozen=True)
class ComfyDownload:
    """Where to download one ComfyUI workflow model file from, and where it
    goes. ``spec`` is ``owner/repo:path/in/repo`` as ``pull_model`` takes it;
    ``size_bytes`` is None when the size could not be read."""
    spec: str
    model_type: str
    comfy_subfolder: str
    size_bytes: Optional[int]
    origin: str

    @property
    def repo(self) -> str:
        return self.spec.split(":", 1)[0]

    @property
    def path(self) -> str:
        return self.spec.split(":", 1)[1]


@dataclass(frozen=True)
class ComfyLookup:
    """The outcome of a source lookup: ``status`` is one of the ``LOOKUP_*``
    values, ``download`` is set only when it is ``LOOKUP_FOUND``, ``reason``
    is one of the ``REASON_*`` values when it is ``LOOKUP_UNSUPPORTED``, and
    ``detail`` is a human-readable explanation for the non-found outcomes."""
    status: str
    download: Optional[ComfyDownload] = None
    reason: str = ""
    detail: str = ""


def comfy_slot_folder(class_type: str, input_name: str) -> Optional[str]:
    """The ComfyUI ``models/<folder>`` a workflow slot reads its file from, or
    None when the input is not one this knows."""
    cls = (class_type or "").lower()
    name = input_name or ""
    if _CLIP_INPUT_RE.match(name):
        return "clip_vision" if "vision" in cls else "clip"
    if name == "model_name" and "upscale" in cls:
        return "upscale_models"
    return _INPUT_FOLDERS.get(name)


def folder_model_type(folder: str) -> str:
    """The registry model type for a file in ComfyUI ``models/<folder>``."""
    return _FOLDER_MODEL_TYPES.get(folder, "unknown")


def _stem(filename: str) -> str:
    lower = filename.lower()
    for ext in SAFE_EXTENSIONS:
        if lower.endswith(ext):
            return filename[: -len(ext)]
    return filename.rsplit(".", 1)[0]


def is_specific_model_name(filename: str) -> bool:
    """False for a bare file name that many unrelated repositories share
    (``model.safetensors``, ``diffusion_pytorch_model.safetensors``) or that is
    too short to identify one model."""
    stem = _stem(filename)
    if len(stem) < _MIN_SPECIFIC_STEM or stem.lower() in _GENERIC_STEMS:
        return False
    return any(ch.isalpha() for ch in stem)


def _search_queries(filename: str) -> list[str]:
    """HuggingFace search strings for *filename*, in order: its full stem; the
    ``_MAX_PREFIX_QUERIES`` shortest leading parts of the stem that span at
    least two words and ``_MIN_TOKEN_QUERY`` characters; its longest word when
    that has at least ``_MIN_TOKEN_QUERY`` characters. Duplicates (ignoring
    case) are dropped."""
    stem = _stem(filename).strip(" _.-")
    prefixes = [stem[:m.start()] for m in _SEPARATOR_RE.finditer(stem)]
    multi = [p for p in prefixes
             if len(p) >= _MIN_TOKEN_QUERY and len(_SEPARATOR_RE.split(p)) >= 2]
    words = sorted((w for w in _SEPARATOR_RE.split(stem) if w), key=len, reverse=True)
    longest = [words[0]] if words and len(words[0]) >= _MIN_TOKEN_QUERY else []
    out: list[str] = []
    for candidate in [stem] + multi[:_MAX_PREFIX_QUERIES] + longest:
        if len(candidate) >= 3 and candidate.lower() not in (o.lower() for o in out):
            out.append(candidate)
    return out


def _safe_repo_path(path: str) -> bool:
    parts = path.split("/")
    return all(p not in (".", "..") and _PATH_PART_RE.match(p) for p in parts)


def _check_request(filename: str, class_type: str, input_name: str
                   ) -> tuple[Optional[str], Optional[ComfyLookup]]:
    """``(folder, None)`` when *filename* may be searched for, else
    ``(None, unsupported-lookup)``."""
    if "/" in filename or "\\" in filename or not _PATH_PART_RE.match(filename):
        return None, ComfyLookup(LOOKUP_UNSUPPORTED, reason=REASON_NAME,
                                 detail="The file name is not a plain file name.")
    if not filename.lower().endswith(SAFE_EXTENSIONS):
        return None, ComfyLookup(
            LOOKUP_UNSUPPORTED, reason=REASON_FORMAT,
            detail="localm downloads only .safetensors and .gguf model files "
                   "automatically.")
    folder = comfy_slot_folder(class_type, input_name)
    if folder is None:
        return None, ComfyLookup(
            LOOKUP_UNSUPPORTED, reason=REASON_FOLDER,
            detail=f"localm does not know which ComfyUI models folder "
                   f"{class_type}.{input_name} reads from.")
    if not is_specific_model_name(filename):
        return None, ComfyLookup(
            LOOKUP_UNSUPPORTED, reason=REASON_NAME,
            detail="The file name is too generic to identify one model.")
    return folder, None


def search_refusal(filename: str, class_type: str, input_name: str
                   ) -> Optional[ComfyLookup]:
    """The ``LOOKUP_UNSUPPORTED`` outcome ``lookup_comfy_download`` gives for
    this slot without searching, or None when it would search. Never touches
    the network."""
    return _check_request((filename or "").strip(), class_type or "",
                          input_name or "")[1]


def curated_comfy_download(filename: str) -> Optional[ComfyDownload]:
    """The curated-table download for *filename*, or None."""
    source = resolve_comfy_model_source(filename)
    if source is None:
        return None
    return ComfyDownload(source.spec, source.model_type, source.comfy_subfolder,
                         source.size_bytes, ORIGIN_CURATED)


def _cache_key(filename: str, folder: str) -> str:
    return f"{folder}\0{filename}"


def _cache_get(key: str) -> Optional[ComfyLookup]:
    with _cache_lock:
        hit = _cache.get(key)
        if hit is None:
            return None
        expires, lookup = hit
        if time.monotonic() >= expires:
            _cache.pop(key, None)
            return None
        return lookup


def _cache_put(key: str, lookup: ComfyLookup) -> None:
    ttl = {LOOKUP_FOUND: _FOUND_TTL, LOOKUP_NOT_FOUND: _MISS_TTL,
           LOOKUP_FAILED: _FAILED_TTL}.get(lookup.status)
    if ttl is None:
        return
    with _cache_lock:
        if len(_cache) >= _CACHE_MAX and key not in _cache:
            oldest = min(_cache, key=lambda k: _cache[k][0])
            _cache.pop(oldest, None)
        _cache[key] = (time.monotonic() + ttl, lookup)


def clear_lookup_cache() -> None:
    """Forget every cached HuggingFace lookup and repository listing."""
    global _org_rows
    with _cache_lock:
        _cache.clear()
        _org_rows = None


def cached_comfy_download(filename: str, class_type: str,
                          input_name: str) -> Optional[ComfyDownload]:
    """The download an earlier ``lookup_comfy_download`` found for this slot
    and has not expired, else None. Never touches the network."""
    folder = comfy_slot_folder(class_type, input_name)
    if folder is None:
        return None
    lookup = _cache_get(_cache_key(filename, folder))
    return lookup.download if lookup is not None else None


def _comfy_org_rows(token: Optional[str]) -> list:
    """The ``COMFY_ORG`` repositories with their file lists, most downloaded
    first, fetched at most once per ``_COMFY_ORG_TTL`` seconds. Raises
    ``discover.DiscoverError``."""
    global _org_rows
    from localm import discover
    with _cache_lock:
        if _org_rows is not None and time.monotonic() < _org_rows[0]:
            return _org_rows[1]
    rows = discover._get(f"{discover.HF_API}/api/models", {
        "author": COMFY_ORG, "limit": _COMFY_ORG_LIMIT, "sort": "downloads",
        "direction": "-1", "expand[]": _EXPAND,
    }, token=token)
    rows = rows if isinstance(rows, list) else []
    with _cache_lock:
        _org_rows = (time.monotonic() + _COMFY_ORG_TTL, rows)
    return rows


def _matches(rows: list, filename: str) -> list[tuple[str, str, int]]:
    """``(repo_id, path, downloads)`` for every file named exactly *filename*
    in a public, ungated, enabled repository among *rows*."""
    out = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        repo = row.get("id")
        if (not isinstance(repo, str) or not _REPO_ID_RE.match(repo) or ".." in repo
                or row.get("gated") or row.get("private") or row.get("disabled")):
            continue
        downloads = row.get("downloads")
        downloads = downloads if isinstance(downloads, int) else 0
        for sib in row.get("siblings") or []:
            path = sib.get("rfilename") if isinstance(sib, dict) else None
            if (isinstance(path, str) and path.rsplit("/", 1)[-1] == filename
                    and _safe_repo_path(path)):
                out.append((repo, path, downloads))
    return out


def _rank(match: tuple[str, str, int]) -> tuple:
    repo, path, downloads = match
    return (repo.split("/", 1)[0] != COMFY_ORG, -downloads, path.count("/"), repo, path)


def _hf_candidates(filename: str, token: Optional[str]) -> list[tuple[str, str, int]]:
    """``(repo_id, path, downloads)`` for every public, ungated HuggingFace
    model repository file named exactly *filename*, best first (``_rank``).
    Looks in the ``COMFY_ORG`` repositories first and stops there on a match;
    otherwise runs the searches from ``_search_queries``. Raises
    ``discover.DiscoverError``."""
    from localm import discover
    found = _matches(_comfy_org_rows(token), filename)
    if not found:
        for query in _search_queries(filename):
            rows = discover._get(f"{discover.HF_API}/api/models", {
                "search": query, "limit": _SEARCH_LIMIT, "sort": "downloads",
                "direction": "-1", "expand[]": _EXPAND,
            }, token=token)
            found.extend(_matches(rows if isinstance(rows, list) else [], filename))
    unique = {(r, p): (r, p, d) for r, p, d in found}
    return sorted(unique.values(), key=_rank)


def _hf_file_size(repo: str, path: str, token: Optional[str]) -> Optional[int]:
    """The size in bytes of *path* in *repo*'s main branch, or None."""
    from localm import discover
    folder, _sep, name = path.rpartition("/")
    url = f"{discover.HF_API}/api/models/{repo}/tree/main"
    if folder:
        url += "/" + urllib.parse.quote(folder, safe="/")
    try:
        entries = discover._get(url, token=token)
    except Exception as e:
        logger.debug("could not read the size of a HuggingFace file: %s", e)
        return None
    if not isinstance(entries, list):
        return None
    for entry in entries:
        if isinstance(entry, dict) and entry.get("path") == path:
            size = (entry.get("lfs") or {}).get("size") or entry.get("size")
            return size if isinstance(size, int) and size > 0 else None
    return None


def lookup_comfy_download(filename: str, class_type: str, input_name: str
                          ) -> ComfyLookup:
    """Find where to download the ComfyUI model file *filename*, needed by the
    workflow input ``class_type.input_name``.

    The curated table answers first. Otherwise, when the request passes the
    checks in the module docstring, HuggingFace is searched; its result is
    cached per slot folder (see ``cached_comfy_download``). Network policy
    refusing the search gives ``LOOKUP_OFFLINE`` and any other search failure
    ``LOOKUP_FAILED``; neither is reported as ``LOOKUP_NOT_FOUND``."""
    filename = (filename or "").strip()
    curated = curated_comfy_download(filename)
    if curated is not None:
        return ComfyLookup(LOOKUP_FOUND, curated)
    folder, refusal = _check_request(filename, class_type, input_name)
    if refusal is not None:
        return refusal
    key = _cache_key(filename, folder)
    cached = _cache_get(key)
    if cached is not None:
        return cached

    from localm import discover
    from localm.model_source_credentials import get_hf_token
    token = get_hf_token()
    try:
        discover._ensure_online()
        candidates = _hf_candidates(filename, token)
    except discover.DiscoverError as e:
        if e.off:
            return ComfyLookup(LOOKUP_OFFLINE, detail=str(e))
        logger.debug("HuggingFace model-file search failed: %s", e)
        lookup = ComfyLookup(LOOKUP_FAILED, detail=str(e))
        _cache_put(key, lookup)
        return lookup
    if not candidates:
        lookup = ComfyLookup(
            LOOKUP_NOT_FOUND,
            detail="No public HuggingFace repository has a file with this name.")
        _cache_put(key, lookup)
        return lookup
    repo, path, _downloads = candidates[0]
    download = ComfyDownload(f"{repo}:{path}", folder_model_type(folder), folder,
                             _hf_file_size(repo, path, token), ORIGIN_HUGGINGFACE)
    lookup = ComfyLookup(LOOKUP_FOUND, download)
    _cache_put(key, lookup)
    return lookup
