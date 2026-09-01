# SPDX-License-Identifier: AGPL-3.0-or-later
"""Model source seam: search() / list_files() / resolve_download(), the same
three operations across every model provider the model manager knows about.

Internal to model_manager, not a plugins/contract.py capability - HFSource
and CivitAISource are the only two implementations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable

CIVITAI_API = "https://civitai.com"
_CIVITAI_TIMEOUT = 20


@dataclass(frozen=True)
class ResolvedDownload:
    """Everything a downloader needs for one file, plus where the registry
    should file it. *comfy_subfolder* is None for a flat MODELS_DIR
    destination (HF's current layout); a source that routes into the
    ComfyUI-subfoldered tree (CivitAI) sets it to a managed_comfy.py
    _MODEL_FOLDER_TYPES name."""

    url: str
    filename: str
    source_tag: str
    model_type: str
    size_bytes: Optional[int] = None
    sha256: Optional[str] = None
    comfy_subfolder: Optional[str] = None


@runtime_checkable
class ModelSource(Protocol):
    """search() returns provider-shaped result rows for browsing. list_files()
    returns the downloadable files for one search result. resolve_download()
    turns one (ref, file) pair into a ResolvedDownload - for HFSource this is
    descriptive only, since the real HF transfer still runs through
    pull_model()/_pull_gguf_file() unchanged; for CivitAISource it performs
    the metadata lookups a download needs (type mapping, minor/format checks)
    but never follows the time-boxed download redirect itself - that happens
    at download time in pull.py, once, immediately before the GET."""

    name: str

    def search(self, query: str = "", *, limit: int = 20, **kwargs) -> dict: ...

    def list_files(self, ref: str, **kwargs) -> list[dict]: ...

    def resolve_download(self, ref: str, file: object, **kwargs) -> ResolvedDownload: ...


class HFSource:
    """HuggingFace, via the existing localm.discover / pull.py machinery."""

    name = "hf"

    def search(self, query: str = "", *, limit: int = 20,
               token: Optional[str] = None, **kwargs) -> dict:
        from .. import discover
        if token is None:
            from ..model_source_credentials import get_hf_token
            token = get_hf_token()
        return {"items": discover.hf_search(query, limit=limit, token=token, **kwargs)}

    def list_files(self, ref: str, *, token: Optional[str] = None,
                   **kwargs) -> list[dict]:
        from .. import discover
        if token is None:
            from ..model_source_credentials import get_hf_token
            token = get_hf_token()
        return discover.hf_gguf_files(ref, token=token)

    def resolve_download(self, ref: str, file: object, *,
                          model_type: str = "llm",
                          token: Optional[str] = None) -> ResolvedDownload:
        """Descriptive only. Real HF pulls dispatch through pull_model(), whose
        resumable local-dir download and split-file assembly stay unchanged."""
        from huggingface_hub import hf_hub_url

        from .pull import _HF_ENDPOINT, _hf_file_sha256
        if token is None:
            from ..model_source_credentials import get_hf_token
            token = get_hf_token()
        filename = file["file"] if isinstance(file, dict) else str(file)
        size = file.get("size_bytes") if isinstance(file, dict) else None
        return ResolvedDownload(
            url=hf_hub_url(ref, filename, endpoint=_HF_ENDPOINT),
            filename=filename,
            source_tag=f"hf:{ref}",
            model_type=model_type,
            size_bytes=size,
            sha256=_hf_file_sha256(ref, filename, token=token),
            comfy_subfolder=None,
        )


# --------------------------------------------------------------------------- #
#  CivitAI                                                                     #
# --------------------------------------------------------------------------- #

# CivitAI ModelType -> (registry.MODEL_TYPES value, ComfyUI models/ subfolder -
# see media/managed_comfy.py _MODEL_FOLDER_TYPES). A CivitAI type absent from
# this table is excluded from search results and refused by resolve_download,
# rather than mis-filed under a folder ComfyUI does not scan for that role.
#
# Controlnet/Upscaler have no registry.MODEL_TYPES value (the frozenset has no
# "controlnet"/"upscaler" member); they still route to a real ComfyUI
# subfolder under model_type="unknown" rather than being excluded outright.
CIVITAI_TYPE_MAP: dict[str, tuple[str, str]] = {
    "Checkpoint": ("diffusion-unet", "checkpoints"),
    "LORA": ("lora", "loras"),
    "LoCon": ("lora", "loras"),
    "DoRA": ("lora", "loras"),
    "TextualInversion": ("embedding", "embeddings"),
    "VAE": ("vae", "vae"),
    "Controlnet": ("unknown", "controlnet"),
    "Upscaler": ("unknown", "upscale_models"),
}

# File formats that cannot execute code on load: safetensors is a plain tensor
# container, GGUF is llama.cpp's own binary format, Diffusers is a directory
# layout of the same. Everything else (PickleTensor, Other, and a missing
# format) sits behind include_legacy_formats.
CIVITAI_SAFE_FORMATS = frozenset({"SafeTensor", "GGUF", "Diffusers"})


def civitai_searchable_types() -> list[str]:
    """The CivitAI `types` values search()/list_files()/resolve_download()
    will accept and place - see CIVITAI_TYPE_MAP."""
    return sorted(CIVITAI_TYPE_MAP)


class ModelSourceError(Exception):
    """A model-source request failed, was refused by policy, or named
    something this source cannot resolve (an unknown source name; for
    CivitAISource specifically, also an excluded type, a hard-excluded minor
    flag, or a legacy format with include_legacy_formats not set)."""

    def __init__(self, message: str, *, off: bool = False):
        super().__init__(message)
        self.off = off


class CivitAIError(ModelSourceError):
    """A CivitAI-specific ModelSourceError - see the base class docstring."""


def _civitai_get(path: str, params: Optional[dict] = None, *,
                  api_key: Optional[str] = None) -> dict:
    """Policy-checked GET against the CivitAI public API, returning parsed
    JSON. Same SSRF/redirect-revalidation treatment discover._get gives every
    HF metadata call - every hop is re-checked, so a redirect from civitai.com
    cannot bounce this into a private address.

    *api_key*: optional CivitAI API key, sent as an Authorization header
    (CivitAI's own convention). Raises rate limits and lets gated resources
    resolve; omitted, the request runs anonymously exactly as before."""
    import json as _json
    import urllib.parse

    from .. import netpolicy
    full = CIVITAI_API + path
    if params:
        full += "?" + urllib.parse.urlencode(params, doseq=True)
    extra_headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
    try:
        _final, _ctype, body = netpolicy.safe_fetch_bytes(
            full, max_bytes=32 * 1024 * 1024, timeout=_CIVITAI_TIMEOUT,
            allow_when_off=netpolicy.downloads_allowed_when_off(),
            extra_headers=extra_headers)
        return _json.loads(body.decode("utf-8"))
    except netpolicy.NetworkPolicyError as e:
        raise CivitAIError(f"CivitAI request failed: {e}", off=e.off)
    except Exception as e:
        raise CivitAIError(f"CivitAI request failed: {e}")


def _passes_content_policy(item: dict) -> bool:
    """Hard exclusion for minor-tagged content, applied regardless of the nsfw
    toggle - a distinct policy category, not a maturity level."""
    return item.get("minor") is not True


def civitai_search(query: str = "", *, limit: int = 20,
                    types: "Optional[list[str]]" = None,
                    base_models: "Optional[list[str]]" = None,
                    tag: Optional[str] = None, period: Optional[str] = None,
                    sort: Optional[str] = None, nsfw: bool = False,
                    cursor: Optional[str] = None,
                    api_key: Optional[str] = None) -> dict:
    """Search CivitAI models. *nsfw* defaults to False regardless of CivitAI's
    own server default, mirroring rather than merely inheriting it. Every
    result also passes the hard minor-exclusion check independent of *nsfw*.

    *types*, when given, is intersected with CIVITAI_TYPE_MAP so a caller
    cannot ask for (and cannot receive) a type this source does not place;
    omitted entirely, results are filtered to only the mapped types after the
    fact, since CivitAI's own `types` filter is not required for correctness
    here - the client-side filter is.

    Returns {"items": [...], "next_cursor": str | None}."""
    params: dict = {"limit": max(1, min(int(limit), 100)), "nsfw": "true" if nsfw else "false"}
    if query.strip():
        params["query"] = query.strip()
    requested_types = ([t for t in types if t in CIVITAI_TYPE_MAP]
                        if types else list(CIVITAI_TYPE_MAP))
    if not requested_types:
        raise CivitAIError("No valid model type requested for CivitAI search.")
    params["types"] = requested_types
    if base_models:
        params["baseModels"] = list(base_models)
    if tag:
        params["tag"] = tag
    if period:
        params["period"] = period
    if sort:
        params["sort"] = sort
    if cursor:
        params["cursor"] = cursor

    data = _civitai_get("/api/v1/models", params, api_key=api_key)
    items = [it for it in data.get("items", [])
             if isinstance(it, dict) and it.get("type") in CIVITAI_TYPE_MAP
             and _passes_content_policy(it)]
    return {"items": items,
            "next_cursor": (data.get("metadata") or {}).get("nextCursor")}


def civitai_list_files(version_id: object, *,
                        include_legacy_formats: bool = False,
                        api_key: Optional[str] = None) -> list[dict]:
    """The downloadable files for one CivitAI model VERSION id, each still
    carrying its own hashes/scan-status/format fields for display. Defaults to
    CIVITAI_SAFE_FORMATS only; include_legacy_formats=True also returns
    PickleTensor/Other/unset-format files."""
    data = _civitai_get(f"/api/v1/model-versions/{version_id}", api_key=api_key)
    out = []
    for f in data.get("files", []):
        if not isinstance(f, dict):
            continue
        fmt = (f.get("metadata") or {}).get("format")
        if not include_legacy_formats and fmt not in CIVITAI_SAFE_FORMATS:
            continue
        out.append(f)
    return out


def _pick_civitai_file(files: "list[dict]", file_id: Optional[object]) -> Optional[dict]:
    if file_id is not None:
        for f in files:
            if str(f.get("id")) == str(file_id):
                return f
        return None
    for f in files:
        if f.get("primary"):
            return f
    return files[0] if files else None


class CivitAISource:
    """CivitAI, via its public v1 API. A token only raises rate limits and
    unlocks gated resources - never a precondition for search or download.
    Resolved from model_source_credentials.get_civitai_api_key() when a
    caller does not pass api_key explicitly."""

    name = "civitai"

    def search(self, query: str = "", *, limit: int = 20,
               api_key: Optional[str] = None, **kwargs) -> dict:
        if api_key is None:
            from ..model_source_credentials import get_civitai_api_key
            api_key = get_civitai_api_key()
        return civitai_search(query, limit=limit, api_key=api_key, **kwargs)

    def list_files(self, ref: object, *, api_key: Optional[str] = None,
                   **kwargs) -> list[dict]:
        if api_key is None:
            from ..model_source_credentials import get_civitai_api_key
            api_key = get_civitai_api_key()
        return civitai_list_files(ref, api_key=api_key, **kwargs)

    def resolve_download(self, ref: object, file: object, *,
                          include_legacy_formats: bool = False,
                          api_key: Optional[str] = None) -> ResolvedDownload:
        """*ref* is a CivitAI model-VERSION id. *file* is a file id, or a file
        dict already carrying one (from list_files()). Re-fetches the version
        (and the owning model, for the authoritative minor flag) rather than
        trusting a caller-supplied dict for anything that decides placement or
        the content-policy exclusion - list_files()'s own filtering already
        applied to what it returned, but a caller may hold a file id from
        anywhere.

        Never resolves the actual download redirect (that target is a
        time-boxed pre-signed URL - see pull.py's _ssrf_resolve_final_url,
        called once, immediately before the transfer)."""
        if api_key is None:
            from ..model_source_credentials import get_civitai_api_key
            api_key = get_civitai_api_key()
        version_id = ref
        version = _civitai_get(f"/api/v1/model-versions/{version_id}", api_key=api_key)
        model_id = version.get("modelId")
        if model_id is None:
            raise CivitAIError(
                f"CivitAI version {version_id} has no owning model id - "
                "cannot confirm its content-policy flags, refusing to resolve.")
        model = _civitai_get(f"/api/v1/models/{model_id}", api_key=api_key)
        if not _passes_content_policy(model):
            raise CivitAIError(
                f"CivitAI model {model_id} is excluded from download "
                "(flagged as depicting a minor).")

        civitai_type = (version.get("model") or {}).get("type") or model.get("type")
        mapped = CIVITAI_TYPE_MAP.get(civitai_type)
        if mapped is None:
            raise CivitAIError(
                f"CivitAI type '{civitai_type}' has no known ComfyUI placement "
                "and cannot be downloaded through this source.")
        model_type, subfolder = mapped

        file_id = file.get("id") if isinstance(file, dict) else file
        all_files = [f for f in version.get("files", []) if isinstance(f, dict)]
        if file_id is not None:
            # An explicitly-named file is looked up among every file on the
            # version, regardless of format - the legacy-format gate below
            # still applies to it, rather than silently substituting a
            # different file the caller did not ask for.
            candidates = all_files
        else:
            safe_files = [f for f in all_files
                          if (f.get("metadata") or {}).get("format") in CIVITAI_SAFE_FORMATS]
            candidates = safe_files if safe_files or not include_legacy_formats else all_files
        picked = _pick_civitai_file(candidates, file_id)
        if picked is None:
            raise CivitAIError(f"No matching file on CivitAI version {version_id}.")
        fmt = (picked.get("metadata") or {}).get("format")
        if not include_legacy_formats and fmt not in CIVITAI_SAFE_FORMATS:
            raise CivitAIError(
                f"{picked.get('name', '(unnamed file)')} is a legacy "
                f"({fmt or 'unknown'}) format - pass include_legacy_formats=True "
                "to download it anyway.")

        download_url = picked.get("downloadUrl")
        if not download_url:
            raise CivitAIError(f"CivitAI file {file_id} has no download URL.")
        sha256 = ((picked.get("hashes") or {}).get("SHA256") or "").lower() or None
        size_kb = picked.get("sizeKB")
        size_bytes = int(size_kb * 1024) if isinstance(size_kb, (int, float)) else None

        return ResolvedDownload(
            url=download_url,
            filename=picked.get("name") or f"civitai-{version_id}",
            source_tag=f"civitai:{version_id}",
            model_type=model_type,
            size_bytes=size_bytes,
            sha256=sha256,
            comfy_subfolder=subfolder,
        )


SOURCES: "dict[str, ModelSource]" = {"hf": HFSource(), "civitai": CivitAISource()}


def get_source(name: str) -> ModelSource:
    try:
        return SOURCES[name]
    except KeyError:
        raise ModelSourceError(f"Unknown model source: {name}") from None
