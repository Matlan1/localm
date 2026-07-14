# SPDX-License-Identifier: AGPL-3.0-or-later
"""ComfyUI model directory scanner to auto-discover and register media models."""

from pathlib import Path
from typing import Optional, NamedTuple, Dict

from localm.config import load_config, load_registry
from localm.model_manager.registry import _register_with_dedup, _entry_path
from localm.media.comfy_client import comfy_object_info, _MODEL_FILE_EXTS, _looks_like_model_files
from localm.debuglog import logger

# Table mapping ComfyUI model subdirectories to localm model_types. Deliberately
# narrower than ComfyUI's OWN folder set (see media/managed_comfy.py's
# _MODEL_FOLDER_TYPES, which lists controlnet/upscale_models/embeddings/
# clip_vision/style_models for ComfyUI's own extra_model_paths.yaml): those
# conventions fall through to "unknown"/Other here BY DESIGN, not by omission.
# MODEL_TYPES (registry.py) is a closed enum wired through plugins/engine.py's
# routing, contract.py, and the CLI/GUI type-selector - localm's own generation
# pipeline never picks "which controlnet" or "which upscaler" through localm's
# model registry; ComfyUI resolves those itself from the workflow JSON via
# extra_model_paths.yaml, which already points at the same folders. Widening
# this table would need every one of those consumers taught new types for no
# functional gain today - the files still get registered (as unknown/Other)
# and remain fully usable by ComfyUI either way.
SUBFOLDER_MAPPING = {
    "unet": "diffusion-unet",
    "unet_gguf": "diffusion-unet",
    "clip": "text-encoder",
    "clip_gguf": "text-encoder",
    "text_encoders": "text-encoder",
    "vae": "vae",
    "vae_approx": "vae",
    "loras": "lora",
    "lora": "lora",
    "checkpoints": "diffusion-unet",
}

class ScanResult(NamedTuple):
    added: int
    skipped: int
    method: str

class ScanPreview(NamedTuple):
    """Dry-run result: counts by model_type for NEW (not-yet-registered) files
    only, plus how many discovered files are already registered. `method`
    carries the same human-decodable reason ScanResult.method does."""
    counts: Dict[str, int]
    already_registered: int
    method: str

def get_comfy_workdir() -> Optional[str]:
    cfg = load_config()
    # Check plugin-specific config overrides
    for p in ("image", "video", "music"):
        workdir = cfg.get("plugins", {}).get(p, {}).get("comfy", {}).get("workdir")
        if workdir:
            return workdir
    return cfg.get("comfy_workdir")

def get_comfy_api_url() -> str:
    from localm.media.comfy_client import default_api_url
    cfg = load_config()
    for p in ("image", "video", "music"):
        api_url = cfg.get("plugins", {}).get(p, {}).get("comfy", {}).get("api_url")
        if api_url:
            return api_url
    return default_api_url()

def _existing_registered_paths(reg: dict) -> set:
    """Resolved on-disk paths of every validly-pathed registry entry, skipping a
    malformed entry (non-dict, or a null / non-string / empty path). `"path" in
    entry` TypeErrors on a null entry and `Path(entry["path"])` raises on a null
    / int path; routing every entry through _entry_path keeps one corrupt row
    from crashing a scan or preview. Mirrors #562's registry consumers."""
    existing = set()
    for entry in reg.values():
        epath = _entry_path(entry)
        if epath is None:
            continue
        try:
            existing.add(Path(epath).resolve())
        except OSError:
            continue
    return existing


def _discover_comfy_files(workdir: str, comfy_url: Optional[str] = None):
    """Walk *workdir*/models (pass 1) and, if ComfyUI answers /object_info,
    reconcile types against its loader specs (pass 2). Shared by scan_comfy_models
    and preview_comfy_models so both use the EXACT same discovery logic.

    Returns (found_files, method). found_files is None (method explains why)
    only when the models folder itself is missing - the one discovery-level
    error a caller must treat specially rather than as an empty result."""
    models_path = Path(workdir) / "models"
    if not models_path.is_dir():
        return None, f"none (models folder not found under {workdir})"

    # Pass 1: Walk the local folders
    found_files: Dict[Path, str] = {}
    for sub, mtype in SUBFOLDER_MAPPING.items():
        sub_dir = models_path / sub
        if not sub_dir.is_dir():
            continue
        # Walk recursively to find all files with model extensions
        for path in sub_dir.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in _MODEL_FILE_EXTS:
                continue
            found_files[path] = mtype

    # Also check the rest of the folders for unknown files
    for child in models_path.iterdir():
        if child.is_dir() and child.name not in SUBFOLDER_MAPPING:
            for path in child.rglob("*"):
                if path.is_file() and path.suffix.lower() in _MODEL_FILE_EXTS:
                    if path not in found_files:
                        found_files[path] = "unknown"

    # Pass 2: Reconcile with /object_info if ComfyUI is online
    api_url = comfy_url or get_comfy_api_url()
    info = comfy_object_info(api_url)
    method = "folder-walk"

    if info:
        method = "hybrid (folder-walk + /object_info)"
        # Try to refine types based on loader specifications
        for node_name, spec in info.items():
            inputs = spec.get("input", {})
            for section in ("required", "optional"):
                sec = inputs.get(section, {})
                for input_name, input_def in sec.items():
                    if isinstance(input_def, list) and input_def and isinstance(input_def[0], list):
                        options = [o for o in input_def[0] if isinstance(o, str)]
                        if _looks_like_model_files(options):
                            inferred_type = "unknown"
                            node_lower = node_name.lower()
                            if "unet" in node_lower or "diffusion" in node_lower:
                                inferred_type = "diffusion-unet"
                            elif "clip" in node_lower or "textencode" in node_lower:
                                inferred_type = "text-encoder"
                            elif "vae" in node_lower:
                                inferred_type = "vae"
                            elif "lora" in node_lower:
                                inferred_type = "lora"

                            if inferred_type != "unknown":
                                for opt in options:
                                    opt_norm = opt.replace("\\", "/").lower()
                                    for path, cur_type in list(found_files.items()):
                                        try:
                                            path_rel = path.relative_to(models_path).as_posix().lower()
                                        except ValueError:
                                            continue
                                        if path_rel.endswith(opt_norm):
                                            found_files[path] = inferred_type

    return found_files, method


def scan_comfy_models(comfy_url: Optional[str] = None, workdir: Optional[str] = None) -> ScanResult:
    """Scan ComfyUI folders and/or /object_info and register newly discovered
    files. *workdir* overrides the configured comfy_workdir for a one-off scan
    of an arbitrary folder (e.g. the guided Import-from-ComfyUI flow) WITHOUT
    reading or mutating the persistent comfy_workdir config value."""
    wd = workdir or get_comfy_workdir()
    if not wd:
        return ScanResult(added=0, skipped=0, method="none (comfy_workdir not configured)")

    found_files, method = _discover_comfy_files(wd, comfy_url)
    if found_files is None:
        return ScanResult(added=0, skipped=0, method=method)

    # Now register the found files
    existing_paths = _existing_registered_paths(load_registry())

    added = 0
    skipped = 0

    for path, mtype in found_files.items():
        resolved_path = path.resolve()
        if resolved_path in existing_paths:
            skipped += 1
            continue

        model_name = path.stem
        try:
            reg = load_registry()
            from localm.model_manager.registry import _unique_registry_name
            unique_name = _unique_registry_name(reg, model_name)
            _register_with_dedup(
                unique_name,
                path,
                source="local-scanned",
                on_duplicate="register",
                model_type=mtype
            )
            added += 1
        except Exception as e:
            logger.error("Failed to register scanned model %s: %s", path, e)
            skipped += 1

    return ScanResult(added=added, skipped=skipped, method=method)


def preview_comfy_models(comfy_url: Optional[str] = None, workdir: Optional[str] = None) -> ScanPreview:
    """Dry-run of scan_comfy_models: identical discovery, but registers NOTHING.
    Returns per-type counts of NEW (not-yet-registered) files plus how many
    discovered files are already registered, so the guided Import-from-ComfyUI
    flow can show what a real scan WOULD do before the user confirms it."""
    wd = workdir or get_comfy_workdir()
    if not wd:
        return ScanPreview(counts={}, already_registered=0,
                            method="none (comfy_workdir not configured)")

    found_files, method = _discover_comfy_files(wd, comfy_url)
    if found_files is None:
        return ScanPreview(counts={}, already_registered=0, method=method)

    existing_paths = _existing_registered_paths(load_registry())
    counts: Dict[str, int] = {}
    already_registered = 0
    for path, mtype in found_files.items():
        if path.resolve() in existing_paths:
            already_registered += 1
            continue
        counts[mtype] = counts.get(mtype, 0) + 1

    return ScanPreview(counts=counts, already_registered=already_registered, method=method)
