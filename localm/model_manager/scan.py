# SPDX-License-Identifier: AGPL-3.0-or-later
"""ComfyUI model directory scanner to auto-discover and register media models."""

import json
import urllib.request
from pathlib import Path
from typing import Optional, NamedTuple, Dict, Any, List

from localm.config import load_config, load_registry
from localm.model_manager.registry import _register_with_dedup, MODEL_TYPES
from localm.media.comfy_client import comfy_object_info, _MODEL_FILE_EXTS, _combo_options, _looks_like_model_files
from localm.debuglog import logger

# Table mapping ComfyUI model subdirectories to localm model_types
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

def scan_comfy_models(comfy_url: Optional[str] = None) -> ScanResult:
    """Scan ComfyUI folders and/or /object_info and register newly discovered files."""
    workdir = get_comfy_workdir()
    if not workdir:
        return ScanResult(added=0, skipped=0, method="none (comfy_workdir not configured)")

    models_path = Path(workdir) / "models"
    if not models_path.is_dir():
        return ScanResult(added=0, skipped=0, method=f"none (models folder not found under {workdir})")

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

    # Now register the found files
    reg = load_registry()
    existing_paths = {Path(entry["path"]).resolve() for entry in reg.values() if "path" in entry}

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
