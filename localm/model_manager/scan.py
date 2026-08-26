# SPDX-License-Identifier: AGPL-3.0-or-later
"""ComfyUI model directory scanner to auto-discover and register media models."""

from pathlib import Path
from typing import Optional, NamedTuple, Dict

from localm.config import load_config, load_registry
from localm.model_manager.registry import _register_with_dedup, _entry_path
from localm.media.comfy_client import (
    comfy_object_info, model_type_for_node, _MODEL_FILE_EXTS, _looks_like_model_files,
)
from localm.debuglog import logger

# Table mapping ComfyUI model subdirectories to localm model_types. Narrower
# than ComfyUI's OWN folder set (see media/managed_comfy.py's
# _MODEL_FOLDER_TYPES): controlnet/upscale_models/embeddings/clip_vision/
# style_models fall through to "unknown"/Other here. Those files are still
# registered and stay usable by ComfyUI, which resolves them itself from the
# workflow JSON via extra_model_paths.yaml.
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
    # Managed routing is absolute: when localm's own ComfyUI is selected and
    # installed, that IS the folder to scan, and no per-plugin or global workdir
    # shadows it. This scan has no single "current plugin" to resolve against,
    # so the non-managed fallback below checks all three plugin blocks itself
    # instead of calling resolve_comfy_target() with a specific plugin.
    from localm.media.managed_comfy import resolve_comfy_target
    target = resolve_comfy_target()
    if target.managed:
        return target.workdir
    cfg = load_config()
    # Check plugin-specific config overrides
    for p in ("image", "music", "video"):
        workdir = cfg.get("plugins", {}).get(p, {}).get("comfy", {}).get("workdir")
        if workdir:
            return workdir
    return cfg.get("comfy_workdir")

def _resolve_explicit_workdir_models_path(workdir: str) -> Path:
    """An explicit *workdir* override (the guided Import-from-ComfyUI flow) is
    normally a real ComfyUI checkout root, so its models live at
    <workdir>/models. The one exception is the "Use localm's own ComfyUI"
    quick-fill, which hands back the managed checkout root itself. A managed
    install's models live in a SIBLING directory, never inside a `models`
    subfolder under the checkout (managed_comfy_provision.py's copy step
    excludes "models"), so that one root is recognized and redirected to the
    real managed models dir."""
    from localm.media.managed_comfy import managed_comfy_paths
    paths = managed_comfy_paths()
    if Path(workdir).resolve() == paths.root.resolve():
        return paths.models_dir
    return Path(workdir) / "models"


def _resolve_scan_models_path(workdir: Optional[str]) -> Optional[Path]:
    """Resolve the actual directory scan_comfy_models/preview_comfy_models
    should walk. An explicit *workdir* is a one-off override (see
    _resolve_explicit_workdir_models_path). With none given, this makes the same
    managed-routing check get_comfy_workdir() makes but resolves straight to the
    managed ComfyUI's actual models directory: the managed instance's models
    live directly under <LOCALM_HOME>/comfyui-models, a SIBLING of the checkout
    root, never inside a `models` subfolder under it. Falls back to
    get_comfy_workdir() + "/models" when not managed. Returns None when nothing
    is configured or resolvable."""
    if workdir:
        return _resolve_explicit_workdir_models_path(workdir)
    from localm.media.managed_comfy import managed_comfy_active, managed_comfy_paths
    if managed_comfy_active():
        return managed_comfy_paths().models_dir
    wd = get_comfy_workdir()
    return Path(wd) / "models" if wd else None


def get_comfy_api_url() -> str:
    from localm.media.comfy_client import default_api_url
    from localm.media.managed_comfy import managed_comfy_active
    cfg = load_config()
    # Same "own means own" rule as get_comfy_workdir() above: a per-plugin/
    # global api_url override is only consulted when NOT routed to the
    # managed instance - default_api_url() alone already resolves the managed
    # URL correctly when it is.
    if not managed_comfy_active(cfg):
        for p in ("image", "music", "video"):
            api_url = cfg.get("plugins", {}).get(p, {}).get("comfy", {}).get("api_url")
            if api_url:
                return api_url
    return default_api_url()

def _existing_registered_paths(reg: dict) -> set:
    """Resolved on-disk paths of every validly-pathed registry entry, skipping a
    malformed entry (non-dict, or a null / non-string / empty path). `"path" in
    entry` TypeErrors on a null entry and `Path(entry["path"])` raises on a null
    / int path; routing every entry through _entry_path keeps one corrupt row
    from crashing a scan or preview."""
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


def _discover_comfy_files(models_path: Path, comfy_url: Optional[str] = None):
    """Walk *models_path* (pass 1) and, if ComfyUI answers /object_info,
    reconcile types against its loader specs (pass 2). Shared by scan_comfy_models
    and preview_comfy_models so both use the EXACT same discovery logic.
    *models_path* is the already-resolved directory to walk (see
    _resolve_scan_models_path) - a real ComfyUI's <checkout>/models, or the
    managed instance's own models dir; this function never derives it.

    Returns (found_files, method). found_files is None (method explains why)
    only when the models folder itself is missing - the one discovery-level
    error a caller must treat specially rather than as an empty result."""
    if not models_path.is_dir():
        return None, f"none (models folder not found under {models_path})"

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
                            # ONE shared node-name -> model_type inference: the
                            # media plugins' model-role wiring reads the SAME
                            # table (comfy_client.model_type_for_node).
                            inferred_type = model_type_for_node(node_name)
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


def scan_comfy_models(comfy_url: Optional[str] = None, workdir: Optional[str] = None,
                       *, progress_cb=None) -> ScanResult:
    """Scan ComfyUI folders and/or /object_info and register newly discovered
    files. *workdir* overrides the configured comfy_workdir for a one-off scan
    of an arbitrary folder (e.g. the guided Import-from-ComfyUI flow) WITHOUT
    reading or mutating the persistent comfy_workdir config value.

    *progress_cb*, when given, is called as ``progress_cb(done, total, name)``
    once per discovered file as it is registered (or found already registered),
    which is the one point in this function with a denominator - the directory
    walk above it has none. The GUI route wires this to Job.progress(). A caller
    that passes nothing sees no behavior change."""
    models_path = _resolve_scan_models_path(workdir)
    if models_path is None:
        return ScanResult(added=0, skipped=0, method="none (comfy_workdir not configured)")

    found_files, method = _discover_comfy_files(models_path, comfy_url)
    if found_files is None:
        return ScanResult(added=0, skipped=0, method=method)

    # Now register the found files
    existing_paths = _existing_registered_paths(load_registry())

    added = 0
    skipped = 0
    total = len(found_files)

    for i, (path, mtype) in enumerate(found_files.items(), start=1):
        if progress_cb is not None:
            progress_cb(i, total, path.name)
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
    models_path = _resolve_scan_models_path(workdir)
    if models_path is None:
        return ScanPreview(counts={}, already_registered=0,
                            method="none (comfy_workdir not configured)")

    found_files, method = _discover_comfy_files(models_path, comfy_url)
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
