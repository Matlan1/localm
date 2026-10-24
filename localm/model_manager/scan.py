# SPDX-License-Identifier: AGPL-3.0-or-later
"""ComfyUI model directory scanner to auto-discover and register media models."""

from pathlib import Path
from typing import Optional, NamedTuple

from localm.config import load_config, load_registry
from localm.model_manager.registry import (
    _register_with_dedup, _entry_path, _MEDIA_KIND_TYPES,
)
from localm.media.comfy_client import (
    comfy_object_info, model_type_for_node, _MODEL_FILE_EXTS, _looks_like_model_files,
)
from localm.debuglog import logger
from localm.model_manager.sam import is_sam_checkpoint

# Folder name (lower case) -> localm model_type, for ComfyUI's models/ tree and
# StabilityMatrix's Data/Models tree. Folders with no registry type map to
# "unknown"; they are still recognised as model folders. A file's own header
# (see _type_for_file) overrides its folder; the folder types a file the header
# cannot classify.
SUBFOLDER_MAPPING = {
    "unet": "diffusion-unet",
    "unet_gguf": "diffusion-unet",
    "diffusion_models": "diffusion-unet",
    "diffusionmodels": "diffusion-unet",
    "stablediffusion": "diffusion-unet",
    "checkpoints": "diffusion-unet",
    "clip": "text-encoder",
    "clip_gguf": "text-encoder",
    "text_encoders": "text-encoder",
    "textencoders": "text-encoder",
    "clip_vision": "text-encoder",
    "clipvision": "text-encoder",
    "vae": "vae",
    "vae_approx": "vae",
    "approxvae": "vae",
    "loras": "lora",
    "lora": "lora",
    "lycoris": "lora",
    "controlnet": "controlnet",
    "t2iadapter": "unknown",
    "ipadapter": "unknown",
    "ipadapters15": "unknown",
    "ipadaptersxl": "unknown",
    "embeddings": "textual-inversion",
    "upscale_models": "upscaler",
    "esrgan": "upscaler",
    "realesrgan": "upscaler",
    "style_models": "unknown",
    "stylemodels": "unknown",
    "hypernetwork": "unknown",
    "hypernetworks": "unknown",
    "diffusers": "unknown",
    "sam": "segmentation",
    "sams": "segmentation",
    "segmentation": "segmentation",
}

_HEADER_TYPED_EXTS = frozenset({".safetensors", ".sft", ".gguf"})
_COMPONENT_DIR_MARKERS = ("config.json", "model_index.json", "adapter_config.json")

class ScanResult(NamedTuple):
    added: int
    skipped: int
    method: str

class ScanPreview(NamedTuple):
    """Dry-run result: counts by model_type for NEW (not-yet-registered) files
    only, plus how many discovered files are already registered. `method`
    carries the same human-decodable reason ScanResult.method does."""
    counts: dict[str, int]
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
    return _models_dir_under(Path(workdir))


def _child_dir(parent: Path, name: str) -> Optional[Path]:
    """The subdirectory of *parent* named *name*, matched case-insensitively
    (StabilityMatrix writes ``Data/Models``, ComfyUI ``models``)."""
    try:
        for child in parent.iterdir():
            if child.name.lower() == name and child.is_dir():
                return child
    except OSError:
        return None
    return None


def _looks_like_models_dir(path: Path) -> bool:
    try:
        return any(c.is_dir() and c.name.lower() in SUBFOLDER_MAPPING
                   for c in path.iterdir())
    except OSError:
        return False


def _models_dir_under(root: Path) -> Path:
    """The models directory for a scan source *root*: a ComfyUI checkout
    (``root/models``), a StabilityMatrix data directory (``root/Models``) or
    install folder (``root/Data/Models``), or *root* itself when it already
    holds model folders. When none exists, ``root/models`` is returned so the
    caller reports it as not found."""
    direct = _child_dir(root, "models")
    if direct is not None:
        return direct
    data = _child_dir(root, "data")
    if data is not None:
        nested = _child_dir(data, "models")
        if nested is not None:
            return nested
    if _looks_like_models_dir(root):
        return root
    return root / "models"


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
    return _models_dir_under(Path(wd)) if wd else None


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


def _in_component_dir(path: Path, cache: dict) -> bool:
    """True when *path* sits in a HuggingFace/diffusers component directory
    (one that carries a config.json or model_index.json), whose files are parts
    of one model rather than standalone model files."""
    parent = path.parent
    hit = cache.get(parent)
    if hit is None:
        hit = any((parent / m).is_file() for m in _COMPONENT_DIR_MARKERS)
        cache[parent] = hit
    return hit


def _type_for_file(path: Path, folder_type: str, component_dirs: dict) -> str:
    """The model_type for one scanned file: what its header says (a diffusion
    model, text encoder, VAE or LoRA, per localm.media.sdcpp.families; a Segment
    Anything checkpoint is "segmentation"), else the type its folder names. Files in a component directory, and LoRAs of
    language models, take the folder type."""
    if is_sam_checkpoint(path):
        return "segmentation"
    if path.suffix.lower() in _HEADER_TYPED_EXTS and not _in_component_dir(path, component_dirs):
        from localm.media.sdcpp.families import inspect_file
        info = inspect_file(path)
        if not (info.kind == "lora" and info.family == "llm"):
            header_type = _MEDIA_KIND_TYPES.get(info.kind)
            if header_type is not None:
                return header_type
    return folder_type


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
    found_files: dict[Path, str] = {}
    component_dirs: dict[Path, bool] = {}
    try:
        folders = sorted(models_path.iterdir())
    except OSError:
        folders = []
    for folder in folders:
        if not folder.is_dir():
            continue
        folder_type = SUBFOLDER_MAPPING.get(folder.name.lower(), "unknown")
        for path in folder.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in _MODEL_FILE_EXTS:
                continue
            found_files[path] = _type_for_file(path, folder_type, component_dirs)

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
                for _input_name, input_def in sec.items():
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
                                    for path in list(found_files):
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
    counts: dict[str, int] = {}
    already_registered = 0
    for path, mtype in found_files.items():
        if path.resolve() in existing_paths:
            already_registered += 1
            continue
        counts[mtype] = counts.get(mtype, 0) + 1

    return ScanPreview(counts=counts, already_registered=already_registered, method=method)
