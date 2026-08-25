# SPDX-License-Identifier: AGPL-3.0-or-later
"""localm-managed ComfyUI: paths, install detection, coexistence routing."""

from __future__ import annotations

import os
import shutil
import stat
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from localm.config import home_dir, load_config

# Serializes remove_managed_comfy() against itself: the CLI (`localm comfy
# remove`) and the GUI's /api/comfy/remove + /api/comfy/repair routes are two
# independent callers of the SAME rmtree target, and the GUI routes now wrap
# their call in run_in_threadpool_bounded (a client-side deadline - see
# localm/inference/_threadpool_timeout.py). That deadline only makes the
# AWAITING request give up; the real rmtree keeps running on its abandoned
# worker thread. Without this lock, a user who sees a timeout and retries
# Remove/Repair would start a SECOND concurrent rmtree against the same
# directory tree while the first is still deleting files - shutil.rmtree
# racing itself, the same "offloading without serializing" shape
# media_workflows.py's _lock_for already fixed for the workflow routes (see
# its module comment). One lock is enough here (unlike per-media there):
# there is only ever ONE managed ComfyUI install, not one per media type.
_remove_lock = threading.Lock()

# Directory names under LOCALM_HOME. Kept as constants so S2/S3 and the CLI all
# agree on the one layout.
_INSTALL_DIRNAME = "comfyui"
_MODELS_DIRNAME = "comfyui-models"

# The managed instance runs on its OWN loopback port, distinct from the ComfyUI
# default (8188) so localm's managed ComfyUI and a user's own ComfyUI can run at
# the same time (coexistence). localm claims 8642-8741 for its own server; the
# managed ComfyUI sits just above the ComfyUI default instead, on 8189, so it
# reads as "a second ComfyUI" to anyone inspecting ports. S2/S3 launch it here.
MANAGED_COMFY_PORT = 8189
MANAGED_COMFY_API_URL = f"http://127.0.0.1:{MANAGED_COMFY_PORT}"

# Standard ComfyUI model-folder types written into extra_model_paths.yaml. Each
# maps to a same-named subfolder under the entry's base_path. This is the common
# core set; the generated file carries a header telling the user how to add more.
_MODEL_FOLDER_TYPES = (
    "checkpoints", "clip", "clip_vision", "configs", "controlnet",
    "diffusion_models", "embeddings", "loras", "style_models",
    "text_encoders", "unet", "upscale_models", "vae", "vae_approx",
)


@dataclass(frozen=True)
class ManagedComfyPaths:
    """Resolved on-disk locations for the managed ComfyUI."""
    root: Path                # <HOME>/comfyui  (checkout + venv)
    models_dir: Path          # <HOME>/comfyui-models
    main_py: Path             # <root>/main.py         (ComfyUI entry point)
    venv_python: Path         # the managed venv interpreter (platform-specific)
    extra_model_paths: Path   # <root>/extra_model_paths.yaml


def _venv_python_path(root: Path) -> Path:
    """The managed venv's interpreter path for this platform (not required to exist here)."""
    if os.name == "nt":
        return root / "venv" / "Scripts" / "python.exe"
    return root / "venv" / "bin" / "python"


def managed_comfy_paths() -> ManagedComfyPaths:
    """Where a localm-managed ComfyUI lives under the CURRENT LOCALM_HOME."""
    root = home_dir() / _INSTALL_DIRNAME
    return ManagedComfyPaths(
        root=root,
        models_dir=home_dir() / _MODELS_DIRNAME,
        main_py=root / "main.py",
        venv_python=_venv_python_path(root),
        extra_model_paths=root / "extra_model_paths.yaml",
    )


def rmtree_robust(path: Path) -> None:
    """``shutil.rmtree`` that also removes read-only files."""
    def _onerror(func, p, _exc_info):
        # Clear the read-only bit and retry; let a genuine failure propagate.
        os.chmod(p, stat.S_IWRITE)
        func(p)
    shutil.rmtree(str(path), onerror=_onerror)


def is_managed_comfy_installed() -> bool:
    """True when a usable managed ComfyUI is actually present on disk AND the provisioning pipeline (S2/S3) actually finished."""
    paths = managed_comfy_paths()
    try:
        if not (paths.main_py.is_file() and paths.venv_python.is_file()):
            return False
        from localm.media.managed_comfy_provision import MARKER_FILENAME
        return (paths.root / MARKER_FILENAME).is_file()
    except OSError:
        return False


def managed_comfy_remove_targets(with_models: bool = False) -> list:
    """The on-disk paths ``remove`` would delete: the managed ComfyUI checkout, and (only with ``with_models``) the managed models folder."""
    paths = managed_comfy_paths()
    targets = []
    if paths.root.exists():
        targets.append(paths.root)
    if with_models and paths.models_dir.exists():
        targets.append(paths.models_dir)
    return targets


def remove_managed_comfy(with_models: bool = False) -> tuple:
    """Delete the managed ComfyUI (and its models folder with ``with_models``) under the localm data dir."""
    with _remove_lock:
        removed = []
        failed = []
        for t in managed_comfy_remove_targets(with_models):
            try:
                rmtree_robust(t)
                removed.append(t)
            except OSError as e:
                failed.append(f"{t} ({e})")
        return removed, failed


def managed_comfy_api_url() -> str:
    """The managed instance's base URL."""
    return MANAGED_COMFY_API_URL


def managed_comfy_workdir() -> str:
    """The working directory to launch/target the managed ComfyUI from (its checkout root)."""
    return str(managed_comfy_paths().root)


def managed_comfy_launch_cmd(config: Optional[dict] = None) -> str:
    """The command that starts the managed instance: its OWN venv interpreter running its OWN ``main.py``, on the managed port - never the user's ``comfy_launch_cmd``/auto-discovered launcher script, which only make sense for a user-provided ComfyUI (their own launcher, possibly ZLUDA-wrapped)."""
    paths = managed_comfy_paths()
    cmd = (f'"{paths.venv_python}" "{paths.main_py}" '
           f'--listen 127.0.0.1 --port {MANAGED_COMFY_PORT}')
    from localm.discover import resolve_preferred_device
    device = resolve_preferred_device(config)
    if device is not None:
        cmd += f" --default-device {device}"
    return cmd


def managed_comfy_active(cfg: Optional[dict] = None) -> bool:
    """True when media calls should target the MANAGED instance (decision 6): the target is 'own' AND an instance is installed."""
    cfg = cfg if cfg is not None else load_config()
    if cfg.get("comfy_target", "own") != "own":
        return False
    return is_managed_comfy_installed()


def legacy_comfy_value(key: str, full_config: dict) -> str:
    """A LEGACY GLOBAL comfy_* config value (``comfy_workdir``, ``comfy_launch_cmd``), suppressed to '' whenever the managed instance is active."""
    if managed_comfy_active(full_config):
        return ""
    return full_config.get(key, "") or ""


def managed_comfy_api_url_if_active(cfg: Optional[dict] = None) -> Optional[str]:
    """The managed api_url when managed routing is active, else None."""
    return managed_comfy_api_url() if managed_comfy_active(cfg) else None


@dataclass(frozen=True)
class ComfyTarget:
    """Which ComfyUI localm targets: its URL, working dir, launch command (None when the caller should fall back to its own discovery, e.g. the user's own install), and whether it is the managed instance (True) or the user's own (False)."""
    api_url: str
    workdir: Optional[str]
    launch_cmd: Optional[str]
    managed: bool


def resolve_comfy_target(cfg: Optional[dict] = None,
                         plugin: Optional[str] = None) -> ComfyTarget:
    """THE single coexistence resolver (decision 6)."""
    cfg = cfg if cfg is not None else load_config()
    if managed_comfy_active(cfg):
        return ComfyTarget(api_url=managed_comfy_api_url(),
                           workdir=managed_comfy_workdir(),
                           launch_cmd=managed_comfy_launch_cmd(), managed=True)
    # User's ComfyUI. Import here (not at module load) to avoid a circular
    # import: comfy_client imports this module lazily too.
    from localm.media.comfy_client import default_api_url
    workdir = cfg.get("comfy_workdir")
    if plugin:
        from localm.plugins.media_config import resolve_config
        block, _warning = resolve_config(plugin, cfg)
        comfy_blk = block.get("comfy") if isinstance(block.get("comfy"), dict) else {}
        workdir = comfy_blk.get("workdir") or workdir
    return ComfyTarget(api_url=default_api_url(),
                       workdir=workdir, launch_cmd=None,
                       managed=False)


def comfy_models_dest_dir(subfolder: str, cfg: Optional[dict] = None,
                          plugin: Optional[str] = None) -> Optional[Path]:
    """Absolute ``models/<subfolder>`` directory for whichever ComfyUI ``resolve_comfy_target()`` says is currently active - the destination a downloaded model file needs to land in for THAT ComfyUI to see it."""
    target = resolve_comfy_target(cfg, plugin=plugin)
    if target.managed:
        return managed_comfy_paths().models_dir / subfolder
    if target.workdir:
        return Path(target.workdir) / "models" / subfolder
    return None


# --------------------------------------------------------------------------- #
#  extra_model_paths.yaml generator (decision 9)                              #
#                                                                             #
#  ComfyUI-native model sharing: point the managed ComfyUI at the user's      #
#  existing models dir (no copy, no symlink, no re-download) AND the           #
#  localm-managed models dir. This just WRITES the file; S2/S3 invoke it as    #
#  part of an install. Written by hand (no PyYAML dependency in shipped code - #
#  AGENTS.md rule 4: self-contained; PyYAML is only a transitive dep).        #
# --------------------------------------------------------------------------- #

def build_extra_model_paths_config(cfg: Optional[dict] = None) -> dict:
    """The extra_model_paths structure as a plain dict: one entry per model source, each ``{base_path, <type>: <subdir>, ...}``."""
    cfg = cfg if cfg is not None else load_config()
    paths = managed_comfy_paths()

    def _entry(base: Path) -> dict:
        e = {"base_path": str(base)}
        for t in _MODEL_FOLDER_TYPES:
            e[t] = t
        return e

    config: dict = {"localm_managed": _entry(paths.models_dir)}

    workdir = cfg.get("comfy_workdir")
    if workdir:
        # Standard ComfyUI stores models under <workdir>/models/<type>.
        config["user_comfyui"] = _entry(Path(workdir) / "models")
    return config


def _yaml_squote(s: str) -> str:
    """Single-quote a YAML scalar (backslash is LITERAL inside single quotes, so Windows paths survive; an embedded single quote is doubled per the spec)."""
    return "'" + s.replace("'", "''") + "'"


def render_extra_model_paths(cfg: Optional[dict] = None) -> str:
    """Render the extra_model_paths.yaml TEXT from build_extra_model_paths_config."""
    config = build_extra_model_paths_config(cfg)
    lines = [
        "# extra_model_paths.yaml - written by localm (managed ComfyUI, stage S1).",
        "# Tells this ComfyUI where to find models WITHOUT copying them:",
        "#   localm_managed = localm's own managed models dir",
        "#   user_comfyui   = your existing ComfyUI's models (when localm knows it)",
        "# Add your own sources by copying an entry below and editing base_path.",
        "",
    ]
    for name, entry in config.items():
        lines.append(f"{name}:")
        # base_path first for readability, then the folder types in order.
        lines.append(f"  base_path: {_yaml_squote(str(entry['base_path']))}")
        for key, val in entry.items():
            if key == "base_path":
                continue
            lines.append(f"  {key}: {_yaml_squote(str(val))}")
        lines.append("")
    return "\n".join(lines)


def write_extra_model_paths(cfg: Optional[dict] = None) -> Path:
    """Write extra_model_paths.yaml into the managed ComfyUI dir and return its path."""
    paths = managed_comfy_paths()
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.extra_model_paths.write_text(render_extra_model_paths(cfg), encoding="utf-8")
    return paths.extra_model_paths
