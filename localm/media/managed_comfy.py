# SPDX-License-Identifier: AGPL-3.0-or-later
"""
localm-managed ComfyUI: paths, install detection, coexistence routing.

STAGE S1 (scaffolding). This module knows WHERE a localm-owned ComfyUI would
live (under LOCALM_HOME, AGENTS.md rule 4: self-contained), how to tell whether
one is actually installed, and which ComfyUI localm should target when it is
(the coexistence resolver, decision 6). It deliberately does NOT provision
anything: copying the user's stack is S2, the fresh hardware-matched install is
S3. Everything here is inert until a managed instance exists on disk.

The single source of truth for "which ComfyUI does localm talk to" is
``resolve_comfy_target()``. The rule (locked decision 6):

    target the MANAGED instance  IFF  managed_comfy_enabled
                                  AND  comfy_target == "own"
                                  AND  a managed instance is actually installed
    otherwise                         the user's ComfyUI, exactly as today.

When nothing is set up, behaviour is byte-identical to before this module
existed: the user's ComfyUI, untouched. The user's ComfyUI is NEVER modified.

Layout under LOCALM_HOME:
  <HOME>/comfyui/          the managed ComfyUI checkout + its own venv (S2/S3)
  <HOME>/comfyui-models/   the localm-managed models dir (extra_model_paths.yaml
                           also points ComfyUI at the user's existing models)
"""

from __future__ import annotations

import os
import shutil
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from localm.config import home_dir, load_config

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
    """Resolved on-disk locations for the managed ComfyUI. Pure paths - reading
    this NEVER creates anything (S1 must not materialize dirs on a normal call)."""
    root: Path                # <HOME>/comfyui  (checkout + venv)
    models_dir: Path          # <HOME>/comfyui-models
    main_py: Path             # <root>/main.py         (ComfyUI entry point)
    venv_python: Path         # the managed venv interpreter (platform-specific)
    extra_model_paths: Path   # <root>/extra_model_paths.yaml


def _venv_python_path(root: Path) -> Path:
    """The managed venv's interpreter path for this platform (not required to
    exist here). Mirrors comfy_client._venv_python's layout (venv/, not .venv)."""
    if os.name == "nt":
        return root / "venv" / "Scripts" / "python.exe"
    return root / "venv" / "bin" / "python"


def managed_comfy_paths() -> ManagedComfyPaths:
    """Where a localm-managed ComfyUI lives under the CURRENT LOCALM_HOME.

    Uses the lazy ``home_dir()`` (not the import-frozen HOME_DIR) so it always
    tracks the configured data dir - and so a test pointing LOCALM_HOME at a
    throwaway dir is honoured immediately."""
    root = home_dir() / _INSTALL_DIRNAME
    return ManagedComfyPaths(
        root=root,
        models_dir=home_dir() / _MODELS_DIRNAME,
        main_py=root / "main.py",
        venv_python=_venv_python_path(root),
        extra_model_paths=root / "extra_model_paths.yaml",
    )


def rmtree_robust(path: Path) -> None:
    """``shutil.rmtree`` that also removes read-only files. A managed ComfyUI is a
    git checkout (S2), and git marks its object store read-only - on Windows plain
    rmtree then raises PermissionError (WinError 5) and the delete half-fails. Both
    ``localm comfy remove`` and S2's rollback-on-failed-provision must be able to
    remove such a tree, or a partial/old install lingers and reversibility breaks.
    Re-raises the underlying OSError when a file genuinely cannot be removed (rule 5:
    a delete that fails must NOT be reported as success by the caller)."""
    def _onerror(func, p, _exc_info):
        # Clear the read-only bit and retry; let a genuine failure propagate.
        os.chmod(p, stat.S_IWRITE)
        func(p)
    shutil.rmtree(str(path), onerror=_onerror)


def is_managed_comfy_installed() -> bool:
    """True when a usable managed ComfyUI is actually present on disk.

    S1 requires BOTH the ComfyUI entry point (``main.py``) AND the managed venv
    interpreter - i.e. code present AND runnable - so a half-copied or aborted
    install (S2/S3) does NOT read as installed and reroute media to a broken
    instance. A future stage may tighten this further (e.g. a marker file
    recording the pinned commit), but the layout check is the honest floor."""
    paths = managed_comfy_paths()
    try:
        return paths.main_py.is_file() and paths.venv_python.is_file()
    except OSError:
        return False


def managed_comfy_remove_targets(with_models: bool = False) -> list:
    """The on-disk paths ``remove`` would delete: the managed ComfyUI checkout, and
    (only with ``with_models``) the managed models folder. Pure path checks, deletes
    nothing - the CLI uses this to show what a remove WILL do before confirming, and
    the GUI/route uses it (via ``remove_managed_comfy``) for the actual delete. The
    user's own ComfyUI (comfy_workdir) is never a target."""
    paths = managed_comfy_paths()
    targets = []
    if paths.root.exists():
        targets.append(paths.root)
    if with_models and paths.models_dir.exists():
        targets.append(paths.models_dir)
    return targets


def remove_managed_comfy(with_models: bool = False) -> tuple:
    """Delete the managed ComfyUI (and its models folder with ``with_models``) under
    the localm data dir. Returns ``(removed, failed)`` - the paths deleted and, per
    rule 5, the ones that could NOT be removed with the reason, so a caller never
    reports success for a delete that failed. Both empty means nothing was installed
    (an honest no-op, not a success). The single source of truth for the removal that
    ``localm comfy remove`` and the GUI remove route share."""
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
    """The managed instance's base URL. No-op-safe: returns the fixed managed
    loopback URL without touching disk or config (callers gate on
    ``managed_comfy_active()`` / ``is_managed_comfy_installed()`` first)."""
    return MANAGED_COMFY_API_URL


def managed_comfy_workdir() -> str:
    """The working directory to launch/target the managed ComfyUI from (its
    checkout root). No-op-safe: a pure path string, creates nothing."""
    return str(managed_comfy_paths().root)


def managed_comfy_active(cfg: Optional[dict] = None) -> bool:
    """True when media calls should target the MANAGED instance (decision 6):
    the toggle is on, the target is "own", AND an instance is installed. Any of
    those false -> the user's ComfyUI, exactly as today."""
    cfg = cfg if cfg is not None else load_config()
    if not cfg.get("managed_comfy_enabled"):
        return False
    if cfg.get("comfy_target", "own") != "own":
        return False
    return is_managed_comfy_installed()


def managed_comfy_api_url_if_active(cfg: Optional[dict] = None) -> Optional[str]:
    """The managed api_url when managed routing is active, else None. This is the
    hook ``comfy_client.default_api_url()`` consults so that EVERY existing caller
    of default_api_url() targets the managed instance when active, without the
    launch/spawn path having to change (S1 confines its comfy_client.py touch to
    the target-resolution path)."""
    return managed_comfy_api_url() if managed_comfy_active(cfg) else None


@dataclass(frozen=True)
class ComfyTarget:
    """Which ComfyUI localm targets: its URL, working dir, and whether it is the
    managed instance (True) or the user's own (False)."""
    api_url: str
    workdir: Optional[str]
    managed: bool


def resolve_comfy_target(cfg: Optional[dict] = None) -> ComfyTarget:
    """THE single coexistence resolver (decision 6). Returns the managed
    instance's (api_url, workdir) when managed routing is active, else the user's
    ComfyUI exactly as today: the same api_url ``default_api_url()`` yields and
    the ``comfy_workdir`` config value.

    Media modules can call this to get BOTH the URL and the workdir at once; the
    URL alone also flows automatically through ``default_api_url()`` (which
    consults ``managed_comfy_api_url_if_active``)."""
    cfg = cfg if cfg is not None else load_config()
    if managed_comfy_active(cfg):
        return ComfyTarget(api_url=managed_comfy_api_url(),
                           workdir=managed_comfy_workdir(), managed=True)
    # User's ComfyUI, untouched. Import here (not at module load) to avoid a
    # circular import: comfy_client imports this module lazily too.
    from localm.media.comfy_client import default_api_url
    return ComfyTarget(api_url=default_api_url(),
                       workdir=cfg.get("comfy_workdir"), managed=False)


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
    """The extra_model_paths structure as a plain dict: one entry per model
    source, each ``{base_path, <type>: <subdir>, ...}``.

    Always includes the localm-managed models dir. When ``comfy_workdir`` is
    known, also includes the user's existing ComfyUI models dir
    (``<comfy_workdir>/models``), so the managed ComfyUI sees the user's models
    with zero copying. Structured so the user can hand-add more entries."""
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
    """Single-quote a YAML scalar (backslash is LITERAL inside single quotes, so
    Windows paths survive; an embedded single quote is doubled per the spec)."""
    return "'" + s.replace("'", "''") + "'"


def render_extra_model_paths(cfg: Optional[dict] = None) -> str:
    """Render the extra_model_paths.yaml TEXT from build_extra_model_paths_config.
    Hand-rolled (no PyYAML): two levels of mapping, all string scalars quoted."""
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
    """Write extra_model_paths.yaml into the managed ComfyUI dir and return its
    path. The install dir must already exist (S2/S3 create it); this writer only
    drops the file in. Overwrites any prior generated file."""
    paths = managed_comfy_paths()
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.extra_model_paths.write_text(render_extra_model_paths(cfg), encoding="utf-8")
    return paths.extra_model_paths
