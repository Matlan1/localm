# SPDX-License-Identifier: AGPL-3.0-or-later
"""
localm-managed ComfyUI: paths, install detection, coexistence routing.

STAGE S1 (scaffolding). This module knows WHERE a localm-owned ComfyUI would
live (under LOCALM_HOME), how to tell whether one is actually installed, and
which ComfyUI localm should target when it is. It provisions nothing: copying
the user's stack is S2, the fresh hardware-matched install is S3. Everything
here is inert until a managed instance exists on disk.

The single source of truth for "which ComfyUI does localm talk to" is
``resolve_comfy_target()``. The rule:

    target the MANAGED instance  IFF  comfy_target == "own"
                                  AND  a managed instance is actually installed
    otherwise                         the user's ComfyUI, exactly as today.

When nothing is set up, the user's ComfyUI is used, untouched. The user's
ComfyUI is NEVER modified.

Layout under LOCALM_HOME:
  <HOME>/comfyui/          the managed ComfyUI checkout + its own venv (S2/S3)
  <HOME>/comfyui-models/   the localm-managed models dir (extra_model_paths.yaml
                           also points ComfyUI at the user's existing models)
"""

from __future__ import annotations

import os
import shutil
import stat
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from localm.config import home_dir, load_config

# Serializes remove_managed_comfy() against itself. The CLI (`localm comfy
# remove`) and the GUI's /api/comfy/remove + /api/comfy/repair routes are two
# independent callers of the SAME rmtree target, and the GUI routes wrap their
# call in run_in_threadpool_bounded, whose deadline only makes the AWAITING
# request give up while the real rmtree keeps running on its abandoned worker
# thread. Without this lock a retry after a timeout starts a SECOND concurrent
# rmtree against the same tree. One lock is enough: there is only ever ONE
# managed ComfyUI install, not one per media type.
_remove_lock = threading.Lock()

# Directory names under LOCALM_HOME. Kept as constants so S2/S3 and the CLI all
# agree on the one layout.
_INSTALL_DIRNAME = "comfyui"
_MODELS_DIRNAME = "comfyui-models"

# The managed instance runs on its OWN loopback port, distinct from the ComfyUI
# default (8188), so localm's managed ComfyUI and a user's own ComfyUI can run at
# the same time. localm claims 8642-8741 for its own server; the managed ComfyUI
# sits just above the ComfyUI default, on 8189. S2/S3 launch it here.
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

    Uses the lazy ``home_dir()``, not the import-frozen HOME_DIR, so it always
    tracks the configured data dir."""
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
    remove such a tree. Re-raises the underlying OSError when a file genuinely
    cannot be removed, so a failed delete is never reported as success."""
    def _onerror(func, p, _exc_info):
        # Clear the read-only bit and retry; let a genuine failure propagate.
        os.chmod(p, stat.S_IWRITE)
        func(p)
    shutil.rmtree(str(path), onerror=_onerror)


def is_managed_comfy_installed() -> bool:
    """True when a usable managed ComfyUI is actually present on disk AND the
    provisioning pipeline (S2/S3) actually finished.

    Requires the completion marker (S2/S3's ``MARKER_FILENAME``, written as the
    SECOND-TO-LAST step of provision_fresh()/provision_by_copy(), right before
    their own final self-check) IN ADDITION TO the ComfyUI entry point
    (``main.py``) and the managed venv interpreter existing.

    main.py and venv_python both exist after step 2 of a 7-8 step pipeline (the
    git clone, then `python -m venv`), so on their own they report an instance
    ready while torch, ComfyUI's own requirements, custom nodes and localm's
    patches are still installing - and `localm comfy status`, the Settings page
    pill and Generate-button routing all read this function. The marker is
    written only once every earlier step has succeeded."""
    paths = managed_comfy_paths()
    try:
        if not (paths.main_py.is_file() and paths.venv_python.is_file()):
            return False
        from localm.media.managed_comfy_provision import MARKER_FILENAME
        return (paths.root / MARKER_FILENAME).is_file()
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
    the localm data dir. Returns ``(removed, failed)`` - the paths deleted and the
    ones that could NOT be removed with the reason, so a caller never reports
    success for a delete that failed. Both empty means nothing was installed (an
    honest no-op, not a success). The single source of truth for the removal that
    ``localm comfy remove`` and the GUI remove route share.

    Holds ``_remove_lock`` for the whole delete so two concurrent callers (the
    CLI and a GUI retry, or two GUI clicks) can never rmtree the same tree at
    once."""
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
    """The managed instance's base URL. No-op-safe: returns the fixed managed
    loopback URL without touching disk or config (callers gate on
    ``managed_comfy_active()`` / ``is_managed_comfy_installed()`` first)."""
    return MANAGED_COMFY_API_URL


def managed_comfy_workdir() -> str:
    """The working directory to launch/target the managed ComfyUI from (its
    checkout root). No-op-safe: a pure path string, creates nothing."""
    return str(managed_comfy_paths().root)


def managed_comfy_launch_cmd(config: Optional[dict] = None) -> str:
    """The command that starts the managed instance: its OWN venv interpreter
    running its OWN ``main.py``, on the managed port - never the user's
    ``comfy_launch_cmd``/auto-discovered launcher script, which only make sense
    for a user-provided ComfyUI. A fresh/copied managed install is a raw
    checkout with no bundled .bat/.sh launcher of its own. Quoted the same way a
    user's own launch command is expected to be (see ensure_comfy's shlex/cmd
    handling).

    Names a PREFERRED GPU via ``--default-device`` when the user configured a
    split or a main GPU, so the swap gate (which reads COMBINED free VRAM across
    the split) and the actual loader agree on which card the work lands on.
    Without it a media job that "fit" in 2x4 GB combined would OOM on one 4 GB
    card.

    ``--default-device``, NOT ``--cuda-device``. Both flags express "use this
    card", but ``--cuda-device`` sets ``CUDA_VISIBLE_DEVICES`` to that one id
    (``main.py:78-81``), DELETING the other cards from torch's view, which
    disables ComfyUI core's own per-component placement nodes
    (``SelectModelDevice``/``SelectCLIPDevice``/``SelectVAEDevice``,
    ``comfy_extras/nodes_multigpu.py``, registered ``nodes.py:2440``) because a
    ``gpu:1`` that no longer exists is a no-op. ``--default-device`` REORDERS the
    visible list so the chosen card leads and the rest stay usable
    (``main.py:69-76``).

    Creates nothing on disk, but is not pure: resolving the device reads config
    and probes the GPU via ``discover.list_gpus()``. That probe runs the driver
    query on a helper thread but BLOCKS THIS CALLING THREAD up to the probe
    deadline (15s default) - fine from the media job worker threads that call it,
    but it must NOT be called inline on the server's event loop.
    """
    paths = managed_comfy_paths()
    cmd = (f'"{paths.venv_python}" "{paths.main_py}" '
           f'--listen 127.0.0.1 --port {MANAGED_COMFY_PORT}')
    from localm.discover import resolve_preferred_device
    device = resolve_preferred_device(config)
    if device is not None:
        cmd += f" --default-device {device}"
    return cmd


def managed_comfy_active(cfg: Optional[dict] = None) -> bool:
    """True when media calls should target the MANAGED instance: the target is
    "own" AND an instance is installed. Either false -> the user's ComfyUI,
    exactly as today."""
    cfg = cfg if cfg is not None else load_config()
    if cfg.get("comfy_target", "own") != "own":
        return False
    return is_managed_comfy_installed()


def legacy_comfy_value(key: str, full_config: dict) -> str:
    """A LEGACY GLOBAL comfy_* config value (``comfy_workdir``, ``comfy_launch_cmd``),
    suppressed to "" whenever the managed instance is active.

    Every media backend.py (image/music/video) falls back to these global keys
    to seed defaults for setups that pre-date per-plugin config, and
    ``comfy_client.ensure_comfy()``'s "an explicit caller workdir/launch_cmd
    always wins over managed routing" precedence cannot tell a deliberate
    override from a years-old global default. Honouring the legacy value here
    would pass it through as if it WERE a deliberate override, defeating
    managed-instance auto-routing for anyone who ever set
    comfy_workdir/comfy_launch_cmd.

    A per-plugin override (the caller's own ``comfy_blk.get(...)``) carries the
    identical ambiguity, so each backend.py suppresses its OWN per-plugin
    comfy_blk value too, gated on the same ``managed_comfy_active()`` check this
    function uses. "own" means own for the per-plugin field as well."""
    if managed_comfy_active(full_config):
        return ""
    return full_config.get(key, "") or ""


def managed_comfy_api_url_if_active(cfg: Optional[dict] = None) -> Optional[str]:
    """The managed api_url when managed routing is active, else None. This is the
    hook ``comfy_client.default_api_url()`` consults so that EVERY existing caller
    of default_api_url() targets the managed instance when active, without the
    launch/spawn path having to change."""
    return managed_comfy_api_url() if managed_comfy_active(cfg) else None


@dataclass(frozen=True)
class ComfyTarget:
    """Which ComfyUI localm targets: its URL, working dir, launch command (None
    when the caller should fall back to its own discovery, e.g. the user's own
    install), and whether it is the managed instance (True) or the user's own
    (False)."""
    api_url: str
    workdir: Optional[str]
    launch_cmd: Optional[str]
    managed: bool


def resolve_comfy_target(cfg: Optional[dict] = None,
                         plugin: Optional[str] = None) -> ComfyTarget:
    """THE single coexistence resolver. Returns the managed instance's (api_url,
    workdir, launch_cmd) when managed routing is active, else the user's ComfyUI
    exactly as today: the same api_url ``default_api_url()`` yields, the
    ``comfy_workdir`` config value, and no launch_cmd (the caller resolves or
    discovers its own).

    *plugin* ("image"/"music"/"video"), when given, resolves THAT plugin's own
    ``comfy.workdir`` (via ``media_config.resolve_config``, honouring
    "use config from") ahead of the legacy global ``comfy_workdir`` for the
    non-managed branch, which is what the plugin's own launch path uses. Omit it
    when there is no plugin context (e.g. the CLI's generic status command); that
    caller still gets a correct answer whenever the global key or the managed
    instance covers it.

    Media modules can call this to get the URL, workdir, AND launch command at
    once; the URL alone also flows automatically through ``default_api_url()``
    (which consults ``managed_comfy_api_url_if_active``)."""
    cfg = cfg if cfg is not None else load_config()
    if managed_comfy_active(cfg):
        return ComfyTarget(api_url=managed_comfy_api_url(),
                           workdir=managed_comfy_workdir(),
                           launch_cmd=managed_comfy_launch_cmd(), managed=True)
    # User's ComfyUI. Imported here, not at module load: comfy_client imports
    # this module lazily too, so a top-level import would be circular.
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
    """Absolute ``models/<subfolder>`` directory for whichever ComfyUI
    ``resolve_comfy_target()`` says is currently active - the destination a
    downloaded model file needs to land in for THAT ComfyUI to see it.

    Managed -> ``<LOCALM_HOME>/comfyui-models/<subfolder>``. External with a
    configured workdir (per-plugin when *plugin* is given, else the legacy
    global ``comfy_workdir``) -> ``<workdir>/models/<subfolder>``. External
    with no workdir configured anywhere -> None: there is no known-safe
    filesystem location to write into, and the caller must say so plainly
    rather than guessing."""
    target = resolve_comfy_target(cfg, plugin=plugin)
    if target.managed:
        return managed_comfy_paths().models_dir / subfolder
    if target.workdir:
        return Path(target.workdir) / "models" / subfolder
    return None


# --------------------------------------------------------------------------- #
#  extra_model_paths.yaml generator                                           #
#                                                                             #
#  ComfyUI-native model sharing: point the managed ComfyUI at the user's      #
#  existing models dir (no copy, no symlink, no re-download) AND the           #
#  localm-managed models dir. This just WRITES the file; S2/S3 invoke it as    #
#  part of an install. Written by hand, with no PyYAML dependency in shipped   #
#  code.                                                                       #
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
