# SPDX-License-Identifier: AGPL-3.0-or-later
"""ComfyUI (Wan) backend for the video plugin.

Mirrors the image/music plugin backends: a thin wrapper over the shared Comfy
HTTP plumbing fed with THIS plugin's per-plugin config (resolved through
``media_config``, honouring the "use config from" share-config selector). Only
the config binding lives here, so a future non-ComfyUI video backend is just
another module selected by ``backend`` name.

Legacy global keys (comfy_launch_cmd / comfy_workdir / comfy_output_dir /
reload_llm_after_imagine) seed the defaults until the user saves per-plugin
values - EXCEPT comfy_launch_cmd/comfy_workdir specifically, which are
suppressed while the managed ComfyUI instance is active (see
managed_comfy.legacy_comfy_value): otherwise a global value left over from
before the managed instance existed would silently defeat its auto-launch
routing forever. A genuine per-plugin override is unaffected and still
always wins.

Per-plugin output containment (FAC-3): the shared ``generate_video`` has no
``comfy_output_dir`` parameter, so the only way to feed it this plugin's own
output dir is the ``COMFY_OUTPUT_DIR`` env var that ``comfy._comfy_output_root``
resolves from. The backend therefore publishes the per-plugin value on that env
var for the duration of the generation (restoring whatever was there before), so
ComfyUI's on-disk copy AND any uploaded image-to-video source actually get
deleted rather than the knob being silently ignored.
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

from localm import video_gen as _video_gen
from localm.image_gen import comfy as _comfy
from localm.media.managed_comfy import legacy_comfy_value
from localm.plugins import media_config
from localm.vram import media_estimate_bytes, resolve_swap_policy


@contextlib.contextmanager
def _comfy_output_dir_env(output_dir: Optional[str]):
    """Publish the per-plugin ComfyUI output dir on ``COMFY_OUTPUT_DIR`` for the
    duration of the block, restoring the prior value afterwards.

    A no-op when no per-plugin output dir is configured, so a value inherited
    from the environment or global config keeps working untouched."""
    if not output_dir:
        yield
        return
    prev = os.environ.get("COMFY_OUTPUT_DIR")
    os.environ["COMFY_OUTPUT_DIR"] = output_dir
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop("COMFY_OUTPUT_DIR", None)
        else:
            os.environ["COMFY_OUTPUT_DIR"] = prev


def settings(full_config: dict) -> dict:
    """Resolve the video plugin's effective backend settings."""
    block, warning = media_config.resolve_config("video", full_config)
    comfy_blk = block.get("comfy") if isinstance(block.get("comfy"), dict) else {}
    backend_name = block.get("backend", "comfy")
    # We do not hide problems: when the configured backend cannot be loaded the
    # job still falls back to comfy (best-effort), but say so instead of silently
    # pretending the chosen backend is active.
    warning = media_config.combine_warnings(
        warning, media_config.backend_unavailable_warning(__package__, backend_name))
    return {
        "backend": backend_name,
        # Sanitise the RESOLVED url: a per-plugin comfy.api_url short-circuits
        # before default_api_url()'s guard, so an admin-set link-local/metadata
        # host would otherwise reach the outbound comfy calls (CHK-COMFY-APIURL).
        "api_url": _comfy.sanitize_comfy_url(
            (comfy_blk.get("api_url") or _comfy.default_api_url()).rstrip("/")),
        # The legacy global fallback is suppressed while the managed instance is
        # active - a genuine per-plugin comfy_blk override still always wins.
        "launch_cmd": comfy_blk.get("launch_cmd")
        or legacy_comfy_value("comfy_launch_cmd", full_config) or "",
        "workdir": comfy_blk.get("workdir")
        or legacy_comfy_value("comfy_workdir", full_config) or "",
        "output_dir": comfy_blk.get("output_dir")
        or full_config.get("comfy_output_dir", "") or "",
        # Opt-in output containment: per-plugin delete_outputs wins, else the
        # global comfy_delete_outputs (default False = keep ComfyUI's own copy).
        # Privacy mode forces deletion later in plug.py regardless of this.
        "delete_outputs": bool(comfy_blk.get(
            "delete_outputs", full_config.get("comfy_delete_outputs", False))),
        "float_type": comfy_blk.get("float_type")
        or full_config.get("comfy_float_type"),
        "reload_after": bool(block.get(
            "reload_llm_after_generate",
            full_config.get("reload_llm_after_imagine", True))),
        "swap_policy": resolve_swap_policy(block, full_config),
        "vram_estimate_bytes": media_estimate_bytes("video", block),
        "warning": warning,
    }


# --- ComfyUI (Wan) reference implementation (the default "comfy" backend) ----

def _comfy_ensure_available(s: dict, on_progress=None) -> tuple[bool, str]:
    return _comfy.ensure_comfy(
        s["api_url"], on_progress=on_progress,
        launch_cmd=s["launch_cmd"] or None, workdir=s["workdir"] or None)


def _comfy_free_vram(s: dict) -> bool:
    return _comfy.free_comfy_vram(s["api_url"])


def _comfy_generate(s: dict, prompt: str, out_path: Path, *,
                    self_url: str, write_sidecar: bool, on_progress=None,
                    input_image: Optional[Path] = None,
                    swap: bool = True,
                    delete_outputs: Optional[bool] = None,
                    cancel_check=None,
                    **kwargs) -> tuple[bool, str]:
    # delete_outputs: None means "use the configured preference"; the privacy-
    # aware caller (plug.py) passes an explicit bool that already folds in
    # privacy mode, so honour that when given and fall back to the setting only
    # when it is not.
    if delete_outputs is None:
        delete_outputs = bool(s.get("delete_outputs", False))
    # generate_video has no comfy_output_dir param; feed the per-plugin value
    # through the env var its containment step resolves from (FAC-3). This is
    # also what lets the uploaded image-to-video source be cleaned up, since the
    # input/ dir is located relative to the resolved output dir.
    with _comfy_output_dir_env(s.get("output_dir") or None):
        return _video_gen.generate_video(
            prompt, out_path,
            input_image=input_image,
            api_url=s["api_url"],
            localm_url=self_url,
            on_progress=on_progress,
            write_sidecar=write_sidecar,
            launch_cmd=s["launch_cmd"] or None,
            workdir=s["workdir"] or None,
            float_type=s.get("float_type"),
            swap=swap,
            delete_outputs=delete_outputs,
            cancel_check=cancel_check,
            **kwargs,
        )


_COMFY_REF = SimpleNamespace(
    ensure_available=_comfy_ensure_available,
    free_vram=_comfy_free_vram,
    generate=_comfy_generate,
)


# --- backend facade: dispatch to the configured backend (the I1 seam) --------
# Shared with the image/music plugins - see media_config.make_backend_facade.

_facade = media_config.make_backend_facade(__package__, _COMFY_REF)
_impl = _facade.resolve
ensure_available = _facade.ensure_available
free_vram = _facade.free_vram
generate = _facade.generate
