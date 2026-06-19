# SPDX-License-Identifier: AGPL-3.0-or-later
"""ComfyUI (Wan) backend for the video plugin.

Mirrors the image/music plugin backends: a thin wrapper over the shared Comfy
HTTP plumbing fed with THIS plugin's per-plugin config (resolved through
``media_config``, honouring the "use config from" share-config selector). Only
the config binding lives here, so a future non-ComfyUI video backend is just
another module selected by ``backend`` name.

Legacy global keys (comfy_launch_cmd / comfy_workdir / comfy_output_dir /
reload_llm_after_imagine) seed the defaults until the user saves per-plugin
values.

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
from typing import Optional

from localm import video_gen as _video_gen
from localm.image_gen import comfy as _comfy
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
    return {
        "backend": block.get("backend", "comfy"),
        "api_url": (comfy_blk.get("api_url") or _comfy.default_api_url()).rstrip("/"),
        "launch_cmd": comfy_blk.get("launch_cmd")
        or full_config.get("comfy_launch_cmd", "") or "",
        "workdir": comfy_blk.get("workdir")
        or full_config.get("comfy_workdir", "") or "",
        "output_dir": comfy_blk.get("output_dir")
        or full_config.get("comfy_output_dir", "") or "",
        "reload_after": bool(block.get(
            "reload_llm_after_generate",
            full_config.get("reload_llm_after_imagine", True))),
        "swap_policy": resolve_swap_policy(block, full_config),
        "vram_estimate_bytes": media_estimate_bytes("video", block),
        "warning": warning,
    }


def ensure_available(s: dict, on_progress=None) -> tuple[bool, str]:
    return _comfy.ensure_comfy(
        s["api_url"], on_progress=on_progress,
        launch_cmd=s["launch_cmd"] or None, workdir=s["workdir"] or None)


def free_vram(s: dict) -> bool:
    return _comfy.free_comfy_vram(s["api_url"])


def generate(s: dict, prompt: str, out_path: Path, *,
             self_url: str, write_sidecar: bool, on_progress=None,
             input_image: Optional[Path] = None,
             swap: bool = True,
             **kwargs) -> tuple[bool, str]:
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
            swap=swap,
            **kwargs,
        )
