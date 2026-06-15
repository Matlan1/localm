"""ComfyUI (Wan) backend for the video plugin.

Mirrors the image/music plugin backends: a thin wrapper over the shared Comfy
HTTP plumbing fed with THIS plugin's per-plugin config (resolved through
``media_config``, honouring the "use config from" share-config selector). Only
the config binding lives here, so a future non-ComfyUI video backend is just
another module selected by ``backend`` name.

Legacy global keys (comfy_launch_cmd / comfy_workdir / reload_llm_after_imagine)
seed the defaults until the user saves per-plugin values. (Video never wrote to a
comfy output dir, so it has no output_dir setting.)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from localm import video_gen as _video_gen
from localm.image_gen import comfy as _comfy
from localm.plugins import media_config


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
        "reload_after": bool(block.get(
            "reload_llm_after_generate",
            full_config.get("reload_llm_after_imagine", True))),
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
             **kwargs) -> tuple[bool, str]:
    return _video_gen.generate_video(
        prompt, out_path,
        input_image=input_image,
        api_url=s["api_url"],
        localm_url=self_url,
        on_progress=on_progress,
        write_sidecar=write_sidecar,
        launch_cmd=s["launch_cmd"] or None,
        workdir=s["workdir"] or None,
        **kwargs,
    )
