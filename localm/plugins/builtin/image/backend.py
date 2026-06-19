# SPDX-License-Identifier: AGPL-3.0-or-later
"""ComfyUI backend for the image plugin.

Thin wrapper over the shared Comfy HTTP plumbing (``localm.image_gen.comfy``)
that feeds it THIS plugin's per-plugin config (resolved through
``media_config``, honouring the "use config from" share-config selector). The
generic transport stays shared; only the config binding lives here, so a future
non-ComfyUI image backend is just another module selected by ``backend`` name.

Legacy global keys (comfy_launch_cmd / comfy_workdir / comfy_output_dir /
reload_llm_after_imagine) seed the defaults until the user saves per-plugin
values, so existing setups keep working with no migration step.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from localm.image_gen import comfy as _comfy
from localm.plugins import media_config
from localm.vram import media_estimate_bytes, resolve_swap_policy


def settings(full_config: dict) -> dict:
    """Resolve the image plugin's effective backend settings."""
    block, warning = media_config.resolve_config("image", full_config)
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
        "vram_estimate_bytes": media_estimate_bytes("image", block),
        "warning": warning,
    }


def ensure_available(s: dict, on_progress=None) -> tuple[bool, str]:
    return _comfy.ensure_comfy(
        s["api_url"], on_progress=on_progress,
        launch_cmd=s["launch_cmd"] or None, workdir=s["workdir"] or None)


def free_vram(s: dict) -> bool:
    return _comfy.free_comfy_vram(s["api_url"])


def generate(s: dict, prompt: str, out_path: Path, *,
             self_url: str, write_sidecar: bool,
             guidance: Optional[float] = None,
             negative_prompt: Optional[str] = None,
             seed: Optional[int] = None,
             input_image: Optional[Path] = None,
             denoise: Optional[float] = None,
             swap: bool = True) -> tuple[bool, str]:
    return _comfy.generate_image(
        prompt, out_path,
        api_url=s["api_url"],
        guidance=guidance,
        negative_prompt=negative_prompt,
        seed=seed,
        input_image=input_image,
        denoise=denoise,
        localm_url=self_url,
        write_sidecar=write_sidecar,
        launch_cmd=s["launch_cmd"] or None,
        workdir=s["workdir"] or None,
        comfy_output_dir=s["output_dir"] or None,
        swap=swap,
    )
