# SPDX-License-Identifier: AGPL-3.0-or-later
"""Image-generation tool: a thin wrapper that delegates to
localm.image_gen.comfy.generate_image."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .base import ToolResult, _confine

def tool_generate_image(
    cwd: Path,
    prompt: str,
    output_path: str = "output.png",
    guidance: Optional[float] = None,
    negative_prompt: Optional[str] = None,
    seed: Optional[int] = None,
    lora_name: Optional[str] = None,
    lora_strength_model: float = 1.0,
    lora_strength_clip: float = 0.5,
    input_image: Optional[str] = None,
    denoise: Optional[float] = None,
    _privacy: bool = False,
) -> ToolResult:
    """Thin wrapper - delegates to localm.image_gen.comfy.generate_image.

    In privacy mode (``_privacy=True``, injected by the agent) the prompt
    sidecar is suppressed so no prompt trace is written to disk."""
    import os
    from localm.image_gen.comfy import generate_image

    try:
        out_p = _confine(cwd, output_path)
        input_p = _confine(cwd, input_image) if input_image else None
    except PermissionError as e:
        return ToolResult.error(str(e))
    api_url = os.environ.get("FLUX_API_URL", "http://127.0.0.1:8188")
    ok, message = generate_image(
        prompt, out_p,
        api_url=api_url,
        guidance=guidance,
        negative_prompt=negative_prompt,
        seed=seed,
        lora_name=lora_name,
        lora_strength_model=lora_strength_model,
        lora_strength_clip=lora_strength_clip,
        input_image=input_p,
        denoise=denoise,
        write_sidecar=not _privacy,
    )
    if ok:
        rel = out_p.relative_to(cwd) if out_p.is_relative_to(cwd) else out_p
        return ToolResult.success(message, summary=f"generated {rel}")
    return ToolResult.error(message)
