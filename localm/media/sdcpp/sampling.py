# SPDX-License-Identifier: AGPL-3.0-or-later
"""Sampler, scheduler and hires-upscaler names accepted by the native media
backends, and the validation of a request's choice among them.

The sampler and scheduler names are the ones the library at the commit pinned
in ``pins.py`` reports through ``sd_sample_method_name`` and
``sd_scheduler_name`` (their order is the order of the library's enums, which
``_binding._SAMPLE_METHOD_COUNT`` and ``_SCHEDULER_COUNT`` count). The hires
upscalers are addressed by a stable id; ``HIRES_UPSCALER_LIBRARY_NAMES`` maps
it to ``sd_hires_upscaler_name``.
"""

from __future__ import annotations

from typing import Optional

SAMPLE_METHODS: tuple[str, ...] = (
    "euler", "euler_a", "heun", "dpm2", "dpm++2s_a", "dpm++2m", "dpm++2mv2", "ipndm",
    "ipndm_v", "lcm", "ddim_trailing", "tcd", "res_multistep", "res_2s", "er_sde",
    "euler_cfg_pp", "euler_a_cfg_pp", "euler_ge", "dpm++2m_sde", "dpm++2m_sde_bt", "lms",
)

SCHEDULERS: tuple[str, ...] = (
    "discrete", "karras", "exponential", "ays", "gits", "sgm_uniform", "simple",
    "smoothstep", "kl_optimal", "lcm", "bong_tangent", "ltx2", "logit_normal", "flux2",
    "flux", "beta", "llada_image",
)

HIRES_UPSCALER_LIBRARY_NAMES: dict[str, str] = {
    "latent": "Latent",
    "latent-nearest": "Latent (nearest)",
    "latent-nearest-exact": "Latent (nearest-exact)",
    "latent-antialiased": "Latent (antialiased)",
    "latent-bicubic": "Latent (bicubic)",
    "latent-bicubic-antialiased": "Latent (bicubic antialiased)",
    "lanczos": "Lanczos",
    "nearest": "Nearest",
    "model": "Model",
}

HIRES_MODEL_UPSCALER = "model"
MIN_HIRES_SCALE = 1.0
MAX_HIRES_SCALE = 4.0
MAX_HIRES_STEPS = 150
MAX_HIRES_SIDE = 4096
MIN_HIRES_DENOISE = 0.05


def check_sample_method(name: Optional[str]) -> Optional[str]:
    """*name* when it is a known sampler, None for an empty value. Raises
    ``ValueError`` naming the accepted samplers otherwise."""
    name = (name or "").strip()
    if not name:
        return None
    if name not in SAMPLE_METHODS:
        raise ValueError(f"Unknown sampler '{name}'. Allowed: {', '.join(SAMPLE_METHODS)}.")
    return name


def check_scheduler(name: Optional[str]) -> Optional[str]:
    """*name* when it is a known scheduler, None for an empty value. Raises
    ``ValueError`` naming the accepted schedulers otherwise."""
    name = (name or "").strip()
    if not name:
        return None
    if name not in SCHEDULERS:
        raise ValueError(f"Unknown scheduler '{name}'. Allowed: {', '.join(SCHEDULERS)}.")
    return name


def hires_side(side: int, scale: float) -> int:
    """The hires target for one *side* at *scale*, rounded up to a multiple of 16,
    the widest alignment the library applies to a hires target."""
    return -(-int(round(side * scale)) // 16) * 16


def check_hires_target(width: int, height: int, scale: float) -> Optional[str]:
    """Why a hires pass of *scale* over *width* x *height* is too large, or None."""
    w, h = hires_side(width, scale), hires_side(height, scale)
    if max(w, h) > MAX_HIRES_SIDE:
        return (f"Hires fix would make a {w}x{h} image; each side is limited to "
                f"{MAX_HIRES_SIDE} pixels. Lower the hires scale or the size.")
    return None
