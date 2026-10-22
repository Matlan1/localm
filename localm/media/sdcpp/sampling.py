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

SUGGESTED_HIRES_UPSCALER = "lanczos"
SUGGESTED_HIRES_DENOISE = 0.35
SUGGESTED_LATENT_HIRES_DENOISE = 0.55
SUGGESTED_HIRES_SCALE_SMALL = 2.0
SUGGESTED_HIRES_SCALE_LARGE = 1.5
SMALL_MODEL_MAX_SIDE = 640


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


def is_latent_upscaler(upscaler: Optional[str]) -> bool:
    """Whether hires upscaler id *upscaler* scales the latent rather than pixels."""
    return (upscaler or "").startswith("latent")


def suggested_hires(native_side: Optional[int]) -> dict:
    """The hires pass suggested for a model whose native image size has the
    longer side *native_side* (None when unknown): ``{"upscaler", "scale",
    "steps", "denoise", "latent_denoise"}``. The upscaler is
    ``SUGGESTED_HIRES_UPSCALER`` with ``SUGGESTED_HIRES_DENOISE``; *scale* is
    ``SUGGESTED_HIRES_SCALE_SMALL`` up to ``SMALL_MODEL_MAX_SIDE`` (and for an
    unknown size), ``SUGGESTED_HIRES_SCALE_LARGE`` above; *steps* 0 keeps the
    job's steps; *latent_denoise* is the denoise used when a latent upscaler
    is chosen without one."""
    small = native_side is None or int(native_side) <= SMALL_MODEL_MAX_SIDE
    return {"upscaler": SUGGESTED_HIRES_UPSCALER,
            "scale": SUGGESTED_HIRES_SCALE_SMALL if small else SUGGESTED_HIRES_SCALE_LARGE,
            "steps": 0, "denoise": SUGGESTED_HIRES_DENOISE,
            "latent_denoise": SUGGESTED_LATENT_HIRES_DENOISE}


def _fitting_scale(scale: float, width: int, height: int) -> float:
    """*scale* lowered in steps of 0.01 until the hires target of *width* x
    *height* fits ``MAX_HIRES_SIDE``, never below ``MIN_HIRES_SCALE``."""
    scale = min(float(scale), MAX_HIRES_SCALE)
    while scale > MIN_HIRES_SCALE and check_hires_target(width, height, scale):
        scale = max(MIN_HIRES_SCALE, round(scale - 0.01, 2))
    return scale


def fill_hires(hires: dict, suggestion: Optional[dict], width: int, height: int) -> dict:
    """*hires* (``native_models.validate_hires``, None for a field the request
    left out) with every unset field taken from *suggestion*
    (``suggested_hires``; None uses ``suggested_hires(None)``). An unset
    denoise is the suggestion's ``latent_denoise`` when the upscaler is a
    latent one. A suggested scale is lowered so the target of *width* x
    *height* fits ``MAX_HIRES_SIDE``; a requested scale is kept as given.
    Returns a new ``{"upscaler", "model", "scale", "steps", "denoise"}``."""
    sug = suggestion or suggested_hires(None)
    upscaler = hires.get("upscaler") or sug["upscaler"]
    scale = hires.get("scale")
    if scale is None:
        scale = _fitting_scale(sug["scale"], width, height)
    steps = hires.get("steps")
    denoise = hires.get("denoise")
    if denoise is None:
        denoise = sug["latent_denoise"] if is_latent_upscaler(upscaler) else sug["denoise"]
    return {"upscaler": upscaler, "model": hires.get("model"), "scale": float(scale),
            "steps": int(sug["steps"] if steps is None else steps),
            "denoise": float(denoise)}
