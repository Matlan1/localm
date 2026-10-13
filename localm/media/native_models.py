# SPDX-License-Identifier: AGPL-3.0-or-later
"""Per-request model choices for the native image and video backends, and the
route the media pages fill their model pickers from.

A request may name the main model and any component slot by REGISTERED model
name only (never a path), plus registered LoRAs with strengths. The choice is
laid over the plugin's ``native`` settings block for that one job: an empty
value clears the setting, so the slot is picked automatically.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from localm.media.sdcpp import sampling
from localm.media.sdcpp.pairing import MEDIA_SLOTS

MAX_LORAS = 8
MAX_BATCH = 16
UPSCALER_TYPE = "upscaler"

MUSIC_KEYS = ("dit", "text_encoder", "vae", "lm")


def media_keys(media: str) -> tuple:
    """The choice keys a *media* request may set."""
    if media == "music":
        return MUSIC_KEYS
    return ("model",) + MEDIA_SLOTS[media]


class LoraChoice(BaseModel):
    name: str
    strength: float = Field(default=1.0, ge=-4.0, le=4.0, allow_inf_nan=False)


class HiresChoice(BaseModel):
    """The hires-fix pass of a native image job: the first pass is upscaled by
    *scale* with *upscaler* (a ``sampling.HIRES_UPSCALER_LIBRARY_NAMES`` id; the
    ``model`` id needs *model*, a registered upscaler) and sampled again for
    *steps* steps (0 keeps the job's steps) at *denoise* strength."""
    upscaler: str = "latent"
    model: Optional[str] = None
    scale: float = Field(default=2.0, ge=sampling.MIN_HIRES_SCALE, le=sampling.MAX_HIRES_SCALE,
                         allow_inf_nan=False)
    steps: int = Field(default=0, ge=0, le=sampling.MAX_HIRES_STEPS)
    denoise: float = Field(default=0.7, ge=sampling.MIN_HIRES_DENOISE, le=1.0,
                           allow_inf_nan=False)


def upscaler_options(reg: Optional[dict] = None) -> list:
    """``[{"name", "file"}]``: the registered upscaler models whose file exists."""
    from localm.model_manager.registry import _entry_path
    from localm.pathsafe import is_unc_or_device_path
    if reg is None:
        from localm.config import load_registry
        reg = load_registry()
    out = []
    for name, entry in sorted(reg.items()):
        if not isinstance(entry, dict) or entry.get("model_type") != UPSCALER_TYPE:
            continue
        raw = _entry_path(entry)
        if not raw or is_unc_or_device_path(raw):
            continue
        try:
            if not Path(raw).is_file():
                continue
        except OSError:
            continue
        out.append({"name": str(name), "file": raw.replace("\\", "/").rsplit("/", 1)[-1]})
    return out


def validate_upscaler(name: Optional[str], reg: Optional[dict] = None) -> str:
    """*name* when it is a registered upscaler model. Raises ``ValueError``."""
    name = (name or "").strip()
    if reg is None:
        from localm.config import load_registry
        reg = load_registry()
    entry = reg.get(name) if name else None
    if not isinstance(entry, dict) or entry.get("model_type") != UPSCALER_TYPE:
        raise ValueError(f"'{name}' is not a registered upscaler model." if name
                         else "Pick an upscaler model.")
    return name


def validate_hires(hires: Optional[HiresChoice], reg: Optional[dict] = None) -> Optional[dict]:
    """*hires* as ``{"upscaler", "model", "scale", "steps", "denoise"}``, or None
    without one. Raises ``ValueError`` with a user-facing reason."""
    if hires is None:
        return None
    upscaler = hires.upscaler.strip()
    if upscaler not in sampling.HIRES_UPSCALER_LIBRARY_NAMES:
        raise ValueError(f"Unknown hires upscaler '{upscaler}'. Allowed: "
                         + ", ".join(sampling.HIRES_UPSCALER_LIBRARY_NAMES) + ".")
    model = None
    if upscaler == sampling.HIRES_MODEL_UPSCALER:
        model = validate_upscaler(hires.model, reg)
    return {"upscaler": upscaler, "model": model, "scale": float(hires.scale),
            "steps": int(hires.steps), "denoise": float(hires.denoise)}


def check_generation_extras(sample_method: Optional[str], scheduler: Optional[str],
                            hires: Optional[HiresChoice], batch: int = 1) -> dict:
    """A request's sampler, scheduler, hires pass and batch count checked:
    ``{"sample_method", "scheduler", "hires", "batch"}`` with None for an unset
    one. A bad value is a 400."""
    try:
        if not 1 <= int(batch) <= MAX_BATCH:
            raise ValueError(f"Batch count must be between 1 and {MAX_BATCH}.")
        return {"sample_method": sampling.check_sample_method(sample_method),
                "scheduler": sampling.check_scheduler(scheduler),
                "hires": validate_hires(hires), "batch": int(batch)}
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


def sampling_options() -> dict:
    """The sampler, scheduler and hires upscaler choices the media pages offer."""
    return {"sample_methods": list(sampling.SAMPLE_METHODS),
            "schedulers": list(sampling.SCHEDULERS),
            "hires_upscalers": [{"id": k, "label": v}
                                for k, v in sampling.HIRES_UPSCALER_LIBRARY_NAMES.items()]}


def validate_choice(native_models: Optional[dict], media: str,
                    reg: Optional[dict] = None) -> dict:
    """*native_models* checked for *media*: keys ``model`` and that media's
    slots, values registered model names or ``""``. Returns the cleaned dict.
    Raises ``ValueError`` with a user-facing reason."""
    if not native_models:
        return {}
    allowed = media_keys(media)
    if reg is None:
        from localm.config import load_registry
        reg = load_registry()
    out = {}
    for key, value in native_models.items():
        if key not in allowed:
            raise ValueError(f"Unknown native {media} model slot '{key}'. Allowed: "
                             + ", ".join(allowed) + ".")
        value = str(value or "").strip()
        if value and value not in reg:
            raise ValueError(f"'{value}' is not a registered model.")
        out[key] = value
    return out


def validate_loras(loras: Optional[list], reg: Optional[dict] = None) -> list:
    """*loras* (``LoraChoice`` items) as ``[{"name", "strength"}]``; each name
    must be a registered ``lora`` entry. Raises ``ValueError``."""
    if not loras:
        return []
    if len(loras) > MAX_LORAS:
        raise ValueError(f"At most {MAX_LORAS} LoRAs per job.")
    if reg is None:
        from localm.config import load_registry
        reg = load_registry()
    out = []
    for item in loras:
        name = item.name.strip()
        entry = reg.get(name)
        if not isinstance(entry, dict) or entry.get("model_type") != "lora":
            raise ValueError(f"'{name}' is not a registered LoRA.")
        out.append({"name": name, "strength": float(item.strength)})
    return out


def apply_choice(s: dict, choice: dict) -> dict:
    """Settings *s* with *choice* laid over its ``native`` block (a copy).
    When *choice* names a different main ``model`` than the settings, the
    settings' component slots are dropped, so only the choice's own slots and
    the automatic pick apply to that model."""
    if not choice:
        return s
    out = dict(s)
    blk = dict(s.get("native") or {}) if isinstance(s.get("native"), dict) else {}
    if "model" in choice and choice["model"] != str(blk.get("model") or "").strip():
        for slot in set(MEDIA_SLOTS["image"]) | set(MEDIA_SLOTS["video"]):
            blk.pop(slot, None)
    blk.update(choice)
    out["native"] = blk
    return out


def check_request(native_models: Optional[dict], loras: Optional[list],
                  media: str) -> tuple[dict, list]:
    """Validate a request's choices; a bad one is a 400."""
    try:
        return validate_choice(native_models, media), validate_loras(loras)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


def music_options(reg: Optional[dict] = None) -> dict:
    """``{component: [{"name", "file"}]}``: the registered GGUFs whose
    recorded architecture is that ACE-Step component's."""
    from localm.media.koboldcpp.models import ARCHITECTURES
    if reg is None:
        from localm.config import load_registry
        reg = load_registry()
    out: dict = {c: [] for c in MUSIC_KEYS}
    for name, entry in sorted(reg.items()):
        if not isinstance(entry, dict):
            continue
        arch = entry.get("architecture")
        for comp, want in ARCHITECTURES.items():
            if arch == want:
                raw = str(entry.get("path") or "")
                out[comp].append({"name": name,
                                  "file": raw.replace("\\", "/").rsplit("/", 1)[-1]})
    return out


ACESTEP_V1_REASON = ("ACE-Step v1: the native music backend runs ACE-Step 1.5 GGUF parts "
                     "only, so this checkpoint works in ComfyUI only")

_ACESTEP_V1_MARKERS = (b"genre_embedder", b"lyric_embs", b"umt5base")
_MAX_HEADER_BYTES = 64 * 1024 * 1024
_V1_CACHE: dict = {}
_V1_CACHE_MAX = 512


def is_acestep_v1_checkpoint(path) -> bool:
    """True when the ``.safetensors`` file at *path* has an ACE-Step v1 tensor
    layout (the all-in-one ComfyUI checkpoint: ``genre_embedder``,
    ``lyric_embs`` and a UMT5 text encoder in one file). Reads only the
    header; False for any other file or an unreadable one. Never raises."""
    import struct
    from pathlib import Path
    p = Path(path)
    try:
        st = p.stat()
        key = (str(p), st.st_size, st.st_mtime_ns)
        if key in _V1_CACHE:
            return _V1_CACHE[key]
        with open(p, "rb") as f:
            raw = f.read(8)
            n = struct.unpack("<Q", raw)[0] if len(raw) == 8 else 0
            header = f.read(n) if 0 < n <= _MAX_HEADER_BYTES else b""
    except (OSError, struct.error):
        return False
    found = bool(header) and all(m in header for m in _ACESTEP_V1_MARKERS)
    if len(_V1_CACHE) >= _V1_CACHE_MAX:
        _V1_CACHE.clear()
    _V1_CACHE[key] = found
    return found


def unsupported_music(reg: Optional[dict] = None) -> list:
    """``[{"name", "file", "reason"}]``: registered music checkpoints the native
    music backend cannot run (today: ACE-Step v1 ``.safetensors`` files), so
    the picker can say why they are not offered."""
    from pathlib import Path

    from localm.pathsafe import is_unc_or_device_path
    if reg is None:
        from localm.config import load_registry
        reg = load_registry()
    out = []
    for name, entry in sorted(reg.items()):
        if not isinstance(entry, dict) or entry.get("model_type") == "lora":
            continue
        raw = str(entry.get("path") or "")
        if Path(raw).suffix.lower() not in (".safetensors", ".sft"):
            continue
        if is_unc_or_device_path(raw):
            continue
        if is_acestep_v1_checkpoint(raw):
            out.append({"name": name, "file": raw.replace("\\", "/").rsplit("/", 1)[-1],
                        "reason": ACESTEP_V1_REASON})
    return out


def make_music_router(plugin_backend) -> APIRouter:
    """``GET /api/music/native-models``: ``{"components", "unsupported",
    "current"}`` - the registered files for each ACE-Step component
    (``music_options``), the registered music checkpoints the native backend
    cannot run with a reason (``unsupported_music``), and the settings' current
    choice (file names only)."""
    router = APIRouter()

    @router.get("/api/music/native-models")
    async def music_native_models():
        from localm.inference._threadpool_timeout import (
            ThreadCallTimeout, run_in_threadpool_bounded,
        )

        def _read():
            from localm.config import load_config
            from localm.media.koboldcpp.models import shown_name
            s = plugin_backend.settings(load_config())
            blk = s.get("native") if isinstance(s.get("native"), dict) else {}
            return {"components": music_options(),
                    "unsupported": unsupported_music(),
                    "current": {k: shown_name(str(blk.get(k) or "")) for k in MUSIC_KEYS}}
        try:
            return await run_in_threadpool_bounded(_read, timeout=30.0)
        except ThreadCallTimeout as e:
            raise HTTPException(504, f"Reading the native models timed out: {e}") from e

    return router


def make_router(media: str, prefix: str, plugin_backend) -> APIRouter:
    """``GET /api/<prefix>/native-models``: the picker options for *media*
    (``pairing.native_options``), the settings' current choice, and a preview
    (``pairing.preview``) of what a job would load for the model and slots in
    the query string (or the settings when the query names no model)."""
    router = APIRouter()

    @router.get(f"/api/{prefix}/native-models")
    async def native_models(request: Request):
        from localm.inference._threadpool_timeout import (
            ThreadCallTimeout, run_in_threadpool_bounded,
        )
        keys = ("model",) + MEDIA_SLOTS[media]
        query = {k: v for k, v in request.query_params.items() if k in keys}
        try:
            query = validate_choice(query, media)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e

        def _read():
            from localm.config import load_config
            from localm.media.sdcpp import pairing
            from localm.media.sdcpp.files import display_name
            s = plugin_backend.settings(load_config())
            blk = s.get("native") if isinstance(s.get("native"), dict) else {}
            configured = {k: str(blk.get(k) or "").strip() for k in keys}
            cands = pairing.registry_candidates()
            other_model = "model" in query and query["model"] != configured["model"]
            picked = {k: query.get(k, "" if other_model and k != "model" else configured[k])
                      for k in keys}
            model = picked.pop("model", "")
            recommended = getattr(plugin_backend, "recommended_defaults", None)
            preview = (pairing.preview(media, model, picked, cands,
                                       recommended() if recommended else None)
                       if model else None)
            if preview and preview["defaults"]:
                for key in ("steps", "cfg_scale"):
                    if blk.get(key) is not None:
                        preview["defaults"][key] = blk[key]
            out = pairing.native_options(
                media, cands, family=preview["family"] if preview else None,
                variant=preview["variant"] if preview else None)
            out["sampling"] = sampling_options()
            if media == "image":
                out["upscalers"] = upscaler_options()
            out["current"] = {k: display_name(v) if v else "" for k, v in configured.items()}
            out["preview"] = preview
            return out
        try:
            return await run_in_threadpool_bounded(_read, timeout=30.0)
        except ThreadCallTimeout as e:
            raise HTTPException(504, f"Reading the native models timed out: {e}") from e

    return router
