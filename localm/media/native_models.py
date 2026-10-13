# SPDX-License-Identifier: AGPL-3.0-or-later
"""Per-request model choices for the native image and video backends, and the
route the media pages fill their model pickers from.

A request may name the main model and any component slot by REGISTERED model
name only (never a path), plus registered LoRAs with strengths. The choice is
laid over the plugin's ``native`` settings block for that one job: an empty
value clears the setting, so the slot is picked automatically.
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from localm.media.sdcpp.pairing import MEDIA_SLOTS

MAX_LORAS = 8

MUSIC_KEYS = ("dit", "text_encoder", "vae", "lm")


def media_keys(media: str) -> tuple:
    """The choice keys a *media* request may set."""
    if media == "music":
        return MUSIC_KEYS
    return ("model",) + MEDIA_SLOTS[media]


class LoraChoice(BaseModel):
    name: str
    strength: float = Field(1.0, ge=-4.0, le=4.0, allow_inf_nan=False)


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


def make_music_router(plugin_backend) -> APIRouter:
    """``GET /api/music/native-models``: ``{"components", "current"}`` - the
    registered files for each ACE-Step component (``music_options``) and the
    settings' current choice (file names only)."""
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
            out = pairing.native_options(media, cands)
            out["current"] = {k: display_name(v) if v else "" for k, v in configured.items()}
            other_model = "model" in query and query["model"] != configured["model"]
            picked = {k: query.get(k, "" if other_model and k != "model" else configured[k])
                      for k in keys}
            model = picked.pop("model", "")
            out["preview"] = (pairing.preview(media, model, picked, cands) if model
                              else None)
            return out
        try:
            return await run_in_threadpool_bounded(_read, timeout=30.0)
        except ThreadCallTimeout as e:
            raise HTTPException(504, f"Reading the native models timed out: {e}") from e

    return router
