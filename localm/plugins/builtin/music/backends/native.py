# SPDX-License-Identifier: AGPL-3.0-or-later
"""Native music backend: ACE-Step 1.5 through the managed KoboldCpp runtime.

Implements the media backend seam (``ensure_available`` / ``free_vram`` /
``generate``) plus ``refusal`` (inputs only ComfyUI can honour), ``status`` (what
it would use, for the GUI) and ``vram_estimate_bytes``. The settings dict comes
from the music plugin's ``backend.settings``; this module reads its ``native``
block (model files, runtime, planner, low-VRAM mode).
"""

from __future__ import annotations

import json
import secrets
import time
from pathlib import Path
from typing import Optional

OUTPUT_SUFFIX = ".wav"


def _native(s: dict) -> dict:
    blk = s.get("native")
    return blk if isinstance(blk, dict) else {}


def refusal(*, model_overrides=None, sampler_name=None, scheduler=None,
            lyrics_strength=None, placement=None, **_ignored) -> Optional[str]:
    """Why the native backend cannot honour a request with these inputs, or
    None. Checked before any download, VRAM handover or load."""
    if model_overrides:
        return ("Workflow model choices apply to ComfyUI. The native backend uses "
                "the models set in Settings > Music.")
    named = [n for n, v in (("sampler", sampler_name), ("scheduler", scheduler),
                            ("lyrics strength", lyrics_strength)) if v is not None]
    if named:
        return (f"The {', '.join(named)} setting applies to the ComfyUI workflow only; "
                "leave it unset or use the ComfyUI backend.")
    if placement:
        return ("Per-component GPU placement applies to ComfyUI only; turn it off or "
                "use the ComfyUI backend.")
    return None


def ensure_available(s: dict, on_progress=None) -> tuple[bool, str]:
    """Install the runtime and pull missing default models, so the job's later
    VRAM handover is not followed by a long download."""
    from localm.media.koboldcpp import music, pins
    say = on_progress or (lambda _m: None)
    try:
        backend, _models = music.prepare(_native(s), s.get("native_runtime") or "auto",
                                         plan=bool(s.get("plan", True)), on_progress=say)
    except music.Cancelled:
        return False, "Cancelled."
    except (music.ProvisionError, music.ModelError, music.NativeMusicError) as e:
        return False, f"Native music generation is not available: {e}"
    return True, f"Native music runtime ready (KoboldCpp {pins.VERSION}, {backend})."


def free_vram(s: dict) -> bool:
    """Stop the music server; process exit returns all of its VRAM."""
    from localm.media.koboldcpp import server
    server.stop()
    return True


def _model_paths(s: dict) -> dict:
    """component -> file for every component the job would load, without
    pulling; a component that is not available yet is absent."""
    from localm.media.koboldcpp.models import COMPONENTS, find_default, resolve_path
    blk = _native(s)
    found = {}
    for comp in COMPONENTS:
        if comp == "lm" and not s.get("plan", True):
            continue
        configured = str(blk.get(comp) or "").strip()
        p = resolve_path(configured) if configured else find_default(comp)
        if p is not None:
            found[comp] = p
    return found


def vram_estimate_bytes(s: dict) -> int:
    """Peak VRAM estimate for the model files the job would load, or for the
    default set when they are not downloaded yet."""
    from localm.media.koboldcpp.models import _OVERHEAD_WITH_LM, _OVERHEAD_WITHOUT_LM
    from localm.media.koboldcpp.models import estimate_bytes
    from localm.media.koboldcpp.server import ModelSet
    plan = bool(s.get("plan", True))
    paths = _model_paths(s)
    needed = {"text_encoder", "dit", "vae"} | ({"lm"} if plan else set())
    if needed <= set(paths):
        return estimate_bytes(ModelSet(text_encoder=str(paths["text_encoder"]),
                                       dit=str(paths["dit"]), vae=str(paths["vae"]),
                                       lm=str(paths["lm"]) if plan else None))
    from localm.media.koboldcpp.models import DEFAULT_SIZES
    default_files = sum(v for k, v in DEFAULT_SIZES.items() if plan or k != "lm")
    return default_files + (_OVERHEAD_WITH_LM if plan else _OVERHEAD_WITHOUT_LM)


def status(s: dict) -> dict:
    """What the native backend would use, without installing, pulling or
    loading anything: ``runtime`` (the configured choice and the installed
    build, if any), ``models`` (component -> registry name or file name, or
    None when not downloaded) and ``missing`` (the components a job would pull
    first, each with its download spec)."""
    from localm.media.koboldcpp import models, runtime
    from localm.media.koboldcpp.music import NativeMusicError, backend_order
    choice = s.get("native_runtime") or "auto"
    try:
        backend = backend_order(choice)[0]
        build = runtime.build_for(backend)
        installed = runtime.installed(build) is not None
    except (NativeMusicError, runtime.ProvisionError):
        backend, build, installed = None, None, False
    paths = _model_paths(s)
    shown, missing = {}, []
    blk = _native(s)
    for comp in models.COMPONENTS:
        if comp == "lm" and not s.get("plan", True):
            continue
        p = paths.get(comp)
        shown[comp] = (str(blk.get(comp)).strip() if blk.get(comp) else p.name) if p else None
        if p is None and not str(blk.get(comp) or "").strip():
            spec, name = models.default_pull(comp)
            missing.append({"component": comp, "file": models.DEFAULT_FILES[comp],
                            "repo": models.DEFAULT_REPO, "spec": spec, "name": name,
                            "size_bytes": models.DEFAULT_SIZES[comp],
                            "model_type": models.REGISTRY_TYPES[comp]})
    return {"runtime": {"choice": choice, "backend": backend, "build": build,
                        "installed": installed},
            "models": shown, "missing": missing}


def generate(s: dict, tags: str, out_path: Path, *, self_url: str = "",
             write_sidecar: bool = True, on_progress=None,
             lyrics: Optional[str] = None, duration_seconds: float = 120.0,
             swap: bool = False, cancel_check=None, seed: Optional[int] = None,
             steps: Optional[int] = None, cfg: Optional[float] = None,
             shift: Optional[float] = None, **kwargs) -> tuple[bool, str]:
    """Generate one track into *out_path* (a WAV). Returns (ok, message)."""
    from localm.media.koboldcpp import music
    say = on_progress or (lambda _m: None)
    refused = refusal(**kwargs)
    if refused:
        return False, refused
    if swap:
        say("The chat model could not be unloaded first; the native runtime may not "
            "fit in the remaining VRAM.")
    if seed is None or seed <= 0:
        seed = secrets.randbelow(2 ** 31 - 2) + 1
    instrumental = not (lyrics or "").strip()
    request = {
        "caption": tags,
        "lyrics": "[Instrumental]" if instrumental else lyrics,
        "instrumental": instrumental,
        "duration": float(duration_seconds),
        "seed": int(seed),
        "stereo": True,
    }
    if steps is not None:
        request["inference_steps"] = int(steps)
    if cfg is not None:
        request["guidance_scale"] = float(cfg)
    if shift is not None:
        request["shift"] = float(shift)
    plan = bool(s.get("plan", True))
    t0 = time.monotonic()
    try:
        data, backend = music.generate_wav(
            _native(s), s.get("native_runtime") or "auto", request, plan=plan,
            lowvram=bool(s.get("lowvram")), on_progress=say, cancel_check=cancel_check)
        seconds = music.write_wav(data, out_path)
    except music.Cancelled:
        return False, "Cancelled."
    except (music.ProvisionError, music.ModelError, music.NativeMusicError,
            music.ServerError) as e:
        return False, f"Native music generation failed: {e}"
    length = f"{seconds:.1f} s track"
    if plan and abs(seconds - float(duration_seconds)) >= 0.5:
        length += f" (requested {float(duration_seconds):g} s; the planner ended the song there)"
    message = f"Track saved to {out_path} ({length}, seed {seed} - reuse it to reproduce)"
    if not write_sidecar:
        return True, message
    sidecar = {
        "tags": tags,
        "lyrics": None if instrumental else lyrics,
        "duration_seconds": duration_seconds,
        "seed": seed,
        "steps": steps,
        "cfg": cfg,
        "shift": shift,
        "backend": f"native ({backend})",
        "planner": plan,
        "length_seconds": round(seconds, 2),
        "elapsed_seconds": round(time.monotonic() - t0, 1),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    try:
        out_path.with_suffix(out_path.suffix + ".json").write_text(
            json.dumps({k: v for k, v in sidecar.items() if v is not None},
                       indent=2, ensure_ascii=False), encoding="utf-8")
    except OSError as e:
        message += (f"\nThe reproducibility sidecar could not be saved ({e}); "
                    "the track itself was saved.")
    return True, message
