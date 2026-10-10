# SPDX-License-Identifier: AGPL-3.0-or-later
"""Native music backend: ACE-Step 1.5 through the managed KoboldCpp runtime.

Implements the media backend seam (``ensure_available`` / ``free_vram`` /
``generate``). The settings dict comes from the music plugin's
``backend.settings``; this module reads its ``native`` block (model files,
compute backend, planner, low-VRAM mode).
"""

from __future__ import annotations

import json
import secrets
import time
from pathlib import Path
from typing import Optional

OUTPUT_SUFFIX = ".wav"

_COMFY_ONLY = ("lyrics_strength", "sampler_name", "scheduler", "model_overrides")


def _native(s: dict) -> dict:
    blk = s.get("native")
    return blk if isinstance(blk, dict) else {}


def ensure_available(s: dict, on_progress=None) -> tuple[bool, str]:
    """Install the runtime and pull missing default models, so the job's later
    VRAM handover is not followed by a long download."""
    from localm.media.koboldcpp import music, pins
    say = on_progress or (lambda _m: None)
    blk = _native(s)
    try:
        backend, _models = music.prepare(blk, s.get("native_backend") or "auto",
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


def generate(s: dict, tags: str, out_path: Path, *, self_url: str = "",
             write_sidecar: bool = True, on_progress=None,
             lyrics: Optional[str] = None, duration_seconds: float = 120.0,
             swap: bool = False, cancel_check=None, seed: Optional[int] = None,
             steps: Optional[int] = None, cfg: Optional[float] = None,
             shift: Optional[float] = None, **kwargs) -> tuple[bool, str]:
    """Generate one track into *out_path* (a WAV). Returns (ok, message)."""
    from localm.media.koboldcpp import music
    say = on_progress or (lambda _m: None)
    ignored = [k for k in _COMFY_ONLY if kwargs.get(k)]
    if ignored:
        say(f"Not used by the native backend: {', '.join(ignored)}.")
    if swap:
        say("The chat model could not be unloaded first; the native runtime may not "
            "fit in the remaining VRAM.")
    blk = _native(s)
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
            blk, s.get("native_backend") or "auto", request, plan=plan,
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
