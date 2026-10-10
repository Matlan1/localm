# SPDX-License-Identifier: AGPL-3.0-or-later
"""Generating one ACE-Step track natively: pick the backend, make sure the
runtime and model files are present, run the managed server, and write the WAV.

An explicit backend (``cuda``, ``vulkan``, ``cpu``, ``metal``) is the only one
tried; if it cannot run, the reason is raised. ``auto`` tries the recommended
backend, then vulkan, then cpu, skipping one already recorded as not working,
records the outcome, and reports every fallback through the progress callback.
"""

from __future__ import annotations

import io
import wave
from pathlib import Path
from typing import Callable, Optional

from . import runtime, server
from .models import ModelError, resolve_models
from .runtime import ProvisionError
from .server import Cancelled, ServerError, StartError

Progress = Callable[[str], None]
CancelCheck = Callable[[], bool]

__all__ = ["Cancelled", "ModelError", "NativeMusicError", "ProvisionError", "ServerError",
           "backend_order", "prepare", "generate_wav", "write_wav"]


class NativeMusicError(RuntimeError):
    """Native generation cannot run here; the message says why."""


def backend_order(choice: str) -> list[str]:
    """Backends to try for *choice*, skipping (under ``auto``) any recorded as
    not working unless every one is."""
    choice = (choice or "auto").strip().lower()
    if choice != "auto":
        if choice not in runtime.BACKENDS:
            raise NativeMusicError(f"unknown native music backend '{choice}' "
                                   f"(choose from auto, {', '.join(runtime.BACKENDS)})")
        if choice not in runtime.available_backends():
            raise NativeMusicError(
                f"the '{choice}' backend has no KoboldCpp build for this platform "
                f"(available: {', '.join(runtime.available_backends()) or 'none'})")
        return [choice]
    order = runtime.fallback_order("auto")
    if not order:
        raise NativeMusicError("KoboldCpp publishes no build for this platform")
    usable = [b for b in order if runtime.backend_failed(b) is None]
    return usable or order


def work_dir() -> Path:
    return runtime.runtimes_root() / "work"


def prepare(native_cfg: dict, choice: str, *, plan: bool = True,
            on_progress: Optional[Progress] = None,
            cancel_check: Optional[CancelCheck] = None) -> tuple[str, server.ModelSet]:
    """Install the runtime for the first backend to try and pull any missing
    default models. Returns (backend, model set). Raises :class:`ProvisionError`,
    :class:`ModelError`, :class:`NativeMusicError` or :class:`Cancelled`."""
    say = on_progress or (lambda _m: None)
    backend = backend_order(choice)[0]
    runtime.ensure_for_backend(backend, on_progress=say)
    models = resolve_models(native_cfg, use_lm=plan, pull_missing=True,
                            on_progress=say, cancel_check=cancel_check)
    return backend, models


def generate_wav(native_cfg: dict, choice: str, request: dict, *, plan: bool = True,
                 lowvram: bool = False, timeout: float = 3600.0,
                 on_progress: Optional[Progress] = None,
                 cancel_check: Optional[CancelCheck] = None) -> tuple[bytes, str]:
    """Generate one track. Returns (WAV bytes, backend that produced it)."""
    say = on_progress or (lambda _m: None)
    order = backend_order(choice)
    models = resolve_models(native_cfg, use_lm=plan, pull_missing=True,
                            on_progress=say, cancel_check=cancel_check)
    errors: list[str] = []
    for i, backend in enumerate(order):
        rt = runtime.ensure_for_backend(backend, on_progress=say)
        try:
            data = server.run(rt, backend, models, work_dir(), prepare=plan,
                              request=request, timeout=timeout, lowvram=lowvram,
                              on_progress=say, cancel_check=cancel_check)
        except StartError as e:
            if choice.strip().lower() == "auto":
                runtime.record_backend(backend, False, str(e))
            errors.append(f"{backend}: {e}")
            if i + 1 < len(order):
                say(f"The {backend} backend did not start here ({e}); "
                    f"trying {order[i + 1]}.")
                continue
            raise NativeMusicError("; ".join(errors)) from e
        if choice.strip().lower() == "auto" and not runtime.backend_worked(backend):
            runtime.record_backend(backend, True)
        return data, backend
    raise NativeMusicError("; ".join(errors) or "no backend to try")


def write_wav(data: bytes, out_path: Path) -> float:
    """Write the PCM of the WAV *data* to *out_path* as a WAV holding only the
    format and data chunks. Returns the duration in seconds. Raises
    :class:`ServerError` when *data* is not a readable WAV."""
    try:
        with wave.open(io.BytesIO(data), "rb") as src:
            params = src.getparams()
            frames = src.readframes(params.nframes)
    except (wave.Error, EOFError) as e:
        raise ServerError(f"the music runtime returned audio that is not a readable WAV: {e}") from e
    if params.nframes == 0 or params.framerate <= 0:
        raise ServerError("the music runtime returned an empty track")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".part")
    with wave.open(str(tmp), "wb") as dst:
        dst.setnchannels(params.nchannels)
        dst.setsampwidth(params.sampwidth)
        dst.setframerate(params.framerate)
        dst.writeframes(frames)
    tmp.replace(out_path)
    return params.nframes / params.framerate
