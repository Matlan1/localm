# SPDX-License-Identifier: AGPL-3.0-or-later
"""Generating one ACE-Step track natively: pick the backend, make sure the
runtime and model files are present, run the managed server, and write the WAV.

An explicit backend (``cuda``, ``vulkan``, ``cpu``, ``metal``) is the only one
tried; if it cannot run, the reason is raised. ``auto`` tries the recommended
backend, then vulkan, then cpu, skipping one already recorded as not working,
records the outcome, and reports every fallback through the progress callback.
A track that comes back broken (see ``broken_reason``) is a failure: ``auto``
generates it again on the next backend, an explicit backend raises the reason.
"""

from __future__ import annotations

import array
import io
import sys
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
           "backend_order", "broken_reason", "prepare", "generate_wav", "write_wav"]

PINNED_SECONDS = 0.25


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
    """Install the runtime for the first backend to try and check the model
    files are present (default models are never downloaded here; see
    ``models.resolve_models``). Returns (backend, model set). Raises
    :class:`ProvisionError`, :class:`ModelError`, :class:`NativeMusicError` or
    :class:`Cancelled`."""
    say = on_progress or (lambda _m: None)
    backend = backend_order(choice)[0]
    models = resolve_models(native_cfg, use_lm=plan, pull_missing=False,
                            on_progress=say, cancel_check=cancel_check)
    _ensure_runtime(backend, say, cancel_check)
    return backend, models


def _ensure_runtime(backend: str, say: Progress,
                    cancel_check: Optional[CancelCheck]) -> runtime.Runtime:
    try:
        return runtime.ensure_for_backend(backend, on_progress=say,
                                          cancel_check=cancel_check)
    except runtime.InstallCancelled as e:
        raise Cancelled() from e


def generate_wav(native_cfg: dict, choice: str, request: dict, *, plan: bool = True,
                 lowvram: bool = False, timeout: float = 3600.0,
                 on_progress: Optional[Progress] = None,
                 cancel_check: Optional[CancelCheck] = None) -> tuple[bytes, str]:
    """Generate one track. Returns (WAV bytes, backend that produced it)."""
    say = on_progress or (lambda _m: None)
    order = backend_order(choice)
    models = resolve_models(native_cfg, use_lm=plan, pull_missing=False,
                            on_progress=say, cancel_check=cancel_check)
    errors: list[str] = []
    auto = choice.strip().lower() == "auto"
    for i, backend in enumerate(order):
        rt = _ensure_runtime(backend, say, cancel_check)
        try:
            data = server.run(rt, backend, models, work_dir(), prepare=plan,
                              request=request, timeout=timeout, lowvram=lowvram,
                              on_progress=say, cancel_check=cancel_check)
        except StartError as e:
            if auto and e.crashed:
                runtime.record_backend(backend, False, str(e))
            errors.append(f"{backend}: {e}")
            if auto and i + 1 < len(order):
                say(f"The {backend} backend did not start here ({e}); "
                    f"trying {order[i + 1]}.")
                continue
            raise NativeMusicError("; ".join(errors)) from e
        broken = broken_reason(data)
        if broken:
            errors.append(f"{backend}: the track came back broken ({broken})")
            if auto and i + 1 < len(order):
                say(f"The {backend} backend returned a broken track ({broken}); "
                    f"generating it again on {order[i + 1]}, which can take longer.")
                continue
            hint = "" if backend == "cpu" else (
                "; this happens with some prompts on some GPUs: set the native runtime "
                "to cpu in the Music settings, or try other style tags")
            raise NativeMusicError("; ".join(errors) + hint)
        if not runtime.backend_worked(backend):
            runtime.record_backend(backend, True)
        return data, backend
    raise NativeMusicError("; ".join(errors) or "no backend to try")


def broken_reason(data: bytes) -> Optional[str]:
    """Why the 16-bit WAV *data* is not a usable track, or None: a channel
    with no variation at all (digital silence or a constant level), or a run of
    identical full-scale samples on a channel longer than ``PINNED_SECONDS``.
    Other sample widths and unreadable data return None (``write_wav`` reports
    unreadable data)."""
    try:
        with wave.open(io.BytesIO(data), "rb") as src:
            params = src.getparams()
            frames = src.readframes(params.nframes)
    except (wave.Error, EOFError):
        return None
    if params.sampwidth != 2 or params.nframes == 0 or params.nchannels < 1:
        return None
    samples = array.array("h")
    samples.frombytes(frames[:len(frames) - len(frames) % 2])
    if sys.byteorder == "big":
        samples.byteswap()
    run = max(1, int(PINNED_SECONDS * params.framerate))
    for c in range(params.nchannels):
        ch = samples[c::params.nchannels]
        if not ch:
            continue
        raw = ch.tobytes()
        for level in (32767, -32768):
            if level.to_bytes(2, sys.byteorder, signed=True) * run in raw:
                return "the audio is stuck at full scale"
        if max(ch) - min(ch) <= 1:
            return "the audio is a constant level with no sound"
    return None


def write_wav(data: bytes, out_path: Path) -> float:
    """Write the PCM of the WAV *data* to *out_path* as a WAV holding only the
    format and data chunks. Returns the duration in seconds. Raises
    :class:`ServerError` when *data* is not a readable WAV and ``OSError`` when
    the file cannot be written; a partly written file is removed."""
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
    try:
        with wave.open(str(tmp), "wb") as dst:
            dst.setnchannels(params.nchannels)
            dst.setsampwidth(params.sampwidth)
            dst.setframerate(params.framerate)
            dst.writeframes(frames)
        tmp.replace(out_path)
    except BaseException:
        try:
            tmp.unlink(missing_ok=True)
        except OSError as e:
            from localm.debuglog import logger
            logger.warning("koboldcpp: could not remove the partial track %s: %s", tmp, e)
        raise
    return params.nframes / params.framerate
