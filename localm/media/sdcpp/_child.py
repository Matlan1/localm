# SPDX-License-Identifier: AGPL-3.0-or-later
"""The stable-diffusion.cpp worker process. Runs ONLY inside the child spawned
by ``runner.SdRunner``; owns the native library and the loaded model.

Protocol (two ``multiprocessing.Queue``s, tagged tuples). ``req_q``, one
command at a time:
    ("probe", {"runtime_dir", "extra_dirs"})
    ("load", {"runtime_dir", "extra_dirs", "ctx": {<sd_ctx_params_t path fields>, ...}})
    ("generate_image", {"prompt", "negative_prompt", "width", "height", "steps",
                        "cfg_scale", "guidance", "seed", "strength", "sample_method",
                        "scheduler", "clip_skip", "init_image", "loras",
                        "hires": {"upscaler", "model_path", "scale", "steps",
                                  "denoising_strength", "tile_size"}})
    ("generate_video", {"prompt", "negative_prompt", "width", "height", "video_frames",
                        "fps", "steps", "cfg_scale", "flow_shift", "seed",
                        "sample_method", "scheduler", "init_image", "loras"})
    ("upscale_image", {"runtime_dir", "extra_dirs", "model_path", "image", "tile_size"})
    ("shutdown", None)

``resp_q``: zero or more events, then exactly one final reply per command:
    ("progress", {"phase", "step", "steps", "secs"})   event
    ("log", level, text)                                event, WARN and ERROR only
    ("preview", {"step", "steps", "width", "height", "png"})
                                                        event, a small PNG of the
                                                        in-progress image; only when
                                                        the request sets "preview"
    ("ok", value) | ("cancelled", message) | ("error", message)   final

A native abort produces no reply; the parent sees the dead process.
"""

from __future__ import annotations

import ctypes
import io
import os
import queue as _queue
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional, TextIO

_FAULT_ENV = "LOCALM_SDCPP_FAULT_FOR_TEST"

_PROGRESS_MIN_INTERVAL = 0.5

_PREVIEW_MAX_SIDE = 256
_PREVIEW_TARGET_COUNT = 8

_HOST_VIS_VAR = "GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM"
_HOST_VIS_OPT_OUT = frozenset({"0", "false", "off", "no", ""})


class _State:
    lib: Optional[ctypes.CDLL] = None
    ctx: Optional[int] = None
    crash_fh: Optional[TextIO] = None
    generating = False
    sample_steps = 0
    hires_steps = 0
    upscaling = False
    decoding = False
    redact: tuple = ()
    preview_cb: Any = None
    preview_failed = False
    paths: tuple = ()
    model_desc = ""
    last_errors: list = []
    last_progress = 0.0


def _simulate_fault(mode: str) -> None:
    if mode == "hang":
        while True:
            time.sleep(3600)
    if mode == "exit":
        os._exit(134)
    os.abort()


def _arm_crash_trace(path) -> None:
    """Point faulthandler at *path* so a native fault leaves a trace."""
    if path is None:
        return
    import faulthandler
    from localm.debuglog import logger
    try:
        fh = open(path, "w", encoding="utf-8")
        _State.crash_fh = fh
        faulthandler.enable(file=fh, all_threads=True)
    except Exception as e:  # noqa: BLE001
        logger.warning("sd.cpp worker: could not arm the native-fault trace (%s: %s)",
                       type(e).__name__, e)


def _prepare_environment(runtime_dir: Path) -> None:
    """Environment the native runtime reads at initialisation: on a Windows
    Vulkan build, keep weights in dedicated VRAM unless the user opted out
    (ggml switches on the variable's presence, so an opt-out value removes it)."""
    if sys.platform != "win32":
        return
    if not (runtime_dir / "ggml-vulkan.dll").exists():
        return
    raw = os.environ.get(_HOST_VIS_VAR)
    if raw is None:
        os.environ[_HOST_VIS_VAR] = "1"
    elif raw.strip().lower() in _HOST_VIS_OPT_OUT:
        os.environ.pop(_HOST_VIS_VAR, None)


def _set_rocblas_tensile(extra_dirs) -> None:
    if os.environ.get("ROCBLAS_TENSILE_LIBPATH"):
        return
    for d in extra_dirs or []:
        lib = Path(d) / "rocblas" / "library"
        if lib.is_dir():
            os.environ["ROCBLAS_TENSILE_LIBPATH"] = str(lib)
            return


_MODEL_DESC_RE = re.compile(r"\AWan\d\.[\dx]-[A-Za-z0-9.]+-[A-Za-z0-9.]+\Z")


def _redacted(text: str) -> str:
    """*text* with the prompt replaced by ``<prompt>`` and every loaded model
    path reduced to its file name."""
    for secret in _State.redact:
        if secret:
            text = text.replace(secret, "<prompt>")
    for path in _State.paths:
        name = path.replace("\\", "/").rsplit("/", 1)[-1]
        for spelling in {path, path.replace("\\", "/"), path.replace("/", "\\")}:
            text = text.replace(spelling, name)
    return text


def preview_interval(steps: int) -> int:
    """Denoiser steps between previews: about ``_PREVIEW_TARGET_COUNT`` per
    sampling pass, at least every step."""
    return max(1, -(-int(steps) // _PREVIEW_TARGET_COUNT))


def preview_png(width: int, height: int, channel: int, data: bytes) -> bytes:
    """*data* (interleaved 8-bit pixels) as a PNG no larger than
    ``_PREVIEW_MAX_SIDE`` on a side, aspect ratio kept."""
    from PIL import Image
    mode = {1: "L", 3: "RGB", 4: "RGBA"}.get(channel)
    if mode is None:
        raise ValueError(f"a preview with {channel} channels")
    im = Image.frombytes(mode, (width, height), data)
    im.thumbnail((_PREVIEW_MAX_SIDE, _PREVIEW_MAX_SIDE), Image.Resampling.BILINEAR)
    out = io.BytesIO()
    im.save(out, format="PNG", compress_level=1)
    return out.getvalue()


def _make_callbacks(resp_q):
    from . import _binding as b

    def _log(level, text, _data):
        try:
            msg = (text or b"").decode("utf-8", "replace").strip()
        except Exception:
            return
        if not msg:
            return
        low = msg.lower()
        if "decoding" in low and "latent" in low:
            _State.decoding = True
        head = msg.split(" --- ", 1)[0].strip()
        if not _State.model_desc and _MODEL_DESC_RE.match(head):
            _State.model_desc = head
        if level >= b.SD_LOG_WARN:
            msg = _redacted(msg)
            if level >= b.SD_LOG_ERROR:
                _State.last_errors = (_State.last_errors + [msg])[-5:]
            try:
                resp_q.put(("log", int(level), msg))
            except Exception:
                pass

    def _progress(step, steps, secs, _data):
        now = time.monotonic()
        if step not in (1, steps) and now - _State.last_progress < _PROGRESS_MIN_INTERVAL:
            return
        _State.last_progress = now
        if _State.upscaling:
            phase = "upscaling"
        elif steps and steps in (_State.sample_steps, _State.hires_steps) and not _State.decoding:
            phase = "sampling"
        elif _State.decoding:
            phase = "decoding"
        else:
            phase = "loading"
        try:
            resp_q.put(("progress", {"phase": phase, "step": int(step),
                                     "steps": int(steps), "secs": float(secs)}))
        except Exception:
            pass

    def _preview(step, frame_count, frames, _is_noisy, _data):
        if not _State.generating or frame_count < 1 or not frames:
            return
        try:
            img = frames[frame_count // 2]
            w, h, c = int(img.width), int(img.height), int(img.channel)
            if not img.data or w < 1 or h < 1:
                return
            png = preview_png(w, h, c, ctypes.string_at(img.data, w * h * c))
            resp_q.put(("preview", {"step": abs(int(step)), "steps": int(_State.sample_steps),
                                    "width": w, "height": h, "png": png}))
        except Exception as e:  # noqa: BLE001
            if not _State.preview_failed:
                _State.preview_failed = True
                from localm.debuglog import logger
                logger.warning("sd.cpp worker: a live preview frame could not be made "
                               "(%s: %s); further failures are not logged",
                               type(e).__name__, e)

    return b.SD_LOG_CB(_log), b.SD_PROGRESS_CB(_progress), b.SD_PREVIEW_CB(_preview)


def _arm_preview(lib, payload, steps: int) -> bool:
    """Ask the library for in-progress frames every ``preview_interval(steps)``
    denoiser steps when the request sets ``"preview"``. True when armed."""
    if not payload.get("preview") or _State.preview_cb is None:
        return False
    from . import _binding as b
    _State.preview_failed = False
    lib.sd_set_preview_callback(_State.preview_cb, b.PREVIEW_PROJ, preview_interval(steps),
                                True, False, None)
    return True


def _disarm_preview(lib) -> None:
    from . import _binding as b
    lib.sd_set_preview_callback(b.SD_PREVIEW_CB(), b.PREVIEW_NONE, 0, False, False, None)


def _cancel_watcher(cancel_event) -> None:
    """Forward a parent cancel request to the running generation, once per
    request: waits for the event, cancels, then waits for the parent to clear it."""
    from . import _binding as b
    while True:
        cancel_event.wait()
        lib, ctx = _State.lib, _State.ctx
        if lib is not None and ctx is not None and _State.generating:
            try:
                lib.sd_cancel_generation(ctx, b.SD_CANCEL_ALL)
            except Exception:
                pass
        while cancel_event.is_set():
            time.sleep(0.05)


def _load_lib(payload):
    from . import _binding as b
    runtime_dir = Path(payload["runtime_dir"])
    extra = [Path(d) for d in payload.get("extra_dirs") or []]
    _prepare_environment(runtime_dir)
    _set_rocblas_tensile(extra)
    try:
        return b.load_library(runtime_dir, extra)
    except OSError as e:
        raise RuntimeError(
            f"could not load the stable-diffusion.cpp runtime from {runtime_dir} ({e}). "
            "Reinstall it with 'localm setup-sdcpp --force', or pick another runtime "
            "backend ('localm setup-sdcpp --backend vulkan').") from e


def _error_with_native(message: str) -> str:
    if _State.last_errors:
        return f"{message}: {' | '.join(_State.last_errors[-3:])}"
    return message


def _do_probe(payload):
    from . import _binding as b
    lib = _State.lib or _load_lib(payload)
    _State.lib = lib
    info = (lib.sd_get_system_info() or b"").decode("utf-8", "replace")
    return {"commit": (lib.sd_commit() or b"").decode("utf-8", "replace"),
            "devices": b.list_devices(lib), "system_info": info}


_CTX_STRING_FIELDS = ("model_path", "clip_l_path", "clip_g_path", "clip_vision_path",
                      "t5xxl_path", "llm_path", "llm_vision_path", "diffusion_model_path",
                      "high_noise_diffusion_model_path", "vae_path", "taesd_path",
                      "backend", "params_backend", "max_vram", "tokenizer")
_CTX_BOOL_FIELDS = ("flash_attn", "diffusion_flash_attn", "enable_mmap", "vae_conv_direct",
                    "force_sdxl_vae_conv_scale",
                    "diffusion_conv_direct", "auto_fit")


def _do_load(payload):
    from . import _binding as b
    lib = _State.lib or _load_lib(payload)
    _State.lib = lib
    if not b.list_devices(lib):
        raise RuntimeError(
            f"the stable-diffusion.cpp runtime at {payload.get('runtime_dir')} loaded "
            "but reports no compute device, so nothing can run on it. Reinstall it "
            "with 'localm setup-sdcpp --force', or pick another runtime backend "
            "('localm setup-sdcpp --backend vulkan').")
    if _State.ctx is not None:
        lib.free_sd_ctx(_State.ctx)
        _State.ctx = None
    params = b._init_struct(lib, "sd_ctx_params_init", b.sd_ctx_params_t)
    ctx_fields = dict(payload.get("ctx") or {})
    keep = []
    for name in _CTX_STRING_FIELDS:
        value = ctx_fields.get(name)
        if value:
            enc = str(value).encode("utf-8")
            keep.append(enc)
            setattr(params, name, enc)
    for name in _CTX_BOOL_FIELDS:
        if name in ctx_fields and ctx_fields[name] is not None:
            setattr(params, name, bool(ctx_fields[name]))
    if ctx_fields.get("n_threads"):
        params.n_threads = int(ctx_fields["n_threads"])
    _State.last_errors = []
    _State.model_desc = ""
    _State.paths = tuple(str(v) for k, v in ctx_fields.items()
                         if k.endswith("_path") and isinstance(v, str) and v)
    ctx = lib.new_sd_ctx(ctypes.byref(params))
    if not ctx:
        raise RuntimeError(_error_with_native("stable-diffusion.cpp could not load the model"))
    _State.ctx = ctx
    version = (lib.sd_get_model_version_name(ctx) or b"").decode("utf-8", "replace")
    method = lib.sd_get_default_sample_method(ctx)
    scheduler = lib.sd_get_default_scheduler(ctx, method)
    return {
        "version": version,
        "desc": _State.model_desc,
        "image": bool(lib.sd_ctx_supports_image_generation(ctx)),
        "video": bool(lib.sd_ctx_supports_video_generation(ctx)),
        "default_sample_method": (lib.sd_sample_method_name(method) or b"").decode(),
        "default_scheduler": (lib.sd_scheduler_name(scheduler) or b"").decode(),
        "devices": b.list_devices(lib),
    }


def _enum_from_name(lib, kind: str, name):
    from . import _binding as b
    if not name:
        return None
    if kind == "sample_method":
        value, count = lib.str_to_sample_method(str(name).encode()), b._SAMPLE_METHOD_COUNT
    else:
        value, count = lib.str_to_scheduler(str(name).encode()), b._SCHEDULER_COUNT
    if value < 0 or value >= count:
        raise ValueError(f"unknown {kind.replace('_', ' ')} {name!r}")
    return value


def _set_hires(lib, b, p, payload):
    """Enable the hires pass of image params *p* from ``payload["hires"]``.
    Returns the bytes objects that must stay alive until generation returns
    (None without a hires request)."""
    hires = payload.get("hires")
    if not hires:
        return None
    upscaler = lib.str_to_sd_hires_upscaler(str(hires["upscaler"]).encode())
    if upscaler < 1 or upscaler >= b._SD_HIRES_UPSCALER_COUNT:
        raise ValueError(f"unknown hires upscaler {hires['upscaler']!r}")
    h = p.hires
    h.enabled = True
    h.upscaler = upscaler
    keep = []
    if hires.get("model_path"):
        enc = str(hires["model_path"]).encode("utf-8")
        keep.append(enc)
        h.model_path = enc
        if str(hires["model_path"]) not in _State.paths:
            _State.paths = _State.paths + (str(hires["model_path"]),)
    if hires.get("scale") is not None:
        h.scale = float(hires["scale"])
    if hires.get("steps") is not None:
        h.steps = int(hires["steps"])
    if hires.get("denoising_strength") is not None:
        h.denoising_strength = float(hires["denoising_strength"])
    if hires.get("tile_size"):
        h.upscale_tile_size = int(hires["tile_size"])
    return keep


def _set_loras(b, p, payload):
    """Fill *p*'s ``loras``/``lora_count`` from ``payload["loras"]`` (each
    ``{"path", "multiplier", "is_high_noise"}``). Returns the ctypes objects
    that must stay alive until generation returns."""
    loras = payload.get("loras") or []
    if not loras:
        return None
    arr = (b.sd_lora_t * len(loras))()
    names = []
    for i, lora in enumerate(loras):
        enc = str(lora["path"]).encode("utf-8")
        names.append(enc)
        arr[i].path = enc
        arr[i].multiplier = float(lora.get("multiplier", 1.0))
        arr[i].is_high_noise = bool(lora.get("is_high_noise", False))
    p.loras = ctypes.cast(arr, ctypes.POINTER(b.sd_lora_t))
    p.lora_count = len(loras)
    new = tuple(str(lora["path"]) for lora in loras if str(lora["path"]) not in _State.paths)
    _State.paths = _State.paths + new
    return arr, names


def _do_generate_image(payload, cancel_event):
    from . import _binding as b
    lib, ctx = _State.lib, _State.ctx
    if lib is None or ctx is None:
        raise RuntimeError("no model is loaded in the media worker")
    p = b._init_struct(lib, "sd_img_gen_params_init", b.sd_img_gen_params_t)
    prompt = str(payload.get("prompt") or "")
    negative = str(payload.get("negative_prompt") or "")
    _State.redact = tuple(s for s in (prompt, negative) if len(s) >= 4)
    p.prompt = prompt.encode("utf-8")
    p.negative_prompt = negative.encode("utf-8")
    p.width = int(payload["width"])
    p.height = int(payload["height"])
    p.batch_count = 1
    lora_refs = _set_loras(b, p, payload)
    seed = payload.get("seed")
    p.seed = int(seed) if seed is not None else -1
    if payload.get("steps"):
        p.sample_params.sample_steps = int(payload["steps"])
    if payload.get("cfg_scale") is not None:
        p.sample_params.guidance.txt_cfg = float(payload["cfg_scale"])
    if payload.get("guidance") is not None:
        p.sample_params.guidance.distilled_guidance = float(payload["guidance"])
    if payload.get("clip_skip") is not None:
        p.clip_skip = int(payload["clip_skip"])
    method = _enum_from_name(lib, "sample_method", payload.get("sample_method"))
    if method is not None:
        p.sample_params.sample_method = method
    sched = _enum_from_name(lib, "scheduler", payload.get("scheduler"))
    if sched is not None:
        p.sample_params.scheduler = sched
    hires_refs = _set_hires(lib, b, p, payload)
    keep = None
    init = payload.get("init_image")
    if init:
        w, h, c, data = int(init["width"]), int(init["height"]), int(init["channel"]), init["data"]
        if len(data) != w * h * c:
            raise ValueError("init image buffer does not match its size")
        keep = (ctypes.c_uint8 * len(data)).from_buffer_copy(data)
        p.init_image = b.sd_image_t(w, h, c, ctypes.cast(keep, ctypes.POINTER(ctypes.c_uint8)))
        if payload.get("strength") is not None:
            p.strength = float(payload["strength"])

    out = ctypes.POINTER(b.sd_image_t)()
    count = ctypes.c_int(0)
    lib.sd_cancel_generation(ctx, b.SD_CANCEL_RESET)
    _State.last_errors = []
    _State.sample_steps = int(p.sample_params.sample_steps)
    _State.hires_steps = (int(p.hires.steps) or _State.sample_steps) if p.hires.enabled else 0
    _State.decoding = False
    previewing = _arm_preview(lib, payload, _State.sample_steps)
    _State.generating = True
    try:
        ok = lib.generate_image(ctx, ctypes.byref(p), ctypes.byref(out), ctypes.byref(count))
    finally:
        _State.generating = False
        if previewing:
            _disarm_preview(lib)
        _State.redact = ()
        del keep, lora_refs, hires_refs
    cancelled = cancel_event.is_set()
    try:
        if not ok or count.value < 1 or not out:
            if cancelled:
                return "cancelled", "Generation cancelled."
            raise RuntimeError(_error_with_native("stable-diffusion.cpp generation failed"))
        img = out[0]
        if not img.data:
            if cancelled:
                return "cancelled", "Generation cancelled."
            raise RuntimeError(_error_with_native("stable-diffusion.cpp returned no image"))
        size = int(img.width) * int(img.height) * int(img.channel)
        data = ctypes.string_at(img.data, size)
        result = {"width": int(img.width), "height": int(img.height),
                  "channel": int(img.channel), "data": data, "seed": int(p.seed)}
    finally:
        if out:
            lib.free_sd_images(out, count.value)
    if cancelled:
        return "cancelled", "Generation cancelled."
    return "ok", result


def _do_upscale_image(payload):
    """Upscale ``payload["image"]`` with the ESRGAN model ``payload["model_path"]``.
    Loads the library when no job has yet; needs no diffusion model. Returns the
    image as ``{"width", "height", "channel", "data", "scale"}``."""
    from . import _binding as b
    lib = _State.lib or _load_lib(payload)
    _State.lib = lib
    path = str(payload["model_path"])
    scale = int(lib.get_upscaler_model_scale(path.encode("utf-8")))
    if scale < 1:
        raise ValueError(f"{Path(path).name} is not an RGB ESRGAN upscaler model")
    img = payload["image"]
    w, h, c, data = int(img["width"]), int(img["height"]), int(img["channel"]), img["data"]
    if len(data) != w * h * c:
        raise ValueError("input image buffer does not match its size")
    max_side = int(payload.get("max_output_side") or 0)
    if max_side and max(w, h) * scale > max_side:
        raise ValueError(f"upscaling {w}x{h} by {scale}x would make a {w * scale}x{h * scale} "
                         f"image; each side is limited to {max_side} pixels")
    _State.last_errors = []
    _State.decoding = False
    _State.sample_steps = _State.hires_steps = 0
    if path not in _State.paths:
        _State.paths = _State.paths + (path,)
    keep = (ctypes.c_uint8 * len(data)).from_buffer_copy(data)
    src = b.sd_image_t(w, h, c, ctypes.cast(keep, ctypes.POINTER(ctypes.c_uint8)))
    uctx = lib.new_upscaler_ctx(path.encode("utf-8"), False, 0,
                                int(payload.get("tile_size") or 128), None, None)
    if not uctx:
        raise RuntimeError(_error_with_native("stable-diffusion.cpp could not load the upscaler"))
    out = ctypes.POINTER(b.sd_image_t)()
    count = ctypes.c_int(0)
    _State.generating = True
    _State.upscaling = True
    try:
        ok = lib.upscale(uctx, src, scale, ctypes.byref(out), ctypes.byref(count))
    finally:
        _State.upscaling = False
        _State.generating = False
        lib.free_upscaler_ctx(uctx)
        del keep
    try:
        if not ok or count.value < 1 or not out or not out[0].data:
            raise RuntimeError(_error_with_native("stable-diffusion.cpp upscaling failed"))
        res = out[0]
        size = int(res.width) * int(res.height) * int(res.channel)
        return {"width": int(res.width), "height": int(res.height),
                "channel": int(res.channel), "data": ctypes.string_at(res.data, size),
                "scale": scale}
    finally:
        if out:
            lib.free_sd_images(out, count.value)


def _do_generate_video(payload, cancel_event):
    from . import _binding as b
    lib, ctx = _State.lib, _State.ctx
    if lib is None or ctx is None:
        raise RuntimeError("no model is loaded in the media worker")
    p = b._init_struct(lib, "sd_vid_gen_params_init", b.sd_vid_gen_params_t)
    prompt = str(payload.get("prompt") or "")
    negative = str(payload.get("negative_prompt") or "")
    _State.redact = tuple(s for s in (prompt, negative) if len(s) >= 4)
    p.prompt = prompt.encode("utf-8")
    p.negative_prompt = negative.encode("utf-8")
    p.width = int(payload["width"])
    p.height = int(payload["height"])
    p.video_frames = int(payload["video_frames"])
    p.fps = int(payload.get("fps") or 16)
    lora_refs = _set_loras(b, p, payload)
    seed = payload.get("seed")
    p.seed = int(seed) if seed is not None else -1
    if payload.get("steps"):
        p.sample_params.sample_steps = int(payload["steps"])
    if payload.get("cfg_scale") is not None:
        p.sample_params.guidance.txt_cfg = float(payload["cfg_scale"])
    if payload.get("flow_shift") is not None:
        p.sample_params.flow_shift = float(payload["flow_shift"])
    method = _enum_from_name(lib, "sample_method", payload.get("sample_method"))
    if method is not None:
        p.sample_params.sample_method = method
    sched = _enum_from_name(lib, "scheduler", payload.get("scheduler"))
    if sched is not None:
        p.sample_params.scheduler = sched
    keep = None
    init = payload.get("init_image")
    if init:
        w, h, c, data = int(init["width"]), int(init["height"]), int(init["channel"]), init["data"]
        if len(data) != w * h * c:
            raise ValueError("init image buffer does not match its size")
        keep = (ctypes.c_uint8 * len(data)).from_buffer_copy(data)
        p.init_image = b.sd_image_t(w, h, c, ctypes.cast(keep, ctypes.POINTER(ctypes.c_uint8)))
    frames = ctypes.POINTER(b.sd_image_t)()
    count = ctypes.c_int(0)
    audio = ctypes.POINTER(b.sd_audio_t)()
    fps_out = ctypes.c_int(0)
    lib.sd_cancel_generation(ctx, b.SD_CANCEL_RESET)
    _State.last_errors = []
    _State.sample_steps = int(p.sample_params.sample_steps)
    _State.hires_steps = 0
    _State.decoding = False
    previewing = _arm_preview(lib, payload, _State.sample_steps)
    _State.generating = True
    try:
        ok = lib.generate_video(ctx, ctypes.byref(p), ctypes.byref(frames), ctypes.byref(count),
                                ctypes.byref(audio), ctypes.byref(fps_out))
    finally:
        _State.generating = False
        if previewing:
            _disarm_preview(lib)
        _State.redact = ()
        del keep, lora_refs
    cancelled = cancel_event.is_set()
    try:
        if not ok or count.value < 1 or not frames:
            if cancelled:
                return "cancelled", "Generation cancelled."
            raise RuntimeError(_error_with_native("stable-diffusion.cpp video generation failed"))
        first = frames[0]
        w, h, c = int(first.width), int(first.height), int(first.channel)
        out_frames = []
        for i in range(count.value):
            img = frames[i]
            if not img.data or (int(img.width), int(img.height), int(img.channel)) != (w, h, c):
                raise RuntimeError("stable-diffusion.cpp returned an incomplete frame")
            out_frames.append(ctypes.string_at(img.data, w * h * c))
        out_audio = None
        if audio and audio.contents.data and audio.contents.sample_count:
            a = audio.contents
            n = int(a.sample_count) * int(a.channels)
            out_audio = {"sample_rate": int(a.sample_rate), "channels": int(a.channels),
                         "sample_count": int(a.sample_count),
                         "data": ctypes.string_at(a.data, n * 4)}
        result = {"width": w, "height": h, "channel": c, "frames": out_frames,
                  "fps": int(fps_out.value) or int(p.fps), "audio": out_audio,
                  "seed": int(p.seed)}
    finally:
        if frames:
            lib.free_sd_images(frames, count.value)
        if audio:
            lib.free_sd_audio(audio)
    if cancelled:
        return "cancelled", "Generation cancelled."
    return "ok", result


def worker_main(req_q, resp_q, cancel_event, crash_trace_path=None) -> None:
    """Child entry point: serve commands until "shutdown" or a closed queue."""
    _arm_crash_trace(crash_trace_path)
    from localm.debuglog import attach_child_logging
    attach_child_logging()
    from localm._mp_spawn import (ignore_interrupt_signals, install_parent_death_watchdog,
                                   suppress_native_error_dialogs)
    install_parent_death_watchdog()
    ignore_interrupt_signals()
    suppress_native_error_dialogs()

    log_cb, progress_cb, _State.preview_cb = _make_callbacks(resp_q)
    callbacks_set = False
    threading.Thread(target=_cancel_watcher, args=(cancel_event,),
                     name="localm-sdcpp-cancel", daemon=True).start()

    while True:
        try:
            cmd = req_q.get()
        except (EOFError, OSError, _queue.Empty):
            return
        if cmd is None:
            return
        name, payload = cmd[0], (cmd[1] if len(cmd) > 1 else None)
        fault = os.environ.get(_FAULT_ENV)
        if fault and name != "shutdown":
            _simulate_fault(fault)
        if name == "shutdown":
            lib, ctx = _State.lib, _State.ctx
            if lib is not None and ctx is not None:
                lib.free_sd_ctx(ctx)
                _State.ctx = None
            return
        try:
            if name in ("probe", "load", "upscale_image") and not callbacks_set:
                lib = _State.lib or _load_lib(payload)
                _State.lib = lib
                lib.sd_set_log_callback(log_cb, None)
                lib.sd_set_progress_callback(progress_cb, None)
                callbacks_set = True
            if name == "probe":
                resp_q.put(("ok", _do_probe(payload)))
            elif name == "load":
                resp_q.put(("ok", _do_load(payload)))
            elif name == "upscale_image":
                resp_q.put(("ok", _do_upscale_image(payload)))
            elif name == "generate_image":
                kind, value = _do_generate_image(payload, cancel_event)
                resp_q.put((kind, value))
            elif name == "generate_video":
                kind, value = _do_generate_video(payload, cancel_event)
                resp_q.put((kind, value))
            else:
                resp_q.put(("error", f"unknown sd.cpp worker command: {name!r}"))
        except Exception as e:  # noqa: BLE001
            resp_q.put(("error", f"{type(e).__name__}: {e}" if not str(e) else str(e)))
