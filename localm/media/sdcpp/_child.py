# SPDX-License-Identifier: AGPL-3.0-or-later
"""The stable-diffusion.cpp worker process. Runs ONLY inside the child spawned
by ``runner.SdRunner``; owns the native library and the loaded model.

Protocol (two ``multiprocessing.Queue``s, tagged tuples). ``req_q``, one
command at a time:
    ("probe", {"runtime_dir", "extra_dirs"})
    ("load", {"runtime_dir", "extra_dirs", "ctx": {<sd_ctx_params_t path fields>, ...}})
    ("generate_image", {"prompt", "negative_prompt", "width", "height", "steps",
                        "cfg_scale", "guidance", "seed", "strength", "sample_method",
                        "scheduler", "clip_skip", "init_image"})
    ("shutdown", None)

``resp_q``: zero or more events, then exactly one final reply per command:
    ("progress", {"phase", "step", "steps", "secs"})   event
    ("log", level, text)                                event, WARN and ERROR only
    ("ok", value) | ("cancelled", message) | ("error", message)   final

A native abort produces no reply; the parent sees the dead process.
"""

from __future__ import annotations

import ctypes
import os
import queue as _queue
import sys
import threading
import time
from pathlib import Path

_FAULT_ENV = "LOCALM_SDCPP_FAULT_FOR_TEST"

_PROGRESS_MIN_INTERVAL = 0.5

_HOST_VIS_VAR = "GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM"
_HOST_VIS_OPT_OUT = frozenset({"0", "false", "off", "no", ""})


class _State:
    lib = None
    ctx = None
    generating = False
    sample_steps = 0
    decoding = False
    redact: tuple = ()
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


def _redacted(text: str) -> str:
    for secret in _State.redact:
        if secret:
            text = text.replace(secret, "<prompt>")
    return text


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
        if _State.sample_steps and steps == _State.sample_steps and not _State.decoding:
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

    return b.SD_LOG_CB(_log), b.SD_PROGRESS_CB(_progress)


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
                    "diffusion_conv_direct", "auto_fit")


def _do_load(payload):
    from . import _binding as b
    if _State.lib is None:
        _State.lib = _load_lib(payload)
    lib = _State.lib
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
    ctx = lib.new_sd_ctx(ctypes.byref(params))
    if not ctx:
        raise RuntimeError(_error_with_native("stable-diffusion.cpp could not load the model"))
    _State.ctx = ctx
    version = (lib.sd_get_model_version_name(ctx) or b"").decode("utf-8", "replace")
    method = lib.sd_get_default_sample_method(ctx)
    return {
        "version": version,
        "image": bool(lib.sd_ctx_supports_image_generation(ctx)),
        "video": bool(lib.sd_ctx_supports_video_generation(ctx)),
        "default_sample_method": (lib.sd_sample_method_name(method) or b"").decode(),
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


def _do_generate_image(payload, cancel_event):
    from . import _binding as b
    lib, ctx = _State.lib, _State.ctx
    if lib is None or ctx is None:
        raise RuntimeError("no model is loaded in the image worker")
    p = b._init_struct(lib, "sd_img_gen_params_init", b.sd_img_gen_params_t)
    prompt = str(payload.get("prompt") or "")
    negative = str(payload.get("negative_prompt") or "")
    _State.redact = tuple(s for s in (prompt, negative) if len(s) >= 4)
    p.prompt = prompt.encode("utf-8")
    p.negative_prompt = negative.encode("utf-8")
    p.width = int(payload["width"])
    p.height = int(payload["height"])
    p.batch_count = 1
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
    _State.decoding = False
    _State.generating = True
    try:
        ok = lib.generate_image(ctx, ctypes.byref(p), ctypes.byref(out), ctypes.byref(count))
    finally:
        _State.generating = False
        _State.redact = ()
        del keep
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

    log_cb, progress_cb = _make_callbacks(resp_q)
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
            if _State.lib is not None and _State.ctx is not None:
                _State.lib.free_sd_ctx(_State.ctx)
                _State.ctx = None
            return
        try:
            if name in ("probe", "load") and not callbacks_set:
                if _State.lib is None:
                    _State.lib = _load_lib(payload)
                _State.lib.sd_set_log_callback(log_cb, None)
                _State.lib.sd_set_progress_callback(progress_cb, None)
                callbacks_set = True
            if name == "probe":
                resp_q.put(("ok", _do_probe(payload)))
            elif name == "load":
                resp_q.put(("ok", _do_load(payload)))
            elif name == "generate_image":
                kind, value = _do_generate_image(payload, cancel_event)
                resp_q.put((kind, value))
            else:
                resp_q.put(("error", f"unknown sd.cpp worker command: {name!r}"))
        except Exception as e:  # noqa: BLE001
            resp_q.put(("error", f"{type(e).__name__}: {e}" if not str(e) else str(e)))
