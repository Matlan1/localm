# SPDX-License-Identifier: AGPL-3.0-or-later
"""ctypes binding of stable-diffusion.cpp's C API (``include/stable-diffusion.h``)
at the commit pinned in ``pins.py``.

Imported and loaded ONLY inside the sd.cpp worker process (``_child.py``). The
runtime's ``stable-diffusion`` library imports ``ggml`` and ``ggml-base`` by
name, the same module names as llama.cpp's own ggml, so the two runtimes must
never be loaded into one process.

The structs mirror the header field for field. ``verify_abi`` checks them
against the loaded library before any struct crosses the FFI boundary.
"""

from __future__ import annotations

import ctypes
import math
import os
import sys
from ctypes import (POINTER, Structure, c_bool, c_char_p, c_float, c_int,
                    c_int64, c_size_t, c_uint8, c_uint32, c_uint64, c_void_p)
from pathlib import Path
from typing import Optional

from . import pins

_enum = c_int

# enum sd_cancel_mode_t
SD_CANCEL_ALL = 0
SD_CANCEL_NEW_LATENTS = 1
SD_CANCEL_RESET = 2

# enum sd_log_level_t
SD_LOG_DEBUG, SD_LOG_VERBOSE, SD_LOG_INFO, SD_LOG_WARN, SD_LOG_ERROR = range(5)

# Sentinel counts the library's *_init functions write, used by verify_abi.
_SD_TYPE_COUNT = 45
_RNG_TYPE_COUNT = 3
_CUDA_RNG = 1
_PREDICTION_COUNT = 8
_SAMPLE_METHOD_COUNT = 21
_SCHEDULER_COUNT = 17
_SD_VAE_FORMAT_AUTO = -1
_SD_HIRES_UPSCALER_LATENT = 1


class sd_tiling_params_t(Structure):
    _fields_ = [
        ("enabled", c_bool),
        ("temporal_tiling", c_bool),
        ("tile_size_w", c_int),
        ("tile_size_h", c_int),
        ("target_overlap", c_float),
        ("rel_size_w", c_float),
        ("rel_size_h", c_float),
        ("extra_tiling_args", c_char_p),
    ]


class sd_embedding_t(Structure):
    _fields_ = [("name", c_char_p), ("path", c_char_p)]


class sd_ctx_params_t(Structure):
    _fields_ = [
        ("model_path", c_char_p),
        ("clip_l_path", c_char_p),
        ("clip_g_path", c_char_p),
        ("clip_vision_path", c_char_p),
        ("t5xxl_path", c_char_p),
        ("llm_path", c_char_p),
        ("llm_vision_path", c_char_p),
        ("diffusion_model_path", c_char_p),
        ("high_noise_diffusion_model_path", c_char_p),
        ("uncond_diffusion_model_path", c_char_p),
        ("embeddings_connectors_path", c_char_p),
        ("vae_path", c_char_p),
        ("audio_vae_path", c_char_p),
        ("audio_encoder_path", c_char_p),
        ("taesd_path", c_char_p),
        ("control_net_path", c_char_p),
        ("ip_adapter_path", c_char_p),
        ("motion_module_path", c_char_p),
        ("embeddings", POINTER(sd_embedding_t)),
        ("embedding_count", c_uint32),
        ("photo_maker_path", c_char_p),
        ("pulid_weights_path", c_char_p),
        ("tensor_type_rules", c_char_p),
        ("n_threads", c_int),
        ("wtype", _enum),
        ("rng_type", _enum),
        ("sampler_rng_type", _enum),
        ("prediction", _enum),
        ("lora_apply_mode", _enum),
        ("enable_mmap", c_bool),
        ("flash_attn", c_bool),
        ("diffusion_flash_attn", c_bool),
        ("tae_preview_only", c_bool),
        ("diffusion_conv_direct", c_bool),
        ("vae_conv_direct", c_bool),
        ("force_sdxl_vae_conv_scale", c_bool),
        ("vae_format", _enum),
        ("max_vram", c_char_p),
        ("disable_prefetch", c_bool),
        ("eager_load", c_bool),
        ("backend", c_char_p),
        ("params_backend", c_char_p),
        ("split_mode", c_char_p),
        ("auto_fit", c_bool),
        ("rpc_servers", c_char_p),
        ("model_args", c_char_p),
        ("disable_segmented_compute", c_bool),
        ("linear_scale", c_float),
        ("attn_scale", c_float),
        ("tokenizer", c_char_p),
        ("sage_attn", c_bool),
        ("conditioning_cache_size", c_int),
    ]


class sd_audio_t(Structure):
    _fields_ = [
        ("sample_rate", c_uint32),
        ("channels", c_uint32),
        ("sample_count", c_uint64),
        ("data", POINTER(c_float)),
    ]


class sd_image_t(Structure):
    _fields_ = [
        ("width", c_uint32),
        ("height", c_uint32),
        ("channel", c_uint32),
        ("data", POINTER(c_uint8)),
    ]


class sd_image_preprocess_params_t(Structure):
    _fields_ = [("rules", c_char_p)]


class sd_ref_video_t(Structure):
    _fields_ = [
        ("frames", POINTER(sd_image_t)),
        ("frame_count", c_int),
        ("fps", c_int),
        ("audio", sd_audio_t),
    ]


class sd_slg_params_t(Structure):
    _fields_ = [
        ("layers", POINTER(c_int)),
        ("layer_count", c_size_t),
        ("layer_start", c_float),
        ("layer_end", c_float),
        ("scale", c_float),
    ]


class sd_guidance_params_t(Structure):
    _fields_ = [
        ("txt_cfg", c_float),
        ("img_cfg", c_float),
        ("distilled_guidance", c_float),
        ("slg", sd_slg_params_t),
    ]


class sd_sample_params_t(Structure):
    _fields_ = [
        ("guidance", sd_guidance_params_t),
        ("scheduler", _enum),
        ("sample_method", _enum),
        ("sample_steps", c_int),
        ("eta", c_float),
        ("shifted_timestep", c_int),
        ("custom_sigmas", POINTER(c_float)),
        ("custom_sigmas_count", c_int),
        ("flow_shift", c_float),
        ("extra_sample_args", c_char_p),
    ]


class sd_pm_params_t(Structure):
    _fields_ = [
        ("id_images", POINTER(sd_image_t)),
        ("id_images_count", c_int),
        ("id_embed_path", c_char_p),
        ("style_strength", c_float),
    ]


class sd_pulid_params_t(Structure):
    _fields_ = [("id_embedding_path", c_char_p), ("id_weight", c_float)]


class sd_cache_params_t(Structure):
    _fields_ = [
        ("mode", _enum),
        ("reuse_threshold", c_float),
        ("start_percent", c_float),
        ("end_percent", c_float),
        ("error_decay_rate", c_float),
        ("use_relative_threshold", c_bool),
        ("reset_error_on_compute", c_bool),
        ("Fn_compute_blocks", c_int),
        ("Bn_compute_blocks", c_int),
        ("residual_diff_threshold", c_float),
        ("max_warmup_steps", c_int),
        ("max_cached_steps", c_int),
        ("max_continuous_cached_steps", c_int),
        ("taylorseer_n_derivatives", c_int),
        ("taylorseer_skip_interval", c_int),
        ("scm_mask", c_char_p),
        ("scm_policy_dynamic", c_bool),
        ("spectrum_w", c_float),
        ("spectrum_m", c_int),
        ("spectrum_lam", c_float),
        ("spectrum_window_size", c_int),
        ("spectrum_flex_window", c_float),
        ("spectrum_warmup_steps", c_int),
        ("spectrum_stop_percent", c_float),
    ]


class sd_lora_t(Structure):
    _fields_ = [
        ("is_high_noise", c_bool),
        ("multiplier", c_float),
        ("path", c_char_p),
    ]


class sd_hires_params_t(Structure):
    _fields_ = [
        ("enabled", c_bool),
        ("upscaler", _enum),
        ("model_path", c_char_p),
        ("scale", c_float),
        ("target_width", c_int),
        ("target_height", c_int),
        ("steps", c_int),
        ("denoising_strength", c_float),
        ("upscale_tile_size", c_int),
        ("custom_sigmas", POINTER(c_float)),
        ("custom_sigmas_count", c_int),
    ]


class sd_img_gen_params_t(Structure):
    _fields_ = [
        ("loras", POINTER(sd_lora_t)),
        ("lora_count", c_uint32),
        ("prompt", c_char_p),
        ("negative_prompt", c_char_p),
        ("clip_skip", c_int),
        ("init_image", sd_image_t),
        ("ref_images", POINTER(sd_image_t)),
        ("ref_images_count", c_int),
        ("ref_image_args", c_char_p),
        ("mask_image", sd_image_t),
        ("width", c_int),
        ("height", c_int),
        ("sample_params", sd_sample_params_t),
        ("strength", c_float),
        ("seed", c_int64),
        ("batch_count", c_int),
        ("control_image", sd_image_t),
        ("control_strength", c_float),
        ("ip_adapter_image", sd_image_t),
        ("ip_adapter_strength", c_float),
        ("pm_params", sd_pm_params_t),
        ("pulid_params", sd_pulid_params_t),
        ("vae_tiling_params", sd_tiling_params_t),
        ("cache", sd_cache_params_t),
        ("hires", sd_hires_params_t),
        ("qwen_image_layers", c_int),
        ("circular_x", c_bool),
        ("circular_y", c_bool),
        ("image_preprocess", sd_image_preprocess_params_t),
    ]


class sd_vid_gen_params_t(Structure):
    _fields_ = [
        ("loras", POINTER(sd_lora_t)),
        ("lora_count", c_uint32),
        ("prompt", c_char_p),
        ("negative_prompt", c_char_p),
        ("clip_skip", c_int),
        ("init_image", sd_image_t),
        ("end_image", sd_image_t),
        ("ref_images", POINTER(sd_image_t)),
        ("ref_images_count", c_int),
        ("ref_videos", POINTER(sd_ref_video_t)),
        ("ref_videos_count", c_int),
        ("ref_audios", POINTER(sd_audio_t)),
        ("ref_audios_count", c_int),
        ("control_frames", POINTER(sd_image_t)),
        ("control_frames_size", c_int),
        ("width", c_int),
        ("height", c_int),
        ("sample_params", sd_sample_params_t),
        ("high_noise_sample_params", sd_sample_params_t),
        ("moe_boundary", c_float),
        ("strength", c_float),
        ("seed", c_int64),
        ("video_frames", c_int),
        ("fps", c_int),
        ("vace_strength", c_float),
        ("vae_tiling_params", sd_tiling_params_t),
        ("cache", sd_cache_params_t),
        ("hires", sd_hires_params_t),
        ("circular_x", c_bool),
        ("circular_y", c_bool),
        ("image_preprocess", sd_image_preprocess_params_t),
    ]


SD_LOG_CB = ctypes.CFUNCTYPE(None, c_int, c_char_p, c_void_p)
SD_PROGRESS_CB = ctypes.CFUNCTYPE(None, c_int, c_int, c_float, c_void_p)

# Bytes reserved past ctypes' sizeof for every struct the library initialises,
# so a library whose struct grew writes into padding instead of past the buffer
# before verify_abi can refuse it.
_INIT_PAD = 4096


class AbiMismatch(RuntimeError):
    """The loaded library's struct layout or commit is not the one bound here."""


def lib_filename() -> str:
    """The stable-diffusion library filename for this platform."""
    if sys.platform == "win32":
        return "stable-diffusion.dll"
    if sys.platform == "darwin":
        return "libstable-diffusion.dylib"
    return "libstable-diffusion.so"


def _ggml_filename() -> str:
    if sys.platform == "win32":
        return "ggml.dll"
    if sys.platform == "darwin":
        return "libggml.dylib"
    return "libggml.so"


def _declare(lib: ctypes.CDLL) -> None:
    """Set argtypes/restype for every function this module calls."""
    def fn(name, restype, *argtypes):
        f = getattr(lib, name)
        f.restype = restype
        f.argtypes = list(argtypes)

    fn("sd_set_log_callback", None, SD_LOG_CB, c_void_p)
    fn("sd_set_progress_callback", None, SD_PROGRESS_CB, c_void_p)
    fn("sd_commit", c_char_p)
    fn("sd_version", c_char_p)
    fn("sd_get_system_info", c_char_p)
    fn("sd_list_devices", c_size_t, c_char_p, c_size_t)
    fn("sd_ctx_params_init", None, c_void_p)
    fn("sd_img_gen_params_init", None, c_void_p)
    fn("sd_vid_gen_params_init", None, c_void_p)
    fn("new_sd_ctx", c_void_p, POINTER(sd_ctx_params_t))
    fn("free_sd_ctx", None, c_void_p)
    fn("sd_get_model_version_name", c_char_p, c_void_p)
    fn("sd_ctx_supports_image_generation", c_bool, c_void_p)
    fn("sd_ctx_supports_video_generation", c_bool, c_void_p)
    fn("sd_get_default_sample_method", _enum, c_void_p)
    fn("sd_get_default_scheduler", _enum, c_void_p, _enum)
    fn("str_to_sample_method", _enum, c_char_p)
    fn("str_to_scheduler", _enum, c_char_p)
    fn("sd_sample_method_name", c_char_p, _enum)
    fn("sd_scheduler_name", c_char_p, _enum)
    fn("generate_image", c_bool, c_void_p, POINTER(sd_img_gen_params_t),
       POINTER(POINTER(sd_image_t)), POINTER(c_int))
    fn("generate_video", c_bool, c_void_p, POINTER(sd_vid_gen_params_t),
       POINTER(POINTER(sd_image_t)), POINTER(c_int),
       POINTER(POINTER(sd_audio_t)), POINTER(c_int))
    fn("sd_cancel_generation", None, c_void_p, _enum)
    fn("free_sd_images", None, POINTER(sd_image_t), c_int)
    fn("free_sd_audio", None, POINTER(sd_audio_t))


def _add_dll_dir(directory: Path) -> None:
    """Make *directory* resolvable for the library's transitive imports."""
    if sys.platform == "win32":
        os.environ["PATH"] = str(directory) + os.pathsep + os.environ.get("PATH", "")
        add = getattr(os, "add_dll_directory", None)
        if add is not None:
            try:
                add(str(directory))
            except OSError:
                pass
    else:
        os.environ["LD_LIBRARY_PATH"] = (
            str(directory) + os.pathsep + os.environ.get("LD_LIBRARY_PATH", ""))


def _register_backends(runtime_dir: Path) -> None:
    """Register the runtime's ggml compute backends from *runtime_dir*.

    A split build (``ggml`` beside the main library, backends as loadable
    modules) is registered with ``ggml_backend_load_all_from_path`` on that
    directory: the library's own default search looks beside the host
    executable, which here is the Python interpreter. A monolithic build has no
    separate ``ggml`` and registers its backends when it loads."""
    ggml_path = runtime_dir / _ggml_filename()
    if not ggml_path.exists():
        return
    mode = 0 if sys.platform == "win32" else ctypes.RTLD_GLOBAL
    ggml = ctypes.CDLL(str(ggml_path), mode=mode)
    count = getattr(ggml, "ggml_backend_dev_count", None)
    if count is not None:
        count.restype = c_size_t
        if int(count()) > 0:
            return
    load_from = getattr(ggml, "ggml_backend_load_all_from_path", None)
    if load_from is None:
        return
    load_from.restype = None
    load_from.argtypes = [c_char_p]
    load_from(str(runtime_dir).encode("utf-8"))


def load_library(runtime_dir: Path, extra_dll_dirs: Optional[list] = None) -> ctypes.CDLL:
    """Load the stable-diffusion library from *runtime_dir*, register its ggml
    backends, declare its functions and run :func:`verify_abi`.

    *extra_dll_dirs*: further directories for transitive imports (the ROCm
    build's hipBLAS/rocBLAS). Raises ``OSError`` when the library or one of its
    imports cannot be loaded, and :class:`AbiMismatch` when the layout check
    fails."""
    runtime_dir = Path(runtime_dir)
    _add_dll_dir(runtime_dir)
    for d in extra_dll_dirs or []:
        _add_dll_dir(Path(d))
    mode = 0 if sys.platform == "win32" else ctypes.RTLD_GLOBAL
    lib = ctypes.CDLL(str(runtime_dir / lib_filename()), mode=mode)
    _register_backends(runtime_dir)
    _declare(lib)
    verify_abi(lib)
    return lib


def _close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=0, abs_tol=1e-4)


def _raw_ptr(struct, name: str) -> int:
    """The address stored in pointer field *name* of *struct*, read without
    dereferencing it (a layout mismatch can leave garbage there)."""
    offset = getattr(type(struct), name).offset
    return c_void_p.from_address(ctypes.addressof(struct) + offset).value or 0


def _init_struct(lib, init_name: str, struct_type):
    """A *struct_type* filled by the library's ``init_name``, as a view over a
    buffer padded by ``_INIT_PAD`` bytes. Passing the view by reference hands the
    library the whole buffer, so any field it has past ``sizeof(struct_type)``
    keeps the value its own init wrote."""
    buf = ctypes.create_string_buffer(ctypes.sizeof(struct_type) + _INIT_PAD)
    getattr(lib, init_name)(ctypes.cast(buf, c_void_p))
    return struct_type.from_buffer(buf)


def _sample_defaults_ok(sp: sd_sample_params_t) -> list[str]:
    bad = []
    if not _close(sp.guidance.txt_cfg, 7.0):
        bad.append(f"sample_params.guidance.txt_cfg={sp.guidance.txt_cfg}")
    if not math.isinf(sp.guidance.img_cfg):
        bad.append(f"sample_params.guidance.img_cfg={sp.guidance.img_cfg}")
    if not _close(sp.guidance.distilled_guidance, 3.5):
        bad.append(f"sample_params.guidance.distilled_guidance={sp.guidance.distilled_guidance}")
    if not _close(sp.guidance.slg.layer_start, 0.01) or not _close(sp.guidance.slg.layer_end, 0.2):
        bad.append("sample_params.guidance.slg")
    if sp.scheduler != _SCHEDULER_COUNT:
        bad.append(f"sample_params.scheduler={sp.scheduler}")
    if sp.sample_method != _SAMPLE_METHOD_COUNT:
        bad.append(f"sample_params.sample_method={sp.sample_method}")
    if not math.isinf(sp.eta) or not math.isinf(sp.flow_shift):
        bad.append("sample_params.eta/flow_shift")
    if _raw_ptr(sp, "extra_sample_args"):
        bad.append("sample_params.extra_sample_args")
    return bad


def _cache_defaults_ok(c: sd_cache_params_t, prefix: str) -> list[str]:
    bad = []
    expected = (("start_percent", 0.15), ("end_percent", 0.95), ("error_decay_rate", 1.0),
                ("residual_diff_threshold", 0.08), ("spectrum_w", 0.40),
                ("spectrum_lam", 1.0), ("spectrum_flex_window", 0.50),
                ("spectrum_stop_percent", 0.9))
    for name, want in expected:
        if not _close(getattr(c, name), want):
            bad.append(f"{prefix}.{name}={getattr(c, name)}")
    ints = (("Fn_compute_blocks", 8), ("max_warmup_steps", 8), ("max_cached_steps", -1),
            ("taylorseer_n_derivatives", 1), ("spectrum_m", 3),
            ("spectrum_window_size", 2), ("spectrum_warmup_steps", 4))
    for name, want in ints:
        if getattr(c, name) != want:
            bad.append(f"{prefix}.{name}={getattr(c, name)}")
    if not (c.use_relative_threshold and c.reset_error_on_compute and c.scm_policy_dynamic):
        bad.append(f"{prefix} bool flags")
    return bad


def _hires_defaults_ok(h: sd_hires_params_t, prefix: str) -> list[str]:
    bad = []
    if h.upscaler != _SD_HIRES_UPSCALER_LATENT:
        bad.append(f"{prefix}.upscaler={h.upscaler}")
    if not _close(h.scale, 2.0) or not _close(h.denoising_strength, 0.7):
        bad.append(f"{prefix}.scale/denoising_strength")
    if h.upscale_tile_size != 128:
        bad.append(f"{prefix}.upscale_tile_size={h.upscale_tile_size}")
    return bad


def check_layouts(lib) -> list[str]:
    """Every field whose value, read through this module's structs after the
    library's own ``*_init``, differs from the default that init writes at the
    pinned commit. Empty when the layouts match."""
    bad: list[str] = []

    cp = _init_struct(lib, "sd_ctx_params_init", sd_ctx_params_t)
    if cp.conditioning_cache_size != 4:
        bad.append(f"ctx.conditioning_cache_size={cp.conditioning_cache_size}")
    if cp.wtype != _SD_TYPE_COUNT:
        bad.append(f"ctx.wtype={cp.wtype}")
    if cp.rng_type != _CUDA_RNG or cp.sampler_rng_type != _RNG_TYPE_COUNT:
        bad.append(f"ctx.rng_type={cp.rng_type}/{cp.sampler_rng_type}")
    if cp.prediction != _PREDICTION_COUNT:
        bad.append(f"ctx.prediction={cp.prediction}")
    if cp.vae_format != _SD_VAE_FORMAT_AUTO:
        bad.append(f"ctx.vae_format={cp.vae_format}")
    if not cp.auto_fit:
        bad.append("ctx.auto_fit")
    if cp.n_threads <= 0:
        bad.append(f"ctx.n_threads={cp.n_threads}")
    if _raw_ptr(cp, "model_path") or _raw_ptr(cp, "tokenizer"):
        bad.append("ctx string pointers not null")

    ip = _init_struct(lib, "sd_img_gen_params_init", sd_img_gen_params_t)
    for name, want in (("clip_skip", -1), ("width", 512), ("height", 512), ("seed", -1),
                       ("batch_count", 1), ("qwen_image_layers", 3), ("ref_images_count", 0)):
        if getattr(ip, name) != want:
            bad.append(f"img.{name}={getattr(ip, name)}")
    for name, want in (("strength", 0.75), ("control_strength", 0.9),
                       ("ip_adapter_strength", 1.0)):
        if not _close(getattr(ip, name), want):
            bad.append(f"img.{name}={getattr(ip, name)}")
    if not _raw_ptr(ip, "ref_image_args"):
        bad.append("img.ref_image_args is null")
    if not _close(ip.pm_params.style_strength, 20.0) or not _close(ip.pulid_params.id_weight, 1.0):
        bad.append("img.pm_params/pulid_params")
    if not _close(ip.vae_tiling_params.target_overlap, 0.5):
        bad.append(f"img.vae_tiling_params.target_overlap={ip.vae_tiling_params.target_overlap}")
    if ip.circular_x or ip.circular_y or _raw_ptr(ip.image_preprocess, "rules"):
        bad.append("img.circular/image_preprocess")
    bad += [f"img.{b}" for b in _sample_defaults_ok(ip.sample_params)]
    if ip.sample_params.sample_steps != 20:
        bad.append(f"img.sample_params.sample_steps={ip.sample_params.sample_steps}")
    bad += _cache_defaults_ok(ip.cache, "img.cache")
    bad += _hires_defaults_ok(ip.hires, "img.hires")

    vp = _init_struct(lib, "sd_vid_gen_params_init", sd_vid_gen_params_t)
    for name, want in (("width", 512), ("height", 512), ("seed", -1), ("video_frames", 6),
                       ("fps", 16)):
        if getattr(vp, name) != want:
            bad.append(f"vid.{name}={getattr(vp, name)}")
    for name, want in (("strength", 0.75), ("moe_boundary", 0.875), ("vace_strength", 1.0)):
        if not _close(getattr(vp, name), want):
            bad.append(f"vid.{name}={getattr(vp, name)}")
    if vp.high_noise_sample_params.sample_steps != -1:
        bad.append(f"vid.high_noise_sample_params.sample_steps="
                   f"{vp.high_noise_sample_params.sample_steps}")
    bad += [f"vid.{b}" for b in _sample_defaults_ok(vp.sample_params)]
    bad += _cache_defaults_ok(vp.cache, "vid.cache")
    bad += _hires_defaults_ok(vp.hires, "vid.hires")
    if not _close(vp.vae_tiling_params.target_overlap, 0.5):
        bad.append("vid.vae_tiling_params.target_overlap")
    return bad


def verify_abi(lib) -> None:
    """Refuse a library that is not the pinned commit or whose struct layouts
    differ from this module's. Raises :class:`AbiMismatch` naming what differs."""
    raw = lib.sd_commit()
    commit = (raw or b"").decode("utf-8", "replace").strip().lower()
    if len(commit) < 7 or not pins.COMMIT.startswith(commit):
        raise AbiMismatch(
            f"stable-diffusion.cpp runtime is commit {commit or '(unknown)'}, but localm "
            f"binds commit {pins.COMMIT[:7]} ({pins.TAG}). Reinstall it with "
            "'localm setup-sdcpp --force'.")
    bad = check_layouts(lib)
    if bad:
        raise AbiMismatch(
            "stable-diffusion.cpp struct layout does not match localm's binding "
            f"({', '.join(bad[:8])}{' ...' if len(bad) > 8 else ''}). Reinstall it with "
            "'localm setup-sdcpp --force'.")


def list_devices(lib) -> list[tuple[str, str]]:
    """``(name, description)`` for every ggml device the runtime can use."""
    need = int(lib.sd_list_devices(None, 0))
    if need <= 0:
        return []
    buf = ctypes.create_string_buffer(need + 1)
    lib.sd_list_devices(buf, need + 1)
    out = []
    for line in buf.value.decode("utf-8", "replace").splitlines():
        if not line.strip():
            continue
        name, _, desc = line.partition("\t")
        out.append((name.strip(), desc.strip()))
    return out
