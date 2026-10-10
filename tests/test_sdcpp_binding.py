# SPDX-License-Identifier: AGPL-3.0-or-later
"""The stable-diffusion.cpp ABI check: it must accept a library whose struct
layouts and commit match the binding and refuse one whose layout or commit
differs. The fake library below writes the pinned commit's init defaults through
a chosen layout, the way the real ``*_init`` functions do."""

from __future__ import annotations

import ctypes
import os
from ctypes import Structure, c_int, c_int64
from pathlib import Path

import pytest

from localm.media.sdcpp import _binding as b
from localm.media.sdcpp import pins


def _fill_sample(sp):
    sp.guidance.txt_cfg = 7.0
    sp.guidance.img_cfg = float("inf")
    sp.guidance.distilled_guidance = 3.5
    sp.guidance.slg.layer_start = 0.01
    sp.guidance.slg.layer_end = 0.2
    sp.scheduler = b._SCHEDULER_COUNT
    sp.sample_method = b._SAMPLE_METHOD_COUNT
    sp.sample_steps = 20
    sp.eta = float("inf")
    sp.flow_shift = float("inf")


def _fill_cache(c):
    c.reuse_threshold = float("inf")
    c.start_percent, c.end_percent, c.error_decay_rate = 0.15, 0.95, 1.0
    c.use_relative_threshold = c.reset_error_on_compute = c.scm_policy_dynamic = True
    c.Fn_compute_blocks, c.residual_diff_threshold, c.max_warmup_steps = 8, 0.08, 8
    c.max_cached_steps = c.max_continuous_cached_steps = -1
    c.taylorseer_n_derivatives = c.taylorseer_skip_interval = 1
    c.spectrum_w, c.spectrum_m, c.spectrum_lam = 0.40, 3, 1.0
    c.spectrum_window_size, c.spectrum_flex_window = 2, 0.50
    c.spectrum_warmup_steps, c.spectrum_stop_percent = 4, 0.9


def _fill_hires(h):
    h.upscaler = b._SD_HIRES_UPSCALER_LATENT
    h.scale, h.denoising_strength, h.upscale_tile_size = 2.0, 0.7, 128


def _fill_img(s):
    _fill_sample(s.sample_params)
    s.clip_skip, s.width, s.height, s.seed, s.batch_count = -1, 512, 512, -1, 1
    s.ref_image_args = b""
    s.strength, s.control_strength, s.ip_adapter_strength = 0.75, 0.9, 1.0
    s.qwen_image_layers = 3
    s.pm_params.style_strength = 20.0
    s.pulid_params.id_weight = 1.0
    s.vae_tiling_params.target_overlap = 0.5
    _fill_cache(s.cache)
    _fill_hires(s.hires)


def _fill_vid(s):
    _fill_sample(s.sample_params)
    _fill_sample(s.high_noise_sample_params)
    s.high_noise_sample_params.sample_steps = -1
    s.width, s.height, s.seed, s.video_frames, s.fps = 512, 512, -1, 6, 16
    s.strength, s.moe_boundary, s.vace_strength = 0.75, 0.875, 1.0
    s.vae_tiling_params.target_overlap = 0.5
    _fill_cache(s.cache)
    _fill_hires(s.hires)


def _fill_ctx(s):
    s.n_threads, s.conditioning_cache_size = 6, 4
    s.wtype, s.rng_type, s.sampler_rng_type = b._SD_TYPE_COUNT, b._CUDA_RNG, b._RNG_TYPE_COUNT
    s.prediction, s.vae_format, s.auto_fit = b._PREDICTION_COUNT, b._SD_VAE_FORMAT_AUTO, True


def _with_extra_int_before(struct_type, field_name):
    """*struct_type* with an 8-byte field inserted before *field_name*, so every
    later field moves (a 4-byte one can land in alignment padding and move
    nothing)."""
    fields = []
    for f in struct_type._fields_:
        if f[0] == field_name:
            fields.append(("_inserted", c_int64))
        fields.append(f)
    return type(f"Shifted_{struct_type.__name__}", (Structure,), {"_fields_": fields})


class _FakeLib:
    """Implements the three ``*_init`` functions and ``sd_commit``."""

    def __init__(self, commit=b"f89d9b1", ctx_type=b.sd_ctx_params_t,
                 img_type=b.sd_img_gen_params_t, vid_type=b.sd_vid_gen_params_t):
        self.commit = commit
        self.types = {"ctx": ctx_type, "img": img_type, "vid": vid_type}
        self.keep = []

    def _write(self, kind, ptr, fill):
        s = self.types[kind].from_address(ptr.value)
        fill(s)
        self.keep.append(s)

    def sd_ctx_params_init(self, ptr):
        self._write("ctx", ptr, _fill_ctx)

    def sd_img_gen_params_init(self, ptr):
        self._write("img", ptr, _fill_img)

    def sd_vid_gen_params_init(self, ptr):
        self._write("vid", ptr, _fill_vid)

    def sd_commit(self):
        return self.commit


def test_matching_layout_and_commit_pass():
    lib = _FakeLib()
    assert b.check_layouts(lib) == []
    b.verify_abi(lib)


@pytest.mark.parametrize("kind,struct_type,field,expect", [
    ("img", b.sd_img_gen_params_t, "width", "img.width"),
    ("img", b.sd_img_gen_params_t, "cache", "img.cache"),
    ("ctx", b.sd_ctx_params_t, "conditioning_cache_size", "ctx.conditioning_cache_size"),
    ("ctx", b.sd_ctx_params_t, "n_threads", "ctx."),
    ("vid", b.sd_vid_gen_params_t, "video_frames", "vid.video_frames"),
    ("vid", b.sd_vid_gen_params_t, "sample_params", "vid."),
])
def test_a_shifted_field_is_reported(kind, struct_type, field, expect):
    shifted = _with_extra_int_before(struct_type, field)
    lib = _FakeLib(**{f"{kind}_type": shifted})
    bad = b.check_layouts(lib)
    assert bad, f"inserting an int before {field} went undetected"
    assert any(expect in item for item in bad), bad
    with pytest.raises(b.AbiMismatch, match="struct layout does not match"):
        b.verify_abi(lib)


def test_a_library_struct_larger_than_the_binding_keeps_its_trailing_defaults():
    grown = type("Grown", (Structure,), {"_fields_": list(b.sd_img_gen_params_t._fields_)
                                         + [("tail", c_int)]})

    def fill(s):
        _fill_img(s)
        s.tail = 1234

    lib = _FakeLib(img_type=grown)
    lib.sd_img_gen_params_init = lambda ptr: lib._write("img", ptr, fill)
    view = b._init_struct(lib, "sd_img_gen_params_init", b.sd_img_gen_params_t)
    view.width = 640
    raw = ctypes.string_at(ctypes.addressof(view), ctypes.sizeof(grown))
    back = grown.from_buffer_copy(raw)
    assert back.width == 640
    assert back.tail == 1234


@pytest.mark.parametrize("commit", [b"deadbee", b"", None, b"f89d"])
def test_a_different_commit_is_refused(commit):
    lib = _FakeLib(commit=commit)
    with pytest.raises(b.AbiMismatch, match="localm binds commit f89d9b1"):
        b.verify_abi(lib)


def test_the_pinned_commit_matches_the_tag():
    assert pins.TAG.endswith(pins.COMMIT[:7])


@pytest.mark.integration
def test_layouts_match_a_real_runtime():
    """Set LOCALM_SDCPP_TEST_RUNTIME to an installed runtime directory. The
    library is loaded in a worker process, never in the test process."""
    d = os.environ.get("LOCALM_SDCPP_TEST_RUNTIME")
    if not d:
        pytest.skip("LOCALM_SDCPP_TEST_RUNTIME is not set")
    from localm.media.sdcpp.runner import SdRunner
    info = SdRunner().probe(Path(d))
    assert info["commit"] == pins.COMMIT[:7]
    assert info["devices"]
