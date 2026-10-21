# SPDX-License-Identifier: AGPL-3.0-or-later
"""Native video on the Vulkan runtime: the denoiser's self-attention over every
video token is one GPU job, so a clip whose attention job runs past the graphics
driver's reset timeout loses the device. The worker measures the GPU's flash
attention rate, the backend predicts the job's length and refuses a clip over
the bound before sampling. Also: an unknown Vulkan buffer limit still tiles the
VAE decode, and the HunyuanVideo VAE decode is tiled under the limit."""

from __future__ import annotations

import queue
import threading

import pytest

from localm.media.sdcpp import _binding as b
from localm.media.sdcpp import _child, attention, runner as sd_runner, vaetile
from localm.media.sdcpp.runner import SdWorkerError

from tests.test_sdcpp_vae_decode_tiling import GIB, _Lib, _generate, _generate_video

RX6900XT_VULKAN_RATE = 0.455e12
WAN13 = attention.video_attention("Wan 2.x", "Wan2.1-T2V-1.3B")


# --------------------------------------------------------------------------- #
#  The attention job of a clip                                                 #
# --------------------------------------------------------------------------- #

def test_wan_tokens_follow_the_latent_grid():
    assert WAN13.tokens(832, 480, 17) == 5 * 52 * 30
    assert WAN13.tokens(832, 480, 49) == 13 * 1560
    assert WAN13.tokens(832, 480, 81) == 21 * 1560
    assert WAN13.tokens(832, 480, 1) == 1560


@pytest.mark.parametrize("version,desc,dim,stride", [
    ("Wan 2.x", "Wan2.1-T2V-1.3B", 1536, 16),
    ("Wan 2.x", "Wan2.1-I2V-14B", 5120, 16),
    ("Wan 2.2 I2V", "Wan2.2-I2V-14B", 5120, 16),
    ("Wan 2.2 TI2V", "Wan2.2-TI2V-5B", 3072, 32),
    ("Wan 2.x", "", 5120, 16),
    ("Wan 2.x", "Wan2.x-unknown-7B", 5120, 16),
])
def test_wan_shapes_by_model(version, desc, dim, stride):
    att = attention.video_attention(version, desc)
    assert (att.dim, att.stride, att.text_tokens) == (dim, stride, 0)


def test_hunyuan_shape_comes_from_its_load_log_else_the_largest():
    att = attention.video_attention("Hunyuan Video", "", 2048, 2)
    assert (att.dim, att.stride) == (2048, 32)
    assert att.tokens(848, 480, 1) == 27 * 15 + att.text_tokens
    fallback = attention.video_attention("Hunyuan Video")
    assert (fallback.dim, fallback.stride) == (3072, 16)


def test_an_unknown_video_model_gets_the_largest_shape():
    att = attention.video_attention("LTXAV", "")
    assert att.dim == 5120 and att.stride == 16 and att.text_tokens > 0


def test_measured_rx6900xt_vulkan_times_match_what_the_driver_did():
    seconds = {f: attention.dispatch_seconds(WAN13, 832, 480, f, RX6900XT_VULKAN_RATE)
               for f in (17, 49, 81)}
    assert seconds[17] == pytest.approx(0.82, abs=0.02)
    assert seconds[17] <= attention.SAFE_DISPATCH_SECONDS
    assert seconds[49] == pytest.approx(5.55, abs=0.05)
    assert seconds[81] == pytest.approx(14.5, abs=0.1)
    assert seconds[49] > attention.DRIVER_RESET_SECONDS


def test_the_safe_bound_leaves_half_the_driver_timeout():
    assert attention.SAFE_DISPATCH_SECONDS * 2 <= attention.DRIVER_RESET_SECONDS


@pytest.mark.parametrize("version,desc,w,h,most", [
    ("Wan 2.x", "Wan2.1-T2V-1.3B", 832, 480, 17),
    ("Wan 2.x", "Wan2.1-T2V-1.3B", 480, 320, 53),
    ("Wan 2.x", "Wan2.1-T2V-14B", 832, 480, 9),
    ("Wan 2.2 TI2V", "Wan2.2-TI2V-5B", 1280, 704, 21),
])
def test_max_frames_is_the_longest_clip_within_the_bound(version, desc, w, h, most):
    att = attention.video_attention(version, desc)
    got = attention.max_frames(att, w, h, RX6900XT_VULKAN_RATE)
    assert got == most
    assert (got - 1) % 4 == 0
    assert attention.dispatch_seconds(att, w, h, got, RX6900XT_VULKAN_RATE) \
        <= attention.SAFE_DISPATCH_SECONDS
    assert attention.dispatch_seconds(att, w, h, got + 4, RX6900XT_VULKAN_RATE) \
        > attention.SAFE_DISPATCH_SECONDS


def test_max_frames_is_zero_when_one_frame_is_too_long():
    att = attention.video_attention("Wan 2.x", "Wan2.1-T2V-14B")
    assert attention.max_frames(att, 1920, 1088, 1e9) == 0


# --------------------------------------------------------------------------- #
#  The rate probe                                                              #
# --------------------------------------------------------------------------- #

class _FaGgml:
    """The ggml functions ``attention_rate`` calls. A compute takes
    ``flops / rate`` seconds of a fake clock."""

    def __init__(self, rate, clock, supported=True, status=0):
        self.rate, self.clock, self.supported, self.status = rate, clock, supported, status
        self.tensors, self.ops, self.computed = {}, {}, []
        self.contexts = self.buffers = self.backends = 0

        def ggml_init(params):
            assert params.no_alloc
            self.contexts += 1
            return 100

        def ggml_free(ctx):
            self.contexts -= 1

        def ggml_new_tensor_3d(ctx, typ, ne0, ne1, ne2):
            key = len(self.tensors) + 1
            self.tensors[key] = (typ, ne0, ne1, ne2)
            return key

        def ggml_flash_attn_ext(ctx, q, k, v, mask, scale, max_bias, softcap):
            assert mask is None
            assert [self.tensors[t][0] for t in (q, k, v)] == [0, 1, 1]
            key = 1000 + len(self.ops)
            self.ops[key] = self.tensors[q]
            return key

        def ggml_prec_set_acc(op, prec):
            assert prec == 10
            return True

        def ggml_new_graph(ctx):
            return 200

        def ggml_build_forward_expand(graph, op):
            self.graph_op = op

        def ggml_backend_dev_by_name(name):
            return 7 if name == b"Vulkan0" else None

        def ggml_backend_dev_supports_op(dev, op):
            return self.supported

        def ggml_backend_dev_init(dev, params):
            self.backends += 1
            return 300

        def ggml_backend_free(backend):
            self.backends -= 1

        def ggml_backend_alloc_ctx_tensors(ctx, backend):
            self.buffers += 1
            return 400

        def ggml_backend_buffer_clear(buf, value):
            assert value == 0

        def ggml_backend_buffer_free(buf):
            self.buffers -= 1

        def ggml_backend_graph_compute(backend, graph):
            _typ, d, tokens, heads = self.ops[self.graph_op]
            self.computed.append(tokens)
            self.clock[0] += b.attention_flops(tokens, d * heads) / self.rate
            return self.status

        for f in (ggml_init, ggml_free, ggml_new_tensor_3d, ggml_flash_attn_ext,
                  ggml_prec_set_acc, ggml_new_graph, ggml_build_forward_expand,
                  ggml_backend_dev_by_name, ggml_backend_dev_supports_op, ggml_backend_dev_init,
                  ggml_backend_free, ggml_backend_alloc_ctx_tensors, ggml_backend_buffer_clear,
                  ggml_backend_buffer_free, ggml_backend_graph_compute):
            setattr(self, f.__name__, f)


@pytest.fixture
def fa(monkeypatch):
    import time
    clock = [0.0]
    monkeypatch.setattr(time, "perf_counter", lambda: clock[0])
    fake = _FaGgml(RX6900XT_VULKAN_RATE, clock)
    monkeypatch.setattr(b, "_ggml_handles", lambda lib, runtime_dir: [fake])
    return fake


def test_the_probe_measures_the_flash_attention_rate(fa, tmp_path):
    rate = b.attention_rate(None, tmp_path, "Vulkan0")
    assert rate == pytest.approx(RX6900XT_VULKAN_RATE)
    assert fa.contexts == fa.buffers == fa.backends == 0


def test_the_probe_stops_at_the_first_size_that_takes_long_enough(fa, tmp_path):
    b.attention_rate(None, tmp_path, "Vulkan0")
    sizes = sorted(set(fa.computed))
    assert sizes == [512, 1024, 2048, 4096]
    longest = b.attention_flops(sizes[-1], 12 * 128) / RX6900XT_VULKAN_RATE
    assert b._ATTENTION_PROBE_MIN_SECONDS <= longest < attention.SAFE_DISPATCH_SECONDS


def test_the_probe_on_a_very_slow_gpu_never_runs_a_long_job(fa, tmp_path):
    fa.rate = 5e9
    assert b.attention_rate(None, tmp_path, "Vulkan0") == pytest.approx(5e9)
    assert sorted(set(fa.computed)) == [512]
    assert max(b.attention_flops(n, 12 * 128) / fa.rate for n in fa.computed) \
        < attention.SAFE_DISPATCH_SECONDS


def test_the_probe_on_a_fast_gpu_stops_at_its_largest_size(fa, tmp_path):
    fa.rate = 1e15
    assert b.attention_rate(None, tmp_path, "Vulkan0") == pytest.approx(1e15)
    assert max(fa.computed) == b._ATTENTION_PROBE_MAX_TOKENS


def test_the_probe_gives_none_without_the_functions_device_or_op(fa, tmp_path):
    assert b.attention_rate(None, tmp_path, "Vulkan1") is None
    fa.supported = False
    assert b.attention_rate(None, tmp_path, "Vulkan0") is None
    assert fa.backends == fa.contexts == 0
    del fa.ggml_backend_graph_compute
    assert b.attention_rate(None, tmp_path, "Vulkan0") is None


def test_a_failed_compute_raises_and_frees(fa, tmp_path):
    fa.status = -1
    with pytest.raises(RuntimeError, match="status -1"):
        b.attention_rate(None, tmp_path, "Vulkan0")
    assert fa.contexts == fa.buffers == fa.backends == 0


def test_the_worker_reports_the_slowest_gpu(monkeypatch):
    rates = {"Vulkan0": 2e12, "Vulkan1": 5e11}
    monkeypatch.setattr(b, "attention_rate", lambda lib, rt, name: rates[name])
    monkeypatch.setattr(b, "list_devices", lambda lib: [("Vulkan0", "a"), ("Vulkan1", "b"),
                                                        ("CPU", "cpu")])
    monkeypatch.setattr(_child._State, "lib", object())
    assert _child._do_attention_rate({"runtime_dir": "rt"}) == 5e11
    rates["Vulkan1"] = None
    assert _child._do_attention_rate({"runtime_dir": "rt"}) is None


def test_the_worker_routes_the_attention_rate_command():
    r = sd_runner.SdRunner()
    try:
        r._spawn()
        with pytest.raises(SdWorkerError, match="no stable-diffusion.cpp runtime is loaded"):
            r.attention_rate("rt", timeout=60)
        assert r.is_alive()
    finally:
        r.shutdown(grace=0)


@pytest.mark.integration
def test_a_real_runtime_measures_its_gpu():
    import os
    rt = os.environ.get("LOCALM_SDCPP_TEST_RUNTIME")
    if not rt:
        pytest.skip("LOCALM_SDCPP_TEST_RUNTIME is not set")
    r = sd_runner.SdRunner()
    try:
        r._spawn()
        r._request("probe", {"runtime_dir": rt, "extra_dirs": []}, 120)
        rate = r.attention_rate(rt)
    finally:
        r.shutdown(grace=0)
    assert rate is None or rate > 1e9


def test_the_runner_measures_once_per_worker(monkeypatch):
    r = sd_runner.SdRunner()
    calls = []

    def request(name, payload, timeout, **kw):
        calls.append((name, payload))
        return 4e11
    monkeypatch.setattr(r, "_request", request)
    assert r.attention_rate("rt") == 4e11
    assert r.attention_rate("rt") == 4e11
    assert calls == [("attention_rate", {"runtime_dir": "rt"})]
    r._reset()
    assert r.attention_rate("rt") == 4e11
    assert len(calls) == 2


def test_the_runner_caches_a_rate_that_could_not_be_measured(monkeypatch):
    r = sd_runner.SdRunner()
    calls = []
    monkeypatch.setattr(r, "_request", lambda *a, **k: calls.append(1))
    assert r.attention_rate("rt") is None
    assert r.attention_rate("rt") is None
    assert len(calls) == 1


# --------------------------------------------------------------------------- #
#  The video job                                                               #
# --------------------------------------------------------------------------- #

def _video(monkeypatch, tmp_path, frames, rate=RX6900XT_VULKAN_RATE, backend="vulkan",
           info=None, width=832, height=480):
    from tests import test_sdcpp_vae_decode_tiling as w28
    asked = []

    def attention_rate(self, runtime_dir, timeout=None):
        asked.append(str(runtime_dir))
        if isinstance(rate, Exception):
            raise rate
        return rate
    monkeypatch.setattr(w28._VideoRunner, "attention_rate", attention_rate)
    ok, msg, params, said = _generate_video(
        monkeypatch, tmp_path,
        {"max_tensor_bytes": 2 * GIB, "desc": "Wan2.1-T2V-1.3B", **(info or {})},
        backend=backend, seconds=frames / 16, fps=16, width=width, height=height)
    return ok, msg, params, said, asked


@pytest.mark.parametrize("frames", [49, 81])
def test_a_clip_whose_attention_job_outlasts_the_driver_is_refused_before_sampling(
        monkeypatch, tmp_path, frames):
    ok, msg, params, _said, asked = _video(monkeypatch, tmp_path, frames)
    assert not ok
    assert params == []
    assert asked
    assert f"{frames} frames at 832x480" in msg
    assert "at most 17 frames (1.1 s at 16 fps)" in msg
    assert "about 2 s" in msg and "ROCm or CUDA" in msg


def test_the_longest_safe_clip_runs(monkeypatch, tmp_path):
    ok, msg, params, _said, asked = _video(monkeypatch, tmp_path, 17)
    assert ok, msg
    assert params[0]["video_frames"] == 17
    assert asked


def test_a_fast_gpu_runs_the_default_clip(monkeypatch, tmp_path):
    ok, msg, params, _said, _asked = _video(monkeypatch, tmp_path, 81, rate=40e12)
    assert ok, msg
    assert params[0]["video_frames"] == 81


@pytest.mark.parametrize("backend", ["rocm", "cuda"])
def test_other_runtimes_are_not_measured_or_limited(monkeypatch, tmp_path, backend):
    ok, msg, params, _said, asked = _video(monkeypatch, tmp_path, 81, backend=backend)
    assert ok, msg
    assert params[0]["video_frames"] == 81
    assert asked == []


@pytest.mark.parametrize("rate", [None, 0.0, SdWorkerError("worker died")])
def test_an_unmeasured_rate_refuses_on_vulkan(monkeypatch, tmp_path, rate):
    ok, msg, params, _said, _asked = _video(monkeypatch, tmp_path, 17, rate=rate)
    assert not ok
    assert params == []
    assert "could not measure" in msg and "ROCm or CUDA" in msg
    if isinstance(rate, Exception):
        assert "worker died" in msg


def test_a_14b_model_is_limited_more_than_the_1_3b(monkeypatch, tmp_path):
    ok, msg, params, _said, _asked = _video(monkeypatch, tmp_path, 17,
                                            info={"desc": "Wan2.1-T2V-14B"})
    assert not ok and params == []
    assert "at most 9 frames" in msg


def test_a_size_where_no_clip_fits_says_lower_the_size(monkeypatch, tmp_path):
    ok, msg, params, _said, _asked = _video(monkeypatch, tmp_path, 17, rate=1e9)
    assert not ok and params == []
    assert "Lower the size" in msg and "at most" not in msg


def test_the_worker_reads_the_hunyuan_shape_from_the_load_log(monkeypatch):
    monkeypatch.setattr(_child._State, "dit_hidden", None)
    monkeypatch.setattr(_child._State, "dit_patch", None)
    log_cb, _progress, _preview = _child._make_callbacks(queue.Queue())
    log_cb(b.SD_LOG_VERBOSE, b"hunyuan video: depth = 54, single depth = 0, in_channels = 65, "
           b"out_channels = 32, hidden_size = 2048, context_in_dim = 3584, "
           b"patch_size = 1x2x2 --- hunyuan.hpp:270", None)
    assert (_child._State.dit_hidden, _child._State.dit_patch) == (2048, 2)


# --------------------------------------------------------------------------- #
#  An unknown Vulkan buffer limit still tiles                                  #
# --------------------------------------------------------------------------- #

def test_device_limit_assumes_a_bound_only_on_vulkan():
    assert vaetile.device_limit("vulkan", None) == (vaetile.VULKAN_ASSUMED_TENSOR_BYTES, True)
    assert vaetile.device_limit("vulkan", 2 * GIB) == (2 * GIB, False)
    assert vaetile.device_limit("rocm", None) == (None, False)
    assert vaetile.device_limit("cuda", None) == (None, False)


def test_a_vulkan_video_with_an_unknown_limit_is_tiled_and_says_so(monkeypatch, tmp_path):
    ok, msg, params, said, _asked = _video(monkeypatch, tmp_path, 17,
                                           info={"max_tensor_bytes": None})
    assert ok, msg
    expected = vaetile.wan_decode_tile_size(832, 480, 17, vaetile.VULKAN_ASSUMED_TENSOR_BYTES,
                                            vaetile.wan_vae("Wan 2.x", "3D"))
    assert expected and params[0]["vae_tile_size"] == expected
    assert any("did not report" in line and f"{expected} px tiles" in line for line in said)


def test_a_vulkan_image_with_an_unknown_limit_is_tiled_and_says_so(monkeypatch, tmp_path):
    params, said = _generate(monkeypatch, tmp_path,
                             {"max_tensor_bytes": None, "vae_decoder_ch": 128},
                             width=64, height=64, hires={"upscaler": "lanczos", "scale": 16.0,
                                                         "steps": 1, "denoise": 0.5})
    expected = vaetile.decode_tile_size(1024, 1024, vaetile.VULKAN_ASSUMED_TENSOR_BYTES, 128)
    assert expected and params[0]["vae_tile_size"] == expected
    assert any("did not report" in line for line in said)


# --------------------------------------------------------------------------- #
#  HunyuanVideo VAE decode                                                     #
# --------------------------------------------------------------------------- #

def test_hunyuan_decode_peak_bytes_at_the_default_video_size():
    peak = vaetile.hunyuan_decode_peak_bytes(848, 480, 81)
    assert peak == 128 * 27 * 2 * 848 * 480 * 4
    assert peak > 2 * GIB


def test_hunyuan_mid_attention_grows_with_the_clip():
    short = vaetile.hunyuan_decode_peak_bytes(384, 384, 81)
    long = vaetile.hunyuan_decode_peak_bytes(384, 384, 241)
    assert short == 128 * 27 * 2 * 384 * 384 * 4
    assert long == (61 * 24 * 24) ** 2 * 4
    assert long > short


@pytest.mark.parametrize("w,h,frames", [(848, 480, 81), (1280, 720, 121), (848, 480, 241)])
def test_hunyuan_tiles_keep_every_tensor_under_the_limit(w, h, frames):
    tile = vaetile.hunyuan_decode_tile_size(w, h, frames, 2 * GIB)
    assert tile and tile % vaetile.TILE_STEP == 0
    assert vaetile.hunyuan_decode_peak_bytes(min(tile, w), min(tile, h), frames) <= 2 * GIB
    bigger = tile + vaetile.TILE_STEP
    assert vaetile.hunyuan_decode_peak_bytes(min(bigger, w), min(bigger, h), frames) > 2 * GIB


def test_hunyuan_needs_no_tile_without_a_limit_or_when_it_fits():
    assert vaetile.hunyuan_decode_tile_size(848, 480, 81, None) is None
    assert vaetile.hunyuan_decode_tile_size(128, 128, 5, 2 * GIB) is None


def test_hunyuan_no_tile_fits_a_tiny_limit():
    with pytest.raises(vaetile.DecodeTooLarge):
        vaetile.hunyuan_decode_tile_size(848, 480, 81, 1024 ** 2)


def test_a_hunyuan_video_on_vulkan_is_decoded_in_tiles(monkeypatch, tmp_path):
    ok, msg, params, said, _asked = _video(
        monkeypatch, tmp_path, 33, rate=40e12, width=848, height=480,
        info={"version": "Hunyuan Video", "desc": "", "wan_vae_conv": None,
              "dit_hidden": 2048, "dit_patch": 2})
    assert ok, msg
    expected = vaetile.hunyuan_decode_tile_size(848, 480, 33, 2 * GIB)
    assert expected and params[0]["vae_tile_size"] == expected
    assert any(line.startswith(f"Decoding in {expected} px tiles") for line in said)
