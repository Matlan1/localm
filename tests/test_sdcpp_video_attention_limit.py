# SPDX-License-Identifier: AGPL-3.0-or-later
"""Native video on the Vulkan runtime: the denoiser's self-attention over every
video token is one GPU job, so a clip whose attention job runs past the graphics
driver's reset timeout loses the device. The worker measures the GPU's flash
attention rate, the backend predicts the job's length and refuses a clip over
the bound before sampling. Also: an unknown Vulkan buffer limit still tiles the
VAE decode, and the HunyuanVideo VAE decode is tiled under the limit."""

from __future__ import annotations

import queue

import pytest

from localm.media.sdcpp import _binding as b
from localm.media.sdcpp import _child, attention, gpureset, runner as sd_runner, vaetile
from localm.media.sdcpp import runtime as sd_runtime_module
from localm.media.sdcpp.runner import SdWorkerError

from tests.test_sdcpp_gpu_reset_limit import (DMA, GROUP, ON_PCI, RX6900XT, THREAD, Adapter,
                                              FakeHost, dgpu, linux)
from tests.test_sdcpp_vae_decode_tiling import GIB, _generate, _generate_video


@pytest.fixture(autouse=True)
def _reference_box(monkeypatch):
    from tests.test_sdcpp_gpu_reset_limit import pin_rx6900xt
    pin_rx6900xt(monkeypatch)


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
    assert seconds[49] > gpureset.WINDOWS_DEFAULT_TDR_DELAY


def test_the_safe_bound_leaves_half_the_driver_timeout():
    assert attention.SAFE_DISPATCH_SECONDS * 2 <= gpureset.ASSUMED_RESET_SECONDS


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


def test_the_probe_runs_the_requested_head_size(fa, tmp_path):
    b.attention_rate(None, tmp_path, "Vulkan0", 64)
    assert {shape[1] for shape in fa.ops.values()} == {64}
    b.attention_rate(None, tmp_path, "Vulkan0")
    assert 128 in {shape[1] for shape in fa.ops.values()}


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


class _DevGgml:
    """The ggml device-list functions ``default_device`` calls."""

    def __init__(self, devices):
        self.devices = devices
        self.ggml_backend_dev_count = lambda: len(self.devices)
        self.ggml_backend_dev_get = lambda i: i + 1
        self.ggml_backend_dev_type = lambda dev: self.devices[dev - 1][1]
        self.ggml_backend_dev_name = lambda dev: self.devices[dev - 1][0].encode()


@pytest.mark.parametrize("devices,expected", [
    ([("CPU", 0), ("Vulkan0", 2), ("Vulkan1", 1)], "Vulkan1"),
    ([("CPU", 0), ("Vulkan0", 1), ("Vulkan1", 1)], "Vulkan0"),
    ([("Vulkan0", 2), ("CPU", 0)], "Vulkan0"),
    ([("CPU", 0)], None),
])
def test_the_default_device_is_the_first_discrete_then_integrated_gpu(
        monkeypatch, tmp_path, devices, expected):
    monkeypatch.setattr(b, "_ggml_handles", lambda lib, runtime_dir: [_DevGgml(devices)])
    assert b.default_device(None, tmp_path) == expected


def test_the_default_device_is_unknown_without_the_device_functions(monkeypatch, tmp_path):
    fake = _DevGgml([("Vulkan0", 1)])
    del fake.ggml_backend_dev_type
    monkeypatch.setattr(b, "_ggml_handles", lambda lib, runtime_dir: [fake])
    assert b.default_device(None, tmp_path) is None


def test_the_worker_measures_only_the_device_the_model_runs_on(monkeypatch):
    rates = {"Vulkan0": 5e11, "Vulkan1": 2e12}
    monkeypatch.setattr(b, "attention_rate", lambda lib, rt, name, hd=128: rates[name])
    monkeypatch.setattr(b, "default_device", lambda lib, rt: "Vulkan1")
    monkeypatch.setattr(_child._State, "lib", object())
    assert _child._do_attention_rate({"runtime_dir": "rt"}) == 2e12
    rates["Vulkan0"] = None
    assert _child._do_attention_rate({"runtime_dir": "rt"}) == 2e12


def test_the_worker_reports_the_slowest_gpu(monkeypatch):
    rates = {"Vulkan0": 2e12, "Vulkan1": 5e11}
    monkeypatch.setattr(b, "attention_rate", lambda lib, rt, name, hd=128: rates[name])
    monkeypatch.setattr(b, "default_device", lambda lib, rt: None)
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
    assert calls == [("attention_rate", {"runtime_dir": "rt", "head_dim": 128})]
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
           info=None, width=832, height=480, host=None):
    from tests import test_sdcpp_vae_decode_tiling as w28
    asked = []

    def attention_rate(self, runtime_dir, head_dim=128, timeout=None, devices=None):
        asked.append((str(runtime_dir), devices))
        if isinstance(rate, Exception):
            raise rate
        return rate
    monkeypatch.setattr(w28._VideoRunner, "attention_rate", attention_rate)
    if host is not None:
        monkeypatch.setattr(gpureset, "_HOST", host)
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
    assert "Windows resets this GPU when one job runs longer than 2 s" in msg
    assert msg.endswith("or pick the ROCm runtime in Settings > Video.")


def test_a_clip_just_over_the_bound_names_its_time_and_the_bound(monkeypatch, tmp_path):
    ok, msg, params, _said, _asked = _video(monkeypatch, tmp_path, 21)
    assert not ok and params == []
    assert "about 1.2 s" in msg and "under 1 s" in msg and "longer than 2 s" in msg


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
    assert "could not measure" in msg and "Pick the ROCm runtime in Settings > Video." in msg
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


# --------------------------------------------------------------------------- #
#  The time limit follows what can reset the device                            #
# --------------------------------------------------------------------------- #

ON_RX6900XT = {"compute_devices": [RX6900XT]}
OTHER_GPU_RUNTIMES = sd_runtime_module.other_gpu_runtimes


@pytest.mark.parametrize("granularity", [GROUP, THREAD])
def test_a_gpu_windows_pauses_mid_dispatch_runs_any_clip_without_measuring(
        monkeypatch, tmp_path, granularity):
    host = FakeHost(adapters=[dgpu(granularity, name="Intel(R) Arc(TM) A770 Graphics")])
    ok, msg, params, _said, asked = _video(monkeypatch, tmp_path, 81, rate=0.2e12,
                                           info=ON_RX6900XT, host=host)
    assert ok, msg
    assert params[0]["video_frames"] == 81
    assert asked == []


def test_an_intel_igpu_windows_pauses_mid_dispatch_runs_the_default_clip(monkeypatch, tmp_path):
    igpu = {"name": "Vulkan0", "description": "Intel(R) UHD Graphics", "type": "igpu",
            "device_id": "0000:00:02.0"}
    host = FakeHost(adapters=[Adapter(0, 2, 0, "Intel(R) UHD Graphics", False, THREAD)])
    ok, msg, _params, _said, asked = _video(monkeypatch, tmp_path, 17, rate=0.2e12,
                                            info={"compute_devices": [igpu]}, host=host)
    assert ok, msg
    assert asked == []


def test_this_box_still_refuses_49_frames(monkeypatch, tmp_path):
    host = FakeHost(adapters=[dgpu(DMA)])
    ok, msg, params, _said, asked = _video(monkeypatch, tmp_path, 49, info=ON_RX6900XT,
                                           host=host)
    assert not ok and params == []
    assert asked == [(str(tmp_path / "rt"), ["Vulkan0"])]
    assert "only between command buffers" in msg


def test_a_longer_tdr_delay_allows_a_longer_clip(monkeypatch, tmp_path):
    host = FakeHost(tdr={"TdrDelay": 12}, adapters=[dgpu(DMA)])
    ok, msg, _params, _said, _asked = _video(monkeypatch, tmp_path, 49, info=ON_RX6900XT,
                                             host=host)
    assert ok, msg
    ok, msg, _params, _said, _asked = _video(monkeypatch, tmp_path, 81, info=ON_RX6900XT,
                                             host=host)
    assert not ok and "under 6 s" in msg and "longer than 12 s" in msg


def test_linux_amdgpu_is_bounded_by_its_job_timeout(monkeypatch, tmp_path):
    host = FakeHost(**linux("amdgpu", release="6.8.0-45-generic"))
    ok, msg, _params, _said, _asked = _video(monkeypatch, tmp_path, 33,
                                             info={"compute_devices": [ON_PCI]}, host=host)
    assert ok, msg
    ok, msg, params, _said, _asked = _video(monkeypatch, tmp_path, 49,
                                            info={"compute_devices": [ON_PCI]}, host=host)
    assert not ok and params == []
    assert "amdgpu driver resets this GPU when one job runs longer than 10 s" in msg
    assert "under 5 s" in msg and "at most 45 frames" in msg


def test_linux_amdgpu_on_6_19_is_bounded_like_windows(monkeypatch, tmp_path):
    host = FakeHost(**linux("amdgpu", release="6.19.1"))
    ok, msg, _params, _said, _asked = _video(monkeypatch, tmp_path, 49,
                                             info={"compute_devices": [ON_PCI]}, host=host)
    assert not ok and "at most 17 frames" in msg and "longer than 2 s" in msg


def test_an_unreadable_limit_keeps_the_conservative_bound_and_says_so(monkeypatch, tmp_path):
    host = FakeHost(**linux("nvidia"))
    ok, msg, _params, said, _asked = _video(monkeypatch, tmp_path, 17,
                                            info={"compute_devices": [ON_PCI]}, host=host)
    assert ok, msg
    assert any(line.startswith("localm could not read how long this GPU may run one job")
               for line in said)
    ok, msg, _params, _said, _asked = _video(monkeypatch, tmp_path, 21,
                                             info={"compute_devices": [ON_PCI]}, host=host)
    assert not ok and "under 1 s" in msg and "assumes about 2 s" in msg


def test_a_readable_limit_is_not_announced(monkeypatch, tmp_path):
    ok, msg, _params, said, _asked = _video(monkeypatch, tmp_path, 17, info=ON_RX6900XT,
                                            host=FakeHost(adapters=[dgpu(DMA)]))
    assert ok, msg
    assert not [line for line in said if "could not read" in line]


def test_a_model_planned_on_the_cpu_is_not_limited(monkeypatch, tmp_path):
    ok, msg, _params, _said, asked = _video(
        monkeypatch, tmp_path, 81,
        info={"compute_devices": [{"name": "CPU", "type": "cpu", "description": "",
                                   "device_id": None}]},
        host=FakeHost(adapters=[dgpu(DMA)]))
    assert ok, msg
    assert asked == []


@pytest.mark.parametrize("runtimes,hint", [
    (["ROCm"], ", or pick the ROCm runtime in Settings > Video."),
    (["CUDA"], ", or pick the CUDA runtime in Settings > Video."),
    (["CUDA", "ROCm"], ", or pick the CUDA or ROCm runtime in Settings > Video."),
])
def test_the_refusal_names_only_the_runtimes_this_machine_has(monkeypatch, tmp_path, runtimes,
                                                             hint):
    from localm.media.sdcpp import runtime as sd_runtime
    monkeypatch.setattr(sd_runtime, "other_gpu_runtimes", lambda det=None: list(runtimes))
    ok, msg, _params, _said, _asked = _video(monkeypatch, tmp_path, 49, info=ON_RX6900XT)
    assert not ok and msg.endswith(hint)


def test_an_intel_only_machine_is_offered_no_other_runtime(monkeypatch, tmp_path):
    from localm.media.sdcpp import runtime as sd_runtime
    monkeypatch.setattr(sd_runtime, "other_gpu_runtimes", lambda det=None: [])
    ok, msg, _params, _said, _asked = _video(monkeypatch, tmp_path, 49, info=ON_RX6900XT)
    assert not ok and "runtime in Settings" not in msg
    assert msg.endswith("or lower the size.")
    ok, msg, _params, _said, _asked = _video(monkeypatch, tmp_path, 17, rate=None,
                                             info=ON_RX6900XT)
    assert not ok and "runtime in Settings" not in msg


@pytest.mark.parametrize("vendors,plat,expected", [
    (["amd"], "windows", ["ROCm"]),
    (["nvidia"], "windows", ["CUDA"]),
    (["nvidia", "amd"], "linux", ["ROCm"]),
    (["intel"], "windows", []),
    (["amd"], "macos-arm64", []),
])
def test_other_gpu_runtimes_follow_the_vendors_and_shipped_builds(monkeypatch, vendors, plat,
                                                                  expected):
    from localm.hwdetect import Detection
    monkeypatch.setattr(sd_runtime_module, "platform_key", lambda: plat)
    assert OTHER_GPU_RUNTIMES(Detection(vendors=vendors)) == expected


# --------------------------------------------------------------------------- #
#  The device the model computes on                                            #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("line,expected", [
    (b'auto-fit: --backend "diffusion=Vulkan1,te=Vulkan1,vae=Vulkan1"', ["Vulkan1"]),
    (b'auto-fit: --backend "diffusion=Vulkan0,te=Vulkan1,vae=Vulkan1" --params-backend '
     b'"te=CPU"', ["Vulkan0"]),
    (b'auto-fit: --backend "diffusion=Vulkan0&Vulkan1,te=Vulkan0"', ["Vulkan0", "Vulkan1"]),
    (b'auto-fit: --backend "(default)"', None),
    (b"auto-fit: no GPU memory budget available; using CPU", ["CPU"]),
])
def test_the_worker_reads_the_diffusion_device_from_the_auto_fit_plan(monkeypatch, line,
                                                                       expected):
    monkeypatch.setattr(_child._State, "diffusion_backends", "unset")
    log_cb, _progress, _preview = _child._make_callbacks(queue.Queue())
    log_cb(b.SD_LOG_INFO, line + b" --- backend_fit.cpp:470", None)
    assert _child._State.diffusion_backends == expected


TWO_DGPUS = [
    {"name": "CPU", "description": "cpu", "type": "cpu", "device_id": None},
    {"name": "Vulkan0", "description": "NVIDIA GeForce RTX 4070", "type": "gpu",
     "device_id": "0000:01:00.0"},
    {"name": "Vulkan1", "description": "AMD Radeon RX 6900 XT", "type": "gpu",
     "device_id": "0000:2d:00.0"},
]
IGPU_AND_DGPU = [
    {"name": "Vulkan0", "description": "AMD Radeon(TM) Graphics", "type": "igpu",
     "device_id": "0000:0e:00.0"},
    {"name": "Vulkan1", "description": "AMD Radeon RX 7900 XTX", "type": "gpu",
     "device_id": "0000:03:00.0"},
]


@pytest.mark.parametrize("props,plan,default,expected", [
    (TWO_DGPUS, ["Vulkan1"], "Vulkan0", ["Vulkan1"]),
    (TWO_DGPUS, ["Vulkan0"], "Vulkan0", ["Vulkan0"]),
    (TWO_DGPUS, None, "Vulkan0", ["Vulkan0"]),
    (IGPU_AND_DGPU, ["Vulkan1"], "Vulkan1", ["Vulkan1"]),
    (TWO_DGPUS, ["CPU"], "Vulkan0", ["CPU"]),
    (TWO_DGPUS, None, None, []),
])
def test_the_compute_devices_are_the_planned_ones_else_the_default(monkeypatch, props, plan,
                                                                   default, expected):
    monkeypatch.setattr(b, "device_props", lambda lib, rt: props)
    monkeypatch.setattr(b, "default_device", lambda lib, rt: default)
    monkeypatch.setattr(_child._State, "diffusion_backends", plan)
    got = _child._compute_devices(object(), "rt")
    assert [d["name"] for d in got] == expected
    by_name = {p["name"]: p for p in props}
    assert all(d == by_name[d["name"]] for d in got)


def test_a_planned_device_missing_from_the_props_keeps_its_name(monkeypatch):
    monkeypatch.setattr(b, "device_props", lambda lib, rt: [])
    monkeypatch.setattr(_child._State, "diffusion_backends", ["Vulkan2"])
    assert _child._compute_devices(object(), "rt") == [
        {"name": "Vulkan2", "description": "", "type": None, "device_id": None}]


def test_a_failed_device_probe_gives_no_devices(monkeypatch):
    def boom(lib, rt):
        raise OSError("no ggml")
    monkeypatch.setattr(b, "device_props", boom)
    assert _child._compute_devices(object(), "rt") == []


def test_the_worker_measures_the_devices_it_is_asked_for(monkeypatch):
    rates = {"Vulkan0": 5e12, "Vulkan1": 4e11}
    monkeypatch.setattr(b, "attention_rate", lambda lib, rt, name, hd=128: rates[name])
    monkeypatch.setattr(b, "default_device", lambda lib, rt: "Vulkan0")
    monkeypatch.setattr(_child._State, "lib", object())
    assert _child._do_attention_rate({"runtime_dir": "rt", "devices": ["Vulkan1"]}) == 4e11
    assert _child._do_attention_rate({"runtime_dir": "rt", "devices": ["CPU", "Vulkan1"]}) == 4e11
    assert _child._do_attention_rate({"runtime_dir": "rt",
                                      "devices": ["Vulkan0", "Vulkan1"]}) == 4e11
    assert _child._do_attention_rate({"runtime_dir": "rt", "devices": []}) == 5e12


def test_the_runner_caches_per_device_list(monkeypatch):
    r = sd_runner.SdRunner()
    calls = []

    def request(name, payload, timeout, **kw):
        calls.append(payload)
        return 4e11
    monkeypatch.setattr(r, "_request", request)
    r.attention_rate("rt", devices=["Vulkan1"])
    r.attention_rate("rt", devices=["Vulkan1"])
    r.attention_rate("rt", devices=["Vulkan0"])
    r.attention_rate("rt")
    assert calls == [{"runtime_dir": "rt", "head_dim": 128, "devices": ["Vulkan1"]},
                     {"runtime_dir": "rt", "head_dim": 128, "devices": ["Vulkan0"]},
                     {"runtime_dir": "rt", "head_dim": 128}]


class _PropsGgml:
    """The ggml functions ``device_props`` calls."""

    def __init__(self, devices):
        self.ggml_backend_dev_count = lambda: len(devices)
        self.ggml_backend_dev_get = lambda i: i + 1

        def ggml_backend_dev_get_props(dev, ref):
            name, desc, kind, device_id = devices[dev - 1]
            props = ref._obj
            props.name, props.description = name.encode(), desc.encode()
            props.type, props.device_id = kind, device_id
        self.ggml_backend_dev_get_props = ggml_backend_dev_get_props


def test_device_props_reads_names_types_and_pci_ids(monkeypatch, tmp_path):
    fake = _PropsGgml([("CPU", "AMD Ryzen", 0, None),
                       ("Vulkan0", "AMD Radeon RX 6900 XT", 1, b"0000:2D:00.0"),
                       ("Vulkan1", "AMD Radeon(TM) Graphics", 2, None)])
    monkeypatch.setattr(b, "_ggml_handles", lambda lib, runtime_dir: [fake])
    assert b.device_props(None, tmp_path) == [
        {"name": "CPU", "description": "AMD Ryzen", "type": "cpu", "device_id": None},
        {"name": "Vulkan0", "description": "AMD Radeon RX 6900 XT", "type": "gpu",
         "device_id": "0000:2d:00.0"},
        {"name": "Vulkan1", "description": "AMD Radeon(TM) Graphics", "type": "igpu",
         "device_id": None},
    ]


def test_device_props_is_empty_without_the_functions(monkeypatch, tmp_path):
    fake = _PropsGgml([])
    del fake.ggml_backend_dev_get_props
    monkeypatch.setattr(b, "_ggml_handles", lambda lib, runtime_dir: [fake])
    assert b.device_props(None, tmp_path) == []


def test_device_props_matches_the_ggml_struct_layout():
    import ctypes
    fields = [f[0] for f in b._GgmlDevProps._fields_]
    assert fields == ["name", "description", "memory_free", "memory_total", "type", "device_id",
                      "caps"]
    assert ctypes.sizeof(b._GgmlDevCaps) == 5


def test_two_dgpus_measure_the_card_the_plan_runs_on(monkeypatch, tmp_path):
    host = FakeHost(adapters=[dgpu(THREAD, bus=0x01, name="NVIDIA GeForce RTX 4070"),
                              dgpu(DMA, bus=0x2D)])
    ok, msg, _params, _said, asked = _video(monkeypatch, tmp_path, 49,
                                            info={"compute_devices": [TWO_DGPUS[2]]}, host=host)
    assert not ok and asked == [(str(tmp_path / "rt"), ["Vulkan1"])]
    ok, msg, _params, _said, asked = _video(monkeypatch, tmp_path, 49,
                                            info={"compute_devices": [TWO_DGPUS[1]]}, host=host)
    assert ok, msg
    assert asked == []
