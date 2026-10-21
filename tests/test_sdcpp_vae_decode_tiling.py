# SPDX-License-Identifier: AGPL-3.0-or-later
"""The native image job log during the VAE decode, and decode tiling on a GPU
runtime that refuses tensors above a size (Vulkan's maxBufferSize): the worker
reports the device limit and the decoder width, the backend picks a tile that
keeps every decode tensor under the limit, and the worker passes it to the
library."""

from __future__ import annotations

import queue

import pytest

from localm import discover
from localm.media.sdcpp import _binding as b
from localm.media.sdcpp import _child, shared, vaetile
from localm.media.sdcpp import budget as sd_budget
from localm.media.sdcpp import runtime as sd_runtime
from localm.plugins.builtin.image.backends import native as image_native

GIB = 1024 ** 3


def _p(phase, step, steps, secs=0.0):
    return ("progress", {"phase": phase, "step": step, "steps": steps, "secs": secs})


PINNED = (b.SD_LOG_WARN, "ggml: ggml_vulkan: Failed to allocate pinned memory")

# Recorded from FLUX.1-dev Q8 1024x1024 on Vulkan (worker events, end of
# sampling to the start of the decode compute): the decoding-phase counts are
# the VAE weight load, after which the decode runs with no further event.
RECORDED_DECODE = [
    _p("sampling", 1, 2, 47.17), _p("sampling", 2, 2, 5.95),
    _p("decoding", 114, 138, 0.0036), _p("decoding", 138, 138, 0.0044),
    ("log", *PINNED), ("log", *PINNED),
]
# The same phase when the worker's rate limit dropped the first decoding count.
RECORDED_DECODE_THROTTLED = [
    _p("sampling", 19, 20, 5.86), _p("sampling", 20, 20, 5.85),
    _p("decoding", 138, 138, 0.0044), ("log", *PINNED), ("log", *PINNED),
]


@pytest.mark.parametrize("events", [RECORDED_DECODE, RECORDED_DECODE_THROTTLED],
                         ids=["recorded", "recorded-throttled"])
def test_the_job_log_says_decoding_until_the_decode_ends(events):
    said = []
    relay = image_native._event_relay(said.append)
    for event in events:
        relay(event)
    after = said[[i for i, line in enumerate(said) if line.startswith("Step ")][-1] + 1:]
    assert "Decoding..." in after
    assert not [line for line in said if "Decoding: done" in line]


def test_the_first_sampling_event_is_never_rate_limited(monkeypatch):
    monkeypatch.setattr(_child._State, "upscaling", False)
    monkeypatch.setattr(_child._State, "decoding", False)
    monkeypatch.setattr(_child._State, "sample_steps", 2)
    monkeypatch.setattr(_child._State, "hires_steps", 0)
    q = queue.Queue()
    _log, progress_cb, _preview = _child._make_callbacks(q)
    progress_cb(219, 219, 0.04, None)
    progress_cb(0, 2, 0.0, None)
    events = [q.get_nowait() for _ in range(q.qsize())]
    assert events[-1] == _p("sampling", 0, 2)


# Recorded (FLUX.1-dev Q8, Vulkan): the first denoiser step streams the DiT's
# weights in blocks, after the step-0 sampling event.
RECORDED_FIRST_STEP = ([_p("loading", 219, 219), _p("sampling", 0, 2)]
                       + [e for _ in range(20) for e in (_p("loading", 16, 24),
                                                         _p("loading", 24, 24))]
                       + [_p("sampling", 1, 2, 46.35)])


def test_weights_streamed_during_the_first_step_add_no_lines():
    said = []
    relay = image_native._event_relay(said.append)
    for event in RECORDED_FIRST_STEP:
        relay(event)
    assert said == ["Loading weights: done", "Step 0/2", "Step 1/2 (46.35 s/step)"]


def test_weight_loads_before_sampling_still_say_done():
    said = []
    relay = image_native._event_relay(said.append)
    for event in (_p("loading", 0, 196), _p("loading", 196, 196), _p("sampling", 0, 2)):
        relay(event)
    assert said[:2] == ["Loading weights...", "Loading weights: done"]


def test_a_hires_pass_says_decoding_again_for_its_decode():
    said = []
    relay = image_native._event_relay(said.append)
    for event in (_p("sampling", 2, 2), _p("decoding", 138, 138), _p("upscaling", 1, 1),
                  _p("sampling", 2, 2), _p("decoding", 138, 138)):
        relay(event)
    assert said.count("Decoding...") == 2


# --------------------------------------------------------------------------- #
#  tile size                                                                  #
# --------------------------------------------------------------------------- #

def test_kl_decoder_peak_bytes_per_pixel():
    # Recorded at 1024x1024 (decoder ch 128): decoder.up.1.upsample.conv runs on a
    # 1 GiB F32 input of 256 channels, so its 3x3 im2col matrix is 9 GiB.
    assert 1024 * 1024 * 256 * 4 == 1 * GIB
    assert vaetile.kl_decode_bytes_per_pixel(128) * 1024 * 1024 == 9 * GIB


def test_a_2_gib_vulkan_limit_tiles_a_1024_decode_under_the_limit():
    limit, bpp = 2 * GIB, vaetile.kl_decode_bytes_per_pixel(128)
    tile = vaetile.decode_tile_size(1024, 1024, limit, 128)
    assert tile == 448
    assert tile * tile * bpp <= limit < (tile + vaetile.TILE_STEP) ** 2 * bpp


@pytest.mark.parametrize("w,h,limit,ch", [
    (448, 448, 2 * GIB, 128),
    (1024, 1024, b.TENSOR_PROBE_CEILING, 128),
    (1024, 1024, None, 128),
    (1024, 1024, 2 * GIB, None),
])
def test_no_tiling_when_the_image_fits_or_a_value_is_unknown(w, h, limit, ch):
    assert vaetile.decode_tile_size(w, h, limit, ch) is None


def test_a_tiny_limit_still_gives_the_smallest_tile():
    assert vaetile.decode_tile_size(1024, 1024, 64 * 1024 ** 2, 128) == vaetile.TILE_STEP


# --------------------------------------------------------------------------- #
#  device probe                                                               #
# --------------------------------------------------------------------------- #

class _Ggml:
    """The ggml functions ``max_tensor_bytes`` calls; ops over *limit* bytes are
    refused, as ggml-vulkan's supports_op does past maxBufferSize."""

    def __init__(self, limit, device="Vulkan0"):
        self.limit, self.device = limit, device
        self.contexts = 0
        self.tensors = {}

        def ggml_init(params):
            assert params.no_alloc
            self.contexts += 1
            return 1000 + self.contexts

        def ggml_free(ctx):
            self.contexts -= 1

        def ggml_new_tensor_1d(ctx, typ, ne):
            assert typ == 0
            key = len(self.tensors) + 1
            self.tensors[key] = ne * 4
            return key

        def ggml_scale(ctx, t, s):
            key = len(self.tensors) + 1
            self.tensors[key] = self.tensors[t]
            return key

        def ggml_backend_dev_by_name(name):
            return 7 if name == self.device.encode() else None

        def ggml_backend_dev_supports_op(dev, op):
            assert dev == 7
            return self.limit is None or self.tensors[op] <= self.limit

        for f in (ggml_init, ggml_free, ggml_new_tensor_1d, ggml_scale,
                  ggml_backend_dev_by_name, ggml_backend_dev_supports_op):
            setattr(self, f.__name__, f)


@pytest.fixture
def ggml(monkeypatch):
    fake = _Ggml(2 * GIB)
    monkeypatch.setattr(b, "_ggml_handles", lambda lib, runtime_dir: [fake])
    return fake


def test_the_probe_finds_the_devices_tensor_limit(ggml, tmp_path):
    assert b.max_tensor_bytes(None, tmp_path, "Vulkan0") == 2 * GIB
    assert ggml.contexts == 0


def test_the_probe_rounds_down_to_its_step(ggml, tmp_path):
    ggml.limit = 2 * GIB - 1
    assert b.max_tensor_bytes(None, tmp_path, "Vulkan0") == 2 * GIB - b._TENSOR_PROBE_STEP


def test_a_device_without_a_limit_reports_the_ceiling(ggml, tmp_path):
    ggml.limit = None
    assert b.max_tensor_bytes(None, tmp_path, "Vulkan0") == b.TENSOR_PROBE_CEILING


def test_an_unknown_device_or_missing_function_gives_none(ggml, tmp_path, monkeypatch):
    assert b.max_tensor_bytes(None, tmp_path, "Vulkan1") is None
    del ggml.ggml_scale
    assert b.max_tensor_bytes(None, tmp_path, "Vulkan0") is None


class _Lib:
    def __init__(self, devices):
        self.devices = devices

    def sd_list_devices(self, buf, size):
        text = "".join(f"{n}\t{d}\n" for n, d in self.devices).encode()
        if buf is not None:
            buf.value = text
        return len(text)


def test_the_worker_reports_the_smallest_gpu_limit(ggml, monkeypatch):
    limits = {"Vulkan0": 2 * GIB, "Vulkan1": 4 * GIB}
    monkeypatch.setattr(b, "max_tensor_bytes", lambda lib, rt, name: limits[name])
    lib = _Lib([("Vulkan0", "a"), ("Vulkan1", "b"), ("CPU", "cpu")])
    assert _child._max_tensor_bytes(lib, "rt") == 2 * GIB


def test_a_probe_error_reports_no_limit(monkeypatch, caplog):
    def broken(lib, rt, name):
        raise RuntimeError("ggml_init failed")
    monkeypatch.setattr(b, "max_tensor_bytes", broken)
    assert _child._max_tensor_bytes(_Lib([("Vulkan0", "a")]), "rt") is None


def test_the_worker_reads_the_decoder_width_from_the_load_log(monkeypatch):
    monkeypatch.setattr(_child._State, "vae_decoder_ch", None)
    log_cb, _progress, _preview = _child._make_callbacks(queue.Queue())
    log_cb(b.SD_LOG_INFO, b"vae decoder: ch = 128 --- auto_encoder_kl.hpp:529", None)
    assert _child._State.vae_decoder_ch == 128


def test_the_worker_passes_the_tile_size_to_the_library():
    p = b.sd_img_gen_params_t()
    _child._set_vae_tiling(p, {"vae_tile_size": 448})
    t = p.vae_tiling_params
    assert (t.enabled, t.tile_size_w, t.tile_size_h) == (True, 448, 448)
    q = b.sd_img_gen_params_t()
    _child._set_vae_tiling(q, {"vae_tile_size": None})
    assert not q.vae_tiling_params.enabled


# --------------------------------------------------------------------------- #
#  backend                                                                    #
# --------------------------------------------------------------------------- #

class _Runner:
    def __init__(self, info):
        self.info = info
        self.loaded_key = self.loaded_ctx = None
        self.alive = False
        self.pid = 1
        self.params = []

    def is_alive(self):
        return self.alive

    def ensure_loaded(self, key, runtime_dir, ctx, **kw):
        self.loaded_key, self.loaded_ctx, self.alive = key, dict(ctx), True
        return dict(self.info)

    def generate_image(self, params, *, on_event=None, cancel_check=None, timeout=None):
        self.params.append(params)
        w, h = params["width"], params["height"]
        return {"width": w, "height": h, "channel": 3, "data": bytes([9, 9, 9]) * (w * h),
                "seed": params["seed"]}

    def shutdown(self, grace=5.0):
        self.alive = False


def _generate(monkeypatch, tmp_path, info, **kw):
    runner = _Runner({"version": "Flux", "image": True, **info})
    monkeypatch.setattr(shared, "runner", runner)
    monkeypatch.setattr(shared, "arm_idle_timer", lambda seconds=None: None)
    monkeypatch.setattr(sd_runtime, "resolve", lambda choice="auto": sd_runtime.Runtime(
        backend="vulkan", path=tmp_path / "rt"))
    monkeypatch.setattr(sd_budget, "_platform", lambda: "linux")
    monkeypatch.setattr(discover, "list_gpus", lambda **k: ([], discover.GPU_PROBE_OK))
    model = tmp_path / "model.gguf"
    model.write_bytes(b"weights")
    said = []
    ok, msg = image_native.generate({"backend": "native", "native": {"model": str(model)}},
                                    "p", tmp_path / "a.png", write_sidecar=False,
                                    on_progress=said.append, **kw)
    assert ok, msg
    return runner.params, said


def test_a_decode_over_the_limit_is_tiled_and_says_so(monkeypatch, tmp_path):
    params, said = _generate(monkeypatch, tmp_path,
                             {"max_tensor_bytes": 2 * GIB, "vae_decoder_ch": 128},
                             width=64, height=64, hires={"upscaler": "lanczos", "scale": 16.0,
                                                         "steps": 1, "denoise": 0.5})
    assert params[0]["vae_tile_size"] == 448
    assert any(line.startswith("Decoding in 448 px tiles") and "1024x1024" in line
               for line in said)


def test_a_decode_that_fits_is_not_tiled(monkeypatch, tmp_path):
    params, said = _generate(monkeypatch, tmp_path,
                             {"max_tensor_bytes": 2 * GIB, "vae_decoder_ch": 128},
                             width=64, height=64)
    assert params[0]["vae_tile_size"] is None
    assert not [line for line in said if "tiles" in line]


def test_a_runtime_without_a_limit_is_not_tiled(monkeypatch, tmp_path):
    params, _said = _generate(monkeypatch, tmp_path,
                              {"max_tensor_bytes": None, "vae_decoder_ch": 128},
                              width=64, height=64, hires={"upscaler": "lanczos", "scale": 16.0,
                                                          "steps": 1, "denoise": 0.5})
    assert params[0]["vae_tile_size"] is None
