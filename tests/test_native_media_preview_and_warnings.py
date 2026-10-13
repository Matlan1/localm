# SPDX-License-Identifier: AGPL-3.0-or-later
"""Native media: the live-preview pieces of the worker, the collapse of
recovered runtime warnings in the job log, and the component picker ordering.

The log lines below are the formats stable-diffusion.cpp writes at the pinned
commit (``file:line - message``), in the order a VAE decode that does not fit in
VRAM writes them before it recovers by tiling."""

from __future__ import annotations

import ctypes
import io
import queue
import threading
from pathlib import Path

import pytest

from localm.media.sdcpp import _binding as b
from localm.media.sdcpp import _child, families, logrelay, pairing
from localm.media.sdcpp.runner import SdRunner
from localm.plugins.builtin.image.backends import native as image_native
from localm.plugins.builtin.video.backends import native as video_native

MEMORY_INFO = ("model manager memory on ROCm0: reported free 3705.81 MB / total 16368.00 MB, "
               "tracked weights 12125.39 MB / other runtime 0.00 MB / current runtime 0.00 MB "
               "--- model_manager.cpp:1753")
NO_MEMORY = ("model manager cannot make enough memory available on ROCm0: need 4351.14 MB "
             "device / 3839.14 MB budget, available 3705.81 MB device / 3589.48 MB budget "
             "--- model_manager.cpp:1919")
SEGMENT = "vae segment 1/1 (graph) failed during weight preparation --- ggml_runner.cpp:899"
COMPUTE = "vae decode compute failed --- vae.hpp:377"
REDUCING = "Reducing VAE decode tiles from 768x768 to 256x256 image pixels --- backend_fit.cpp:519"
RETRY = "VAE decode ran out of memory; retrying with spatial tiling --- backend_fit.cpp:523"
TILE_FAILED = "vae decode compute failed while processing a tile --- vae.hpp:363"
RECORDED = [MEMORY_INFO, NO_MEMORY, SEGMENT, COMPUTE, REDUCING, RETRY]
TILES = "Not enough VRAM to decode in one piece; decoding in tiles."


class _DebugLog:
    def __init__(self):
        self.lines = []

    def debug(self, fmt, *args):
        self.lines.append(fmt % args if args else fmt)


@pytest.fixture
def debug_log(monkeypatch):
    log = _DebugLog()
    monkeypatch.setattr("localm.debuglog.logger", log)
    return log


def _relay():
    said = []
    return said, logrelay.RecoveredWarnings(said.append)


# --------------------------------------------------------------------------- #
#  recovered warnings                                                         #
# --------------------------------------------------------------------------- #

def test_the_recorded_vae_recovery_becomes_one_plain_line(debug_log):
    said, relay = _relay()
    for line in RECORDED:
        relay.line(line)
    relay.flush()
    assert said == [TILES]


def test_every_raw_line_is_kept_in_the_debug_log(debug_log):
    said, relay = _relay()
    for line in RECORDED:
        relay.line(line)
    assert debug_log.lines == [f"sd.cpp: {line}" for line in RECORDED]


def test_precursors_without_a_recovery_are_shown_unchanged_when_the_job_ends(debug_log):
    said, relay = _relay()
    for line in (MEMORY_INFO, NO_MEMORY, SEGMENT, COMPUTE):
        relay.line(line)
    assert said == []
    relay.flush()
    assert said == [f"stable-diffusion.cpp: {line}"
                    for line in (MEMORY_INFO, NO_MEMORY, SEGMENT, COMPUTE)]


def test_sampling_resuming_shows_held_lines_so_an_early_warning_is_not_swallowed(debug_log):
    said, relay = _relay()
    relay.line(NO_MEMORY)
    relay.sampling()
    relay.line(RETRY)
    assert said == [f"stable-diffusion.cpp: {NO_MEMORY}", TILES]


def test_an_unrelated_warning_passes_through_after_held_lines(debug_log):
    said, relay = _relay()
    relay.line(COMPUTE)
    relay.line("sd.cpp: something else went wrong")
    assert said == [f"stable-diffusion.cpp: {COMPUTE}",
                    "stable-diffusion.cpp: sd.cpp: something else went wrong"]


def test_a_wan_vae_failure_is_collapsed_too(debug_log):
    said, relay = _relay()
    for line in (NO_MEMORY,
                 "ggml_runner.cpp:898  - wan_vae segment 1/2 (g) failed during compute",
                 "hunyuan_video_vae segment 2/2 (g) failed during weight preparation",
                 COMPUTE, RETRY):
        relay.line(line)
    relay.flush()
    assert said == [TILES]


def test_two_separate_decodes_each_get_the_first_message(debug_log):
    said, relay = _relay()
    for line in RECORDED + RECORDED:
        relay.line(line)
    relay.flush()
    assert said == [TILES, TILES]


def test_a_model_without_a_preview_projection_says_so_once(debug_log):
    said, relay = _relay()
    line = "diffusion_engine.cpp:2017 - No latent to RGB projection known for this model (dim = 32)"
    for _ in range(3):
        relay.line(line)
    assert said == [logrelay.NO_PREVIEW]


def test_a_second_retry_says_the_tiles_are_getting_smaller(debug_log):
    said, relay = _relay()
    for line in RECORDED:
        relay.line(line)
    relay.line(TILE_FAILED)
    relay.line("Reducing VAE decode tiles from 256x256 to 128x128 image pixels "
               "--- backend_fit.cpp:519")
    relay.line(RETRY)
    assert said == [TILES, "Still not enough VRAM; decoding in smaller tiles."]


@pytest.mark.parametrize("mode,words", [("temporal", "time slices"),
                                        ("spatial+temporal", "tiles and time slices")])
def test_temporal_retries_are_worded_for_what_they_do(debug_log, mode, words):
    said, relay = _relay()
    relay.line(f"backend_fit.cpp:522 - VAE decode ran out of memory; retrying with {mode} tiling")
    assert said == [f"Not enough VRAM to decode in one piece; decoding in {words}."]


def test_a_failure_after_the_recovery_still_shows_its_lines(debug_log):
    said, relay = _relay()
    for line in RECORDED:
        relay.line(line)
    relay.line(TILE_FAILED)
    relay.flush()
    assert said == [TILES, f"stable-diffusion.cpp: {TILE_FAILED}"]


@pytest.mark.parametrize("module", [image_native, video_native])
def test_the_backend_relays_collapse_the_recovery_and_flush_on_end(debug_log, module):
    said = []
    relay = module._event_relay(said.append)
    relay(("progress", {"phase": "sampling", "step": 1, "steps": 2, "secs": 0.5}))
    for line in RECORDED:
        relay(("log", 3, line))
    assert [s for s in said if "stable-diffusion.cpp" in s] == []
    assert TILES in said
    relay(("log", 4, COMPUTE))
    relay.flush()
    assert said[-1] == f"stable-diffusion.cpp: {COMPUTE}"


@pytest.mark.parametrize("module", [image_native, video_native])
def test_the_backend_relays_hand_preview_frames_to_the_preview_callback(module):
    said, frames = [], []
    relay = module._event_relay(said.append, frames.append)
    frame = {"step": 4, "steps": 20, "png": b"x"}
    relay(("preview", frame))
    assert frames == [frame] and said == []


@pytest.mark.parametrize("module", [image_native, video_native])
def test_the_backend_relays_ignore_a_preview_nobody_asked_for(module):
    said = []
    module._event_relay(said.append)(("preview", {"step": 1, "steps": 2, "png": b"x"}))
    assert said == []


# --------------------------------------------------------------------------- #
#  worker preview                                                             #
# --------------------------------------------------------------------------- #

def test_preview_interval_gives_about_eight_previews_per_pass():
    assert _child.preview_interval(20) == 3
    assert _child.preview_interval(8) == 1
    assert _child.preview_interval(2) == 1
    assert _child.preview_interval(0) == 1
    assert _child.preview_interval(100) == 13


def test_preview_png_is_a_small_png_of_the_same_aspect():
    from PIL import Image
    data = bytes([10, 200, 30]) * (1024 * 512)
    png = _child.preview_png(1024, 512, 3, data)
    with Image.open(io.BytesIO(png)) as im:
        assert im.format == "PNG"
        assert im.size == (256, 128)
        assert im.getpixel((0, 0)) == (10, 200, 30)


def test_preview_png_does_not_enlarge_a_small_frame():
    from PIL import Image
    png = _child.preview_png(16, 16, 3, bytes(16 * 16 * 3))
    with Image.open(io.BytesIO(png)) as im:
        assert im.size == (16, 16)


def test_preview_png_refuses_a_channel_count_it_cannot_show():
    with pytest.raises(ValueError):
        _child.preview_png(4, 4, 2, bytes(32))


@pytest.fixture
def worker_state(monkeypatch):
    monkeypatch.setattr(_child._State, "generating", True)
    monkeypatch.setattr(_child._State, "sample_steps", 20)
    monkeypatch.setattr(_child._State, "preview_failed", False)


def _frames(width, height, channel, count=1, fill=77):
    bufs = [(ctypes.c_uint8 * (width * height * channel))(*([fill] * (width * height * channel)))
            for _ in range(count)]
    arr = (b.sd_image_t * count)()
    for i, buf in enumerate(bufs):
        arr[i] = b.sd_image_t(width, height, channel,
                              ctypes.cast(buf, ctypes.POINTER(ctypes.c_uint8)))
    return arr, bufs


def test_the_native_preview_callback_queues_a_png_with_the_step(worker_state):
    from PIL import Image
    q = queue.Queue()
    _log, _progress, preview_cb = _child._make_callbacks(q)
    arr, keep = _frames(64, 48, 3)
    preview_cb(6, 1, arr, False, None)
    kind, info = q.get_nowait()
    assert kind == "preview"
    assert (info["step"], info["steps"], info["width"], info["height"]) == (6, 20, 64, 48)
    with Image.open(io.BytesIO(info["png"])) as im:
        assert im.size == (64, 48) and im.getpixel((1, 1)) == (77, 77, 77)
    del keep


def test_a_video_preview_shows_the_middle_frame(worker_state):
    from PIL import Image
    q = queue.Queue()
    _log, _progress, preview_cb = _child._make_callbacks(q)
    arr, keep = _frames(8, 8, 3, count=3)
    for i, fill in enumerate((10, 20, 30)):
        ctypes.memset(arr[i].data, fill, 8 * 8 * 3)
    preview_cb(3, 3, arr, False, None)
    _kind, info = q.get_nowait()
    with Image.open(io.BytesIO(info["png"])) as im:
        assert im.getpixel((0, 0)) == (20, 20, 20)
    del keep


def test_a_second_order_sampler_preview_reports_a_positive_step(worker_state):
    q = queue.Queue()
    _log, _progress, preview_cb = _child._make_callbacks(q)
    arr, keep = _frames(8, 8, 3)
    preview_cb(-3, 1, arr, False, None)
    assert q.get_nowait()[1]["step"] == 3
    del keep


def test_no_preview_is_queued_outside_a_generation(monkeypatch):
    monkeypatch.setattr(_child._State, "generating", False)
    q = queue.Queue()
    _log, _progress, preview_cb = _child._make_callbacks(q)
    arr, keep = _frames(8, 8, 3)
    preview_cb(1, 1, arr, False, None)
    assert q.empty()
    del keep


def test_a_preview_that_cannot_be_made_is_logged_once_and_never_raises(worker_state, monkeypatch):
    warnings = []

    class _Log:
        def warning(self, fmt, *args):
            warnings.append(fmt % args)

    monkeypatch.setattr("localm.debuglog.logger", _Log())
    q = queue.Queue()
    _log, _progress, preview_cb = _child._make_callbacks(q)
    arr, keep = _frames(8, 8, 2)
    preview_cb(1, 1, arr, False, None)
    preview_cb(2, 1, arr, False, None)
    assert q.empty()
    assert len(warnings) == 1 and "live preview" in warnings[0]
    del keep


class _FakeLib:
    def __init__(self):
        self.calls = []

    def sd_set_preview_callback(self, cb, mode, interval, denoised, noisy, data):
        self.calls.append((cb, mode, interval, denoised, noisy, data))


class _RecordingLib:
    """Records, in order, the native calls a generation makes."""

    def __init__(self, ok=False):
        self.events = []
        self.ok = ok

    def __getattr__(self, name):
        def call(*args):
            self.events.append(name)
            if name == "sd_set_preview_callback":
                self.events[-1] = ("preview", bool(args[0]), args[1])
            if name in ("generate_image", "generate_video"):
                return self.ok
        return call


@pytest.mark.parametrize("runner_name,payload", [
    ("_do_generate_image", {"prompt": "p", "width": 64, "height": 64, "steps": 8}),
    ("_do_generate_video", {"prompt": "p", "width": 64, "height": 64, "video_frames": 5,
                            "steps": 8}),
])
def test_a_generation_arms_the_preview_before_and_disarms_it_after(monkeypatch, runner_name,
                                                                   payload):
    lib = _RecordingLib()
    monkeypatch.setattr(_child._State, "lib", lib)
    monkeypatch.setattr(_child._State, "ctx", 1)
    monkeypatch.setattr(_child._State, "preview_cb", object())
    with pytest.raises(RuntimeError):
        getattr(_child, runner_name)({**payload, "preview": True}, threading.Event())
    gen = "generate_image" if runner_name == "_do_generate_image" else "generate_video"
    events = [e for e in lib.events if e in (gen,) or isinstance(e, tuple)]
    assert events == [("preview", True, b.PREVIEW_PROJ), gen, ("preview", False, b.PREVIEW_NONE)]


def test_a_generation_that_did_not_ask_for_previews_never_touches_them(monkeypatch):
    lib = _RecordingLib()
    monkeypatch.setattr(_child._State, "lib", lib)
    monkeypatch.setattr(_child._State, "ctx", 1)
    monkeypatch.setattr(_child._State, "preview_cb", object())
    with pytest.raises(RuntimeError):
        _child._do_generate_image({"prompt": "p", "width": 64, "height": 64, "steps": 8},
                                  threading.Event())
    assert not [e for e in lib.events if isinstance(e, tuple)]


def test_the_library_function_is_declared_with_the_header_signature():
    from unittest.mock import MagicMock
    lib = MagicMock()
    b._declare(lib)
    fn = lib.sd_set_preview_callback
    assert fn.argtypes == [b.SD_PREVIEW_CB, ctypes.c_int, ctypes.c_int, ctypes.c_bool,
                           ctypes.c_bool, ctypes.c_void_p]
    assert fn.restype is None


_PREVIEW_SETTER = ctypes.CFUNCTYPE(None, b.SD_PREVIEW_CB, ctypes.c_int, ctypes.c_int,
                                   ctypes.c_bool, ctypes.c_bool, ctypes.c_void_p)


class _TypedLib:
    """A library whose preview setter converts its arguments the way the real one does."""

    def __init__(self):
        self.seen = []
        self.sd_set_preview_callback = _PREVIEW_SETTER(
            lambda cb, mode, interval, denoised, noisy, data:
            self.seen.append((bool(cb), mode, interval)))


def test_disarming_passes_a_null_function_pointer_the_ffi_accepts():
    lib = _TypedLib()
    _child._disarm_preview(lib)
    assert lib.seen == [(False, b.PREVIEW_NONE, 0)]


def test_arming_converts_through_the_ffi_prototype(monkeypatch):
    lib = _TypedLib()
    monkeypatch.setattr(_child._State, "preview_cb", b.SD_PREVIEW_CB(lambda *a: None))
    assert _child._arm_preview(lib, {"preview": True}, 20) is True
    assert lib.seen == [(True, b.PREVIEW_PROJ, 3)]


def test_python_none_is_not_a_callback_the_ffi_accepts():
    lib = _TypedLib()
    with pytest.raises(ctypes.ArgumentError):
        lib.sd_set_preview_callback(None, 0, 0, False, False, None)


def test_arming_the_preview_asks_the_library_for_projection_frames(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(_child._State, "preview_cb", sentinel)
    lib = _FakeLib()
    assert _child._arm_preview(lib, {"preview": True}, 20) is True
    assert lib.calls == [(sentinel, b.PREVIEW_PROJ, 3, True, False, None)]
    _child._disarm_preview(lib)
    assert not lib.calls[-1][0]
    assert lib.calls[-1][1:] == (b.PREVIEW_NONE, 0, False, False, None)


def test_a_request_without_preview_leaves_the_library_alone(monkeypatch):
    monkeypatch.setattr(_child._State, "preview_cb", object())
    lib = _FakeLib()
    assert _child._arm_preview(lib, {"preview": False}, 20) is False
    assert _child._arm_preview(lib, {}, 20) is False
    assert lib.calls == []


def test_the_preview_callback_signature_matches_the_pinned_header():
    assert b.SD_PREVIEW_CB._argtypes_ == (ctypes.c_int, ctypes.c_int,
                                          ctypes.POINTER(b.sd_image_t), ctypes.c_bool,
                                          ctypes.c_void_p)
    assert (b.PREVIEW_NONE, b.PREVIEW_PROJ, b.PREVIEW_TAE, b.PREVIEW_VAE) == (0, 1, 2, 3)


# --------------------------------------------------------------------------- #
#  runner                                                                     #
# --------------------------------------------------------------------------- #

class _AliveProc:
    def is_alive(self):
        return True


def test_the_runner_relays_preview_events_and_returns_the_final_reply():
    runner = SdRunner()
    runner._proc = _AliveProc()
    runner._resp_q = queue.Queue()
    runner._cancel = threading.Event()
    frame = {"step": 3, "steps": 20, "png": b"p"}
    runner._resp_q.put(("preview", frame))
    runner._resp_q.put(("log", 3, "w"))
    runner._resp_q.put(("ok", {"seed": 1}))
    events = []
    assert runner._wait("generate_image", 5.0, on_event=events.append) == {"seed": 1}
    assert events == [("preview", frame), ("log", 3, "w")]


# --------------------------------------------------------------------------- #
#  component picker order                                                     #
# --------------------------------------------------------------------------- #

def _cand(name, kind, family=None, **detail):
    return pairing.Candidate(name, Path(f"/models/{name}.safetensors"),
                             families.FileInfo(kind, family, detail), "vae")


@pytest.fixture
def vaes():
    return [
        _cand("a-sdxl-vae", "vae", "sdxl", vae_dims="2d", latent=4),
        _cand("wan_2.1_vae", "vae", "wan2.1", vae_dims="3d", latent=16),
        _cand("flux-ae", "vae", "flux", vae_dims="2d", latent=16),
        _cand("ae", "vae", "flux", vae_dims="2d", latent=16),
    ]


def test_fitting_components_come_first_and_the_rest_are_marked(vaes):
    slot = pairing.native_options("image", vaes, {}, family="flux")["slots"]["vae"]
    assert [(o["name"], o["fits"]) for o in slot] == [
        ("ae", True), ("flux-ae", True), ("a-sdxl-vae", False), ("wan_2.1_vae", False)]


def test_without_a_known_family_every_component_is_listed_unmarked(vaes):
    for family in (None, "no-such-family"):
        slot = pairing.native_options("image", vaes, {}, family=family)["slots"]["vae"]
        assert [o["name"] for o in slot] == [c.name for c in vaes]
        assert all("fits" not in o for o in slot)


def test_the_wan22_small_variant_takes_its_own_vae():
    cands = [_cand("vae16", "vae", "wan2.1", vae_dims="3d", latent=16),
             _cand("vae48", "vae", "wan2.2", vae_dims="3d", latent=48)]
    slot = pairing.native_options("video", cands, {}, family="wan2.2", variant="5b")["slots"]["vae"]
    assert [(o["name"], o["fits"]) for o in slot] == [("vae48", True), ("vae16", False)]
