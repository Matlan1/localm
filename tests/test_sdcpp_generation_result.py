# SPDX-License-Identifier: AGPL-3.0-or-later
"""The media worker reads stable-diffusion.cpp's generation result structs:
``generate_image``, ``upscale`` and ``generate_video`` each hand back one
library-owned result through ``result_out``, the worker copies the pixels (and
the audio and frame rate of a video) out of it and frees it exactly once with the
matching ``free_sd_*_generation_result``. A failed call (false, or a NULL
result) is an error unless the request was cancelled, and frees nothing it did
not get."""

from __future__ import annotations

import ctypes
import threading

import pytest

from localm.media.sdcpp import _binding as b
from localm.media.sdcpp import _child


def _image(w, h, c, value):
    data = (ctypes.c_uint8 * (w * h * c))(*([value] * (w * h * c)))
    return b.sd_image_t(w, h, c, ctypes.cast(data, ctypes.POINTER(ctypes.c_uint8))), data


class _Lib:
    """The library functions the worker's generate and upscale paths call."""

    def __init__(self, *, ok=True, null_result=False, images=((8, 4, 3, 7),), audio=None,
                 fps=0, on_generate=None):
        self.ok, self.null_result = ok, null_result
        self.images, self.audio, self.fps = images, audio, fps
        self.on_generate = on_generate
        self.freed_image, self.freed_video = [], []
        self.keep = []
        self.received = None

    def sd_img_gen_params_init(self, ptr):
        pass

    def sd_vid_gen_params_init(self, ptr):
        pass

    def sd_cancel_generation(self, ctx, mode):
        pass

    def get_upscaler_model_scale(self, path):
        return 2

    def new_upscaler_ctx(self, *args):
        return 99

    def free_upscaler_ctx(self, uctx):
        pass

    def _image_result(self, ref):
        if self.null_result:
            return
        items = (b.sd_image_result_t * len(self.images))()
        for i, (w, h, c, v) in enumerate(self.images):
            img, data = _image(w, h, c, v)
            self.keep.append(data)
            items[i].image = img
            items[i].generation_info = b'{"schema": "sdcpp.generation/v1"}'
        res = b.sd_image_generation_result_t(
            ctypes.cast(items, ctypes.POINTER(b.sd_image_result_t)), len(self.images))
        self.keep += [items, res]
        ref._obj.contents = res

    def generate_image(self, ctx, params_ref, result_ref):
        self.received = params_ref._obj
        if self.on_generate:
            self.on_generate()
        self._image_result(result_ref)
        return self.ok

    def upscale(self, uctx, src, scale, result_ref):
        self.received = src
        self._image_result(result_ref)
        return self.ok

    def generate_video(self, ctx, params_ref, result_ref):
        self.received = params_ref._obj
        if self.on_generate:
            self.on_generate()
        if self.null_result:
            return self.ok
        frames = (b.sd_image_t * len(self.images))()
        for i, (w, h, c, v) in enumerate(self.images):
            frames[i], data = _image(w, h, c, v)
            self.keep.append(data)
        audio = None
        if self.audio is not None:
            samples = (ctypes.c_float * len(self.audio))(*self.audio)
            audio = b.sd_audio_t(16000, 1, len(self.audio),
                                 ctypes.cast(samples, ctypes.POINTER(ctypes.c_float)))
            self.keep += [samples, audio]
        res = b.sd_video_generation_result_t(
            ctypes.cast(frames, ctypes.POINTER(b.sd_image_t)), len(self.images),
            ctypes.pointer(audio) if audio is not None else None, self.fps, b"{}")
        self.keep += [frames, res]
        result_ref._obj.contents = res
        return self.ok

    def free_sd_image_generation_result(self, ptr):
        self.freed_image.append(ctypes.addressof(ptr.contents))

    def free_sd_video_generation_result(self, ptr):
        self.freed_video.append(ctypes.addressof(ptr.contents))


@pytest.fixture
def worker(monkeypatch):
    def use(lib):
        monkeypatch.setattr(_child._State, "lib", lib)
        monkeypatch.setattr(_child._State, "ctx", 1234)
        monkeypatch.setattr(_child._State, "redact", ())
        monkeypatch.setattr(_child._State, "paths", ())
        monkeypatch.setattr(_child._State, "last_errors", [])
        return lib
    return use


IMAGE_REQUEST = {"prompt": "a fox", "negative_prompt": "", "width": 64, "height": 32,
                 "seed": 5, "steps": 2}
VIDEO_REQUEST = {"prompt": "a fox", "negative_prompt": "", "width": 64, "height": 32,
                 "video_frames": 3, "fps": 12, "seed": 5, "steps": 2}


def test_an_image_result_is_copied_out_and_freed_once(worker):
    lib = worker(_Lib(images=((8, 4, 3, 7), (8, 4, 3, 9))))
    status, out = _child._do_generate_image(dict(IMAGE_REQUEST), threading.Event())
    assert status == "ok"
    assert (out["width"], out["height"], out["channel"]) == (8, 4, 3)
    assert out["data"] == bytes([7]) * (8 * 4 * 3)
    assert out["seed"] == 5
    assert len(lib.freed_image) == 1 and lib.freed_video == []
    assert lib.received.width == 64 and lib.received.height == 32


@pytest.mark.parametrize("ok,null_result,images", [
    (False, False, ((8, 4, 3, 7),)),
    (True, True, ()),
    (True, False, ()),
])
def test_a_failed_image_generation_is_an_error_and_frees_only_what_it_got(
        worker, ok, null_result, images):
    lib = worker(_Lib(ok=ok, null_result=null_result, images=images))
    with pytest.raises(RuntimeError, match="generation failed"):
        _child._do_generate_image(dict(IMAGE_REQUEST), threading.Event())
    assert len(lib.freed_image) == (0 if null_result else 1)


def test_a_cancelled_image_generation_with_no_result_is_cancelled(worker):
    cancel = threading.Event()
    lib = worker(_Lib(ok=False, null_result=True, on_generate=cancel.set))
    assert _child._do_generate_image(dict(IMAGE_REQUEST), cancel) == (
        "cancelled", "Generation cancelled.")
    assert lib.freed_image == []


def test_an_upscale_result_is_copied_out_and_freed_once(worker):
    lib = worker(_Lib(images=((16, 8, 3, 200),)))
    src = {"width": 8, "height": 4, "channel": 3, "data": bytes(8 * 4 * 3)}
    out = _child._do_upscale_image({"model_path": "x.pth", "image": src})
    assert (out["width"], out["height"], out["scale"]) == (16, 8, 2)
    assert out["data"] == bytes([200]) * (16 * 8 * 3)
    assert len(lib.freed_image) == 1


def test_a_failed_upscale_is_an_error(worker):
    lib = worker(_Lib(ok=False, null_result=True))
    src = {"width": 8, "height": 4, "channel": 3, "data": bytes(8 * 4 * 3)}
    with pytest.raises(RuntimeError, match="upscaling failed"):
        _child._do_upscale_image({"model_path": "x.pth", "image": src})
    assert lib.freed_image == []


def test_a_video_result_carries_frames_audio_and_the_library_frame_rate(worker):
    lib = worker(_Lib(images=((8, 4, 3, 1), (8, 4, 3, 2), (8, 4, 3, 3)),
                      audio=[0.5, -0.5], fps=24))
    status, out = _child._do_generate_video(dict(VIDEO_REQUEST), threading.Event())
    assert status == "ok"
    assert [f[0] for f in out["frames"]] == [1, 2, 3]
    assert out["fps"] == 24
    assert out["audio"]["sample_count"] == 2 and len(out["audio"]["data"]) == 8
    assert len(lib.freed_video) == 1 and lib.freed_image == []


def test_a_video_without_a_library_frame_rate_keeps_the_requested_one(worker):
    worker(_Lib(images=((8, 4, 3, 1),), fps=0))
    status, out = _child._do_generate_video(dict(VIDEO_REQUEST), threading.Event())
    assert status == "ok" and out["fps"] == 12 and out["audio"] is None


def test_a_failed_video_generation_is_an_error_and_a_null_result_is_not_freed(worker):
    lib = worker(_Lib(ok=False, null_result=True))
    with pytest.raises(RuntimeError, match="video generation failed"):
        _child._do_generate_video(dict(VIDEO_REQUEST), threading.Event())
    assert lib.freed_video == []


def test_a_video_with_mismatched_frames_is_an_error_and_is_still_freed(worker):
    lib = worker(_Lib(images=((8, 4, 3, 1), (4, 4, 3, 1))))
    with pytest.raises(RuntimeError, match="incomplete frame"):
        _child._do_generate_video(dict(VIDEO_REQUEST), threading.Event())
    assert len(lib.freed_video) == 1
