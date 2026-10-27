# SPDX-License-Identifier: AGPL-3.0-or-later
"""What llama.cpp says on stderr while it loads a vision projector reaches the
user as a localm warning: that the projector was placed on the CPU although the
GPU was requested, and its advice to raise the minimum image tokens."""

from __future__ import annotations

import os

import pytest

from localm.inference.backends.llamacpp import llama as llama_mod
from localm.inference.backends.llamacpp import mtmd as mtmd_mod
from tests._bare_llama import make_bare_llama


def _projector_that_logs(native_text: str, *, on_gpu: bool = True):
    class _Projector:
        supports_vision = True
        supports_audio = False

        def __init__(self, mmproj_path, model_ptr, gpu_index=0):
            self.on_gpu = on_gpu
            os.write(2, native_text.encode())

    return _Projector


def _load(monkeypatch, native_text, *, on_gpu=True, verbose=False):
    monkeypatch.setattr(mtmd_mod, "MtmdContext", _projector_that_logs(native_text, on_gpu=on_gpu))
    inst = make_bare_llama(_model_ptr=0xBEEF)
    inst._load_mmproj("/fake/mmproj.gguf", verbose=verbose)
    return inst


def _warnings(caplog):
    return [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]


def test_a_projector_clip_places_on_the_cpu_is_marked_and_reported(monkeypatch, caplog):
    with caplog.at_level("WARNING", logger="localm"):
        inst = _load(monkeypatch, "clip_ctx: CLIP using CPU backend\n")
    assert inst._mtmd.on_gpu is False
    assert any("placed the vision projector on the CPU" in m for m in _warnings(caplog))


def test_a_projector_clip_places_on_the_gpu_is_not_reported(monkeypatch, caplog):
    with caplog.at_level("WARNING", logger="localm"):
        inst = _load(monkeypatch, "clip_ctx: CLIP using ROCm0 backend\n")
    assert inst._mtmd.on_gpu is True
    assert not _warnings(caplog)


def test_a_projector_already_on_the_cpu_adds_no_second_warning(monkeypatch, caplog):
    with caplog.at_level("WARNING", logger="localm"):
        inst = _load(monkeypatch, "clip_ctx: CLIP using CPU backend\n", on_gpu=False)
    assert inst._mtmd.on_gpu is False
    assert not _warnings(caplog)


def test_the_image_min_tokens_advice_is_reported(monkeypatch, caplog):
    text = ("clip_ctx: Qwen-VL models require at minimum 1024 image tokens to function "
            "correctly on grounding tasks\n"
            "clip_ctx: if you encounter problems with accuracy, try adding "
            "--image-min-tokens 1024\n")
    with caplog.at_level("WARNING", logger="localm"):
        inst = _load(monkeypatch, text)
    assert inst._mtmd.on_gpu is True
    assert any("at least 1024 image tokens" in m for m in _warnings(caplog))


def test_a_quiet_native_load_adds_no_warning(monkeypatch, caplog):
    with caplog.at_level("WARNING", logger="localm"):
        _load(monkeypatch, "clip_model_loader: tensor[0]\n")
    assert not _warnings(caplog)


def test_the_notices_are_read_from_the_capture(tmp_path):
    log = tmp_path / "native.log"
    log.write_text("x\nCLIP using CPU backend\n", encoding="utf-8")
    assert llama_mod._CapturedStderr(str(log)).projector_notices() == {
        "cpu": True, "min_tokens": False}
    assert llama_mod._CapturedStderr(str(tmp_path / "gone.log")).projector_notices() == {
        "cpu": False, "min_tokens": False}


@pytest.mark.parametrize("verbose", [True])
def test_a_verbose_load_has_no_capture_and_does_not_fail(monkeypatch, verbose):
    inst = _load(monkeypatch, "", verbose=verbose)
    assert inst._mtmd is not None
