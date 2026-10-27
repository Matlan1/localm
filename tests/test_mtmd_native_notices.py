# SPDX-License-Identifier: AGPL-3.0-or-later
"""What llama.cpp says on stderr while it loads a vision projector is acted on:
a projector placed on the CPU although the GPU was requested is marked and
reported, and the minimum image tokens llama.cpp asks for are applied by
rebuilding the projector with them."""

from __future__ import annotations

import os

from localm.inference.backends.llamacpp import llama as llama_mod
from localm.inference.backends.llamacpp import mtmd as mtmd_mod
from tests._bare_llama import make_bare_llama

ADVICE = ("clip_ctx: Qwen-VL models require at minimum 1024 image tokens to function "
          "correctly on grounding tasks\n"
          "clip_ctx: if you encounter problems with accuracy, try adding "
          "--image-min-tokens 1024\n")


def _projector_that_logs(native_text: str, *, on_gpu: bool = True, can_apply: bool = True):
    class _Projector:
        supports_vision = True
        supports_audio = False
        applied: list = []

        def __init__(self, mmproj_path, model_ptr, gpu_index=0):
            self.on_gpu = on_gpu
            self.applied = []
            os.write(2, native_text.encode())

        def apply_min_image_tokens(self, tokens):
            self.applied.append(tokens)
            return can_apply

    return _Projector


def _load(monkeypatch, native_text, *, on_gpu=True, can_apply=True, verbose=False):
    monkeypatch.setattr(
        mtmd_mod, "MtmdContext", _projector_that_logs(native_text, on_gpu=on_gpu,
                                                       can_apply=can_apply))
    inst = make_bare_llama(_model_ptr=0xBEEF)
    inst._load_mmproj("/fake/mmproj.gguf", verbose=verbose)
    return inst


def _records(caplog, level):
    return [r.getMessage() for r in caplog.records if r.levelname == level]


def test_a_projector_clip_places_on_the_cpu_is_marked_and_reported(monkeypatch, caplog):
    monkeypatch.setattr(llama_mod.api, "llama_supports_gpu_offload", lambda: True)
    with caplog.at_level("INFO", logger="localm"):
        inst = _load(monkeypatch, "clip_ctx: CLIP using CPU backend\n")
    assert inst._mtmd.on_gpu is False
    assert any("placed the vision projector on the CPU" in m for m in _records(caplog, "WARNING"))


def test_a_projector_on_the_cpu_of_a_cpu_only_runtime_is_marked_but_not_warned(
        monkeypatch, caplog):
    monkeypatch.setattr(llama_mod.api, "llama_supports_gpu_offload", lambda: False)
    with caplog.at_level("INFO", logger="localm"):
        inst = _load(monkeypatch, "clip_ctx: CLIP using CPU backend\n")
    assert inst._mtmd.on_gpu is False
    assert not _records(caplog, "WARNING")


def test_a_projector_clip_places_on_the_gpu_is_not_reported(monkeypatch, caplog):
    with caplog.at_level("INFO", logger="localm"):
        inst = _load(monkeypatch, "clip_ctx: CLIP using ROCm0 backend\n")
    assert inst._mtmd.on_gpu is True
    assert not _records(caplog, "WARNING")


def test_a_projector_already_on_the_cpu_adds_no_second_warning(monkeypatch, caplog):
    with caplog.at_level("INFO", logger="localm"):
        inst = _load(monkeypatch, "clip_ctx: CLIP using CPU backend\n", on_gpu=False)
    assert inst._mtmd.on_gpu is False
    assert not _records(caplog, "WARNING")


def test_the_advised_minimum_image_tokens_are_applied_without_a_warning(monkeypatch, caplog):
    with caplog.at_level("INFO", logger="localm"):
        inst = _load(monkeypatch, ADVICE)
    assert inst._mtmd.applied == [1024]
    assert inst._mtmd.on_gpu is True
    assert not _records(caplog, "WARNING")
    assert any("at least 1024 image tokens" in m for m in _records(caplog, "INFO"))


def test_the_number_comes_from_the_advice(monkeypatch):
    inst = _load(monkeypatch, "x: try adding --image-min-tokens 2048\n")
    assert inst._mtmd.applied == [2048]


def test_a_runtime_that_cannot_take_the_setting_is_reported(monkeypatch, caplog):
    with caplog.at_level("INFO", logger="localm"):
        inst = _load(monkeypatch, ADVICE, can_apply=False)
    assert inst._mtmd.applied == [1024]
    assert any("could not apply it" in m for m in _records(caplog, "WARNING"))


def test_a_quiet_native_load_changes_nothing(monkeypatch, caplog):
    with caplog.at_level("INFO", logger="localm"):
        inst = _load(monkeypatch, "clip_model_loader: tensor[0]\n")
    assert inst._mtmd.applied == []
    assert not _records(caplog, "WARNING")


def test_the_notices_are_read_from_the_capture(tmp_path):
    log = tmp_path / "native.log"
    log.write_text("x\nCLIP using CPU backend\nuse --image-min-tokens 1024\n", encoding="utf-8")
    assert llama_mod._CapturedStderr(str(log)).projector_notices() == {
        "cpu": True, "min_tokens": 1024}
    assert llama_mod._CapturedStderr(str(tmp_path / "gone.log")).projector_notices() == {
        "cpu": False, "min_tokens": None}


def test_an_absurd_advised_number_is_ignored(tmp_path):
    log = tmp_path / "native.log"
    log.write_text("--image-min-tokens 99999\n", encoding="utf-8")
    assert llama_mod._CapturedStderr(str(log)).projector_notices()["min_tokens"] is None


def test_a_verbose_load_has_no_capture_and_does_not_fail(monkeypatch):
    inst = _load(monkeypatch, "", verbose=True)
    assert inst._mtmd is not None
