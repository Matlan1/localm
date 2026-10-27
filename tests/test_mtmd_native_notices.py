# SPDX-License-Identifier: AGPL-3.0-or-later
"""What llama.cpp says on stderr while it loads a vision projector is acted on:
a projector placed on the CPU although the GPU was requested is marked and
reported, and the minimum image tokens llama.cpp asks for are applied by
rebuilding the projector with them."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from localm.inference.backends.llamacpp import _loader
from localm.inference.backends.llamacpp import llama as llama_mod
from localm.inference.backends.llamacpp import mtmd as mtmd_mod
from tests._bare_llama import make_bare_llama
from tests._real_gguf import require_native_runtime

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


def _runtime_devices(monkeypatch, devices, *, offload_flag=True):
    monkeypatch.setattr(_loader, "compute_devices", lambda: list(devices))
    monkeypatch.setattr(llama_mod.api, "llama_supports_gpu_offload", lambda: offload_flag)


def test_a_projector_clip_places_on_the_cpu_is_marked_and_reported(monkeypatch, caplog):
    _runtime_devices(monkeypatch, [("ROCm0", _loader.GGML_DEV_TYPE_GPU), ("CPU", 0)])
    with caplog.at_level("INFO", logger="localm"):
        inst = _load(monkeypatch, "clip_ctx: CLIP using CPU backend\n")
    assert inst._mtmd.on_gpu is False
    assert any("placed the vision projector on the CPU" in m for m in _records(caplog, "WARNING"))


def test_a_projector_on_the_cpu_of_a_cpu_only_runtime_is_marked_but_not_warned(
        monkeypatch, caplog):
    _runtime_devices(monkeypatch, [("CPU", 0)], offload_flag=False)
    with caplog.at_level("INFO", logger="localm"):
        inst = _load(monkeypatch, "clip_ctx: CLIP using CPU backend\n")
    assert inst._mtmd.on_gpu is False
    assert not _records(caplog, "WARNING")


def test_a_cpu_only_machine_with_the_rpc_backend_registered_is_not_warned(
        monkeypatch, caplog):
    _runtime_devices(monkeypatch, [("CPU", 0)], offload_flag=True)
    with caplog.at_level("INFO", logger="localm"):
        inst = _load(monkeypatch, "clip_ctx: CLIP using CPU backend\n")
    assert inst._mtmd.on_gpu is False
    assert not _records(caplog, "WARNING")


def test_an_unreadable_device_registry_falls_back_to_the_offload_flag(
        monkeypatch, caplog):
    _runtime_devices(monkeypatch, [], offload_flag=True)
    with caplog.at_level("INFO", logger="localm"):
        _load(monkeypatch, "clip_ctx: CLIP using CPU backend\n")
    assert any("placed the vision projector on the CPU" in m
               for m in _records(caplog, "WARNING"))
    caplog.clear()
    _runtime_devices(monkeypatch, [], offload_flag=False)
    with caplog.at_level("INFO", logger="localm"):
        _load(monkeypatch, "clip_ctx: CLIP using CPU backend\n")
    assert not _records(caplog, "WARNING")


_PROBE = """
import json
from localm.inference.backends.llamacpp import _api, _loader
print(json.dumps({"devices": _loader.compute_devices(),
                  "offload": _api.llama_supports_gpu_offload(),
                  "gpu": _loader.has_gpu_device()}))
"""


def _probe_native_runtime(extra_env):
    env = {**os.environ, **extra_env,
           "PYTHONPATH": os.pathsep.join(
               [str(Path(llama_mod.__file__).parents[4]), os.environ.get("PYTHONPATH", "")])}
    out = subprocess.run([sys.executable, "-c", _PROBE], env=env, capture_output=True,
                         text=True, timeout=180)
    assert out.returncode == 0, out.stderr[-2000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_the_gate_reads_the_native_devices_not_the_offload_flag():
    require_native_runtime()
    hidden = {"HIP_VISIBLE_DEVICES": "-1", "CUDA_VISIBLE_DEVICES": "-1",
              "ROCR_VISIBLE_DEVICES": "-1", "GGML_VK_VISIBLE_DEVICES": "-1"}
    for env in ({}, hidden):
        got = _probe_native_runtime(env)
        if not got["devices"]:
            pytest.skip("runtime lacks the ggml_backend_dev_* registry symbols")
        has_non_cpu = any(t != _loader.GGML_DEV_TYPE_CPU for _, t in got["devices"])
        assert got["gpu"] is has_non_cpu, got


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
