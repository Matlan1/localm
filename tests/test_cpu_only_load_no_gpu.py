# SPDX-License-Identifier: AGPL-3.0-or-later
"""A load with zero GPU layers does no GPU work: the GGUF chat model gets an
empty llama.cpp device list (so its contexts have no GPU backend for op offload
to send a prompt batch to) in a worker that hides the GPU, its vision projector and draft model stay on the
CPU, and a zero-layer embedder hides the GPU from its worker. A load with any
GPU layers keeps the default device list and op offload."""

import ctypes
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from localm.inference.backends.llamacpp import llama as llama_mod
from localm.inference.backends.llamacpp import mtmd as mtmd_mod
from localm.inference.backends.llamacpp.llama import LlamaCpp
from tests._bare_llama import make_bare_llama
from tests.test_embedder import _isolated_embedder
from tests.test_mtmd_device_placement import CPU, GPU, _devices, _FakeLib


def _mock_api(seen):
    """A mocked llama.cpp API whose model load records the device list it was
    given: None for the native default (every GPU), else the entries of the
    NULL-terminated list up to and including the terminator."""
    api = MagicMock()
    api.llama_model_default_params.return_value = SimpleNamespace(
        main_gpu=0, n_gpu_layers=0, use_mmap=True, devices=None,
        split_mode=1, tensor_split=None)
    cp = MagicMock()
    cp.op_offload = True
    api.llama_context_default_params.return_value = cp

    def load(path, mp):
        if mp.devices is None:
            seen["devices"] = None
        else:
            arr = ctypes.cast(mp.devices, ctypes.POINTER(ctypes.c_void_p))
            entries = []
            while True:
                entries.append(arr[len(entries)])
                if entries[-1] is None:
                    break
            seen["devices"] = entries
        return ctypes.c_void_p(1)
    api.llama_load_model_from_file.side_effect = load
    return api


def _build(n_gpu_layers, **kwargs):
    seen = {}
    api = _mock_api(seen)
    with patch.object(llama_mod, "api", api), \
            patch("localm.discover.apply_main_gpu") as main, \
            patch("localm.discover.apply_gpu_split", return_value=None) as split, \
            patch("localm.discover.apply_gpu_devices", return_value=None) as devices:
        llm = LlamaCpp("m.gguf", n_ctx=512, n_gpu_layers=n_gpu_layers, verbose=True,
                       **kwargs)
        llm.close()
    calls = SimpleNamespace(main=main, split=split, devices=devices)
    return llm, seen, api.llama_context_default_params.return_value, calls


@pytest.mark.parametrize("kwargs", [{}, {"main_gpu": 1},
                                    {"gpu_devices": [{"slot": 0, "device_id": None,
                                                      "units": 4}]}])
def test_a_cpu_only_load_gets_no_gpu_device(kwargs):
    llm, seen, cp, calls = _build(0, **kwargs)
    assert seen["devices"] == [None]
    calls.main.assert_not_called()
    calls.split.assert_not_called()
    calls.devices.assert_not_called()
    assert llm._cpu_only is True and llm._main_gpu_index == 0


@pytest.mark.parametrize("n_gpu_layers", [5, 99])
def test_a_load_with_gpu_layers_keeps_every_gpu_and_op_offload(n_gpu_layers):
    llm, seen, cp, calls = _build(n_gpu_layers)
    assert seen["devices"] is None
    calls.main.assert_called_once()
    calls.split.assert_called_once()
    assert cp.op_offload is True
    assert llm._cpu_only is False


@pytest.mark.parametrize("cpu_only,use_gpu", [(True, False), (False, True)])
def test_the_vision_projector_follows_a_cpu_only_load(cpu_only, use_gpu):
    llm = make_bare_llama(_model_ptr=ctypes.c_void_p(1), _ctx_ptr=None)
    llm._cpu_only = cpu_only
    llm._main_gpu_index = 0
    fake = MagicMock(supports_vision=True, supports_audio=False)
    with patch.object(mtmd_mod, "MtmdContext", return_value=fake) as ctor, \
            patch.object(mtmd_mod, "compatible_mmproj_path", side_effect=lambda p: p):
        llm._load_mmproj("mm.gguf", verbose=True)
    assert ctor.call_args.kwargs["use_gpu"] is use_gpu
    assert llm._mtmd is fake


class _RecordingLib(_FakeLib):
    """The mtmd library fake, also recording the use_gpu byte of every open."""

    def __init__(self):
        super().__init__()
        self.use_gpu = []

    def mtmd_init_from_file(self, path, model_ptr, params):
        self.use_gpu.append(bytes(params)[0])
        return super().mtmd_init_from_file(path, model_ptr, params)

    def mtmd_support_vision(self, ctx):
        return True

    def mtmd_default_marker(self):
        return b"<__media__>"


def test_a_cpu_projector_opens_once_on_the_cpu_without_a_warning(monkeypatch, caplog):
    _devices(monkeypatch, [("CPU", CPU), ("ROCm0", GPU)])
    lib = _RecordingLib()
    monkeypatch.setattr(mtmd_mod, "_load_lib", lambda: lib)
    monkeypatch.setattr(mtmd_mod, "_input_text_class", mtmd_mod._MtmdInputTextV2)
    with caplog.at_level("INFO", logger="localm"):
        ctx = mtmd_mod.MtmdContext("/fake/mmproj.gguf", 0xBEEF, use_gpu=False)
    assert lib.use_gpu == [0]
    assert ctx.on_gpu is False
    assert lib.seen_env is None
    assert not [r for r in caplog.records if r.levelname == "WARNING"]


def test_a_gpu_projector_still_opens_on_the_gpu(monkeypatch):
    _devices(monkeypatch, [("CPU", CPU), ("ROCm0", GPU)])
    lib = _RecordingLib()
    monkeypatch.setattr(mtmd_mod, "_load_lib", lambda: lib)
    monkeypatch.setattr(mtmd_mod, "_input_text_class", mtmd_mod._MtmdInputTextV2)
    ctx = mtmd_mod.MtmdContext("/fake/mmproj.gguf", 0xBEEF)
    assert lib.use_gpu == [1]
    assert ctx.on_gpu is True


@pytest.mark.parametrize("cpu_only,draft_gpu,on_gpu", [
    (True, True, False), (False, True, True), (False, False, False)])
def test_the_draft_model_follows_a_cpu_only_load(cpu_only, draft_gpu, on_gpu):
    llm = make_bare_llama(_model_ptr=ctypes.c_void_p(1), _ctx_ptr=ctypes.c_void_p(2))
    llm._cpu_only = cpu_only
    llm._spec_source_name = "draft"
    with patch.object(LlamaCpp, "_cache_can_drop_a_speculative_token", return_value=True), \
            patch.object(LlamaCpp, "_load_draft_model") as load, \
            patch.object(LlamaCpp, "_draft_source", return_value=None):
        llm._set_up_spec_source("d.gguf", None, True, draft_gpu)
    assert load.call_args.kwargs["on_gpu"] is on_gpu


@pytest.mark.parametrize("n_gpu_layers,cpu_only", [(0, True), (5, False), (99, False)])
def test_a_zero_layer_embedder_hides_the_gpu_from_its_worker(
        monkeypatch, n_gpu_layers, cpu_only):
    embedder = _isolated_embedder(monkeypatch, n_gpu_layers=n_gpu_layers)
    assert embedder._runner.loaded_params["cpu_only"] is cpu_only


_GPU_VARS = ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")


@pytest.mark.parametrize("n_gpu_layers,seen_value", [(0, "-1"), (5, None), (99, None)])
def test_the_gguf_worker_hides_the_gpu_before_a_cpu_only_load(
        monkeypatch, n_gpu_layers, seen_value):
    import queue

    from localm.inference.backends.llamacpp import _runner
    for name in ("install_parent_death_watchdog", "ignore_interrupt_signals",
                 "suppress_native_error_dialogs"):
        monkeypatch.setattr(f"localm._mp_spawn.{name}", lambda: None)
    monkeypatch.setattr("localm.debuglog.attach_child_logging", lambda: None)
    for var in _GPU_VARS:
        monkeypatch.delenv(var, raising=False)
    seen = {}

    class _Worker:
        def __init__(self, cancel_event=None, **payload):
            seen.update({var: os.environ.get(var) for var in _GPU_VARS})
            self.stream_cancel = None

        def load(self):
            return {}

        def close(self):
            pass

    monkeypatch.setattr("localm.inference.backends.llamacpp._worker.GgufWorker", _Worker)
    req_q, resp_q, ctrl_q = queue.Queue(), queue.Queue(), queue.Queue()
    req_q.put(("load", {"model_path": "m.gguf", "n_gpu_layers": n_gpu_layers}))
    req_q.put(None)
    try:
        _runner._runner_main(req_q, resp_q, ctrl_q)
    finally:
        ctrl_q.put(None)
        for var in _GPU_VARS:
            os.environ.pop(var, None)
    assert resp_q.get_nowait()[0] == "ok"
    assert seen == {var: seen_value for var in _GPU_VARS}
