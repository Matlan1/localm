# SPDX-License-Identifier: AGPL-3.0-or-later
"""A vision or audio projector (mmproj) attached to a GGUF load is charged in
every VRAM estimate of that load: its GGUF's bytes plus a compute buffer
allowance, beside the compute overhead, whatever the layer count.

The model and the projector are real GGUF files built byte by byte; the
sizing runs for real against them and only the VRAM readings are patched.
"""

import os
from unittest.mock import patch

import pytest

from localm.inference.backends.gguf import GgufBackend
from localm.inference.backends.llamacpp._sizing import VramSizingMixin
from tests.test_gguf_moe_vram_sizing import _T_STRING, _gguf_with_tensors
from tests.test_moe_auto_placement import KV, N_CTX, OVERHEAD, _dense_model, _Vram

COMPUTE = 7_000
PROJECTOR_TENSOR = 300_000
SHIPPED_COMPUTE = VramSizingMixin._PROJECTOR_COMPUTE_BYTES


@pytest.fixture(autouse=True)
def _small_compute_allowance():
    with patch.object(VramSizingMixin, "_PROJECTOR_COMPUTE_BYTES", COMPUTE):
        yield


def _projector(tmp_path, name="mmproj-test.gguf"):
    return _gguf_with_tensors(tmp_path / name,
                              [("general.architecture", _T_STRING, "clip")],
                              [("v.patch_embd.weight", [4], 0, PROJECTOR_TENSOR)])


def _backend(path, mmproj=None, **kw):
    kw.setdefault("n_gpu_layers", 99)
    kw.setdefault("n_gpu_layers_auto", True)
    return GgufBackend(str(path), mmproj_path=str(mmproj) if mmproj else None,
                       n_ctx=N_CTX, **kw)


def _charge(projector) -> int:
    return os.path.getsize(projector) + COMPUTE


def test_the_charge_is_the_projectors_file_bytes_and_the_compute_allowance(tmp_path):
    projector = _projector(tmp_path)
    assert os.path.getsize(projector) > PROJECTOR_TENSOR
    b = _backend(_dense_model(tmp_path), projector)
    assert b._projector_vram_bytes() == _charge(projector)


def test_no_projector_no_charge(tmp_path):
    assert _backend(_dense_model(tmp_path))._projector_vram_bytes() == 0


def test_an_unreadable_projector_is_charged_nothing(tmp_path):
    b = _backend(_dense_model(tmp_path), tmp_path / "missing-mmproj.gguf")
    assert b._projector_vram_bytes() == 0


def test_a_projector_kept_on_the_cpu_takes_system_ram_not_vram(tmp_path, monkeypatch):
    projector = _projector(tmp_path)
    b = _backend(_dense_model(tmp_path), projector)
    assert b._cpu_projector_bytes(True) == 0
    assert b._cpu_projector_bytes(False) == os.path.getsize(projector)
    monkeypatch.setenv("LOCALM_MTMD_CPU", "1")
    assert b._projector_vram_bytes() == 0
    assert b._cpu_projector_bytes(True) == os.path.getsize(projector)


def test_a_cpu_load_counts_the_projector_in_system_ram(tmp_path):
    model, projector = _dense_model(tmp_path), _projector(tmp_path)
    plain, vision = _backend(model, n_gpu_layers=0), _backend(model, projector, n_gpu_layers=0)
    host, read, _ = plain._mmap_byte_counts(0, False, 0)
    v_host, v_read, _ = vision._mmap_byte_counts(0, False, 0)
    assert v_host == host + os.path.getsize(projector)
    assert v_read == read
    available = 10**7
    with patch("localm.sysstats.system_ram", return_value=(2 * available, available)), \
            patch("localm.inference.backends.llamacpp._sizing.HOST_RAM_HEADROOM_BYTES", 0):
        assert (vision._ram_ctx_tokens(0, 1)
                == plain._ram_ctx_tokens(0, 1) - os.path.getsize(projector))


def test_the_full_offload_need_carries_the_projector(tmp_path):
    model, projector = _dense_model(tmp_path), _projector(tmp_path)
    plain, vision = _backend(model), _backend(model, projector)
    with _Vram(10**9):
        assert (vision.full_offload_vram_bytes()
                == plain.full_offload_vram_bytes() + _charge(projector))
        full, low = vision.gpu_offload_needs()
    assert full == low == vision._vram_model_bytes(0) + KV + OVERHEAD + _charge(projector)


def test_the_preflight_check_charges_the_projector(tmp_path):
    model, projector = _dense_model(tmp_path), _projector(tmp_path)
    with _Vram(10**9):
        plain = _backend(model)._planned_vram(99)
        vision = _backend(model, projector)._planned_vram(99)
    assert vision[0] == plain[0] + _charge(projector)
    assert vision[5] == plain[5] + _charge(projector)


def test_auto_sizing_keeps_room_for_the_projector(tmp_path):
    model, projector = _dense_model(tmp_path), _projector(tmp_path)
    fits_text_only = _backend(model)._vram_model_bytes(0) + KV + OVERHEAD
    with _Vram(fits_text_only):
        assert _backend(model)._auto_gpu_layers() == 99
        assert 0 < _backend(model, projector)._auto_gpu_layers() < 99
    with _Vram(fits_text_only + _charge(projector)):
        assert _backend(model, projector)._auto_gpu_layers() == 99


def test_the_sizing_record_names_the_projector(tmp_path):
    from types import SimpleNamespace
    from localm.inference import http_server as hs
    projector = _projector(tmp_path)
    b = _backend(_dense_model(tmp_path), projector)
    with _Vram(10**9):
        budget = b._auto_gpu_layers_budget()
    b._record_gpu_sizing("auto", budget.layers, budget)
    record = b.last_gpu_sizing
    assert record["projector_bytes"] == _charge(projector)
    assert record["overhead_bytes"] == OVERHEAD + _charge(projector)
    engine = SimpleNamespace(gpu_placement=None, gpu_sizing=record, mmap_state=None)
    line = hs._describe_load_placement("vl", engine)
    mb = 1024 ** 2
    assert f"overhead {record['overhead_bytes'] // mb}, of which projector " \
           f"{record['projector_bytes'] // mb})" in line


def test_the_automatic_context_ceiling_leaves_room_for_the_projector(tmp_path):
    model, projector = _dense_model(tmp_path), _projector(tmp_path)
    kw = dict(ctx_auto=True, n_ctx_max=0)
    with _Vram(10**7), patch(
            "localm.inference.backends.llamacpp._sizing.embedder_ctx_reservation_bytes",
            return_value=0):
        plain = _backend(model, **kw)
        vision = _backend(model, projector, **kw)
        per_token = plain._kv_bytes_per_token()
        plain_ctx, _ = plain._auto_ctx_sized(capped=False)
        vision_ctx, _ = vision._auto_ctx_sized(capped=False)
    expected = ((10**7 - plain._effective_model_bytes_for_vram() - OVERHEAD
                 - _charge(projector)) // per_token // 1024) * 1024
    assert vision_ctx == expected < plain_ctx


def test_a_load_over_two_gpus_shares_the_projector_among_them(tmp_path):
    from tests.test_implicit_split_fit import _write_gguf
    tensors = [("token_embd.weight", 5_000), ("output.weight", 5_000)]
    tensors += [(f"blk.{il}.attn_q.weight", 110_000) for il in range(4)]
    extra = [("embedding_length", 64), ("attention.head_count", 4),
             ("attention.head_count_kv", 4)]
    model = _write_gguf(tmp_path / "two.gguf", arch="dense", block_count=4, vocab=8,
                        tensors=tensors, extra_kv=extra)
    projector = _projector(tmp_path)
    plain = _backend(model)._placement_units()
    vision = _backend(model, projector)._placement_units()
    assert vision.spread_bytes == plain.spread_bytes + _charge(projector)


def test_the_projector_charge_covers_the_measured_buffers():
    """The compute buffers mtmd reserved on the GPU for the projectors below,
    as llama.cpp printed them ("mtmd_context: <device> compute buffer size"):
    b12676 on Vulkan, and the amd-rocm build b1384 on ROCm. The allowance
    covers each of them."""
    measured_mib = {
        "Qwen3VL-8B F16 projector, Vulkan": 387.06,
        "Qwen3VL-8B F16 projector, ROCm": 372.08,
        "Gemma-4-E4B F16 vision and audio projector, ROCm": 143.65,
        "Qwen3.5-9B BF16 projector, ROCm": 248.10,
    }
    for name, mib in measured_mib.items():
        assert SHIPPED_COMPUTE >= mib * 1024 ** 2, name
