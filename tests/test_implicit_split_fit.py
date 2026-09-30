# SPDX-License-Identifier: AGPL-3.0-or-later
"""Per-device fit for llama.cpp's implicit multi-GPU layer split.

With no tensor_split, llama.cpp weights devices by free memory and puts the
output layer, plus the logits buffer the context reserves for it, on the
device holding the LAST share. On a heterogeneous box that is often the
smallest card: the reported layout (5 devices with 13.7 / 21.9 / 21.9 / 14.4 /
2.7 GB free, a 35B MoE model with a 248k vocabulary) gives the 2.7 GB device
only the output layer and a 1.9 GB logits buffer. The fit predicts that
placement, and when a device cannot hold its charge, writes an explicit split
that leaves it out.
"""

import ctypes
import struct
from unittest import mock

import pytest

from localm import discover
from localm.inference.backends.gguf import GgufBackend
from localm.inference.backends.llamacpp import _loader
from localm.inference.backends.llamacpp._split_fit import (
    layer_devices, logits_buffer_bytes, plan_split)
from localm.model_manager.gguf import gguf_split_layout

GiB = 1024 ** 3
MiB = 1024 ** 2

# The reported free readings, device order as logged.
_REPORTED_FREE = [13.7, 21.9, 21.9, 14.4, 2.7]
# The reported model: 41 blocks (40 + 1 nextn), ~855.5 MiB per block, a
# 515.3 MiB Q8_0 output head, a 248320-token vocabulary.
_N_LAYER_ALL = 41
_N_VOCAB = 248320


def _devices(frees_gib):
    return [{"index": i, "free": int(f * GiB), "total": int((f + 2) * GiB)}
            for i, f in enumerate(frees_gib)]


def _reported_plan(**overrides):
    kw = dict(
        layer_bytes=[int(855.5 * MiB)] * 40 + [0],
        layer_kv_bytes=[int(4096 * 22528 / 40)] * 40 + [0],
        output_bytes=int(515.3 * MiB),
        n_gpu_layers=99,
        logits_bytes=logits_buffer_bytes(_N_VOCAB, 4096),
        reserve_bytes=int(1.5e9))
    kw.update(overrides)
    return plan_split(_devices(_REPORTED_FREE), **kw)


class TestLayerDevices:
    def test_reported_layout_puts_only_the_output_layer_on_the_small_device(self):
        frees = [int(f * GiB) for f in _REPORTED_FREE]
        positions, out = layer_devices(frees, _N_LAYER_ALL, 99)
        counts = [sum(1 for p in positions if p == i) for i in range(5)]
        assert counts == [8, 13, 12, 8, 0]
        assert out == 4

    def test_zero_share_device_gets_no_layers_and_output_moves_back(self):
        shares = [int(f * GiB) for f in _REPORTED_FREE[:4]] + [0]
        positions, out = layer_devices(shares, _N_LAYER_ALL, 99)
        assert 4 not in positions
        assert out == 3

    def test_partial_offload_keeps_the_first_layers_on_cpu(self):
        positions, out = layer_devices([1.0, 1.0], 10, 4)
        assert positions[:7] == [None] * 7
        assert all(p is not None for p in positions[7:])
        assert out == 1

    def test_no_devices_is_all_cpu(self):
        positions, out = layer_devices([], 5, 99)
        assert positions == [None] * 5 and out is None


class TestPlanSplit:
    def test_reported_layout_leaves_the_small_device_out(self):
        plan = _reported_plan()
        assert not plan.default_fits
        short = [c for c in plan.default if not c.fits]
        assert [c.index for c in short] == [4]
        assert short[0].holds_output and short[0].layers == 0
        assert plan.excluded == [4]
        assert sorted(plan.tensor_split) == [0, 1, 2, 3]
        assert abs(sum(plan.tensor_split.values()) - 1.0) < 1e-9
        assert all(c.fits for c in plan.chosen)
        assert next(c for c in plan.chosen if c.holds_output).index == 3

    def test_logits_buffer_size_for_the_reported_model(self):
        assert logits_buffer_bytes(_N_VOCAB, 4096) == _N_VOCAB * 2048 * 4
        assert logits_buffer_bytes(_N_VOCAB, 1024) == _N_VOCAB * 1024 * 4
        assert logits_buffer_bytes(_N_VOCAB, 4096, contexts=2) == 2 * _N_VOCAB * 2048 * 4

    def test_a_layout_that_fits_keeps_llamacpps_default(self):
        plan = plan_split(_devices([22.0, 22.0]), layer_bytes=[100 * MiB] * 32,
                          layer_kv_bytes=[MiB] * 32, output_bytes=300 * MiB,
                          n_gpu_layers=99, logits_bytes=GiB,
                          reserve_bytes=int(1.5e9))
        assert plan.default_fits
        assert plan.tensor_split is None and plan.excluded == []

    def test_no_split_is_written_when_fewer_than_two_devices_would_remain(self):
        plan = plan_split(_devices([2.0, 2.0]), layer_bytes=[500 * MiB] * 8,
                          layer_kv_bytes=[0] * 8, output_bytes=500 * MiB,
                          n_gpu_layers=99, logits_bytes=GiB,
                          reserve_bytes=int(1.5e9))
        assert not plan.default_fits
        assert plan.tensor_split is None

    def test_excluding_the_output_device_can_move_the_output_to_another_small_one(self):
        # Device 3 gets the output first; leaving it out moves the output to
        # device 2, which is too small as well.
        plan = plan_split(_devices([20.0, 20.0, 1.2, 1.5]),
                          layer_bytes=[200 * MiB] * 40, layer_kv_bytes=[0] * 40,
                          output_bytes=500 * MiB, n_gpu_layers=99,
                          logits_bytes=GiB, reserve_bytes=int(0.5e9))
        assert next(c for c in plan.default if c.holds_output).index == 3
        assert sorted(plan.excluded) == [2, 3]
        assert sorted(plan.tensor_split) == [0, 1]
        assert next(c for c in plan.chosen if c.holds_output).index == 1

    def test_a_small_device_that_receives_nothing_is_kept(self):
        # Device 3's share is too small to receive a layer, so it needs no
        # more than the reserve and stays in the split.
        plan = plan_split(_devices([20.0, 20.0, 1.0, 1.0]),
                          layer_bytes=[200 * MiB] * 20, layer_kv_bytes=[0] * 20,
                          output_bytes=500 * MiB, n_gpu_layers=99,
                          logits_bytes=GiB, reserve_bytes=int(0.5e9))
        assert plan.excluded == [2]
        assert sorted(plan.tensor_split) == [0, 1, 3]


class TestApplyGpuSplitMapping:
    """A {device_index: share} mapping applies without a configured split and
    without re-probing, leaving unnamed devices at a zero share."""

    def _mp(self):
        class _MP:
            tensor_split = None
            split_mode = 0
            main_gpu = 0
        return _MP()

    def test_mapping_writes_zero_for_the_left_out_device(self):
        mp = self._mp()
        with mock.patch.object(discover, "list_gpus",
                               side_effect=AssertionError("must not probe")) as lg, \
                mock.patch.object(discover, "_tensor_split_capacity", return_value=16):
            arr = discover.apply_gpu_split(
                mp, config={}, ratios_override={0: 0.2, 1: 0.3, 2: 0.3, 3: 0.2})
        lg.assert_not_called()
        assert arr is not None
        values = [arr[i] for i in range(5)]
        assert values[4] == 0.0
        assert values[:4] == pytest.approx([0.2, 0.3, 0.3, 0.2])
        assert mp.split_mode == discover._LLAMA_SPLIT_MODE_LAYER
        assert ctypes.cast(mp.tensor_split, ctypes.c_void_p).value == \
            ctypes.addressof(arr)

    def test_list_override_still_pairs_with_the_configured_indices(self):
        mp = self._mp()
        gpus = [{"index": i, "free": 1, "total": 2} for i in range(3)]
        with mock.patch.object(discover, "list_gpus", return_value=gpus), \
                mock.patch.object(discover, "_native_gpu_index_space_is_opaque",
                                  return_value=False), \
                mock.patch.object(discover, "_tensor_split_capacity", return_value=16):
            arr = discover.apply_gpu_split(
                mp, config={"gpu_split_indices": [1, 2]}, ratios_override=[0.25, 0.75])
        assert [arr[i] for i in range(3)] == pytest.approx([0.0, 0.25, 0.75])

    def test_a_one_entry_mapping_applies_no_split(self):
        mp = self._mp()
        assert discover.apply_gpu_split(mp, config={}, ratios_override={0: 1.0}) is None
        assert mp.tensor_split is None


# --- a real, internally consistent GGUF written byte by byte ---------------

_T_UINT32 = 4
_T_STRING = 8
_T_ARRAY = 9


def _s(text):
    raw = text.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _write_gguf(path, *, arch, block_count, vocab, tensors, nextn=0):
    kv = [("general.architecture", _T_STRING, arch),
          (f"{arch}.block_count", _T_UINT32, block_count)]
    if nextn:
        kv.append((f"{arch}.nextn_predict_layers", _T_UINT32, nextn))
    kv.append(("tokenizer.ggml.tokens", _T_ARRAY, [f"t{i}" for i in range(vocab)]))
    out = [b"GGUF", struct.pack("<I", 3), struct.pack("<QQ", len(tensors), len(kv))]
    for key, vtype, val in kv:
        out += [_s(key), struct.pack("<I", vtype)]
        if vtype == _T_STRING:
            out.append(_s(val))
        elif vtype == _T_UINT32:
            out.append(struct.pack("<I", val))
        else:
            out += [struct.pack("<I", _T_STRING), struct.pack("<Q", len(val))]
            out += [_s(v) for v in val]
    off = 0
    for name, size in tensors:
        out += [_s(name), struct.pack("<I", 1), struct.pack("<Q", size),
                struct.pack("<I", 0), struct.pack("<Q", off)]
        off += size
    head = b"".join(out)
    head += b"\0" * ((32 - len(head) % 32) % 32)
    path.write_bytes(head + b"\1" * off)
    return path


def _tiny_model(tmp_path, *, vocab=1000, nextn=1):
    tensors = [("token_embd.weight", 4000)]
    for il in range(4):
        tensors += [(f"blk.{il}.attn_q.weight", 3000), (f"blk.{il}.ffn.weight", 1000)]
    tensors += [("output_norm.weight", 64), ("output.weight", 5000)]
    return _write_gguf(tmp_path / "m.gguf", arch="qwen35moe", block_count=4,
                       vocab=vocab, tensors=tensors, nextn=nextn)


class TestGgufSplitLayout:
    def test_reads_blocks_vocab_and_every_tensor_size(self, tmp_path):
        layout = gguf_split_layout(_tiny_model(tmp_path))
        assert layout["block_count"] == 4
        assert layout["n_vocab"] == 1000
        sizes = layout["tensor_bytes"]
        assert sizes["blk.2.attn_q.weight"] == 3000
        assert sizes["output.weight"] == 5000
        assert sizes["output_norm.weight"] == 64

    def test_not_a_gguf_is_none(self, tmp_path):
        p = tmp_path / "x.gguf"
        p.write_bytes(b"nope" * 100)
        assert gguf_split_layout(p) is None

    def test_split_parts_are_summed(self, tmp_path):
        a = _write_gguf(tmp_path / "m-00001-of-00002.gguf", arch="llama",
                        block_count=2, vocab=10,
                        tensors=[("blk.0.a.weight", 100), ("output.weight", 50)])
        _write_gguf(tmp_path / "m-00002-of-00002.gguf", arch="llama",
                    block_count=2, vocab=10, tensors=[("blk.1.a.weight", 70)])
        layout = gguf_split_layout(a)
        assert layout["tensor_bytes"]["blk.1.a.weight"] == 70
        assert layout["tensor_bytes"]["blk.0.a.weight"] == 100

    def test_a_missing_split_part_is_none(self, tmp_path):
        a = _write_gguf(tmp_path / "m-00001-of-00002.gguf", arch="llama",
                        block_count=2, vocab=10, tensors=[("blk.0.a.weight", 100)])
        assert gguf_split_layout(a) is None


class TestBackendWiring:
    """The parent computes the plan and hands the worker the mapping."""

    # 4 blocks (the last one nextn): with shares 50 / 50 / 30 the default split
    # puts blocks 0-1 on device 0, blocks 2-3 on device 1 and the output layer
    # on device 2. A 5000-token vocabulary at a 2048-token batch is a 41 MB
    # logits buffer, more than device 2's 30 MB.
    _FREES = [50_000_000, 50_000_000, 30_000_000]
    _VOCAB = 5000

    def _backend(self, tmp_path, frees, **kw):
        path = _tiny_model(tmp_path, vocab=self._VOCAB)
        b = GgufBackend(str(path), n_ctx=4096, n_gpu_layers=99, **kw)
        b._VRAM_OVERHEAD_BYTES = 1000
        b._gguf_kv_bpt = 10
        devices = [{"index": i, "free": f, "total": f + 10} for i, f in enumerate(frees)]
        return b, devices

    def test_small_last_device_is_left_out_of_the_worker_split(self, tmp_path):
        b, devices = self._backend(tmp_path, self._FREES)
        with mock.patch.object(discover, "implicit_split_devices",
                               return_value=devices), \
                mock.patch.object(_loader, "native_lib_loaded", return_value=False):
            plan = b._implicit_split_fit(99)
        assert plan is not None
        out = next(c for c in plan.default if c.holds_output)
        assert out.index == 2 and not out.fits
        assert out.logits == self._VOCAB * 2048 * 4
        assert out.output == 5064
        assert [c.layers for c in plan.default] == [2, 2, 0]
        assert plan.excluded == [2]
        assert sorted(plan.tensor_split) == [0, 1]
        assert next(c for c in plan.chosen if c.holds_output).index == 1

    def test_mtp_layer_weights_are_charged_only_with_mtp_enabled(self, tmp_path):
        b, devices = self._backend(tmp_path, [10**9, 10**9])
        with mock.patch.object(discover, "implicit_split_devices",
                               return_value=devices), \
                mock.patch.object(_loader, "native_lib_loaded", return_value=False):
            off = b._implicit_split_fit(99)
            b.mtp_enabled = True
            on = b._implicit_split_fit(99)
        assert sum(c.weights for c in on.default) - sum(c.weights for c in off.default) == 4000
        out_on = next(c for c in on.default if c.holds_output)
        assert out_on.logits == 2 * self._VOCAB * 2048 * 4

    def test_configured_split_or_cpu_load_skips_the_fit(self, tmp_path):
        b, devices = self._backend(tmp_path, [10**9, 10**9])
        with mock.patch.object(discover, "implicit_split_devices",
                               return_value=devices):
            assert b._implicit_split_fit(0) is None
            b.n_cpu_moe = 2
            assert b._implicit_split_fit(99) is None

    def test_load_native_passes_the_mapping_and_reports_it(self, tmp_path):
        b, devices = self._backend(tmp_path, self._FREES)
        captured = {}

        def _fake_spawn(self_runner, params, cancel_event=None, timeout=None,
                        on_progress=None):
            captured.update(params)
            return {"n_layers": 4}

        with mock.patch.object(discover, "implicit_split_devices",
                               return_value=devices), \
                mock.patch.object(_loader, "native_lib_loaded", return_value=False), \
                mock.patch.object(discover, "list_gpus", return_value=([], "ok")), \
                mock.patch.object(discover, "resolve_auto_split_ratios",
                                  return_value=None), \
                mock.patch.object(GgufBackend, "_effective_ctx_max",
                                  return_value=4096), \
                mock.patch("localm.inference.backends.llamacpp._runner."
                           "ModelRunner.spawn_and_load", _fake_spawn), \
                mock.patch("localm.model_meta.store_n_layers"):
            b.effective_gpu_layers = 99
            b._load_native()
        assert isinstance(captured["gpu_split_ratios"], dict)
        assert sorted(captured["gpu_split_ratios"]) == [0, 1]
        assert b.applied_gpu_split["source"] == "auto"
        assert [d["index"] for d in b.applied_gpu_split["devices"]] == [0, 1]
        assert b._split_fit_note == ""


class TestReportingTheFit:
    def _backend(self, tmp_path):
        return GgufBackend(str(_tiny_model(tmp_path)), n_ctx=4096, n_gpu_layers=99)

    def test_every_left_out_device_is_named(self, tmp_path, caplog, capsys):
        plan = plan_split(_devices([20.0, 20.0, 1.2, 1.5]),
                          layer_bytes=[200 * MiB] * 40, layer_kv_bytes=[0] * 40,
                          output_bytes=500 * MiB, n_gpu_layers=99,
                          logits_bytes=GiB, reserve_bytes=int(0.5e9))
        b = self._backend(tmp_path)
        with caplog.at_level("INFO", logger="localm"):
            b._report_split_fit(plan)
        logged = "\n".join(r.getMessage() for r in caplog.records)
        assert "leaving out device(s) 2, 3, splitting over 0, 1" in logged
        shown = " ".join(capsys.readouterr().out.split())
        assert "device 3 has 1.5 GB free but would hold about" in shown
        assert "device 2 would then be short of memory as well" in shown
        assert b._split_fit_note == ""

    def test_no_fitting_split_leaves_a_note_for_the_failure_message(self, tmp_path):
        # Only device 1 is short, and leaving it out would leave one device.
        plan = plan_split(_devices([10.0, 2.0]), layer_bytes=[500 * MiB] * 8,
                          layer_kv_bytes=[0] * 8, output_bytes=500 * MiB,
                          n_gpu_layers=99, logits_bytes=GiB,
                          reserve_bytes=int(1.5e9))
        b = self._backend(tmp_path)
        b._report_split_fit(plan)
        assert b._split_fit_note.startswith("Device 1 has 2.0 GB free")
        assert "gpu_split_indices" in b._split_fit_note

    def test_the_note_reaches_the_load_failure_message(self, tmp_path):
        b = self._backend(tmp_path)
        b._split_fit_note = "Device 4 has 2.7 GB free but the split would place about 3.8 GB on it."
        with mock.patch.object(GgufBackend, "_effective_gpu_layers", return_value=99), \
                mock.patch.object(GgufBackend, "_check_vram"), \
                mock.patch.object(GgufBackend, "_split_free_total_bytes",
                                  return_value=(None, None, 0)), \
                mock.patch.object(GgufBackend, "_free_vram_bytes", return_value=None), \
                mock.patch.object(GgufBackend, "_load_native",
                                  side_effect=RuntimeError("Failed to create llama context")):
            with pytest.raises(RuntimeError) as ei:
                b.load()
        assert "Device 4 has 2.7 GB free" in str(ei.value)
        assert "setup-llama" not in str(ei.value)


class TestImplicitSplitDevices:
    def test_nvidia_smi_indices_are_not_offered_as_split_slots(self):
        gpus = [{"index": 0, "free": 10, "total": 20, "source": "nvidia-smi"},
                {"index": 1, "free": 10, "total": 20, "source": "nvidia-smi"}]
        with mock.patch.object(discover, "_native_gpu_index_space_is_opaque",
                               return_value=False), \
                mock.patch.object(discover, "_list_gpus_kw",
                                  return_value=(gpus, discover.GPU_PROBE_OK)):
            assert discover.implicit_split_devices({}) is None
            assert discover.implicit_split_capacity({})["devices"] == 2

    def test_torch_readings_are_returned_per_device(self):
        gpus = [{"index": 0, "name": "a", "free": 10, "total": 20},
                {"index": 1, "name": "b", "free": 5, "total": 20}]
        with mock.patch.object(discover, "_native_gpu_index_space_is_opaque",
                               return_value=False), \
                mock.patch.object(discover, "_list_gpus_kw",
                                  return_value=(gpus, discover.GPU_PROBE_OK)):
            assert discover.implicit_split_devices({}) == [
                {"index": 0, "free": 10, "total": 20},
                {"index": 1, "free": 5, "total": 20}]

    def test_configured_split_answers_none(self):
        assert discover.implicit_split_devices({"gpu_split_indices": [0, 1]}) is None
