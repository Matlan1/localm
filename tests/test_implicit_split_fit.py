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

import contextlib
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


# The ggml device type llama.cpp b11118 gives an integrated GPU.
_GGML_DEV_TYPE_IGPU = 2


def _torch_readings(*specs):
    """list_gpus()-shaped torch readings, one per CUDA ordinal, from
    ``(free_gib, integrated)`` pairs."""
    return [{"index": i, "name": f"gpu{i}", "free": int(f * GiB),
             "total": int((f + 1) * GiB), "free_scope": discover.FREE_SCOPE_DEVICE,
             "integrated": ig}
            for i, (f, ig) in enumerate(specs)]


def _registry(gpus):
    """The native registry for the same box: ggml-cuda registers devices in
    ordinal order and types an integrated one IGPU."""
    return [{"index": i, "name": f"CUDA{i}", "description": "",
             "type": _GGML_DEV_TYPE_IGPU if g["integrated"] else _loader.GGML_DEV_TYPE_GPU,
             "free": g["free"], "total": g["total"]}
            for i, g in enumerate(gpus)]


@contextlib.contextmanager
def _torch_box(gpus, registry, *, daemon_running=True):
    """A CUDA/HIP build (index space not opaque) whose torch probe reports
    *gpus* and whose native registry reports *registry*."""
    with mock.patch.object(discover, "_native_gpu_index_space_is_opaque",
                           return_value=False), \
            mock.patch.object(discover, "_list_gpus_kw",
                              return_value=(gpus, discover.GPU_PROBE_OK)), \
            mock.patch.object(_loader, "gpu_devices_isolated",
                              return_value=registry) as reg, \
            mock.patch.object(_loader, "probe_daemon_running",
                              return_value=daemon_running), \
            mock.patch.object(_loader, "stop_probe_daemon") as stop:
        yield reg, stop


def _config(**overrides):
    """The real config with *overrides* applied."""
    from localm.config import load_config
    cfg = dict(load_config())
    cfg.update(overrides)
    return cfg


def _load_capturing(b, gpus, registry, registry_reads=None, cfg=None):
    """Run the backend's real ``_load_native`` on a torch box and return the
    params handed to the worker. Appends the number of registry reads to
    *registry_reads* when given. *cfg* replaces the loaded config."""
    captured = {}

    def _fake_spawn(self_runner, params, cancel_event=None, timeout=None,
                    on_progress=None):
        captured.update(params)
        return {"n_layers": 4}

    cfg_patch = (mock.patch("localm.config.load_config", return_value=cfg)
                 if cfg is not None else contextlib.nullcontext())
    def _no_gpus(*_a, **kw):
        return ([], "ok") if kw.get("return_status") else []

    with _torch_box(gpus, registry) as (reg, _stop), cfg_patch, \
            mock.patch.object(discover, "list_gpus", side_effect=_no_gpus), \
            mock.patch.object(_loader, "native_lib_loaded", return_value=False), \
            mock.patch.object(discover, "resolve_auto_split_ratios", return_value=None), \
            mock.patch.object(GgufBackend, "_effective_ctx_max", return_value=4096), \
            mock.patch("localm.inference.backends.llamacpp._runner."
                       "ModelRunner.spawn_and_load", _fake_spawn), \
            mock.patch("localm.model_meta.store_n_layers"):
        b.effective_gpu_layers = 99
        b._load_native()
    if registry_reads is not None:
        registry_reads.append(reg.call_count)
    return captured


def _worker_view(params):
    """``(split_mode, main_gpu, tensor_split values or None)`` the worker's
    ``apply_gpu_split`` writes into llama_model_params from *params*."""
    class _MP:
        tensor_split = None
        split_mode = discover._LLAMA_SPLIT_MODE_LAYER
        main_gpu = 0
    mp = _MP()
    with mock.patch.object(discover, "_tensor_split_capacity", return_value=8), \
            mock.patch.object(discover, "list_gpus",
                              side_effect=AssertionError("the worker must not probe")):
        arr = discover.apply_gpu_split(mp, config={},
                                       ratios_override=params["gpu_split_ratios"])
    values = None if arr is None else [arr[i] for i in range(8)]
    return mp.split_mode, mp.main_gpu, values


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

    def test_no_plan_is_written_when_no_device_holds_the_model_alone(self):
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

    def test_a_one_entry_mapping_loads_on_that_device_alone(self):
        mp = self._mp()
        mp.split_mode = discover._LLAMA_SPLIT_MODE_LAYER
        with mock.patch.object(discover, "list_gpus",
                               side_effect=AssertionError("must not probe")):
            assert discover.apply_gpu_split(
                mp, config={}, ratios_override={2: 1.0}) is None
        assert mp.tensor_split is None
        assert mp.split_mode == discover._LLAMA_SPLIT_MODE_NONE
        assert mp.main_gpu == 2

    def test_a_split_the_fit_chose_moves_an_unconfigured_main_gpu_quietly(self, caplog):
        mp = self._mp()
        with caplog.at_level("WARNING", logger="localm"), \
                mock.patch.object(discover, "_tensor_split_capacity", return_value=16):
            discover.apply_gpu_split(mp, config={}, ratios_override={1: 0.5, 2: 0.5})
        assert mp.main_gpu == 1
        assert not [r for r in caplog.records if "main_gpu_index" in r.getMessage()]


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

    @pytest.fixture(autouse=True)
    def _numbering_confirmed(self):
        with mock.patch.object(discover, "runtime_split_devices_match",
                               return_value=True):
            yield

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

    def test_a_cpu_load_skips_the_fit(self, tmp_path):
        b, devices = self._backend(tmp_path, [10**9, 10**9])
        with mock.patch.object(discover, "implicit_split_devices",
                               return_value=devices):
            assert b._implicit_split_fit(0) is None

    def test_experts_kept_in_system_ram_are_not_charged_to_any_device(self, tmp_path):
        tensors = [("token_embd.weight", 4000)]
        for il in range(4):
            tensors += [(f"blk.{il}.attn_q.weight", 3000),
                        (f"blk.{il}.ffn_up_exps.weight", 2000),
                        (f"blk.{il}.ffn_down_exps.weight", 2000)]
        tensors += [("output_norm.weight", 64), ("output.weight", 5000)]
        path = _write_gguf(tmp_path / "moe.gguf", arch="qwen3moe", block_count=4,
                           vocab=self._VOCAB, tensors=tensors)
        devices = [{"index": i, "free": 10**9, "total": 10**9 + 10} for i in range(2)]
        plans = {}
        for n_cpu_moe in (0, 3):
            b = GgufBackend(str(path), n_ctx=4096, n_gpu_layers=99, n_cpu_moe=n_cpu_moe)
            b._VRAM_OVERHEAD_BYTES = 1000
            b._gguf_kv_bpt = 10
            with mock.patch.object(discover, "implicit_split_devices",
                                   return_value=devices), \
                    mock.patch.object(_loader, "native_lib_loaded", return_value=False):
                plans[n_cpu_moe] = b._implicit_split_fit(99)
        weights = {n: sum(c.weights for c in plan.default) for n, plan in plans.items()}
        assert weights[0] == 4 * 7000
        assert weights[0] - weights[3] == 3 * 4000

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
        # Only device 1 is short, and device 0 cannot hold everything alone.
        plan = plan_split(_devices([5.0, 2.0]), layer_bytes=[500 * MiB] * 8,
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
        gpus = _torch_readings((10.0, False), (5.0, False))
        with _torch_box(gpus, _registry(gpus)):
            assert discover.implicit_split_devices({}) == [
                {"index": 0, "free": gpus[0]["free"], "total": gpus[0]["total"]},
                {"index": 1, "free": gpus[1]["free"], "total": gpus[1]["total"]}]

    def test_configured_split_answers_none(self):
        assert discover.implicit_split_devices({"gpu_split_indices": [0, 1]}) is None

    def test_a_single_gpu_answers_none_without_reading_the_registry(self):
        gpus = _torch_readings((23.0, False))
        with _torch_box(gpus, _registry(gpus)) as (reg, _stop):
            assert discover.implicit_split_devices({}) is None
        reg.assert_not_called()

    def test_vulkan_numbering_comes_from_the_native_registry_unchanged(self):
        native = [{"index": 0, "name": "a", "free": 10, "total": 20,
                   "type": _loader.GGML_DEV_TYPE_GPU},
                  {"index": 1, "name": "b", "free": 5, "total": 20,
                   "type": _loader.GGML_DEV_TYPE_GPU}]
        with mock.patch.object(discover, "_native_gpu_index_space_is_opaque",
                               return_value=True), \
                mock.patch.object(discover, "native_gpu_devices", return_value=native), \
                mock.patch.object(discover, "_runtime_device_registry") as reg:
            assert discover.implicit_split_devices({}) == [
                {"index": 0, "free": 10, "total": 20},
                {"index": 1, "free": 5, "total": 20}]
        reg.assert_not_called()

    def test_a_failure_logs_its_message(self, caplog):
        with caplog.at_level("DEBUG", logger="localm"), \
                mock.patch.object(discover, "_implicit_split_readings",
                                  side_effect=ValueError("bad reading")):
            assert discover.implicit_split_devices({}) is None
        assert any("bad reading" in r.getMessage() for r in caplog.records)


class TestTorchNumberingMatchesLlamaCpp:
    """llama.cpp drops integrated GPUs from its device list whenever a
    discrete GPU exists, while torch numbers every CUDA/HIP device. The fit
    writes a split only in llama.cpp's own numbering."""

    def _backend(self, tmp_path):
        b = GgufBackend(str(_tiny_model(tmp_path)), n_ctx=4096, n_gpu_layers=99)
        b._VRAM_OVERHEAD_BYTES = int(1.5e9)
        b._gguf_kv_bpt = 10
        return b

    def test_an_integrated_gpu_first_does_not_shift_the_split(self, tmp_path, capsys):
        # Torch ordinals: 0 = iGPU, 1 and 2 = discrete. llama.cpp's list is
        # [ordinal 1, ordinal 2]; a split keyed by torch ordinal would put
        # every layer on one card.
        gpus = _torch_readings((0.45, True), (23.0, False), (23.0, False))
        params = _load_capturing(self._backend(tmp_path), gpus, _registry(gpus))
        split_mode, _main, values = _worker_view(params)
        assert values is None, f"a tensor_split was written: {values}"
        assert split_mode == discover._LLAMA_SPLIT_MODE_LAYER
        assert "gpu split" not in capsys.readouterr().out

    @pytest.mark.parametrize("igpu_at", [1, 2])
    def test_an_integrated_gpu_elsewhere_writes_no_split(self, tmp_path, igpu_at):
        specs = [(23.0, False), (23.0, False)]
        specs.insert(igpu_at, (0.45, True))
        gpus = _torch_readings(*specs)
        params = _load_capturing(self._backend(tmp_path), gpus, _registry(gpus))
        assert _worker_view(params)[2] is None

    def test_one_discrete_gpu_beside_an_integrated_one_is_not_split_or_warned(
            self, tmp_path, capsys):
        gpus = _torch_readings((23.0, False), (0.45, True))
        b = self._backend(tmp_path)
        params = _load_capturing(b, gpus, _registry(gpus))
        assert params["gpu_split_ratios"] is None
        assert b._split_fit_note == ""
        assert "gpu split" not in capsys.readouterr().out

    def test_integrated_gpus_are_not_counted_in_the_sizing_budget(self):
        gpus = _torch_readings((0.45, True), (23.0, False), (23.0, False))
        with _torch_box(gpus, _registry(gpus)):
            info = discover.implicit_split_capacity({})
        assert info["devices"] == 2
        assert info["free"] == gpus[1]["free"] + gpus[2]["free"]

    @pytest.mark.parametrize("igpu_first", [True, False])
    def test_one_discrete_gpu_beside_an_integrated_one_is_what_sizing_reads(
            self, igpu_first):
        specs = [(2.0, True), (23.0, False)]
        if not igpu_first:
            specs.reverse()
        gpus = _torch_readings(*specs)
        dgpu = next(g for g in gpus if not g["integrated"])
        with _torch_box(gpus, _registry(gpus)):
            assert discover.implicit_split_capacity({}) == {
                "free": dgpu["free"], "total": dgpu["total"], "devices": 1}
            with mock.patch("localm.config.load_config", return_value=_config(
                    gpu_split_indices=None)), \
                    mock.patch.object(_loader, "native_lib_loaded", return_value=False):
                assert GgufBackend._split_free_total_bytes() == (
                    dgpu["free"], dgpu["total"], 1)
            assert discover.implicit_split_devices({}) is None

    def test_a_split_after_an_integrated_gpu_is_written_in_llamacpps_numbering(
            self, tmp_path):
        # Torch ordinal 0 is an iGPU; the discrete GPUs (ordinals 1-3) are
        # llama.cpp devices 0-2. The default split puts the output layer and a
        # 41 MB logits buffer on the 30 MB device, which is left out.
        b = GgufBackend(str(_tiny_model(tmp_path, vocab=5000)), n_ctx=4096,
                        n_gpu_layers=99)
        b._VRAM_OVERHEAD_BYTES = 1000
        b._gguf_kv_bpt = 10
        frees = [(2_000_000_000, True), (50_000_000, False), (50_000_000, False),
                 (30_000_000, False)]
        gpus = [{"index": i, "name": f"gpu{i}", "free": f, "total": f + 10,
                 "free_scope": discover.FREE_SCOPE_DEVICE, "integrated": ig}
                for i, (f, ig) in enumerate(frees)]
        params = _load_capturing(b, gpus, _registry(gpus))
        split_mode, main_gpu, values = _worker_view(params)
        assert values == pytest.approx([0.5, 0.5, 0, 0, 0, 0, 0, 0])
        assert (split_mode, main_gpu) == (discover._LLAMA_SPLIT_MODE_LAYER, 0)

    def test_discrete_gpus_are_renumbered_into_llamacpps_list(self):
        gpus = _torch_readings((0.45, True), (20.0, False), (10.0, False))
        with _torch_box(gpus, _registry(gpus)):
            devices = discover.implicit_split_devices({})
        assert devices == [
            {"index": 0, "free": gpus[1]["free"], "total": gpus[1]["total"]},
            {"index": 1, "free": gpus[2]["free"], "total": gpus[2]["total"]}]

    @pytest.mark.parametrize("case", [
        "runtime-has-more-gpus", "totals-differ", "registry-unreadable",
        "no-integrated-flag", "missing-ordinal", "process-scoped-free",
        "nvidia-smi"])
    def test_an_unproven_numbering_keeps_llamacpps_default(self, case, caplog):
        gpus = _torch_readings((20.0, False), (10.0, False))
        registry = _registry(gpus)
        if case == "runtime-has-more-gpus":
            registry.append(dict(registry[1], index=2))
        elif case == "totals-differ":
            registry[0]["total"] //= 2
        elif case == "registry-unreadable":
            registry = None
        elif case == "no-integrated-flag":
            del gpus[1]["integrated"]
        elif case == "missing-ordinal":
            gpus[1]["index"] = 2
        elif case == "process-scoped-free":
            gpus[0]["free_scope"] = discover.FREE_SCOPE_PROCESS
        else:
            for g in gpus:
                g["source"] = discover.GPU_SOURCE_NVIDIA_SMI
        with caplog.at_level("INFO", logger="localm"), _torch_box(gpus, registry):
            assert discover.implicit_split_devices({}) is None
        assert any("keeping llama.cpp's default split" in r.getMessage()
                   for r in caplog.records)

    def test_the_registry_read_stops_a_probe_daemon_it_started(self):
        gpus = _torch_readings((20.0, False), (10.0, False))
        with _torch_box(gpus, _registry(gpus), daemon_running=False) as (reg, stop):
            assert discover.implicit_split_devices({}) is not None
        reg.assert_called_once()
        stop.assert_called_once()
        with _torch_box(gpus, _registry(gpus), daemon_running=True) as (reg, stop):
            assert discover.implicit_split_devices({}) is not None
        stop.assert_not_called()

    def test_two_discrete_gpus_that_fit_keep_the_default_split(self, tmp_path):
        gpus = _torch_readings((20.0, False), (20.0, False))
        b = self._backend(tmp_path)
        reads = []
        params = _load_capturing(b, gpus, _registry(gpus), reads)
        assert params["gpu_split_ratios"] is None
        assert _worker_view(params) == (discover._LLAMA_SPLIT_MODE_LAYER, 0, None)
        assert b.applied_gpu_split is None
        assert reads == [0], "a plan that changes nothing must not read the registry"

    def test_a_plan_is_dropped_when_the_runtime_lists_other_gpus(self, tmp_path, capsys):
        # Without the registry check this layout writes {1: .5, 2: .5}.
        gpus = _torch_readings((0.45, False), (23.0, False), (23.0, False))
        registry = _registry(gpus)[1:]
        b = self._backend(tmp_path)
        reads = []
        params = _load_capturing(b, gpus, registry, reads)
        assert params["gpu_split_ratios"] is None
        assert b._split_fit_note == ""
        assert "gpu split" not in capsys.readouterr().out
        assert reads == [1]


class TestTorchProbeReportsIntegrated:
    def _torch(self, props):
        cuda = mock.Mock()
        cuda.is_available.return_value = True
        cuda.device_count.return_value = len(props)
        cuda.mem_get_info.side_effect = lambda i: (1, 2)
        cuda.get_device_name.side_effect = lambda i: f"gpu{i}"
        cuda.get_device_properties.side_effect = lambda i: props[i]
        return mock.Mock(cuda=cuda)

    def test_both_probe_paths_carry_the_flag_only_when_torch_reports_it(self):
        from localm import _torch_gpu_probe
        fake = self._torch([mock.Mock(is_integrated=1), mock.Mock(is_integrated=0),
                            mock.Mock(spec=[])])
        with mock.patch.dict("sys.modules", {"torch": fake}):
            for read in (_torch_gpu_probe._enumerate, discover._torch_gpus_resident):
                out = read()
                assert out[0]["integrated"] is True
                assert out[1]["integrated"] is False
                assert "integrated" not in out[2]


class TestNothingPlacedIsNotShort:
    """A device llama.cpp's default split gives no layer and no output holds
    nothing, so it is neither charged the per-device reserve nor reported."""

    def _backend(self, tmp_path):
        b = GgufBackend(str(_tiny_model(tmp_path)), n_ctx=4096, n_gpu_layers=99)
        b._VRAM_OVERHEAD_BYTES = int(1.5e9)
        b._gguf_kv_bpt = 10
        return b

    def test_a_small_device_given_nothing_is_not_warned_about(self, tmp_path, capsys):
        gpus = _torch_readings((23.0, False), (0.4, False))
        b = self._backend(tmp_path)
        params = _load_capturing(b, gpus, _registry(gpus))
        out = " ".join(capsys.readouterr().out.split())
        assert "creating the context may fail" not in out
        assert b._split_fit_note == ""
        assert params["gpu_split_ratios"] is None

    def test_a_third_small_device_given_nothing_is_not_left_out(self, tmp_path):
        gpus = _torch_readings((23.0, False), (23.0, False), (0.4, False))
        params = _load_capturing(self._backend(tmp_path), gpus, _registry(gpus))
        assert _worker_view(params) == (discover._LLAMA_SPLIT_MODE_LAYER, 0, None)

    def test_plan_charges_nothing_to_a_device_without_layers(self):
        plan = plan_split(_devices([20.0, 20.0, 1.0, 1.0]),
                          layer_bytes=[200 * MiB] * 20, layer_kv_bytes=[0] * 20,
                          output_bytes=500 * MiB, n_gpu_layers=99,
                          logits_bytes=GiB, reserve_bytes=int(1.5e9))
        empty = [c for c in plan.default if not c.layers and not c.holds_output]
        assert [c.index for c in empty] == [3]
        assert empty[0].need == 0 and empty[0].fits
        assert plan.excluded == [2]

    def test_negative_gpu_layers_offloads_every_layer(self):
        positions, out = layer_devices([1.0, 1.0], 8, -1)
        assert None not in positions and out == 1


class TestOneGpuWhenOnlyOneFits:
    """On two GPUs where only one can hold the model, the load runs on that
    GPU alone (llama.cpp's single-GPU mode), and a 1-entry gpu_split_indices
    does the same."""

    # The default split puts blocks 0-3 on device 0 and the output layer plus
    # a 41 MB logits buffer on device 1, which has 30 MB free.
    _VOCAB = 5000

    def test_the_load_runs_on_the_gpu_that_holds_everything(self, tmp_path, capsys):
        b = GgufBackend(str(_tiny_model(tmp_path, vocab=self._VOCAB)), n_ctx=4096,
                        n_gpu_layers=99)
        b._VRAM_OVERHEAD_BYTES = 1000
        b._gguf_kv_bpt = 10
        gpus = [{"index": i, "name": f"gpu{i}", "free": f, "total": f + 10,
                 "free_scope": discover.FREE_SCOPE_DEVICE, "integrated": False}
                for i, f in enumerate([60_000_000, 30_000_000])]
        params = _load_capturing(b, gpus, _registry(gpus))
        assert _worker_view(params) == (discover._LLAMA_SPLIT_MODE_NONE, 0, None)
        out = " ".join(capsys.readouterr().out.split())
        assert "loading on device 0 only" in out
        assert b._split_fit_note == ""
        assert b.applied_gpu_split is None

    def test_plan_keeps_one_device_only_when_it_holds_the_whole_charge(self):
        kw = dict(layer_bytes=[500 * MiB] * 8, layer_kv_bytes=[0] * 8,
                  output_bytes=500 * MiB, n_gpu_layers=99, logits_bytes=GiB,
                  reserve_bytes=int(1.5e9))
        plan = plan_split(_devices([10.0, 2.0]), **kw)
        assert plan.tensor_split == {0: 1.0} and plan.excluded == [1]
        assert plan.chosen[0].holds_output and plan.chosen[0].layers == 8
        assert plan_split(_devices([5.0, 2.0]), **kw).tensor_split is None

    def _one_entry_load(self, tmp_path, gpus, registry, index):
        b = GgufBackend(str(_tiny_model(tmp_path)), n_ctx=4096, n_gpu_layers=99)
        b._VRAM_OVERHEAD_BYTES = int(1.5e9)
        b._gguf_kv_bpt = 10
        params = _load_capturing(b, gpus, registry,
                                 cfg=_config(gpu_split_indices=[index],
                                             gpu_split_ratios=None))
        return b, params

    @pytest.mark.parametrize("specs,index,slot", [
        ([(20.0, False), (20.0, False)], 1, 1),
        ([(2.0, True), (20.0, False)], 1, 0),
        ([(2.0, True), (20.0, False), (20.0, False)], 2, 1),
    ])
    def test_a_one_entry_gpu_split_indices_loads_on_that_gpu_alone(
            self, tmp_path, specs, index, slot):
        gpus = _torch_readings(*specs)
        b, params = self._one_entry_load(tmp_path, gpus, _registry(gpus), index)
        assert params["gpu_split_ratios"] == {slot: 1.0}
        assert _worker_view(params) == (discover._LLAMA_SPLIT_MODE_NONE, slot, None)
        assert b.applied_gpu_split is None

    @pytest.mark.parametrize("case", ["integrated-gpu-unproven", "runtime-differs"])
    def test_an_unmatched_single_gpu_keeps_the_default(self, tmp_path, case, caplog):
        gpus = _torch_readings((2.0, True), (20.0, False))
        registry = _registry(gpus)
        index = 0 if case == "integrated-gpu-unproven" else 1
        registry.append(dict(registry[1], index=2))
        with caplog.at_level("INFO", logger="localm"):
            _b, params = self._one_entry_load(tmp_path, gpus, registry, index)
        assert params["gpu_split_ratios"] is None
        assert _worker_view(params) == (discover._LLAMA_SPLIT_MODE_LAYER, 0, None)
        assert any("keeping llama.cpp's default split" in r.getMessage()
                   for r in caplog.records)

    def test_the_worker_never_applies_a_one_entry_config_itself(self):
        class _MP:
            tensor_split = None
            split_mode = discover._LLAMA_SPLIT_MODE_LAYER
            main_gpu = 0
        mp = _MP()
        with mock.patch.object(discover, "list_gpus",
                               return_value=[{"index": 0}, {"index": 1}]),                 mock.patch.object(discover, "_native_gpu_index_space_is_opaque",
                                  return_value=False):
            assert discover.apply_gpu_split(
                mp, config={"gpu_split_indices": [1]}) is None
        assert (mp.split_mode, mp.main_gpu, mp.tensor_split) == (
            discover._LLAMA_SPLIT_MODE_LAYER, 0, None)

    def test_an_undetected_single_index_is_warned_and_unused(self, caplog):
        with caplog.at_level("WARNING", logger="localm"),                 mock.patch.object(discover, "_native_gpu_index_space_is_opaque",
                                  return_value=False):
            assert discover.single_gpu_index(
                [3], gpus=[{"index": 0}, {"index": 1}]) is None
        assert any("not one of the 2 GPU(s)" in r.getMessage() for r in caplog.records)

    def test_sizing_reads_the_gpu_a_one_entry_list_names(self):
        with mock.patch.object(discover, "list_gpus",
                               return_value=[{"index": 0}, {"index": 1}]), \
                mock.patch.object(discover, "_native_gpu_index_space_is_opaque",
                                  return_value=False):
            assert discover.resolve_load_gpu_index(
                {"gpu_split_indices": [1], "main_gpu_index": None}) == 1
            assert discover.resolve_load_gpu_index(
                {"gpu_split_indices": [0, 1], "main_gpu_index": 1}) == 1
            assert discover.resolve_load_gpu_index(
                {"gpu_split_indices": None, "main_gpu_index": None}) == 0


    def test_the_torch_free_read_uses_that_gpu(self):
        cuda = mock.Mock()
        cuda.is_available.return_value = True
        cuda.mem_get_info.side_effect = lambda i: (i * 100, i * 1000)
        with mock.patch.dict("sys.modules", {"torch": mock.Mock(cuda=cuda)}),                 mock.patch.object(discover, "resolve_load_gpu_index", return_value=1):
            assert GgufBackend._torch_free_total_uncapped() == (100, 1000)
        cuda.mem_get_info.assert_called_once_with(1)

    def test_the_device_global_correction_uses_that_gpu(self):
        seen = []

        def _used(entries):
            seen.extend(e["index"] for e in entries)
            return {1: 400}

        with mock.patch("localm.gpu_usage.raw_reading_is_process_scoped",
                        return_value=True),                 mock.patch("localm.gpu_usage.device_global_used_bytes", _used),                 mock.patch.object(discover, "resolve_load_gpu_index", return_value=1):
            assert GgufBackend._device_global_free_bytes(1000) == 600
        assert seen == [1]


class TestProbeDaemonStop:
    def test_stop_ends_a_running_daemon_process(self, monkeypatch):
        import subprocess
        import sys
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL)
        try:
            monkeypatch.setattr(_loader, "_PROBE_PROC", proc)
            assert _loader.probe_daemon_running()
            _loader.stop_probe_daemon(wait=10.0)
            assert proc.poll() is not None
            assert _loader._PROBE_PROC is None
            assert not _loader.probe_daemon_running()
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=10)
            for stream in (proc.stdin, proc.stdout):
                stream.close()
