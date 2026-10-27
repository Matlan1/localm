# SPDX-License-Identifier: AGPL-3.0-or-later
"""Placement of one GGUF model over several GPUs.

The planner (llamacpp/_gpu_plan.py) chooses, for each GPU a load uses, a
contiguous run of layers that fits its free memory; discover.gguf_split_devices
says which GPUs those are and where each sits in llama.cpp's own device list
(matched by PCI bus, so a runtime that numbers the cards in another order still
gets the cards the user chose); discover.apply_gpu_devices hands llama.cpp
exactly those devices and layer counts. The reported box: five NVIDIA cards of
15.5 / 23.5 / 23.5 / 15.5 / 5.6 GiB.
"""

import contextlib
import ctypes
import random
import time
from unittest import mock

import pytest

from localm import discover
from localm.inference.backends.llamacpp import _loader
from localm.inference.backends.llamacpp._gpu_plan import (ModelUnits, PlanDevice,
                                                         plan_placement)
from localm.inference.backends.llamacpp._split_fit import layer_devices

GiB = 1024 ** 3
MiB = 1024 ** 2

_TOTALS = [15.5, 23.5, 23.5, 15.5, 5.6]
_IDLE = [14.9, 23.1, 23.1, 15.1, 5.4]
_BUSY = [14.5, 22.5, 18.0, 4.0, 0.3]


def _devices(frees, ratios=None):
    return [PlanDevice(slot=i, index=i, name=f"gpu{i}", free=int(f * GiB),
                       total=int(_TOTALS[i] * GiB) if i < len(_TOTALS) else int(f * GiB),
                       device_id=f"0000:{0x10 * (i + 1):02x}:00.0",
                       ratio=None if ratios is None else ratios[i])
            for i, f in enumerate(frees)]


def _dense(n_layer=64, layer=444 * MiB, kv=64 * MiB, per_token=16384,
           output=int(1.5 * GiB), logits=int(1.2 * GiB), reserve=int(1.5e9)):
    """A ~30 GB dense model (Qwen2.5-32B shape) at a 4096-token window."""
    return ModelUnits(layer_bytes=[layer] * n_layer, expert_bytes=[0] * n_layer,
                      layer_kv=[kv] * n_layer, kv_per_token=[per_token] * n_layer,
                      output_bytes=output, logits_bytes=logits, reserve_bytes=reserve)


def _moe(n_layer=48, dense=200 * MiB, experts=1500 * MiB):
    return ModelUnits(layer_bytes=[dense + experts] * n_layer,
                      expert_bytes=[experts] * n_layer, layer_kv=[16 * MiB] * n_layer,
                      kv_per_token=[4096] * n_layer, output_bytes=600 * MiB,
                      logits_bytes=int(1.2 * GiB), reserve_bytes=int(1.5e9))


def _llama_counts(placement):
    """Layers per device and the output layer's device as llama.cpp places
    them from the placement's tensor_split (its own float32 arithmetic,
    _split_fit.layer_devices)."""
    positions, out = layer_devices([float(u) for u in placement.tensor_split],
                                   placement.n_layer, placement.n_gpu_layers)
    counts = [0] * len(placement.loads)
    for pos in positions:
        if pos is not None:
            counts[pos] += 1
    return counts, out


class TestPlanner:
    def test_reported_box_full_offload_fits_every_card(self):
        # A 32B model's real output layer (Q4) and one batch's logits.
        p = plan_placement(_devices(_IDLE), _dense(output=int(0.45 * GiB),
                                                   logits=int(0.3 * GiB)))
        assert p.fits and p.full_offload and p.n_cpu_moe == 0
        assert len(p.loads) == 5
        assert all(d.need <= d.device.free for d in p.loads)
        assert sum(d.units for d in p.loads) == 65
        assert [d.device.index for d in p.loads] == [0, 1, 2, 3, 4]
        out = p.loads[-1]
        assert out.output and out.device.index == 4, "output on the last card"

    def test_tensor_split_reproduces_the_planned_layers_in_llama_cpp(self):
        p = plan_placement(_devices(_BUSY), _dense())
        counts, out = _llama_counts(p)
        assert counts == [len(d.layers) for d in p.loads]
        assert out == len(p.loads) - 1

    def test_a_nearly_full_card_is_left_out_entirely(self):
        p = plan_placement(_devices(_BUSY), _dense())
        assert p.fits
        assert 4 not in [d.device.index for d in p.loads], "0.3 GB free holds nothing"

    def test_a_last_card_the_even_fill_does_not_reach_is_left_out(self):
        p = plan_placement(_devices(_IDLE), _dense())
        assert p.fits and p.full_offload
        assert [d.device.index for d in p.loads] == [0, 1, 2, 3], \
            "the even fill places every unit, the output layer included, on cards 0-3"
        assert p.loads[-1].output

    def test_the_cards_take_the_layers_in_the_order_given(self):
        devices = _devices(_IDLE)
        order = [3, 1, 4, 0, 2]
        p = plan_placement([devices[i] for i in order], _dense())
        assert p.fits and p.full_offload
        assert [d.device.index for d in p.loads] == order
        layers = [il for d in p.loads for il in d.layers]
        assert layers == sorted(layers) and p.loads[0].layers[0] == 0, \
            "card 3 holds the first layers, card 2 the last"
        assert p.loads[-1].output and not any(d.output for d in p.loads[:-1])
        counts, out = _llama_counts(p)
        assert counts == [len(d.layers) for d in p.loads] and out == len(p.loads) - 1

    def test_a_last_card_with_no_room_leaves_the_output_on_the_card_before(self):
        frees = [23.1, 23.1, 0.3]
        p = plan_placement(_devices(frees), _dense(n_layer=40))
        assert p.fits and p.full_offload
        assert [d.device.index for d in p.loads] == [0, 1]
        assert p.loads[-1].output and p.loads[-1].device.index == 1

    def test_moe_experts_go_to_ram_before_any_layer_leaves_the_gpus(self):
        frees = [8, 8]
        p = plan_placement(_devices(frees), _moe())
        assert p.fits and p.full_offload
        assert 0 < p.n_cpu_moe <= 48
        fewer = plan_placement(_devices(frees), _moe(), n_cpu_moe=p.n_cpu_moe - 1,
                               n_gpu_layers=49)
        assert not fewer.fits, "the chosen n_cpu_moe is the smallest that fits"

    def test_dense_model_too_big_offloads_the_most_layers_that_fit(self):
        p = plan_placement(_devices([6.0, 6.0]), _dense())
        assert p.fits and not p.full_offload and p.n_gpu_layers > 0
        more = plan_placement(_devices([6.0, 6.0]), _dense(),
                              n_gpu_layers=p.n_gpu_layers + 1)
        assert not more.fits

    def test_nothing_fits_is_cpu_only(self):
        p = plan_placement(_devices([1.0, 1.0]), _dense())
        assert p.loads == [] and p.n_gpu_layers == 0 and p.fits

    def test_fixed_layer_count_that_does_not_fit_reports_each_device(self):
        p = plan_placement(_devices([6.0, 6.0]), _dense(), n_gpu_layers=65)
        assert not p.fits and p.n_gpu_layers == 65
        assert any(not d.fits for d in p.loads)

    def test_pinned_ratios_set_the_layer_counts(self):
        p = plan_placement(_devices([20, 20], ratios=[3.0, 1.0]), _dense(),
                           n_gpu_layers=65)
        units = {d.device.index: d.units for d in p.loads}
        assert units[0] + units[1] == 65
        assert abs(units[0] - 3 * units[1]) <= 3

    def test_context_ceiling_is_set_by_the_tightest_card(self):
        p = plan_placement(_devices(_IDLE), _dense())
        ctx = p.max_ctx(4096)
        tight = min((d.device.free - d.need) // d.kv_per_token for d in p.loads)
        assert ctx == 4096 + tight

    def test_a_tight_box_still_gets_every_layer(self):
        # 10 units of 1 GiB (9 layers + the output layer) over cards with 4.0
        # and 9.6 GiB free: every unit fits only when each card keeps its 1.5 GB
        # reserve free, which leaves room for 2 and 8 units.
        units = ModelUnits(layer_bytes=[GiB] * 9, expert_bytes=[0] * 9,
                           layer_kv=[0] * 9, kv_per_token=[1] * 9,
                           output_bytes=GiB, logits_bytes=0,
                           reserve_bytes=int(1.5e9))
        p = plan_placement(_devices([4.0, 9.6]), units)
        assert p.fits and p.full_offload
        assert [d.units for d in p.loads] == [2, 8]

    def test_random_boxes_never_overfill_a_card(self):
        rng = random.Random(20261027)
        checked = 0
        for _ in range(400):
            n_dev = rng.randint(2, 6)
            frees = [rng.uniform(0.2, 24.0) for _ in range(n_dev)]
            n_layer = rng.randint(8, 80)
            units = ModelUnits(
                layer_bytes=[rng.randint(50, 900) * MiB for _ in range(n_layer)],
                expert_bytes=[0] * n_layer,
                layer_kv=[rng.randint(1, 120) * MiB for _ in range(n_layer)],
                kv_per_token=[rng.randint(1, 40000) for _ in range(n_layer)],
                output_bytes=rng.randint(100, 2000) * MiB,
                logits_bytes=rng.randint(10, 1500) * MiB,
                reserve_bytes=int(1.5e9), spread_bytes=rng.choice([0, 300 * MiB]))
            p = plan_placement(_devices(frees), units)
            assert p.fits
            assert all(d.need <= d.device.free for d in p.loads)
            if p.loads:
                counts, out = _llama_counts(p)
                assert counts == [len(d.layers) for d in p.loads]
                assert out == len(p.loads) - 1 and p.loads[-1].output
                checked += 1
        assert checked > 100


def _torch_reading(frees, order=None):
    """torch's reading of the box, cards in *order* (positions into _TOTALS)."""
    order = order or list(range(len(frees)))
    return [{"index": i, "name": f"card{c}", "total": int(_TOTALS[c] * GiB),
             "free": int(frees[c] * GiB), "free_scope": discover.FREE_SCOPE_DEVICE,
             "integrated": False, "pci_bus_id": 0x10 * (c + 1)}
            for i, c in enumerate(order)]


def _registry(frees, order):
    """The runtime's own device list: cards in *order*, with PCI ids."""
    return [{"index": k, "name": f"CUDA{k}", "description": f"card{c}",
             "type": _loader.GGML_DEV_TYPE_GPU, "free": int(frees[c] * GiB),
             "total": int(_TOTALS[c] * GiB),
             "device_id": f"0000:{0x10 * (c + 1):02x}:00.0"}
            for k, c in enumerate(order)]


@contextlib.contextmanager
def _box(readings, registry):
    discover._LLAMA_SLOTS_CACHE.clear()
    with mock.patch.object(discover, "_native_gpu_index_space_is_opaque",
                           return_value=False), \
            mock.patch.object(discover, "_list_gpus_kw",
                              return_value=(readings, discover.GPU_PROBE_OK)), \
            mock.patch.object(_loader, "gpu_devices_isolated", return_value=registry) as reg, \
            mock.patch.object(_loader, "probe_daemon_running", return_value=True), \
            mock.patch.object(_loader, "stop_probe_daemon"):
        yield reg
    discover._LLAMA_SLOTS_CACHE.clear()


class TestDeviceMatching:
    # llama.cpp's runtime lists the cards as torch cards 1, 2, 0, 3, 4.
    _LLAMA_ORDER = [1, 2, 0, 3, 4]

    def test_cards_are_matched_by_pci_bus_when_the_orders_differ(self):
        with _box(_torch_reading(_IDLE), _registry(_IDLE, self._LLAMA_ORDER)):
            devices, reason = discover.gguf_split_devices({"gpu_split_indices": None})
        assert reason == ""
        slot_of = {d["index"]: d["slot"] for d in devices}
        assert slot_of == {0: 2, 1: 0, 2: 1, 3: 3, 4: 4}
        assert all(d["device_id"] == f"0000:{0x10 * (d['index'] + 1):02x}:00.0"
                   for d in devices)

    def test_chosen_cards_are_the_ones_used_whatever_llama_cpp_calls_them(self):
        with _box(_torch_reading(_IDLE), _registry(_IDLE, self._LLAMA_ORDER)):
            devices, _ = discover.gguf_split_devices({"gpu_split_indices": [1, 2, 3, 4]})
        assert [d["index"] for d in devices] == [1, 2, 3, 4]
        assert 2 not in [d["slot"] for d in devices], "llama.cpp's slot 2 is card 0"

    def test_without_pci_ids_the_totals_must_match_position_by_position(self):
        readings = _torch_reading(_IDLE)
        for r in readings:
            r.pop("pci_bus_id")
        with _box(readings, _registry(_IDLE, self._LLAMA_ORDER)):
            devices, reason = discover.gguf_split_devices({"gpu_split_indices": None})
        assert devices is None and "memory" in reason

    def test_without_pci_ids_matching_totals_keep_the_positions(self):
        readings = _torch_reading(_IDLE)
        for r in readings:
            r.pop("pci_bus_id")
        reg = _registry(_IDLE, [0, 1, 2, 3, 4])
        for r in reg:
            r.pop("device_id")
        with _box(readings, reg):
            devices, reason = discover.gguf_split_devices({"gpu_split_indices": None})
        assert reason == "" and [d["slot"] for d in devices] == [0, 1, 2, 3, 4]

    def test_a_card_missing_from_the_runtime_is_not_guessed(self):
        reg = _registry(_IDLE, [1, 2, 3, 4])
        with _box(_torch_reading(_IDLE), reg):
            devices, reason = discover.gguf_split_devices({"gpu_split_indices": None})
        assert devices is None and "not in the llama.cpp runtime" in reason

    def test_chosen_cards_keep_the_order_they_were_chosen_in(self):
        with _box(_torch_reading(_IDLE), _registry(_IDLE, self._LLAMA_ORDER)):
            devices, _ = discover.gguf_split_devices({"gpu_split_indices": [3, 1, 4, 0]})
        assert [d["index"] for d in devices] == [3, 1, 4, 0]
        assert [d["slot"] for d in devices] == [3, 0, 4, 2]

    def test_with_no_card_chosen_the_cards_go_by_number(self):
        readings = list(reversed(_torch_reading(_IDLE)))
        with _box(readings, _registry(_IDLE, self._LLAMA_ORDER)):
            devices, _ = discover.gguf_split_devices({"gpu_split_indices": None})
        assert [d["index"] for d in devices] == [0, 1, 2, 3, 4]

    def test_ratios_pair_with_the_chosen_cards(self):
        with _box(_torch_reading(_IDLE), _registry(_IDLE, self._LLAMA_ORDER)):
            devices, _ = discover.gguf_split_devices(
                {"gpu_split_indices": [3, 1], "gpu_split_ratios": [1.0, 2.0]})
        assert [(d["index"], d["ratio"]) for d in devices] == [(3, 1.0), (1, 2.0)]

    def test_one_chosen_card_is_not_a_split(self):
        with _box(_torch_reading(_IDLE), _registry(_IDLE, self._LLAMA_ORDER)):
            devices, reason = discover.gguf_split_devices({"gpu_split_indices": [2]})
        assert devices is None and reason == "one GPU is chosen"

    def test_a_chosen_integrated_gpu_is_refused(self):
        readings = _torch_reading(_IDLE)
        readings[0]["integrated"] = True
        with _box(readings, _registry(_IDLE, [1, 2, 3, 4])), \
                pytest.raises(discover.GpuSplitConfigError):
            discover.gguf_split_devices({"gpu_split_indices": [0, 1]})

    def test_the_mapping_reads_the_registry_once(self):
        with _box(_torch_reading(_IDLE), _registry(_IDLE, self._LLAMA_ORDER)) as reg:
            discover.gguf_split_devices({"gpu_split_indices": None})
            discover.gguf_split_devices({"gpu_split_indices": [0, 1]})
        assert reg.call_count == 1


def _warnings(caplog):
    """The text of the WARNING records caplog holds."""
    return " | ".join(r.getMessage() for r in caplog.records if r.levelname == "WARNING")


class TestHfGpuOrder:
    """discover.hf_gpu_order: the GPUs an HF load must see, in the listed order."""

    @pytest.fixture(autouse=True)
    def _torch_numbering(self):
        with mock.patch.object(discover, "_native_gpu_index_space_is_opaque",
                               return_value=False):
            yield

    @pytest.mark.parametrize("listed, expected", [
        ([3, 1], [3, 1]),
        ([3, 1, 4, 0], [3, 1, 4, 0]),
        ([1, 3], None),
        ([3], None),
        ([3, 9], None),
        ([3, 9, 1], [3, 1]),
        (None, None),
    ])
    def test_the_listed_order_when_it_is_not_by_number(self, listed, expected):
        cfg = {"gpu_split_indices": listed}
        assert discover.hf_gpu_order(cfg, gpus=_torch_reading(_IDLE)) == expected

    def test_nvidia_smi_numbers_give_no_order_and_say_so(self, caplog):
        readings = [dict(g, numbering=discover.GPU_NUMBERING_NVIDIA_SMI)
                    for g in _torch_reading(_IDLE)]
        assert discover.hf_gpu_order({"gpu_split_indices": [3, 1]}, gpus=readings) is None
        assert "nvidia-smi numbers the GPUs its own way" in _warnings(caplog)
        assert "[3, 1]" in _warnings(caplog)

    def test_a_vulkan_or_sycl_build_gives_no_order_and_says_so(self, caplog):
        with mock.patch.object(discover, "_native_gpu_index_space_is_opaque",
                               return_value=True):
            assert discover.hf_gpu_order({"gpu_split_indices": [3, 1]},
                                         gpus=_torch_reading(_IDLE)) is None
        assert "numbers the GPUs its own way" in _warnings(caplog)

    def test_a_stale_reading_or_a_missing_gpu_says_so(self, caplog):
        with mock.patch.object(discover, "list_gpus",
                               return_value=([], discover.GPU_PROBE_TIMEOUT)):
            assert discover.hf_gpu_order({"gpu_split_indices": [3, 1]}) is None
        assert "no fresh GPU reading" in _warnings(caplog)
        caplog.clear()
        assert discover.hf_gpu_order({"gpu_split_indices": [9, 3]},
                                     gpus=_torch_reading(_IDLE)) is None
        assert "[9] are not detected" in _warnings(caplog)

    @pytest.mark.parametrize("listed", [[1, 3], [3], None])
    def test_a_list_by_number_or_none_says_nothing(self, caplog, listed):
        assert discover.hf_gpu_order({"gpu_split_indices": listed},
                                     gpus=_torch_reading(_IDLE)) is None
        assert caplog.text == ""


class _MP:
    devices = None
    tensor_split = None
    split_mode = 0
    main_gpu = 3


class TestApplyGpuDevices:
    @staticmethod
    @contextlib.contextmanager
    def _runtime(ids):
        """A runtime whose GPU-type devices are handles 101.. with PCI *ids*;
        yields the list of handles asked for their id."""
        asked = []

        def _pci(handle):
            asked.append(handle)
            return ids[handle - 101]
        with mock.patch.object(_loader, "gpu_device_handles",
                               return_value=[101 + k for k in range(len(ids))]), \
                mock.patch.object(_loader, "device_pci_id", side_effect=_pci), \
                mock.patch.object(discover, "_tensor_split_capacity", return_value=16):
            yield asked

    def test_planned_devices_in_the_planned_order(self):
        mp = _MP()
        with self._runtime(["0000:20:00.0", "0000:30:00.0", "0000:10:00.0"]):
            arr, split = discover.apply_gpu_devices(mp, [
                {"slot": 2, "device_id": "0000:10:00.0", "units": 12},
                {"slot": 1, "device_id": "0000:30:00.0", "units": 53}])
        assert [arr[0], arr[1], arr[2]] == [103, 102, None]
        assert mp.devices == ctypes.cast(arr, ctypes.c_void_p).value
        assert [split[0], split[1], split[2]] == [12.0, 53.0, 0.0]
        assert mp.split_mode == discover._LLAMA_SPLIT_MODE_LAYER and mp.main_gpu == 0

    def test_only_the_planned_devices_are_asked_for_their_id(self):
        # Asking a CUDA device for its id initialises it; a card outside the
        # plan must not be touched.
        with self._runtime(["0000:10:00.0", "0000:20:00.0", "0000:30:00.0",
                            "0000:40:00.0", "0000:50:00.0"]) as asked:
            discover.apply_gpu_devices(_MP(), [
                {"slot": 1, "device_id": "0000:20:00.0", "units": 3},
                {"slot": 3, "device_id": "0000:40:00.0", "units": 4}])
        assert sorted(asked) == [102, 104]

    def test_without_pci_ids_devices_are_found_by_slot(self):
        with self._runtime([None, None]) as asked:
            arr, _ = discover.apply_gpu_devices(_MP(), [
                {"slot": 1, "device_id": None, "units": 3},
                {"slot": 0, "device_id": None, "units": 4}])
        assert [arr[0], arr[1]] == [102, 101] and asked == []

    def test_a_device_with_another_pci_id_fails_the_load(self):
        with self._runtime(["0000:20:00.0", "0000:10:00.0"]), \
                pytest.raises(RuntimeError, match="not the planned"):
            discover.apply_gpu_devices(_MP(), [
                {"slot": 0, "device_id": "0000:10:00.0", "units": 3},
                {"slot": 1, "device_id": "0000:20:00.0", "units": 4}])

    def test_a_device_this_runtime_does_not_have_fails_the_load(self):
        with self._runtime(["0000:20:00.0"]), \
                pytest.raises(RuntimeError, match="not among"):
            discover.apply_gpu_devices(_MP(), [
                {"slot": 0, "device_id": "0000:20:00.0", "units": 3},
                {"slot": 1, "device_id": "0000:10:00.0", "units": 4}])


class TestCapacity:
    def test_a_stubbed_vram_info_is_still_the_answer_for_one_gpu(self):
        with mock.patch.object(discover, "vram_info", return_value={"total": 7}),                 mock.patch.object(discover, "_list_gpus_kw",
                                  side_effect=AssertionError("no second reading")):
            assert discover.vram_capacity({"gpu_split_indices": None}) == {"total": 7}

    def test_no_split_sums_every_discrete_gpu(self):
        readings = _torch_reading(_IDLE)
        with mock.patch.object(discover, "_list_gpus_kw", return_value=readings):
            info = discover.vram_capacity({"gpu_split_indices": None,
                                           "main_gpu_index": None})
        assert info["total"] == sum(int(t * GiB) for t in _TOTALS)
        assert info["free"] == sum(int(f * GiB) for f in _IDLE)

    def test_one_discrete_gpu_beside_an_integrated_one_is_that_gpu(self):
        readings = _torch_reading(_IDLE)[:2]
        readings[0]["integrated"] = True
        with mock.patch.object(discover, "_list_gpus_kw", return_value=readings):
            info = discover.vram_capacity({"gpu_split_indices": None,
                                           "main_gpu_index": None})
        assert info["total"] == int(_TOTALS[1] * GiB)

    def test_the_sum_takes_one_reading(self):
        readings = _torch_reading(_IDLE)
        with mock.patch.object(discover, "_list_gpus_kw", return_value=readings) as probe:
            discover.vram_capacity({"gpu_split_indices": None, "main_gpu_index": None})
        assert probe.call_count == 1

    def test_a_vulkan_build_sums_the_runtimes_own_gpus(self):
        # torch cannot see the cards of a vulkan build; the runtime can.
        native = [{"index": i, "name": f"Vulkan{i}", "total": 20 * GiB,
                   "free": 18 * GiB, "type": _loader.GGML_DEV_TYPE_GPU}
                  for i in range(2)]
        with mock.patch.object(discover, "_list_gpus_kw", return_value=[]), \
                mock.patch.object(discover, "_windows_largest_adapter_registry_entry",
                                  return_value=None), \
                mock.patch.object(discover, "_native_gpu_index_space_is_opaque",
                                  return_value=True), \
                mock.patch.object(discover, "native_gpu_devices", return_value=native):
            info = discover.vram_capacity({"gpu_split_indices": None,
                                           "main_gpu_index": None})
        assert info == {"total": 40 * GiB, "free": 36 * GiB}

    def test_a_chosen_main_gpu_is_that_gpu_alone(self):
        readings = _torch_reading(_IDLE)
        cfg = {"gpu_split_indices": None, "main_gpu_index": 1}
        with mock.patch.object(discover, "_list_gpus_kw", return_value=readings),                 mock.patch("localm.config.load_config", return_value=cfg):
            info = discover.vram_capacity(cfg)
        assert info["total"] == int(_TOTALS[1] * GiB)


# Three cards on PCI buses 0x10, 0x20 and 0x30. torch (fastest first) numbers
# them 0x30, 0x10, 0x20; nvidia-smi numbers them in bus order; llama.cpp lists
# them 0x20, 0x30, 0x10. The selector shows torch's numbers.
_TORCH_BUSES = [0x30, 0x10, 0x20]
_LLAMA_BUSES = [0x20, 0x30, 0x10]


def _torch_cards():
    return [{"index": i, "name": f"card{b:02x}", "total": 24 * GiB, "free": 20 * GiB,
             "free_scope": discover.FREE_SCOPE_DEVICE, "integrated": False,
             "pci_bus_id": b} for i, b in enumerate(_TORCH_BUSES)]


def _llama_cards():
    return [{"index": k, "name": f"CUDA{k}", "description": "",
             "type": _loader.GGML_DEV_TYPE_GPU, "free": 20 * GiB, "total": 24 * GiB,
             "device_id": f"0000:{b:02x}:00.0"} for k, b in enumerate(_LLAMA_BUSES)]


class _SmiProc:
    returncode = 0

    def __init__(self, text):
        self._text = text

    def communicate(self, timeout=None):
        return self._text, ""

    def kill(self):
        pass


def _probe(torch_out, *, torch_present=True, torch_unanswered=False):
    """discover._list_gpus_probe on the three-card box, torch answering
    *torch_out* (with *torch_unanswered*, an empty list is torch failing to
    answer) and nvidia-smi answering in bus order."""
    smi = "".join(f"{i}, card{b:02x}, 24576, 20480, 00000000:{b:02X}:00.0\n"
                  for i, b in enumerate(sorted(_TORCH_BUSES)))
    with mock.patch.object(discover, "_torch_gpu_probe_known_doomed",
                           return_value=not torch_present), \
            mock.patch.object(discover, "_isolated_torch_unavailable", torch_unanswered), \
            mock.patch.object(discover, "_torch_is_resident", return_value=False), \
            mock.patch.object(discover, "_torch_gpus_isolated_once",
                              return_value=torch_out), \
            mock.patch.object(discover, "_apply_device_global_free"), \
            mock.patch("subprocess.Popen", return_value=_SmiProc(smi)):
        return discover._list_gpus_probe()


class _ProbeProc(_SmiProc):
    def __init__(self, text, returncode):
        super().__init__(text)
        self.returncode = returncode


def _probe_with_child(torch_reply, torch_rc):
    """discover._list_gpus_probe on the three-card box with the real isolated
    torch probe path: its child answers *torch_reply* with exit code
    *torch_rc*, and nvidia-smi answers in bus order."""
    smi = "".join(f"{i}, card{b:02x}, 24576, 20480, 00000000:{b:02X}:00.0\n"
                  for i, b in enumerate(sorted(_TORCH_BUSES)))

    def _popen(args, *a, **k):
        if "nvidia-smi" in str(args[0]):
            return _ProbeProc(smi, 0)
        return _ProbeProc(torch_reply, torch_rc)

    with mock.patch.object(discover, "_torch_gpu_probe_known_doomed", return_value=False), \
            mock.patch.object(discover, "_torch_is_resident", return_value=False), \
            mock.patch.object(discover, "_apply_device_global_free"), \
            mock.patch("subprocess.Popen", side_effect=_popen):
        return discover._list_gpus_probe()


@pytest.fixture
def _fresh_probe_state():
    discover._reset_gpu_probe_cache()
    yield
    discover._reset_gpu_probe_cache()


@pytest.mark.usefixtures("_fresh_probe_state")
class TestSelectorNumbering:
    """A GPU number the user saved is the one the selector showed (torch's),
    whichever tool answered the reading a load is planned from."""

    def test_nvidia_smi_takes_torchs_numbers_after_a_torch_reading(self):
        first = _probe(_torch_cards())
        assert [(d["index"], d["pci_bus_id"]) for d in first] == \
            [(0, 0x30), (1, 0x10), (2, 0x20)]
        second = _probe([], torch_unanswered=True)
        assert [d["source"] for d in second] == [discover.GPU_SOURCE_NVIDIA_SMI] * 3
        assert [(d["index"], d["pci_bus_id"]) for d in second] == \
            [(0, 0x30), (1, 0x10), (2, 0x20)]
        assert {d["numbering"] for d in second} == {discover.GPU_NUMBERING_TORCH}

    def test_nvidia_smi_alone_on_a_torch_box_cannot_name_the_chosen_cards(self):
        readings = _probe([], torch_unanswered=True)
        assert [(d["index"], d["pci_bus_id"]) for d in readings] == \
            [(0, 0x10), (1, 0x20), (2, 0x30)]
        with _box(readings, _llama_cards()):
            for cfg in ({"gpu_split_indices": [1, 2]},
                        {"gpu_split_indices": None, "main_gpu_index": 1}):
                devices, reason = discover.gguf_split_devices(cfg, gpus=readings)
                assert devices is None and "cannot be matched" in reason, cfg
            devices, reason = discover.gguf_split_devices(
                {"gpu_split_indices": None}, gpus=readings)
        assert reason == "" and sorted(d["slot"] for d in devices) == [0, 1, 2]

    def test_a_failed_torch_child_marks_the_nvidia_smi_numbers(self):
        readings = _probe_with_child("[]", 1)
        assert [d.get("numbering") for d in readings] == \
            [discover.GPU_NUMBERING_NVIDIA_SMI] * 3
        with _box(readings, _llama_cards()):
            devices, reason = discover.gguf_split_devices(
                {"gpu_split_indices": [1, 2]}, gpus=readings)
        assert devices is None and "cannot be matched" in reason

    def test_torch_answering_no_gpu_leaves_the_nvidia_smi_numbers_unmarked(self):
        readings = _probe_with_child("[]", 0)
        assert [(d["index"], d["pci_bus_id"]) for d in readings] == \
            [(0, 0x10), (1, 0x20), (2, 0x30)]
        assert all("numbering" not in d for d in readings)

    @pytest.mark.parametrize("hang", [True, False])
    def test_a_resident_torch_read_that_fails_marks_the_round(self, hang):
        def _read():
            if hang:
                time.sleep(0.5)
                return []
            raise RuntimeError("driver gone")

        with mock.patch.object(discover, "_torch_gpus_resident", side_effect=_read):
            assert discover._torch_gpus_resident_bounded(timeout=0.05) == []
        assert discover._torch_round_unanswered is True

    def test_auto_ratios_are_equal_on_nvidia_smi_numbers(self):
        readings = _probe([], torch_unanswered=True)
        cfg = {"gpu_split_indices": [1, 2], "gpu_split_ratios": None}
        with mock.patch.object(discover, "_native_gpu_index_space_is_opaque",
                               return_value=False):
            assert discover.resolve_auto_split_ratios(cfg, gpus=readings) is None
            first = _probe(_torch_cards())
            assert discover.resolve_auto_split_ratios(cfg, gpus=first) is not None

    @pytest.mark.parametrize("torch_present", [True, False])
    def test_a_box_without_torch_keeps_nvidia_smis_numbers(self, torch_present):
        readings = _probe([], torch_present=torch_present)
        assert [(d["index"], d["pci_bus_id"]) for d in readings] == \
            [(0, 0x10), (1, 0x20), (2, 0x30)]
        assert all("numbering" not in d for d in readings)
        with _box(readings, _llama_cards()):
            devices, reason = discover.gguf_split_devices(
                {"gpu_split_indices": [1, 2]}, gpus=readings)
        assert reason == ""
        assert [(d["index"], d["slot"]) for d in devices] == [(1, 0), (2, 1)]


class TestBackendLoad:
    """GgufBackend sizes and loads a model over the chosen GPUs and hands the
    worker exactly those devices and layer counts."""

    def _backend(self, tmp_path, **kw):
        from localm.inference.backends.gguf import GgufBackend
        from tests.test_implicit_split_fit import _write_gguf
        tensors = [("token_embd.weight", 4000)]
        for il in range(8):
            tensors += [(f"blk.{il}.attn_q.weight", 3000), (f"blk.{il}.ffn.weight", 1000)]
        tensors += [("output_norm.weight", 64), ("output.weight", 5000)]
        path = _write_gguf(tmp_path / "m.gguf", arch="llama", block_count=8,
                           vocab=1, tensors=tensors)
        b = GgufBackend(str(path), n_ctx=4096, n_gpu_layers=99,
                        n_gpu_layers_auto=True, **kw)
        b._VRAM_OVERHEAD_BYTES = 1000
        b._gguf_kv_bpt = 1
        return b

    def _load(self, b, readings, registry, cfg):
        captured = {}

        def _spawn(self_runner, params, cancel_event=None, timeout=None,
                   on_progress=None):
            captured.update(params)
            return {"n_layers": 8}

        with _box(readings, registry),                 mock.patch("localm.config.load_config", return_value=cfg),                 mock.patch.object(discover, "list_gpus",
                                  side_effect=lambda *a, **k: (readings, "ok")
                                  if k.get("return_status") else readings),                 mock.patch.object(_loader, "native_lib_loaded", return_value=False),                 mock.patch("localm.inference.backends.llamacpp._runner."
                           "ModelRunner.spawn_and_load", _spawn),                 mock.patch("localm.model_meta.store_n_layers"):
            b.effective_gpu_layers = b._effective_gpu_layers()
            b._load_native()
        return captured

    def _world(self, frees_bytes, order):
        readings = [{"index": i, "name": f"card{i}", "total": f + 1000, "free": f,
                     "free_scope": discover.FREE_SCOPE_DEVICE, "integrated": False,
                     "pci_bus_id": 0x10 * (i + 1)} for i, f in enumerate(frees_bytes)]
        registry = [{"index": k, "name": f"CUDA{k}", "description": "",
                     "type": _loader.GGML_DEV_TYPE_GPU, "free": readings[c]["free"],
                     "total": readings[c]["total"],
                     "device_id": f"0000:{0x10 * (c + 1):02x}:00.0"}
                    for k, c in enumerate(order)]
        return readings, registry

    def test_worker_gets_the_chosen_cards_by_pci_id_and_their_layer_counts(self, tmp_path):
        from localm.config import load_config
        b = self._backend(tmp_path)
        readings, registry = self._world([30_000, 30_000, 30_000], [2, 0, 1])
        cfg = dict(load_config(), gpu_split_indices=[0, 2], gpu_split_ratios=None,
                   ctx_auto=False)
        params = self._load(b, readings, registry, cfg)
        devices = params["gpu_devices"]
        assert {d["device_id"] for d in devices} == {"0000:10:00.0", "0000:30:00.0"}
        assert {d["slot"] for d in devices} == {1, 0}, "card 0 is slot 1, card 2 is slot 0"
        assert sum(d["units"] for d in devices) == 9
        assert params["gpu_split_ratios"] is None
        assert params["main_gpu"] in {0, 1}, "the projector goes on a placed card"
        assert params["n_gpu_layers"] >= 9
        assert b.applied_gpu_split["source"] == "auto"
        assert sorted(d["index"] for d in b.applied_gpu_split["devices"]) == [0, 2]

    def test_a_model_too_big_for_the_cards_offloads_fewer_layers(self, tmp_path):
        from localm.config import load_config
        b = self._backend(tmp_path)
        readings, registry = self._world([20_000, 20_000], [0, 1])
        cfg = dict(load_config(), gpu_split_indices=None, ctx_auto=False)
        params = self._load(b, readings, registry, cfg)
        units = sum(d["units"] for d in params["gpu_devices"])
        assert 0 < units < 9 and params["n_gpu_layers"] == units
        assert b.last_gpu_sizing["layers"] == units

    @pytest.mark.usefixtures("_fresh_probe_state")
    @pytest.mark.parametrize("answered_by", ["torch", "nvidia-smi"])
    def test_a_saved_gpu_number_reaches_the_card_the_selector_shows(
            self, tmp_path, answered_by):
        from localm.config import load_config
        readings = _probe(_torch_cards())
        if answered_by == "nvidia-smi":
            readings = _probe([], torch_unanswered=True)
        assert next(d for d in readings if d["index"] == 1)["pci_bus_id"] == 0x10
        cfg = dict(load_config(), gpu_split_indices=None, gpu_split_ratios=None,
                   main_gpu_index=1, ctx_auto=False)
        params = self._load(self._backend(tmp_path), readings, _llama_cards(), cfg)
        assert len(params["gpu_devices"]) == 3
        assert params["main_gpu"] == _LLAMA_BUSES.index(0x10)
        cfg = dict(cfg, gpu_split_indices=[1, 2], main_gpu_index=None)
        params = self._load(self._backend(tmp_path), readings, _llama_cards(), cfg)
        assert {d["device_id"] for d in params["gpu_devices"]} == \
            {"0000:10:00.0", "0000:20:00.0"}
        assert {d["slot"] for d in params["gpu_devices"]} == \
            {_LLAMA_BUSES.index(0x10), _LLAMA_BUSES.index(0x20)}

    @pytest.mark.usefixtures("_fresh_probe_state")
    def test_a_split_on_nvidia_smi_numbers_is_refused_with_a_warning(self, tmp_path, caplog):
        import logging

        from localm.config import load_config
        readings = _probe([], torch_unanswered=True)
        cfg = dict(load_config(), gpu_split_indices=[1, 2], gpu_split_ratios=None,
                   ctx_auto=False)
        with caplog.at_level(logging.WARNING, logger="localm"):
            params = self._load(self._backend(tmp_path), readings, _llama_cards(), cfg)
        assert "gpu_devices" not in params or params["gpu_devices"] is None
        assert any(r.levelno == logging.WARNING and "cannot be matched" in r.getMessage()
                   for r in caplog.records)

    def test_the_worker_gets_the_chosen_cards_in_the_chosen_order(self, tmp_path):
        from localm.config import load_config
        b = self._backend(tmp_path)
        readings, registry = self._world([30_000, 30_000, 30_000], [2, 0, 1])
        cfg = dict(load_config(), gpu_split_indices=[2, 0], gpu_split_ratios=None,
                   ctx_auto=False)
        params = self._load(b, readings, registry, cfg)
        devices = params["gpu_devices"]
        assert [d["device_id"] for d in devices] == ["0000:30:00.0", "0000:10:00.0"]
        assert [d["slot"] for d in devices] == [0, 1], "card 2 is slot 0, card 0 is slot 1"
        assert [d["index"] for d in b.applied_gpu_split["devices"]] == [2, 0]

    @pytest.mark.usefixtures("_fresh_probe_state")
    def test_a_fallback_says_the_listed_order_is_not_applied(self, tmp_path, caplog):
        import logging

        from localm.config import load_config
        readings = _probe([], torch_unanswered=True)
        cfg = dict(load_config(), gpu_split_indices=[2, 1], gpu_split_ratios=None,
                   ctx_auto=False)
        with caplog.at_level(logging.WARNING, logger="localm"):
            self._load(self._backend(tmp_path), readings, _llama_cards(), cfg)
        assert any(r.levelno == logging.WARNING
                   and "not in the listed order [2, 1]" in r.getMessage()
                   for r in caplog.records)

    def test_the_projector_goes_on_the_placed_card_with_the_most_room(self, tmp_path):
        from localm.config import load_config
        b = self._backend(tmp_path)
        readings, registry = self._world([30_000, 90_000, 30_000], [2, 0, 1])
        cfg = dict(load_config(), gpu_split_indices=None, gpu_split_ratios=None,
                   main_gpu_index=None, ctx_auto=False)
        params = self._load(b, readings, registry, cfg)
        assert 2 in [d["slot"] for d in params["gpu_devices"]]
        assert params["main_gpu"] == 2, "card 1 (the most room left) is llama.cpp's slot 2"
