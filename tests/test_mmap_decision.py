# SPDX-License-Identifier: AGPL-3.0-or-later
"""The RAM-aware mmap decision for a GGUF load.

A load with every layer on the GPU reads its host-resident weights (the input
layer and any routed experts n_cpu_moe pins to system RAM) into memory when
they fit available RAM, and keeps the runtime's mmap default when they do not,
so a model whose CPU-resident part is larger than RAM pages from the file
instead of failing or swapping. A configured use_mmap of on or off always wins.

The decision runs against REAL GGUF files built byte by byte; only the system
RAM reading is patched. The load-param layer drives LlamaCpp.__init__ with the
real llama_model_params structs of every layout.
"""

from unittest.mock import MagicMock, patch

import pytest

from localm.inference.backends.gguf import GgufBackend
from localm.inference.backends.llamacpp import llama as llama_mod
from localm.inference.backends.llamacpp._sizing import (
    HOST_RAM_HEADROOM_BYTES, MmapDecision, decide_use_mmap)
from localm.inference.backends.llamacpp._structs import (
    LLAMA_LOAD_MODE_AUTO, LLAMA_LOAD_MODE_MMAP, LLAMA_LOAD_MODE_NONE,
    LlamaModelParamsV1, LlamaModelParamsV2, LlamaModelParamsV3)
from tests.test_moe_auto_placement import EXPERT_BYTES, _dense_model, _moe_model

GIB = 1024 ** 3
PINNED_TWO = EXPERT_BYTES[0] + EXPERT_BYTES[1]


# --------------------------------------------------------------------------- #
#  decide_use_mmap: the rule itself                                           #
# --------------------------------------------------------------------------- #

class TestDecideUseMmap:
    @pytest.mark.parametrize("setting", ["on", "ON", " On "])
    @pytest.mark.parametrize("full", [True, False])
    def test_on_forces_mmap_whatever_the_ram(self, setting, full):
        d = decide_use_mmap(setting, full, 1, 64 * GIB, 64 * GIB)
        assert (d.use_mmap, d.reason) == (True, "user_on")

    @pytest.mark.parametrize("setting", ["off", "OFF"])
    @pytest.mark.parametrize("full", [True, False])
    def test_off_forces_no_mmap_even_when_the_weights_exceed_ram(self, setting, full):
        d = decide_use_mmap(setting, full, 100 * GIB, 8 * GIB, 1 * GIB)
        assert (d.use_mmap, d.reason) == (False, "user_off")

    def test_auto_full_offload_that_fits_reads_the_weights_into_memory(self):
        d = decide_use_mmap("auto", True, 4 * GIB, 32 * GIB, 16 * GIB)
        assert d == MmapDecision(False, "fits_ram", 4 * GIB, 32 * GIB, 16 * GIB)

    def test_auto_full_offload_that_exceeds_keeps_the_runtime_default(self):
        d = decide_use_mmap("auto", True, 40 * GIB, 32 * GIB, 16 * GIB)
        assert d == MmapDecision(None, "exceeds_ram", 40 * GIB, 32 * GIB, 16 * GIB)

    def test_the_headroom_boundary_is_inclusive(self):
        avail = 10 * GIB
        fits = decide_use_mmap("auto", True, avail - HOST_RAM_HEADROOM_BYTES, None, avail)
        over = decide_use_mmap("auto", True, avail - HOST_RAM_HEADROOM_BYTES + 1, None, avail)
        assert (fits.use_mmap, fits.reason) == (False, "fits_ram")
        assert (over.use_mmap, over.reason) == (None, "exceeds_ram")

    def test_unknown_ram_is_never_read_as_plenty(self):
        d = decide_use_mmap("auto", True, 1, None, None)
        assert (d.use_mmap, d.reason) == (None, "ram_unknown")

    def test_unknown_host_bytes_keep_the_runtime_default(self):
        d = decide_use_mmap("auto", True, None, 64 * GIB, 64 * GIB)
        assert (d.use_mmap, d.reason) == (None, "host_unknown")

    def test_auto_partial_offload_keeps_the_runtime_default(self):
        d = decide_use_mmap("auto", False, 1 * GIB, 64 * GIB, 32 * GIB)
        assert d == MmapDecision(None, "partial_offload", 1 * GIB, 64 * GIB, 32 * GIB)

    def test_a_partial_offload_whose_cpu_weights_exceed_ram_says_so(self):
        d = decide_use_mmap("auto", False, 100 * GIB, 8 * GIB, 1 * GIB)
        assert d == MmapDecision(None, "cpu_exceeds_ram", 100 * GIB, 8 * GIB, 1 * GIB)

    def test_the_partial_offload_headroom_boundary_is_inclusive(self):
        avail = 10 * GIB
        fits = decide_use_mmap("auto", False, avail - HOST_RAM_HEADROOM_BYTES, None, avail)
        over = decide_use_mmap("auto", False, avail - HOST_RAM_HEADROOM_BYTES + 1, None, avail)
        assert fits.reason == "partial_offload"
        assert over.reason == "cpu_exceeds_ram"

    @pytest.mark.parametrize("host, avail", [(None, 1), (100 * GIB, None)])
    def test_a_partial_offload_with_an_unknown_reading_is_not_flagged(self, host, avail):
        d = decide_use_mmap("auto", False, host, None, avail)
        assert (d.use_mmap, d.reason) == (None, "partial_offload")

    def test_off_keeps_the_ram_reading_for_the_load_warning(self):
        d = decide_use_mmap("off", False, 100 * GIB, 8 * GIB, 1 * GIB)
        assert d == MmapDecision(False, "user_off", 100 * GIB, 8 * GIB, 1 * GIB)

    @pytest.mark.parametrize("setting", [None, True, False, "", "maybe", 1])
    def test_anything_but_on_or_off_reads_as_auto(self, setting):
        d = decide_use_mmap(setting, True, 4 * GIB, 32 * GIB, 16 * GIB)
        assert (d.use_mmap, d.reason) == (False, "fits_ram")


# --------------------------------------------------------------------------- #
#  The sizing layer: host-resident bytes from the real file, RAM read once    #
# --------------------------------------------------------------------------- #

GPU = [{"index": 0, "name": "GPU 0", "total": 24 * GIB, "free": 20 * GIB}]


@pytest.fixture(autouse=True)
def _gpu_runtime():
    """A native runtime that ships a GPU backend, on a build whose GPU index
    space list_gpus can see."""
    with patch("localm.discover.native_backend_has_gpu", return_value=True), \
         patch("localm.discover._native_gpu_index_space_is_opaque", return_value=False):
        yield


def _backend(path, **kw):
    return GgufBackend(str(path), n_ctx=64, **kw)


def _ram(total, available):
    return patch("localm.sysstats.system_ram", MagicMock(return_value=(total, available)))


class TestHostResidentBytes:
    def test_a_dense_full_offload_keeps_only_the_input_layer_in_ram(self, tmp_path):
        host = _backend(_dense_model(tmp_path))._host_resident_bytes()
        assert 5_000 <= host < 5_000 + 32   # token_embd, padded to the alignment

    def test_pinned_experts_add_exactly_their_bytes(self, tmp_path):
        path = _moe_model(tmp_path)
        none_pinned = _backend(path)._host_resident_bytes()
        two_pinned = _backend(path, n_cpu_moe=2)._host_resident_bytes()
        assert two_pinned - none_pinned == PINNED_TWO

    def test_an_auto_chosen_n_cpu_moe_counts_like_a_configured_one(self, tmp_path):
        b = _backend(_moe_model(tmp_path))
        b.effective_n_cpu_moe = 2
        assert b._host_resident_bytes() == _backend(
            _moe_model(tmp_path), n_cpu_moe=2)._host_resident_bytes()

    def test_pinned_experts_with_an_unreadable_header_are_unknown(self, tmp_path):
        bad = tmp_path / "bad.gguf"
        bad.write_bytes(b"\0" * 4096)
        assert _backend(bad, n_cpu_moe=2)._host_resident_bytes() is None

    def test_a_missing_file_is_unknown(self, tmp_path):
        assert _backend(tmp_path / "gone.gguf")._host_resident_bytes() is None


class TestCpuHeldWeightBytes:
    """llama.cpp puts blocks from n_layer + 1 - n_gpu_layers up, and the
    output layer, on the GPU; the rest stays in system RAM."""

    def test_no_gpu_layers_hold_the_whole_file(self, tmp_path):
        b = _backend(_dense_model(tmp_path))
        assert b._cpu_held_weight_bytes(0) == b._model_bytes()

    def test_each_gpu_layer_takes_the_next_block_off_the_cpu(self, tmp_path):
        b = _backend(_moe_model(tmp_path))
        blocks = b._block_bytes()
        assert len({total for total, _ in blocks.values()}) == len(blocks)
        assert b._cpu_held_weight_bytes(1) - b._cpu_held_weight_bytes(2) == blocks[3][0]
        assert b._cpu_held_weight_bytes(2) - b._cpu_held_weight_bytes(3) == blocks[2][0]
        assert b._cpu_held_weight_bytes(4) - b._cpu_held_weight_bytes(5) == blocks[0][0]

    def test_every_block_on_the_gpu_leaves_the_input_layer(self, tmp_path):
        b = _backend(_dense_model(tmp_path))
        assert b._cpu_held_weight_bytes(5) == b._host_resident_bytes()
        assert b._cpu_held_weight_bytes(98) == b._host_resident_bytes()

    def test_pinned_experts_in_gpu_blocks_stay_in_ram(self, tmp_path):
        path = _moe_model(tmp_path)
        two = _backend(path, n_cpu_moe=2)._cpu_held_weight_bytes(3)
        three = _backend(path, n_cpu_moe=3)._cpu_held_weight_bytes(3)
        assert three - two == _backend(path)._block_bytes()[2][1]

    def test_pinned_experts_in_cpu_blocks_are_not_counted_twice(self, tmp_path):
        path = _moe_model(tmp_path)
        assert (_backend(path, n_cpu_moe=2)._cpu_held_weight_bytes(2)
                == _backend(path)._cpu_held_weight_bytes(2))

    def test_an_unreadable_header_is_unknown(self, tmp_path):
        bad = tmp_path / "bad.gguf"
        bad.write_bytes(b"\0" * 4096)
        assert _backend(bad)._cpu_held_weight_bytes(24) is None
        assert _backend(bad)._cpu_held_weight_bytes(0) == 4096

    def test_a_missing_file_is_unknown(self, tmp_path):
        assert _backend(tmp_path / "gone.gguf")._cpu_held_weight_bytes(0) is None


class TestCpuDraftWeightBytes:
    def _draft_backend(self, tmp_path, on_gpu):
        draft = tmp_path / "draft.gguf"
        draft.write_bytes(b"\0" * 7_000)
        b = _backend(_dense_model(tmp_path))
        b.spec_source = "draft"
        b.spec_draft_model = str(draft)
        b.draft_model_on_gpu = on_gpu
        return b

    def test_a_cpu_draft_model_counts_its_file(self, tmp_path):
        assert self._draft_backend(tmp_path, False)._cpu_draft_weight_bytes() == 7_000

    def test_a_gpu_draft_model_counts_nothing(self, tmp_path):
        assert self._draft_backend(tmp_path, True)._cpu_draft_weight_bytes() == 0

    def test_a_missing_draft_file_counts_nothing(self, tmp_path):
        b = self._draft_backend(tmp_path, False)
        b.spec_draft_model = str(tmp_path / "gone.gguf")
        assert b._cpu_draft_weight_bytes() == 0

    def test_a_cpu_draft_joins_the_ram_check(self, tmp_path):
        b = self._draft_backend(tmp_path, False)
        base = b._host_resident_bytes()
        with _ram(64 * GIB, base + 7_000 + HOST_RAM_HEADROOM_BYTES - 1):
            d = b._resolve_use_mmap(99, GPU)
        assert (d.reason, d.host_bytes) == ("cpu_exceeds_ram", base + 7_000)

    def test_a_cpu_draft_that_fits_is_a_plain_full_offload(self, tmp_path):
        b = self._draft_backend(tmp_path, False)
        with _ram(64 * GIB, 32 * GIB):
            d = b._resolve_use_mmap(99, GPU)
        assert d.reason == "fits_ram"

    def test_with_no_gpu_the_draft_counts_wherever_it_was_placed(self, tmp_path):
        b = self._draft_backend(tmp_path, True)
        with _ram(64 * GIB, 64 * GIB):
            d = b._resolve_use_mmap(99, [])
        assert (d.reason, d.host_bytes) == ("no_gpu", b._model_bytes() + 7_000)


class TestResolveUseMmap:
    def test_moe_experts_that_fit_ram_load_unmapped(self, tmp_path):
        b = _backend(_moe_model(tmp_path), n_cpu_moe=2)
        host = b._host_resident_bytes()
        with _ram(64 * GIB, host + HOST_RAM_HEADROOM_BYTES) as probe:
            d = b._resolve_use_mmap(99, GPU)
        assert d == MmapDecision(False, "fits_ram", host, 64 * GIB,
                                 host + HOST_RAM_HEADROOM_BYTES)
        assert b.last_mmap_decision == d
        assert probe.call_count == 1

    def test_moe_experts_larger_than_ram_keep_mmap(self, tmp_path):
        b = _backend(_moe_model(tmp_path), n_cpu_moe=2)
        host = b._host_resident_bytes()
        with _ram(64 * GIB, host + HOST_RAM_HEADROOM_BYTES - 1):
            d = b._resolve_use_mmap(99, GPU)
        assert (d.use_mmap, d.reason, d.host_bytes) == (None, "cpu_exceeds_ram", host)

    def test_an_unreadable_ram_reading_keeps_mmap(self, tmp_path):
        b = _backend(_moe_model(tmp_path), n_cpu_moe=2)
        with _ram(None, None):
            d = b._resolve_use_mmap(99, GPU)
        assert (d.use_mmap, d.reason) == (None, "ram_unknown")

    def test_a_dense_full_offload_with_room_stays_unmapped(self, tmp_path):
        b = _backend(_dense_model(tmp_path))
        with _ram(16 * GIB, 8 * GIB):
            d = b._resolve_use_mmap(99, GPU)
        assert (d.use_mmap, d.reason) == (False, "fits_ram")

    def test_a_dense_full_offload_on_a_full_box_keeps_mmap(self, tmp_path):
        b = _backend(_dense_model(tmp_path))
        with _ram(16 * GIB, 1 * GIB):
            d = b._resolve_use_mmap(99, GPU)
        assert (d.use_mmap, d.reason) == (None, "exceeds_ram")

    def test_on_wins_and_reads_no_ram(self, tmp_path):
        b = _backend(_moe_model(tmp_path), n_cpu_moe=2)
        b.use_mmap = "on"
        with _ram(64 * GIB, 1024 * GIB) as probe:
            d = b._resolve_use_mmap(99, GPU)
        assert d.use_mmap is True
        assert probe.call_count == 0

    def test_off_wins_and_keeps_the_ram_reading(self, tmp_path):
        b = _backend(_moe_model(tmp_path), n_cpu_moe=2)
        b.use_mmap = "off"
        with _ram(64 * GIB, 1) as probe:
            d = b._resolve_use_mmap(99, GPU)
        assert d == MmapDecision(False, "user_off", b._host_resident_bytes(), 64 * GIB, 1)
        assert probe.call_count == 1

    @pytest.mark.parametrize("layers", [0, 3, 98])
    def test_a_partial_offload_keeps_the_default(self, tmp_path, layers):
        b = _backend(_moe_model(tmp_path))
        with _ram(64 * GIB, 64 * GIB) as probe:
            d = b._resolve_use_mmap(layers, GPU)
        assert d == MmapDecision(None, "partial_offload", b._cpu_held_weight_bytes(layers),
                                 64 * GIB, 64 * GIB)
        assert probe.call_count == 1

    @pytest.mark.parametrize("layers", [0, 3, 98])
    def test_a_partial_offload_whose_cpu_weights_exceed_ram_says_so(self, tmp_path, layers):
        b = _backend(_moe_model(tmp_path))
        held = b._cpu_held_weight_bytes(layers)
        with _ram(64 * GIB, held + HOST_RAM_HEADROOM_BYTES - 1):
            d = b._resolve_use_mmap(layers, GPU)
        assert (d.use_mmap, d.reason, d.host_bytes) == (None, "cpu_exceeds_ram", held)

    def test_a_failing_host_byte_count_never_raises(self, tmp_path):
        b = _backend(_moe_model(tmp_path))
        with patch.object(GgufBackend, "_host_resident_bytes",
                          side_effect=ValueError("corrupt")), _ram(64 * GIB, 64 * GIB) as probe:
            d = b._resolve_use_mmap(99, GPU)
        assert (d.use_mmap, d.reason) == (None, "host_unknown")
        assert probe.call_count == 0


class TestNoGpu:
    """99 GPU layers with no GPU to hold them puts the whole model in system
    RAM, so the load keeps the runtime default however little the input layer
    and experts weigh."""

    def test_no_gpu_reading_keeps_the_default(self, tmp_path):
        b = _backend(_dense_model(tmp_path))
        with _ram(64 * GIB, 64 * GIB) as probe:
            d = b._resolve_use_mmap(99, [])
        assert d == MmapDecision(None, "no_gpu", b._model_bytes(), 64 * GIB, 64 * GIB)
        assert probe.call_count == 1

    def test_no_gpu_with_a_model_larger_than_ram_says_so(self, tmp_path):
        b = _backend(_dense_model(tmp_path))
        with _ram(64 * GIB, b._model_bytes() + HOST_RAM_HEADROOM_BYTES - 1):
            d = b._resolve_use_mmap(99, [])
        assert (d.use_mmap, d.reason) == (None, "cpu_exceeds_ram")

    def test_a_cpu_only_runtime_keeps_the_default_despite_a_gpu_reading(self, tmp_path):
        b = _backend(_dense_model(tmp_path))
        with patch("localm.discover.native_backend_has_gpu", return_value=False), \
             _ram(64 * GIB, 64 * GIB):
            d = b._resolve_use_mmap(99, GPU)
        assert (d.use_mmap, d.reason) == (None, "no_gpu")

    def test_an_unreadable_runtime_dir_falls_back_to_the_gpu_reading(self, tmp_path):
        b = _backend(_dense_model(tmp_path))
        with patch("localm.discover.native_backend_has_gpu", return_value=None), \
             _ram(64 * GIB, 64 * GIB):
            d = b._resolve_use_mmap(99, GPU)
        assert (d.use_mmap, d.reason) == (False, "fits_ram")

    @pytest.mark.parametrize("registry, want", [
        (None, "no_gpu"), ([], "no_gpu"), ([{"index": 0, "name": "Vulkan0"}], "fits_ram")])
    def test_a_vulkan_or_sycl_build_asks_the_native_registry(self, tmp_path, registry, want):
        b = _backend(_dense_model(tmp_path))
        with patch("localm.discover._native_gpu_index_space_is_opaque", return_value=True), \
             patch("localm.discover.native_gpu_devices", return_value=registry), \
             _ram(64 * GIB, 64 * GIB):
            d = b._resolve_use_mmap(99, [])
        assert d.reason == want

    def test_a_failing_gpu_check_never_raises(self, tmp_path):
        b = _backend(_dense_model(tmp_path))
        with patch("localm.discover.native_backend_has_gpu",
                   side_effect=OSError("unreadable")), _ram(64 * GIB, 64 * GIB):
            d = b._resolve_use_mmap(99, GPU)
        assert (d.use_mmap, d.reason) == (None, "no_gpu")

    @pytest.mark.parametrize("layers", [1, 3, 98])
    def test_partial_layers_with_no_gpu_count_the_whole_file(self, tmp_path, layers):
        b = _backend(_moe_model(tmp_path))
        with _ram(64 * GIB, 64 * GIB):
            d = b._resolve_use_mmap(layers, [])
        assert (d.use_mmap, d.reason, d.host_bytes) == (None, "no_gpu", b._model_bytes())

    def test_partial_layers_with_no_gpu_and_a_model_larger_than_ram_say_so(self, tmp_path):
        b = _backend(_moe_model(tmp_path))
        with _ram(64 * GIB, b._model_bytes() + HOST_RAM_HEADROOM_BYTES - 1):
            d = b._resolve_use_mmap(3, [])
        assert d.reason == "cpu_exceeds_ram"

    def test_an_explicit_off_still_wins_without_a_gpu(self, tmp_path):
        b = _backend(_dense_model(tmp_path))
        b.use_mmap = "off"
        with _ram(64 * GIB, 64 * GIB):
            d = b._resolve_use_mmap(99, [])
        assert (d.use_mmap, d.reason, d.host_bytes) == (False, "user_off", b._model_bytes())


# --------------------------------------------------------------------------- #
#  GgufBackend._load_native: the decision reaches the worker, the report back #
# --------------------------------------------------------------------------- #

def _load(backend, meta_mmap, gpus=GPU):
    seen = {}

    def _spawn(params, **_kw):
        seen.update(params)
        return {"n_layers": 4, "kv_bytes_per_token": 0, "supports_images": False,
                "weight_placement": [], "moe_skip_reason": None, "mmap": meta_mmap}

    with patch("localm.discover.list_gpus", return_value=(list(gpus), "ok")), \
         patch("localm.inference.backends.llamacpp._runner.ModelRunner."
               "spawn_and_load", side_effect=_spawn):
        backend._load_native()
    return seen


class TestLoadNativeWiring:
    def test_experts_larger_than_ram_reach_the_worker_as_the_default(self, tmp_path):
        b = _backend(_moe_model(tmp_path), n_cpu_moe=2)
        with _ram(64 * GIB, 1 * GIB):
            params = _load(b, True)
        assert "use_mmap" in params and params["use_mmap"] is None
        assert b.effective_use_mmap is True
        assert b.mmap_forced_by_ram is True

    def test_experts_that_fit_reach_the_worker_as_forced_off(self, tmp_path):
        b = _backend(_moe_model(tmp_path), n_cpu_moe=2)
        with _ram(64 * GIB, 32 * GIB):
            params = _load(b, None)
        assert params["use_mmap"] is False
        assert b.effective_use_mmap is False
        assert b.mmap_forced_by_ram is False

    def test_the_worker_report_wins_over_the_requested_mode(self, tmp_path):
        b = _backend(_moe_model(tmp_path), n_cpu_moe=2)
        b.use_mmap = "on"
        with _ram(64 * GIB, 32 * GIB):
            params = _load(b, False)
        assert params["use_mmap"] is True
        assert b.effective_use_mmap is False
        assert b.mmap_forced_by_ram is False

    def test_an_unreported_default_stays_unknown(self, tmp_path):
        b = _backend(_moe_model(tmp_path), n_cpu_moe=2)
        with _ram(64 * GIB, 1 * GIB):
            _load(b, None)
        assert b.effective_use_mmap is None
        assert b.mmap_forced_by_ram is False

    @pytest.mark.parametrize("setting, want", [("off", False), ("on", True)])
    def test_the_configured_setting_reaches_the_worker_through_the_constructor(
            self, tmp_path, setting, want):
        b = _backend(_moe_model(tmp_path), n_cpu_moe=2, use_mmap=setting)
        free = 1 * GIB if setting == "off" else 64 * GIB
        with _ram(64 * GIB, free):
            params = _load(b, want)
        assert params["use_mmap"] is want
        assert b.last_mmap_decision.reason == f"user_{setting}"
        assert b.effective_use_mmap is want
        assert b.mmap_forced_by_ram is False

    def test_the_loads_own_gpu_reading_decides_whether_layers_reach_a_gpu(self, tmp_path):
        b = _backend(_moe_model(tmp_path), n_cpu_moe=2)
        with _ram(64 * GIB, 32 * GIB):
            params = _load(b, True, gpus=[])
        assert "use_mmap" in params and params["use_mmap"] is None
        assert b.last_mmap_decision.reason == "no_gpu"
        assert b.effective_use_mmap is True
        assert b.mmap_forced_by_ram is False

    def test_a_mapped_load_with_an_unknown_ram_reading_is_not_flagged(self, tmp_path):
        b = _backend(_moe_model(tmp_path))
        with _ram(None, None):
            _load(b, True)
        assert b.effective_use_mmap is True
        assert b.mmap_forced_by_ram is False

    def test_a_partial_load_whose_cpu_weights_exceed_ram_is_flagged(self, tmp_path):
        b = _backend(_moe_model(tmp_path), n_gpu_layers=2)
        with _ram(64 * GIB, 1 * GIB):
            params = _load(b, True)
        assert params["n_gpu_layers"] == 2 and params["use_mmap"] is None
        assert b.last_mmap_decision.reason == "cpu_exceeds_ram"
        assert b.mmap_forced_by_ram is True
        assert b.mmap_cpu_from_disk is True

    def test_a_partial_load_that_fits_is_not_flagged(self, tmp_path):
        b = _backend(_moe_model(tmp_path), n_gpu_layers=2)
        with _ram(64 * GIB, 32 * GIB):
            _load(b, True)
        assert b.mmap_forced_by_ram is False
        assert b.mmap_cpu_from_disk is False

    def test_a_dense_full_offload_exceeding_ram_is_not_a_cpu_layer_load(self, tmp_path):
        b = _backend(_dense_model(tmp_path))
        with _ram(64 * GIB, 1 * GIB):
            _load(b, True)
        assert b.mmap_forced_by_ram is True
        assert b.mmap_cpu_from_disk is False

    def test_a_full_offload_with_pinned_experts_exceeding_ram_reads_them_every_token(
            self, tmp_path):
        b = _backend(_moe_model(tmp_path), n_cpu_moe=2)
        with _ram(64 * GIB, 1 * GIB):
            _load(b, True)
        assert b.mmap_forced_by_ram is True
        assert b.mmap_cpu_from_disk is True

    def test_an_unmapped_load_larger_than_ram_warns_before_it_starts(self, tmp_path):
        b = _backend(_moe_model(tmp_path), use_mmap="off")
        order = []
        with _ram(64 * GIB, 1 * GIB), \
             patch("localm.inference.backends.gguf.console.print",
                   lambda *a, **k: order.append(str(a[0]) if a else "")), \
             patch("localm.debuglog.logger.warning") as warned:
            _load_ordered(b, order)
        first = next(i for i, line in enumerate(order) if "use_mmap is off" in line)
        assert first < order.index("spawn")
        assert "may fail or swap" in order[first]
        assert warned.call_count == 1
        assert b.mmap_ram_short_note and "may fail or swap" in b.mmap_ram_short_note

    def test_an_unmapped_load_that_fits_does_not_warn(self, tmp_path):
        b = _backend(_moe_model(tmp_path), use_mmap="off")
        order = []
        with _ram(64 * GIB, 32 * GIB), \
             patch("localm.inference.backends.gguf.console.print",
                   lambda *a, **k: order.append(str(a[0]) if a else "")):
            _load_ordered(b, order)
        assert not any("use_mmap is off" in line for line in order)
        assert b.mmap_ram_short_note == ""


def _load_ordered(backend, order):
    def _spawn(params, **_kw):
        order.append("spawn")
        return {"n_layers": 4, "kv_bytes_per_token": 0, "supports_images": False,
                "weight_placement": [], "moe_skip_reason": None, "mmap": False}

    with patch("localm.discover.list_gpus", return_value=(list(GPU), "ok")), \
         patch("localm.inference.backends.llamacpp._runner.ModelRunner."
               "spawn_and_load", side_effect=_spawn):
        backend._load_native()


# --------------------------------------------------------------------------- #
#  LlamaCpp.__init__: the load mode written into the real struct              #
# --------------------------------------------------------------------------- #

def _native_default(cls):
    """A llama_model_params of *cls* with the native default load mode: V1
    use_mmap true, V2 (b10270) MMAP, V3 (b11118) AUTO."""
    mp = cls()
    if cls is LlamaModelParamsV1:
        mp.use_mmap = True
    elif cls is LlamaModelParamsV2:
        mp.load_mode = LLAMA_LOAD_MODE_MMAP
    else:
        mp.load_mode = LLAMA_LOAD_MODE_AUTO
    return mp


def _load_params(monkeypatch, cls, **kw):
    """The params LlamaCpp.__init__ hands llama_load_model_from_file for *kw*
    (the native load is faked to return NULL, ending __init__)."""
    seen = []
    monkeypatch.setattr(llama_mod.api, "llama_backend_init", lambda: None)
    monkeypatch.setattr(llama_mod.api, "llama_model_default_params",
                        lambda: _native_default(cls))
    monkeypatch.setattr(llama_mod.api, "llama_load_model_from_file",
                        lambda path, params: seen.append(params))
    with patch("localm.discover.apply_main_gpu", lambda mp, **_k: None), \
         patch("localm.discover.apply_gpu_split", lambda mp, ratios_override=None: None):
        with pytest.raises(RuntimeError, match="Failed to load model"):
            llama_mod.LlamaCpp("m.gguf", n_ctx=64, verbose=False, **kw)
    (mp,) = seen
    return mp


def _mode(mp):
    return mp.use_mmap if isinstance(mp, LlamaModelParamsV1) else mp.load_mode


LAYOUTS = [LlamaModelParamsV1, LlamaModelParamsV2, LlamaModelParamsV3]


class TestLoadParamLayer:
    @pytest.mark.parametrize("cls", LAYOUTS)
    def test_false_writes_no_mmap(self, monkeypatch, cls):
        mp = _load_params(monkeypatch, cls, n_gpu_layers=99, use_mmap=False)
        assert _mode(mp) == (False if cls is LlamaModelParamsV1 else LLAMA_LOAD_MODE_NONE)

    @pytest.mark.parametrize("cls", LAYOUTS)
    def test_true_writes_mmap(self, monkeypatch, cls):
        mp = _load_params(monkeypatch, cls, n_gpu_layers=24, use_mmap=True)
        assert _mode(mp) == (True if cls is LlamaModelParamsV1 else LLAMA_LOAD_MODE_MMAP)

    @pytest.mark.parametrize("cls", LAYOUTS)
    def test_none_at_full_offload_keeps_the_native_default(self, monkeypatch, cls):
        mp = _load_params(monkeypatch, cls, n_gpu_layers=99, use_mmap=None)
        assert _mode(mp) == _mode(_native_default(cls))


# --------------------------------------------------------------------------- #
#  _CapturedStderr.mapped: what the native load log says                      #
# --------------------------------------------------------------------------- #

def _mapped(tmp_path, text):
    p = tmp_path / "captured.log"
    p.write_text(text, encoding="utf-8")
    return llama_mod._CapturedStderr(str(p)).mapped()


class TestMappedFromTheLoadLog:
    @pytest.mark.parametrize("mode, want", [
        ("mmap", True), ("mmap+mlock", True),
        ("none", False), ("mlock", False), ("dio", False),
    ])
    def test_the_reported_load_mode(self, tmp_path, mode, want):
        line = (f"load_tensors: loading model tensors, this can take a while... "
                f"(load_mode = {mode})\n")
        assert _mapped(tmp_path, line) is want

    @pytest.mark.parametrize("value, want", [("true", True), ("false", False)])
    def test_the_older_mmap_flag_report(self, tmp_path, value, want):
        line = f"load_tensors: loading model tensors, this can take a while... (mmap = {value})\n"
        assert _mapped(tmp_path, line) is want

    def test_auto_falls_back_to_a_mapped_cpu_buffer(self, tmp_path):
        text = ("load_tensors: loading model tensors, this can take a while... "
                "(load_mode = auto)\n"
                "load_tensors:   CPU_Mapped model buffer size =   200.00 MiB\n")
        assert _mapped(tmp_path, text) is True

    def test_an_unsupported_platform_is_unmapped_whatever_was_asked(self, tmp_path):
        text = ("llama_model_loader: mmap is not supported on this platform\n"
                "load_tensors: loading model tensors, this can take a while... "
                "(load_mode = mmap)\n")
        assert _mapped(tmp_path, text) is False

    def test_nothing_reported_is_unknown(self, tmp_path):
        text = "load_tensors:        ROCm0 model buffer size =   3.35 MiB\n"
        assert _mapped(tmp_path, text) is None

    def test_a_missing_capture_is_unknown(self, tmp_path):
        assert llama_mod._CapturedStderr(str(tmp_path / "gone.log")).mapped() is None

    def test_adversarial_input_stays_linear_time(self, tmp_path):
        import time
        text = "(load_mode = " + "mmap" * 200_000 + "(mmap = " * 50_000
        start = time.perf_counter()
        assert _mapped(tmp_path, text) is None
        assert time.perf_counter() - start < 2.0
