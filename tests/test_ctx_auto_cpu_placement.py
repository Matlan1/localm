# SPDX-License-Identifier: AGPL-3.0-or-later
"""ctx_auto sizes the context ceiling from the memory the KV cache lands in:
free VRAM for the layers on the GPU, free system RAM for the layers on the CPU.

The model's shape (file bytes, per-block bytes, KV bytes per token, layer
count) is set through the backend's own per-instance caches; the free-VRAM and
free-RAM readers are the injected seams."""

import pytest

from localm.inference.backends.llamacpp import _sizing
from localm.inference.backends.llamacpp._sizing import HOST_RAM_HEADROOM_BYTES

MIB = 1024 ** 2
GIB = 1024 ** 3
LAYERS = 30
BLOCK = 100 * MIB
MODEL = (LAYERS + 1) * BLOCK          # 30 blocks plus 100 MiB outside them
INPUT = 50 * MIB                      # input-layer tensors, part of the outside bytes
PER_TOKEN = LAYERS * 4096             # KV bytes per token over every layer


def _ceiling(budget: int, per_token: int, capped: bool = False) -> int:
    tokens = (max(0, budget) // per_token // 1024) * 1024
    if capped:
        tokens = min(_sizing.VramSizingMixin._AUTO_CTX_MAX, tokens)
    return max(_sizing.VramSizingMixin._AUTO_CTX_MIN, tokens)


class _Readers:
    """Counts and answers the free-VRAM and free-RAM readings."""

    def __init__(self, vram, ram):
        self.vram, self.ram = vram, ram
        self.vram_calls = 0
        self.ram_calls = 0

    def split(self):
        self.vram_calls += 1
        return None, None, 1

    def free_vram(self):
        self.vram_calls += 1
        return self.vram

    def system_ram(self):
        self.ram_calls += 1
        return 64 * GIB, self.ram


@pytest.fixture
def make(monkeypatch, tmp_path):
    monkeypatch.setattr(_sizing, "embedder_ctx_reservation_bytes", lambda: 0)

    def _make(gpu_layers, *, vram, ram, n_ctx_max=None):
        from localm.inference.backends.gguf import GgufBackend
        readers = _Readers(vram, ram)
        monkeypatch.setattr("localm.sysstats.system_ram", readers.system_ram)
        b = GgufBackend.__new__(GgufBackend)
        b.model_path = str(tmp_path / "absent.gguf")
        b.n_ctx = 4096
        b.n_ctx_max = n_ctx_max
        b.ctx_auto = True
        b.n_gpu_layers = gpu_layers
        b.effective_gpu_layers = gpu_layers
        b._model_bytes = lambda: MODEL
        b._gguf_kv_bpt = PER_TOKEN
        b._header_layer_count = LAYERS
        b._block_bytes_cache = {i: (BLOCK, 0) for i in range(LAYERS)}
        b._gguf_input_layer_bytes = INPUT
        b._gguf_parsed_entries_cache = None
        b._split_free_total_bytes = readers.split
        b._free_vram_bytes = readers.free_vram
        return b, readers
    return _make


class TestCpuOnly:
    def test_g0_sizes_from_free_ram_and_never_reads_vram(self, make):
        b, readers = make(0, vram=80 * GIB, ram=6 * GIB)
        tokens, source = b._auto_ctx_sized(capped=False)
        assert readers.vram_calls == 0
        assert readers.ram_calls == 1
        assert source == "from free RAM"
        assert tokens == _ceiling(6 * GIB - MODEL - HOST_RAM_HEADROOM_BYTES, PER_TOKEN)
        assert tokens == 8192

    def test_g0_ceiling_never_exceeds_what_free_ram_holds(self, make):
        b, _ = make(0, vram=80 * GIB, ram=6 * GIB)
        tokens = b._auto_ctx_max()
        assert MODEL + tokens * PER_TOKEN <= 6 * GIB

    def test_g0_prints_from_free_ram(self, make, monkeypatch):
        printed = []
        monkeypatch.setattr(_sizing.console, "print", lambda s, *a, **k: printed.append(s))
        b, _ = make(0, vram=80 * GIB, ram=6 * GIB)
        assert b._effective_ctx_max() == 8192
        assert printed == ["[dim]  ctx auto : window may grow to 8,192 tokens "
                           "(from free RAM)[/dim]"]

    def test_g0_with_ram_unreadable_falls_back_and_says_so(self, make):
        b, readers = make(0, vram=80 * GIB, ram=None)
        assert b._auto_ctx_sized() == (16384, "free RAM not measurable, default")
        assert readers.vram_calls == 0

    @pytest.mark.parametrize("gpu_layers", [99, 16])
    def test_runtime_without_a_gpu_backend_sizes_everything_from_ram(
            self, make, gpu_layers):
        b, readers = make(gpu_layers, vram=80 * GIB, ram=6 * GIB)
        b.runtime_has_no_gpu = True
        assert b._auto_ctx_sized() == (8192, "from free RAM")
        assert readers.vram_calls == 0

    def test_full_offload_with_vram_unreadable_keeps_the_fallback(self, make):
        b, readers = make(99, vram=None, ram=6 * GIB)
        assert b._auto_ctx_sized() == (16384, "free VRAM not measurable, default")
        assert readers.ram_calls == 0


class TestHeadroomLeavesNoBudget:
    """Free RAM that holds the weights and a useful window but not the headroom
    beside them still gets that window, never more than the 16384 fallback."""

    def test_a_window_that_fits_is_not_cut_to_the_floor(self, make):
        ram = MODEL + 3 * GIB // 2
        assert ram - MODEL - HOST_RAM_HEADROOM_BYTES < 0
        b, _ = make(0, vram=80 * GIB, ram=ram)
        tokens, source = b._auto_ctx_sized()
        assert source == "from free RAM"
        assert tokens == _ceiling(ram - MODEL, PER_TOKEN) == 12288
        assert MODEL + tokens * PER_TOKEN <= ram

    def test_the_ceiling_never_exceeds_the_fallback(self, make):
        b, _ = make(0, vram=80 * GIB, ram=MODEL + GIB)
        b._gguf_kv_bpt = 4096
        assert b._auto_ctx_sized(capped=True)[0] == b._AUTO_CTX_FALLBACK == 16384
        assert b._auto_ctx_sized(capped=False)[0] == 16384

    @pytest.mark.parametrize("ram", [MODEL - 1, MODEL, 0])
    def test_weights_that_leave_no_room_keep_the_floor(self, make, ram):
        b, _ = make(0, vram=80 * GIB, ram=ram)
        assert b._auto_ctx_sized() == (4096, "from free RAM")

    def test_a_budget_after_the_headroom_is_still_held_to_it(self, make):
        ram = MODEL + HOST_RAM_HEADROOM_BYTES + 1500 * MIB
        b, _ = make(0, vram=80 * GIB, ram=ram)
        assert b._auto_ctx_sized(capped=False)[0] == _ceiling(1500 * MIB, PER_TOKEN) == 12288

    def test_a_partial_offload_with_a_short_headroom_keeps_its_window(self, make):
        cpu_held = 15 * BLOCK + INPUT
        ram = cpu_held + 3 * GIB // 2
        assert ram - cpu_held - HOST_RAM_HEADROOM_BYTES < 0
        b, _ = make(16, vram=16 * GIB, ram=ram)
        tokens, source = b._auto_ctx_sized(capped=False)
        assert source == "from free VRAM and RAM"
        assert tokens == min(16384, _ceiling(ram - cpu_held, PER_TOKEN - PER_TOKEN // 2)) == 16384


class TestPartialOffload:
    # -g 16 on 30 blocks: blocks 15..29 (15 of them) keep their KV in VRAM.
    GPU_LAYERS = 16
    CPU_HELD = 15 * BLOCK + INPUT
    HALF = PER_TOKEN // 2

    def test_ram_share_limits_the_ceiling(self, make):
        b, readers = make(self.GPU_LAYERS, vram=16 * GIB, ram=4 * GIB)
        tokens, source = b._auto_ctx_sized(capped=False)
        assert source == "from free VRAM and RAM"
        assert readers.vram_calls >= 1 and readers.ram_calls == 1
        assert tokens == _ceiling(4 * GIB - self.CPU_HELD - HOST_RAM_HEADROOM_BYTES,
                                  PER_TOKEN - self.HALF)

    def test_vram_share_is_charged_only_the_gpu_blocks(self, make):
        b, _ = make(self.GPU_LAYERS, vram=4 * GIB, ram=64 * GIB)
        tokens, source = b._auto_ctx_sized(capped=False)
        assert source == "from free VRAM and RAM"
        vram_weights = MODEL - self.CPU_HELD
        assert tokens == _ceiling(4 * GIB - vram_weights - b._split_overhead_bytes(1),
                                  self.HALF)

    def test_ram_unreadable_keeps_the_vram_share_and_says_so(self, make):
        b, _ = make(self.GPU_LAYERS, vram=4 * GIB, ram=None)
        tokens, source = b._auto_ctx_sized(capped=False)
        assert source == "from free VRAM; free RAM not measurable"
        assert tokens == _ceiling(4 * GIB - (MODEL - self.CPU_HELD)
                                  - b._split_overhead_bytes(1), self.HALF)

    def test_vram_unreadable_keeps_the_ram_share_and_says_so(self, make):
        b, _ = make(self.GPU_LAYERS, vram=None, ram=4 * GIB)
        tokens, source = b._auto_ctx_sized(capped=False)
        assert source == "from free RAM; free VRAM not measurable"
        assert tokens == _ceiling(4 * GIB - self.CPU_HELD - HOST_RAM_HEADROOM_BYTES,
                                  PER_TOKEN - self.HALF)


    def test_unknown_block_count_charges_all_kv_to_ram(self, make):
        b, _ = make(self.GPU_LAYERS, vram=16 * GIB, ram=8 * GIB)
        b._header_layer_count = None
        b._block_bytes_cache = {}
        tokens, source = b._auto_ctx_sized(capped=False)
        assert source == "from free VRAM and RAM"
        assert tokens == _ceiling(8 * GIB - MODEL - HOST_RAM_HEADROOM_BYTES, PER_TOKEN)
        assert tokens == 25600


class TestFullOffloadUnchanged:
    @pytest.mark.parametrize("gpu_layers", [99, -1])
    def test_full_offload_reads_vram_only(self, make, gpu_layers):
        b, readers = make(gpu_layers, vram=6 * GIB, ram=1 * GIB)
        tokens, source = b._auto_ctx_sized(capped=False)
        assert readers.ram_calls == 0
        assert source == "from free VRAM"
        weights = b._effective_model_bytes_for_vram()
        assert weights == MODEL - INPUT
        assert tokens == _ceiling(6 * GIB - weights - b._split_overhead_bytes(1),
                                  PER_TOKEN)

    def test_full_offload_prints_from_free_vram(self, make, monkeypatch):
        printed = []
        monkeypatch.setattr(_sizing.console, "print", lambda s, *a, **k: printed.append(s))
        b, _ = make(99, vram=80 * GIB, ram=1 * GIB)
        assert b._effective_ctx_max() == 65536
        assert printed == ["[dim]  ctx auto : window may grow to 65,536 tokens "
                           "(from free VRAM)[/dim]"]


class TestMultiGpuPlacement:
    """A placement over two GPUs that leaves blocks on the CPU is also held to
    the window whose KV cache for those blocks fits free system RAM."""

    def _placement(self, gpu_blocks, free_each):
        from localm.inference.backends.llamacpp._gpu_plan import (
            DeviceLoad, Placement, PlanDevice)
        start = LAYERS - gpu_blocks
        half = gpu_blocks // 2
        per_layer = PER_TOKEN // LAYERS
        loads = []
        for slot, layers in enumerate((list(range(start, start + half)),
                                       list(range(start + half, LAYERS)))):
            dev = PlanDevice(slot=slot, index=slot, name=f"gpu{slot}",
                             free=free_each, total=free_each)
            loads.append(DeviceLoad(device=dev, layers=layers, output=slot == 1,
                                    weights=len(layers) * BLOCK,
                                    kv_per_token=len(layers) * per_layer))
        return Placement(loads=loads, n_gpu_layers=gpu_blocks + 1, n_cpu_moe=0,
                         n_layer=LAYERS, fits=True)

    def _units(self):
        from localm.inference.backends.llamacpp._gpu_plan import ModelUnits
        return ModelUnits(layer_bytes=[BLOCK] * LAYERS, expert_bytes=[0] * LAYERS,
                          layer_kv=[0] * LAYERS,
                          kv_per_token=[PER_TOKEN // LAYERS] * LAYERS,
                          output_bytes=0, logits_bytes=0, reserve_bytes=0)

    def test_cpu_blocks_are_held_to_free_ram(self, make, monkeypatch):
        printed = []
        monkeypatch.setattr(_sizing.console, "print", lambda s, *a, **k: printed.append(s))
        b, readers = make(16, vram=None, ram=4 * GIB, n_ctx_max=0)
        b._placement_units = self._units
        placement = self._placement(gpu_blocks=15, free_each=40 * GIB)
        tokens = b._placement_ctx_max(placement)
        cpu_held = 15 * BLOCK + INPUT
        assert tokens == _ceiling(4 * GIB - cpu_held - HOST_RAM_HEADROOM_BYTES,
                                  PER_TOKEN // 2)
        assert readers.ram_calls == 1
        assert printed == [f"[dim]  ctx auto : window may grow to {tokens:,} tokens "
                           "(from the free VRAM of each GPU and free RAM; "
                           "no max (n_ctx_max=0))[/dim]"]

    def test_full_offload_placement_reads_no_ram(self, make):
        b, readers = make(99, vram=None, ram=1 * GIB, n_ctx_max=0)
        b._placement_units = self._units
        placement = self._placement(gpu_blocks=LAYERS, free_each=40 * GIB)
        assert b._placement_ctx_max(placement) > 65536
        assert readers.ram_calls == 0


class TestLiveContextKvPlacement:
    @pytest.mark.parametrize("gpu_layers, offload_kqv, in_vram", [
        (0, True, False), (12, True, True), (99, True, True), (99, False, False)])
    def test_kv_in_vram_follows_the_layers_on_the_gpu(self, gpu_layers, offload_kqv,
                                                      in_vram):
        from types import SimpleNamespace

        from localm.inference.backends.llamacpp._worker import GgufWorker
        w = GgufWorker.__new__(GgufWorker)
        w.effective_gpu_layers = gpu_layers
        w._llm = SimpleNamespace(_ctx_ptr=1, _ctx_capacity=4096, _offload_kqv=offload_kqv)
        assert w.live_context == {"n_ctx": 4096, "kv_in_vram": in_vram}
