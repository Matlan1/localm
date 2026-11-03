# SPDX-License-Identifier: AGPL-3.0-or-later
"""Context first: a load keeps its reserved context window usable before it
puts more routed experts on the GPU, and admission judges a Mixture-of-Experts
model by the placement its backend will actually use.

The models are real GGUF files built byte by byte (see
tests/test_moe_auto_placement.py); the sizing runs for real against them and
only the VRAM readings are patched.
"""

import logging
from unittest.mock import patch

import pytest

from localm.inference.backends.gguf import GgufBackend
from localm.inference.engine import (Engine, describe_gpu_placement,
                                     placement_moves_layers)
from tests.test_gguf_moe_vram_sizing import _T_UINT32, _gguf_with_tensors
from tests.test_moe_auto_placement import (EXPERT_BYTES, KV_PER_TOKEN, N_CTX,
                                           OVERHEAD, _dense_model, _kv,
                                           _moe_model, _Vram)

RESERVE = 256


def _backend(path, **kw):
    kw.setdefault("n_gpu_layers", 99)
    kw.setdefault("n_gpu_layers_auto", True)
    kw.setdefault("ctx_auto", True)
    kw.setdefault("n_ctx_max", 0)
    return GgufBackend(str(path), n_ctx=N_CTX, **kw)


def _free_for(b, n_cpu_moe, tokens):
    """Free VRAM that fits every layer on the GPU with the experts of the first
    *n_cpu_moe* layers in system RAM and a KV cache of *tokens*, and not one
    byte more."""
    return b._vram_model_bytes(n_cpu_moe) + tokens * KV_PER_TOKEN + OVERHEAD


def _trained_moe(tmp_path, trained):
    tensors = [("token_embd.weight", [4], 0, 5_000)]
    for il, size in enumerate(EXPERT_BYTES):
        tensors += [(f"blk.{il}.attn_q.weight", [4], 0, 2_000),
                    (f"blk.{il}.ffn_gate_exps.weight", [4], 0, size)]
    kv = _kv("testmoe") + [("testmoe.context_length", _T_UINT32, trained)]
    return _gguf_with_tensors(tmp_path / "trained.gguf", kv, tensors)


class TestReservedWindowComesFirst:
    def test_more_experts_move_to_ram_to_hold_the_reserved_window(self, tmp_path):
        plain = _backend(_moe_model(tmp_path))
        free = _free_for(plain, 1, N_CTX)
        with _Vram(free):
            assert plain._auto_gpu_layers_budget().moe_cpu_layers == 1
        b = _backend(_moe_model(tmp_path, "r.gguf"), ctx_reserve=RESERVE)
        with _Vram(free):
            budget = b._auto_gpu_layers_budget()
        assert budget.layers == 99
        assert budget.kv == RESERVE * KV_PER_TOKEN
        assert budget.moe_cpu_layers > 1
        assert (b._vram_model_bytes(budget.moe_cpu_layers) + budget.kv + budget.overhead
                <= free)

    def test_the_context_ceiling_holds_the_reserved_window(self, tmp_path):
        b = _backend(_moe_model(tmp_path), ctx_reserve=RESERVE)
        with _Vram(_free_for(b, 2, RESERVE)):
            b.effective_gpu_layers = b._effective_gpu_layers()
            ceiling = b._effective_ctx_max()
        assert b.effective_n_cpu_moe == 2
        assert ceiling >= RESERVE

    def test_a_window_that_never_fits_keeps_every_expert_in_ram_and_every_layer(
            self, tmp_path):
        b = _backend(_moe_model(tmp_path), ctx_reserve=10**6)
        all_layers = len(EXPERT_BYTES)
        with _Vram(_free_for(b, all_layers, N_CTX)):
            budget = b._auto_gpu_layers_budget()
        assert (budget.layers, budget.moe_cpu_layers) == (99, all_layers)
        assert budget.kv == N_CTX * KV_PER_TOKEN

    def test_a_dense_model_keeps_its_layers_and_only_raises_the_ceiling(self, tmp_path):
        b = _backend(_dense_model(tmp_path), ctx_reserve=RESERVE)
        free = b._vram_model_bytes(0) + N_CTX * KV_PER_TOKEN + OVERHEAD
        with _Vram(free):
            budget = b._auto_gpu_layers_budget()
            ceiling = b._effective_ctx_max()
        assert (budget.layers, budget.moe_cpu_layers) == (99, 0)
        assert budget.kv == N_CTX * KV_PER_TOKEN
        assert ceiling >= RESERVE

    def test_a_configured_n_cpu_moe_is_sized_for_n_ctx_only(self, tmp_path):
        b = _backend(_moe_model(tmp_path), ctx_reserve=RESERVE, n_cpu_moe=1)
        assert b._trades_experts_for_context() is False
        assert b._plan_kv_tokens() == N_CTX


class TestReserveLimit:
    def test_the_trained_window_caps_the_reserve(self, tmp_path):
        b = _backend(_trained_moe(tmp_path, 128), ctx_reserve=RESERVE)
        assert b.ctx_reserve_limit() == (128, "trained")
        assert b._ctx_reserve_tokens() == 128

    def test_a_fixed_ceiling_caps_it_when_auto_sizing_is_off(self, tmp_path):
        b = _backend(_moe_model(tmp_path), ctx_reserve=RESERVE, ctx_auto=False,
                     n_ctx_max=100)
        assert b.ctx_reserve_limit() == (100, "n_ctx_max")
        assert b._ctx_reserve_tokens() == 100

    def test_the_automatic_ceiling_caps_it_unless_n_ctx_max_is_0(self, tmp_path):
        capped = _backend(_moe_model(tmp_path), ctx_reserve=10**6, n_ctx_max=16384)
        assert capped.ctx_reserve_limit() == (capped._AUTO_CTX_MAX, "auto_cap")
        assert _backend(_moe_model(tmp_path, "u.gguf"),
                        ctx_reserve=10**6).ctx_reserve_limit() == (None, None)

    def test_never_below_n_ctx(self, tmp_path):
        b = _backend(_trained_moe(tmp_path, 16), ctx_reserve=RESERVE)
        assert b._ctx_reserve_tokens() == N_CTX


class TestOffloadNeeds:
    def test_a_moe_minimum_keeps_every_expert_in_ram_with_an_n_ctx_cache(self, tmp_path):
        b = _backend(_moe_model(tmp_path), ctx_reserve=RESERVE)
        with _Vram(10**9):
            full, low = b.gpu_offload_needs()
            assert b.full_offload_vram_bytes() == full
        assert full == b._vram_model_bytes(0) + RESERVE * KV_PER_TOKEN + OVERHEAD
        assert low == (b._vram_model_bytes(len(EXPERT_BYTES))
                       + N_CTX * KV_PER_TOKEN + OVERHEAD)

    def test_a_dense_model_has_one_figure(self, tmp_path):
        b = _backend(_dense_model(tmp_path), ctx_reserve=RESERVE)
        with _Vram(10**9):
            full, low = b.gpu_offload_needs()
        assert full == low == b._vram_model_bytes(0) + N_CTX * KV_PER_TOKEN + OVERHEAD

    def test_a_configured_n_cpu_moe_has_one_figure(self, tmp_path):
        b = _backend(_moe_model(tmp_path), n_cpu_moe=2)
        with _Vram(10**9):
            full, low = b.gpu_offload_needs()
        assert full == low

    def test_none_when_fewer_than_all_layers_are_asked_for(self, tmp_path):
        assert _backend(_moe_model(tmp_path), n_gpu_layers=3).gpu_offload_needs() is None


class TestPlacementIsInformation:
    def test_experts_in_ram_log_at_info_and_print_dim(self, tmp_path, capsys, caplog):
        b = _backend(_moe_model(tmp_path))
        with _Vram(_free_for(b, 2, N_CTX)), \
                caplog.at_level(logging.DEBUG, logger="localm"):
            layers = b._effective_gpu_layers()
        assert layers == 99
        records = [r for r in caplog.records if "gpu layers auto" in r.getMessage()]
        assert records and all(r.levelno == logging.INFO for r in records)
        out = " ".join(capsys.readouterr().out.split())
        assert "moe placement: every layer on the GPU; the routed experts of 2 of 4 " \
               "layers stay in system RAM" in out
        assert "[yellow]" not in out

    def test_layers_on_the_cpu_still_warn(self, tmp_path, caplog):
        b = _backend(_moe_model(tmp_path))
        with _Vram(OVERHEAD + N_CTX * KV_PER_TOKEN + b._vram_model_bytes(4) // 2), \
                caplog.at_level(logging.DEBUG, logger="localm"):
            layers = b._effective_gpu_layers()
        assert 0 < layers < 99
        records = [r for r in caplog.records if "gpu layers auto" in r.getMessage()]
        assert records and all(r.levelno == logging.WARNING for r in records)

    def test_a_moe_placement_is_described_without_slower(self):
        placement = {"gpu_layers_offloaded": 48, "gpu_layers_total": 48,
                     "degraded": True, "moe_cpu_layers": 25,
                     "moe_ram_read_bytes": 805109760}
        assert placement_moves_layers(placement) is False
        assert describe_gpu_placement(placement) == (
            "48/48 layers on the GPU, the routed experts of 25 of 48 layers in system "
            "RAM; each token reads only its active experts there (about 0.75 GB)")

    def test_layers_on_the_cpu_are_named(self):
        placement = {"gpu_layers_offloaded": 30, "gpu_layers_total": 48,
                     "degraded": True}
        assert placement_moves_layers(placement) is True
        assert describe_gpu_placement(placement) == "30/48 layers on the GPU, the rest on the CPU"


class TestEngineReserve:
    def test_reserve_context_raises_the_backends_window_now_and_at_the_next_load(
            self, tmp_path):
        engine = Engine(str(_moe_model(tmp_path)))
        assert engine.can_reserve_context() is True
        engine.reserve_context(RESERVE)
        assert engine.ctx_reserve == RESERVE
        assert engine._backend_args["ctx_reserve"] == RESERVE
        assert engine._backend_fresh is False

    @pytest.mark.parametrize("config_value,override,expected", [
        (16384, None, 16384), (16384, 32768, 32768), (16384, 4096, 16384),
        (0, None, 0), (True, None, 0), (-5, 8192, 8192),
    ])
    def test_the_config_value_is_raised_by_an_override(self, config_value, override,
                                                      expected):
        from localm.inference.engine import _resolve_ctx_reserve
        assert _resolve_ctx_reserve({"ctx_reserve": config_value}, override) == expected


def test_the_real_moe_file_plans_every_layer_on_the_gpu_with_its_reserved_window():
    """Bound to a real Qwen3-Coder-30B-A3B Q6_K file named by
    LOCALM_TEST_MOE_GGUF: on a 16 GB card's free reading, the 32k window an
    agent asks for keeps every layer on the GPU and leaves the expert plan's
    minimum far below the full offload need."""
    import os
    from pathlib import Path
    path = Path(os.environ.get("LOCALM_TEST_MOE_GGUF", ""))
    if not path.name or not path.is_file():
        pytest.skip("LOCALM_TEST_MOE_GGUF does not name a Qwen3-Coder-30B-A3B GGUF")
    b = GgufBackend(str(path), n_ctx=4096, n_gpu_layers=99, n_gpu_layers_auto=True,
                    ctx_auto=True, n_ctx_max=0, ctx_reserve=32768,
                    vram_overhead_bytes=1500 * 1024 ** 2)
    free = 15_730_737_152
    with patch.object(GgufBackend, "_split_free_total_bytes", return_value=(None, None, 0)), \
            patch.object(GgufBackend, "_free_vram_bytes", return_value=free), \
            patch.object(GgufBackend, "_total_vram_bytes", return_value=17_163_091_968), \
            patch.object(GgufBackend, "_placement_full_offload_need", return_value=None):
        budget = b._auto_gpu_layers_budget()
        full, low = b.gpu_offload_needs()
    assert budget.layers == 99
    assert budget.kv == 32768 * b._kv_bytes_per_token()
    assert 23 < budget.moe_cpu_layers < 48
    assert low < 5 * 1024 ** 3 < free < full


class TestTwoGpuPlanKeepsEveryLayer:
    """The placement planner for two or more GPUs falls back to an n_ctx KV
    cache when the reserved window does not fit beside every layer."""

    def _model(self, tmp_path):
        # gguf_split_layout reads the vocabulary size from the token list.
        from tests.test_implicit_split_fit import _write_gguf
        tensors = [("token_embd.weight", 5_000), ("output.weight", 5_000)]
        for il, size in enumerate(EXPERT_BYTES):
            tensors += [(f"blk.{il}.attn_q.weight", 2_000),
                        (f"blk.{il}.ffn_gate_exps.weight", size)]
        extra = [("expert_count", 8), ("expert_used_count", 2), ("embedding_length", 64),
                 ("attention.head_count", 4), ("attention.head_count_kv", 4)]
        return _write_gguf(tmp_path / "two.gguf", arch="testmoe", block_count=4,
                           vocab=8, tensors=tensors, extra_kv=extra)

    def _plan(self, b, free_each):
        devices = [{"slot": i, "index": i, "name": f"gpu{i}", "free": free_each,
                    "total": free_each * 2} for i in range(2)]
        with patch("localm.discover.gguf_split_devices", return_value=(devices, "")), \
                patch("localm.discover.listed_gpu_order", return_value=None), \
                patch("localm.inference.backends.llamacpp._loader.native_lib_loaded",
                      return_value=False), \
                patch.object(GgufBackend, "_VRAM_OVERHEAD_BYTES", OVERHEAD):
            return b._plan_gpu_placement()

    def test_a_window_that_never_fits_keeps_every_layer_on_the_gpus(self, tmp_path):
        b = _backend(self._model(tmp_path), ctx_reserve=10**6)
        placement = self._plan(b, 400_000)
        assert placement is not None
        assert placement.full_offload and placement.fits

    def test_a_window_that_fits_is_kept(self, tmp_path):
        b = _backend(self._model(tmp_path), ctx_reserve=RESERVE)
        placement = self._plan(b, 10**8)
        assert placement.full_offload and placement.fits
        kv = sum(d.kv for d in placement.loads)
        assert kv >= RESERVE * KV_PER_TOKEN


class TestUnlimitedCeiling:
    def test_no_ceiling_reports_the_trained_window_not_n_ctx(self, tmp_path):
        engine = Engine(str(_trained_moe(tmp_path, 40960)))
        backend = engine._backend
        backend.ctx_auto = False
        backend.n_ctx_max = 0
        backend.effective_ctx_max = 0
        assert engine.context_capacity() == 40960
        assert backend.ctx_reserve_limit() == (40960, "trained")

    def test_no_ceiling_and_no_trained_window_has_no_capacity_bound(self, tmp_path):
        engine = Engine(str(_moe_model(tmp_path)))
        engine._backend.ctx_auto = False
        engine._backend.n_ctx_max = 0
        engine._backend.effective_ctx_max = 0
        assert engine.context_capacity() is None
