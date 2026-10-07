# SPDX-License-Identifier: AGPL-3.0-or-later
"""Automatic MoE expert placement in GPU-layer auto sizing.

When n_gpu_layers is auto and a Mixture-of-Experts model does not fit whole,
auto sizing keeps routed experts in system RAM (the smallest n_cpu_moe that
fits every layer on the GPU) before it moves whole layers to the CPU. A
configured n_cpu_moe or n_gpu_layers is never overridden, and a dense model is
sized exactly as before.

The models are REAL GGUF files built byte by byte (header, KV block,
tensor-info entries, tensor data); the sizing runs for real against them and
only the VRAM readings are patched.
"""

from unittest.mock import patch

import pytest

from localm.inference.backends.gguf import GgufBackend
from localm.model_manager.gguf import (gguf_expert_counts,
                                       gguf_moe_expert_bytes_by_layer,
                                       gguf_moe_pinned_expert_bytes)
from tests.test_gguf_moe_vram_sizing import _T_STRING, _T_UINT32, _gguf_with_tensors

OVERHEAD = 10_000
N_CTX = 64
# block_count 4, head_count_kv 4, head_dim 64/4 = 16 -> 4 * 4 * 16 * 2 * 2 bytes.
KV_PER_TOKEN = 1024
KV = N_CTX * KV_PER_TOKEN
EXPERT_BYTES = (100_000, 110_000, 120_000, 130_000)


def _kv(arch, *, experts=True):
    kv = [("general.architecture", _T_STRING, arch),
          (f"{arch}.block_count", _T_UINT32, 4),
          (f"{arch}.embedding_length", _T_UINT32, 64),
          (f"{arch}.attention.head_count", _T_UINT32, 4),
          (f"{arch}.attention.head_count_kv", _T_UINT32, 4)]
    if experts:
        kv += [(f"{arch}.expert_count", _T_UINT32, 8),
               (f"{arch}.expert_used_count", _T_UINT32, 2)]
    return kv


def _moe_model(tmp_path, name="moe.gguf"):
    tensors = [("token_embd.weight", [4], 0, 5_000)]
    for il, size in enumerate(EXPERT_BYTES):
        tensors += [(f"blk.{il}.attn_q.weight", [4], 0, 2_000),
                    (f"blk.{il}.ffn_gate_inp.weight", [4], 0, 300),
                    (f"blk.{il}.ffn_gate_exps.weight", [4], 0, size // 2),
                    (f"blk.{il}.ffn_down_exps.weight", [4], 0, size - size // 2)]
    return _gguf_with_tensors(tmp_path / name, _kv("testmoe"), tensors)


def _dense_model(tmp_path):
    tensors = [("token_embd.weight", [4], 0, 5_000)]
    for il in range(4):
        tensors += [(f"blk.{il}.attn_q.weight", [4], 0, 2_000),
                    (f"blk.{il}.ffn_up.weight", [4], 0, 110_000)]
    return _gguf_with_tensors(tmp_path / "dense.gguf", _kv("dense", experts=False), tensors)


def _backend(path, **kw):
    kw.setdefault("n_gpu_layers", 99)
    kw.setdefault("n_gpu_layers_auto", True)
    return GgufBackend(str(path), n_ctx=N_CTX, **kw)


class _Vram:
    """Patch the free/total VRAM readings for one sizing call."""

    def __init__(self, free, total=10**12):
        self._patches = [
            patch.object(GgufBackend, "_split_free_total_bytes",
                         return_value=(None, None, 0)),
            patch.object(GgufBackend, "_free_vram_bytes", return_value=free),
            patch.object(GgufBackend, "_total_vram_bytes", return_value=total),
            patch.object(GgufBackend, "_VRAM_OVERHEAD_BYTES", OVERHEAD),
        ]

    def __enter__(self):
        for p in self._patches:
            p.start()
        return self

    def __exit__(self, *exc):
        for p in reversed(self._patches):
            p.stop()
        return False


def _flat(capsys) -> str:
    return " ".join(capsys.readouterr().out.split())


def _free_fitting(b, n_cpu_moe):
    """Free VRAM that fits every layer on the GPU with exactly the experts of
    the first *n_cpu_moe* layers in system RAM, and not one byte more."""
    return b._vram_model_bytes(0) - sum(EXPERT_BYTES[:n_cpu_moe]) + KV + OVERHEAD


class TestFitMath:
    def test_the_backend_reads_the_per_layer_expert_bytes_of_the_file(self, tmp_path):
        b = _backend(_moe_model(tmp_path))
        assert b._moe_expert_bytes_by_layer() == dict(enumerate(EXPERT_BYTES))
        assert b._kv_bytes_per_token() == KV_PER_TOKEN
        assert b._vram_model_bytes(2) == b._vram_model_bytes(0) - 210_000

    @pytest.mark.parametrize("n", [1, 2, 3, 4])
    def test_exact_fit_picks_that_n_and_one_byte_less_picks_the_next(self, tmp_path, n):
        b = _backend(_moe_model(tmp_path))
        free = _free_fitting(b, n)
        with _Vram(free):
            budget = b._auto_gpu_layers_budget()
        assert (budget.layers, budget.moe_cpu_layers) == (99, n)
        if n < 4:
            with _Vram(free - 1):
                budget = b._auto_gpu_layers_budget()
            assert (budget.layers, budget.moe_cpu_layers) == (99, n + 1)

    def test_budget_model_bytes_are_the_unpinned_weights(self, tmp_path):
        b = _backend(_moe_model(tmp_path))
        with _Vram(_free_fitting(b, 2)):
            budget = b._auto_gpu_layers_budget()
        assert budget.model == b._vram_model_bytes(0)
        assert (budget.kv, budget.overhead) == (KV, OVERHEAD)


class TestAutoChoice:
    def test_picks_the_smallest_n_cpu_moe_with_every_layer_on_the_gpu(self, tmp_path, capsys):
        b = _backend(_moe_model(tmp_path))
        with _Vram(_free_fitting(b, 2)):
            layers = b._effective_gpu_layers()
        assert b.effective_n_cpu_moe == 2
        assert layers == 99
        assert b.last_gpu_sizing["n_cpu_moe"] == 2
        assert b.last_gpu_sizing["n_cpu_moe_auto"] is True
        assert b.last_gpu_sizing["mode"] == "auto"
        assert b.last_gpu_sizing["layers"] == 99
        out = _flat(capsys)
        assert "every layer on the GPU, with the routed experts of 2/4 layers in system RAM" in out
        assert "Set n_cpu_moe or n_gpu_layers to override" in out

    def test_the_choice_reaches_every_later_charge_of_the_load(self, tmp_path, capsys):
        b = _backend(_moe_model(tmp_path))
        free = _free_fitting(b, 2)
        with _Vram(free):
            b.effective_gpu_layers = b._effective_gpu_layers()
            assert b._effective_model_bytes_for_vram() == b._vram_model_bytes(0) - 210_000
            b._check_vram()
        assert "Low VRAM" not in _flat(capsys)

    def test_a_model_that_fits_whole_pins_nothing(self, tmp_path, capsys):
        b = _backend(_moe_model(tmp_path))
        with _Vram(10**9):
            layers = b._effective_gpu_layers()
        assert (layers, b.effective_n_cpu_moe) == (99, 0)
        assert b.last_gpu_sizing["n_cpu_moe_auto"] is False
        assert "gpu layers auto" not in _flat(capsys)

    def test_the_full_offload_need_ignores_an_earlier_automatic_choice(self, tmp_path):
        b = _backend(_moe_model(tmp_path))
        with _Vram(_free_fitting(b, 2)):
            b._effective_gpu_layers()
            assert b.effective_n_cpu_moe == 2
            need = b.full_offload_vram_bytes()
        assert need == b._vram_model_bytes(0) + KV + OVERHEAD

    def test_a_reload_sizes_again_from_the_configured_value(self, tmp_path):
        b = _backend(_moe_model(tmp_path))
        with _Vram(_free_fitting(b, 3)):
            b._effective_gpu_layers()
        assert b.effective_n_cpu_moe == 3
        with _Vram(10**9):
            b._effective_gpu_layers()
        assert b.effective_n_cpu_moe == 0

    def test_every_expert_in_ram_then_whole_layers_when_that_still_does_not_fit(
            self, tmp_path, capsys):
        b = _backend(_moe_model(tmp_path))
        rest = b._vram_model_bytes(4)
        with _Vram(KV + OVERHEAD + rest // 2):
            layers = b._effective_gpu_layers()
        assert b.effective_n_cpu_moe == 4
        assert 0 < layers < 99
        assert layers == int((rest // 2) / rest * 32)
        out = _flat(capsys)
        assert "with every layer's routed experts in system RAM" in out
        assert "the rest on CPU" in out

    def test_no_room_for_any_weights_loads_on_the_cpu_and_pins_nothing(self, tmp_path):
        b = _backend(_moe_model(tmp_path))
        with _Vram(KV + OVERHEAD - 1):
            layers = b._effective_gpu_layers()
        assert (layers, b.effective_n_cpu_moe) == (0, 0)

    def test_an_unreadable_expert_layout_falls_back_to_layer_offload(self, tmp_path, capsys):
        b = _backend(_moe_model(tmp_path))
        free = _free_fitting(b, 2)
        b2 = _backend(_moe_model(tmp_path, name="moe2.gguf"))
        with _Vram(free), patch("localm.model_manager.gguf.gguf_moe_expert_bytes_by_layer",
                                side_effect=ValueError("simulated probe failure")):
            layers = b2._effective_gpu_layers()
        assert b2.effective_n_cpu_moe == 0
        assert layers < 99
        assert "experts" not in _flat(capsys)


class TestExplicitChoiceWins:
    def test_a_configured_n_cpu_moe_is_never_raised(self, tmp_path, capsys):
        b = _backend(_moe_model(tmp_path), n_cpu_moe=1)
        with _Vram(_free_fitting(b, 3)):
            layers = b._effective_gpu_layers()
        assert b.effective_n_cpu_moe == 1
        assert layers < 99
        assert b.last_gpu_sizing["n_cpu_moe"] == 1
        assert b.last_gpu_sizing["n_cpu_moe_auto"] is False
        assert "routed experts of" not in _flat(capsys)

    def test_a_configured_n_cpu_moe_that_fits_loads_every_layer(self, tmp_path):
        b = _backend(_moe_model(tmp_path), n_cpu_moe=3)
        with _Vram(_free_fitting(b, 2)):
            layers = b._effective_gpu_layers()
        assert (layers, b.effective_n_cpu_moe) == (99, 3)

    def test_an_explicit_n_gpu_layers_turns_auto_placement_off(self, tmp_path):
        b = _backend(_moe_model(tmp_path), n_gpu_layers=3)
        with _Vram(_free_fitting(b, 2)):
            layers = b._effective_gpu_layers()
        assert (layers, b.effective_n_cpu_moe) == (3, 0)
        assert b.last_gpu_sizing["mode"] == "configured"

    def test_auto_sizing_off_places_nothing_automatically(self, tmp_path):
        b = _backend(_moe_model(tmp_path), n_gpu_layers_auto=False)
        with _Vram(_free_fitting(b, 2)):
            layers = b._effective_gpu_layers()
        assert (layers, b.effective_n_cpu_moe) == (99, 0)


class TestDenseModelUnchanged:
    def test_a_dense_model_that_does_not_fit_offloads_layers_as_before(self, tmp_path, capsys):
        b = _backend(_dense_model(tmp_path))
        weights = b._vram_model_bytes(0)
        with _Vram(KV + OVERHEAD + weights // 2):
            layers = b._effective_gpu_layers()
        assert b.effective_n_cpu_moe == 0
        assert layers == int((weights // 2) / weights * 32)
        assert b.last_gpu_sizing["n_cpu_moe_auto"] is False
        out = _flat(capsys)
        assert "the rest on CPU (slower)" in out
        assert "experts" not in out


class TestLoadCarriesTheChoice:
    def _spawn(self, captured, meta):
        def _fake(self_runner, params, cancel_event=None, timeout=None, on_progress=None):
            captured.update(params)
            return dict(meta)
        return _fake

    def test_the_worker_gets_the_auto_choice_and_placement_reports_it(self, tmp_path, capsys):
        from localm.inference.engine import Engine
        b = _backend(_moe_model(tmp_path))
        captured = {}
        meta = {"n_layers": 4, "weight_placement": [
            {"backend": "ROCm0", "mib": 1.0, "is_ram": False},
            {"backend": "CPU", "mib": 0.5, "is_ram": True}]}
        with _Vram(_free_fitting(b, 2)), \
                patch("localm.discover.list_gpus", return_value=([], "ok")), \
                patch("localm.discover.resolve_auto_split_ratios", return_value=None), \
                patch.object(GgufBackend, "_implicit_split_fit", return_value=None), \
                patch.object(GgufBackend, "_effective_ctx_max", return_value=N_CTX), \
                patch("localm.inference.backends.llamacpp._runner.ModelRunner."
                      "spawn_and_load", self._spawn(captured, meta)), \
                patch("localm.model_meta.store_n_layers"):
            b.load()
        assert captured["n_cpu_moe"] == 2
        assert captured["n_gpu_layers"] == 99
        assert b.moe_cpu_layers == 2
        out = _flat(capsys)
        assert "moe placement: 0.50 MiB system RAM / 1.00 MiB VRAM" in out
        # 210,000 expert bytes in RAM, 2 of 8 experts read per token.
        assert "each generated token reads about 0.00 GB of expert weights" in out
        engine = object.__new__(Engine)
        engine._backend = b
        assert engine.gpu_placement == {"gpu_layers_offloaded": 4, "gpu_layers_total": 4,
                                        "degraded": True, "moe_cpu_layers": 2}

    def test_a_skipped_override_reports_no_experts_in_ram(self, tmp_path):
        b = _backend(_moe_model(tmp_path))
        meta = {"n_layers": 4, "weight_placement": [], "moe_skip_reason": "buffer_unresolved"}
        with _Vram(_free_fitting(b, 2)), \
                patch("localm.discover.list_gpus", return_value=([], "ok")), \
                patch("localm.discover.resolve_auto_split_ratios", return_value=None), \
                patch.object(GgufBackend, "_implicit_split_fit", return_value=None), \
                patch.object(GgufBackend, "_effective_ctx_max", return_value=N_CTX), \
                patch("localm.inference.backends.llamacpp._runner.ModelRunner."
                      "spawn_and_load", self._spawn({}, meta)), \
                patch("localm.model_meta.store_n_layers"):
            b.load()
        assert b.effective_n_cpu_moe == 2
        assert b.moe_cpu_layers == 0


class TestRamBytesPerToken:
    def test_scales_the_pinned_bytes_by_the_routed_share(self, tmp_path):
        b = _backend(_moe_model(tmp_path))
        assert b._moe_ram_bytes_per_token(2) == 210_000 * 2 // 8
        assert b._moe_ram_bytes_per_token(0) == 0

    def test_zero_for_a_dense_model(self, tmp_path):
        assert _backend(_dense_model(tmp_path))._moe_ram_bytes_per_token(4) == 0


class TestPreflightOptions:
    def test_the_refusal_names_a_context_that_fits_below_the_current_one(self, tmp_path):
        b = _backend(_dense_model(tmp_path), n_gpu_layers_auto=False)
        b.n_ctx = 32768
        weights = b._vram_model_bytes(0)
        total = weights + OVERHEAD + 8192 * KV_PER_TOKEN + 100
        with _Vram(total, total=total), pytest.raises(RuntimeError) as exc:
            b._check_vram()
        msg = str(exc.value)
        assert "Lower the context:  -c 8192" in msg
        assert "-c 32768" not in msg

    def test_no_context_option_when_the_weights_alone_exceed_the_card(self, tmp_path):
        b = _backend(_dense_model(tmp_path), n_gpu_layers_auto=False)
        total = b._vram_model_bytes(0) // 2
        with _Vram(total, total=total), pytest.raises(RuntimeError) as exc:
            b._check_vram()
        msg = str(exc.value)
        assert "Lower the context" not in msg
        assert "Offload fewer layers" in msg

    def test_the_refusal_names_the_n_cpu_moe_that_fits(self, tmp_path):
        b = _backend(_moe_model(tmp_path), n_gpu_layers_auto=False)
        b.n_ctx = 32768
        total = _free_fitting(b, 2) - KV + 32768 * KV_PER_TOKEN
        with _Vram(total, total=total), pytest.raises(RuntimeError) as exc:
            b._check_vram()
        assert "Keep MoE experts in system RAM:  localm config n_cpu_moe 2" in str(exc.value)

    def test_the_low_vram_warning_names_a_context_that_fits(self, tmp_path, capsys):
        b = _backend(_dense_model(tmp_path), n_gpu_layers_auto=False)
        b.n_ctx = 32768
        weights = b._vram_model_bytes(0)
        free = weights + OVERHEAD + 4096 * KV_PER_TOKEN + 100
        with _Vram(free, total=10**12), \
                patch.object(GgufBackend, "_vram_holder_hint", return_value="test"):
            b._check_vram()
        out = _flat(capsys)
        assert "Low VRAM" in out
        assert "Lower the context: -c 4096" in out
        assert "-c 32768" not in out


class TestGgufExpertProbes:
    def test_by_layer_sums_every_expert_projection(self, tmp_path):
        assert gguf_moe_expert_bytes_by_layer(_moe_model(tmp_path)) == dict(
            enumerate(EXPERT_BYTES))

    def test_by_layer_is_empty_for_a_dense_model(self, tmp_path):
        assert gguf_moe_expert_bytes_by_layer(_dense_model(tmp_path)) == {}

    def test_by_layer_is_none_for_a_file_that_does_not_parse(self, tmp_path):
        p = tmp_path / "x.gguf"
        p.write_bytes(b"\0" * 4096)
        assert gguf_moe_expert_bytes_by_layer(p) is None

    def test_by_layer_sums_split_parts(self, tmp_path):
        a = _gguf_with_tensors(tmp_path / "m-00001-of-00002.gguf", _kv("testmoe"),
                               [("blk.0.ffn_up_exps.weight", [4], 0, 700),
                                ("blk.0.attn_q.weight", [4], 0, 50)])
        _gguf_with_tensors(tmp_path / "m-00002-of-00002.gguf", _kv("testmoe"),
                           [("blk.1.ffn_down_exps.weight", [4], 0, 900),
                            ("blk.0.ffn_gate_exps.weight", [4], 0, 300)])
        assert gguf_moe_expert_bytes_by_layer(a) == {0: 1000, 1: 900}

    def test_by_layer_is_none_when_a_split_part_is_missing(self, tmp_path):
        a = _gguf_with_tensors(tmp_path / "m-00001-of-00002.gguf", _kv("testmoe"),
                               [("blk.0.ffn_up_exps.weight", [4], 0, 700)])
        assert gguf_moe_expert_bytes_by_layer(a) is None

    @pytest.mark.parametrize("name", ["ffn_gate_up_exps", "ffn_up_chexps",
                                      "ffn_down_chexps", "ffn_gate_chexps"])
    def test_the_fused_and_chunked_expert_tensors_count(self, tmp_path, name):
        f = _gguf_with_tensors(tmp_path / "m.gguf", _kv("testmoe"),
                               [(f"blk.0.{name}.weight", [4], 0, 400),
                                ("blk.0.ffn_gate_inp.weight", [4], 0, 30),
                                ("blk.0.ffn_up_shexp.weight", [4], 0, 60)])
        assert gguf_moe_expert_bytes_by_layer(f) == {0: 400}
        assert gguf_moe_pinned_expert_bytes(f, 1) == 400

    def test_expert_counts(self, tmp_path):
        assert gguf_expert_counts(_moe_model(tmp_path)) == (8, 2)
        assert gguf_expert_counts(_dense_model(tmp_path)) == (0, 0)
        only_count = _gguf_with_tensors(
            tmp_path / "c.gguf", _kv("testmoe", experts=False)
            + [("testmoe.expert_count", _T_UINT32, 8)], [("blk.0.a.weight", [4], 0, 10)])
        assert gguf_expert_counts(only_count) == (8, 0)
