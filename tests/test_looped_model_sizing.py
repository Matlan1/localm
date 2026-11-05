# SPDX-License-Identifier: AGPL-3.0-or-later
"""Pre-load sizing of a looped model (Nanbeige4.2, ``<arch>.num_loops``).

llama.cpp expands such a stack to ``block_count * num_loops`` logical layers
(``n_layer_all``), keeps one KV entry per logical layer, and shares the weights
of the first loop: ``llama_model_n_layer`` reports the product after load. The
GGUF header still states the physical ``block_count``, so every figure derived
from it before the load has to apply the loop count itself.
"""

from unittest import mock

import pytest

from localm.inference.backends.gguf import GgufBackend
from localm.inference.backends.llamacpp import _loader
from localm.model_manager.gguf import (
    _gguf_split_layout_meta, gguf_kv_bytes_per_token, gguf_split_layout)
from tests._bare_llama import make_bare_llama
from tests.test_implicit_split_fit import _T_UINT32, _write_gguf
from tests.test_kv_bytes_from_gguf import _T_ARRAY, _T_INT32, _gguf, _shape

PHYS = 4
N_EMBD, N_HEAD, N_HEAD_KV = 256, 8, 2
PER_LAYER_F16 = N_HEAD_KV * (N_EMBD // N_HEAD) * 2 * 2


def _nanbeige(path, loops=None, **kw):
    extra = [] if loops is None else [("nanbeige.num_loops", _T_UINT32, loops)]
    return _gguf(path, _shape("nanbeige", kw.pop("layers", PHYS), N_EMBD, N_HEAD,
                              N_HEAD_KV, extra=extra), **kw)


class TestKvBytesPerToken:
    def test_a_header_without_num_loops_is_the_plain_stack(self, tmp_path):
        assert gguf_kv_bytes_per_token(_nanbeige(tmp_path / "m.gguf")) == PHYS * PER_LAYER_F16

    @pytest.mark.parametrize("loops", [2, 3, 4])
    def test_num_loops_multiplies_the_kv_exactly(self, tmp_path, loops):
        base = gguf_kv_bytes_per_token(_nanbeige(tmp_path / "base.gguf"))
        looped = gguf_kv_bytes_per_token(_nanbeige(tmp_path / "looped.gguf", loops))
        assert base == PHYS * PER_LAYER_F16
        assert looped == loops * base

    @pytest.mark.parametrize("loops", [None, 1])
    def test_one_loop_changes_nothing(self, tmp_path, loops):
        f = _nanbeige(tmp_path / "m.gguf", loops)
        assert gguf_kv_bytes_per_token(f) == PHYS * PER_LAYER_F16

    def test_the_cache_type_scales_the_looped_figure(self, tmp_path):
        base = gguf_kv_bytes_per_token(_nanbeige(tmp_path / "base.gguf"), "q8_0")
        looped = gguf_kv_bytes_per_token(_nanbeige(tmp_path / "m.gguf", 2), "q8_0")
        assert looped == 2 * base == 2 * PHYS * PER_LAYER_F16 * 17 // 32

    def test_a_per_layer_head_array_is_multiplied_too(self, tmp_path):
        kv = [("general.architecture", 8, "nanbeige"),
              ("nanbeige.block_count", _T_UINT32, 4),
              ("nanbeige.embedding_length", _T_UINT32, N_EMBD),
              ("nanbeige.attention.head_count", _T_UINT32, N_HEAD),
              ("nanbeige.attention.head_count_kv", _T_ARRAY, (_T_INT32, [2, 0, 2, 0])),
              ("nanbeige.num_loops", _T_UINT32, 2)]
        f = _gguf(tmp_path / "m.gguf", kv)
        assert gguf_kv_bytes_per_token(f) == 2 * (2 + 2) * (N_EMBD // N_HEAD) * 2 * 2

    def test_only_the_architecture_own_num_loops_counts(self, tmp_path):
        extra = [("clip.num_loops", _T_UINT32, 9), ("llama.num_loops", _T_UINT32, 9)]
        f = _gguf(tmp_path / "m.gguf", _shape("nanbeige", PHYS, N_EMBD, N_HEAD,
                                              N_HEAD_KV, extra=extra))
        assert gguf_kv_bytes_per_token(f) == PHYS * PER_LAYER_F16

    def test_a_zero_num_loops_is_read_as_one(self, tmp_path):
        f = _nanbeige(tmp_path / "m.gguf", 0)
        assert gguf_kv_bytes_per_token(f) == PHYS * PER_LAYER_F16


def _model(tmp_path, loops=None, name="m.gguf"):
    tensors = [("token_embd.weight", 4000)]
    for il in range(PHYS):
        tensors += [(f"blk.{il}.attn_k.weight", 3000), (f"blk.{il}.ffn_down.weight", 1000)]
    tensors += [("output_norm.weight", 64), ("output.weight", 5000)]
    extra = [("embedding_length", N_EMBD), ("attention.head_count", N_HEAD),
             ("attention.head_count_kv", N_HEAD_KV)]
    if loops is not None:
        extra.append(("num_loops", loops))
    return _write_gguf(tmp_path / name, arch="nanbeige", block_count=PHYS, vocab=100,
                       tensors=tensors, extra_kv=extra)


class TestLayerCount:
    def test_the_layout_counts_the_logical_layers(self, tmp_path):
        layout = gguf_split_layout(_model(tmp_path, 3))
        assert layout["block_count"] == 3 * PHYS

    def test_without_num_loops_the_count_is_unchanged(self, tmp_path):
        assert _gguf_split_layout_meta(_model(tmp_path)) == (PHYS, 100)
        assert _gguf_split_layout_meta(_model(tmp_path, 1, "one.gguf")) == (PHYS, 100)

    def test_more_layers_than_llama_cpp_allows_is_unreadable(self, tmp_path):
        assert _gguf_split_layout_meta(_model(tmp_path, 512 // PHYS)) == (512, 100)
        assert _gguf_split_layout_meta(_model(tmp_path, 512 // PHYS + 1, "big.gguf")) is None

    def test_the_cached_layer_count_is_the_logical_one(self, tmp_path):
        b = GgufBackend(str(_model(tmp_path, 2)), n_ctx=512, n_gpu_layers=99)
        with mock.patch("localm.model_meta.cached_n_layers", return_value=None):
            assert b._cached_layer_count() == 2 * PHYS


class TestAgreesWithTheLoadedModel:
    def test_pre_load_and_post_load_kv_figures_are_equal(self, tmp_path):
        loops = 2
        f = _nanbeige(tmp_path / "m.gguf", loops)
        api = mock.MagicMock()
        api.has_kv_head_api.return_value = True
        api.has_hybrid_api.return_value = False
        api.llama_model_n_embd.return_value = N_EMBD
        api.llama_model_n_head.return_value = N_HEAD
        api.llama_model_n_head_kv.return_value = N_HEAD_KV
        llm = make_bare_llama(_model_ptr=111, n_layers=PHYS * loops,
                              is_encoder_decoder=False)
        with mock.patch("localm.inference.backends.llamacpp.llama.api", api):
            post = llm._read_kv_bytes_per_token()
        assert post == gguf_kv_bytes_per_token(f) == loops * PHYS * PER_LAYER_F16


class TestPlacement:
    N_CTX = 512

    def _units(self, tmp_path, loops):
        b = GgufBackend(str(_model(tmp_path, loops, f"m{loops}.gguf")),
                        n_ctx=self.N_CTX, n_gpu_layers=99)
        with mock.patch.object(_loader, "native_lib_loaded", return_value=False):
            return b, b._placement_units(kv_tokens=1)

    def test_every_logical_layer_carries_one_physical_layers_kv(self, tmp_path):
        _, plain = self._units(tmp_path, None)
        _, looped = self._units(tmp_path, 3)
        assert len(plain.layer_kv) == PHYS and len(looped.layer_kv) == 3 * PHYS
        assert set(plain.layer_kv) == set(looped.layer_kv) == {PER_LAYER_F16}
        assert sum(looped.layer_kv) == 3 * sum(plain.layer_kv)

    def test_the_weights_are_charged_once(self, tmp_path):
        _, plain = self._units(tmp_path, None)
        _, looped = self._units(tmp_path, 3)
        assert looped.layer_bytes[:PHYS] == plain.layer_bytes
        assert looped.layer_bytes[PHYS:] == [0] * (2 * PHYS)

    def test_the_implicit_split_fit_sees_the_same_layers(self, tmp_path):
        b, _ = self._units(tmp_path, 3)
        devices = [{"index": 0, "free": 10**9, "total": 10**9 + 10}]
        from localm import discover
        with mock.patch.object(discover, "implicit_split_devices", return_value=devices), \
                mock.patch.object(_loader, "native_lib_loaded", return_value=False):
            inputs = b._implicit_split_inputs(99)
        assert inputs["n_layer_all"] == 3 * PHYS
        kv = inputs["fit_kw"]["layer_kv_bytes"]
        assert len(kv) == 3 * PHYS
        assert set(kv) == {self.N_CTX * PER_LAYER_F16}
