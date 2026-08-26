# SPDX-License-Identifier: AGPL-3.0-or-later
"""MoE-aware VRAM preflight (n_cpu_moe): _check_vram, _auto_gpu_layers and
_auto_ctx_max all read self.n_cpu_moe rather than charging VRAM for the WHOLE
file, so routed-expert weights pinned to system RAM (llama.py's _apply_cpu_moe)
are not counted against the VRAM budget. Without that a model that fits once its
experts are pinned is refused outright (_check_vram), under-offloaded
(_auto_gpu_layers), or under-budgeted for context (_auto_ctx_max).

These tests build REAL GGUF files byte by byte (header + KV block + real
tensor-info entries + real tensor data, offsets and file size all internally
consistent) and drive the real parser and the real sizing path. They follow
test_kv_bytes_from_gguf.py's _gguf/_s/_shape helper conventions.
"""

import struct

from unittest.mock import patch

import pytest

from localm.inference.backends.gguf import GgufBackend
from localm.model_manager.gguf import gguf_moe_pinned_expert_bytes


GB = 1024 ** 3

_T_UINT32 = 4
_T_STRING = 8
_T_ARRAY = 9


def _s(text: str) -> bytes:
    raw = text.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _tensor_info(name: str, dims, ggml_type: int, offset: int) -> bytes:
    """One GGUF tensor-info entry: name, n_dims, dims[], ggml_type, offset."""
    out = [_s(name), struct.pack("<I", len(dims))]
    for d in dims:
        out.append(struct.pack("<Q", d))
    out.append(struct.pack("<I", ggml_type))
    out.append(struct.pack("<Q", offset))
    return b"".join(out)


def _gguf_with_tensors(path, kv, tensors, *, version=3, magic=b"GGUF",
                       alignment=32):
    """Write a REAL, internally-consistent GGUF file: header, KV block,
    tensor-info block, alignment padding, then real tensor DATA bytes -
    offsets and the final file size are exactly what a real writer would
    produce, which is exactly what gguf_moe_pinned_expert_bytes reads back.

    *kv* is an ordered list of (key, type, value) - value is an int for
    _T_UINT32, a str for _T_STRING, or a list[str] for _T_ARRAY (string
    arrays only - the one shape this file's tests need, to simulate a large
    tokenizer vocab sitting before the tensor-info section).

    *tensors* is an ordered list of (name, dims, ggml_type, size_bytes) -
    offsets are assigned here from cumulative size_bytes, contiguous, in
    list order (mirrors how a real GGUF writer lays out tensor data)."""
    header = [magic, struct.pack("<I", version),
              struct.pack("<QQ", len(tensors), len(kv))]
    for key, vtype, val in kv:
        header.append(_s(key))
        header.append(struct.pack("<I", vtype))
        if vtype == _T_STRING:
            header.append(_s(val))
        elif vtype == _T_UINT32:
            header.append(struct.pack("<I", val))
        elif vtype == _T_ARRAY:
            header.append(struct.pack("<I", _T_STRING))
            header.append(struct.pack("<Q", len(val)))
            for s in val:
                header.append(_s(s))
        else:
            raise AssertionError(f"test helper does not emit type {vtype}")

    offsets = []
    off = 0
    for _name, _dims, _ggml_type, size in tensors:
        offsets.append(off)
        off += size
    tensor_info = [_tensor_info(name, dims, ggml_type, offset)
                   for (name, dims, ggml_type, _size), offset
                   in zip(tensors, offsets)]

    body = b"".join(header) + b"".join(tensor_info)
    remainder = len(body) % alignment
    if remainder:
        body += b"\0" * (alignment - remainder)
    data = b"".join(b"\xAB" * size for _n, _d, _t, size in tensors)
    path.write_bytes(body + data)
    return path


# --------------------------------------------------------------------------- #
#  gguf_moe_pinned_expert_bytes: the header/tensor-info read                    #
# --------------------------------------------------------------------------- #

class TestGgufMoePinnedExpertBytes:
    def test_sums_only_matching_tensors_within_the_pinned_layer_range(
            self, tmp_path):
        tensors = [
            ("blk.0.attn_q.weight", [4, 4], 0, 100),           # not an expert tensor
            ("blk.0.ffn_gate_exps.weight", [4, 4, 8], 0, 500),
            ("blk.0.ffn_down_exps.weight", [4, 4, 8], 0, 600),
            ("blk.1.ffn_gate_exps.weight", [4, 4, 8], 0, 700),
            ("blk.1.ffn_up_exps.weight", [4, 4, 8], 0, 900),
            ("blk.5.ffn_gate_exps.weight", [4, 4, 8], 0, 1234),  # layer 5 out of range
        ]
        f = _gguf_with_tensors(
            tmp_path / "moe.gguf",
            [("general.architecture", _T_STRING, "testmoe")], tensors)
        result = gguf_moe_pinned_expert_bytes(f, n_pinned_layers=2)
        assert result == 500 + 600 + 700 + 900

    def test_excludes_router_and_shared_expert_tensors(self, tmp_path):
        # The router (ffn_gate_inp) and a shared expert are never pinned by
        # _apply_cpu_moe, so they are not counted here either.
        tensors = [
            ("blk.0.ffn_gate_inp.weight", [4, 4], 0, 50),
            ("blk.0.ffn_gate_exps.weight", [4, 4, 8], 0, 500),
        ]
        f = _gguf_with_tensors(
            tmp_path / "moe.gguf",
            [("general.architecture", _T_STRING, "testmoe")], tensors)
        assert gguf_moe_pinned_expert_bytes(f, n_pinned_layers=1) == 500

    def test_out_of_offset_order_tensor_info_still_sums_correctly(
            self, tmp_path):
        # The tensor-info LIST order need not match offset order; the function
        # sorts by offset itself.
        tensors = [
            ("blk.1.ffn_gate_exps.weight", [4], 0, 700),
            ("blk.0.ffn_gate_exps.weight", [4], 0, 500),
        ]
        f = _gguf_with_tensors(
            tmp_path / "moe.gguf",
            [("general.architecture", _T_STRING, "testmoe")], tensors)
        assert gguf_moe_pinned_expert_bytes(f, n_pinned_layers=2) == 1200

    def test_last_tensor_in_file_sized_from_file_size(self, tmp_path):
        # The final tensor has no next offset to diff against, so its size comes
        # from the file's own total size.
        tensors = [
            ("blk.0.attn_q.weight", [4], 0, 64),
            ("blk.0.ffn_gate_exps.weight", [4], 0, 500),   # last -> sized from EOF
        ]
        f = _gguf_with_tensors(
            tmp_path / "moe.gguf",
            [("general.architecture", _T_STRING, "testmoe")], tensors)
        assert gguf_moe_pinned_expert_bytes(f, n_pinned_layers=1) == 500

    def test_zero_for_dense_model_no_expert_tensors_present(self, tmp_path):
        # Parsing succeeded and nothing matched: 0 is a real answer, not a
        # failure.
        tensors = [("blk.0.attn_q.weight", [4, 4], 0, 100)]
        f = _gguf_with_tensors(
            tmp_path / "dense.gguf",
            [("general.architecture", _T_STRING, "dense")], tensors)
        assert gguf_moe_pinned_expert_bytes(f, n_pinned_layers=5) == 0

    def test_reaches_tensor_infos_past_a_large_kv_array(self, tmp_path):
        """This function streams the file rather than reading a bounded prefix
        (as gguf_kv_bytes_per_token / gguf_expert_count do), because a real
        tokenizer vocab array sits BEFORE the tensor-info section and can be
        several MB for a 100k+-token vocabulary. Builds one bigger than
        _GGUF_META_PROBE_BYTES (4MB) and reads the tensor-info section on the
        far side of it."""
        big_vocab = [f"tok{i:08d}" for i in range(250_000)]   # ~4.5 MB of strings
        kv = [
            ("general.architecture", _T_STRING, "testmoe"),
            ("tokenizer.ggml.tokens", _T_ARRAY, big_vocab),
        ]
        tensors = [("blk.0.ffn_gate_exps.weight", [4, 4, 8], 0, 500)]
        f = _gguf_with_tensors(tmp_path / "moe.gguf", kv, tensors)
        assert gguf_moe_pinned_expert_bytes(f, n_pinned_layers=1) == 500

    # --- refusals: every one of these must answer None, never a wrong number ---

    def test_none_when_n_pinned_layers_not_positive(self, tmp_path):
        tensors = [("blk.0.ffn_gate_exps.weight", [4], 0, 10)]
        f = _gguf_with_tensors(tmp_path / "m.gguf", [], tensors)
        assert gguf_moe_pinned_expert_bytes(f, 0) is None
        assert gguf_moe_pinned_expert_bytes(f, -1) is None

    def test_none_when_not_a_gguf(self, tmp_path):
        f = tmp_path / "m.gguf"
        f.write_bytes(b"\0" * 4096)
        assert gguf_moe_pinned_expert_bytes(f, 5) is None

    def test_none_when_file_missing(self, tmp_path):
        assert gguf_moe_pinned_expert_bytes(tmp_path / "nope.gguf", 5) is None

    def test_none_on_gguf_v1(self, tmp_path):
        tensors = [("blk.0.ffn_gate_exps.weight", [4], 0, 10)]
        f = _gguf_with_tensors(tmp_path / "m.gguf", [], tensors, version=1)
        assert gguf_moe_pinned_expert_bytes(f, 5) is None

    def test_does_not_raise_on_truncated_metadata(self, tmp_path):
        # Truncated WITHIN the header/KV/tensor-info section answers None rather
        # than crashing. Cutting mid-tensor DATA would not exercise this: that
        # section is never parsed.
        tensors = [("blk.0.ffn_gate_exps.weight", [4, 4, 8], 0, 500)]
        full = _gguf_with_tensors(
            tmp_path / "m.gguf",
            [("general.architecture", _T_STRING, "testmoe")], tensors)
        data = full.read_bytes()
        cut = tmp_path / "cut.gguf"
        cut.write_bytes(data[:40])   # well inside the KV block, before tensor-info
        assert gguf_moe_pinned_expert_bytes(cut, 1) is None   # no signal, no exception

    def test_does_not_raise_on_implausible_string_length(self, tmp_path):
        # A misaligned stream can decode a garbage 8-byte length prefix as an
        # enormous number; it must refuse rather than attempt a huge read.
        f = tmp_path / "m.gguf"
        f.write_bytes(b"GGUF" + struct.pack("<I", 3) + struct.pack("<QQ", 1, 0)
                      + struct.pack("<Q", 0xFFFFFFFFFFFF))  # implausible name length
        assert gguf_moe_pinned_expert_bytes(f, 5) is None


# --------------------------------------------------------------------------- #
#  VramSizingMixin._effective_model_bytes_for_vram: the memoized adapter        #
# --------------------------------------------------------------------------- #

class TestEffectiveModelBytesForVram:
    def _backend(self, tmp_path, *, n_cpu_moe=0, tensors=None):
        f = tmp_path / "m.gguf"
        tensors = tensors if tensors is not None else []
        _gguf_with_tensors(
            f, [("general.architecture", _T_STRING, "testmoe")], tensors)
        b = GgufBackend(str(f), n_cpu_moe=n_cpu_moe)
        return b, f

    def test_unchanged_when_n_cpu_moe_is_zero(self, tmp_path):
        tensors = [("blk.0.ffn_gate_exps.weight", [4], 0, 500)]
        b, f = self._backend(tmp_path, n_cpu_moe=0, tensors=tensors)
        assert b._effective_model_bytes_for_vram() == b._model_bytes()

    def test_subtracts_pinned_bytes_when_n_cpu_moe_set(self, tmp_path):
        tensors = [
            ("blk.0.attn_q.weight", [4], 0, 100),
            ("blk.0.ffn_gate_exps.weight", [4], 0, 500),
            ("blk.0.ffn_down_exps.weight", [4], 0, 600),
        ]
        b, f = self._backend(tmp_path, n_cpu_moe=1, tensors=tensors)
        raw = b._model_bytes()
        assert b._effective_model_bytes_for_vram() == raw - 1100

    def test_memoised_across_repeated_calls(self, tmp_path):
        tensors = [("blk.0.ffn_gate_exps.weight", [4], 0, 500)]
        b, f = self._backend(tmp_path, n_cpu_moe=1, tensors=tensors)
        with patch("localm.model_manager.gguf.gguf_moe_pinned_expert_bytes",
                   wraps=gguf_moe_pinned_expert_bytes) as spy:
            first = b._effective_model_bytes_for_vram()
            second = b._effective_model_bytes_for_vram()
        assert first == second
        assert spy.call_count == 1

    def test_falls_back_to_whole_file_when_header_unreadable(self, tmp_path):
        f = tmp_path / "opaque.gguf"
        f.write_bytes(b"\0" * 4096)
        b = GgufBackend(str(f), n_cpu_moe=10)
        assert b._effective_model_bytes_for_vram() == b._model_bytes()

    def test_falls_back_to_whole_file_for_a_dense_model(self, tmp_path):
        # n_cpu_moe is a no-op on a dense model, so the preflight charges the
        # whole file with no discount.
        tensors = [("blk.0.attn_q.weight", [4], 0, 100)]
        b, f = self._backend(tmp_path, n_cpu_moe=10, tensors=tensors)
        assert b._effective_model_bytes_for_vram() == b._model_bytes()


# --------------------------------------------------------------------------- #
#  _check_vram and an MoE model that fits once its experts are pinned          #
# --------------------------------------------------------------------------- #

class TestCheckVramHonoursNCpuMoe:
    """KB-scale, not the GB-scale a real MoE would be: conftest.py refuses to
    leave files over 100MB in tmp_path, and truncate() is not sparse on
    Windows/NTFS. The arithmetic under test is pure ratios and sums, so a
    KB-scale model with a proportionally KB-scale VRAM budget exercises the
    identical code paths. Every constant below is worked out against the real
    formulas (_kv_bytes_per_token's header shape, _check_vram's need/total
    compare) - see the inline arithmetic notes."""

    def _backend(self, tmp_path, tensors, *, n_cpu_moe, n_ctx=64, n_gpu_layers=99):
        f = tmp_path / "moe.gguf"
        # block_count=2, embedding_length=64, head_count=4, head_count_kv=4 ->
        # head_dim=16 -> kv_bytes_per_token = n_layers(2)*n_head_kv(4)*head_dim(16)*2*2 = 512.
        kv = [("general.architecture", _T_STRING, "testmoe"),
              ("testmoe.block_count", _T_UINT32, 2),
              ("testmoe.embedding_length", _T_UINT32, 64),
              ("testmoe.attention.head_count", _T_UINT32, 4),
              ("testmoe.attention.head_count_kv", _T_UINT32, 4)]
        _gguf_with_tensors(f, kv, tensors)
        b = GgufBackend(str(f), n_ctx=n_ctx, n_gpu_layers=n_gpu_layers,
                        n_cpu_moe=n_cpu_moe)
        return b

    def test_refuses_without_n_cpu_moe_but_fits_with_it(self, tmp_path, capsys):
        """A model whose bulk is expert weights is refused when the check
        charges the whole file, and fits once n_cpu_moe pins those exact bytes
        off the VRAM budget - same free VRAM, same model, only the setting
        differs.

        tensors: attn 2,000 B x2 layers, ffn_gate_exps 900,000 B x2 layers ->
        model_bytes = 1,804,000. kv_bytes_per_token=512 (see _backend), n_ctx=64
        -> kv_cache = 32,768. overhead = 10,000.
          need WITHOUT n_cpu_moe (weights = whole model, gpu_layers=99):
            1,804,000 + 32,768 + 10,000 = 1,846,768
          total = 1,500,000 -> need > total -> hard refusal ("cannot fit").
          need WITH n_cpu_moe=2 (both layers pinned, effective weights =
          just the 2x2,000 B attn tensors = 4,000):
            4,000 + 32,768 + 10,000 = 46,768
          total (1,500,000) >= 46,768 -> no hard refusal.
          free = 500,000 >= 46,768 -> fits cleanly, no "Low VRAM" warning either.
        """
        tensors = [
            ("blk.0.attn_q.weight", [4], 0, 2_000),
            ("blk.0.ffn_gate_exps.weight", [4], 0, 900_000),
            ("blk.1.attn_q.weight", [4], 0, 2_000),
            ("blk.1.ffn_gate_exps.weight", [4], 0, 900_000),
        ]
        free, total, overhead = 500_000, 1_500_000, 10_000

        def _run(n_cpu_moe):
            b = self._backend(tmp_path, tensors, n_cpu_moe=n_cpu_moe)
            with patch.object(GgufBackend, "_split_free_total_bytes",
                              return_value=(None, None, 0)), \
                 patch.object(GgufBackend, "_free_vram_bytes", return_value=free), \
                 patch.object(GgufBackend, "_total_vram_bytes", return_value=total), \
                 patch.object(GgufBackend, "_VRAM_OVERHEAD_BYTES", overhead):
                b._check_vram()
            return b

        with pytest.raises(RuntimeError, match="cannot fit"):
            _run(n_cpu_moe=0)

        _run(n_cpu_moe=2)   # must NOT raise
        out = capsys.readouterr().out
        assert "Low VRAM" not in out


# --------------------------------------------------------------------------- #
#  _auto_gpu_layers and _auto_ctx_max: the same discount, applied consistently #
# --------------------------------------------------------------------------- #

class TestAutoGpuLayersHonoursNCpuMoe:
    @staticmethod
    def _vram(free, total):
        return patch.object(
            GgufBackend, "_split_free_total_bytes",
            return_value=(None, None, 0)), \
            patch.object(GgufBackend, "_free_vram_bytes", return_value=free), \
            patch.object(GgufBackend, "_total_vram_bytes", return_value=total)

    def test_full_offload_fits_once_experts_are_pinned(self, tmp_path):
        """Single layer: attn 2,000 B, ffn_gate_exps 900,000 B -> model_bytes
        = 902,000. block_count=1, head_dim=16, head_count_kv=4 ->
        kv_bytes_per_token = 1*4*16*4 = 256. n_ctx=64 -> kv_cache = 16,384.
        overhead = 10,000.
          WITHOUT n_cpu_moe: need_full = 902,000+16,384+10,000 = 928,384.
          free=500,000 < 928,384 -> full offload does NOT fit -> partial (<99).
          WITH n_cpu_moe=1 (the only layer pinned): effective weights = 2,000
          (just attn). need_full_effective = 2,000+16,384+10,000 = 28,384.
          free (500,000) >= 28,384 -> full offload FITS -> 99."""
        tensors = [
            ("blk.0.attn_q.weight", [4], 0, 2_000),
            ("blk.0.ffn_gate_exps.weight", [4], 0, 900_000),
        ]
        kv = [("general.architecture", _T_STRING, "testmoe"),
              ("testmoe.block_count", _T_UINT32, 1),
              ("testmoe.embedding_length", _T_UINT32, 64),
              ("testmoe.attention.head_count", _T_UINT32, 4),
              ("testmoe.attention.head_count_kv", _T_UINT32, 4)]
        f = tmp_path / "moe.gguf"
        _gguf_with_tensors(f, kv, tensors)

        b_off = GgufBackend(str(f), n_ctx=64, n_cpu_moe=0)
        p1, p2, p3 = self._vram(500_000, 1_500_000)
        with p1, p2, p3, \
             patch.object(GgufBackend, "_VRAM_OVERHEAD_BYTES", 10_000):
            n_off = b_off._auto_gpu_layers()
        assert n_off is not None and n_off < 99   # cannot fully offload the whole file

        b_on = GgufBackend(str(f), n_ctx=64, n_cpu_moe=1)
        p1, p2, p3 = self._vram(500_000, 1_500_000)
        with p1, p2, p3, \
             patch.object(GgufBackend, "_VRAM_OVERHEAD_BYTES", 10_000):
            n_on = b_on._auto_gpu_layers()
        assert n_on == 99   # full offload fits once the experts are pinned


class TestAutoCtxMaxHonoursNCpuMoe:
    def test_larger_ceiling_once_experts_are_pinned(self, tmp_path):
        """Same single-layer model as TestAutoGpuLayersHonoursNCpuMoe
        (model_bytes=902,000, kv_bytes_per_token=256). free=5,000,000,
        overhead=10,000, embedder reservation pinned to 0 for determinism.
          budget WITHOUT n_cpu_moe = 5,000,000-902,000-10,000 = 4,088,000
            -> auto = 4,088,000//256 = 15,968, rounded to 1024s = 15,360.
          budget WITH n_cpu_moe=1 (effective weights=2,000) =
            5,000,000-2,000-10,000 = 4,988,000
            -> auto = 4,988,000//256 = 19,484, rounded to 1024s = 19,456.
          Both clear the _AUTO_CTX_MIN(4096) floor, so the comparison is a
          real one, not two floored answers reading as equal."""
        tensors = [
            ("blk.0.attn_q.weight", [4], 0, 2_000),
            ("blk.0.ffn_gate_exps.weight", [4], 0, 900_000),
        ]
        kv = [("general.architecture", _T_STRING, "testmoe"),
              ("testmoe.block_count", _T_UINT32, 1),
              ("testmoe.embedding_length", _T_UINT32, 64),
              ("testmoe.attention.head_count", _T_UINT32, 4),
              ("testmoe.attention.head_count_kv", _T_UINT32, 4)]
        f = tmp_path / "moe.gguf"
        _gguf_with_tensors(f, kv, tensors)

        with patch("localm.inference.backends.llamacpp._sizing."
                   "embedder_ctx_reservation_bytes", return_value=0):
            b_off = GgufBackend(str(f), n_ctx=64, n_cpu_moe=0)
            with patch.object(GgufBackend, "_split_free_total_bytes",
                              return_value=(None, None, 0)), \
                 patch.object(GgufBackend, "_free_vram_bytes",
                              return_value=5_000_000), \
                 patch.object(GgufBackend, "_VRAM_OVERHEAD_BYTES", 10_000):
                ctx_off = b_off._auto_ctx_max()
            assert ctx_off == 15360

            b_on = GgufBackend(str(f), n_ctx=64, n_cpu_moe=1)
            with patch.object(GgufBackend, "_split_free_total_bytes",
                              return_value=(None, None, 0)), \
                 patch.object(GgufBackend, "_free_vram_bytes",
                              return_value=5_000_000), \
                 patch.object(GgufBackend, "_VRAM_OVERHEAD_BYTES", 10_000):
                ctx_on = b_on._auto_ctx_max()
            assert ctx_on == 19456

        assert ctx_on > ctx_off
