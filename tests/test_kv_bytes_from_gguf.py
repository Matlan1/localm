# SPDX-License-Identifier: AGPL-3.0-or-later
"""Pre-load KV sizing from the GGUF header (gguf_kv_bytes_per_token) and its use
by VramSizingMixin._kv_bytes_per_token / _auto_gpu_layers.

KV cost is a function of the ATTENTION shape only, never of the model's FILE
SIZE: a file-size estimate over-charges a sparse MoE, whose file is inflated by
expert weights that cost no KV at all, and under-charges a wide-KV dense model.
An over-charged KV shrinks the weight budget, so _auto_gpu_layers offloads fewer
layers.

These tests build REAL GGUF headers byte by byte and drive the real parser and
the real sizing path.
"""

import struct

from unittest.mock import patch

import pytest

from localm.inference.backends.gguf import GgufBackend
from localm.model_manager.gguf import (
    gguf_kv_bytes_per_token,
    gguf_mtp_draft_kv_bytes_per_token,
    gguf_nextn_predict_layers,
)


GB = 1024 ** 3

_T_UINT32 = 4
_T_INT32 = 5
_T_FLOAT32 = 6
_T_STRING = 8
_T_ARRAY = 9


def _s(text: str) -> bytes:
    raw = text.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _gguf(path, kv, *, version=3, magic=b"GGUF", tensors=()):
    """Write a REAL minimal GGUF header: magic, version, tensor/kv counts, then
    the KV block. *kv* is an ordered list of (key, type, value); value is an int
    for _T_UINT32, a float for _T_FLOAT32, a str for _T_STRING, and for _T_ARRAY
    either a list (written as uint32) or a (element_type, list) pair.

    The pair form exists because REAL GGUF writers emit the per-layer
    head_count_kv array as INT32 (type 5), as Granite 4 H and LFM2 headers do."""
    out = [magic, struct.pack("<I", version),
           struct.pack("<QQ", len(tensors), len(kv))]
    for key, vtype, val in kv:
        out.append(_s(key))
        out.append(struct.pack("<I", vtype))
        if vtype == _T_STRING:
            out.append(_s(val))
        elif vtype == _T_UINT32:
            out.append(struct.pack("<I", val))
        elif vtype == _T_FLOAT32:
            out.append(struct.pack("<f", val))
        elif vtype == _T_ARRAY:
            elem_t, items = val if isinstance(val, tuple) else (_T_UINT32, val)
            fmt = {_T_UINT32: "<I", _T_INT32: "<i", _T_FLOAT32: "<f"}[elem_t]
            out.append(struct.pack("<I", elem_t))
            out.append(struct.pack("<Q", len(items)))
            for v in items:
                out.append(struct.pack(fmt, v))
        else:
            raise AssertionError(f"test helper does not emit type {vtype}")
    for name in tensors:
        # name, n_dims, dims[n_dims], ggml type, offset - the real tensor-info
        # record layout, so the parser under test walks real bytes here too.
        out.append(_s(name))
        out.append(struct.pack("<I", 1))
        out.append(struct.pack("<Q", 1))
        out.append(struct.pack("<I", 0))
        out.append(struct.pack("<Q", 0))
    path.write_bytes(b"".join(out))
    return path


def _hybrid_tensors(n_layers, attending):
    """Tensor names for a hybrid stack: EVERY layer carries attn_norm, and only
    *attending* layers carry attn_k/attn_v. Taken from the real Qwen3-Next /
    Granite 4 H / LFM2 files."""
    names = []
    for i in range(n_layers):
        names.append(f"blk.{i}.attn_norm.weight")
        names.append(f"blk.{i}.ffn_down.weight")
        if i in attending:
            names += [f"blk.{i}.attn_k.weight", f"blk.{i}.attn_v.weight",
                      f"blk.{i}.attn_q.weight"]
        else:
            names.append(f"blk.{i}.ssm_in.weight")
    return names


def _shape(arch, n_layers, n_embd, n_head, n_head_kv, extra=()):
    kv = [
        ("general.architecture", _T_STRING, arch),
        (f"{arch}.block_count", _T_UINT32, n_layers),
        (f"{arch}.embedding_length", _T_UINT32, n_embd),
        (f"{arch}.attention.head_count", _T_UINT32, n_head),
        (f"{arch}.attention.head_count_kv", _T_UINT32, n_head_kv),
    ]
    kv.extend(extra)
    return kv


# --------------------------------------------------------------------------- #
#  gguf_kv_bytes_per_token: the header read                                     #
# --------------------------------------------------------------------------- #

class TestGgufKvBytesPerToken:
    def test_gqa_shape_matches_the_loaded_model_formula(self, tmp_path):
        # n_layers * n_head_kv * head_dim * 2 (K and V) * 2 (f16), head_dim =
        # n_embd // n_head.
        f = _gguf(tmp_path / "m.gguf", _shape("llama", 32, 4096, 32, 8))
        assert gguf_kv_bytes_per_token(f) == 32 * 8 * 128 * 2 * 2 == 131072

    def test_explicit_key_value_length_wins_over_n_embd_over_n_head(self, tmp_path):
        # Architectures whose head_dim is NOT n_embd/n_head state it outright.
        # Here n_embd//n_head would give 128, but the explicit keys say 256.
        f = _gguf(tmp_path / "m.gguf", _shape(
            "gemma3", 26, 4096, 32, 4,
            extra=[("gemma3.attention.key_length", _T_UINT32, 256),
                   ("gemma3.attention.value_length", _T_UINT32, 256)]))
        assert gguf_kv_bytes_per_token(f) == 26 * 4 * (256 + 256) * 2
        # Different from the fallback path.
        assert gguf_kv_bytes_per_token(f) != 26 * 4 * (4096 // 32) * 2 * 2

    def test_architecture_scoped_so_an_mmproj_clip_block_cannot_win(self, tmp_path):
        # A vision projector carries its own clip.* attention shape. Picking it up
        # would compute the KV of the wrong tower.
        kv = _shape("llama", 32, 4096, 32, 8, extra=[
            ("clip.block_count", _T_UINT32, 2),
            ("clip.embedding_length", _T_UINT32, 64),
            ("clip.attention.head_count", _T_UINT32, 1),
            ("clip.attention.head_count_kv", _T_UINT32, 1),
        ])
        f = _gguf(tmp_path / "m.gguf", kv)
        assert gguf_kv_bytes_per_token(f) == 131072

    def test_key_order_does_not_matter(self, tmp_path):
        # general.architecture last: the shape keys are collected by full name and
        # resolved against the architecture at the end, so order is irrelevant.
        kv = _shape("llama", 32, 4096, 32, 8)
        f = _gguf(tmp_path / "m.gguf", kv[1:] + [kv[0]])
        assert gguf_kv_bytes_per_token(f) == 131072

    # --- refusals: every one of these must answer 0, never a wrong number ---

    def test_zero_when_not_a_gguf(self, tmp_path):
        f = tmp_path / "m.gguf"
        f.write_bytes(b"\0" * 4096)
        assert gguf_kv_bytes_per_token(f) == 0

    def test_zero_when_file_missing(self, tmp_path):
        assert gguf_kv_bytes_per_token(tmp_path / "nope.gguf") == 0

    def test_zero_when_shape_keys_absent(self, tmp_path):
        f = _gguf(tmp_path / "m.gguf",
                  [("general.architecture", _T_STRING, "llama")])
        assert gguf_kv_bytes_per_token(f) == 0

    def test_per_layer_array_is_summed_not_declined(self, tmp_path):
        # The array states the KV heads of every layer individually (0 = a layer
        # holding no KV cache), so summing it is exact rather than a guess.
        kv = [
            ("general.architecture", _T_STRING, "llama"),
            ("llama.block_count", _T_UINT32, 4),
            ("llama.embedding_length", _T_UINT32, 4096),
            ("llama.attention.head_count", _T_UINT32, 32),
            ("llama.attention.head_count_kv", _T_ARRAY, [8, 8, 0, 8]),
        ]
        f = _gguf(tmp_path / "m.gguf", kv)
        assert gguf_kv_bytes_per_token(f) == (8 + 8 + 0 + 8) * 128 * 2 * 2 == 12288
        # Not the whole-stack formula: 4 layers * 8 heads would charge the silent
        # layer.
        assert gguf_kv_bytes_per_token(f) != 4 * 8 * 128 * 2 * 2

    def test_zero_on_gguf_v1(self, tmp_path):
        f = _gguf(tmp_path / "m.gguf", _shape("llama", 32, 4096, 32, 8), version=1)
        assert gguf_kv_bytes_per_token(f) == 0

    def test_zero_on_zero_valued_shape(self, tmp_path):
        f = _gguf(tmp_path / "m.gguf", _shape("llama", 32, 4096, 0, 8))
        assert gguf_kv_bytes_per_token(f) == 0

    def test_does_not_raise_on_truncated_header(self, tmp_path):
        full = _gguf(tmp_path / "m.gguf", _shape("llama", 32, 4096, 32, 8))
        data = full.read_bytes()
        cut = tmp_path / "cut.gguf"
        cut.write_bytes(data[:len(data) // 2])
        assert gguf_kv_bytes_per_token(cut) == 0     # no signal, and no exception


# --------------------------------------------------------------------------- #
#  gguf_nextn_predict_layers / gguf_mtp_draft_kv_bytes_per_token: the MTP      #
#  draft context's OWN pre-load metadata, distinct from the whole model's.     #
# --------------------------------------------------------------------------- #

class TestGgufNextnPredictLayers:
    def test_declared_single_head(self, tmp_path):
        kv = _shape("qwen35", 48, 2048, 16, 4, extra=[
            ("qwen35.nextn_predict_layers", _T_UINT32, 1)])
        f = _gguf(tmp_path / "m.gguf", kv)
        assert gguf_nextn_predict_layers(f) == ("qwen35", 1)

    def test_declared_multi_head(self, tmp_path):
        kv = _shape("step35", 60, 4096, 32, 8, extra=[
            ("step35.nextn_predict_layers", _T_UINT32, 3)])
        f = _gguf(tmp_path / "m.gguf", kv)
        assert gguf_nextn_predict_layers(f) == ("step35", 3)

    def test_no_nextn_key_at_all(self, tmp_path):
        f = _gguf(tmp_path / "m.gguf", _shape("llama", 32, 4096, 32, 8))
        assert gguf_nextn_predict_layers(f) == ("llama", 0)

    def test_zero_valued_key_reads_as_absent(self, tmp_path):
        kv = _shape("glm4moe", 47, 4096, 32, 8, extra=[
            ("glm4moe.nextn_predict_layers", _T_UINT32, 0)])
        f = _gguf(tmp_path / "m.gguf", kv)
        assert gguf_nextn_predict_layers(f) == ("glm4moe", 0)

    def test_key_order_does_not_matter(self, tmp_path):
        kv = _shape("qwen35", 48, 2048, 16, 4, extra=[
            ("qwen35.nextn_predict_layers", _T_UINT32, 1)])
        # architecture LAST: resolved against it at the end, like the KV path.
        f = _gguf(tmp_path / "m.gguf", kv[1:] + [kv[0]])
        assert gguf_nextn_predict_layers(f) == ("qwen35", 1)

    def test_architecture_scoped_so_an_mmproj_clip_block_cannot_win(self, tmp_path):
        # A vision projector's own (fabricated) nextn key must not attach to
        # the LLM's answer.
        kv = _shape("llama", 32, 4096, 32, 8, extra=[
            ("clip.nextn_predict_layers", _T_UINT32, 5)])
        f = _gguf(tmp_path / "m.gguf", kv)
        assert gguf_nextn_predict_layers(f) == ("llama", 0)

    def test_zero_when_not_a_gguf(self, tmp_path):
        f = tmp_path / "m.gguf"
        f.write_bytes(b"\0" * 4096)
        assert gguf_nextn_predict_layers(f) == ("", 0)

    def test_zero_when_file_missing(self, tmp_path):
        assert gguf_nextn_predict_layers(tmp_path / "nope.gguf") == ("", 0)

    def test_zero_when_no_architecture_declared(self, tmp_path):
        f = _gguf(tmp_path / "m.gguf",
                  [("some.nextn_predict_layers", _T_UINT32, 1)])
        assert gguf_nextn_predict_layers(f) == ("", 0)

    def test_does_not_raise_on_truncated_header(self, tmp_path):
        # architecture is the FIRST key _shape emits, so it can parse clean
        # even when the nextn key past it is cut away - what matters is that
        # this never raises and never reports a nextn count it never read.
        full = _gguf(tmp_path / "m.gguf", _shape("qwen35", 48, 2048, 16, 4, extra=[
            ("qwen35.nextn_predict_layers", _T_UINT32, 1)]))
        data = full.read_bytes()
        cut = tmp_path / "cut.gguf"
        cut.write_bytes(data[:len(data) // 2])
        _arch, nextn = gguf_nextn_predict_layers(cut)     # no exception
        assert nextn == 0


class TestGgufMtpDraftKvBytesPerToken:
    def test_gqa_shape_scaled_by_nextn_layers(self, tmp_path):
        # One draft layer: nextn_layers * n_head_kv * head_dim * 2 (K and V) *
        # 2 (f16) - the SAME per-layer formula gguf_kv_bytes_per_token uses,
        # multiplied by 1 draft layer rather than the model's 48.
        f = _gguf(tmp_path / "m.gguf", _shape("qwen35", 48, 2048, 16, 4))
        assert gguf_mtp_draft_kv_bytes_per_token(f, 1) == 4 * 128 * 2 * 2 == 2048

    def test_scales_linearly_with_nextn_layers(self, tmp_path):
        f = _gguf(tmp_path / "m.gguf", _shape("step35", 60, 4096, 32, 8))
        one = gguf_mtp_draft_kv_bytes_per_token(f, 1)
        three = gguf_mtp_draft_kv_bytes_per_token(f, 3)
        assert three == one * 3
        # NOT the whole-stack rate (60 layers).
        assert three != gguf_kv_bytes_per_token(f)

    def test_zero_when_nextn_layers_is_zero_or_negative(self, tmp_path):
        f = _gguf(tmp_path / "m.gguf", _shape("qwen35", 48, 2048, 16, 4))
        assert gguf_mtp_draft_kv_bytes_per_token(f, 0) == 0
        assert gguf_mtp_draft_kv_bytes_per_token(f, -1) == 0

    def test_explicit_key_value_length_wins_over_n_embd_over_n_head(self, tmp_path):
        f = _gguf(tmp_path / "m.gguf", _shape(
            "gemma3", 26, 4096, 32, 4,
            extra=[("gemma3.attention.key_length", _T_UINT32, 256),
                   ("gemma3.attention.value_length", _T_UINT32, 256)]))
        assert gguf_mtp_draft_kv_bytes_per_token(f, 1) == 4 * (256 + 256) * 2

    def test_hybrid_per_layer_array_uses_the_maximum_not_the_sum(self, tmp_path):
        # Granite-shaped hybrid: attending layers all carry head_count_kv=4,
        # everything else 0. A single draft layer costs the max of the
        # per-layer array, not the whole array summed (that would be
        # gguf_kv_bytes_per_token's own whole-stack answer).
        f = _gguf(tmp_path / "g.gguf", [
            ("general.architecture", _T_STRING, "granitehybrid"),
            ("granitehybrid.block_count", _T_UINT32, 40),
            ("granitehybrid.embedding_length", _T_UINT32, 1536),
            ("granitehybrid.attention.head_count", _T_UINT32, 12),
            ("granitehybrid.attention.head_count_kv", _T_ARRAY,
             (_T_INT32, _granite_layers())),
        ])
        head_dim = 1536 // 12
        assert gguf_mtp_draft_kv_bytes_per_token(f, 1) == 4 * head_dim * 2 * 2 == 2048
        assert gguf_mtp_draft_kv_bytes_per_token(f, 1) != gguf_kv_bytes_per_token(f)

    def test_architecture_scoped_so_an_mmproj_clip_block_cannot_win(self, tmp_path):
        kv = _shape("llama", 32, 4096, 32, 8, extra=[
            ("clip.block_count", _T_UINT32, 2),
            ("clip.embedding_length", _T_UINT32, 64),
            ("clip.attention.head_count", _T_UINT32, 1),
            ("clip.attention.head_count_kv", _T_UINT32, 1),
        ])
        f = _gguf(tmp_path / "m.gguf", kv)
        assert gguf_mtp_draft_kv_bytes_per_token(f, 1) == 8 * 128 * 2 * 2

    def test_zero_when_shape_keys_absent(self, tmp_path):
        f = _gguf(tmp_path / "m.gguf",
                  [("general.architecture", _T_STRING, "llama")])
        assert gguf_mtp_draft_kv_bytes_per_token(f, 1) == 0

    def test_zero_when_not_a_gguf(self, tmp_path):
        f = tmp_path / "m.gguf"
        f.write_bytes(b"\0" * 4096)
        assert gguf_mtp_draft_kv_bytes_per_token(f, 1) == 0

    def test_does_not_raise_on_truncated_header(self, tmp_path):
        full = _gguf(tmp_path / "m.gguf", _shape("qwen35", 48, 2048, 16, 4))
        data = full.read_bytes()
        cut = tmp_path / "cut.gguf"
        cut.write_bytes(data[:len(data) // 2])
        assert gguf_mtp_draft_kv_bytes_per_token(cut, 1) == 0


# --------------------------------------------------------------------------- #
#  File size vs attention shape                                                #
# --------------------------------------------------------------------------- #

class TestSparseMoeIsNotOverCharged:
    def test_same_attention_shape_costs_the_same_kv_whatever_the_file_size(
            self, tmp_path):
        """A sparse MoE and a dense model with the SAME attention shape have the
        SAME KV cost per token, whatever their file sizes."""
        moe = _gguf(tmp_path / "moe.gguf", _shape("qwen3moe", 48, 2048, 16, 4))
        dense = _gguf(tmp_path / "dense.gguf", _shape("qwen3", 48, 2048, 16, 4))
        assert gguf_kv_bytes_per_token(moe) == gguf_kv_bytes_per_token(dense)

        # A file-size estimate disagrees wildly for the same attention shape,
        # because the MoE file carries expert weights.
        heur_moe = GgufBackend._bytes_per_token(18 * GB)     # ~30B-A3B on disk
        heur_dense = GgufBackend._bytes_per_token(3 * GB)    # same attention, dense
        assert heur_moe != heur_dense, "precondition: the heuristic must differ"
        assert heur_moe > gguf_kv_bytes_per_token(moe), (
            "precondition: the heuristic must OVER-charge this MoE, or the "
            "under-offload this test guards cannot occur")


class TestAutoGpuLayersUsesTheRealShape:
    """End-to-end: _auto_gpu_layers offloads the number of layers the REAL
    attention shape allows, not the smaller number a file-size KV estimate
    yields."""

    @staticmethod
    def _vram(free, total):
        from contextlib import ExitStack
        from localm.inference.backends.llamacpp import _loader
        stack = ExitStack()
        stack.enter_context(patch.object(
            GgufBackend, "_free_total_vram_bytes", return_value=(free, total)))
        stack.enter_context(patch.object(
            _loader, "gpu_memory_isolated", return_value=(free, total)))
        stack.enter_context(patch.object(
            GgufBackend, "_device_global_free_bytes", return_value=None))
        return stack

    def _backend(self, path, size_bytes, n_ctx):
        b = GgufBackend(str(path), n_gpu_layers=99, n_gpu_layers_auto=True,
                        n_ctx=n_ctx)
        b._model_bytes = lambda: size_bytes      # faked size, real header on disk
        return b

    def test_moe_offloads_more_layers_than_the_file_size_guess_allowed(
            self, tmp_path):
        # A 30B-A3B-shaped MoE: 48 layers, GQA 4 kv-heads, head_dim 128.
        # True KV  = 48 * 4 * 128 * 2 * 2  =  98,304 B/token
        # Heuristic= 18 GB // 100_000      = 193,273 B/token  (about 2x too high)
        f = _gguf(tmp_path / "moe.gguf", _shape("qwen3moe", 48, 2048, 16, 4))
        size, n_ctx = 18 * GB, 32768

        true_kv = gguf_kv_bytes_per_token(f)
        heur_kv = GgufBackend._bytes_per_token(size)
        assert heur_kv > true_kv, "precondition: heuristic over-charges this model"

        b = self._backend(f, size, n_ctx)
        assert b._kv_bytes_per_token() == true_kv, (
            "sizing must use the header's attention shape, not the file size")

        with self._vram(12 * GB, 16 * GB):
            n = b._auto_gpu_layers()

        # What the file-size path produces, computed the same way
        # _auto_gpu_layers does it.
        free = 12 * GB
        overhead = GgufBackend._VRAM_OVERHEAD_BYTES
        layers = 32                                  # _ASSUMED_LAYERS (none cached)
        old_budget = free - n_ctx * heur_kv - overhead
        old_n = int(min(max(old_budget / size, 0.0), 1.0) * layers)
        assert 0 < n < 99
        assert n > old_n, (
            f"over-charged KV under-offloaded: now {n} layers vs {old_n} before")

    def test_falls_back_to_the_heuristic_when_the_header_is_unreadable(
            self, tmp_path):
        # Not a GGUF: sizing must keep working on the old estimate rather than
        # divide by zero or refuse the load.
        f = tmp_path / "opaque.gguf"
        f.write_bytes(b"\0" * 4096)
        b = self._backend(f, 8 * GB, 4096)
        assert b._kv_bytes_per_token() == GgufBackend._bytes_per_token(8 * GB)
        with self._vram(6 * GB, 16 * GB):
            assert 0 < b._auto_gpu_layers() < 99

    def test_loaded_model_accessor_still_wins_over_the_header(self, tmp_path):
        # Preference order: loaded model > GGUF header > heuristic. A loaded model
        # reports its real shape and must not be overridden by a header re-read.
        f = _gguf(tmp_path / "m.gguf", _shape("llama", 32, 4096, 32, 8))
        b = self._backend(f, 8 * GB, 4096)
        assert b._kv_bytes_per_token() == 131072      # header path
        b._llm = type("L", (), {"kv_bytes_per_token": 4242})()
        assert b._kv_bytes_per_token() == 4242        # loaded model wins


# --------------------------------------------------------------------------- #
#  The GUI estimate (sysstats.estimate_vram): the same shape at a separate     #
#  call site                                                                   #
# --------------------------------------------------------------------------- #

class TestEstimateVramUsesTheRealShape:
    """sysstats.estimate_vram powers /api/vram-estimate (the Settings
    performance sliders' live readout). It cannot read a GGUF header itself, as
    it only ever sees a byte count, so the caller reads the header and passes
    the result in via kv_bytes_per_token."""

    def test_sparse_moe_lands_on_the_attention_shape_not_the_heuristic(
            self, tmp_path):
        from localm.inference.backends.gguf import GgufBackend
        from localm.sysstats import estimate_vram
        f = _gguf(tmp_path / "moe.gguf", _shape("qwen3moe", 48, 2048, 16, 4))
        true_kv = gguf_kv_bytes_per_token(f)
        # A big file (as a real MoE on disk would be) so the two paths disagree.
        big_bytes = 18 * GB
        heuristic_kv = GgufBackend._bytes_per_token(big_bytes)
        assert heuristic_kv != true_kv, (
            "precondition: the heuristic must disagree with the real shape, or "
            "this test cannot tell the fix from the bug")

        est = estimate_vram(big_bytes, n_ctx=4096, kv_bytes_per_token=true_kv)
        assert est["kv_cache"] == 4096 * true_kv

    def test_wide_kv_dense_lands_on_the_attention_shape_not_the_heuristic(
            self, tmp_path):
        from localm.inference.backends.gguf import GgufBackend
        from localm.sysstats import estimate_vram
        # A small file with a wide (large head_dim) KV shape.
        f = _gguf(tmp_path / "dense.gguf", _shape(
            "gemma3", 26, 4096, 32, 4,
            extra=[("gemma3.attention.key_length", _T_UINT32, 256),
                   ("gemma3.attention.value_length", _T_UINT32, 256)]))
        true_kv = gguf_kv_bytes_per_token(f)
        small_bytes = 2 * GB
        heuristic_kv = GgufBackend._bytes_per_token(small_bytes)
        assert heuristic_kv != true_kv, (
            "precondition: the heuristic must disagree with the real shape")

        est = estimate_vram(small_bytes, n_ctx=8192, kv_bytes_per_token=true_kv)
        assert est["kv_cache"] == 8192 * true_kv

    def test_falls_back_to_the_size_class_heuristic_when_no_header_was_read(
            self, tmp_path):
        # kv_bytes_per_token=0 is the caller's signal that no header was readable
        # (missing file, non-GGUF, unresolved shape). estimate_vram still answers
        # with the last-resort heuristic rather than dividing by zero or reporting
        # 0 kv_cache.
        from localm.inference.backends.gguf import GgufBackend
        from localm.sysstats import estimate_vram
        model_bytes = 6 * GB
        est = estimate_vram(model_bytes, n_ctx=4096, kv_bytes_per_token=0)
        assert est["kv_cache"] == 4096 * GgufBackend._bytes_per_token(model_bytes)
        assert est["kv_cache"] > 0

    def test_moe_pinned_bytes_reduces_the_weights_estimate(self, tmp_path):
        """_sizing.py's VramSizingMixin._effective_model_bytes_for_vram
        discounts a Mixture-of-Experts load's weight footprint by whatever
        n_cpu_moe pins to system RAM, and this GUI estimate applies the SAME
        discount. moe_pinned_bytes=0 (the default) is a no-op."""
        from localm.sysstats import estimate_vram
        model_bytes = 800 * 1024 * 1024   # 800 MB file, matches the real MoE test scale
        pinned = 700 * 1024 * 1024        # 700 MB of it pinned to system RAM

        without = estimate_vram(model_bytes, n_ctx=0, n_gpu_layers=99)
        with_moe = estimate_vram(model_bytes, n_ctx=0, n_gpu_layers=99,
                                 moe_pinned_bytes=pinned)
        assert with_moe["weights"] == model_bytes - pinned
        assert with_moe["weights"] < without["weights"]
        assert with_moe["needed"] < without["needed"]

    def test_moe_pinned_bytes_never_goes_negative(self, tmp_path):
        # A pinned count larger than the file itself clamps to zero rather than
        # underflowing into a negative "weights" value.
        from localm.sysstats import estimate_vram
        est = estimate_vram(100, n_ctx=0, n_gpu_layers=99, moe_pinned_bytes=999)
        assert est["weights"] == 0
        assert est["needed"] >= 0


# --------------------------------------------------------------------------- #
#  HYBRID architectures: layers that hold NO KV cache                          #
#                                                                              #
#  A hybrid stack (Qwen3-Next, Granite 4 H, LFM2, Jamba, Falcon-H1 ...) mixes  #
#  attention layers with linear-attention / state-space / short-convolution    #
#  layers that keep a FIXED-size recurrent state and cost no per-token KV at   #
#  all. n_layers * n_head_kv charges every layer, so it over-charges by the    #
#  ratio of attending layers to total layers.                                  #
# --------------------------------------------------------------------------- #

def _granite_layers():
    """The real Granite 4.0 H Tiny pattern: 40 layers, attention at 5/15/25/35
    with 4 KV heads, mamba (no KV cache) everywhere else, matching that model's
    config.json layer_types."""
    return [4 if i in (5, 15, 25, 35) else 0 for i in range(40)]


def _lfm2_layers():
    """The real LFM2-1.2B pattern: 16 layers, 8 KV heads at exactly the indices
    that model's config.json lists in full_attn_idxs, short convolution (no KV
    cache) everywhere else."""
    return [8 if i in (2, 5, 8, 10, 12, 14) else 0 for i in range(16)]


class TestHybridPerLayerKvHeads:
    """The ARRAY form carries the exact per-layer truth, so it is summed."""

    def test_granite_shape_sums_only_the_attending_layers(self, tmp_path):
        # int32 element type, which is what real writers emit.
        f = _gguf(tmp_path / "g.gguf", [
            ("general.architecture", _T_STRING, "granitehybrid"),
            ("granitehybrid.block_count", _T_UINT32, 40),
            ("granitehybrid.embedding_length", _T_UINT32, 1536),
            ("granitehybrid.attention.head_count", _T_UINT32, 12),
            ("granitehybrid.attention.head_count_kv", _T_ARRAY,
             (_T_INT32, _granite_layers())),
            ("granitehybrid.ssm.state_size", _T_UINT32, 128),
        ])
        head_dim = 1536 // 12
        assert gguf_kv_bytes_per_token(f) == (4 * 4) * head_dim * 2 * 2 == 8192
        # Charging all 40 layers is 10x the truth: only 4 of 40 attend.
        assert gguf_kv_bytes_per_token(f) != 40 * 4 * head_dim * 2 * 2

    def test_lfm2_shape_has_no_ssm_keys_at_all(self, tmp_path):
        # lfm2 is hybrid via SHORT CONVOLUTION and declares no ssm.* key, so
        # hybrid detection keyed on ssm.* alone would miss it; the array form
        # needs no detection at all.
        f = _gguf(tmp_path / "l.gguf", [
            ("general.architecture", _T_STRING, "lfm2"),
            ("lfm2.block_count", _T_UINT32, 16),
            ("lfm2.embedding_length", _T_UINT32, 2048),
            ("lfm2.attention.head_count", _T_UINT32, 32),
            ("lfm2.attention.head_count_kv", _T_ARRAY, (_T_INT32, _lfm2_layers())),
            ("lfm2.shortconv.l_cache", _T_UINT32, 3),
        ])
        head_dim = 2048 // 32
        assert gguf_kv_bytes_per_token(f) == (6 * 8) * head_dim * 2 * 2 == 12288
        assert gguf_kv_bytes_per_token(f) != 16 * 8 * head_dim * 2 * 2

    def test_uint32_and_int32_arrays_agree(self, tmp_path):
        # The element-type table covers what real writers emit, int32 included.
        def build(elem_t, name):
            return _gguf(tmp_path / name, [
                ("general.architecture", _T_STRING, "granitehybrid"),
                ("granitehybrid.block_count", _T_UINT32, 40),
                ("granitehybrid.embedding_length", _T_UINT32, 1536),
                ("granitehybrid.attention.head_count", _T_UINT32, 12),
                ("granitehybrid.attention.head_count_kv", _T_ARRAY,
                 (elem_t, _granite_layers())),
            ])
        assert (gguf_kv_bytes_per_token(build(_T_INT32, "i.gguf"))
                == gguf_kv_bytes_per_token(build(_T_UINT32, "u.gguf")) == 8192)

    def test_explicit_key_value_length_still_wins_for_a_hybrid(self, tmp_path):
        f = _gguf(tmp_path / "h.gguf", [
            ("general.architecture", _T_STRING, "hyb"),
            ("hyb.block_count", _T_UINT32, 4),
            ("hyb.embedding_length", _T_UINT32, 1024),
            ("hyb.attention.head_count", _T_UINT32, 8),
            ("hyb.attention.head_count_kv", _T_ARRAY, (_T_INT32, [0, 2, 0, 2])),
            ("hyb.attention.key_length", _T_UINT32, 256),
            ("hyb.attention.value_length", _T_UINT32, 256),
        ])
        assert gguf_kv_bytes_per_token(f) == 4 * (256 + 256) * 2

    # --- refusals ---------------------------------------------------------- #

    def test_zero_when_the_array_length_disagrees_with_block_count(self, tmp_path):
        # One of the two is not describing the same stack, so the probe refuses.
        f = _gguf(tmp_path / "m.gguf", [
            ("general.architecture", _T_STRING, "hyb"),
            ("hyb.block_count", _T_UINT32, 40),
            ("hyb.embedding_length", _T_UINT32, 1536),
            ("hyb.attention.head_count", _T_UINT32, 12),
            ("hyb.attention.head_count_kv", _T_ARRAY, (_T_INT32, [4, 0, 0])),
        ])
        assert gguf_kv_bytes_per_token(f) == 0

    def test_zero_when_no_layer_attends(self, tmp_path):
        # A fully recurrent stack has no growing KV cache. 0 means "no signal"
        # and the caller keeps its heuristic.
        f = _gguf(tmp_path / "m.gguf", [
            ("general.architecture", _T_STRING, "mamba2"),
            ("mamba2.block_count", _T_UINT32, 4),
            ("mamba2.embedding_length", _T_UINT32, 1024),
            ("mamba2.attention.head_count", _T_UINT32, 8),
            ("mamba2.attention.head_count_kv", _T_ARRAY, (_T_INT32, [0, 0, 0, 0])),
        ])
        assert gguf_kv_bytes_per_token(f) == 0

    def test_zero_when_the_array_is_not_integers(self, tmp_path):
        # A float array is not a head count.
        f = _gguf(tmp_path / "m.gguf", [
            ("general.architecture", _T_STRING, "hyb"),
            ("hyb.block_count", _T_UINT32, 4),
            ("hyb.embedding_length", _T_UINT32, 1024),
            ("hyb.attention.head_count", _T_UINT32, 8),
            ("hyb.attention.head_count_kv", _T_ARRAY,
             (_T_FLOAT32, [1.0, 2.0, 3.0, 4.0])),
        ])
        assert gguf_kv_bytes_per_token(f) == 0


class TestHybridStatedAsAScalarWithoutTensorInfo:
    """Qwen3-Next states ONE head_count_kv for a stack whose layers differ, and
    its METADATA records nothing about which layers attend: no
    full_attention_interval, no layer_types, no array.

    These fixtures carry no tensor list either, so nothing in the file can
    answer and the probe declines. When the tensor list IS present it answers
    exactly - see TestScalarHybridResolvedFromTensorNames."""

    @staticmethod
    def _qwen3next(tmp_path, extra=()):
        kv = [
            ("general.architecture", _T_STRING, "qwen3next"),
            ("qwen3next.block_count", _T_UINT32, 48),
            ("qwen3next.embedding_length", _T_UINT32, 2048),
            ("qwen3next.attention.head_count", _T_UINT32, 16),
            ("qwen3next.attention.head_count_kv", _T_UINT32, 2),
            ("qwen3next.attention.key_length", _T_UINT32, 256),
            ("qwen3next.attention.value_length", _T_UINT32, 256),
        ]
        kv.extend(extra)
        return _gguf(tmp_path / "q.gguf", kv)

    def test_scalar_plus_ssm_marker_refuses(self, tmp_path):
        f = self._qwen3next(tmp_path, extra=[
            ("qwen3next.ssm.conv_kernel", _T_UINT32, 4),
            ("qwen3next.ssm.state_size", _T_UINT32, 128),
        ])
        assert gguf_kv_bytes_per_token(f) == 0
        # Only 12 of 48 layers attend, so the whole-stack product is 4.0x the
        # truth (which the header does not record).
        assert gguf_kv_bytes_per_token(f) != 48 * 2 * (256 + 256) * 2 == 98304

    def test_scalar_plus_shortconv_marker_refuses(self, tmp_path):
        f = self._qwen3next(tmp_path,
                            extra=[("qwen3next.shortconv.l_cache", _T_UINT32, 3)])
        assert gguf_kv_bytes_per_token(f) == 0

    def test_without_any_marker_the_scalar_is_still_used(self, tmp_path):
        # Same shape, no recurrent marker, so the refusal does not fire.
        f = self._qwen3next(tmp_path)
        assert gguf_kv_bytes_per_token(f) == 48 * 2 * (256 + 256) * 2 == 98304

    def test_another_architectures_ssm_keys_do_not_veto(self, tmp_path):
        # Arch-scoped exactly like the shape keys: an mmproj or a second tower
        # carrying ssm.* does not make the LLM refuse.
        f = _gguf(tmp_path / "m.gguf", _shape("llama", 32, 4096, 32, 8, extra=[
            ("clip.ssm.state_size", _T_UINT32, 128),
            ("someothermodel.shortconv.l_cache", _T_UINT32, 3),
        ]))
        assert gguf_kv_bytes_per_token(f) == 131072


class TestScalarHybridResolvedFromTensorNames:
    """A hybrid that states ONE head_count_kv does not record which layers attend
    anywhere in its metadata. The TENSOR NAMES do, exactly and with no
    architecture table: an attending layer carries attn_k/attn_v weights, a
    linear-attention / SSM / short-convolution layer does not.

    Broadcasting the single head count across every layer, as some other
    readers do, over-charges Qwen3-Next by 4x. Reading the tensor list gets the
    exact figure."""

    QWEN3NEXT_ATTENDING = frozenset(range(3, 48, 4))    # 12 of 48, from the file

    @classmethod
    def _qwen3next(cls, tmp_path, *, tensors=None, attending=None):
        kv = [
            ("general.architecture", _T_STRING, "qwen3next"),
            ("qwen3next.block_count", _T_UINT32, 48),
            ("qwen3next.embedding_length", _T_UINT32, 2048),
            ("qwen3next.attention.head_count", _T_UINT32, 16),
            ("qwen3next.attention.head_count_kv", _T_UINT32, 2),
            ("qwen3next.attention.key_length", _T_UINT32, 256),
            ("qwen3next.attention.value_length", _T_UINT32, 256),
            ("qwen3next.ssm.conv_kernel", _T_UINT32, 4),
            ("qwen3next.ssm.state_size", _T_UINT32, 128),
        ]
        if tensors is None:
            tensors = _hybrid_tensors(
                48, cls.QWEN3NEXT_ATTENDING if attending is None else attending)
        return _gguf(tmp_path / "q.gguf", kv, tensors=tensors)

    def test_exact_value_from_the_tensor_list(self, tmp_path):
        f = self._qwen3next(tmp_path)
        # 12 attending layers * 2 KV heads * (256 + 256) * 2 bytes.
        assert gguf_kv_bytes_per_token(f) == 12 * 2 * (256 + 256) * 2 == 24576
        # Not the whole-stack product that charging every layer gives.
        assert gguf_kv_bytes_per_token(f) != 48 * 2 * (256 + 256) * 2 == 98304

    def test_attn_norm_on_every_layer_does_not_count_as_attending(self, tmp_path):
        # Every layer of all three real hybrids carries blk.<i>.attn_norm.weight
        # whether it attends or not, so matching a bare "attn" would count the
        # whole stack.
        names = [f"blk.{i}.attn_norm.weight" for i in range(48)]
        names += [f"blk.{i}.attn_k.weight" for i in self.QWEN3NEXT_ATTENDING]
        assert sum("attn_norm" in n for n in names) == 48, "precondition"
        f = self._qwen3next(tmp_path, tensors=names)
        assert gguf_kv_bytes_per_token(f) == 24576

    def test_a_different_attending_pattern_gives_a_different_answer(self, tmp_path):
        # Half the layers attending doubles the answer.
        f = self._qwen3next(tmp_path, attending=frozenset(range(1, 48, 2)))
        assert gguf_kv_bytes_per_token(f) == 24 * 2 * (256 + 256) * 2 == 49152

    def test_zero_when_the_file_carries_no_tensor_list(self, tmp_path):
        # A metadata-only prefix, or a file whose tensor list sits past the bounded
        # read, cannot answer, so the probe declines.
        f = self._qwen3next(tmp_path, tensors=())
        assert gguf_kv_bytes_per_token(f) == 0

    def test_zero_when_no_layer_carries_attention_weights(self, tmp_path):
        # A fully recurrent stack: nothing to charge per token.
        f = self._qwen3next(tmp_path, attending=frozenset())
        assert gguf_kv_bytes_per_token(f) == 0

    def test_zero_when_more_layers_attend_than_the_stack_has(self, tmp_path):
        # block_count and the tensor list disagree about what they describe.
        names = [f"blk.{i}.attn_k.weight" for i in range(60)]
        f = self._qwen3next(tmp_path, tensors=names)
        assert gguf_kv_bytes_per_token(f) == 0

    def test_a_uniform_architecture_never_reads_the_tensor_list(self, tmp_path):
        # No recurrent marker: the scalar already speaks for every layer, the
        # second pass does not run, and a tensor list claiming otherwise does not
        # change the answer.
        f = _gguf(tmp_path / "u.gguf", _shape("llama", 32, 4096, 32, 8),
                  tensors=[f"blk.{i}.attn_k.weight" for i in range(4)])
        assert gguf_kv_bytes_per_token(f) == 131072


# --------------------------------------------------------------------------- #
#  Bound to the REAL published artefacts. The expected values are ground truth  #
#  taken from each model's own config.json, not from this implementation.       #
# --------------------------------------------------------------------------- #

_REAL_HEADERS = [
    # (id, repo path, expected bytes/token, why)
    ("qwen3next",
     "Qwen/Qwen3-Next-80B-A3B-Instruct-GGUF/resolve/main/"
     "Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf",
     12 * 2 * (256 + 256) * 2,
     "48 layers, config.json full_attention_interval 4 so exactly 12 attend "
     "(the tensor list carries attn_k/attn_v on blocks 3,7,...,47), 2 KV heads "
     "each, key and value length 256"),
    ("granitehybrid",
     "ibm-granite/granite-4.0-h-tiny-GGUF/resolve/main/"
     "granite-4.0-h-tiny-Q4_K_M.gguf",
     16 * (1536 // 12) * 2 * 2,
     "40 layers, config.json layer_types marks attention at 5/15/25/35 with 4 "
     "KV heads each: 16 KV heads over the whole stack, head_dim 128"),
    ("lfm2",
     "LiquidAI/LFM2-1.2B-GGUF/resolve/main/LFM2-1.2B-Q4_K_M.gguf",
     48 * (2048 // 32) * 2 * 2,
     "16 layers, config.json full_attn_idxs [2,5,8,10,12,14] with 8 KV heads "
     "each: 48 KV heads over the whole stack, head_dim 64"),
]


@pytest.mark.integration
@pytest.mark.parametrize("name,repo_path,expected,why",
                         _REAL_HEADERS, ids=[h[0] for h in _REAL_HEADERS])
def test_real_published_header(tmp_path, name, repo_path, expected, why):
    """Range-fetch a real GGUF's leading bytes, far enough in to carry both the
    metadata block and the tensor list - the tensor list is what resolves a
    hybrid that states a single head count, and it sits after the tokenizer
    vocab (about 5.7 MiB into the Qwen3-Next file). In production localm has the
    whole file on disk, so this prefix stands in for that."""
    import urllib.request
    from localm.http_ssl import verified_urlopen

    prefix = 16 * 1024 * 1024
    url = f"https://huggingface.co/{repo_path}"
    req = urllib.request.Request(
        url, headers={"Range": f"bytes=0-{prefix - 1}"})
    try:
        with verified_urlopen(req, timeout=60) as r:
            status, body = r.status, r.read()
    except Exception as exc:
        pytest.skip(f"cannot reach {url}: {type(exc).__name__}: {exc}")

    # Verify the fetch before trusting anything computed from it: a probe run over
    # an error page returns 0, which is also the expected answer for qwen3next.
    if status not in (200, 206) or body[:4] != b"GGUF":
        pytest.skip(f"{url} did not serve a GGUF header (status {status}, "
                    f"{len(body)} bytes, magic {body[:4]!r})")

    f = tmp_path / f"{name}.gguf"
    f.write_bytes(body)
    assert gguf_kv_bytes_per_token(f) == expected, why
