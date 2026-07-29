# SPDX-License-Identifier: AGPL-3.0-or-later
"""Pre-load KV sizing from the GGUF header (gguf_kv_bytes_per_token) and its use
by VramSizingMixin._kv_bytes_per_token / _auto_gpu_layers.

The defect these pin: the per-token KV cost used to be derived from the model's
FILE SIZE (_bytes_per_token), but KV cost is a function of the ATTENTION shape
only. That is wrong in both directions - it over-charges a sparse MoE, whose file
is inflated by expert weights that cost no KV at all, and under-charges a wide-KV
dense model. The MoE direction silently UNDER-offloads (weight budget shrinks, so
_auto_gpu_layers picks fewer layers) and nothing warns.

These tests build REAL GGUF headers byte by byte and drive the real parser and the
real sizing path - no mock stands in for the code under test.
"""

import struct

from unittest.mock import patch

from localm.inference.backends.gguf import GgufBackend
from localm.model_manager.gguf import gguf_kv_bytes_per_token


GB = 1024 ** 3

_T_UINT32 = 4
_T_STRING = 8
_T_ARRAY = 9


def _s(text: str) -> bytes:
    raw = text.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _gguf(path, kv, *, version=3, magic=b"GGUF"):
    """Write a REAL minimal GGUF header: magic, version, tensor/kv counts, then
    the KV block. *kv* is an ordered list of (key, type, value); value is an int
    for _T_UINT32, a str for _T_STRING, or a list[int] for _T_ARRAY."""
    out = [magic, struct.pack("<I", version), struct.pack("<QQ", 0, len(kv))]
    for key, vtype, val in kv:
        out.append(_s(key))
        out.append(struct.pack("<I", vtype))
        if vtype == _T_STRING:
            out.append(_s(val))
        elif vtype == _T_UINT32:
            out.append(struct.pack("<I", val))
        elif vtype == _T_ARRAY:
            out.append(struct.pack("<I", _T_UINT32))
            out.append(struct.pack("<Q", len(val)))
            for v in val:
                out.append(struct.pack("<I", v))
        else:
            raise AssertionError(f"test helper does not emit type {vtype}")
    path.write_bytes(b"".join(out))
    return path


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
        # n_embd // n_head. Same number LlamaCpp._read_kv_bytes_per_token would
        # compute post-load for this shape (cf. test_kv_bytes_offload.py).
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
        # and that is genuinely different from the fallback path, or this test
        # would pass for the wrong reason
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

    def test_zero_when_head_count_kv_is_an_array(self, tmp_path):
        # Hybrid/per-layer attention states head_count_kv as an ARRAY. A single
        # number would be wrong, so the probe must decline rather than guess.
        kv = [
            ("general.architecture", _T_STRING, "llama"),
            ("llama.block_count", _T_UINT32, 4),
            ("llama.embedding_length", _T_UINT32, 4096),
            ("llama.attention.head_count", _T_UINT32, 32),
            ("llama.attention.head_count_kv", _T_ARRAY, [8, 8, 0, 8]),
        ]
        f = _gguf(tmp_path / "m.gguf", kv)
        assert gguf_kv_bytes_per_token(f) == 0

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
#  The actual defect: file size vs attention shape                             #
# --------------------------------------------------------------------------- #

class TestSparseMoeIsNotOverCharged:
    def test_same_attention_shape_costs_the_same_kv_whatever_the_file_size(
            self, tmp_path):
        """A sparse MoE and a dense model with the SAME attention shape have the
        SAME KV cost per token. The file-size heuristic says otherwise, and that
        difference is the bug."""
        moe = _gguf(tmp_path / "moe.gguf", _shape("qwen3moe", 48, 2048, 16, 4))
        dense = _gguf(tmp_path / "dense.gguf", _shape("qwen3", 48, 2048, 16, 4))
        assert gguf_kv_bytes_per_token(moe) == gguf_kv_bytes_per_token(dense)

        # The heuristic they replace disagrees wildly for the same attention
        # shape, purely because the MoE file carries expert weights.
        heur_moe = GgufBackend._bytes_per_token(18 * GB)     # ~30B-A3B on disk
        heur_dense = GgufBackend._bytes_per_token(3 * GB)    # same attention, dense
        assert heur_moe != heur_dense, "precondition: the heuristic must differ"
        assert heur_moe > gguf_kv_bytes_per_token(moe), (
            "precondition: the heuristic must OVER-charge this MoE, or the "
            "under-offload this test guards cannot occur")


class TestAutoGpuLayersUsesTheRealShape:
    """End-to-end: the over-charged KV shrank the weight budget, so _auto_gpu_layers
    offloaded FEWER layers than would actually fit. This is the negative case - it
    fails against the file-size-only code."""

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

        # What the old, over-charged path would have produced, computed the same
        # way _auto_gpu_layers does it, so this is a real comparison rather than a
        # magic number.
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
