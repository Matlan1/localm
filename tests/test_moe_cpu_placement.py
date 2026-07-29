# SPDX-License-Identifier: AGPL-3.0-or-later
"""Opt-in MoE expert placement (config ``n_cpu_moe``): keep the EXPERT weights of
the first N layers in system RAM via llama_model_params.tensor_buft_overrides.

What these pin, and why each matters:

  * ``gguf_expert_count`` reads ``<arch>.expert_count`` from the header pre-load,
    and refuses to guess. It is deliberately SEPARATE from the KV-size probe:
    expert weights cost VRAM but contribute nothing to the KV cache, and
    conflating the two is the bug that probe exists to fix.
  * The override ARRAY is built to llama.cpp's layout and NULL-terminated, and the
    pattern bytes are kept alive with it - ctypes does not own those strings, so
    losing them would leave the native side reading freed memory at load time.
  * Only the FUSED expert tensors are matched. The router and any shared expert
    stay put on purpose: they are read for every token and are tiny.
  * Both refusal paths are LOUD and leave the params untouched. A placement the
    user asked for that silently becomes a different placement is exactly the
    kind of false success AGENTS.md rule 5 forbids.
"""

import ctypes
import struct

from localm.model_manager.gguf import gguf_expert_count
from localm.inference.backends.llamacpp._structs import (
    LlamaModelParams, LlamaModelTensorBuftOverride)
from localm.inference.backends.llamacpp import llama as llama_mod


T_UINT32 = 4
T_STRING = 8
T_ARRAY = 9


def _s(text: str) -> bytes:
    raw = text.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _gguf(path, kv, *, version=3, magic=b"GGUF"):
    """A REAL minimal GGUF header, so the actual parser is exercised."""
    out = [magic, struct.pack("<I", version), struct.pack("<QQ", 0, len(kv))]
    for key, vtype, val in kv:
        out.append(_s(key))
        out.append(struct.pack("<I", vtype))
        if vtype == T_STRING:
            out.append(_s(val))
        elif vtype == T_UINT32:
            out.append(struct.pack("<I", val))
        elif vtype == T_ARRAY:
            out.append(struct.pack("<I", T_UINT32))
            out.append(struct.pack("<Q", len(val)))
            for v in val:
                out.append(struct.pack("<I", v))
    path.write_bytes(b"".join(out))
    return path


# --------------------------------------------------------------------------- #
#  expert detection                                                            #
# --------------------------------------------------------------------------- #

class TestExpertCount:
    def test_reads_expert_count(self, tmp_path):
        f = _gguf(tmp_path / "moe.gguf", [
            ("general.architecture", T_STRING, "qwen3moe"),
            ("qwen3moe.expert_count", T_UINT32, 128),
        ])
        assert gguf_expert_count(f) == 128

    def test_dense_model_reports_zero(self, tmp_path):
        f = _gguf(tmp_path / "dense.gguf", [
            ("general.architecture", T_STRING, "llama"),
            ("llama.block_count", T_UINT32, 32),
        ])
        assert gguf_expert_count(f) == 0

    def test_architecture_scoped(self, tmp_path):
        """An mmproj's clip.* block must not be mistaken for the LLM's."""
        f = _gguf(tmp_path / "m.gguf", [
            ("general.architecture", T_STRING, "llama"),
            ("clip.expert_count", T_UINT32, 8),
        ])
        assert gguf_expert_count(f) == 0

    def test_non_gguf_and_missing_are_zero(self, tmp_path):
        bad = tmp_path / "x.gguf"
        bad.write_bytes(b"\0" * 512)
        assert gguf_expert_count(bad) == 0
        assert gguf_expert_count(tmp_path / "nope.gguf") == 0

    def test_array_valued_expert_count_is_refused(self, tmp_path):
        """Non-scalar means we cannot answer; refuse rather than guess."""
        f = _gguf(tmp_path / "m.gguf", [
            ("general.architecture", T_STRING, "llama"),
            ("llama.expert_count", T_ARRAY, [8, 8]),
        ])
        assert gguf_expert_count(f) == 0


# --------------------------------------------------------------------------- #
#  the override array                                                          #
# --------------------------------------------------------------------------- #

class TestApplyCpuMoe:
    @staticmethod
    def _moe(tmp_path, experts=64):
        return _gguf(tmp_path / "moe.gguf", [
            ("general.architecture", T_STRING, "olmoe"),
            ("olmoe.expert_count", T_UINT32, experts),
        ])

    def test_builds_a_null_terminated_array_and_sets_the_field(
            self, tmp_path, monkeypatch):
        monkeypatch.setattr(llama_mod, "cpu_buffer_type", lambda: 0xBEEF,
                            raising=False)
        monkeypatch.setattr(
            "localm.inference.backends.llamacpp._loader.cpu_buffer_type",
            lambda: 0xBEEF)
        mp = LlamaModelParams()
        keep = llama_mod._apply_cpu_moe(mp, 3, str(self._moe(tmp_path)))
        assert keep is not None, "a resolvable CPU buft must produce an override"
        assert mp.tensor_buft_overrides, "the native field must be pointed at it"

        array = ctypes.cast(
            mp.tensor_buft_overrides,
            ctypes.POINTER(LlamaModelTensorBuftOverride))
        for i in range(3):
            assert array[i].buft == 0xBEEF
            pattern = array[i].pattern.decode()
            assert pattern.startswith(r"blk\.") and str(i) in pattern
            # only the FUSED expert tensors, never the router or shared expert
            assert "ffn_(gate|down|up)_exps" in pattern
            assert "ffn_gate_inp" not in pattern
        assert array[3].pattern is None, "array must be NULL-terminated"

    def test_pattern_bytes_are_kept_alive_with_the_array(
            self, tmp_path, monkeypatch):
        """ctypes does not own the pattern strings. If they are not retained the
        native side reads freed memory at load time."""
        monkeypatch.setattr(
            "localm.inference.backends.llamacpp._loader.cpu_buffer_type",
            lambda: 0xBEEF)
        mp = LlamaModelParams()
        keep = llama_mod._apply_cpu_moe(mp, 2, str(self._moe(tmp_path)))
        array, patterns = keep
        assert len(patterns) == 2
        assert all(isinstance(p, bytes) for p in patterns)
        assert array is not None

    def test_dense_model_is_refused_and_params_untouched(
            self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "localm.inference.backends.llamacpp._loader.cpu_buffer_type",
            lambda: 0xBEEF)
        f = _gguf(tmp_path / "dense.gguf", [
            ("general.architecture", T_STRING, "llama"),
            ("llama.block_count", T_UINT32, 32),
        ])
        mp = LlamaModelParams()
        assert llama_mod._apply_cpu_moe(mp, 4, str(f)) is None
        assert not mp.tensor_buft_overrides, (
            "a no-op setting must leave the native params alone")

    def test_unresolvable_cpu_buft_is_refused_and_params_untouched(
            self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "localm.inference.backends.llamacpp._loader.cpu_buffer_type",
            lambda: None)
        mp = LlamaModelParams()
        assert llama_mod._apply_cpu_moe(mp, 4, str(self._moe(tmp_path))) is None
        assert not mp.tensor_buft_overrides


# --------------------------------------------------------------------------- #
#  wiring                                                                      #
# --------------------------------------------------------------------------- #

def test_default_is_off():
    """Off by default: this is a footprint dial, not a free speed-up. At matched
    VRAM it measured throughput-NEUTRAL (52.23 vs 52.04 tok/s)."""
    from localm.config import DEFAULT_CONFIG
    assert DEFAULT_CONFIG["n_cpu_moe"] == 0


def test_setting_is_exposed_in_the_schema():
    from localm.settings_schema import CORE_FIELDS
    field = next((f for f in CORE_FIELDS if f.key == "n_cpu_moe"), None)
    assert field is not None, "n_cpu_moe must be user-settable"
    assert field.group == "Engine"


def test_struct_layout_is_two_pointers():
    """llama_model_tensor_buft_override is {const char *pattern; buft;}."""
    assert ctypes.sizeof(LlamaModelTensorBuftOverride) == 2 * ctypes.sizeof(
        ctypes.c_void_p)
