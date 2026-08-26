# SPDX-License-Identifier: AGPL-3.0-or-later
"""Opt-in MoE expert placement (config ``n_cpu_moe``): keep the EXPERT weights of
the first N layers in system RAM via llama_model_params.tensor_buft_overrides.

What these pin:

  * ``gguf_expert_count`` reads ``<arch>.expert_count`` from the header pre-load,
    and refuses to guess. It is SEPARATE from the KV-size probe: expert weights
    cost VRAM but contribute nothing to the KV cache.
  * The override ARRAY is built to llama.cpp's layout and NULL-terminated, and the
    pattern bytes are kept alive with it - ctypes does not own those strings, so
    losing them would leave the native side reading freed memory at load time.
  * Only the FUSED expert tensors are matched. The router and any shared expert
    stay put: they are read for every token and are tiny.
  * Both refusal paths are LOUD and leave the params untouched.
"""

import ctypes
import struct

from localm.model_manager.gguf import gguf_expert_count
from localm.inference.backends.llamacpp._structs import (
    LlamaModelParamsV1, LlamaModelParamsV2, LlamaModelTensorBuftOverride)

# _apply_cpu_moe only touches `tensor_buft_overrides`, which sits at the same
# offset in both llama_model_params layouts, so these tests are layout-agnostic
# and use V1 as the container.
LlamaModelParams = LlamaModelParamsV1
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
        keep, skip_reason = llama_mod._apply_cpu_moe(mp, 3, str(self._moe(tmp_path)))
        assert keep is not None, "a resolvable CPU buft must produce an override"
        assert skip_reason is None, "a successful override has no skip reason"
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
        keep, skip_reason = llama_mod._apply_cpu_moe(mp, 2, str(self._moe(tmp_path)))
        assert skip_reason is None
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
        keep, skip_reason = llama_mod._apply_cpu_moe(mp, 4, str(f))
        assert keep is None
        assert skip_reason == "no_experts", (
            "a caller (GgufWorker.load(), then the PARENT process) needs this "
            "exact key to render the right message - see MOE_SKIP_MESSAGES")
        assert not mp.tensor_buft_overrides, (
            "a no-op setting must leave the native params alone")

    def test_unresolvable_cpu_buft_is_refused_and_params_untouched(
            self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "localm.inference.backends.llamacpp._loader.cpu_buffer_type",
            lambda: None)
        mp = LlamaModelParams()
        keep, skip_reason = llama_mod._apply_cpu_moe(mp, 4, str(self._moe(tmp_path)))
        assert keep is None
        assert skip_reason == "buffer_unresolved"
        assert not mp.tensor_buft_overrides

    def test_skip_reasons_never_console_print_from_this_function(
            self, tmp_path, monkeypatch):
        """_apply_cpu_moe runs INSIDE the isolated worker child (see its own
        docstring) - it may never console.print (the child's stdout is not
        the server's own console). Both refusal branches report the fact via
        their return value only, asserted by spying on the shared console
        object itself."""
        from localm.console import console as real_console
        calls = []
        monkeypatch.setattr(real_console, "print",
                            lambda *a, **k: calls.append((a, k)))

        mp = LlamaModelParams()
        f = _gguf(tmp_path / "dense.gguf", [
            ("general.architecture", T_STRING, "llama"),
            ("llama.block_count", T_UINT32, 32),
        ])
        llama_mod._apply_cpu_moe(mp, 4, str(f))
        assert calls == [], (
            f"_apply_cpu_moe printed directly from the child: {calls}")

        monkeypatch.setattr(
            "localm.inference.backends.llamacpp._loader.cpu_buffer_type",
            lambda: None)
        llama_mod._apply_cpu_moe(mp, 4, str(self._moe(tmp_path)))
        assert calls == [], (
            f"_apply_cpu_moe printed directly from the child: {calls}")

    def test_every_skip_reason_has_a_rendered_message(self):
        """Every key _apply_cpu_moe can return through skip_reason must have
        a corresponding entry in MOE_SKIP_MESSAGES - the parent
        (GgufBackend._load_native) looks it up by that exact key, and a
        missing entry falls back to a generic message instead of the specific
        one."""
        assert set(llama_mod.MOE_SKIP_MESSAGES) == {"no_experts", "buffer_unresolved"}
        for reason, message in llama_mod.MOE_SKIP_MESSAGES.items():
            assert "n_cpu_moe" in message, reason
            assert message.strip(), reason


def test_tensor_buft_overrides_offset_is_layout_agnostic():
    """The premise of using one layout's class above: the field MoE placement
    writes did not move in the lemonade b1288 -> b1307 reorder. If a future reorder moves
    it, these tests must start covering both layouts instead of silently
    exercising the wrong offset."""
    assert (LlamaModelParamsV1.tensor_buft_overrides.offset
            == LlamaModelParamsV2.tensor_buft_overrides.offset == 8)
    assert (LlamaModelParamsV1.n_gpu_layers.offset
            == LlamaModelParamsV2.n_gpu_layers.offset == 16)


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
