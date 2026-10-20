# SPDX-License-Identifier: AGPL-3.0-or-later
"""Split GGUFs whose first part holds only metadata (no tensors), the layout
``llama-gguf-split --no-tensor-first-split`` writes and large published models
ship in, and split GGUFs whose layers are spread over several parts.

Every per-tensor probe must read the whole set: the input-layer bytes that stay
in system RAM, the routed-expert bytes per block that n_cpu_moe planning needs,
the per-layer split layout, the attending layers behind the KV cost and the
recurrent layers behind the recurrent-state cost. These tests write REAL GGUF
parts byte by byte (header, KV block, tensor-info block, padding, data) and
drive the real parsers and the real sizing code.
"""

import struct
from unittest.mock import patch

from localm import discover
from localm.inference.backends.gguf import GgufBackend
from localm.inference.backends.llamacpp import _loader
from localm.model_manager.gguf import (
    _gguf_tensor_offset_entries,
    gguf_block_bytes,
    gguf_input_layer_bytes,
    gguf_kv_bytes_per_token,
    gguf_moe_pinned_expert_bytes,
    gguf_recurrent_state_bytes,
    gguf_split_layout,
)

KB = 1024
MB = 1024 ** 2

_T_UINT32 = 4
_T_STRING = 8
_T_ARRAY = 9

ARCH = "qwen4exp"
N_LAYERS = 4
ATTENDING = (3,)
EMBD = 64
HEADS_KV = 2
KEY_LEN = 16
CONV_KERNEL, STATE, INNER, GROUPS = 4, 8, 32, 2

INPUT_BYTES = {"token_embd.weight": 48 * KB, "per_layer_token_embd.weight": 640 * KB}
EXPERT_BYTES = 96 * KB         # each of gate/up/down per block
OTHER_BLOCK_BYTES = KB         # each non-expert tensor per block
OUTPUT_BYTES = 48 * KB


def _s(text):
    raw = text.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _kv_block(kv):
    out = []
    for key, vtype, val in kv:
        out.append(_s(key) + struct.pack("<I", vtype))
        if vtype == _T_STRING:
            out.append(_s(val))
        elif vtype == _T_UINT32:
            out.append(struct.pack("<I", val))
        elif vtype == _T_ARRAY:
            out.append(struct.pack("<IQ", _T_STRING, len(val)))
            out.extend(_s(v) for v in val)
        else:
            raise AssertionError(vtype)
    return b"".join(out)


def _write_part(path, kv, tensors):
    """One real GGUF part: *kv* is (key, type, value) triples, *tensors* is
    (name, size_bytes) pairs laid out contiguously in list order."""
    header = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kv)) + _kv_block(kv)
    info, off = [], 0
    for name, size in tensors:
        info.append(_s(name) + struct.pack("<IQIQ", 1, 1, 0, off))
        off += size
    body = header + b"".join(info)
    body += b"\0" * (-len(body) % 32)
    with open(path, "wb") as f:
        f.write(body)
        for _name, size in tensors:
            f.write(b"\xab" * size)
    return path


def _meta_kv():
    return [
        ("general.architecture", _T_STRING, ARCH),
        (f"{ARCH}.block_count", _T_UINT32, N_LAYERS),
        (f"{ARCH}.embedding_length", _T_UINT32, EMBD),
        (f"{ARCH}.attention.head_count", _T_UINT32, 4),
        (f"{ARCH}.attention.head_count_kv", _T_UINT32, HEADS_KV),
        (f"{ARCH}.attention.key_length", _T_UINT32, KEY_LEN),
        (f"{ARCH}.attention.value_length", _T_UINT32, KEY_LEN),
        (f"{ARCH}.expert_count", _T_UINT32, 8),
        (f"{ARCH}.expert_used_count", _T_UINT32, 2),
        (f"{ARCH}.ssm.conv_kernel", _T_UINT32, CONV_KERNEL),
        (f"{ARCH}.ssm.state_size", _T_UINT32, STATE),
        (f"{ARCH}.ssm.inner_size", _T_UINT32, INNER),
        (f"{ARCH}.ssm.group_count", _T_UINT32, GROUPS),
        ("tokenizer.ggml.tokens", _T_ARRAY, [f"t{i}" for i in range(64)]),
    ]


def _block_tensors(layer):
    names = [(f"blk.{layer}.attn_norm.weight", OTHER_BLOCK_BYTES),
             (f"blk.{layer}.ffn_gate_inp.weight", OTHER_BLOCK_BYTES)]
    if layer in ATTENDING:
        names += [(f"blk.{layer}.attn_k.weight", OTHER_BLOCK_BYTES),
                  (f"blk.{layer}.attn_v.weight", OTHER_BLOCK_BYTES)]
    else:
        names += [(f"blk.{layer}.ssm_conv1d.weight", OTHER_BLOCK_BYTES)]
    names += [(f"blk.{layer}.ffn_{w}_exps.weight", EXPERT_BYTES)
              for w in ("gate", "up", "down")]
    return names


def _input_tensors():
    return list(INPUT_BYTES.items())


def _metadata_only_split(tmp_path):
    """Part 1: metadata, no tensors. Part 2: every tensor."""
    first = _write_part(tmp_path / "m-00001-of-00002.gguf", _meta_kv(), [])
    tensors = _input_tensors()
    for layer in range(N_LAYERS):
        tensors += _block_tensors(layer)
    tensors.append(("output.weight", OUTPUT_BYTES))
    _write_part(tmp_path / "m-00002-of-00002.gguf",
                [("split.count", _T_UINT32, 2)], tensors)
    return first


def _layers_over_parts_split(tmp_path):
    """Part 1: metadata, the input tensors and blocks 0-1. Part 2: blocks 2-3
    and the output, so the attending block sits only in part 2."""
    first_tensors = _input_tensors() + _block_tensors(0) + _block_tensors(1)
    first = _write_part(tmp_path / "m-00001-of-00002.gguf", _meta_kv(), first_tensors)
    _write_part(tmp_path / "m-00002-of-00002.gguf", [("split.count", _T_UINT32, 2)],
                _block_tensors(2) + _block_tensors(3) + [("output.weight", OUTPUT_BYTES)])
    return first


def _expected_kv_per_token():
    return len(ATTENDING) * HEADS_KV * (KEY_LEN + KEY_LEN) * 2


def _expected_recurrent_state():
    conv = (CONV_KERNEL - 1) * (INNER + 2 * GROUPS * STATE)
    return (N_LAYERS - len(ATTENDING)) * (conv + STATE * INNER) * 4


# --------------------------------------------------------------------------- #
#  The header probes                                                           #
# --------------------------------------------------------------------------- #

class TestMetadataOnlyFirstPart:
    def test_a_part_without_tensors_parses_to_no_entries(self, tmp_path):
        first = _metadata_only_split(tmp_path)
        parsed = _gguf_tensor_offset_entries(first)
        assert parsed is not None
        assert parsed[0] == []

    def test_input_layer_bytes_are_read_from_the_later_part(self, tmp_path):
        first = _metadata_only_split(tmp_path)
        assert gguf_input_layer_bytes(first) == sum(INPUT_BYTES.values())

    def test_block_bytes_cover_every_block(self, tmp_path):
        blocks = gguf_block_bytes(_metadata_only_split(tmp_path))
        assert blocks is not None
        assert sorted(blocks) == list(range(N_LAYERS))
        assert all(experts == 3 * EXPERT_BYTES for _total, experts in blocks.values())

    def test_split_layout_is_read(self, tmp_path):
        layout = gguf_split_layout(_metadata_only_split(tmp_path))
        assert layout is not None
        assert layout["block_count"] == N_LAYERS
        assert layout["tensor_bytes"]["blk.3.ffn_up_exps.weight"] == EXPERT_BYTES

    def test_pinned_expert_bytes(self, tmp_path):
        first = _metadata_only_split(tmp_path)
        assert gguf_moe_pinned_expert_bytes(first, 2) == 2 * 3 * EXPERT_BYTES

    def test_kv_cost_counts_the_attending_layers_of_the_later_part(self, tmp_path):
        first = _metadata_only_split(tmp_path)
        assert gguf_kv_bytes_per_token(first) == _expected_kv_per_token()

    def test_recurrent_state_counts_the_recurrent_layers_of_the_later_part(self, tmp_path):
        first = _metadata_only_split(tmp_path)
        assert gguf_recurrent_state_bytes(first) == _expected_recurrent_state()

    def test_a_missing_later_part_is_no_answer(self, tmp_path):
        first = _metadata_only_split(tmp_path)
        (tmp_path / "m-00002-of-00002.gguf").unlink()
        assert gguf_block_bytes(first) is None
        assert gguf_split_layout(first) is None
        assert gguf_moe_pinned_expert_bytes(first, 2) is None
        assert gguf_kv_bytes_per_token(first) == 0
        assert gguf_recurrent_state_bytes(first) == 0


class TestLayersSpreadOverParts:
    def test_kv_cost_counts_attending_layers_in_every_part(self, tmp_path):
        first = _layers_over_parts_split(tmp_path)
        assert gguf_kv_bytes_per_token(first) == _expected_kv_per_token()

    def test_recurrent_state_counts_recurrent_layers_in_every_part(self, tmp_path):
        first = _layers_over_parts_split(tmp_path)
        assert gguf_recurrent_state_bytes(first) == _expected_recurrent_state()

    def test_pinned_expert_bytes_reach_into_the_later_part(self, tmp_path):
        first = _layers_over_parts_split(tmp_path)
        assert gguf_moe_pinned_expert_bytes(first, N_LAYERS) == N_LAYERS * 3 * EXPERT_BYTES

    def test_a_truncated_later_part_is_no_answer(self, tmp_path):
        first = _layers_over_parts_split(tmp_path)
        second = tmp_path / "m-00002-of-00002.gguf"
        second.write_bytes(second.read_bytes()[:40])
        assert gguf_recurrent_state_bytes(first) == 0
        assert gguf_kv_bytes_per_token(first) == 0
        assert gguf_moe_pinned_expert_bytes(first, N_LAYERS) is None


# --------------------------------------------------------------------------- #
#  The load planning that reads them                                           #
# --------------------------------------------------------------------------- #

OVERHEAD = 4 * KB


def _backend(first):
    return GgufBackend(str(first), n_ctx=256, n_gpu_layers=99,
                       vram_overhead_bytes=OVERHEAD)


class TestPlanningOnAMetadataOnlyFirstPart:
    def test_the_input_layer_is_not_charged_as_vram(self, tmp_path):
        b = _backend(_metadata_only_split(tmp_path))
        assert b._vram_model_bytes(0) == b._model_bytes() - sum(INPUT_BYTES.values())

    def test_the_layer_count_comes_from_the_header(self, tmp_path):
        b = _backend(_metadata_only_split(tmp_path))
        with patch("localm.model_meta.cached_n_layers", return_value=None):
            assert b._cached_layer_count() == N_LAYERS

    def test_one_gpu_keeps_experts_in_ram_before_moving_layers(self, tmp_path):
        """Free VRAM holds every non-expert weight and the experts of two of the
        four blocks: auto sizing keeps every layer on the GPU and pins the
        experts of the first two blocks."""
        b = _backend(_metadata_only_split(tmp_path))
        resident = b._model_bytes() - sum(INPUT_BYTES.values())
        free = resident - 2 * 3 * EXPERT_BYTES + 64 * KB
        with patch.object(GgufBackend, "_split_free_total_bytes",
                          return_value=(None, None, 0)), \
                patch.object(GgufBackend, "_free_vram_bytes", return_value=free), \
                patch.object(GgufBackend, "_total_vram_bytes", return_value=free + MB):
            budget = b._auto_gpu_layers_budget()
        assert budget.layers == 99
        assert budget.moe_cpu_layers == 2

    def test_an_implicit_split_gets_a_per_device_plan(self, tmp_path):
        """Two GPUs under llama.cpp's implicit split: the per-device fit is
        made from the header (it needs every tensor's size)."""
        b = _backend(_metadata_only_split(tmp_path))
        devices = [{"index": i, "name": f"GPU{i}", "type": _loader.GGML_DEV_TYPE_GPU,
                    "free": 8 * MB, "total": 16 * MB} for i in range(2)]
        with patch.object(discover, "_native_gpu_index_space_is_opaque",
                          return_value=True), \
                patch.object(discover, "native_gpu_devices", return_value=devices), \
                patch.object(_loader, "native_lib_loaded", return_value=False):
            plan = b._implicit_split_fit(99, 0)
        assert plan is not None


# --------------------------------------------------------------------------- #
#  Packing layers of unequal size onto the devices                             #
# --------------------------------------------------------------------------- #

class TestPackedSplit:
    """Blocks whose routed experts stay in system RAM are light, the rest heavy,
    and the heavy ones come last. Weighting devices by free memory hands the
    heavy tail to the last devices; packing the layers in order fits them."""

    GiB = 1024 ** 3

    def _plan(self, frees_gib):
        from localm.inference.backends.llamacpp._split_fit import plan_split
        devices = [{"index": i, "free": int(f * self.GiB)}
                   for i, f in enumerate(frees_gib)]
        return plan_split(devices, layer_bytes=[50 * MB] * 10 + [int(2.5 * self.GiB)] * 6,
                          layer_kv_bytes=[0] * 16, output_bytes=300 * MB,
                          n_gpu_layers=99, logits_bytes=200 * MB,
                          reserve_bytes=int(0.5 * self.GiB))

    def test_a_light_prefix_and_heavy_tail_is_packed_in_order(self):
        plan = self._plan([10.0, 10.0, 2.0])
        assert not plan.default_fits
        assert sorted(plan.tensor_split) == [0, 1]
        assert plan.excluded == [2]
        assert [c.layers for c in plan.chosen] == [13, 3, 0]
        assert plan.chosen[1].holds_output
        assert all(c.fits for c in plan.chosen)

    def test_no_plan_when_the_layers_do_not_fit_even_packed(self):
        plan = self._plan([8.0, 8.0, 2.0])
        assert plan.tensor_split is None


# --------------------------------------------------------------------------- #
#  The load-failure VRAM hint                                                  #
# --------------------------------------------------------------------------- #

class TestLoadFailureVramHint:
    def _failed_load_message(self, tmp_path, *, free, layers=99):
        first = _metadata_only_split(tmp_path)
        b = GgufBackend(str(first), n_ctx=256, n_gpu_layers=layers,
                        vram_overhead_bytes=OVERHEAD)

        def _effective(self_b):
            self_b.effective_n_cpu_moe = 0
            self_b.last_gpu_sizing = {"mode": "auto", "layers": layers,
                                      "free_bytes": free}
            return layers

        with patch.object(GgufBackend, "_effective_gpu_layers", _effective), \
                patch.object(GgufBackend, "_check_vram", lambda self_b: None), \
                patch.object(GgufBackend, "_requested_parallel_slots", lambda self_b: 1), \
                patch.object(GgufBackend, "_split_free_total_bytes",
                             return_value=(10 * free, 10 * free, 2)), \
                patch.object(GgufBackend, "_load_native",
                             side_effect=RuntimeError("worker died")):
            try:
                b.load()
            except RuntimeError as exc:
                return str(exc), b
        raise AssertionError("the load did not fail")

    def test_no_low_memory_claim_when_the_planned_load_fit(self, tmp_path):
        """Free VRAM below the whole file but above what the load planned
        (the input layer stays in system RAM): no claim that VRAM ran short."""
        first = _metadata_only_split(tmp_path)
        b = _backend(first)
        free = b._vram_model_bytes(0) + 128 * KB
        assert free < b._model_bytes()
        msg, _b = self._failed_load_message(tmp_path, free=free)
        assert "worker died" in msg
        assert "low on memory" not in msg
        assert "needed about" not in msg
        assert "-g 24" not in msg

    def test_a_short_plan_names_both_figures_and_a_fitting_layer_count(self, tmp_path):
        msg, b = self._failed_load_message(tmp_path, free=600 * KB)
        assert "needed about" in msg and "GB free" in msg
        assert "the 2 GPUs" in msg
        assert "-g 24" not in msg
        expected = b._gpu_layers_fitting(600 * KB, b.n_ctx * b._kv_bytes_per_token(),
                                         b._split_overhead_bytes(2)
                                         + b._recurrent_state_vram_bytes())
        assert 0 < expected < N_LAYERS
        assert f"-g {expected})" in msg


# --------------------------------------------------------------------------- #
#  A native crash in privacy mode                                              #
# --------------------------------------------------------------------------- #

class _FakeProcess:
    exitcode = -11

    def __init__(self, *a, **kw):
        pass

    def start(self):
        pass

    def is_alive(self):
        return False


class _FakeContext:
    Process = _FakeProcess

    @staticmethod
    def Queue():
        import queue
        return queue.Queue()


class TestNoTraceInPrivacyMode:
    def _runner_death(self, allowed):
        from localm.inference.backends.llamacpp import _runner
        r = _runner.ModelRunner()
        with patch.object(_runner.mp, "get_context", return_value=_FakeContext()), \
                patch("localm.audit.diagnostics_allowed", return_value=allowed), \
                patch("localm._mp_spawn.ensure_spawn_uses_venv_python"):
            r._spawn()
        return r._death_report()[1]

    def test_privacy_mode_says_why_there_is_no_trace(self):
        detail = self._runner_death(allowed=False)
        assert "privacy mode keeps none" in detail
        assert "keep_diagnostics" in detail

    def test_without_privacy_mode_the_plain_sentence_stays(self, tmp_path):
        with patch("localm.debuglog.logs_dir", return_value=tmp_path):
            detail = self._runner_death(allowed=True)
        assert detail == " No native fault trace was captured for this exit."
