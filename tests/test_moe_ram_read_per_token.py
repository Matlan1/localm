# SPDX-License-Identifier: AGPL-3.0-or-later
"""A Mixture-of-Experts load that keeps routed experts in system RAM reports how
much of that RAM each generated token reads: the pinned expert bytes scaled by the
share of experts the router selects (GGUF ``expert_used_count`` of
``expert_count``)."""

from types import SimpleNamespace

from localm.inference.backends.gguf import GgufBackend
from localm.inference.engine import Engine, describe_gpu_placement
from tests.test_gguf_moe_vram_sizing import _T_STRING, _T_UINT32, _gguf_with_tensors

_TENSORS = [
    ("blk.0.attn_q.weight", [4, 4], 0, 100),
    ("blk.0.ffn_gate_exps.weight", [4, 4, 8], 0, 500),
    ("blk.0.ffn_down_exps.weight", [4, 4, 8], 0, 300),
    ("blk.1.ffn_gate_exps.weight", [4, 4, 8], 0, 500),
    ("blk.1.ffn_down_exps.weight", [4, 4, 8], 0, 300),
    ("blk.2.ffn_gate_exps.weight", [4, 4, 8], 0, 800),
]


def _backend(tmp_path, *, experts=8, used=2, n_cpu_moe=2):
    kv = [("general.architecture", _T_STRING, "testmoe")]
    if experts is not None:
        kv.append(("testmoe.expert_count", _T_UINT32, experts))
    if used is not None:
        kv.append(("testmoe.expert_used_count", _T_UINT32, used))
    path = _gguf_with_tensors(tmp_path / "m.gguf", kv, _TENSORS)
    return GgufBackend(str(path), n_cpu_moe=n_cpu_moe)


def test_each_token_reads_the_selected_share_of_the_pinned_experts(tmp_path):
    b = _backend(tmp_path)
    assert b._moe_ram_read_bytes_per_token(2) == (500 + 300 + 500 + 300) * 2 // 8


def test_pinning_fewer_layers_reads_less(tmp_path):
    b = _backend(tmp_path)
    assert b._moe_ram_read_bytes_per_token(1) == (500 + 300) * 2 // 8


def test_no_figure_when_nothing_is_pinned_or_the_header_gives_no_counts(tmp_path):
    assert _backend(tmp_path)._moe_ram_read_bytes_per_token(0) is None
    assert _backend(tmp_path, used=None)._moe_ram_read_bytes_per_token(2) is None
    assert _backend(tmp_path, experts=None, used=None)._moe_ram_read_bytes_per_token(2) is None


def test_the_figure_reaches_the_placement_dict_and_its_sentence():
    gib = 1024 ** 3
    engine = Engine.__new__(Engine)
    engine._backend = SimpleNamespace(
        gpu_layers_offloaded=48, gpu_layers_total=48, moe_cpu_layers=15,
        moe_ram_read_bytes=int(1.5 * gib))
    placement = engine.gpu_placement
    assert placement["moe_ram_read_bytes"] == int(1.5 * gib)
    assert "about 1.50 GB read from RAM per token" in describe_gpu_placement(placement)


def test_a_load_without_the_figure_keeps_the_placement_unchanged():
    engine = Engine.__new__(Engine)
    engine._backend = SimpleNamespace(
        gpu_layers_offloaded=48, gpu_layers_total=48, moe_cpu_layers=15,
        moe_ram_read_bytes=None)
    assert "moe_ram_read_bytes" not in engine.gpu_placement
    assert "per token" not in describe_gpu_placement(engine.gpu_placement)
