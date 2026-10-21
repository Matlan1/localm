# SPDX-License-Identifier: AGPL-3.0-or-later
"""A Mixture-of-Experts load that keeps routed experts in system RAM carries the
per-token RAM read (``moe_ram_read_bytes``) in its placement dict, which the load
response spreads and the toast and placement sentence render."""

from types import SimpleNamespace

from localm.inference.engine import Engine, describe_gpu_placement


def _engine(read):
    engine = Engine.__new__(Engine)
    engine._backend = SimpleNamespace(
        gpu_layers_offloaded=48, gpu_layers_total=48, moe_cpu_layers=15,
        moe_ram_read_bytes=read)
    return engine


def test_the_figure_reaches_the_placement_dict_and_its_sentence():
    gib = 1024 ** 3
    placement = _engine(int(1.5 * gib)).gpu_placement
    assert placement["moe_ram_read_bytes"] == int(1.5 * gib)
    assert "about 1.50 GB read from RAM per token" in describe_gpu_placement(placement)


def test_a_load_without_the_figure_keeps_the_placement_unchanged():
    placement = _engine(None).gpu_placement
    assert "moe_ram_read_bytes" not in placement
    assert "per token" not in describe_gpu_placement(placement)
