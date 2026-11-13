# SPDX-License-Identifier: AGPL-3.0-or-later
"""GgufBackend.predict_placement says what a load would do (layers on the GPU,
routed experts in RAM, weights read from disk) with the same numbers the load
itself computes, and changes nothing a later load reads.

Real GGUF files built byte by byte; only the VRAM and RAM readings are
patched."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from localm.inference.backends.gguf import GgufBackend
from localm.inference.backends.llamacpp._sizing import HOST_RAM_HEADROOM_BYTES
from tests.test_moe_auto_placement import (KV, OVERHEAD, _backend, _dense_model,
                                           _free_fitting, _moe_model, _Vram)

PLENTY_RAM = 10 ** 12


def _real_load_outcome(b, free, ram_available):
    """What a load computes for *free* VRAM and *ram_available* RAM: the layer
    count and n_cpu_moe of ``_effective_gpu_layers`` and the reason of
    ``_resolve_use_mmap``."""
    with _Vram(free), \
            patch("localm.sysstats.system_ram", return_value=(PLENTY_RAM, ram_available)), \
            patch.object(GgufBackend, "_layers_reach_a_gpu", return_value=True):
        layers = b._effective_gpu_layers()
        decision = b._resolve_use_mmap(layers, [object()])
    return layers, b._load_n_cpu_moe(), decision


def _predicted(b, free, ram_available):
    with _Vram(free), patch.object(GgufBackend, "_layers_reach_a_gpu", return_value=True):
        return b.predict_placement(free, 10 ** 12, ram_available, PLENTY_RAM)


@pytest.mark.parametrize("which", ["moe", "dense"])
@pytest.mark.parametrize("free_kind", ["whole", "experts_2", "experts_4", "partial", "none"])
@pytest.mark.parametrize("ram", [PLENTY_RAM, 0])
def test_prediction_matches_what_the_load_computes(tmp_path, which, free_kind, ram):
    path = _moe_model(tmp_path) if which == "moe" else _dense_model(tmp_path)
    probe = _backend(path)
    whole = probe._vram_model_bytes(0) + KV + OVERHEAD
    free = {"whole": whole,
            "experts_2": _free_fitting(probe, 2) if which == "moe" else whole - 1,
            "experts_4": _free_fitting(probe, 4) if which == "moe" else whole // 2,
            "partial": KV + OVERHEAD + probe._vram_model_bytes(0) // 12,
            "none": KV + OVERHEAD - 1}[free_kind]
    predicted = _predicted(_backend(path), free, ram)
    layers, n_cpu_moe, decision = _real_load_outcome(_backend(path), free, ram)
    assert predicted is not None
    assert (predicted["layers"], predicted["n_cpu_moe"], predicted["mmap"]) == (
        layers, n_cpu_moe, decision.reason)
    assert predicted["host_bytes"] == decision.host_bytes
    assert predicted["cpu_read_bytes"] == decision.cpu_read_bytes


def test_prediction_changes_nothing_a_load_reads(tmp_path):
    b = _backend(_moe_model(tmp_path))
    before = {k: v for k, v in vars(b).items() if not k.startswith("_")}
    _predicted(b, _free_fitting(b, 3), PLENTY_RAM)
    after = {k: v for k, v in vars(b).items() if not k.startswith("_")}
    assert after == before


def test_experts_in_ram_that_do_not_fit_ram_read_from_disk(tmp_path):
    b = _backend(_moe_model(tmp_path))
    fits = _predicted(b, _free_fitting(b, 4), PLENTY_RAM)
    assert fits["full"] and fits["n_cpu_moe"] == 4 and not fits["reads_from_disk"]
    short = fits["cpu_read_bytes"] + HOST_RAM_HEADROOM_BYTES - 1
    tight = _predicted(b, _free_fitting(b, 4), short)
    assert tight["reads_from_disk"] is True
    exact = _predicted(b, _free_fitting(b, 4), short + 1)
    assert exact["reads_from_disk"] is False


def test_a_full_dense_offload_never_reads_from_disk(tmp_path):
    b = _backend(_dense_model(tmp_path))
    whole = b._vram_model_bytes(0) + KV + OVERHEAD
    p = _predicted(b, whole, 0)
    assert p["full"] and p["n_cpu_moe"] == 0 and p["reads_from_disk"] is False


def test_a_partial_offload_short_of_ram_reads_from_disk(tmp_path):
    b = _backend(_dense_model(tmp_path))
    p = _predicted(b, KV + OVERHEAD + b._vram_model_bytes(0) // 12, 0)
    assert 0 < p["layers"] <= 4 and not p["full"]
    assert p["reads_from_disk"] is True


def test_a_load_over_two_gpus_is_not_predicted(tmp_path):
    b = _backend(_dense_model(tmp_path))
    with patch.object(GgufBackend, "_split_free_total_bytes",
                      return_value=(10 ** 10, 2 * 10 ** 10, 2)):
        assert b.predict_placement() is None


def test_a_configured_layer_count_is_predicted_as_configured(tmp_path):
    b = _backend(_dense_model(tmp_path), n_gpu_layers=2, n_gpu_layers_auto=False)
    p = _predicted(b, 10 ** 12, PLENTY_RAM)
    assert p["layers"] == 2 and not p["full"]
