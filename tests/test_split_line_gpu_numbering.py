# SPDX-License-Identifier: AGPL-3.0-or-later
"""The GPU split the sidebar shows names GPUs by the numbers the per-GPU rows,
the GPU selector and ``localm gpus`` use (torch's index), whichever branch of
``GgufBackend._load_native`` recorded it and whatever numbering the llama.cpp
runtime gives its own devices."""

import pytest

from localm import discover, sysstats
from localm.inference.backends.gguf import GgufBackend
from tests.test_gpu_placement import TestBackendLoad
from tests.test_implicit_split_fit import (
    _config, _load_capturing, _registry, _tiny_model, _torch_readings)
from tests.test_multigpu_config_placement import _backend as _cfg_backend
from tests.test_multigpu_config_placement import _load as _cfg_load

GIB = 1024 ** 3


def _names(split, gpus):
    """The cards a recorded split names, as the rows and the selector number them."""
    by_index = {g["index"]: g["name"] for g in gpus}
    return [by_index[d["index"]] for d in split["devices"]]


def _row_indices(monkeypatch, gpus):
    """The ``index`` of each per-GPU row ``/api/stats`` builds from *gpus*."""
    monkeypatch.setattr(discover, "last_known_gpus", lambda *a, **k: list(gpus))
    monkeypatch.setattr(
        discover, "vram_capacity",
        lambda *a, **k: ({"total": sum(g["total"] for g in gpus),
                          "free": sum(g["free"] for g in gpus)}, None))
    monkeypatch.setattr(sysstats, "_vram_reading_trusted", lambda *a, **k: True)
    return [d["index"] for d in sysstats._compute_vram()["vram"]["devices"]]


class TestImplicitFitBranch:
    """The implicit per-device fit plans in llama.cpp's device slots, which are
    not torch's numbers once the runtime leaves an integrated GPU out."""

    def _backend(self, tmp_path):
        b = GgufBackend(str(_tiny_model(tmp_path, vocab=5000)), n_ctx=4096,
                        n_gpu_layers=99)
        b._VRAM_OVERHEAD_BYTES = 1000
        b._gguf_kv_bpt = 10
        return b

    @pytest.mark.parametrize("igpu_at", [0, 1, 2])
    def test_the_split_names_the_torch_numbers_of_the_cards_it_uses(
            self, tmp_path, monkeypatch, igpu_at):
        frees = [(50_000_000, False), (50_000_000, False), (30_000_000, False)]
        frees.insert(igpu_at, (2_000_000_000, True))
        gpus = [{"index": i, "name": f"gpu{i}", "free": f, "total": f + 10,
                 "free_scope": discover.FREE_SCOPE_DEVICE, "integrated": ig}
                for i, (f, ig) in enumerate(frees)]
        discrete = [g["index"] for g in gpus if not g["integrated"]]
        b = self._backend(tmp_path)
        params = _load_capturing(b, gpus, _registry(gpus))
        assert params["gpu_split_ratios"] == {0: 0.5, 1: 0.5}, (
            "the 30 MB card must be left out of the worker's split")
        assert b.applied_gpu_split["source"] == "auto"
        recorded = [d["index"] for d in b.applied_gpu_split["devices"]]
        assert recorded == discrete[:2], (
            f"the split line names {recorded}, the cards used are {discrete[:2]}")
        assert _names(b.applied_gpu_split, gpus) == [f"gpu{i}" for i in discrete[:2]]
        assert set(recorded) <= set(_row_indices(monkeypatch, gpus))

    def test_shares_are_kept_with_the_translated_numbers(self, tmp_path):
        gpus = [{"index": i, "name": f"gpu{i}", "free": f, "total": f + 10,
                 "free_scope": discover.FREE_SCOPE_DEVICE, "integrated": ig}
                for i, (f, ig) in enumerate(
                    [(2_000_000_000, True), (60_000_000, False), (40_000_000, False),
                     (30_000_000, False)])]
        b = self._backend(tmp_path)
        params = _load_capturing(b, gpus, _registry(gpus))
        by_index = {d["index"]: d["share"] for d in b.applied_gpu_split["devices"]}
        assert sorted(by_index) == [1, 2]
        assert by_index[1] == pytest.approx(params["gpu_split_ratios"][0])
        assert by_index[2] == pytest.approx(params["gpu_split_ratios"][1])


class TestPlacedBranch:
    """A load spread over 2+ GPUs (gpu_devices) records the torch index."""

    def test_a_reordered_runtime_still_names_the_selected_cards(self, tmp_path):
        from localm.config import load_config
        loader = TestBackendLoad()
        b = loader._backend(tmp_path)
        readings, registry = loader._world([30_000, 30_000, 30_000], [2, 0, 1])
        cfg = dict(load_config(), gpu_split_indices=[0, 2], gpu_split_ratios=None,
                   ctx_auto=False)
        params = loader._load(b, readings, registry, cfg)
        assert {d["slot"] for d in params["gpu_devices"]} == {0, 1}
        assert sorted(_names(b.applied_gpu_split, readings)) == ["card0", "card2"]


class TestConfiguredPairsBranch:
    """A configured split records the configured (torch) numbers."""

    def test_a_split_after_an_integrated_gpu_names_the_configured_cards(
            self, tmp_path, monkeypatch):
        gpus = _torch_readings((2.0, True), (30.0, False), (10.0, False))
        cfg = _config(gpu_split_indices=[1, 2], gpu_split_ratios=None,
                      main_gpu_index=None)
        b = _cfg_backend(tmp_path)
        _cfg_load(b, gpus, _registry(gpus), cfg)
        assert _names(b.applied_gpu_split, gpus) == ["gpu1", "gpu2"]
        assert set(d["index"] for d in b.applied_gpu_split["devices"]) <= \
            set(_row_indices(monkeypatch, gpus))
