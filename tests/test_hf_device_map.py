# SPDX-License-Identifier: AGPL-3.0-or-later
"""_cuda_device_map: the HF (transformers) backend's device_map/max_memory builder,
honouring gpu_split_indices / main_gpu_index the same way the GGUF backend's native
params do (see localm.discover.apply_gpu_split / apply_main_gpu) - closing the gap
where this backend used to hardcode device_map="auto" regardless of either setting.

resolve_gpu_split / resolve_main_gpu_index are the REAL functions here (not mocked -
that would just be re-testing this module's own logic against a stub); only
localm.discover.list_gpus (their shared device-detection dependency) and
torch.cuda.mem_get_info are faked, mirroring the style in tests/test_hf_device_xpu.py
and tests/test_discover.py."""

from unittest.mock import MagicMock

from localm.inference.backends.hf import _cuda_device_map

_HEADROOM = int(0.5e9)   # must match the headroom baked into _cuda_device_map


def _fake_torch(mem_by_device=None):
    """A stand-in for the torch module: torch.cuda.mem_get_info(idx) returns
    mem_by_device[idx] (an (free, total) tuple), or raises when the mapped value
    is an Exception instance. A missing idx raises KeyError, same as a real bug
    in the test (every device _cuda_device_map queries must be accounted for)."""
    fake = MagicMock()

    def _mem_get_info(idx):
        val = (mem_by_device or {})[idx]
        if isinstance(val, Exception):
            raise val
        return val

    fake.cuda.mem_get_info.side_effect = _mem_get_info
    return fake


class TestNoSplitNoMainGpu:
    """Neither gpu_split_indices nor main_gpu_index configured -> today's
    unchanged default, device_map "auto" across every visible device."""

    def test_defaults_to_auto_with_config_kwarg(self, monkeypatch):
        # resolve_gpu_split short-circuits on an empty gpu_split_indices before
        # ever calling list_gpus(), and main_gpu_index is unset so
        # resolve_main_gpu_index is never called either - list_gpus() must not
        # be consulted at all.
        calls = []
        monkeypatch.setattr("localm.discover.list_gpus", lambda: calls.append(1) or [])
        result = _cuda_device_map(
            _fake_torch(), config={"gpu_split_indices": None, "main_gpu_index": None})
        assert result == {"device_map": "auto"}
        assert calls == []

    def test_defaults_to_auto_via_load_config(self, monkeypatch):
        # config=None -> _cuda_device_map reads localm.config.load_config() itself.
        monkeypatch.setattr(
            "localm.config.load_config",
            lambda: {"gpu_split_indices": None, "main_gpu_index": None})
        monkeypatch.setattr("localm.discover.list_gpus", lambda: [])
        assert _cuda_device_map(_fake_torch()) == {"device_map": "auto"}


class TestMainGpuIndexOnly:
    """main_gpu_index configured, no split -> pin the whole model to that one
    resolved device."""

    def test_pins_whole_model_to_resolved_device(self, monkeypatch):
        monkeypatch.setattr(
            "localm.discover.list_gpus", lambda: [{"index": 0}, {"index": 1}])
        result = _cuda_device_map(_fake_torch(), config={"main_gpu_index": 1})
        assert result == {"device_map": {"": 1}}

    def test_invalid_main_gpu_index_falls_back_to_resolved_zero(self, monkeypatch, caplog):
        # resolve_main_gpu_index is the REAL function: index 7 isn't among the
        # detected devices, so it drops to 0 with a warning - _cuda_device_map
        # must pin to whatever resolve_main_gpu_index actually RETURNS, not the
        # raw configured value.
        monkeypatch.setattr("localm.discover.list_gpus", lambda: [{"index": 0}])
        with caplog.at_level("WARNING", logger="localm"):
            result = _cuda_device_map(_fake_torch(), config={"main_gpu_index": 7})
        assert result == {"device_map": {"": 0}}
        assert any("main_gpu_index" in r.message for r in caplog.records)


class TestGpuSplitSuccess:
    """gpu_split_indices configured with 2+ valid devices and mem_get_info
    succeeding for each -> "auto" plus a max_memory dict with a headroom-adjusted
    entry per device."""

    def test_two_devices_produce_max_memory_with_headroom(self, monkeypatch):
        monkeypatch.setattr(
            "localm.discover.list_gpus", lambda: [{"index": 0}, {"index": 1}])
        mem = {0: (8_000_000_000, 16_000_000_000), 1: (4_000_000_000, 8_000_000_000)}
        result = _cuda_device_map(
            _fake_torch(mem), config={"gpu_split_indices": [0, 1]})

        assert result["device_map"] == "auto"
        assert set(result["max_memory"]) == {0, 1}
        for idx, (free, _total) in mem.items():
            got = result["max_memory"][idx]
            assert isinstance(got, int)
            assert got == free - _HEADROOM
            assert got < free   # strictly less than the raw free bytes (headroom)

    def test_three_devices_all_readable(self, monkeypatch):
        monkeypatch.setattr(
            "localm.discover.list_gpus",
            lambda: [{"index": 0}, {"index": 1}, {"index": 2}])
        mem = {0: (8_000_000_000, 16_000_000_000),
               1: (4_000_000_000, 8_000_000_000),
               2: (2_000_000_000, 4_000_000_000)}
        result = _cuda_device_map(
            _fake_torch(mem), config={"gpu_split_indices": [0, 1, 2]})
        assert result == {
            "device_map": "auto",
            "max_memory": {0: 7_500_000_000, 1: 3_500_000_000, 2: 1_500_000_000},
        }


class TestGpuSplitMemGetInfoFailures:
    """torch.cuda.mem_get_info raising for enough devices that fewer than 2 end
    up usable -> falls back to whichever of the device_map branches applies,
    with a warning logged."""

    def test_falls_back_to_auto_when_only_one_device_readable(self, monkeypatch, caplog):
        monkeypatch.setattr(
            "localm.discover.list_gpus", lambda: [{"index": 0}, {"index": 1}])
        mem = {0: (8_000_000_000, 16_000_000_000), 1: RuntimeError("no driver")}
        with caplog.at_level("WARNING", logger="localm"):
            result = _cuda_device_map(
                _fake_torch(mem), config={"gpu_split_indices": [0, 1]})
        assert result == {"device_map": "auto"}
        assert any("gpu_split_indices" in r.message for r in caplog.records)

    def test_falls_back_to_main_gpu_pin_when_also_configured(self, monkeypatch, caplog):
        monkeypatch.setattr(
            "localm.discover.list_gpus", lambda: [{"index": 0}, {"index": 1}])
        mem = {0: RuntimeError("no driver"), 1: RuntimeError("no driver")}
        with caplog.at_level("WARNING", logger="localm"):
            result = _cuda_device_map(
                _fake_torch(mem),
                config={"gpu_split_indices": [0, 1], "main_gpu_index": 1})
        assert result == {"device_map": {"": 1}}
        assert any("gpu_split_indices" in r.message for r in caplog.records)


class TestGpuSplitDropsUnknownDevice:
    """An index resolve_gpu_split() itself would drop (unmatched by list_gpus())
    - confirms _cuda_device_map composes with the REAL resolve_gpu_split rather
    than trusting the raw config."""

    def test_unknown_device_dropped_leaves_single_gpu_default(self, monkeypatch, caplog):
        # list_gpus() only reports device 0; resolve_gpu_split drops the unknown
        # index 99, leaving a single valid device (< 2) -> no split at all.
        monkeypatch.setattr("localm.discover.list_gpus", lambda: [{"index": 0}])
        with caplog.at_level("WARNING", logger="localm"):
            result = _cuda_device_map(
                _fake_torch(), config={"gpu_split_indices": [0, 99]})
        assert result == {"device_map": "auto"}
        assert any("gpu_split_indices" in r.message for r in caplog.records)

    def test_unknown_device_dropped_then_main_gpu_pin_applies(self, monkeypatch, caplog):
        monkeypatch.setattr("localm.discover.list_gpus", lambda: [{"index": 0}])
        with caplog.at_level("WARNING", logger="localm"):
            result = _cuda_device_map(
                _fake_torch(),
                config={"gpu_split_indices": [0, 99], "main_gpu_index": 0})
        assert result == {"device_map": {"": 0}}
        assert any("gpu_split_indices" in r.message for r in caplog.records)
