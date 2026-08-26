# SPDX-License-Identifier: AGPL-3.0-or-later
"""Direct-binding coverage for the multi-GPU tensor-split capacity probe:
has_max_devices()/llama_max_devices() in
localm/inference/backends/llamacpp/_api.py, plus discover._tensor_split_capacity
(the sole caller, via apply_gpu_split) which decides between the live native
answer and the documented fallback constant.

Only load_lib() is mocked - what has_max_devices/llama_max_devices actually
call - so the real hasattr/getattr and ctypes-bind logic is exercised."""

from unittest.mock import MagicMock, patch

import pytest

from localm.inference.backends.llamacpp import _api
from localm.discover import (
    _TENSOR_SPLIT_FALLBACK_CAPACITY,
    _tensor_split_capacity,
)

_LOAD_LIB = "localm.inference.backends.llamacpp._api.load_lib"


class _NoMaxDevices:
    """Stands in for a ctypes.CDLL handle from an older build that never
    exported llama_max_devices. A plain object() (not a Mock) so hasattr()
    genuinely fails instead of a Mock auto-creating the attribute."""


class TestHasMaxDevices:
    def test_true_when_symbol_present(self):
        fake_lib = MagicMock(spec=["llama_max_devices"])
        with patch(_LOAD_LIB, return_value=fake_lib):
            assert _api.has_max_devices() is True

    def test_false_when_symbol_absent(self):
        with patch(_LOAD_LIB, return_value=_NoMaxDevices()):
            assert _api.has_max_devices() is False

    def test_false_when_spec_mock_lacks_symbol(self):
        # A Mock constrained with spec=[] behaves like the real CDLL handle for
        # attributes outside the spec: hasattr/getattr genuinely raise rather
        # than auto-vivifying.
        fake_lib = MagicMock(spec=[])
        with patch(_LOAD_LIB, return_value=fake_lib):
            assert _api.has_max_devices() is False


class TestLlamaMaxDevices:
    def test_calls_through_and_returns_native_int(self):
        fake_fn = MagicMock(return_value=32)
        fake_lib = MagicMock(spec=["llama_max_devices"])
        fake_lib.llama_max_devices = fake_fn
        with patch(_LOAD_LIB, return_value=fake_lib):
            result = _api.llama_max_devices()
        assert result == 32
        assert isinstance(result, int)
        fake_fn.assert_called_once_with()

    def test_return_value_is_coerced_to_python_int(self):
        # The native call returns a ctypes c_size_t-flavoured value; confirm
        # llama_max_devices() hands back a plain Python int for callers such as
        # _tensor_split_capacity's max() and comparisons.
        fake_lib = MagicMock(spec=["llama_max_devices"])
        fake_lib.llama_max_devices = MagicMock(return_value=16)
        with patch(_LOAD_LIB, return_value=fake_lib):
            result = _api.llama_max_devices()
        assert type(result) is int


_HAS_MAX_DEVICES = "localm.inference.backends.llamacpp._api.has_max_devices"
_LLAMA_MAX_DEVICES = "localm.inference.backends.llamacpp._api.llama_max_devices"


class TestTensorSplitCapacity:
    """discover._tensor_split_capacity: the sole call site is
    apply_gpu_split, deciding how many float slots to allocate for
    mp.tensor_split. It imports localm.inference.backends.llamacpp._api
    lazily inside the function, so the probes are monkeypatched on the _api
    module directly (patching localm.discover.<name> would miss it, since
    that name is never bound in discover's own module namespace)."""

    def test_false_falls_back_to_documented_constant(self, monkeypatch):
        monkeypatch.setattr(_HAS_MAX_DEVICES, lambda: False)
        assert _tensor_split_capacity(min_len=0) == _TENSOR_SPLIT_FALLBACK_CAPACITY

    def test_raising_probe_falls_back_to_documented_constant(self, monkeypatch):
        def _boom():
            raise RuntimeError("native probe exploded")
        monkeypatch.setattr(_HAS_MAX_DEVICES, _boom)
        assert _tensor_split_capacity(min_len=0) == _TENSOR_SPLIT_FALLBACK_CAPACITY

    def test_fallback_still_bounded_below_by_min_len(self, monkeypatch):
        monkeypatch.setattr(_HAS_MAX_DEVICES, lambda: False)
        huge = _TENSOR_SPLIT_FALLBACK_CAPACITY + 5
        assert _tensor_split_capacity(min_len=huge) == huge

    def test_true_uses_live_native_value(self, monkeypatch):
        monkeypatch.setattr(_HAS_MAX_DEVICES, lambda: True)
        monkeypatch.setattr(_LLAMA_MAX_DEVICES, lambda: 4)
        assert _tensor_split_capacity(min_len=0) == 4

    def test_true_bounded_below_by_min_len(self, monkeypatch):
        # More configured device indices than the native build's own capacity:
        # min_len wins the max().
        monkeypatch.setattr(_HAS_MAX_DEVICES, lambda: True)
        monkeypatch.setattr(_LLAMA_MAX_DEVICES, lambda: 4)
        assert _tensor_split_capacity(min_len=8) == 8

    def test_true_native_value_wins_when_larger_than_min_len(self, monkeypatch):
        monkeypatch.setattr(_HAS_MAX_DEVICES, lambda: True)
        monkeypatch.setattr(_LLAMA_MAX_DEVICES, lambda: 16)
        assert _tensor_split_capacity(min_len=2) == 16


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
