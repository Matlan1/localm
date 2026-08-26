# SPDX-License-Identifier: AGPL-3.0-or-later
"""HF backend device routing: an Intel Arc/Xe GPU (torch.xpu) must be used instead of
silently falling back to CPU, WITHOUT regressing the cuda/cpu paths. _auto_device takes
torch as an argument so the routing is testable without any real GPU (GPU EXECUTION still
needs a real Arc box - out of scope here)."""

import pytest

from localm.inference.backends._hf_worker import _auto_device


class _Avail:
    def __init__(self, available):
        self._available = available

    def is_available(self):
        return self._available


class _FakeTorch:
    def __init__(self, cuda=False, xpu=None):
        self.cuda = _Avail(cuda)
        if xpu is not None:               # old PyTorch has no torch.xpu at all
            self.xpu = _Avail(xpu)


@pytest.mark.parametrize(
    "cuda,xpu,override,expected",
    [
        pytest.param(True, True, "cpu", "cpu", id="explicit_override_wins"),
        pytest.param(True, True, None, "cuda", id="cuda_preferred_over_xpu"),
        # An Intel box with no CUDA selects xpu.
        pytest.param(False, True, None, "xpu", id="intel_xpu_used_when_no_cuda"),
        pytest.param(False, False, None, "cpu", id="cpu_when_no_gpu"),
        # Older PyTorch without torch.xpu must not crash - falls back to cpu.
        pytest.param(False, None, None, "cpu", id="cpu_when_torch_has_no_xpu_attr"),
    ],
)
def test_auto_device(cuda, xpu, override, expected):
    assert _auto_device(_FakeTorch(cuda=cuda, xpu=xpu), override) == expected
