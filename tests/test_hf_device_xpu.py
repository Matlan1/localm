# SPDX-License-Identifier: AGPL-3.0-or-later
"""HF backend device routing: an Intel Arc/Xe GPU (torch.xpu) must be used instead of
silently falling back to CPU, WITHOUT regressing the cuda/cpu paths. _auto_device takes
torch as an argument so the routing is testable without any real GPU (GPU EXECUTION still
needs a real Arc box - out of scope here)."""

from localm.inference.backends.hf import _auto_device


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


def test_explicit_override_wins():
    assert _auto_device(_FakeTorch(cuda=True, xpu=True), "cpu") == "cpu"


def test_cuda_preferred_over_xpu():
    assert _auto_device(_FakeTorch(cuda=True, xpu=True)) == "cuda"


def test_intel_xpu_used_when_no_cuda():
    # THE fix: before, an Intel box with no CUDA fell through to "cpu".
    assert _auto_device(_FakeTorch(cuda=False, xpu=True)) == "xpu"


def test_cpu_when_no_gpu():
    assert _auto_device(_FakeTorch(cuda=False, xpu=False)) == "cpu"


def test_cpu_when_torch_has_no_xpu_attr():
    # Older PyTorch without torch.xpu must not crash - falls back to cpu.
    assert _auto_device(_FakeTorch(cuda=False, xpu=None)) == "cpu"
