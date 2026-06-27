# SPDX-License-Identifier: AGPL-3.0-or-later
"""The torch wheel SOURCE must be correct for EVERY hardware+OS combination, not
just the maintainer's gfx1030 box (the "works on any hardware" rule). On Windows,
AMD has no single ROCm wheel: gfx103X (RX 6000 / RDNA2) uses localm's bundled
self-contained build, while RX 7000/9000 (RDNA3/RDNA4) use AMD's official Windows
ROCm wheels (public preview). ``hwdetect.torch_pip_args`` is the single tested
source of truth both installers consult via ``hwdetect torch-args <backend>``.

These pin the ROUTING. Executing the RX 7000/9000 Windows wheels is pending real
RDNA3/RDNA4 hardware (the dev box is gfx1030); the gfx103X path is verified live.
"""

from __future__ import annotations

import pytest

from localm import hwdetect


def _det(names, vendors=("amd",)):
    return hwdetect.Detection(vendors=list(vendors), gpu_names=names)


# ------------------------------ amd_gfx_family ---------------------------- #

@pytest.mark.parametrize("name,expected", [
    ("amd radeon rx 6900 xt", "gfx103x"),   # RDNA2
    ("amd radeon rx 6600", "gfx103x"),
    ("amd radeon rx 7900 xtx", "gfx110x"),  # RDNA3
    ("amd radeon rx 9070 xt", "gfx120x"),   # RDNA4
    ("amd radeon rx 5700 xt", ""),          # RDNA1 - no current Windows ROCm wheel
    ("amd radeon vii", ""),
    ("amd instinct mi100", ""),
    ("", ""),
])
def test_amd_gfx_family(name, expected):
    assert hwdetect.amd_gfx_family(name) == expected


# ------------------------- torch_pip_args: vendors ------------------------ #

def test_cuda_uses_cu126_any_os():
    args = hwdetect.torch_pip_args("cuda", _det("nvidia rtx 4090", ("nvidia",)))
    assert args == "torch torchvision --index-url https://download.pytorch.org/whl/cu126"


def test_xpu_for_intel_sycl():
    args = hwdetect.torch_pip_args("sycl", _det("intel arc a770", ("intel",)))
    assert args == "torch torchvision --index-url https://download.pytorch.org/whl/xpu"


def test_cpu_and_neutral_picks_install_nothing():
    # cpu pick, and a vendor-neutral vulkan pick on an AMD-only box: no torch.
    assert hwdetect.torch_pip_args("cpu", _det("amd radeon rx 6900 xt")) == ""
    assert hwdetect.torch_pip_args("vulkan", _det("amd radeon rx 6900 xt")) == ""


# ---------------- torch_pip_args: AMD rocm, per OS + gfx ------------------ #

def test_amd_rocm_on_linux_uses_upstream_index(monkeypatch):
    monkeypatch.setattr(hwdetect.sys, "platform", "linux")
    args = hwdetect.torch_pip_args("hip", _det("amd radeon rx 7900 xtx"))
    # Linux uses upstream wheels (broad gfx) regardless of the exact card.
    assert args == "torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2"


def test_amd_rocm_on_windows_gfx103x_uses_bundled_extra(monkeypatch):
    monkeypatch.setattr(hwdetect.sys, "platform", "win32")
    args = hwdetect.torch_pip_args("amd-rocm", _det("amd radeon rx 6900 xt"))
    assert args == "-e .[gpu]"


@pytest.mark.parametrize("name", ["amd radeon rx 7900 xtx", "amd radeon rx 9070 xt"])
def test_amd_rocm_on_windows_rdna3_4_uses_official_preview(monkeypatch, name):
    monkeypatch.setattr(hwdetect.sys, "platform", "win32")
    args = hwdetect.torch_pip_args("amd-rocm", _det(name))
    # RX 7000 / 9000 are NOT the bundled gfx103X build - they get AMD's official
    # Windows ROCm wheels (public preview), not silently dropped or mis-pinned.
    assert args == "torch torchvision --index-url https://download.pytorch.org/whl/rocm6.4"
    assert args != "-e .[gpu]"


def test_amd_rocm_on_windows_unknown_card_skips_not_mispins(monkeypatch):
    # An RDNA1 / unknown AMD card on Windows has no verified prebuilt: skip and
    # guide, never silently install the gfx103X wheel that will not load.
    monkeypatch.setattr(hwdetect.sys, "platform", "win32")
    assert hwdetect.torch_pip_args("amd-rocm", _det("amd radeon rx 5700 xt")) == ""
