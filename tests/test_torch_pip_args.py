# SPDX-License-Identifier: AGPL-3.0-or-later
"""The torch wheel SOURCE must be correct for EVERY hardware+OS combination, not
just one box. On Windows, AMD has no single ROCm wheel: gfx103X (RX 6000 /
RDNA2) uses localm's bundled self-contained build, while RX 7000/9000
(RDNA3/RDNA4) use AMD's Windows ROCm wheels from their family's index.
``hwdetect.torch_pip_args`` is the single tested source of truth both installers
consult via ``hwdetect torch-args <backend>``.

These pin the ROUTING, and test_amd_rocm_win_index_resolves_live checks
the RX 7000/9000 wheels actually RESOLVE against AMD's live indexes (a package-index
question, no GPU needed for that). Executing them still needs real RDNA3/RDNA4
hardware.
"""

from __future__ import annotations

import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from localm import hwdetect

_GPU_SETUP_DOC = Path(__file__).resolve().parents[1] / "docs" / "gpu-setup.md"


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

def test_cuda_uses_cu126_any_os(monkeypatch):
    # Deterministic regardless of the machine running this test: without mocking,
    # this would pass or fail depending on whether the box has an NVIDIA GPU.
    monkeypatch.setattr(hwdetect, "_cuda_compute_capabilities", lambda: [])
    args = hwdetect.torch_pip_args("cuda", _det("nvidia rtx 4090", ("nvidia",)))
    assert args.replace(f"{hwdetect._FLA_CORE_LINUX_CUDA} ", "") == "torch torchvision --torch-backend=cu126"


def test_cuda_uses_blackwell_line_when_detected(monkeypatch):
    """pytorch_index_url("cuda") must not be a flat cu126 regardless of
    hardware: a real Blackwell GPU then gets a wheel with no kernels for it, so
    torch loads but warns every device is unsupported and runs CPU-only."""
    monkeypatch.setattr(hwdetect, "_cuda_compute_capabilities", lambda: [(12, 0)])
    args = hwdetect.torch_pip_args("cuda", _det("nvidia rtx pro 4000 blackwell", ("nvidia",)))
    assert args.replace(f"{hwdetect._FLA_CORE_LINUX_CUDA} ", "") == "torch torchvision --torch-backend=cu130"


def test_cuda_uses_blackwell_line_if_any_of_several_gpus_is_blackwell(monkeypatch):
    """A mixed box (an older card alongside a Blackwell one) must still pick
    the line the newest card needs - the older card stays covered by the
    broader, newer wheel's SM target list, whereas a Blackwell card on cu126
    does not run."""
    monkeypatch.setattr(hwdetect, "_cuda_compute_capabilities",
                        lambda: [(8, 9), (12, 0)])
    assert hwdetect.pytorch_index_url("cuda") == "https://download.pytorch.org/whl/cu130"


def test_cuda_datacenter_blackwell_also_detected(monkeypatch):
    """Data-center Blackwell (B100/B200) is compute capability 10.x, distinct
    from consumer/workstation Blackwell's 12.x - _CUDA_BLACKWELL_MIN_CAP is
    the lower bound specifically so both are covered by one threshold."""
    monkeypatch.setattr(hwdetect, "_cuda_compute_capabilities", lambda: [(10, 0)])
    assert hwdetect.pytorch_index_url("cuda") == "https://download.pytorch.org/whl/cu130"


def test_cuda_pre_blackwell_stays_on_cu126(monkeypatch):
    monkeypatch.setattr(hwdetect, "_cuda_compute_capabilities", lambda: [(9, 0)])
    assert hwdetect.pytorch_index_url("cuda") == "https://download.pytorch.org/whl/cu126"


def test_cuda_compute_capabilities_probe_failure_is_non_fatal(monkeypatch):
    """nvidia-smi missing/unparseable must fall back to cu126, never raise -
    same fail-open posture as the rest of this module."""
    monkeypatch.setattr(hwdetect, "_run", lambda cmd: "")
    assert hwdetect._cuda_compute_capabilities() == []
    assert hwdetect.pytorch_index_url("cuda") == "https://download.pytorch.org/whl/cu126"


def test_cuda_compute_capabilities_parses_multi_gpu_output(monkeypatch):
    monkeypatch.setattr(hwdetect, "_run", lambda cmd: "8.9\n12.0\n")
    assert hwdetect._cuda_compute_capabilities() == [(8, 9), (12, 0)]


def test_xpu_for_intel_sycl():
    args = hwdetect.torch_pip_args("sycl", _det("intel arc a770", ("intel",)))
    assert args == "torch torchvision --torch-backend=xpu"


def test_torch_backend_avoids_the_setuptools_conflict_index_url_hits():
    """--index-url replaces the resolver's package source for the WHOLE
    install, including ordinary transitive dependencies - every PyTorch wheel
    index (cuda, rocm, xpu, even cpu) caps setuptools at 78.1.0, which
    conflicts with this project's own >=83.0.0 override (pyproject.toml's
    [tool.uv] override-dependencies) the moment `uv pip install` runs from the
    repo root, as setup.sh/setup.bat always do - confirmed live with `uv pip
    install --dry-run`, both the conflict on --index-url and the clean
    resolve on --torch-backend. --torch-backend scopes the substitution to
    the PyTorch-family packages only, leaving setuptools on the normal PyPI
    index. This is the static lock-in that the emitted args never regress."""
    for backend, det in [
        ("cuda", _det("nvidia rtx 4090", ("nvidia",))),
        ("sycl", _det("intel arc a770", ("intel",))),
    ]:
        args = hwdetect.torch_pip_args(backend, det)
        assert "--torch-backend=" in args, f"{backend}: {args!r}"
        assert "--index-url" not in args, f"{backend}: {args!r}"


def test_cpu_and_neutral_picks_install_nothing():
    # cpu pick, and a vendor-neutral vulkan pick on an AMD-only box: no torch.
    assert hwdetect.torch_pip_args("cpu", _det("amd radeon rx 6900 xt")) == ""
    assert hwdetect.torch_pip_args("vulkan", _det("amd radeon rx 6900 xt")) == ""


# ---------------- torch_pip_args: AMD rocm, per OS + gfx ------------------ #

def test_amd_rocm_on_linux_uses_upstream_index(monkeypatch):
    monkeypatch.setattr(hwdetect.sys, "platform", "linux")
    args = hwdetect.torch_pip_args("hip", _det("amd radeon rx 7900 xtx"))
    # Linux uses upstream wheels (broad gfx) regardless of the exact card.
    assert args == "torch torchvision --torch-backend=rocm6.2"


def test_amd_rocm_on_windows_gfx103x_uses_bundled_extra(monkeypatch):
    monkeypatch.setattr(hwdetect.sys, "platform", "win32")
    args = hwdetect.torch_pip_args("amd-rocm", _det("amd radeon rx 6900 xt"))
    assert args == "-e .[gpu]"


@pytest.mark.parametrize("name, family", [("amd radeon rx 7900 xtx", "gfx110x"),
                                          ("amd radeon rx 9070 xt", "gfx120x")])
def test_amd_rocm_on_windows_rdna3_4_uses_its_family_index(monkeypatch, name, family):
    """RX 7000 / 9000 are NOT the bundled gfx103X build: they get the pinned torch
    and torchvision with their family's rocm-sdk packages from that family's AMD
    index, as an extra index next to PyPI."""
    monkeypatch.setattr(hwdetect.sys, "platform", "win32")
    args = hwdetect.torch_pip_args("amd-rocm", _det(name))
    torch, torchvision, _ = hwdetect.amd_rocm_win_torch_packages()
    sdk = " ".join(hwdetect.amd_rocm_win_sdk_packages(family))
    assert args == (f"{torch} {torchvision} {sdk} --extra-index-url "
                    f"https://repo.amd.com/rocm/whl/{family[:5]}{family[5:].upper()}-all/ "
                    "--index-strategy=unsafe-best-match")
    assert "--find-links" not in args and "--torch-backend" not in args


@pytest.mark.parametrize("name", ["amd radeon rx 7900 xtx", "amd radeon rx 9070 xt"])
def test_the_updater_reads_only_packages_out_of_the_amd_torch_args(monkeypatch, name):
    from localm import updater_components
    monkeypatch.setattr(hwdetect.sys, "platform", "win32")
    args = shlex.split(hwdetect.torch_pip_args("amd-rocm", _det(name)), posix=True)
    reqs = updater_components._requirements_in(args)
    assert len(reqs) == 5, reqs
    for req in reqs:
        assert re.fullmatch(r"(torch|torchvision|rocm|rocm-sdk-core|rocm-sdk-libraries-gfx1\d\dx-all)"
                            r"==[\w.+]+", req), req


def test_gpu_setup_doc_amd_windows_command_matches_the_live_pin():
    """docs/gpu-setup.md documents the exact manual install command for RX
    7000/9000 on Windows - it must stay byte-for-byte in sync with hwdetect's
    pin, or a reader copies a command this project no longer resolves the same
    way (the failure mode that motivated this test: the doc kept advertising
    --torch-backend=rocm6.4 after that stopped resolving on Windows)."""
    text = _GPU_SETUP_DOC.read_text(encoding="utf-8")
    torch, torchvision, _ = hwdetect.amd_rocm_win_torch_packages()
    assert torch in text
    assert torchvision in text
    for family in ("gfx110x", "gfx120x"):
        assert hwdetect.amd_rocm_win_index(family) in text
    assert "repo.radeon.com" not in text
    # The doc must not still show the dead command shape for this hardware.
    assert "torch torchvision --torch-backend=rocm6.4" not in text


@pytest.mark.integration
@pytest.mark.parametrize("family", ["gfx110x", "gfx120x"])
def test_amd_rocm_win_index_resolves_live(tmp_path, family):
    """Not mocked: a real uv resolve (metadata only - uv's --dry-run never
    downloads a wheel, unlike pip's) against AMD's live Windows ROCm preview
    repo, cross-targeting win_amd64/cp312 via --python-platform so this runs
    identically on any host OS. Resolves against a DISPOSABLE venv under
    tmp_path (never the suite's own interpreter - tests/conftest.py's installer
    guard blocks that); --python-platform/--python-version override the actual
    resolve target regardless of that venv's own version. Also the
    fires-control for the routing tests above: a genuinely dead URL (or a wrong
    package name/version) must make THIS half fail, which the macOS-target
    assertion below proves by making the same packages fail to resolve for a
    platform AMD does not publish them for."""
    torch, torchvision, _ = hwdetect.amd_rocm_win_torch_packages()
    source = ["--extra-index-url", hwdetect.amd_rocm_win_index(family),
              "--index-strategy", "unsafe-best-match"]
    sdk = list(hwdetect.amd_rocm_win_sdk_packages(family))

    venv = tmp_path / "disposable-venv"
    made = subprocess.run(
        ["uv", "venv", "--python", sys.executable, str(venv)],
        capture_output=True, text=True, timeout=60)
    assert made.returncode == 0, f"stdout: {made.stdout}\nstderr: {made.stderr}"
    venv_python = venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")

    resolves = subprocess.run(
        ["uv", "pip", "install", "--dry-run", "--python", str(venv_python),
         "--python-platform", "windows", "--python-version", "3.12",
         *source, torch, torchvision, *sdk],
        capture_output=True, text=True, timeout=60)
    assert resolves.returncode == 0, (
        f"stdout: {resolves.stdout}\nstderr: {resolves.stderr}")

    # Fires-control: the identical request for macOS, which AMD publishes no
    # ROCm wheels for, must fail.
    doesnt_resolve = subprocess.run(
        ["uv", "pip", "install", "--dry-run", "--python", str(venv_python),
         "--python-platform", "macos", "--python-version", "3.12",
         *source, torch, torchvision, *sdk],
        capture_output=True, text=True, timeout=60)
    assert doesnt_resolve.returncode != 0
    assert "No solution found" in doesnt_resolve.stderr


def test_amd_rocm_on_windows_unknown_card_skips_not_mispins(monkeypatch):
    # An RDNA1 / unknown AMD card on Windows has no verified prebuilt: skip and
    # guide, never silently install the gfx103X wheel that will not load.
    monkeypatch.setattr(hwdetect.sys, "platform", "win32")
    assert hwdetect.torch_pip_args("amd-rocm", _det("amd radeon rx 5700 xt")) == ""
