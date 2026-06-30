# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for localm.hwdetect: GPU-vendor detection and the install/runtime backend
selection policy. hwdetect drives which llama.cpp backend and which PyTorch wheel
the installers provision, so a silent regression here is exactly the costly kind
(TEST-2). Everything is mocked - no real GPU, subprocess, or network - so the
policy is pinned deterministically on any machine, including a GPU-less CI box."""

import subprocess

import pytest

from localm import hwdetect
from localm.hwdetect import Detection


# --------------------------- _run (never raises) ---------------------------

def test_run_returns_empty_on_missing_tool(monkeypatch):
    def boom(*a, **k):
        raise FileNotFoundError("no such tool")
    monkeypatch.setattr(subprocess, "run", boom)
    assert hwdetect._run(["nonexistent-tool"]) == ""


def test_run_returns_empty_on_timeout(monkeypatch):
    def boom(*a, **k):
        raise subprocess.TimeoutExpired(cmd="x", timeout=8)
    monkeypatch.setattr(subprocess, "run", boom)
    assert hwdetect._run(["slow"]) == ""


def test_run_combines_stdout_and_stderr(monkeypatch):
    class R:
        stdout = "out"
        stderr = "err"
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: R())
    assert hwdetect._run(["x"]) == "outerr"


# --------------------------- detect(): win32 ---------------------------

def _win(monkeypatch, names, which=lambda n: None):
    monkeypatch.setattr(hwdetect.sys, "platform", "win32")
    monkeypatch.setattr(hwdetect, "_win_gpu_names", lambda: names.lower())
    monkeypatch.setattr(hwdetect.shutil, "which", which)


def test_detect_win_nvidia(monkeypatch):
    _win(monkeypatch, "NVIDIA GeForce RTX 4090")
    d = hwdetect.detect()
    assert d.vendors == ["nvidia"]
    assert d.recommended == "vulkan"
    assert d.has_gpu is True
    assert d.source == "win32 video controllers"


def test_detect_win_amd_radeon(monkeypatch):
    _win(monkeypatch, "AMD Radeon RX 6800 XT")
    d = hwdetect.detect()
    assert d.vendors == ["amd"]
    assert d.recommended == "vulkan"


def test_detect_win_intel_arc(monkeypatch):
    _win(monkeypatch, "Intel(R) Arc(TM) A770 Graphics")
    assert hwdetect.detect().vendors == ["intel"]


def test_detect_win_no_gpu_is_cpu(monkeypatch):
    _win(monkeypatch, "Microsoft Basic Display Adapter")
    d = hwdetect.detect()
    assert d.vendors == []
    assert d.recommended == "cpu"
    assert d.has_gpu is False


def test_detect_win_vendor_priority_order(monkeypatch):
    # Names mention amd first, but the result is priority-ordered (nvidia, amd).
    _win(monkeypatch, "AMD Radeon RX 6800\nNVIDIA GeForce RTX 4090")
    assert hwdetect.detect().vendors == ["nvidia", "amd"]


def test_detect_win_nvidia_smi_on_path_counts(monkeypatch):
    # No GPU name, but nvidia-smi present -> nvidia detected.
    _win(monkeypatch, "", which=lambda n: "C:/x/nvidia-smi.exe" if n == "nvidia-smi" else None)
    assert hwdetect.detect().vendors == ["nvidia"]


def test_detect_win_mixed_signal_sources(monkeypatch):
    # nvidia from the adapter NAME, amd from a TOOL (rocm-smi): two different
    # signal sources combine, priority-ordered.
    _win(monkeypatch, "NVIDIA GeForce RTX 4090",
         which=lambda n: "rocm-smi.exe" if n == "rocm-smi" else None)
    assert hwdetect.detect().vendors == ["nvidia", "amd"]


# --------------------------- detect(): darwin ---------------------------

def test_detect_macos_arm64_is_metal(monkeypatch):
    monkeypatch.setattr(hwdetect.sys, "platform", "darwin")
    monkeypatch.setattr(hwdetect, "_run", lambda cmd: "arm64\n")
    d = hwdetect.detect()
    assert d.vendors == ["apple"]
    assert d.recommended == "metal"


def test_detect_macos_intel_is_cpu(monkeypatch):
    monkeypatch.setattr(hwdetect.sys, "platform", "darwin")
    monkeypatch.setattr(hwdetect, "_run", lambda cmd: "x86_64\n")
    d = hwdetect.detect()
    assert d.vendors == []
    assert d.recommended == "cpu"


# --------------------------- detect(): linux ---------------------------

def test_detect_linux_nvidia_via_smi(monkeypatch):
    monkeypatch.setattr(hwdetect.sys, "platform", "linux")
    monkeypatch.setattr(hwdetect, "_linux_gpu_names", lambda: "")
    monkeypatch.setattr(hwdetect.shutil, "which",
                        lambda n: "/usr/bin/nvidia-smi" if n == "nvidia-smi" else None)
    # Pin /opt/rocm absent: the detector treats that dir as an AMD signal, and the
    # WSL/Linux test host may actually have it, which would add a spurious "amd".
    monkeypatch.setattr(hwdetect.Path, "is_dir", lambda self: False)
    d = hwdetect.detect()
    assert d.vendors == ["nvidia"]
    assert d.source == "lspci / driver tools"


def test_detect_linux_amd_via_rocminfo(monkeypatch):
    monkeypatch.setattr(hwdetect.sys, "platform", "linux")
    monkeypatch.setattr(hwdetect, "_linux_gpu_names", lambda: "")
    monkeypatch.setattr(hwdetect.shutil, "which",
                        lambda n: "/usr/bin/rocminfo" if n == "rocminfo" else None)
    monkeypatch.setattr(hwdetect.Path, "is_dir", lambda self: False)  # isolate the rocminfo signal
    assert hwdetect.detect().vendors == ["amd"]


def test_detect_linux_no_gpu_is_cpu(monkeypatch):
    monkeypatch.setattr(hwdetect.sys, "platform", "linux")
    monkeypatch.setattr(hwdetect, "_linux_gpu_names", lambda: "")
    monkeypatch.setattr(hwdetect.shutil, "which", lambda n: None)
    # Guard against a real /opt/rocm on the test host skewing the result.
    monkeypatch.setattr(hwdetect.Path, "is_dir", lambda self: False)
    d = hwdetect.detect()
    assert d.vendors == []
    assert d.recommended == "cpu"


def test_detect_never_raises_even_with_dead_probes(monkeypatch):
    # All probes return nothing/raise-internally; detect() must still yield a
    # valid Detection, never propagate.
    monkeypatch.setattr(hwdetect.sys, "platform", "win32")
    monkeypatch.setattr(hwdetect, "_win_gpu_names", lambda: "")
    monkeypatch.setattr(hwdetect.shutil, "which", lambda n: None)
    d = hwdetect.detect()
    assert isinstance(d, Detection)
    assert d.recommended in ("cpu", "vulkan")


# ------------------- recommended_install_backend (policy) -------------------

def test_install_backend_apple_metal():
    assert hwdetect.recommended_install_backend(
        Detection(vendors=["apple"], recommended="metal")) == "metal"


def test_install_backend_no_gpu_cpu():
    assert hwdetect.recommended_install_backend(Detection(vendors=[])) == "cpu"


def test_install_backend_nvidia_vulkan():
    assert hwdetect.recommended_install_backend(
        Detection(vendors=["nvidia"])) == "vulkan"


def test_install_backend_win_amd_rx6000_is_rocm(monkeypatch):
    monkeypatch.setattr(hwdetect.sys, "platform", "win32")
    d = Detection(vendors=["amd"], gpu_names="amd radeon rx 6800 xt")
    assert hwdetect.recommended_install_backend(d) == "amd-rocm"


def test_install_backend_win_amd_rx7000_is_vulkan(monkeypatch):
    # RX 7000 is clearly NOT the bundled gfx103X build -> vulkan, not amd-rocm.
    monkeypatch.setattr(hwdetect.sys, "platform", "win32")
    d = Detection(vendors=["amd"], gpu_names="amd radeon rx 7900 xtx")
    assert hwdetect.recommended_install_backend(d) == "vulkan"


def test_install_backend_win_amd_unknown_name_defaults_to_rocm(monkeypatch):
    # Policy (see docstring): an UNKNOWN AMD card on Windows DEFAULTS to the
    # bundled gfx103X build (amd-rocm), with setup-llama's load-test + vulkan
    # fallback as the net. Empty gpu_names ("could not read the adapter name")
    # is the unknown case, so it must be amd-rocm, NOT vulkan.
    monkeypatch.setattr(hwdetect.sys, "platform", "win32")
    assert hwdetect.recommended_install_backend(
        Detection(vendors=["amd"], gpu_names="")) == "amd-rocm"


def test_install_backend_win_amd_order_must_be_exact(monkeypatch):
    # The amd-rocm case requires vendors == ["amd"] exactly; a mixed list (even
    # ["amd", "nvidia"]) falls through to vulkan.
    monkeypatch.setattr(hwdetect.sys, "platform", "win32")
    assert hwdetect.recommended_install_backend(
        Detection(vendors=["amd", "nvidia"], gpu_names="rx 6800")) == "vulkan"


def test_install_backend_linux_amd_is_vulkan(monkeypatch):
    # The amd-rocm bundle is Windows-only; AMD on Linux -> vulkan.
    monkeypatch.setattr(hwdetect.sys, "platform", "linux")
    d = Detection(vendors=["amd"], gpu_names="amd radeon rx 6800")
    assert hwdetect.recommended_install_backend(d) == "vulkan"


def test_install_backend_win_mixed_amd_nvidia_is_vulkan(monkeypatch):
    # vendors != ["amd"] (mixed) -> the narrow amd-rocm case does not apply.
    monkeypatch.setattr(hwdetect.sys, "platform", "win32")
    d = Detection(vendors=["nvidia", "amd"], gpu_names="rx 6800 rtx 4090")
    assert hwdetect.recommended_install_backend(d) == "vulkan"


# ----------------------- recommended_torch_variant -----------------------

@pytest.mark.parametrize("backend,expected", [
    ("cuda", "cuda"),
    ("amd-rocm", "rocm"),
    ("rocm", "rocm"),
    ("hip", "rocm"),
    ("sycl", "xpu"),
    ("cpu", "none"),
])
def test_torch_variant_by_explicit_backend(backend, expected):
    # Explicit backend wins regardless of detected hardware (pass no GPU).
    assert hwdetect.recommended_torch_variant(backend, Detection(vendors=[])) == expected


def test_torch_variant_vulkan_routes_by_detected_gpu():
    assert hwdetect.recommended_torch_variant("vulkan", Detection(vendors=["nvidia"])) == "cuda"
    assert hwdetect.recommended_torch_variant("vulkan", Detection(vendors=["intel"])) == "xpu"
    assert hwdetect.recommended_torch_variant("vulkan", Detection(vendors=["amd"])) == "none"
    assert hwdetect.recommended_torch_variant("vulkan", Detection(vendors=[])) == "none"


def test_torch_variant_unknown_backend_routes_by_gpu():
    # "own"/empty/unknown are vendor-neutral too; never rocm on a neutral pick.
    assert hwdetect.recommended_torch_variant("", Detection(vendors=["nvidia"])) == "cuda"
    assert hwdetect.recommended_torch_variant("own", Detection(vendors=["amd"])) == "none"


# ------------------------- amd family helpers -------------------------

@pytest.mark.parametrize("name,fam", [
    ("amd radeon rx 9070", "gfx120x"),
    ("amd radeon rx 7900 xtx", "gfx110x"),
    ("amd radeon rx 6800 xt", "gfx103x"),
    ("amd radeon rx 6900 xt", "gfx103x"),
    ("amd radeon rx 5700", ""),
    ("amd radeon vii", ""),
    ("rx", ""),            # substring but no family digit
    ("rx 4", ""),          # undefined family
    ("some old card", ""),  # no rx token at all
    ("", ""),
])
def test_amd_gfx_family(name, fam):
    assert hwdetect.amd_gfx_family(name) == fam


@pytest.mark.parametrize("name,known", [
    ("rx 5700", True),
    ("rx 7900", True),
    ("rx 9070", True),
    ("radeon vii", True),
    ("instinct mi100", True),
    ("rx 6800", False),
    ("", False),
])
def test_amd_known_non_gfx103x(name, known):
    assert hwdetect._amd_known_non_gfx103x(name) is known


# ------------------------- torch_pip_args (wheel source) -------------------

def test_torch_pip_args_cuda():
    args = hwdetect.torch_pip_args("cuda", Detection(vendors=["nvidia"]))
    assert "cu126" in args and args.startswith("torch torchvision")


def test_torch_pip_args_cpu_is_empty():
    assert hwdetect.torch_pip_args("cpu", Detection(vendors=[])) == ""


def test_torch_pip_args_rocm_linux(monkeypatch):
    monkeypatch.setattr(hwdetect.sys, "platform", "linux")
    args = hwdetect.torch_pip_args("rocm", Detection(vendors=["amd"], gpu_names="rx 6800"))
    assert "rocm6.2" in args


def test_torch_pip_args_rocm_win_gfx103x_uses_bundled(monkeypatch):
    monkeypatch.setattr(hwdetect.sys, "platform", "win32")
    args = hwdetect.torch_pip_args("amd-rocm", Detection(vendors=["amd"], gpu_names="rx 6800"))
    assert args == "-e .[gpu]"


def test_torch_pip_args_rocm_win_gfx110x_uses_amd_wheels(monkeypatch):
    monkeypatch.setattr(hwdetect.sys, "platform", "win32")
    args = hwdetect.torch_pip_args("amd-rocm", Detection(vendors=["amd"], gpu_names="rx 7900"))
    assert "rocm6.4" in args


def test_torch_pip_args_rocm_win_unknown_amd_is_empty(monkeypatch):
    monkeypatch.setattr(hwdetect.sys, "platform", "win32")
    args = hwdetect.torch_pip_args("amd-rocm", Detection(vendors=["amd"], gpu_names="some old card"))
    assert args == ""


# ------------------------------- main() CLI -------------------------------

def test_main_prints_vendor_and_backend(monkeypatch, capsys):
    monkeypatch.setattr(hwdetect, "detect",
                        lambda: Detection(vendors=["nvidia"], recommended="vulkan"))
    assert hwdetect.main([]) == 0
    assert capsys.readouterr().out.strip() == "nvidia vulkan"


def test_main_no_gpu_prints_none_cpu(monkeypatch, capsys):
    monkeypatch.setattr(hwdetect, "detect", lambda: Detection(vendors=[], recommended="cpu"))
    hwdetect.main([])
    assert capsys.readouterr().out.strip() == "none cpu"


def test_main_torch_subcommand(monkeypatch, capsys):
    monkeypatch.setattr(hwdetect, "detect", lambda: Detection(vendors=["nvidia"]))
    hwdetect.main(["torch", "vulkan"])
    assert capsys.readouterr().out.strip() == "cuda"


def test_main_torch_args_subcommand(monkeypatch, capsys):
    monkeypatch.setattr(hwdetect, "detect", lambda: Detection(vendors=["nvidia"]))
    hwdetect.main(["torch-args", "cuda"])
    assert "cu126" in capsys.readouterr().out


def test_main_torch_args_missing_backend_defaults_empty(monkeypatch, capsys):
    # "torch-args" with no backend -> backend="" -> vendor-neutral routing by GPU.
    monkeypatch.setattr(hwdetect, "detect", lambda: Detection(vendors=["nvidia"]))
    hwdetect.main(["torch-args"])
    assert "cu126" in capsys.readouterr().out   # neutral + nvidia -> cuda wheels


def test_main_torch_missing_backend_defaults_empty(monkeypatch, capsys):
    monkeypatch.setattr(hwdetect, "detect", lambda: Detection(vendors=[]))
    hwdetect.main(["torch"])
    assert capsys.readouterr().out.strip() == "none"   # neutral + no gpu -> none


def test_main_unknown_subcommand_prints_vendor_line(monkeypatch, capsys):
    # An unrecognised first arg is not a subcommand: fall through to the default
    # "<vendor> <install-backend>" form, never an error.
    monkeypatch.setattr(hwdetect.sys, "platform", "linux")
    monkeypatch.setattr(hwdetect, "detect",
                        lambda: Detection(vendors=["amd"], recommended="vulkan",
                                          gpu_names="rx 6800"))
    assert hwdetect.main(["wat"]) == 0
    assert capsys.readouterr().out.strip() == "amd vulkan"
