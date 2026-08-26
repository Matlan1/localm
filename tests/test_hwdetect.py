# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for localm.hwdetect: GPU-vendor detection and the install/runtime backend
selection policy. hwdetect drives which llama.cpp backend and which PyTorch wheel
the installers provision. Everything is mocked - no real GPU, subprocess, or
network - so the policy is pinned deterministically on any machine, including a
GPU-less CI box."""

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
        returncode = 0
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: R())
    assert hwdetect._run(["x"]) == "outerr"


def test_run_ok_is_false_on_missing_tool(monkeypatch):
    def boom(*a, **k):
        raise FileNotFoundError("no such tool")
    monkeypatch.setattr(subprocess, "run", boom)
    assert hwdetect._run_ok(["nonexistent-tool"]) == ("", False)


def test_run_ok_is_false_on_nonzero_exit(monkeypatch):
    """A tool that ran and FAILED did not answer the question either. Its output
    is still returned unchanged (long-standing behaviour detect() tolerates); only
    the flag records that it cannot be read as an answer."""
    class R:
        stdout = "some error text"
        stderr = ""
        returncode = 1
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: R())
    assert hwdetect._run_ok(["x"]) == ("some error text", False)


def test_run_ok_is_true_on_clean_exit_with_no_output(monkeypatch):
    """The load-bearing case: exit 0 with nothing to say IS an answer ("this box
    has no display adapters"), and must not read as a failed probe."""
    class R:
        stdout = ""
        stderr = ""
        returncode = 0
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: R())
    assert hwdetect._run_ok(["x"]) == ("", True)


# ------------- the three-state result: found / none / unknown -------------
#
# has_gpu is a BOOLEAN and cannot express "we could not ask": it is False both
# for a box with no GPU and for a box whose enumeration never ran. gpu_state is
# the discriminator, and these tests pin that those two inputs produce DIFFERENT
# values.

def test_probe_that_ran_and_found_nothing_is_none(monkeypatch):
    monkeypatch.setattr(hwdetect.sys, "platform", "win32")
    monkeypatch.setattr(hwdetect, "_win_gpu_names",
                        lambda: ("microsoft basic display adapter", True))
    monkeypatch.setattr(hwdetect.shutil, "which", lambda n: None)
    d = hwdetect.detect()
    assert d.vendors == []
    assert d.gpu_state == "none"
    assert d.probe_ok is True
    assert d.probe_error == ""


def test_probe_that_could_not_run_is_unknown_not_none(monkeypatch):
    """Same empty vendor list, opposite epistemic state."""
    monkeypatch.setattr(hwdetect.sys, "platform", "win32")
    monkeypatch.setattr(hwdetect, "_win_gpu_names", lambda: ("", False))
    monkeypatch.setattr(hwdetect.shutil, "which", lambda n: None)
    d = hwdetect.detect()
    assert d.vendors == []
    assert d.gpu_state == "unknown"
    assert d.probe_ok is False
    assert d.probe_error, "a failed probe must record WHY, not just that it failed"
    # The recommendation stays conservative and has_gpu stays False: this makes
    # the state VISIBLE without turning a best-effort probe into a hard failure
    # or a positive claim.
    assert d.has_gpu is False
    assert d.recommended == "cpu"


def test_linux_probe_that_could_not_run_is_unknown(monkeypatch):
    """lspci is absent on plenty of minimal installs and containers, where
    "no display controllers" would be fabricated rather than measured."""
    monkeypatch.setattr(hwdetect.sys, "platform", "linux")
    monkeypatch.setattr(hwdetect, "_linux_gpu_names", lambda: ("", False))
    monkeypatch.setattr(hwdetect.shutil, "which", lambda n: None)
    monkeypatch.setattr(hwdetect.Path, "is_dir", lambda self: False)
    d = hwdetect.detect()
    assert d.gpu_state == "unknown"


def test_vendor_found_by_smi_stays_found_despite_a_failed_enumeration(monkeypatch):
    """A failed probe cannot RETRACT evidence another probe supplied: nvidia-smi
    on PATH is positive proof even when the adapter enumeration did not run.
    probe_ok stays factual (the enumeration really did fail); gpu_state resolves
    the two into the answer."""
    monkeypatch.setattr(hwdetect.sys, "platform", "win32")
    monkeypatch.setattr(hwdetect, "_win_gpu_names", lambda: ("", False))
    monkeypatch.setattr(hwdetect.shutil, "which",
                        lambda n: "C:/w/nvidia-smi.exe" if n == "nvidia-smi" else None)
    d = hwdetect.detect()
    assert d.vendors == ["nvidia"]
    assert d.gpu_state == "found"
    assert d.probe_ok is False


def test_macos_uname_failure_is_unknown_not_intel(monkeypatch):
    """uname producing anything other than arm64 must not be read as
    source="macos intel" when uname did not run at all. On an Apple Silicon box
    that asserts the wrong architecture and costs the Metal recommendation."""
    monkeypatch.setattr(hwdetect.sys, "platform", "darwin")
    monkeypatch.setattr(hwdetect, "_run_ok", lambda cmd: ("", False))
    d = hwdetect.detect()
    assert d.gpu_state == "unknown"
    assert "intel" not in d.source
    assert d.recommended == "cpu"       # still conservative, just not mislabelled


def test_macos_intel_still_reports_none_when_uname_answered(monkeypatch):
    """The guard on the test above: a real Intel Mac is a genuine negative and
    must keep saying so."""
    monkeypatch.setattr(hwdetect.sys, "platform", "darwin")
    monkeypatch.setattr(hwdetect, "_run_ok", lambda cmd: ("x86_64\n", True))
    d = hwdetect.detect()
    assert d.gpu_state == "none"
    assert d.source == "macos intel"


# --------------------------- detect(): win32 ---------------------------

def _win(monkeypatch, names, which=lambda n: None):
    monkeypatch.setattr(hwdetect.sys, "platform", "win32")
    monkeypatch.setattr(hwdetect, "_win_gpu_names", lambda: (names.lower(), True))
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
    _win(monkeypatch, "", which=lambda n: "Z:/x/nvidia-smi.exe" if n == "nvidia-smi" else None)
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
    monkeypatch.setattr(hwdetect, "_run_ok", lambda cmd: ("arm64\n", True))
    d = hwdetect.detect()
    assert d.vendors == ["apple"]
    assert d.recommended == "metal"


def test_detect_macos_intel_is_cpu(monkeypatch):
    monkeypatch.setattr(hwdetect.sys, "platform", "darwin")
    monkeypatch.setattr(hwdetect, "_run_ok", lambda cmd: ("x86_64\n", True))
    d = hwdetect.detect()
    assert d.vendors == []
    assert d.recommended == "cpu"


# --------------------------- detect(): linux ---------------------------

def test_detect_linux_nvidia_via_smi(monkeypatch):
    monkeypatch.setattr(hwdetect.sys, "platform", "linux")
    monkeypatch.setattr(hwdetect, "_linux_gpu_names", lambda: ("", True))
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
    monkeypatch.setattr(hwdetect, "_linux_gpu_names", lambda: ("", True))
    monkeypatch.setattr(hwdetect.shutil, "which",
                        lambda n: "/usr/bin/rocminfo" if n == "rocminfo" else None)
    monkeypatch.setattr(hwdetect.Path, "is_dir", lambda self: False)  # isolate the rocminfo signal
    assert hwdetect.detect().vendors == ["amd"]


def test_detect_linux_no_gpu_is_cpu(monkeypatch):
    monkeypatch.setattr(hwdetect.sys, "platform", "linux")
    monkeypatch.setattr(hwdetect, "_linux_gpu_names", lambda: ("", True))
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
    monkeypatch.setattr(hwdetect, "_win_gpu_names", lambda: ("", True))
    monkeypatch.setattr(hwdetect.shutil, "which", lambda n: None)
    d = hwdetect.detect()
    assert isinstance(d, Detection)
    assert d.recommended in ("cpu", "vulkan")


# ------------------- recommended_install_backend (policy) -------------------

@pytest.mark.parametrize("vendors,expected", [
    (["apple"], "metal"),
    ([], "cpu"),
])
def test_install_backend_early_return_branches(vendors, expected):
    # apple/no-gpu are platform-independent; nvidia is platform-scoped (cuda on
    # Windows, vulkan elsewhere) and covered by its own tests below.
    assert hwdetect.recommended_install_backend(Detection(vendors=vendors)) == expected


def test_install_backend_win_nvidia_is_cuda(monkeypatch):
    # NVIDIA on Windows -> cuda: the release ships a self-contained cudart
    # bundle. The setup-llama driver preflight + load-test fall back to vulkan
    # if unsupported.
    monkeypatch.setattr(hwdetect.sys, "platform", "win32")
    d = Detection(vendors=["nvidia"], gpu_names="nvidia geforce rtx 4090")
    assert hwdetect.recommended_install_backend(d) == "cuda"


def test_install_backend_linux_nvidia_is_cuda(monkeypatch):
    # NVIDIA on Linux -> cuda: llama.cpp ships a self-contained cudart bundle on
    # Linux too, so setup-llama fetches the CUDA runtime libraries itself and no
    # system Toolkit is needed.
    monkeypatch.setattr(hwdetect.sys, "platform", "linux")
    d = Detection(vendors=["nvidia"], gpu_names="nvidia geforce rtx 4090")
    assert hwdetect.recommended_install_backend(d) == "cuda"


def test_install_backend_linux_nvidia_blackwell_is_cuda(monkeypatch):
    # Compute capability alone, not the platform, decides NVIDIA gets cuda.
    # recommended_install_backend does not branch on architecture itself (that is
    # NvidiaInfo.cuda_line's job), so this confirms a Blackwell-named/CC-bearing
    # card is not excluded by name-based reasoning.
    monkeypatch.setattr(hwdetect.sys, "platform", "linux")
    d = Detection(vendors=["nvidia"], gpu_names="nvidia rtx pro 4000 blackwell")
    assert hwdetect.recommended_install_backend(d) == "cuda"


def test_install_backend_win_amd_rx6000_is_rocm(monkeypatch):
    # gfx103X keeps the self-contained amd-rocm bundle EVEN when a system ROCm
    # toolkit is also present - it needs no toolkit at all, so it wins outright.
    monkeypatch.setattr(hwdetect.sys, "platform", "win32")
    monkeypatch.setattr(hwdetect, "_rocm_toolkit_present", lambda: True)
    d = Detection(vendors=["amd"], gpu_names="amd radeon rx 6800 xt")
    assert hwdetect.recommended_install_backend(d) == "amd-rocm"


def test_install_backend_win_amd_rx7000_without_rocm_is_vulkan(monkeypatch):
    # RX 7000 is not the bundled gfx103X build, and with no system ROCm toolkit
    # detected hip cannot run here, so vulkan is the fallback.
    monkeypatch.setattr(hwdetect.sys, "platform", "win32")
    monkeypatch.setattr(hwdetect, "_rocm_toolkit_present", lambda: False)
    d = Detection(vendors=["amd"], gpu_names="amd radeon rx 7900 xtx")
    assert hwdetect.recommended_install_backend(d) == "vulkan"


def test_install_backend_win_amd_rx7000_with_rocm_is_hip(monkeypatch):
    # gfx110X has no self-contained build, but `hip` is a real downloadable
    # binary that works when the AMD HIP SDK is already installed, detected via
    # the same signal detect() probes for vendor identification.
    monkeypatch.setattr(hwdetect.sys, "platform", "win32")
    monkeypatch.setattr(hwdetect, "_rocm_toolkit_present", lambda: True)
    d = Detection(vendors=["amd"], gpu_names="amd radeon rx 7900 xtx")
    assert hwdetect.recommended_install_backend(d) == "hip"


def test_install_backend_win_amd_unknown_name_defaults_to_rocm(monkeypatch):
    # An UNKNOWN AMD card on Windows defaults to the bundled gfx103X build
    # (amd-rocm), with setup-llama's load-test + vulkan fallback as the net.
    # Empty gpu_names ("could not read the adapter name") is the unknown case, so
    # it is amd-rocm and not vulkan, even with a system ROCm toolkit present.
    monkeypatch.setattr(hwdetect.sys, "platform", "win32")
    monkeypatch.setattr(hwdetect, "_rocm_toolkit_present", lambda: True)
    assert hwdetect.recommended_install_backend(
        Detection(vendors=["amd"], gpu_names="")) == "amd-rocm"


def test_install_backend_win_amd_order_must_be_exact(monkeypatch):
    # The amd-rocm case requires vendors == ["amd"] exactly. A mixed list that
    # includes nvidia resolves to cuda on Windows, NOT the AMD build.
    monkeypatch.setattr(hwdetect.sys, "platform", "win32")
    assert hwdetect.recommended_install_backend(
        Detection(vendors=["amd", "nvidia"], gpu_names="rx 6800")) == "cuda"


def test_install_backend_linux_amd_without_rocm_is_vulkan(monkeypatch):
    # No self-contained bundle exists for Linux AMD, and with no system ROCm
    # toolkit detected, hip genuinely cannot run here -> vulkan is correct.
    monkeypatch.setattr(hwdetect.sys, "platform", "linux")
    monkeypatch.setattr(hwdetect, "_rocm_toolkit_present", lambda: False)
    d = Detection(vendors=["amd"], gpu_names="amd radeon rx 6800")
    assert hwdetect.recommended_install_backend(d) == "vulkan"


def test_install_backend_linux_amd_with_rocm_is_hip(monkeypatch):
    # hip is a real downloadable binary on Linux too
    # (setup_llama._BACKEND_ASSETS), viable the moment a working system ROCm
    # install is detected.
    monkeypatch.setattr(hwdetect.sys, "platform", "linux")
    monkeypatch.setattr(hwdetect, "_rocm_toolkit_present", lambda: True)
    d = Detection(vendors=["amd"], gpu_names="amd radeon rx 6800")
    assert hwdetect.recommended_install_backend(d) == "hip"


# ------------------------- _rocm_toolkit_present -------------------------

def test_rocm_toolkit_present_via_rocminfo(monkeypatch):
    monkeypatch.setattr(hwdetect.shutil, "which",
                        lambda n: "/usr/bin/rocminfo" if n == "rocminfo" else None)
    assert hwdetect._rocm_toolkit_present() is True


def test_rocm_toolkit_present_via_rocm_smi(monkeypatch):
    monkeypatch.setattr(hwdetect.shutil, "which",
                        lambda n: "/usr/bin/rocm-smi" if n == "rocm-smi" else None)
    assert hwdetect._rocm_toolkit_present() is True


def test_rocm_toolkit_present_via_opt_rocm_dir_linux_only(monkeypatch):
    monkeypatch.setattr(hwdetect.shutil, "which", lambda n: None)
    monkeypatch.setattr(hwdetect.Path, "is_dir", lambda self: True)
    monkeypatch.setattr(hwdetect.sys, "platform", "linux")
    assert hwdetect._rocm_toolkit_present() is True
    # The /opt/rocm check is Linux-only: confirm it is NOT consulted on Windows.
    monkeypatch.setattr(hwdetect.sys, "platform", "win32")
    assert hwdetect._rocm_toolkit_present() is False


def test_rocm_toolkit_absent_is_false(monkeypatch):
    monkeypatch.setattr(hwdetect.shutil, "which", lambda n: None)
    monkeypatch.setattr(hwdetect.Path, "is_dir", lambda self: False)
    assert hwdetect._rocm_toolkit_present() is False


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

def test_torch_pip_args_cuda(monkeypatch):
    # Deterministic regardless of the test-running machine's own hardware.
    monkeypatch.setattr(hwdetect, "_cuda_compute_capabilities", lambda: [])
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
    # Pin Linux so the install-backend is deterministic; this asserts the OUTPUT
    # FORMAT "<vendor> <backend>", and NVIDIA is cuda on both platforms, so this
    # and the Windows path share one expectation. main() calls
    # recommended_install_backend(d), NOT d.recommended, so the "vulkan" passed
    # to Detection() below is the WRONG legacy value and main() must ignore it.
    monkeypatch.setattr(hwdetect.sys, "platform", "linux")
    monkeypatch.setattr(hwdetect, "detect",
                        lambda: Detection(vendors=["nvidia"], recommended="vulkan"))
    assert hwdetect.main([]) == 0
    assert capsys.readouterr().out.strip() == "nvidia cuda"


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
    # Deterministic regardless of the test-running machine's own hardware.
    monkeypatch.setattr(hwdetect, "_cuda_compute_capabilities", lambda: [])
    hwdetect.main(["torch-args", "cuda"])
    assert "cu126" in capsys.readouterr().out


def test_main_torch_args_missing_backend_defaults_empty(monkeypatch, capsys):
    # "torch-args" with no backend -> backend="" -> vendor-neutral routing by GPU.
    monkeypatch.setattr(hwdetect, "detect", lambda: Detection(vendors=["nvidia"]))
    # Deterministic regardless of the test-running machine's own hardware.
    monkeypatch.setattr(hwdetect, "_cuda_compute_capabilities", lambda: [])
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
