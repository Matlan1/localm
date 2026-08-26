# SPDX-License-Identifier: AGPL-3.0-or-later
"""doctor's smi probe: a FAILING vendor tool is not a working GPU.

`_check_gpu_driver` must look at the return code, not only at
`subprocess.run(...).stdout`. nvidia-smi and rocm-smi report a broken driver by
printing an error and exiting non-zero, so a stdout-only check answers "did the
tool print something" while the reader hears "you have a working GPU driver".

That return value feeds `smi_or_torch_gpu`, and `_check_gpu_verdict`'s step (3)
returns EARLY when it is True, so a broken driver counted as a GPU also
suppresses the "No GPU detected ... CPU mode only" line.

Absent and broken are NOT treated alike: absent is the normal case on nearly
every box and stays silent, while present-and-failing is surfaced.
"""

from __future__ import annotations

import importlib
import subprocess

import localm.cli as cli

doctor_mod = importlib.import_module("localm.cli.doctor")

# The nvidia-smi wording for a driver updated without a reboot.
_NVML_ERROR = "Failed to initialize NVML: Driver/library version mismatch"


def _smi(monkeypatch, *, tool="nvidia-smi", returncode=0, stdout="", stderr=""):
    """Make exactly one vendor tool answer; every other command looks absent.

    Patches subprocess.run wholesale, which also makes doctor's venv-creation
    probe report an error. That error is noise here and is not asserted on."""
    def _run(cmd, *a, **k):
        if cmd and cmd[0] == tool:
            return subprocess.CompletedProcess(cmd, returncode, stdout, stderr)
        raise FileNotFoundError(cmd[0] if cmd else "?")

    monkeypatch.setattr(subprocess, "run", _run)


# --------------------------------------------------------------------------- #
#  The check itself                                                            #
# --------------------------------------------------------------------------- #

def test_a_failing_smi_is_not_counted_as_a_gpu(monkeypatch, capsys):
    """Non-zero exit means no GPU was proven, whatever landed on stdout."""
    _smi(monkeypatch, returncode=9, stdout=_NVML_ERROR)

    assert doctor_mod._check_gpu_driver() is False


def test_a_failing_smi_never_prints_its_error_as_a_device_name(monkeypatch, capsys):
    """A failing tool's error text is never printed as the device name the OK
    line names. The error may appear elsewhere (see the next test)."""
    _smi(monkeypatch, returncode=9, stdout=_NVML_ERROR)

    doctor_mod._check_gpu_driver()
    out = capsys.readouterr().out

    assert "NVIDIA GPU:" not in out, out


def test_a_failing_smi_is_surfaced_not_silenced(monkeypatch, capsys):
    """'Installed but broken' is surfaced, not collapsed into the same silence
    as 'not installed'."""
    _smi(monkeypatch, returncode=9, stdout=_NVML_ERROR)

    doctor_mod._check_gpu_driver()
    out = capsys.readouterr().out

    assert "nvidia-smi" in out
    assert "installed but failed" in out
    assert "Driver/library version mismatch" in out


def test_a_tool_that_fails_with_only_stderr_still_reports_what_it_said(
        monkeypatch, capsys):
    """Some builds put the reason on stderr instead; the fallback reaches for
    it rather than printing a bare exit code."""
    _smi(monkeypatch, returncode=1, stdout="", stderr="NVML library not found")

    doctor_mod._check_gpu_driver()
    out = capsys.readouterr().out

    assert "NVML library not found" in out


def test_a_clean_smi_is_still_reported_as_a_gpu(monkeypatch, capsys):
    """Exit 0 with a device name returns True and prints the name."""
    _smi(monkeypatch, returncode=0, stdout="NVIDIA GeForce RTX 4090, 24564 MiB")

    assert doctor_mod._check_gpu_driver() is True
    out = capsys.readouterr().out
    assert "NVIDIA GeForce RTX 4090" in out
    assert "installed but failed" not in out


def test_an_absent_tool_stays_silent(monkeypatch, capsys):
    """An absent tool is not a fault and produces no warning; the default GPU
    paths (Vulkan/Metal/bundled-ROCm) do not need these tools."""
    _smi(monkeypatch, tool="__nothing_matches__")

    assert doctor_mod._check_gpu_driver() is False
    out = capsys.readouterr().out
    assert "installed but failed" not in out
    assert "nvidia-smi" not in out


def test_a_clean_exit_with_no_output_is_not_a_gpu(monkeypatch, capsys):
    """Exit 0 with no output must not become a tick with an empty device
    name."""
    _smi(monkeypatch, returncode=0, stdout="   \n")

    assert doctor_mod._check_gpu_driver() is False
# --------------------------------------------------------------------------- #
#  Effect on the CPU-only verdict                                              #
# --------------------------------------------------------------------------- #

def test_a_failing_smi_does_not_suppress_the_cpu_only_verdict(
        cli_runner, monkeypatch):
    """_check_gpu_verdict's step (3) returns EARLY when smi/torch claim a GPU,
    so a failing smi must not suppress the 'No GPU detected ... CPU mode only'
    line."""
    monkeypatch.setattr(cli, "find_binary_dir", lambda: None)
    monkeypatch.setattr(doctor_mod, "_check_native_abi", lambda: None)
    monkeypatch.setattr(doctor_mod, "_check_vram_torch", lambda: False)
    monkeypatch.setattr(doctor_mod, "_check_hf_backend_usable", lambda *a, **k: None)
    _smi(monkeypatch, returncode=9, stdout=_NVML_ERROR)

    out = cli_runner.invoke(cli.doctor, []).output

    assert "CPU mode only" in out, out
    assert "installed but failed" in out, out


def test_a_working_smi_still_suppresses_the_cpu_only_verdict(
        cli_runner, monkeypatch):
    """The other direction: a genuinely working driver is positive proof of a
    GPU and still vetoes the CPU-only line."""
    monkeypatch.setattr(cli, "find_binary_dir", lambda: None)
    monkeypatch.setattr(doctor_mod, "_check_native_abi", lambda: None)
    monkeypatch.setattr(doctor_mod, "_check_vram_torch", lambda: False)
    monkeypatch.setattr(doctor_mod, "_check_hf_backend_usable", lambda *a, **k: None)
    _smi(monkeypatch, returncode=0, stdout="NVIDIA GeForce RTX 4090, 24564 MiB")

    out = cli_runner.invoke(cli.doctor, []).output

    assert "No GPU detected" not in out, out
    assert "NVIDIA GeForce RTX 4090" in out, out
