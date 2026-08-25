# SPDX-License-Identifier: AGPL-3.0-or-later
"""doctor's smi probe: a FAILING vendor tool is not a working GPU (ADR-0008).

`_check_gpu_driver` read `subprocess.run(...).stdout` and never looked at the
return code, so any non-empty stdout became BOTH a green tick and the reported
device name. nvidia-smi and rocm-smi report a broken driver by printing an error
and exiting non-zero, so the check answered "did the tool print something" while
the reader heard "you have a working GPU driver" - the R2/R3 shape, diverging
precisely in the case worth reporting.

The downstream cost is the worse half and has its own test below: this return
value feeds `smi_or_torch_gpu`, and `_check_gpu_verdict`'s step (3) returns EARLY
when it is True. So a broken driver ALSO suppressed the "No GPU detected ... CPU
mode only" line - the check that would have told the user something is wrong.

Absent and broken are deliberately NOT treated alike: absent is the normal case
on nearly every box and stays silent, while present-and-failing is surfaced,
because only the second is actionable and they need opposite responses.
"""

from __future__ import annotations

import importlib
import subprocess

import localm.cli as cli

doctor_mod = importlib.import_module("localm.cli.doctor")

# The nvidia-smi wording for a driver updated without a reboot: a plausible-
# looking line that a stdout-only check renders as a device NAME.
_NVML_ERROR = "Failed to initialize NVML: Driver/library version mismatch"


def _smi(monkeypatch, *, tool="nvidia-smi", returncode=0, stdout="", stderr=""):
    """Make exactly one vendor tool answer; every other command looks absent.

    Patches subprocess.run wholesale, as the sibling doctor tests already do -
    which also makes doctor's venv-creation probe report an error. That is
    pre-existing, noisy rather than wrong, and irrelevant to the lines asserted
    here."""
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
    """The visible defect: a green tick reading 'NVIDIA GPU: Failed to
    initialize NVML...'. The error may appear (see the next test), but never as
    the thing the OK line names."""
    _smi(monkeypatch, returncode=9, stdout=_NVML_ERROR)

    doctor_mod._check_gpu_driver()
    out = capsys.readouterr().out

    assert "NVIDIA GPU:" not in out, out


def test_a_failing_smi_is_surfaced_not_silenced(monkeypatch, capsys):
    """Rule 5: surface it. 'Installed but broken' is actionable and must not be
    collapsed into the same silence as 'not installed'."""
    _smi(monkeypatch, returncode=9, stdout=_NVML_ERROR)

    doctor_mod._check_gpu_driver()
    out = capsys.readouterr().out

    assert "nvidia-smi" in out
    assert "installed but failed" in out
    assert "Driver/library version mismatch" in out


def test_a_tool_that_fails_with_only_stderr_still_reports_what_it_said(
        monkeypatch, capsys):
    """Some builds put the reason on stderr instead. The fallback must reach for
    it rather than printing a bare exit code the user cannot act on."""
    _smi(monkeypatch, returncode=1, stdout="", stderr="NVML library not found")

    doctor_mod._check_gpu_driver()
    out = capsys.readouterr().out

    assert "NVML library not found" in out


def test_a_clean_smi_is_still_reported_as_a_gpu(monkeypatch, capsys):
    """No regression: exit 0 with a device name is exactly what this check is
    for, and must still return True and print the name."""
    _smi(monkeypatch, returncode=0, stdout="NVIDIA GeForce RTX 4090, 24564 MiB")

    assert doctor_mod._check_gpu_driver() is True
    out = capsys.readouterr().out
    assert "NVIDIA GeForce RTX 4090" in out
    assert "installed but failed" not in out


def test_an_absent_tool_stays_silent(monkeypatch, capsys):
    """The normal case on nearly every box. Not a fault, so not a warning - the
    default GPU paths (Vulkan/Metal/bundled-ROCm) never need these tools."""
    _smi(monkeypatch, tool="__nothing_matches__")

    assert doctor_mod._check_gpu_driver() is False
    out = capsys.readouterr().out
    assert "installed but failed" not in out
    assert "nvidia-smi" not in out


def test_a_clean_exit_with_no_output_is_not_a_gpu(monkeypatch, capsys):
    """Exit 0 and nothing to say proves nothing. It must not become a tick with
    an empty device name."""
    _smi(monkeypatch, returncode=0, stdout="   \n")

    assert doctor_mod._check_gpu_driver() is False
# --------------------------------------------------------------------------- #
#  Effect on the CPU-only verdict                                              #
# --------------------------------------------------------------------------- #

def test_a_failing_smi_does_not_suppress_the_cpu_only_verdict(
        cli_runner, monkeypatch):
    """The load-bearing one. _check_gpu_verdict's step (3) returns EARLY when
    smi/torch claim a GPU, so a broken driver counted as a GPU ALSO swallowed the
    'No GPU detected ... CPU mode only' line. The user then had a machine with no
    working GPU, a doctor showing a green NVIDIA tick, and no warning anywhere."""
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
    """The other direction, so the fix cannot be 'always warn'. A genuinely
    working driver is positive proof of a GPU and must still veto the CPU-only
    line, exactly as before."""
    monkeypatch.setattr(cli, "find_binary_dir", lambda: None)
    monkeypatch.setattr(doctor_mod, "_check_native_abi", lambda: None)
    monkeypatch.setattr(doctor_mod, "_check_vram_torch", lambda: False)
    monkeypatch.setattr(doctor_mod, "_check_hf_backend_usable", lambda *a, **k: None)
    _smi(monkeypatch, returncode=0, stdout="NVIDIA GeForce RTX 4090, 24564 MiB")

    out = cli_runner.invoke(cli.doctor, []).output

    assert "No GPU detected" not in out, out
    assert "NVIDIA GeForce RTX 4090" in out, out
