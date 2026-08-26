# SPDX-License-Identifier: AGPL-3.0-or-later
"""setup.sh's detect_gpu(): NVIDIA must win over leftover ROCm tooling.

Checking rocminfo/rocm-smi//opt/rocm BEFORE nvidia-smi makes a box with an
NVIDIA GPU but ALSO some ROCm tooling on PATH (a shared ML rig, a base image
bundling both vendor stacks) report "rocm" - matching neither hwdetect.py's
vendor priority ("nvidia", "amd", "intel") nor the actual hardware. The real
backend recommendation comes from `python -m localm.hwdetect` further down in
setup.sh; $GPU from detect_gpu() only becomes load-bearing as setup.sh's OWN
fallback if that probe's output is unparseable - see the
`case "$REC" in ... *) case "$GPU" in ...` guard.

Extracts and runs ONLY the detect_gpu() function body (not the whole script,
which has side effects and prompts) with a controlled PATH holding stub
commands - no real GPU hardware needed either way.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SETUP_SH = ROOT / "setup.sh"


def _bash() -> str | None:
    return shutil.which("bash")


def _function_body(name: str) -> str:
    src = SETUP_SH.read_text(encoding="utf-8")
    start = src.index(f"{name}() {{")
    end = src.index("\n}\n", start) + len("\n}\n")
    return src[start:end]


def _make_stub(bin_dir: Path, name: str) -> None:
    stub = bin_dir / name
    stub.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


def _run_detect_gpu(tmp_path: Path, *, stub_names: list[str]) -> str:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in stub_names:
        _make_stub(bin_dir, name)
    script = _function_body("detect_gpu") + "\ndetect_gpu\n"
    env = dict(os.environ)
    env["PATH"] = str(bin_dir)   # isolate: no real nvidia-smi/rocminfo must leak in
    result = subprocess.run([_bash(), "-c", script], capture_output=True,
                            text=True, env=env, timeout=15)
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


pytestmark = pytest.mark.skipif(_bash() is None, reason="no bash on PATH")


def test_nvidia_wins_over_leftover_rocm_tooling(tmp_path):
    # rocminfo present ALONGSIDE nvidia-smi must still detect nvidia, not rocm.
    assert _run_detect_gpu(tmp_path, stub_names=["nvidia-smi", "rocminfo"]) == "cuda"


def test_nvidia_wins_over_rocm_smi_too(tmp_path):
    assert _run_detect_gpu(tmp_path, stub_names=["nvidia-smi", "rocm-smi"]) == "cuda"


def test_rocm_alone_is_still_detected(tmp_path):
    assert _run_detect_gpu(tmp_path, stub_names=["rocminfo"]) == "rocm"


def test_nvidia_alone_is_still_detected(tmp_path):
    assert _run_detect_gpu(tmp_path, stub_names=["nvidia-smi"]) == "cuda"


def test_neither_is_cpu(tmp_path):
    assert _run_detect_gpu(tmp_path, stub_names=[]) == "cpu"


def test_nvidia_checked_before_rocm_in_source():
    """Static backstop that holds even without bash on PATH: the nvidia-smi
    check must appear (and therefore short-circuit) before the rocm probes."""
    body = _function_body("detect_gpu")
    assert body.index("nvidia-smi") < body.index("rocminfo"), (
        "detect_gpu() must check nvidia-smi before rocminfo/rocm-smi/opt-rocm")


# ---------------------------------------------------------------------------
# detect_gpu()'s pre-venv guess must not be presented as its own detection
# verdict.
#
# setup.sh has two things that can each claim to say what GPU acceleration is in
# play: this bash-level detect_gpu(), which runs before the venv exists, and the
# authoritative Recommended for your hardware line, sourced from
# `python -m localm.hwdetect` once the venv exists. $GPU's only job is gating the
# pre-venv Y/n prompt, plus setup.sh's own fallback if the hwdetect probe fails.
# ---------------------------------------------------------------------------


def _pre_venv_gpu_block() -> str:
    src = SETUP_SH.read_text(encoding="utf-8")
    start = src.index("# ---- detect GPU acceleration")
    end = src.index("# ---- create the venv", start)
    return src[start:end]


def _run_pre_venv_gpu_block(tmp_path: Path, *, stub_names: list[str], yes: bool) -> str:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in stub_names:
        _make_stub(bin_dir, name)
    # Minimal stand-ins for setup.sh's own say()/ask(), which the extracted
    # block calls but which are defined earlier in the real script (outside
    # this slice). ask() always echoes its default here, matching --yes mode.
    script = (
        'say() { printf "%s\\n" "$*"; }\n'
        'ask() { echo "$2"; }\n'
        f'YES={"1" if yes else "0"}\n'
        + _pre_venv_gpu_block()
        + '\nprintf "GPU=%s\\n" "$GPU"\n'
    )
    env = dict(os.environ)
    env["PATH"] = str(bin_dir)   # isolate: no real nvidia-smi/rocminfo must leak in
    result = subprocess.run([_bash(), "-c", script], capture_output=True,
                            text=True, env=env, timeout=15)
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_pre_venv_block_never_prints_a_detection_verdict_line(tmp_path):
    out = _run_pre_venv_gpu_block(tmp_path, stub_names=["rocminfo"], yes=True)
    assert "Detected acceleration" not in out
    assert "GPU=rocm" in out


def test_pre_venv_block_gpu_var_still_correct_for_the_probe_failed_fallback(tmp_path):
    # $GPU still has to be right even though it is not printed as a verdict:
    # setup.sh's own fallback on $REC falls through to a case on $GPU if the
    # hwdetect probe's output is unparseable.
    out = _run_pre_venv_gpu_block(tmp_path, stub_names=["nvidia-smi", "rocminfo"], yes=True)
    assert "GPU=cuda" in out


def test_authoritative_recommendation_still_sourced_from_hwdetect():
    """Static lock-in: the one line users should trust as a verdict must stay
    driven by python -m localm.hwdetect, not by detect_gpu()."""
    src = SETUP_SH.read_text(encoding="utf-8")
    assert 'Recommended for your hardware: $REC' in src
    assert "localm.hwdetect" in src


def test_detect_gpu_guess_not_printed_as_its_own_verdict_line_in_source():
    """Static backstop that holds even without bash on PATH."""
    src = SETUP_SH.read_text(encoding="utf-8")
    assert 'say "  Detected acceleration: $GPU"' not in src
