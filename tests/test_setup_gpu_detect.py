# SPDX-License-Identifier: AGPL-3.0-or-later
"""setup.sh's detect_gpu(): NVIDIA must win over leftover ROCm tooling.

Regression for a filed ordering bug: detect_gpu() checked rocminfo/rocm-smi/
/opt/rocm BEFORE nvidia-smi, so a box with an NVIDIA GPU but ALSO some ROCm
tooling on PATH (a shared ML rig, a base image bundling both vendor stacks)
reported "rocm" - matching neither hwdetect.py's vendor priority
("nvidia", "amd", "intel") nor the actual hardware. The real backend
recommendation comes from `python -m localm.hwdetect` further down in
setup.sh, which gets this right; $GPU from detect_gpu() only becomes
load-bearing as setup.sh's OWN fallback if that probe's output is
unparseable - see the `case "$REC" in ... *) case "$GPU" in ...` guard.

Extracts and runs ONLY the detect_gpu() function body (not the whole
script, which has side effects and prompts) with a controlled PATH holding
stub commands - no real GPU hardware needed either way.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
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
    # The filed bug, reproduced directly: rocminfo present ALONGSIDE nvidia-smi
    # must still detect nvidia, not rocm.
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
