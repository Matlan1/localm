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


def _mark_executable(path: Path) -> None:
    """os.chmod()/Path.chmod() cannot set a real POSIX execute bit on NTFS -
    it silently no-ops the S_IEXEC/S_IXGRP/S_IXOTH bits, which some Git-Bash/
    MSYS builds require before they will invoke a shebang script via command
    substitution (others are lenient about it, which is why this was never
    caught locally). Route through bash's own chmod, which does set a bit
    MSYS itself recognizes."""
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    subprocess.run(["chmod", "+x", str(path)], check=True)


def _make_stub(bin_dir: Path, name: str) -> None:
    stub = bin_dir / name
    stub.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    _mark_executable(stub)


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


# ---------------------------------------------------------------------------
# Apple Silicon: detect_gpu() must recognize Darwin/arm64 so an Apple Silicon
# machine is actually offered the metal backend through the normal
# automatic/interactive path, instead of always falling through to "cpu"
# (detect_gpu() had no Darwin branch at all - every real Mac silently landed
# on the CPU-only build, with the correct hwdetect.py "metal" recommendation
# clobbered by the "$GPU = cpu -> REC=cpu" guard further down in setup.sh).
# ---------------------------------------------------------------------------


def _make_uname_stub(bin_dir: Path, *, os_name: str, arch: str) -> None:
    stub = bin_dir / "uname"
    stub.write_text(
        "#!/bin/sh\n"
        f'if [ "$1" = "-s" ]; then echo {os_name}; fi\n'
        f'if [ "$1" = "-m" ]; then echo {arch}; fi\n',
        encoding="utf-8",
    )
    _mark_executable(stub)


def _run_detect_gpu_with_uname(tmp_path: Path, *, os_name: str, arch: str,
                                stub_names: list[str] | None = None) -> str:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in stub_names or []:
        _make_stub(bin_dir, name)
    _make_uname_stub(bin_dir, os_name=os_name, arch=arch)
    script = _function_body("detect_gpu") + "\ndetect_gpu\n"
    env = dict(os.environ)
    env["PATH"] = str(bin_dir)   # isolate: no real nvidia-smi/rocminfo/uname must leak in
    result = subprocess.run([_bash(), "-c", script], capture_output=True,
                            text=True, env=env, timeout=15)
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_apple_silicon_is_metal(tmp_path):
    assert _run_detect_gpu_with_uname(tmp_path, os_name="Darwin", arch="arm64") == "metal"


def test_intel_mac_is_cpu_not_metal(tmp_path):
    # Matches hwdetect.py's own policy: Intel Macs get "cpu", not metal - the
    # official llama.cpp macOS Metal build targets Apple Silicon.
    assert _run_detect_gpu_with_uname(tmp_path, os_name="Darwin", arch="x86_64") == "cpu"


def test_linux_uname_never_mistaken_for_darwin(tmp_path):
    assert _run_detect_gpu_with_uname(tmp_path, os_name="Linux", arch="x86_64") == "cpu"


def test_nvidia_still_wins_over_a_darwin_arm64_uname(tmp_path):
    # Priority order: a real vendor tool detected earlier in detect_gpu() must
    # still win even if uname would otherwise match the Darwin/arm64 branch
    # (not reachable on real hardware - Apple Silicon has no nvidia-smi - but
    # this pins that the Darwin check is an elif appended AFTER the vendor
    # checks, not a check that could ever shadow them).
    assert _run_detect_gpu_with_uname(
        tmp_path, os_name="Darwin", arch="arm64", stub_names=["nvidia-smi"]
    ) == "cuda"


def test_darwin_branch_checked_after_vendor_tools_in_source():
    """Static backstop that holds even without bash on PATH."""
    body = _function_body("detect_gpu")
    assert body.index("nvidia-smi") < body.index('"Darwin"')
    assert body.index("rocminfo") < body.index('"Darwin"')


# ---------------------------------------------------------------------------
# The "probe failed" fallback (fires only if `python -m localm.hwdetect`
# itself produced no known backend token) must not recommend "vulkan" for a
# GPU value of "metal" - localm has no vulkan build for darwin at all (see
# setup_llama.py's _BACKEND_ASSETS), so that fallback would recommend a
# backend nothing can ever provision on Apple Silicon.
# ---------------------------------------------------------------------------


def _rec_fallback_case() -> str:
    src = SETUP_SH.read_text(encoding="utf-8")
    marker = 'case "$REC" in\n  vulkan|cuda|hip|cpu|metal|amd-rocm) ;;'
    start = src.index(marker)
    end = src.index("esac\n", start) + len("esac\n")
    return src[start:end]


def _run_rec_fallback(tmp_path: Path, *, gpu: str, rec: str = "") -> str:
    script = f'REC="{rec}"\nGPU="{gpu}"\n' + _rec_fallback_case() + '\nprintf "REC=%s\\n" "$REC"\n'
    result = subprocess.run([_bash(), "-c", script], capture_output=True,
                            text=True, cwd=tmp_path, timeout=15)
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_probe_failed_fallback_metal_gpu_stays_metal(tmp_path):
    assert _run_rec_fallback(tmp_path, gpu="metal") == "REC=metal"


def test_probe_failed_fallback_cpu_gpu_stays_cpu(tmp_path):
    assert _run_rec_fallback(tmp_path, gpu="cpu") == "REC=cpu"


def test_probe_failed_fallback_other_gpu_is_vulkan(tmp_path):
    assert _run_rec_fallback(tmp_path, gpu="cuda") == "REC=vulkan"


def test_probe_succeeded_metal_is_trusted_directly(tmp_path):
    # The probe SUCCEEDED (REC is already a known backend) - the case
    # statement's first arm must leave it alone rather than falling into the
    # GPU-based fallback at all.
    assert _run_rec_fallback(tmp_path, gpu="cpu", rec="metal") == "REC=metal"


def test_probe_failed_fallback_metal_case_precedes_generic_vulkan_fallback_in_source():
    """Static backstop that holds even without bash on PATH."""
    body = _rec_fallback_case()
    assert body.index("metal) REC=metal") < body.index("REC=vulkan")


# ---------------------------------------------------------------------------
# The backend-choice menu must offer metal as a real, selectable numbered
# entry on Apple Silicon (not only reachable via the recommended-default
# shortcut [1]) - and must never offer it on a platform with no darwin build
# to provision at all.
# ---------------------------------------------------------------------------


def _backend_menu_block() -> str:
    src = SETUP_SH.read_text(encoding="utf-8")
    start = src.index('_same="   (same as [1])"')
    end = src.index("esac\n", src.index('case "$bpick" in')) + len("esac\n")
    return src[start:end]


def _run_backend_menu(tmp_path: Path, *, rec: str, is_apple_silicon: bool, bpick: str) -> dict:
    script = (
        'say() { :; }\n'
        f'ask() {{ echo "{bpick}"; }}\n'
        f'REC="{rec}"\n'
        f'GPU=irrelevant\n'
        f'IS_APPLE_SILICON={1 if is_apple_silicon else 0}\n'
        + _backend_menu_block()
        + '\nprintf "BACKEND=%s\\n" "$BACKEND"\n'
    )
    result = subprocess.run([_bash(), "-c", script], capture_output=True,
                            text=True, cwd=tmp_path, timeout=15)
    assert result.returncode == 0, result.stderr
    return dict(line.split("=", 1) for line in result.stdout.strip().splitlines() if "=" in line)


def test_menu_bpick_7_selects_metal_on_apple_silicon(tmp_path):
    out = _run_backend_menu(tmp_path, rec="cpu", is_apple_silicon=True, bpick="7")
    assert out["BACKEND"] == "metal"


def test_menu_default_pick_on_apple_silicon_with_rec_metal_is_metal(tmp_path):
    # ask() here returns bpick itself (default "1" simulated directly), so this
    # covers the recommended-default path staying correct too.
    out = _run_backend_menu(tmp_path, rec="metal", is_apple_silicon=True, bpick="1")
    assert out["BACKEND"] == "metal"


def test_menu_shows_metal_line_only_on_apple_silicon(tmp_path):
    script_apple = (
        'say() { printf "%s\\n" "$*"; }\n'
        'ask() { echo 1; }\n'
        'REC="metal"\nGPU=irrelevant\nIS_APPLE_SILICON=1\n'
        + _backend_menu_block()
    )
    script_other = script_apple.replace("IS_APPLE_SILICON=1", "IS_APPLE_SILICON=0")
    out_apple = subprocess.run([_bash(), "-c", script_apple], capture_output=True,
                               text=True, cwd=tmp_path, timeout=15)
    out_other = subprocess.run([_bash(), "-c", script_other], capture_output=True,
                               text=True, cwd=tmp_path, timeout=15)
    assert out_apple.returncode == 0, out_apple.stderr
    assert out_other.returncode == 0, out_other.stderr
    assert "[7] metal" in out_apple.stdout
    assert "[7] metal" not in out_other.stdout


def test_menu_pick_range_extends_to_7_only_on_apple_silicon(tmp_path):
    script_apple = (
        'say() { :; }\n'
        'ask() { printf "%s" "$1" >&2; echo 1; }\n'
        'REC="metal"\nGPU=irrelevant\nIS_APPLE_SILICON=1\n'
        + _backend_menu_block()
    )
    result = subprocess.run([_bash(), "-c", script_apple], capture_output=True,
                            text=True, cwd=tmp_path, timeout=15)
    assert result.returncode == 0, result.stderr
    assert "Pick 1-7" in result.stderr


def test_menu_shows_metal_even_when_gpu_was_downgraded_to_cpu(tmp_path):
    # $GPU can be downgraded to "cpu" by the earlier Y/n prompt (a declined
    # GPU); IS_APPLE_SILICON is a separate flag computed once from uname, so a
    # declined GPU must not also remove the [7] metal choice from the menu.
    script = (
        'say() { printf "%s\\n" "$*"; }\n'
        'ask() { echo 1; }\n'
        'REC="cpu"\nGPU=cpu\nIS_APPLE_SILICON=1\n'
        + _backend_menu_block()
    )
    result = subprocess.run([_bash(), "-c", script], capture_output=True,
                            text=True, cwd=tmp_path, timeout=15)
    assert result.returncode == 0, result.stderr
    assert "[7] metal" in result.stdout
