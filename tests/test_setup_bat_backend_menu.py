# SPDX-License-Identifier: AGPL-3.0-or-later
"""setup.bat lists only the llama.cpp backends setup-llama has a build of.

The backend menu, its prompt range and the "provision later" hint come from
``python -m localm.hwdetect backends``, the same list setup.sh uses. These tests
cut the real blocks out of setup.bat and run them under cmd.exe, so what they
pin is the batch behaviour itself.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BAT = (ROOT / "setup.bat").read_text(encoding="utf-8")

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="runs setup.bat blocks under cmd.exe")

PROBE_START = "rem ---- the backends setup-llama has a build of for this machine"
MENU_START = "rem ---- choose the llama.cpp backend"
MENU_END = "rem ---- PyTorch + transformers for HuggingFace models"
HELPERS_START = "\n:backend_pickrange\n"
HELPERS_END = "rem ===========================================================================\nrem  Setup journal helpers."

X64_ALL = "vulkan cuda amd-rocm sycl cpu"
ARM64 = "vulkan cuda cpu"


def _between(start: str, end: str) -> str:
    text = BAT.replace("\r\n", "\n")
    a = text.index(start)
    return text[a:text.index(end, a)]


def _run(tmp_path: Path, body: str, stdin: str = "\n", env: dict[str, str] | None = None) -> str:
    helpers = _between(HELPERS_START, HELPERS_END)
    script = (
        "@echo off\n"
        "setlocal EnableDelayedExpansion\n"
        'set "STMP=stmp"\n'
        + body
        + "\nexit /b 0\n"
        ":flush\nexit /b 0\n"
        + helpers
    )
    (tmp_path / "stmp").mkdir(exist_ok=True)
    driver = tmp_path / "driver.bat"
    driver.write_bytes(script.replace("\n", "\r\n").encode("utf-8"))
    run_env = {**os.environ, **(env or {})}
    done = subprocess.run(
        ["cmd.exe", "/d", "/c", str(driver)], cwd=tmp_path, input=stdin,
        capture_output=True, text=True, timeout=60, env=run_env, check=False,
    )
    assert done.returncode == 0, done.stderr
    return done.stdout


def _menu(tmp_path: Path, avail: str, rec: str, pick: str = "") -> str:
    body = (
        f'set "REC={rec}"\nset "AVAIL={avail}"\n'
        + _between(MENU_START, MENU_END)
        + "\necho BACKEND=!BACKEND!\n"
    )
    return _run(tmp_path, body, stdin=pick + "\n")


def _listed(out: str) -> list[str]:
    return [line.strip().split("]")[0].lstrip("[") for line in out.splitlines()
            if line.strip().startswith("[") and "]" in line]


def test_x64_menu_lists_every_entry(tmp_path):
    out = _menu(tmp_path, X64_ALL, "vulkan")
    assert _listed(out) == ["1", "2", "3", "4", "5", "6", "7"]
    assert "Pick 1-7 [1]:" in out
    assert "BACKEND=vulkan" in out


def test_arm64_menu_drops_the_backends_with_no_arm64_build(tmp_path):
    out = _menu(tmp_path, ARM64, "vulkan")
    assert _listed(out) == ["1", "2", "3", "6", "7"]
    assert "amd-rocm" not in out
    assert "sycl" not in out
    assert "Pick 1, 2, 3, 6 or 7 [1]:" in out


def test_numbers_stay_fixed_when_entries_are_hidden(tmp_path):
    out = _menu(tmp_path, ARM64, "cpu", pick="6")
    assert "[6] cpu" in out
    assert "BACKEND=cpu" in out
    out = _menu(tmp_path, ARM64, "vulkan", pick="3")
    assert "BACKEND=cuda" in out
    assert "BACKEND=own" in _menu(tmp_path, ARM64, "vulkan", pick="7")


@pytest.mark.parametrize("pick", ["4", "5"])
def test_an_unlisted_number_returns_to_the_recommendation_with_a_message(tmp_path, pick):
    out = _menu(tmp_path, ARM64, "vulkan", pick=pick)
    assert f"[{pick}] is not offered on this machine; using vulkan." in out
    assert "BACKEND=vulkan" in out


def test_a_listed_number_prints_no_unoffered_message(tmp_path):
    out = _menu(tmp_path, X64_ALL, "vulkan", pick="4")
    assert "is not offered" not in out
    assert "BACKEND=amd-rocm" in out


def test_blank_and_junk_picks_take_the_recommendation(tmp_path):
    assert "BACKEND=cuda" in _menu(tmp_path, ARM64, "cuda")
    assert "BACKEND=cuda" in _menu(tmp_path, ARM64, "cuda", pick="zz")


def test_the_recommended_twin_is_marked_on_a_listed_entry(tmp_path):
    out = _menu(tmp_path, ARM64, "cuda")
    assert "[3] cuda" in out
    assert "(same as [1])" in out


def _probe(tmp_path: Path, with_python: bool, arch: str) -> str:
    env = {"PROCESSOR_ARCHITECTURE": arch, "PYTHONPATH": str(ROOT)}
    if with_python:
        scripts = tmp_path / ".venv" / "Scripts"
        scripts.mkdir(parents=True, exist_ok=True)
        shutil.copy2(sys.executable, scripts / "python.exe")
        if (Path(sys.prefix) / "pyvenv.cfg").is_file():
            shutil.copy2(Path(sys.prefix) / "pyvenv.cfg", tmp_path / ".venv" / "pyvenv.cfg")
    body = _between(PROBE_START, MENU_START) + "\necho AVAIL=!AVAIL!\n"
    return _run(tmp_path, body, env=env)


def test_probe_output_becomes_the_available_list(tmp_path):
    from localm import hwdetect

    expected = " ".join(hwdetect.prebuilt_backends())
    assert expected
    assert f"AVAIL={expected}" in _probe(tmp_path, True, "AMD64")


def test_failed_probe_falls_back_by_cpu_architecture(tmp_path):
    assert f"AVAIL={ARM64}" in _probe(tmp_path, False, "ARM64")
    assert f"AVAIL={X64_ALL}" in _probe(tmp_path, False, "AMD64")


def test_provision_later_hint_lists_only_offered_backends(tmp_path):
    body = 'set "CHK= vulkan cuda cpu "\ncall :backend_hint\necho SLB=!SLB!\n'
    assert "SLB=vulkan|cuda|cpu" in _run(tmp_path, body)
    body = 'set "CHK= none "\ncall :backend_hint\necho SLB=!SLB!\n'
    assert "SLB=backend" in _run(tmp_path, body)


def test_the_menu_reads_the_same_list_setup_sh_does():
    assert "localm.hwdetect backends" in BAT
    assert "localm.hwdetect backends" in (ROOT / "setup.sh").read_text(encoding="utf-8")
