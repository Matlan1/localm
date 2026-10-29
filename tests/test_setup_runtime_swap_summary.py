# SPDX-License-Identifier: AGPL-3.0-or-later
"""Every setup variant says in its summary when the runtime installed is not
the one picked (setup-llama fell back, e.g. vulkan that does not load -> cpu).

Each check runs the real block: setup.sh's under bash, setup.bat's under
cmd.exe, and the window's provisioning step through its own module, against a
real runtime marker file.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

from tests.test_setup_gpu_detect import _bash

ROOT = Path(__file__).resolve().parents[1]
MARKER = Path("runtime") / "localm_llama_runtime" / "lib" / ".localm-backend"


def _block(name: str, start: str, end: str) -> str:
    text = (ROOT / name).read_text(encoding="utf-8").replace("\r\n", "\n")
    a = text.index(start)
    b = text.index(end, a) + len(end)
    return text[a:b]


def _marker(folder: Path, backend: str) -> None:
    p = folder / MARKER
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(f"{backend} b11928\n", encoding="utf-8")


CASES = [  # (picked, installed, runtime_ok, swap reported)
    ("vulkan", "cpu", "1", True),
    ("vulkan", "vulkan", "1", False),
    ("own", "cpu", "1", False),
    ("vulkan", "cpu", "0", False),
    ("vulkan", None, "1", False),
]


@pytest.mark.skipif(_bash() is None, reason="no bash on PATH")
@pytest.mark.parametrize("picked,installed,ok,swapped", CASES)
def test_setup_sh_reports_a_swapped_runtime(tmp_path, picked, installed, ok, swapped):
    block = _block("setup.sh", "  # The backend setup-llama recorded as installed",
                   "--backend $BACKEND --force\"\n  fi\n")
    if installed:
        _marker(tmp_path, installed)
    script = ("set -euo pipefail\nsay() { printf '%s\\n' \"$*\"; }\n"
              f"RUNTIME_OK={ok}; BACKEND={picked}\n" + block)
    out = subprocess.run([_bash(), "-c", script], cwd=tmp_path, capture_output=True,
                         text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    line = f"The {picked} runtime did not load here; {installed} is installed and in use."
    assert (line in out.stdout) is swapped, out.stdout
    if swapped:
        assert f".venv/bin/localm setup-llama --backend {picked} --force" in out.stdout


@pytest.mark.skipif(sys.platform != "win32", reason="runs a setup.bat block under cmd.exe")
@pytest.mark.parametrize("picked,installed,ok,swapped", CASES)
def test_setup_bat_reports_a_swapped_runtime(tmp_path, picked, installed, ok, swapped):
    block = _block("setup.bat", "rem  The backend setup-llama recorded as installed",
                   "--backend %BACKEND% --force\n")
    if installed:
        _marker(tmp_path, installed)
    driver = tmp_path / "driver.bat"
    driver.write_bytes(("@echo off\nsetlocal EnableDelayedExpansion\n"
                        f'set "RUNTIME_OK={ok}"\nset "BACKEND={picked}"\n'
                        + block + "exit /b 0\n").replace("\n", "\r\n").encode("utf-8"))
    out = subprocess.run(["cmd.exe", "/d", "/c", str(driver)], cwd=tmp_path,
                         capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    line = f"The {picked} runtime did not load here; {installed} is installed and in use."
    assert (line in out.stdout) is swapped, out.stdout
    if swapped:
        assert f"[!] {line}" in out.stdout, out.stdout
        assert f"setup-llama --backend {picked} --force" in out.stdout


@pytest.fixture()
def gui(tmp_path, monkeypatch):
    path = ROOT / "installer" / "gui.py"
    s = importlib.util.spec_from_file_location("localm_installer_gui_swap", path)
    mod = importlib.util.module_from_spec(s)
    sys.modules["localm_installer_gui_swap"] = mod
    s.loader.exec_module(mod)
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    return mod


@pytest.mark.parametrize("installed,swapped", [("cpu", True), ("vulkan", False)])
def test_the_window_reports_a_swapped_runtime(gui, tmp_path, monkeypatch, installed, swapped):
    def fake_run(cmd, emit, plan, **kw):
        _marker(tmp_path, installed)
        return 0
    monkeypatch.setattr(gui, "_run", fake_run)
    step = [s for s in gui.build_steps(gui.Plan(backend="vulkan"))
            if s.label.startswith("Provisioning the vulkan")][0]
    failures, stopped = gui.run_steps([step], lambda s: None)
    assert stopped is None
    if swapped:
        assert failures and "vulkan runtime did not load here; cpu is installed" in failures[0]
    else:
        assert failures == []
