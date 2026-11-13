# SPDX-License-Identifier: AGPL-3.0-or-later
"""The AMD Windows PyTorch pins live in localm/hwdetect.py, one set per GPU family.

Every other file that names a ``torch``/``torchvision``/``torchaudio`` ROCm build names
one of those, every setup variant asks ``hwdetect torch-args`` for the wheel source, and
the managed ComfyUI installs the same wheels.
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

from localm import hwdetect

ROOT = Path(__file__).resolve().parent.parent

_SCANNED = ("localm", "docs", "scripts", "installer", "docker")
_SCANNED_FILES = ("README.md", "pyproject.toml", "uv.lock", "setup.bat", "setup.sh",
                  "setup-gui.bat", "setup-gui.sh")
_SUFFIXES = {".py", ".md", ".toml", ".lock", ".bat", ".sh", ".txt", ".yml", ".yaml", ".cfg"}
_ROCM_BUILD = re.compile(
    r"\b(torch|torchvision|torchaudio)(?:==|-)(\d[\d.]*)(?:\+|%2[Bb])(rocm\d[\d.]*\d)")


def _pins() -> set:
    pins = set()
    for req in (*hwdetect.amd_rocm_win_torch_packages(),
                *hwdetect.amd_gfx103x_win_torch_packages()):
        name, _, version = req.partition("==")
        pins.add((name, version))
    return pins


def _scanned_paths():
    for rel in _SCANNED_FILES:
        yield ROOT / rel
    for top in _SCANNED:
        for path in sorted((ROOT / top).rglob("*")):
            if path.is_file() and path.suffix in _SUFFIXES and "__pycache__" not in path.parts:
                yield path


def test_every_amd_windows_torch_pin_matches_hwdetect():
    pins = _pins()
    found, wrong = 0, []
    for path in _scanned_paths():
        text = path.read_text(encoding="utf-8", errors="replace")
        for lineno, line in enumerate(text.splitlines(), 1):
            for name, base, rocm in _ROCM_BUILD.findall(line):
                found += 1
                if (name, f"{base}+{rocm}") not in pins:
                    wrong.append(f"{path.relative_to(ROOT).as_posix()}:{lineno}: "
                                 f"{name} {base}+{rocm}")
    assert found >= 6, "the scan found fewer ROCm builds than hwdetect itself names"
    assert not wrong, ("these name an AMD torch build hwdetect.py does not pin: "
                       + "; ".join(wrong))


def test_the_gpu_extra_and_its_wheel_sources_pin_the_hwdetect_gfx103x_wheels():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extra = {r.split(";")[0].strip() for r in data["project"]["optional-dependencies"]["gpu"]}
    sources = data["tool"]["uv"]["sources"]
    for req in hwdetect.amd_gfx103x_win_torch_packages():
        name, _, version = req.partition("==")
        assert req in extra, f"the gpu extra does not pin {req}"
        urls = [s["url"] for s in sources[name] if "url" in s]
        assert [u for u in urls if u.startswith(hwdetect.amd_gfx103x_win_index())
                and f"/{name}-{version.replace('+', '%2B')}-" in u], (name, urls)


def test_the_managed_comfyui_installs_the_hwdetect_gfx103x_wheels(monkeypatch):
    from localm.media import managed_comfy_fresh as fresh
    monkeypatch.setattr(sys, "platform", "win32")
    det = SimpleNamespace(vendors=["amd"], gpu_names=["AMD Radeon RX 6800"],
                          gpu_state="found", probe_ok=True)
    spec = fresh.comfy_torch_spec(det)
    assert spec.variant == "rocm"
    assert spec.packages[:3] == hwdetect.amd_gfx103x_win_torch_packages()
    assert spec.extra_index_url == hwdetect.amd_gfx103x_win_index()


def test_the_rx_7000_9000_instructions_in_the_docs_name_the_hwdetect_wheels():
    doc = (ROOT / "docs" / "gpu-setup.md").read_text(encoding="utf-8")
    torch, torchvision, _ = hwdetect.amd_rocm_win_torch_packages()
    assert torch in doc and torchvision in doc and hwdetect.amd_rocm_win_find_links() in doc


@pytest.mark.parametrize("rel", ["setup.bat", "setup.sh", "installer/gui.py"])
def test_every_setup_variant_asks_hwdetect_for_the_torch_wheel_source(rel):
    text = (ROOT / rel).read_text(encoding="utf-8")
    assert re.search(r"localm\.hwdetect[\"',\s]+torch-args", text), rel


@pytest.mark.parametrize("rel", ["setup-gui.bat", "setup-gui.sh"])
def test_the_gui_setup_variants_hand_the_install_to_the_gui_installer(rel):
    text = (ROOT / rel).read_text(encoding="utf-8")
    assert re.search(r"installer[\\/]gui\.py", text), rel
