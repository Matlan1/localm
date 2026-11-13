# SPDX-License-Identifier: AGPL-3.0-or-later
"""The install step after the PyTorch install (the ``hf`` and ``audio`` extras)
is held to the torch, torchvision and torchaudio versions the PyTorch step left,
in every setup variant and in the updater, through one constraints file
(``hwdetect.TORCH_PINS_FILE``) written by ``python -m localm.hwdetect torch-pins``."""

from __future__ import annotations

from pathlib import Path

import pytest

from localm import hwdetect, updater, updater_components as uc

ROOT = Path(__file__).resolve().parent.parent


def test_torch_pins_names_each_installed_torch_package_exactly_in_order():
    assert hwdetect.torch_pins({"torchaudio": "2.11.0+rocm7.13.0", "numpy": "2.4.4",
                                "torch": "2.11.0+rocm7.13.0"}) == [
        "torch==2.11.0+rocm7.13.0", "torchaudio==2.11.0+rocm7.13.0"]
    assert hwdetect.torch_pins({"torch": "", "torchvision": None}) == []
    assert hwdetect.torch_pins({}) == []


def test_torch_pins_writes_the_constraints_file(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(hwdetect, "torch_pins", lambda installed=None: ["torch==2.11.0+cu126"])
    out = tmp_path / "pins.txt"
    assert hwdetect.main(["torch-pins", "--out", str(out)]) == 0
    assert out.read_text(encoding="utf-8") == "torch==2.11.0+cu126\n"
    assert capsys.readouterr().out == ""
    monkeypatch.setattr(hwdetect, "torch_pins", lambda installed=None: [])
    assert hwdetect.main(["torch-pins", "--out", str(out)]) == 0
    assert out.read_text(encoding="utf-8") == ""
    assert hwdetect.main(["torch-pins", "--out", str(tmp_path / "missing" / "p.txt")]) == 1
    assert "could not write" in capsys.readouterr().err
    assert hwdetect.main(["torch-pins", "--out"]) == 2


def _second_step(text: str, extras: str) -> tuple:
    """(the line before, the line of) the install of *extras* in a setup script."""
    lines = [ln.strip() for ln in text.splitlines()]
    hits = [i for i, ln in enumerate(lines)
            if ln.startswith("uv pip install") and f'-e ".[{extras}]"' in ln]
    assert len(hits) == 1, hits
    i = hits[0]
    before = next(lines[j] for j in range(i - 1, -1, -1) if lines[j])
    return before, lines[i]


@pytest.mark.parametrize("script, python, pins", [
    ("setup.bat", r".venv\Scripts\python", r'".venv\localm-torch-pins.txt"'),
    ("setup.sh", ".venv/bin/python", ".venv/localm-torch-pins.txt"),
])
def test_every_console_setup_holds_the_hf_install_to_the_installed_torch(script, python,
                                                                         pins):
    assert pins.strip('"').replace("\\", "/") == hwdetect.TORCH_PINS_FILE
    before, line = _second_step((ROOT / script).read_text(encoding="utf-8"), "hf,audio")
    assert before == f"{python} -m localm.hwdetect torch-pins --out {pins}", before
    assert f" -c {pins}" in line, line


def test_the_gui_installer_names_the_same_pins_file():
    text = (ROOT / "installer" / "gui.py").read_text(encoding="utf-8")
    assert f'TORCH_PINS_FILE = "{hwdetect.TORCH_PINS_FILE}"' in text


@pytest.fixture
def install(tmp_path):
    root = tmp_path / "inst"
    (root / ".venv" / "Lib" / "site-packages").mkdir(parents=True)
    return root


def _dist(root: Path, name: str, version: str) -> None:
    d = root / ".venv" / "Lib" / "site-packages" / f"{name}-{version}.dist-info"
    d.mkdir()
    (d / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n",
                                encoding="utf-8")


def test_the_updater_holds_its_hf_install_to_the_torch_its_first_step_left(install,
                                                                          monkeypatch):
    monkeypatch.setattr(updater, "_uv_for_deps", lambda root: "uv")
    _reqs, commands, _line = uc.torch_step(ROOT, "torch==2.11.0+rocm7.13.0 --index-url https://x/")
    assert commands[1][-2:] == ["-c", hwdetect.TORCH_PINS_FILE]
    pins = install / hwdetect.TORCH_PINS_FILE
    gen = uc._commands({"name": "pytorch", "items": commands}, install)
    first = next(gen)
    assert first[1:4] == ["pip", "install", "-p"] and not pins.exists()
    _dist(install, "torch", "2.11.0+rocm7.13.0")
    _dist(install, "numpy", "2.4.4")
    second = next(gen)
    assert second[-2:] == ["-c", hwdetect.TORCH_PINS_FILE]
    assert pins.read_text(encoding="utf-8") == "torch==2.11.0+rocm7.13.0\n"
    assert list(gen) == []


def test_the_updater_writes_an_empty_pins_file_when_no_torch_is_installed(install,
                                                                         monkeypatch):
    monkeypatch.setattr(updater, "_uv_for_deps", lambda root: "uv")
    _reqs, commands, _line = uc.torch_step(ROOT, "torch torchvision --torch-backend=cpu")
    assert len(list(uc._commands({"name": "pytorch", "items": commands}, install))) == 2
    assert (install / hwdetect.TORCH_PINS_FILE).read_text(encoding="utf-8") == ""
