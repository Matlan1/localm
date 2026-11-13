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


def test_torch_pins_reads_this_interpreters_distributions(monkeypatch):
    from importlib import metadata
    versions = {"torch": "2.11.0+cu126", "torchaudio": "2.11.0+cu126"}

    def version(name):
        if name not in versions:
            raise metadata.PackageNotFoundError(name)
        return versions[name]
    monkeypatch.setattr(metadata, "version", version)
    assert hwdetect.torch_pins() == ["torch==2.11.0+cu126", "torchaudio==2.11.0+cu126"]

    def unreadable(name):
        raise ValueError("bad METADATA")
    monkeypatch.setattr(metadata, "version", unreadable)
    with pytest.raises(ValueError):
        hwdetect.torch_pins()


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


def _logical_lines(text: str) -> list:
    """The lines of a setup script, with shell continuation lines joined."""
    out, cur = [], ""
    for ln in text.splitlines():
        ln = ln.strip()
        if ln.endswith("\\"):
            cur += ln[:-1].strip() + " "
            continue
        out.append(cur + ln)
        cur = ""
    return out


@pytest.mark.parametrize("script, python, pins", [
    ("setup.bat", r".venv\Scripts\python", r'".venv\localm-torch-pins.txt"'),
    ("setup.sh", ".venv/bin/python", ".venv/localm-torch-pins.txt"),
])
def test_every_console_setup_holds_the_hf_install_to_the_installed_torch(script, python,
                                                                         pins):
    assert pins.strip('"').replace("\\", "/") == hwdetect.TORCH_PINS_FILE
    text = (ROOT / script).read_text(encoding="utf-8")
    lines = (_logical_lines(text) if script.endswith(".sh")
             else [ln.strip() for ln in text.splitlines()])
    hits = [ln for ln in lines if 'uv pip install -p .venv -e ".[hf,audio]"' in ln]
    assert len(hits) == 1, hits
    step = (f"{python} -m localm.hwdetect torch-pins --out {pins} && "
            f'uv pip install -p .venv -e ".[hf,audio]" -c {pins} ||')
    assert hits[0].startswith(step), hits[0]


def test_the_gui_installer_names_the_same_pins_file():
    text = (ROOT / "installer" / "gui.py").read_text(encoding="utf-8")
    assert f'TORCH_PINS_FILE = "{hwdetect.TORCH_PINS_FILE}"' in text


@pytest.fixture
def install(tmp_path):
    root = tmp_path / "inst"
    (root / ".venv" / "Lib" / "site-packages").mkdir(parents=True)
    (root / "pyproject.toml").write_text('[project]\nname = "x"\nversion = "1"\n',
                                         encoding="utf-8")
    return root


def _dist(root: Path, name: str, version: str) -> None:
    d = root / ".venv" / "Lib" / "site-packages" / f"{name}-{version}.dist-info"
    d.mkdir()
    (d / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n",
                                encoding="utf-8")


def test_the_updater_holds_its_hf_install_to_the_torch_its_first_step_left(install,
                                                                          monkeypatch):
    monkeypatch.setattr(updater, "_uv_for_deps", lambda root: "uv")
    _reqs, commands, _line = uc.torch_step(install, "torch==2.11.0+rocm7.13.0 --index-url https://x/")
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
    _reqs, commands, _line = uc.torch_step(install, "torch torchvision --torch-backend=cpu")
    assert len(list(uc._commands({"name": "pytorch", "items": commands}, install))) == 2
    assert (install / hwdetect.TORCH_PINS_FILE).read_text(encoding="utf-8") == ""
