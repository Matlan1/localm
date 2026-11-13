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
    for req in hwdetect.amd_rocm_win_torch_packages():
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
    assert found >= 3, "the scan found fewer ROCm builds than hwdetect itself names"
    assert not wrong, ("these name an AMD torch build hwdetect.py does not pin: "
                       + "; ".join(wrong))


def test_the_gpu_extra_and_its_wheel_sources_pin_the_hwdetect_wheels():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extra = {r.split(";")[0].strip() for r in data["project"]["optional-dependencies"]["gpu"]}
    sources = data["tool"]["uv"]["sources"]
    for req in hwdetect.amd_rocm_win_torch_packages():
        name, _, version = req.partition("==")
        assert req in extra, f"the gpu extra does not pin {req}"
        urls = [s["url"] for s in sources[name] if "url" in s]
        assert [u for u in urls if u.startswith(hwdetect.amd_rocm_win_index("gfx103x"))
                and f"/{name}-{version.replace('+', '%2B')}-" in u], (name, urls)


@pytest.mark.parametrize("card, family", [("AMD Radeon RX 6800", "gfx103x"),
                                          ("AMD Radeon RX 7900 XTX", "gfx110x"),
                                          ("AMD Radeon RX 9070 XT", "gfx120x")])
def test_the_managed_comfyui_installs_the_hwdetect_wheels_of_each_family(monkeypatch, card,
                                                                        family):
    from localm.media import managed_comfy_fresh as fresh
    monkeypatch.setattr(sys, "platform", "win32")
    det = hwdetect.Detection(vendors=["amd"], gpu_names=card.lower(), probe_ok=True)
    spec = fresh.comfy_torch_spec(det)
    assert spec.variant == "rocm"
    assert spec.packages == (hwdetect.amd_rocm_win_torch_packages()
                             + hwdetect.amd_rocm_win_sdk_packages(family))
    assert spec.index_url == hwdetect.amd_rocm_win_index(family)


def test_the_rx_7000_9000_instructions_in_the_docs_name_the_hwdetect_wheels():
    doc = (ROOT / "docs" / "gpu-setup.md").read_text(encoding="utf-8")
    torch, torchvision, _ = hwdetect.amd_rocm_win_torch_packages()
    assert torch in doc and torchvision in doc
    for family in ("gfx110x", "gfx120x"):
        assert hwdetect.amd_rocm_win_index(family) in doc
        assert f"rocm-sdk-libraries-{family}-all" in doc


def test_the_wheels_bump_checks_every_family_hwdetect_installs_from():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "bump_amd_wheels_pin_for_family_test", ROOT / "scripts" / "bump_amd_wheels_pin.py")
    bump = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bump)
    families = {bump.INDEX_BASE.rsplit("/", 1)[1], *bump.OTHER_FAMILIES}
    assert {u.rstrip("/").rsplit("/", 1)[1] for u in hwdetect._AMD_ROCM_WIN_INDEX.values()} \
        == families


@pytest.mark.parametrize("rel", ["setup.bat", "setup.sh", "installer/gui.py"])
def test_every_setup_variant_asks_hwdetect_for_the_torch_wheel_source(rel):
    text = (ROOT / rel).read_text(encoding="utf-8")
    assert re.search(r"localm\.hwdetect[\"',\s]+torch-args", text), rel


@pytest.mark.parametrize("rel", ["setup-gui.bat", "setup-gui.sh"])
def test_the_gui_setup_variants_hand_the_install_to_the_gui_installer(rel):
    text = (ROOT / rel).read_text(encoding="utf-8")
    assert re.search(r"installer[\\/]gui\.py", text), rel


_OTHER_SOURCES = ("--extra-index-url", "--find-links", "-f", "--index-strategy")


def _amd_install_args():
    """(where, family, pip arguments) of every AMD Windows PyTorch install: setup's
    torch-args for the families it installs from AMD's index, and the managed
    ComfyUI's for every family."""
    import shlex
    from localm.media import managed_comfy_fresh as fresh
    for card, family in (("AMD Radeon RX 6800", "gfx103x"), ("AMD Radeon RX 7900 XTX", "gfx110x"),
                         ("AMD Radeon RX 9070 XT", "gfx120x")):
        det = hwdetect.Detection(vendors=["amd"], gpu_names=card.lower(), probe_ok=True)
        spec = fresh.comfy_torch_spec(det)
        yield "managed ComfyUI", family, fresh.comfy_torch_install_args(spec)
        args = hwdetect.torch_pip_args("amd-rocm", det)
        if args != "-e .[gpu]":
            yield "torch-args", family, shlex.split(args, posix=True)


def test_every_amd_package_comes_from_its_family_index_alone(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    seen = []
    for where, family, args in _amd_install_args():
        seen.append((where, family))
        flat = " ".join(args)
        assert not any(a == o or a.startswith(o + "=") for a in args for o in _OTHER_SOURCES), \
            f"{where} {family}: {flat}"
        assert "unsafe" not in flat, f"{where} {family}: {flat}"
        indexes = [args[i + 1] for i, a in enumerate(args) if a == "--index-url"]
        assert indexes == [hwdetect.amd_rocm_win_index(family)], f"{where} {family}: {flat}"
        reqs = [a for i, a in enumerate(args)
                if not a.startswith("-") and (i == 0 or args[i - 1] != "--index-url")]
        assert reqs and all(re.fullmatch(r"[a-z0-9-]+==[\w.+]+", r) for r in reqs), reqs
    assert ("managed ComfyUI", "gfx103x") in seen and ("torch-args", "gfx110x") in seen \
        and ("torch-args", "gfx120x") in seen, seen


def test_the_manual_amd_commands_in_the_docs_use_the_family_index_alone():
    doc = (ROOT / "docs" / "gpu-setup.md").read_text(encoding="utf-8")
    for family in ("gfx110x", "gfx120x"):
        index = hwdetect.amd_rocm_win_index(family)
        assert f"--index-url {index}" in doc
        assert f"--extra-index-url {index}" not in doc
    assert "unsafe-best-match" not in doc
