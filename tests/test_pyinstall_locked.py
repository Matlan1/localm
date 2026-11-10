# SPDX-License-Identifier: AGPL-3.0-or-later
"""Python package installs localm starts: the network policy check runs first,
the update's dependency step installs from the shipped uv.lock, and a plugin's
dependencies are installed at their locked versions with --require-hashes.

The lock reads run the real uv against the real uv.lock of this tree; only the
installing child process is faked."""

from __future__ import annotations

import importlib.metadata as md
import re
import shutil
from pathlib import Path

import pytest

from localm import _apply_update as au, pyinstall, uv_tool
from localm.plugins import deps

ROOT = uv_tool.install_root()


def _uv():
    uv = uv_tool.find_uv(ROOT)
    if not uv or pyinstall.lock_file(ROOT) is None:
        pytest.skip("needs uv and the tree's uv.lock")
    return uv


def _lock_hashes(name: str, version: str) -> set:
    """Every sha256 the tree's uv.lock records for *name* at *version*."""
    text = (ROOT / "uv.lock").read_text(encoding="utf-8")
    m = re.search(r'\[\[package\]\]\r?\nname = "' + re.escape(name) + r'"\r?\nversion = "'
                  + re.escape(version) + r'"(.*?)(?=\r?\n\[\[package\]\]|\Z)', text, re.S)
    assert m, (name, version)
    return set(re.findall(r'hash = "(sha256:[0-9a-f]{64})"', m.group(1)))


# ------------------------------ the helper --------------------------------- #

def test_index_urls_names_pypi_and_every_index_option():
    cmd = ["uv", "pip", "install", "x", "--index-url", "https://a.example/simple",
           "--extra-index-url=https://b.example/simple", "-f", "https://c.example/wheels",
           "--find-links", "./local", "torch @ https://d.example/t.whl"]
    urls = pyinstall.index_urls(cmd)
    assert urls[:2] == list(pyinstall.PYPI_URLS)
    for u in ("https://a.example/simple", "https://b.example/simple",
              "https://c.example/wheels", "https://d.example/t.whl"):
        assert u in urls
    assert "./local" not in urls


def test_net_mode_off_refuses_an_install(monkeypatch):
    monkeypatch.setenv("LOCALM_NET_MODE", "off")
    with pytest.raises(pyinstall.InstallRefused, match="net_mode=off"):
        pyinstall.check_command(["uv", "pip", "install", "x"])


def test_net_allow_model_downloads_exempts_off(monkeypatch):
    from localm.config import update_config
    update_config(lambda c: c.update({"net_allow_model_downloads": True}))
    monkeypatch.setenv("LOCALM_NET_MODE", "off")
    pyinstall.check_command(["uv", "pip", "install", "x"])


def test_the_deny_list_refuses_an_extra_index(monkeypatch):
    from localm.config import update_config
    update_config(lambda c: c.update({"net_deny": ["evil.example"]}))
    with pytest.raises(pyinstall.InstallRefused, match="deny list"):
        pyinstall.check_command(["pip", "install", "x", "-i", "https://evil.example/simple"])


def test_parse_export_refuses_an_entry_without_a_hash():
    with pytest.raises(ValueError):
        pyinstall.parse_export("six==1.16.0 ; sys_platform == 'win32'\n")
    with pytest.raises(ValueError):
        pyinstall.parse_export("six>=1.16.0 --hash=sha256:" + "a" * 64 + "\n")


def test_locked_requirements_reads_exact_hashed_pins_from_the_lock():
    uv = _uv()
    entries = pyinstall.locked_requirements(uv, ROOT, ["voice"])
    by_name = {e.name: e for e in entries}
    assert "faster-whisper" in by_name and "localm" not in by_name
    for e in entries:
        assert e.hashes and set(e.hashes) <= _lock_hashes(e.name, e.version), e.name


def test_locked_requirements_refuses_a_root_without_a_lock(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    with pytest.raises(pyinstall.InstallRefused):
        pyinstall.locked_requirements("uv", tmp_path, ["voice"])


# --------------------------- the update's step ----------------------------- #

def test_the_update_dependency_step_syncs_from_the_lock():
    cmd = au.post_swap_command("deps", uv="/x/uv")
    assert cmd[:2] == ["/x/uv", "sync"]
    assert "--frozen" in cmd and "--inexact" in cmd and "--no-config" in cmd
    extras = [cmd[i + 1] for i, a in enumerate(cmd) if a == "--extra"]
    assert extras == list(au.DEPS_EXTRAS)
    assert not any("verify-hashes" in a or a == "pip" for a in cmd)


def test_the_update_dependency_step_installs_what_the_lock_names():
    """The extras the step syncs resolve, in the shipped lock, to hashed pins."""
    uv = _uv()
    assert pyinstall.locked_requirements(uv, ROOT, au.DEPS_EXTRAS)


# --------------------------- plugin dependencies --------------------------- #

@pytest.fixture
def capture(monkeypatch):
    """Records each install attempt list and the requirements file it names."""
    seen: dict = {"attempts": None, "file": None}

    def fake_run(attempts, env, on_progress):
        seen["attempts"] = attempts
        for i, a in enumerate(attempts[0]):
            if a == "-r":
                seen["file"] = Path(attempts[0][i + 1]).read_text(encoding="utf-8")
        return True, "ok"
    monkeypatch.setattr(deps, "_run_attempts", fake_run)
    return seen


def test_plugin_install_takes_the_locked_hashed_versions(monkeypatch, capture):
    _uv()
    monkeypatch.setattr(deps, "_unmet_dependencies", lambda names: [])
    real_version = md.version

    def version(name):
        if name.lower().replace("_", "-") == "faster-whisper":
            raise md.PackageNotFoundError(name)
        return real_version(name)
    monkeypatch.setattr(md, "version", version)
    ok, _out = deps._run_pip(["faster-whisper>=1.2.1"], extras=["voice"])
    assert ok
    uv_cmd, pip_cmd = capture["attempts"]
    for cmd in (uv_cmd, pip_cmd):
        assert "--require-hashes" in cmd and "--no-deps" in cmd
        assert "faster-whisper>=1.2.1" not in cmd
    line = next(ln for ln in capture["file"].splitlines() if ln.startswith("faster-whisper=="))
    version_ = line.split("==", 1)[1].split()[0]
    hashes = set(re.findall(r"--hash=(sha256:[0-9a-f]{64})", line))
    assert hashes and hashes <= _lock_hashes("faster-whisper", version_)


def test_plugin_install_refused_under_net_mode_off(monkeypatch, capture):
    monkeypatch.setenv("LOCALM_NET_MODE", "off")
    ok, out = deps._run_pip(["faster-whisper>=1.2.1"], extras=["voice"])
    assert ok is False and "net_mode=off" in out
    assert capture["attempts"] is None


def test_plugin_install_with_an_unreadable_lock_installs_nothing(monkeypatch, tmp_path, capture):
    uv = _uv()
    shutil.copy(ROOT / "pyproject.toml", tmp_path / "pyproject.toml")
    (tmp_path / "uv.lock").write_text("not a lock\n", encoding="utf-8")
    monkeypatch.setattr(deps.uv_tool, "install_root", lambda: tmp_path)
    monkeypatch.setattr(deps.uv_tool, "find_uv", lambda root: uv)
    ok, out = deps._run_pip(["faster-whisper>=1.2.1"], extras=["voice"])
    assert ok is False and "uv.lock" in out
    assert capture["attempts"] is None


def test_plugin_install_without_a_lock_says_hashes_are_not_checked(monkeypatch, tmp_path,
                                                                    capture):
    monkeypatch.setattr(deps.uv_tool, "install_root", lambda: tmp_path)
    monkeypatch.setattr(deps.uv_tool, "find_uv", lambda root: "/x/uv")
    lines: list = []
    ok, _out = deps._run_pip(["faster-whisper>=1.2.1"], extras=["voice"],
                             on_progress=lines.append)
    assert ok
    assert capture["attempts"][0] == ["/x/uv", "pip", "install", "--python",
                                      deps.sys.executable, "faster-whisper>=1.2.1"]
    assert any("not checked against known hashes" in ln for ln in lines)


def test_plugin_install_fails_when_an_installed_package_has_an_unmet_dependency(
        monkeypatch, capture):
    _uv()
    real_version = md.version

    def version(name):
        if name.lower().replace("_", "-") == "faster-whisper":
            raise md.PackageNotFoundError(name)
        return real_version(name)
    monkeypatch.setattr(md, "version", version)
    monkeypatch.setattr(md, "requires",
                        lambda name: ["no-such-dist-xyz>=1"] if name == "faster-whisper" else [])
    ok, out = deps._run_pip(["faster-whisper>=1.2.1"], extras=["voice"])
    assert ok is False and "faster-whisper needs no-such-dist-xyz>=1" in out
