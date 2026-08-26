# SPDX-License-Identifier: AGPL-3.0-or-later
"""The optional global `localm` command must add itself SAFELY: idempotent,
reversible, respecting an existing command, and NEVER via ``setx`` (which
truncates and silently corrupts the user PATH). These tests use an in-memory
fake for the registry, so they never touch the real machine's PATH."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from localm import globalcmd as gc


class _FakePath:
    """In-memory stand-in for HKCU\\Environment\\Path, patched over the module's
    read/write helpers so no test can touch the real registry."""

    def __init__(self, value=""):
        self.value = value
        self.regtype = 2  # REG_EXPAND_SZ sentinel

    def read(self):
        return self.value, self.regtype

    def write(self, value, regtype):
        self.value = value
        self.regtype = regtype


@pytest.fixture
def fake_path(monkeypatch):
    fp = _FakePath()
    monkeypatch.setattr(gc, "_win_read_user_path", fp.read)
    monkeypatch.setattr(gc, "_win_write_user_path", fp.write)
    return fp


# --------------------------- the setx guard ------------------------------- #

def test_path_edited_via_registry_not_setx():
    """PATH is edited through the Windows registry (winreg), never by shelling
    out to `setx` (which truncates and corrupts the user PATH). Enforced
    structurally: the module spawns NO external process for a PATH edit."""
    src = Path(gc.__file__).read_text(encoding="utf-8")
    assert "import subprocess" not in src         # no shelling out at all
    assert ("os." + "system") not in src          # concat avoids a scanner false-positive
    assert "winreg" in src                        # uses the registry API instead


# ----------------------- platform-agnostic helpers ------------------------ #

def test_split_and_same_dir():
    assert gc._split_path("a" + os.pathsep + os.pathsep + "b") == ["a", "b"]
    assert not gc._same_dir("/a/b", "/a/c")
    if sys.platform == "win32":
        assert gc._same_dir("Z:\\Foo\\", "Z:/foo")   # case + slash + trailing sep


@pytest.mark.skipif(sys.platform != "win32",
                    reason="Windows registry PATH semantics (; separator, case-insensitive)")
def test_win_path_add_is_idempotent(fake_path):
    d = r"Z:\clone\bin"
    assert gc._win_path_add(d) is True          # first add changes PATH
    assert d in fake_path.value
    assert gc._win_path_add(d) is False         # second add is a no-op
    assert fake_path.value.count(d) == 1        # never duplicated


@pytest.mark.skipif(sys.platform != "win32",
                    reason="Windows registry PATH semantics (; separator, case-insensitive)")
def test_win_path_remove_only_ours(fake_path):
    fake_path.value = os.pathsep.join([r"Z:\keep\me", r"Z:\clone\bin", r"Z:\also\keep"])
    assert gc._win_path_remove(r"Z:\clone\bin") is True
    assert r"Z:\keep\me" in fake_path.value      # every OTHER entry preserved
    assert r"Z:\also\keep" in fake_path.value
    assert r"Z:\clone\bin" not in fake_path.value
    assert gc._win_path_remove(r"Z:\clone\bin") is False   # already gone -> no-op


def test_existing_localm_ignores_our_own_shim(monkeypatch, tmp_path):
    other = tmp_path / "other" / "localm"
    monkeypatch.setenv("PATH", os.pathsep.join([str(tmp_path), str(other.parent)]))
    shim = tmp_path / "localm.cmd"
    shim.write_text("x", encoding="utf-8")
    monkeypatch.setattr(gc.shutil, "which", lambda name: str(shim))
    assert gc.existing_localm(shim) is None            # our own -> not a conflict
    monkeypatch.setattr(gc.shutil, "which", lambda name: str(other))
    assert gc.existing_localm(shim) == str(other)      # a different one -> reported


def test_deterministic_locations(tmp_path):
    d = gc.bin_dir(tmp_path)
    s = gc.shim_path(tmp_path)
    assert s.parent == d
    if sys.platform == "win32":
        assert d == tmp_path / "bin"
        assert s.name == "localm.cmd"
    else:
        assert s.name == "localm"


# ----------------------------- install path ------------------------------- #

@pytest.mark.skipif(sys.platform != "win32", reason="Windows shim + registry PATH")
def test_install_uninstall_roundtrip_windows(fake_path, tmp_path, monkeypatch):
    monkeypatch.setattr(gc.shutil, "which", lambda name: None)   # no pre-existing localm
    res = gc.install(tmp_path)
    shim = Path(res["shim"])
    assert shim.exists()
    assert "localm.exe" in shim.read_text(encoding="utf-8")
    assert res["path_modified"] is True
    assert res["path_dir"] in fake_path.value
    assert res["conflict"] is None
    # reverse it: shim gone, our PATH entry gone, everything else untouched.
    rep = gc.uninstall_command(res["path_dir"], res["shim"])
    assert not shim.exists()
    assert res["path_dir"] not in fake_path.value
    assert any("PATH entry" in x for x in rep["removed"])


@pytest.mark.skipif(sys.platform == "win32",
                    reason="POSIX symlink usually needs privilege on Windows")
def test_install_posix_symlink(monkeypatch, tmp_path):
    bindir = tmp_path / "localbin"
    monkeypatch.setattr(gc, "bin_dir", lambda root: bindir)
    monkeypatch.setattr(gc, "shim_path", lambda root: bindir / "localm")
    monkeypatch.setattr(gc, "_posix_ensure_on_path", lambda d: (False, None))
    monkeypatch.setattr(gc.shutil, "which", lambda name: None)
    target = tmp_path / ".venv" / "bin" / "localm"
    target.parent.mkdir(parents=True)
    target.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setattr(gc, "_venv_localm", lambda root: target)
    res = gc.install(tmp_path)
    link = Path(res["shim"])
    assert link.is_symlink()
    assert Path(os.path.realpath(link)) == target.resolve()


def test_install_cli_exit_code_reflects_path_modified(monkeypatch):
    """setup keys off this exit code to record --path-modified accurately: 0 =
    PATH changed, 20 = installed but PATH already set, 1 = failed. install() is
    mocked so no real PATH/registry is touched."""
    monkeypatch.setattr(gc, "install", lambda root, precedence="append": {
        "path_dir": "d", "shim": "s", "path_modified": True, "conflict": None})
    assert gc.main(["install", "--root", "."]) == 0

    monkeypatch.setattr(gc, "install", lambda root, precedence="append": {
        "path_dir": "d", "shim": "s", "path_modified": False, "conflict": None})
    assert gc.main(["install", "--root", "."]) == 20

    def _boom(root, precedence="append"):
        raise OSError("cannot write shim")
    monkeypatch.setattr(gc, "install", _boom)
    assert gc.main(["install", "--root", "."]) == 1


# --------------- honesty: PATH-edit-failed is not a false success ---------- #

def test_posix_ensure_on_path_reports_edit_failure(monkeypatch, tmp_path):
    """When bindir is NOT on PATH and every shell-rc edit fails, the function must
    return (False, <note>) - the 'could not add it' case - NOT the same (False,
    None) as 'already on PATH'. Home points at a nonexistent dir (via the env vars
    Path.home() reads) so every rc write AND the ~/.profile fallback raise
    FileNotFoundError (an OSError)."""
    missing_home = tmp_path / "no_such_home"           # never created
    monkeypatch.setenv("HOME", str(missing_home))
    monkeypatch.setenv("USERPROFILE", str(missing_home))
    monkeypatch.delenv("HOMEDRIVE", raising=False)
    monkeypatch.delenv("HOMEPATH", raising=False)
    assert Path.home() == missing_home                 # sanity: env override took
    bindir = missing_home / ".local" / "bin"           # not on the real PATH
    changed, note = gc._posix_ensure_on_path(bindir)
    assert changed is False
    assert note and "PATH" in note and "manually" in note


def test_posix_ensure_on_path_already_on_path_has_no_note(monkeypatch, tmp_path):
    """The already-on-PATH case stays (False, None) - no false warning."""
    bindir = tmp_path / "localbin"
    monkeypatch.setattr(gc, "_posix_on_path", lambda d: True)
    assert gc._posix_ensure_on_path(bindir) == (False, None)


def test_install_cli_surfaces_path_edit_failure(monkeypatch, capsys):
    """main() must NOT claim 'already on PATH' when the shim was created but
    its dir could not be added to PATH; it prints the manual-add note and still
    exits 20 (installed, PATH not modified by us)."""
    monkeypatch.setattr(gc, "install", lambda root, precedence="append": {
        "path_dir": "/home/u/.local/bin", "shim": "s", "path_modified": False,
        "conflict": None,
        "path_note": "could not add /home/u/.local/bin to your PATH (could not "
                     "edit your shell startup files); add it manually"})
    rc = gc.main(["install", "--root", "."])
    out = capsys.readouterr().out
    assert rc == 20
    assert "already on PATH" not in out          # the falsehood must be gone
    assert "add it manually" in out
