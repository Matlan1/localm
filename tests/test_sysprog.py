# SPDX-License-Identifier: AGPL-3.0-or-later
"""localm.sysprog: system programs by full path, found without PATH."""

import os
import sys
from pathlib import Path

import pytest

from localm import sysprog


def _exe(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    path.chmod(0o755)
    return path


def test_a_system_folder_program_is_found_with_an_empty_path(tmp_path, monkeypatch):
    want = _exe(tmp_path / "system" / ("tool.exe" if sysprog.IS_WINDOWS else "tool"))
    monkeypatch.setattr(sysprog, "system_candidates", lambda name: [str(want)])
    monkeypatch.setenv("PATH", "")
    assert sysprog.find("tool") == str(want)


def test_path_is_the_last_place_looked(tmp_path, monkeypatch):
    on_path = _exe(tmp_path / "bin" / ("tool.exe" if sysprog.IS_WINDOWS else "tool"))
    monkeypatch.setattr(sysprog, "system_candidates", lambda name: [str(tmp_path / "none")])
    monkeypatch.setenv("PATH", str(on_path.parent))
    found = sysprog.find("tool")
    assert found is not None and os.path.isabs(found) and Path(found).samefile(on_path)


def test_a_missing_program_is_none(tmp_path, monkeypatch):
    monkeypatch.setattr(sysprog, "system_candidates", lambda name: [str(tmp_path / "none")])
    monkeypatch.setenv("PATH", str(tmp_path / "emptypath"))
    assert sysprog.find("tool") is None


@pytest.mark.skipif(sys.platform != "win32", reason="the Windows system directory")
def test_windows_powershell_is_found_without_path(monkeypatch):
    monkeypatch.setenv("PATH", "")
    found = sysprog.find("powershell")
    assert found and Path(found).name.lower() == "powershell.exe"
    assert Path(found).is_relative_to(sysprog.windows_system_dir())


@pytest.mark.skipif(sys.platform == "win32", reason="the POSIX default search path")
def test_posix_sh_is_found_without_path(monkeypatch):
    monkeypatch.setenv("PATH", "")
    found = sysprog.find("sh")
    assert found and os.path.isabs(found) and os.path.dirname(found) in sysprog.posix_system_dirs()
