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


def test_a_system_folder_program_wins_over_one_on_path(tmp_path, monkeypatch):
    exe = "tool.exe" if sysprog.IS_WINDOWS else "tool"
    system = _exe(tmp_path / "system" / exe)
    on_path = _exe(tmp_path / "bin" / exe)
    monkeypatch.setattr(sysprog, "system_candidates", lambda name: [str(system)])
    monkeypatch.setenv("PATH", str(on_path.parent))
    assert sysprog.find("tool") == str(system)


def test_a_program_in_the_current_folder_is_never_run_instead(tmp_path, monkeypatch):
    exe = "tool.exe" if sysprog.IS_WINDOWS else "tool"
    here = _exe(tmp_path / "cwd" / exe)
    _exe(tmp_path / "cwd" / "relbin" / exe)
    on_path = _exe(tmp_path / "bin" / exe)
    monkeypatch.setattr(sysprog, "system_candidates", lambda name: [])
    monkeypatch.chdir(here.parent)
    monkeypatch.setenv("PATH", os.pathsep.join(["", ".", "relbin", str(on_path.parent)]))
    found = sysprog.find("tool")
    assert found is not None and Path(found).samefile(on_path)
    monkeypatch.setenv("PATH", os.pathsep.join(["", ".", "relbin"]))
    assert sysprog.find("tool") is None


@pytest.mark.skipif(sys.platform != "win32", reason="PATHEXT is Windows-only")
def test_path_entries_are_tried_with_each_pathext_extension(tmp_path, monkeypatch):
    script = _exe(tmp_path / "bin" / "tool.cmd")
    monkeypatch.setattr(sysprog, "system_candidates", lambda name: [])
    monkeypatch.setenv("PATH", str(script.parent))
    monkeypatch.setenv("PATHEXT", ".EXE;.CMD")
    found = sysprog.find("tool")
    assert found is not None and Path(found).samefile(script)
    assert Path(sysprog.find("tool.cmd")).samefile(script)
    monkeypatch.setenv("PATHEXT", ".EXE")
    assert sysprog.find("tool") is None


def test_a_name_with_a_folder_is_not_looked_up_on_path(tmp_path, monkeypatch):
    _exe(tmp_path / "bin" / "sub" / ("tool.exe" if sysprog.IS_WINDOWS else "tool"))
    monkeypatch.setattr(sysprog, "system_candidates", lambda name: [])
    monkeypatch.setenv("PATH", str(tmp_path / "bin"))
    assert sysprog.find("sub/tool") is None
    assert sysprog.path_candidates("") == []


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


@pytest.mark.skipif(sys.platform != "win32", reason="icacls is the Windows path")
def test_restricting_a_file_without_icacls_reports_failure(monkeypatch, tmp_path):
    from localm import config
    target = tmp_path / "secret.key"
    target.write_text("x", encoding="utf-8")
    monkeypatch.setattr(sysprog, "find", lambda name: None)
    monkeypatch.setenv("USERNAME", "someone")
    assert config.restrict_file_perms(target) is False
