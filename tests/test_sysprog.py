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


def test_comfy_git_steps_report_a_missing_git_without_starting_anything(monkeypatch):
    import subprocess

    from localm.media import managed_comfy_provision as prov
    monkeypatch.setattr(sysprog, "find", lambda name: None)
    started = []
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: started.append(a))
    ok, out = prov._run(prov.git_argv("status"))
    assert not ok and "git was not found" in out and started == []


def test_comfy_git_steps_run_git_by_its_full_path(monkeypatch, tmp_path):
    from localm.media import managed_comfy_provision as prov
    git = str(tmp_path / ("git.exe" if sysprog.IS_WINDOWS else "git"))
    monkeypatch.setattr(sysprog, "find", lambda name: git if name == "git" else None)
    assert prov.git_argv("-C", "x", "status") == [git, "-C", "x", "status"]


def test_component_updates_raise_file_not_found_when_git_is_missing(monkeypatch, tmp_path):
    from localm import updater_components as uc
    monkeypatch.setattr(sysprog, "find", lambda name: None)
    with pytest.raises(FileNotFoundError, match="git was not found"):
        uc._git(tmp_path, "status")


@pytest.mark.skipif(sys.platform != "win32", reason="icacls is the Windows path")
def test_restricting_a_file_without_icacls_reports_failure(monkeypatch, tmp_path):
    from localm import config
    target = tmp_path / "secret.key"
    target.write_text("x", encoding="utf-8")
    monkeypatch.setattr(sysprog, "find", lambda name: None)
    monkeypatch.setenv("USERNAME", "someone")
    assert config.restrict_file_perms(target) is False
