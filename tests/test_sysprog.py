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


def test_path_is_the_last_place_looked_in_a_source_checkout(tmp_path, monkeypatch):
    on_path = _exe(tmp_path / "bin" / ("tool.exe" if sysprog.IS_WINDOWS else "tool"))
    monkeypatch.setattr(sysprog, "system_candidates", lambda name: [str(tmp_path / "none")])
    monkeypatch.setattr(sysprog, "setup_install", lambda: False)
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


def test_a_setup_install_never_takes_a_program_from_path(tmp_path, monkeypatch):
    on_path = _exe(tmp_path / "bin" / ("tool.exe" if sysprog.IS_WINDOWS else "tool"))
    monkeypatch.setattr(sysprog, "system_candidates", lambda name: [str(tmp_path / "none")])
    monkeypatch.setattr(sysprog, "setup_install", lambda: True)
    monkeypatch.setenv("PATH", str(on_path.parent))
    assert sysprog.find("tool") is None
    assert sysprog.find_system("tool") is None
    with pytest.raises(FileNotFoundError, match="tool"):
        sysprog.require_system("tool")


def test_a_gpu_driver_tool_comes_from_the_driver_folder_first(tmp_path, monkeypatch):
    name = "nvidia-smi"
    driver = _exe(tmp_path / "driver" / (name + (".exe" if sysprog.IS_WINDOWS else "")))
    on_path = _exe(tmp_path / "bin" / driver.name)
    monkeypatch.setattr(sysprog, "system_candidates", lambda n: [str(tmp_path / "none")])
    monkeypatch.setattr(sysprog, "setup_install", lambda: True)
    monkeypatch.setenv("PATH", str(on_path.parent))
    monkeypatch.setattr(sysprog, "driver_candidates", lambda n: [str(driver)])
    assert sysprog.find(name) == str(driver)
    monkeypatch.setattr(sysprog, "driver_candidates", lambda n: [])
    assert Path(sysprog.find(name)).samefile(on_path)


def test_the_driver_folders_name_only_driver_tools():
    assert sysprog.driver_candidates("git") == []
    assert sysprog.driver_candidates("taskkill") == []


def test_setup_install_reads_the_install_record_beside_the_package():
    root = Path(sysprog.__file__).resolve().parents[1]
    assert sysprog.setup_install() == (root / ".localm-install.json").is_file()


def test_restricting_a_file_without_icacls_reports_failure(monkeypatch, tmp_path):
    from localm import config
    target = tmp_path / "secret.key"
    target.write_text("x", encoding="utf-8")
    monkeypatch.setattr(sysprog, "find_system", lambda name: None)
    monkeypatch.setenv("USERNAME", "someone")
    assert config.restrict_file_perms(target) is False
