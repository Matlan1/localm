# SPDX-License-Identifier: AGPL-3.0-or-later
"""localm.uv_tool: which uv an install runs, and installing the pinned one."""

import hashlib
import json
import re
from pathlib import Path

import pytest

from localm import uv_tool

ROOT = Path(__file__).resolve().parents[1]
UV = uv_tool.UV_NAME


def _uv(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    exe = directory / UV
    exe.write_bytes(b"")
    exe.chmod(0o755)
    return exe


@pytest.fixture
def machine(tmp_path, monkeypatch):
    """No uv on PATH and none in the home directory; returns (root, home)."""
    home = tmp_path / "userhome"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setenv("PATH", str(tmp_path / "emptypath"))
    root = tmp_path / "clone"
    root.mkdir()
    return root, home


def _record(root: Path, **fields):
    (root / ".localm-install.json").write_text(json.dumps({"schema": 3, **fields}), encoding="utf-8")


class TestFindUv:
    def test_nothing_anywhere_is_none(self, machine):
        root, _ = machine
        assert uv_tool.find_uv(root) is None

    def test_the_contained_copy(self, machine):
        root, _ = machine
        exe = _uv(root / ".uv")
        assert uv_tool.find_uv(root) == str(exe)

    def test_the_contained_copy_in_a_bin_subfolder(self, machine):
        root, _ = machine
        exe = _uv(root / ".uv" / "bin")
        assert uv_tool.find_uv(root) == str(exe)

    def test_a_uv_on_path_is_returned_as_a_full_path(self, machine, tmp_path, monkeypatch):
        root, _ = machine
        on_path = _uv(tmp_path / "bin")
        monkeypatch.setenv("PATH", str(on_path.parent))
        found = uv_tool.find_uv(root)
        assert Path(found).is_absolute() and Path(found).samefile(on_path)

    def test_the_contained_copy_beats_one_on_path(self, machine, tmp_path, monkeypatch):
        root, _ = machine
        monkeypatch.setenv("PATH", str(_uv(tmp_path / "bin").parent))
        exe = _uv(root / ".uv")
        assert uv_tool.find_uv(root) == str(exe)

    def test_astral_default_directories_when_path_lacks_uv(self, machine):
        root, home = machine
        exe = _uv(home / ".local" / "bin")
        assert uv_tool.find_uv(root) == str(exe)

    def test_path_beats_the_default_directories(self, machine, tmp_path, monkeypatch):
        root, home = machine
        _uv(home / ".local" / "bin")
        on_path = _uv(tmp_path / "bin")
        monkeypatch.setenv("PATH", str(on_path.parent))
        assert Path(uv_tool.find_uv(root)).samefile(on_path)

    def test_the_recorded_shared_uv_beats_path_and_defaults(self, machine, tmp_path, monkeypatch):
        root, home = machine
        own = _uv(tmp_path / "mine")
        monkeypatch.setenv("PATH", str(_uv(tmp_path / "bin").parent))
        _uv(home / ".local" / "bin")
        _record(root, uv_exe=str(own))
        assert uv_tool.find_uv(root) == str(own)

    def test_a_recorded_uv_that_is_gone_falls_through(self, machine, tmp_path):
        root, _ = machine
        _record(root, uv_exe=str(tmp_path / "deleted" / UV), uv_dir=str(root / ".uv"))
        assert uv_tool.find_uv(root) is None
        exe = _uv(root / ".uv")
        assert uv_tool.find_uv(root) == str(exe)

    def test_a_damaged_record_is_ignored(self, machine):
        root, _ = machine
        (root / ".localm-install.json").write_text("{not json", encoding="utf-8")
        exe = _uv(root / ".uv")
        assert uv_tool.find_uv(root) == str(exe)

    @pytest.mark.skipif(uv_tool.IS_WINDOWS, reason="POSIX execute bit")
    def test_a_file_that_cannot_be_executed_is_not_a_uv(self, machine):
        root, _ = machine
        (root / ".uv").mkdir()
        (root / ".uv" / UV).write_bytes(b"")
        (root / ".uv" / UV).chmod(0o644)
        assert uv_tool.find_uv(root) is None
        assert uv_tool.contained_uv(root) is None

    def test_a_recorded_contained_folder_outside_the_install_is_ignored(self, machine, tmp_path):
        root, _ = machine
        old_clone = tmp_path / "old-clone"
        _uv(old_clone / ".uv")
        _record(root, uv_dir=str(old_clone / ".uv"))
        assert uv_tool.find_uv(root) is None
        assert uv_tool.find_uv(root) != str(old_clone / ".uv" / UV)
        own = _uv(root / ".uv")
        assert uv_tool.find_uv(root) == str(own)

    def test_describe_search_names_every_place(self, machine, tmp_path):
        root, home = machine
        _record(root, uv_exe=str(tmp_path / "mine" / UV))
        text = uv_tool.describe_search(root)
        for part in (str(tmp_path / "mine" / UV), str(root / ".uv" / UV), "PATH",
                     str(home / ".local" / "bin" / UV)):
            assert part in text


class TestPin:
    def test_the_pin_is_read_from_the_shipped_setup_script(self):
        version, sha = uv_tool.read_pin(ROOT)
        assert re.fullmatch(r"\d+\.\d+\.\d+", version) and re.fullmatch(r"[0-9a-f]{64}", sha)

    def test_a_missing_setup_script_is_an_error_that_names_the_cause(self, tmp_path):
        with pytest.raises(uv_tool.UvBootstrapError, match="pins could not be read|could not be read"):
            uv_tool.read_pin(tmp_path)

    def test_a_setup_script_that_is_not_utf8_is_reported_not_raised(self, tmp_path):
        (tmp_path / "setup.sh").write_bytes(bytes([0xFF, 0xFE, 0x80]) + b" not utf-8")
        (tmp_path / "setup.bat").write_bytes(bytes([0xFF, 0xFE, 0x80]) + b" not utf-8")
        with pytest.raises(uv_tool.UvBootstrapError):
            uv_tool.read_pin(tmp_path)

    def test_the_module_carries_no_pin_of_its_own(self):
        text = (ROOT / "localm" / "uv_tool.py").read_text(encoding="utf-8")
        assert not re.search(r"\b[0-9a-f]{64}\b", text)
        assert not re.search(r'UV_INSTALLER_VERSION\s*=\s*"?\d', text)


def _clone_with_pin(tmp_path: Path, installer: bytes) -> Path:
    clone = tmp_path / "pinned"
    clone.mkdir()
    sha = hashlib.sha256(installer).hexdigest()
    (clone / "setup.sh").write_text(
        f'UV_INSTALLER_VERSION="9.8.7"\nUV_INSTALLER_SHA256="{sha}"\n', encoding="utf-8")
    (clone / "setup.bat").write_text(
        f'set "UV_INSTALLER_VERSION=9.8.7"\nset "UV_INSTALLER_SHA256={sha}"\n', encoding="utf-8")
    return clone


class TestBootstrap:
    def test_installs_into_dot_uv_when_the_checksum_matches(self, tmp_path):
        installer = b"the pinned installer"
        clone = _clone_with_pin(tmp_path, installer)
        seen = {}

        def download(url, dest):
            seen["url"] = url
            dest.write_bytes(installer)

        def run(script, env):
            seen["env"] = env
            _uv(Path(env["UV_INSTALL_DIR"]))
            return 0

        exe = uv_tool.bootstrap_uv(clone, download=download, run_installer=run)
        assert exe == clone / ".uv" / UV
        assert "/9.8.7/uv-installer." in seen["url"]
        assert seen["env"]["UV_INSTALL_DIR"] == str(clone / ".uv")
        assert seen["env"]["UV_UNMANAGED_INSTALL"] == str(clone / ".uv")

    def test_a_checksum_mismatch_never_runs_the_installer(self, tmp_path):
        clone = _clone_with_pin(tmp_path, b"the pinned installer")
        ran = []
        with pytest.raises(uv_tool.UvBootstrapError, match="checksum"):
            uv_tool.bootstrap_uv(clone, download=lambda u, d: d.write_bytes(b"tampered"),
                                 run_installer=lambda s, e: ran.append(s) or 0)
        assert ran == [] and not (clone / ".uv").exists()

    def test_a_failed_download_is_named(self, tmp_path):
        clone = _clone_with_pin(tmp_path, b"x")

        def download(url, dest):
            raise OSError("network is unreachable")

        with pytest.raises(uv_tool.UvBootstrapError, match="download.*network is unreachable"):
            uv_tool.bootstrap_uv(clone, download=download)

    def test_an_installer_that_fails_is_reported(self, tmp_path):
        clone = _clone_with_pin(tmp_path, b"x")
        with pytest.raises(uv_tool.UvBootstrapError, match="exited 3"):
            uv_tool.bootstrap_uv(clone, download=lambda u, d: d.write_bytes(b"x"),
                                 run_installer=lambda s, e: 3)

    def test_an_installer_that_leaves_no_uv_is_not_a_success(self, tmp_path):
        clone = _clone_with_pin(tmp_path, b"x")
        with pytest.raises(uv_tool.UvBootstrapError, match="no .* appeared"):
            uv_tool.bootstrap_uv(clone, download=lambda u, d: d.write_bytes(b"x"),
                                 run_installer=lambda s, e: 0)

    def test_the_scratch_download_is_removed(self, tmp_path, monkeypatch):
        clone = _clone_with_pin(tmp_path, b"x")
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        monkeypatch.setenv("TMP", str(scratch))
        monkeypatch.setenv("TEMP", str(scratch))
        monkeypatch.setenv("TMPDIR", str(scratch))
        import tempfile
        monkeypatch.setattr(tempfile, "tempdir", None)
        with pytest.raises(uv_tool.UvBootstrapError):
            uv_tool.bootstrap_uv(clone, download=lambda u, d: d.write_bytes(b"x"),
                                 run_installer=lambda s, e: 1)
        assert list(scratch.iterdir()) == []


class TestBundledUv:
    """The uv the ``uv`` Python package installed into this environment."""

    @staticmethod
    def _package(monkeypatch, find_uv_bin):
        import sys
        import types
        module = types.ModuleType("uv")
        module.find_uv_bin = find_uv_bin
        monkeypatch.setitem(sys.modules, "uv", module)

    def test_the_package_binary_is_returned_by_full_path(self, tmp_path, monkeypatch):
        exe = _uv(tmp_path / "venv" / "Scripts")
        self._package(monkeypatch, lambda: str(exe))
        monkeypatch.setenv("PATH", str(tmp_path / "emptypath"))
        assert uv_tool.bundled_uv() == str(exe)

    def test_no_package_means_none(self, monkeypatch):
        import sys
        monkeypatch.setitem(sys.modules, "uv", None)
        assert uv_tool.bundled_uv() is None

    def test_a_package_without_its_binary_means_none(self, monkeypatch):
        def missing():
            raise FileNotFoundError("Could not find the uv binary")
        self._package(monkeypatch, missing)
        assert uv_tool.bundled_uv() is None

    @pytest.mark.parametrize("answer", ["uv", "gone"])
    def test_a_bare_or_absent_answer_means_none(self, tmp_path, monkeypatch, answer):
        path = answer if answer == "uv" else str(tmp_path / "gone" / UV)
        self._package(monkeypatch, lambda: path)
        assert uv_tool.bundled_uv() is None


class TestInstallerShell:
    """The uv installer runs under the system shell named by full path."""

    def test_the_installer_runs_the_shell_sysprog_finds(self, tmp_path, monkeypatch):
        import subprocess
        from localm import sysprog
        shell = str(tmp_path / ("powershell.exe" if uv_tool.IS_WINDOWS else "sh"))
        monkeypatch.setattr(sysprog, "find", lambda name: shell)
        seen = []
        monkeypatch.setattr(subprocess, "run", lambda argv, **kw: seen.append(argv)
                            or subprocess.CompletedProcess(argv, 0))
        assert uv_tool._run_installer(tmp_path / "uv-installer", {}) == 0
        assert seen[0][0] == shell

    def test_a_missing_shell_fails_the_bootstrap_by_name(self, tmp_path, monkeypatch):
        from localm import sysprog
        monkeypatch.setattr(sysprog, "find", lambda name: None)
        installer = b"the pinned installer"
        clone = _clone_with_pin(tmp_path, installer)
        with pytest.raises(uv_tool.UvBootstrapError) as ei:
            uv_tool.bootstrap_uv(clone, download=lambda u, d: d.write_bytes(installer))
        assert "could not run the uv installer" in str(ei.value)
        assert ("powershell" if uv_tool.IS_WINDOWS else "sh") in str(ei.value)


class TestOwnUv:
    def test_the_bundled_uv_comes_first(self, machine, monkeypatch):
        root, _ = machine
        _uv(root / ".uv")
        monkeypatch.setattr(uv_tool, "bundled_uv", lambda: "/venv/bin/uv")
        assert uv_tool.own_uv(root) == "/venv/bin/uv"

    def test_without_a_bundled_uv_the_install_uv_is_used(self, machine, monkeypatch):
        root, _ = machine
        exe = _uv(root / ".uv")
        monkeypatch.setattr(uv_tool, "bundled_uv", lambda: None)
        assert uv_tool.own_uv(root) == str(exe)
