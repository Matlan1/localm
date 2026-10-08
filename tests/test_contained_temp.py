# SPDX-License-Identifier: AGPL-3.0-or-later
"""Where localm's temporary files go.

``localm.config`` points tempfile and TMP/TEMP/TMPDIR at :func:`config.temp_dir`
when it is imported, so nothing localm (or a library or child process it starts)
writes to a temp location lands outside the chosen folder. In order: the
``LOCALM_TMPDIR`` variable; the ``temp_location`` setting (``data``, ``system`` or
an absolute folder); then ``auto``, which keeps temp files in ``<data dir>/tmp``
for a self-contained checkout and leaves the system temp folder for any other
install. Each case runs in a fresh interpreter, because the containment happens at
import time.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent

_PROBE = r"""
import json, os, subprocess, sys, tempfile
import localm.config
fd, made = tempfile.mkstemp(prefix="probe_")
os.close(fd)
child = subprocess.run(
    [sys.executable, "-c", "import tempfile; print(tempfile.gettempdir())"],
    capture_output=True, text=True).stdout.strip()
out = {
    "gettempdir": tempfile.gettempdir(),
    "env": [os.environ.get(v) for v in ("TMP", "TEMP", "TMPDIR")],
    "made": made,
    "child": child,
    "portable": localm.config._is_portable_home(localm.config.home_dir()),
}
if sys.platform == "win32":
    import ctypes
    buf = ctypes.create_unicode_buffer(1024)
    ctypes.windll.kernel32.GetTempPathW(1024, buf)
    out["win32_temp_path"] = buf.value
print(json.dumps(out))
"""


def _run(tmp_path, *, home, system_temp, extra_env=None, config=None):
    system_temp.mkdir(exist_ok=True)
    if config is not None:
        Path(home).mkdir(parents=True, exist_ok=True)
        (Path(home) / "config.json").write_text(json.dumps(config), encoding="utf-8")
    env = dict(os.environ)
    env.pop("LOCALM_TMPDIR", None)
    env["LOCALM_HOME"] = str(home)
    for var in ("TMP", "TEMP", "TMPDIR"):
        env[var] = str(system_temp)
    env["PYTHONPATH"] = str(_REPO_ROOT)
    env.update(extra_env or {})
    r = subprocess.run([sys.executable, "-c", _PROBE], env=env, cwd=str(_REPO_ROOT),
                       capture_output=True, text=True, timeout=180)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout.strip().splitlines()[-1]), r.stderr


def _same(a, b) -> bool:
    return os.path.normcase(os.path.normpath(str(a))) == os.path.normcase(
        os.path.normpath(str(b)))


class TestChoosingTheDataFolder:
    """``temp_location = data``: temp files go to ``<data dir>/tmp``."""

    def _data(self, tmp_path):
        home = tmp_path / "data"
        out, err = _run(tmp_path, home=home, system_temp=tmp_path / "systemtemp",
                        config={"temp_location": "data"})
        return home, out, err

    def test_tempfile_and_env_point_into_the_data_dir(self, tmp_path):
        home, out, _ = self._data(tmp_path)
        want = home / "tmp"
        assert _same(out["gettempdir"], want)
        assert all(_same(v, want) for v in out["env"]), out["env"]
        assert _same(Path(out["made"]).parent, want)
        assert want.is_dir()

    def test_nothing_lands_in_the_system_temp_folder(self, tmp_path):
        _data = self._data(tmp_path)
        assert list((tmp_path / "systemtemp").iterdir()) == []

    def test_a_child_process_inherits_it(self, tmp_path):
        home, out, _ = self._data(tmp_path)
        assert _same(out["child"], home / "tmp")

    @pytest.mark.skipif(sys.platform != "win32", reason="Win32 temp path")
    def test_native_code_asking_windows_for_the_temp_path_gets_it(self, tmp_path):
        home, out, _ = self._data(tmp_path)
        assert _same(out["win32_temp_path"], home / "tmp")


class TestAutoDefault:
    def test_an_install_outside_a_checkout_leaves_the_system_temp_folder(self, tmp_path):
        system_temp = tmp_path / "systemtemp"
        out, _ = _run(tmp_path, home=tmp_path / "data", system_temp=system_temp)
        assert out["portable"] is False
        assert _same(out["gettempdir"], system_temp)
        assert not (tmp_path / "data" / "tmp").exists()

    def test_an_explicit_auto_is_the_same_as_no_setting(self, tmp_path):
        system_temp = tmp_path / "systemtemp"
        out, _ = _run(tmp_path, home=tmp_path / "data", system_temp=system_temp,
                      config={"temp_location": "auto"})
        assert _same(out["gettempdir"], system_temp)

    def test_a_data_folder_inside_a_checkout_keeps_its_temp_files_inside_it(
            self, tmp_path):
        home = _REPO_ROOT / "scratch" / f"portable-temp-{os.getpid()}"
        try:
            out, _ = _run(tmp_path, home=home, system_temp=tmp_path / "systemtemp")
            assert out["portable"] is True
            assert _same(out["gettempdir"], home / "tmp")
            assert list((tmp_path / "systemtemp").iterdir()) == []
        finally:
            import shutil
            shutil.rmtree(home, ignore_errors=True)


class TestChoosingSystemOrAFolder:
    def test_system_leaves_the_system_temp_folder(self, tmp_path):
        system_temp = tmp_path / "systemtemp"
        out, _ = _run(tmp_path, home=tmp_path / "data", system_temp=system_temp,
                      config={"temp_location": "system"})
        assert _same(out["gettempdir"], system_temp)
        assert not (tmp_path / "data" / "tmp").exists()

    def test_an_absolute_folder_is_used(self, tmp_path):
        chosen = tmp_path / "chosen"
        out, _ = _run(tmp_path, home=tmp_path / "data", system_temp=tmp_path / "st",
                      config={"temp_location": str(chosen)})
        assert _same(out["gettempdir"], chosen)
        assert chosen.is_dir()

    def test_a_relative_folder_is_reported_and_treated_as_auto(self, tmp_path):
        system_temp = tmp_path / "systemtemp"
        out, err = _run(tmp_path, home=tmp_path / "data", system_temp=system_temp,
                        config={"temp_location": "relative/folder"})
        assert "temp_location" in err and "absolute" in err
        assert _same(out["gettempdir"], system_temp)

    def test_a_relative_folder_in_the_setting_never_breaks_the_import(self, tmp_path):
        system_temp = tmp_path / "systemtemp"
        out, err = _run(tmp_path, home=tmp_path / "data", system_temp=system_temp,
                        config={"temp_location": "relative/folder"})
        assert "temp_location" in err
        assert _same(out["gettempdir"], system_temp)

    def test_a_relative_folder_in_the_environment_is_reported_and_ignored(self, tmp_path):
        system_temp = tmp_path / "systemtemp"
        out, err = _run(tmp_path, home=tmp_path / "data", system_temp=system_temp,
                        extra_env={"LOCALM_TMPDIR": "relative/folder"})
        assert "LOCALM_TMPDIR" in err
        assert _same(out["gettempdir"], system_temp)

    def test_the_environment_variable_wins_over_the_setting(self, tmp_path):
        chosen = tmp_path / "from-env"
        out, _ = _run(tmp_path, home=tmp_path / "data", system_temp=tmp_path / "st",
                      config={"temp_location": "system"},
                      extra_env={"LOCALM_TMPDIR": str(chosen)})
        assert _same(out["gettempdir"], chosen)
        assert chosen.is_dir()


class TestFailureIsSaidNotHidden:
    def test_an_uncreatable_temp_dir_warns_and_leaves_temp_alone(self, tmp_path):
        home = tmp_path / "data"
        home.mkdir()
        (home / "tmp").write_text("a file where the folder belongs", encoding="utf-8")
        system_temp = tmp_path / "systemtemp"
        out, err = _run(tmp_path, home=home, system_temp=system_temp,
                        config={"temp_location": "data"})
        assert "cannot create the temp folder" in err
        assert _same(out["gettempdir"], system_temp)

    def test_an_unreadable_config_file_falls_back_to_auto(self, tmp_path):
        home = tmp_path / "data"
        home.mkdir()
        (home / "config.json").write_text("{ not json", encoding="utf-8")
        system_temp = tmp_path / "systemtemp"
        out, _ = _run(tmp_path, home=home, system_temp=system_temp)
        assert _same(out["gettempdir"], system_temp)


class TestAHomeFolderThatCannotBeResolved:
    """``~user`` for a user that does not exist makes ``Path.expanduser`` raise
    ``RuntimeError`` on POSIX. Simulated here instead of with a real unknown user,
    because Windows resolves ``~name`` to a sibling of the current user folder."""

    @pytest.fixture
    def failing_expanduser(self, monkeypatch):
        def boom(self):
            raise RuntimeError("Could not determine home directory.")
        monkeypatch.setattr(Path, "expanduser", boom)

    def test_the_folder_helper_reports_none(self, failing_expanduser):
        from localm import config
        assert config._absolute_folder("~nosuchuser/tmp") is None

    def test_the_setting_falls_back_to_auto_with_a_warning(
            self, failing_expanduser, tmp_path, capsys):
        from localm import config
        home = tmp_path / "data"
        home.mkdir()
        (home / "config.json").write_text(
            json.dumps({"temp_location": "~nosuchuser/tmp"}), encoding="utf-8")
        assert config.temp_dir(home) is None          # auto outside a checkout
        assert "temp_location" in capsys.readouterr().err

    def test_the_environment_variable_is_ignored_with_a_warning(
            self, failing_expanduser, tmp_path, monkeypatch, capsys):
        from localm import config
        monkeypatch.setenv("LOCALM_TMPDIR", "~nosuchuser/tmp")
        assert config.temp_dir(tmp_path / "data") is None
        assert "LOCALM_TMPDIR" in capsys.readouterr().err
