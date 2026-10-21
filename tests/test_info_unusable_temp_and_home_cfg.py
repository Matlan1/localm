# SPDX-License-Identifier: AGPL-3.0-or-later
"""``localm info`` names the temp folder that is actually in use, and an unreadable
``localm-home.cfg`` is reported once per process."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from localm import config

ROOT = Path(__file__).resolve().parents[1]


def _info(tmp_path, tmpdir_value):
    env = dict(os.environ)
    env["LOCALM_HOME"] = str(tmp_path / "home")
    env["LOCALM_TMPDIR"] = tmpdir_value
    env["COLUMNS"] = "400"
    env["PYTHONPATH"] = str(ROOT)
    for var in ("LOCALM_SETUP",):
        env.pop(var, None)
    r = subprocess.run([sys.executable, "-m", "localm", "info"], env=env, cwd=str(tmp_path),
                       capture_output=True, text=True, timeout=180)
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout, r.stderr


def _temp_row(stdout: str) -> str:
    for line in stdout.splitlines():
        if line.strip().startswith("temp dir"):
            return "".join(line.split())
    raise AssertionError(f"no temp dir row in: {stdout!r}")


def test_info_reports_the_system_temp_folder_when_the_chosen_one_cannot_be_created(tmp_path):
    blocker = tmp_path / "a-file"
    blocker.write_text("x", encoding="utf-8")
    unusable = blocker / "tmp"
    out, err = _info(tmp_path, str(unusable))
    assert err.count("cannot create the temp folder") == 1
    assert _temp_row(out) == "tempdirsystemtempfolder"


def test_info_reports_a_temp_folder_that_was_created(tmp_path):
    chosen = tmp_path / "chosen"
    out, err = _info(tmp_path, str(chosen))
    assert "cannot create" not in err
    assert chosen.is_dir()
    assert _temp_row(out) == "tempdir" + "".join(str(chosen).split())


def test_temp_dir_in_use_ignores_a_folder_that_failed_at_start_up(tmp_path, monkeypatch):
    chosen = tmp_path / "chosen"
    monkeypatch.setenv(config.TEMP_DIR_ENV, str(chosen))
    assert config.temp_dir_in_use(tmp_path / "data") == chosen
    monkeypatch.setattr(config, "_unusable_temp_dirs", {os.path.abspath(str(chosen))})
    assert config.temp_dir_in_use(tmp_path / "data") is None


def test_an_unreadable_home_cfg_is_reported_once(tmp_path, monkeypatch, capsys):
    (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
    (tmp_path / "localm-home.cfg").write_text("x", encoding="utf-8")
    monkeypatch.setattr(config, "__file__", str(tmp_path / "localm" / "config.py"))
    monkeypatch.delenv("LOCALM_HOME", raising=False)
    monkeypatch.setattr(config, "_warned_unreadable_home_cfg", set())

    def unreadable(path):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(config, "_read_home_cfg", unreadable)
    capsys.readouterr()
    for _ in range(7):
        config._detect_home()
    err = capsys.readouterr().err
    assert err.count("cannot read") == 1
    assert "falling back to the default data dir" in err


def test_temp_dir_in_use_does_not_repeat_the_start_up_warnings(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv(config.TEMP_DIR_ENV, "relative/folder")
    capsys.readouterr()
    config.temp_dir_in_use(tmp_path / "data")
    assert capsys.readouterr().err == ""
    config.temp_dir(tmp_path / "data")
    assert "is not an absolute folder" in capsys.readouterr().err
