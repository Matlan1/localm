# SPDX-License-Identifier: AGPL-3.0-or-later
"""In-process tests for the temp-folder helpers in localm.config: temp_dir,
contain_temp_dir, _is_portable_home and _stored_temp_location.

test_contained_temp.py drives them through fresh interpreters; these call the
functions directly so the checks run against the module under test."""

import json
import os
import tempfile
from pathlib import Path

import pytest

from localm import config


@pytest.fixture(autouse=True)
def _no_env_override(monkeypatch):
    monkeypatch.delenv(config.TEMP_DIR_ENV, raising=False)


def _home(tmp_path, settings=None, raw=None):
    home = tmp_path / "data"
    home.mkdir()
    if raw is not None:
        (home / "config.json").write_bytes(raw)
    elif settings is not None:
        (home / "config.json").write_text(json.dumps(settings), encoding="utf-8")
    return home


def _checkout_home():
    return Path(config.__file__).resolve().parents[1] / "scratch-home-that-does-not-exist"


class TestStoredTempLocation:
    def test_the_stored_value_is_returned_stripped(self, tmp_path):
        home = _home(tmp_path, {"temp_location": "  data  "})
        assert config._stored_temp_location(home) == "data"

    def test_a_byte_order_mark_is_tolerated(self, tmp_path):
        home = _home(tmp_path, raw=b"\xef\xbb\xbf" + b'{"temp_location": "system"}')
        assert config._stored_temp_location(home) == "system"

    def test_a_missing_file_gives_an_empty_string(self, tmp_path):
        assert config._stored_temp_location(_home(tmp_path)) == ""

    @pytest.mark.parametrize("raw", [b"not json", b"[1, 2]", b'{"temp_location": 5}',
                                     b'{"other": "data"}'])
    def test_unusable_content_gives_an_empty_string(self, tmp_path, raw):
        assert config._stored_temp_location(_home(tmp_path, raw=raw)) == ""

    def test_the_file_is_read_from_the_given_home(self, tmp_path):
        home = _home(tmp_path, {"temp_location": "data"})
        assert config._stored_temp_location(home) == "data"
        assert config._stored_temp_location(tmp_path) == ""


class TestIsPortableHome:
    def test_a_folder_inside_a_checkout_is_portable(self):
        assert config._is_portable_home(_checkout_home()) is True

    def test_a_folder_outside_the_checkout_is_not(self, tmp_path):
        assert config._is_portable_home(tmp_path / "data") is False

    def test_the_package_must_sit_beside_a_pyproject(self, tmp_path, monkeypatch):
        pkg = tmp_path / "pkg"
        pkg.mkdir()
        monkeypatch.setattr(config, "__file__", str(pkg / "config.py"))
        assert config._is_portable_home(tmp_path / "data") is False
        (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
        assert config._is_portable_home(tmp_path / "data") is True

    def test_only_the_parent_of_the_package_folder_is_the_root(self, tmp_path, monkeypatch):
        top = tmp_path / "top"
        pkg = top / "pkg"
        pkg.mkdir(parents=True)
        (top / "pyproject.toml").write_text("", encoding="utf-8")
        monkeypatch.setattr(config, "__file__", str(pkg / "config.py"))
        assert config._is_portable_home(tmp_path / "elsewhere") is False
        assert config._is_portable_home(top / "data") is True


class TestTempDir:
    def test_an_absolute_environment_folder_wins(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv(config.TEMP_DIR_ENV, str(tmp_path / "env"))
        home = _home(tmp_path, {"temp_location": "data"})
        assert config.temp_dir(home) == tmp_path / "env"
        assert capsys.readouterr().err == ""

    def test_a_relative_environment_folder_is_reported_and_ignored(
            self, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv(config.TEMP_DIR_ENV, "relative/folder")
        assert config.temp_dir(_home(tmp_path)) is None
        assert capsys.readouterr().err == (
            "[localm] WARNING: LOCALM_TMPDIR='relative/folder' is not an absolute "
            "folder; ignoring it.\n")

    def test_no_environment_variable_prints_nothing(self, tmp_path, capsys):
        assert config.temp_dir(_home(tmp_path)) is None
        assert capsys.readouterr().err == ""

    @pytest.mark.parametrize("word", ["data", "DATA", " Data "])
    def test_data_means_a_tmp_folder_in_the_data_dir(self, tmp_path, word):
        home = _home(tmp_path, {"temp_location": word})
        assert config.temp_dir(home) == home / "tmp"

    @pytest.mark.parametrize("word", ["system", "SYSTEM"])
    def test_system_leaves_the_system_temp_folder_even_in_a_checkout(self, tmp_path, word):
        home = _home(tmp_path, {"temp_location": word})
        assert config.temp_dir(home) is None

    def test_system_wins_inside_a_checkout(self, monkeypatch):
        monkeypatch.setattr(config, "_stored_temp_location", lambda home: "system")
        assert config.temp_dir(_checkout_home()) is None

    def test_an_absolute_setting_is_used(self, tmp_path):
        home = _home(tmp_path, {"temp_location": str(tmp_path / "chosen")})
        assert config.temp_dir(home) == tmp_path / "chosen"

    @pytest.mark.parametrize("word", ["auto", "AUTO"])
    def test_auto_follows_the_checkout_rule(self, tmp_path, word, capsys):
        outside = _home(tmp_path, {"temp_location": word})
        assert config.temp_dir(outside) is None
        assert capsys.readouterr().err == ""
        inside = _checkout_home()
        os.environ.pop(config.TEMP_DIR_ENV, None)
        assert config.temp_dir(inside) == inside / "tmp"
        assert capsys.readouterr().err == ""

    def test_a_relative_setting_is_reported_and_treated_as_auto(self, tmp_path, capsys):
        home = _home(tmp_path, {"temp_location": "some/where"})
        assert config.temp_dir(home) is None
        assert capsys.readouterr().err == (
            "[localm] WARNING: temp_location 'some/where' in config.json is not "
            "'auto', 'data', 'system' or an absolute folder; using 'auto'.\n")

    def test_the_default_home_is_the_active_data_dir(self, monkeypatch):
        seen = []
        monkeypatch.setattr(config, "home_dir", lambda: _checkout_home())
        monkeypatch.setattr(config, "_stored_temp_location",
                            lambda home: seen.append(home) or "data")
        assert config.temp_dir() == _checkout_home() / "tmp"
        assert seen == [_checkout_home()]

    def test_a_given_home_is_used_as_given(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "home_dir", lambda: pytest.fail("home_dir used"))
        home = _home(tmp_path, {"temp_location": "data"})
        assert config.temp_dir(str(home)) == home / "tmp"


class TestContainTempDir:
    @pytest.fixture(autouse=True)
    def _restore(self, monkeypatch):
        for var in ("TMP", "TEMP", "TMPDIR"):
            monkeypatch.setenv(var, os.environ.get(var, ""))
        monkeypatch.setattr(tempfile, "tempdir", tempfile.tempdir)

    def test_nothing_changes_when_there_is_no_target(self, monkeypatch):
        monkeypatch.setattr(config, "temp_dir", lambda: None)
        before = (dict(os.environ), tempfile.tempdir)
        config.contain_temp_dir()
        assert (dict(os.environ), tempfile.tempdir) == before

    def test_the_folder_is_created_and_every_variable_points_at_it(self, tmp_path, monkeypatch):
        target = tmp_path / "made" / "tmp"
        monkeypatch.setattr(config, "temp_dir", lambda: target)
        config.contain_temp_dir()
        assert target.is_dir()
        for var in ("TMP", "TEMP", "TMPDIR"):
            assert os.environ[var] == os.path.abspath(str(target))
        assert tempfile.tempdir == os.path.abspath(str(target))

    def test_an_existing_folder_is_accepted(self, tmp_path, monkeypatch):
        target = tmp_path / "already" / "tmp"
        target.mkdir(parents=True)
        monkeypatch.setattr(config, "temp_dir", lambda: target)
        config.contain_temp_dir()
        assert tempfile.tempdir == os.path.abspath(str(target))
