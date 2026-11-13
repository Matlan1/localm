# SPDX-License-Identifier: AGPL-3.0-or-later
"""Reading the data folder and its files: a relative LOCALM_HOME or
localm-home.cfg line does not follow the working folder, a config file saved
with a byte-order mark is read, and an instance id file that cannot be read or
written never changes the id within a run or gets overwritten."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from localm import config, storekit


def test_a_relative_localm_home_stays_where_it_was_first_read(tmp_path, monkeypatch):
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    monkeypatch.setenv("LOCALM_HOME", "rel-home")
    monkeypatch.chdir(first)
    home = config.home_dir()
    assert home == first / "rel-home"
    monkeypatch.chdir(second)
    assert config.home_dir() == first / "rel-home"
    assert Path(config.os.environ["LOCALM_HOME"]) == first / "rel-home"


def test_an_absolute_localm_home_is_kept_as_given(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path / "abs"))
    assert config.home_dir() == tmp_path / "abs"
    assert config.os.environ["LOCALM_HOME"] == str(tmp_path / "abs")


def test_a_relative_home_cfg_line_is_relative_to_the_checkout(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    repo = tmp_path / "checkout"
    assert config._home_from_cfg_line("data", repo) == repo / "data"
    assert config._home_from_cfg_line(str(tmp_path / "x"), repo) == tmp_path / "x"


def test_a_config_file_with_a_byte_order_mark_is_read(tmp_path):
    path = tmp_path / "config.json"
    path.write_bytes(b"\xef\xbb\xbf" + json.dumps({"net_mode": "off"}).encode("utf-8"))
    value, state = storekit.read_json_state(path, {})
    assert state == storekit.READ_CURRENT
    assert value == {"net_mode": "off"}


@pytest.fixture
def id_home(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "HOME_DIR", tmp_path)
    monkeypatch.setattr(config, "ensure_dirs", lambda: None)
    monkeypatch.setattr(config, "_RUN_INSTANCE_IDS", {})
    return tmp_path / "instance_id.txt"


def test_an_unreadable_instance_id_file_is_not_overwritten_and_the_id_holds(id_home, monkeypatch):
    id_home.write_text("the-stored-id", encoding="utf-8")
    real = Path.read_text

    def refuse(self, *a, **kw):
        if self == id_home:
            raise PermissionError(13, "held by another program")
        return real(self, *a, **kw)

    monkeypatch.setattr(Path, "read_text", refuse)
    first = config.instance_id()
    assert config.instance_id() == first
    assert first != "the-stored-id"
    monkeypatch.setattr(Path, "read_text", real)
    assert id_home.read_text(encoding="utf-8") == "the-stored-id"
    assert config.instance_id() == "the-stored-id"


def test_an_unwritable_instance_id_file_gives_one_id_per_run(id_home, monkeypatch):
    real = Path.write_text
    writes = []

    def refuse(self, *a, **kw):
        if self == id_home:
            writes.append(self)
            raise PermissionError(13, "read-only folder")
        return real(self, *a, **kw)

    monkeypatch.setattr(Path, "write_text", refuse)
    first = config.instance_id()
    assert [config.instance_id() for _ in range(3)] == [first] * 3
    assert len(writes) == 1
    assert not id_home.exists()


def test_a_readable_instance_id_file_is_reused(id_home):
    first = config.instance_id()
    assert id_home.read_text(encoding="utf-8") == first
    assert config.instance_id() == first
