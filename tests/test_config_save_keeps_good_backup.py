# SPDX-License-Identifier: AGPL-3.0-or-later
"""A save after a read that fell back to the .bak keeps that .bak and keeps the
unreadable file under its own name."""

from __future__ import annotations

import json

from localm import config


def test_a_save_over_an_unreadable_file_keeps_the_good_backup(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"n_ctx": 8192, "max_tokens": 512, "theme": broken', encoding="utf-8")
    bak = tmp_path / "config.json.bak"
    bak.write_text('{"plugins_enabled": ["chat"], "n_ctx": 4096}', encoding="utf-8")

    stored, state = config._read_json_state(path, {})
    assert state == config.READ_FROM_BACKUP
    assert stored == {"plugins_enabled": ["chat"], "n_ctx": 4096}
    stored["max_tokens"] = 1024
    config._atomic_write_json(path, stored)

    assert json.loads(path.read_text(encoding="utf-8")) == {
        "plugins_enabled": ["chat"], "n_ctx": 4096, "max_tokens": 1024}
    assert json.loads(bak.read_text(encoding="utf-8")) == {
        "plugins_enabled": ["chat"], "n_ctx": 4096}
    kept = list(tmp_path.glob("config.json.unreadable-*"))
    assert len(kept) == 1
    assert kept[0].read_text(encoding="utf-8").startswith('{"n_ctx": 8192, "max_tokens": 512')


def test_a_save_over_a_readable_file_still_rotates_it_into_the_backup(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"max_tokens": 512}', encoding="utf-8")
    config._atomic_write_json(path, {"max_tokens": 1024})
    assert json.loads((tmp_path / "config.json.bak").read_text(encoding="utf-8")) == {
        "max_tokens": 512}
    assert not list(tmp_path.glob("config.json.unreadable-*"))


def test_the_file_is_present_at_every_step_of_a_save(tmp_path, monkeypatch):
    path = tmp_path / "config.json"
    path.write_text('{"max_tokens": 512}', encoding="utf-8")
    present = []
    real = config._replace_atomic

    def spy(src, dst):
        present.append(path.is_file())
        real(src, dst)
        present.append(path.is_file())

    monkeypatch.setattr(config, "_replace_atomic", spy)
    config._atomic_write_json(path, {"max_tokens": 1024})
    config._atomic_write_json(path, {"max_tokens": 2048})
    assert present and all(present), present
    assert json.loads(path.read_text(encoding="utf-8")) == {"max_tokens": 2048}
    assert json.loads((tmp_path / "config.json.bak").read_text(encoding="utf-8")) == {
        "max_tokens": 1024}
    assert not list(tmp_path.glob("*.tmp"))


def test_a_failed_backup_copy_leaves_no_temp_and_still_saves(tmp_path, monkeypatch):
    path = tmp_path / "config.json"
    path.write_text('{"max_tokens": 512}', encoding="utf-8")

    def refuse(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(config.shutil, "copyfile", refuse)
    config._atomic_write_json(path, {"max_tokens": 1024})
    assert json.loads(path.read_text(encoding="utf-8")) == {"max_tokens": 1024}
    assert not (tmp_path / "config.json.bak").exists()
    assert not list(tmp_path.glob("*.tmp"))
