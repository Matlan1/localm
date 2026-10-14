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

    stored, read_ok = config._read_json_checked(path, {})
    assert read_ok and stored == {"plugins_enabled": ["chat"], "n_ctx": 4096}
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
