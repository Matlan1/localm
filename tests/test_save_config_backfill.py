# SPDX-License-Identifier: AGPL-3.0-or-later
"""save_config backfills missing top-level DEFAULT_CONFIG keys.

A config dict built by hand (rather than via load_config(), which merges the
defaults) must not silently drop a default key from disk on save. Explicit
user values, including non-default ones, are preserved verbatim, and nested
user-owned blocks like "plugins" are written as-is, never deep-merged."""

import json

import pytest

import localm.config as cfg


@pytest.fixture
def config_file(tmp_path, monkeypatch):
    f = tmp_path / "config.json"
    monkeypatch.setattr(cfg, "CONFIG_FILE", f)
    return f


def test_missing_default_keys_are_backfilled(config_file):
    cfg.save_config({"n_ctx": 2048})              # hand-built, misses most defaults
    stored = json.loads(config_file.read_text(encoding="utf-8"))
    assert stored["n_ctx"] == 2048
    for key in cfg.DEFAULT_CONFIG:
        assert key in stored
    assert stored["port"] == cfg.DEFAULT_CONFIG["port"]
    assert cfg.load_config()["port"] == cfg.DEFAULT_CONFIG["port"]


def test_explicit_non_default_value_is_preserved(config_file):
    cfg.save_config({"port": 9999, "ctx_auto": False})
    stored = json.loads(config_file.read_text(encoding="utf-8"))
    assert stored["port"] == 9999                 # user value, not the default
    assert stored["ctx_auto"] is False            # falsy user value survives too
    assert cfg.load_config()["port"] == 9999


def test_nested_user_block_is_not_deep_merged(config_file):
    # The user owns "plugins": a hand-written block is stored verbatim.
    cfg.save_config({"plugins": {"mine": {"enabled": True}}})
    stored = json.loads(config_file.read_text(encoding="utf-8"))
    assert stored["plugins"] == {"mine": {"enabled": True}}


def test_caller_dict_is_not_mutated(config_file):
    partial = {"n_ctx": 2048}
    cfg.save_config(partial)
    assert partial == {"n_ctx": 2048}


def test_backfill_does_not_corrupt_default_config(config_file):
    # The backfilled values must be copies: mutating the config on disk (or a
    # later loaded dict) must never write through into DEFAULT_CONFIG.
    cfg.save_config({})
    loaded = cfg.load_config()
    loaded["key_presets"].append({"name": "evil", "scopes": []})
    assert all(p.get("name") != "evil" for p in cfg.DEFAULT_CONFIG["key_presets"])
