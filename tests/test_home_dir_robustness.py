# SPDX-License-Identifier: AGPL-3.0-or-later
"""LOCALM_HOME is an explicit "keep my data here" choice, so localm creates it on
first use. ``ensure_dirs`` creates the whole path, parents included, so a nested
home whose PARENT does not exist yet works exactly like one whose parent does.
These pin that mkdir -p behavior."""

import json

import pytest

import localm.config as cfg


@pytest.fixture()
def home_at(tmp_path, monkeypatch):
    """Point config at a chosen home path (which may not exist yet)."""
    def _set(path):
        monkeypatch.setenv("LOCALM_HOME", str(path))
        monkeypatch.setattr(cfg, "HOME_DIR", path)
        monkeypatch.setattr(cfg, "MODELS_DIR", path / "models")
        monkeypatch.setattr(cfg, "CONFIG_FILE", path / "config.json")
        monkeypatch.setattr(cfg, "REGISTRY_FILE", path / "registry.json")
        return path
    return _set


def test_ensure_dirs_creates_nested_home(home_at, tmp_path):
    """A home several levels below an existing dir is created, parents and all."""
    home = tmp_path / "a" / "b" / "c" / "localm-data"
    home_at(home)
    assert not home.exists()
    cfg.ensure_dirs()  # must not raise
    assert home.is_dir()
    assert (home / "models").is_dir()


def test_load_config_on_nested_home(home_at, tmp_path):
    """The end-to-end read path works on a not-yet-existing nested home."""
    home = tmp_path / "deep" / "nested" / "home"
    home_at(home)
    conf = cfg.load_config()  # must not raise
    assert conf["port"] == cfg.DEFAULT_CONFIG["port"]
    assert home.is_dir()


def test_save_config_on_nested_home(home_at, tmp_path):
    """A write to a not-yet-existing nested home creates it and persists."""
    home = tmp_path / "x" / "y" / "z" / "home"
    home_at(home)
    cfg.save_config({**cfg.load_config(), "port": 9191})
    assert home.is_dir()
    on_disk = json.loads((home / "config.json").read_text())
    assert on_disk["port"] == 9191


def test_ensure_dirs_idempotent(home_at, tmp_path):
    """Re-running on an existing home is a no-op, not an error."""
    home = tmp_path / "home"
    home_at(home)
    cfg.ensure_dirs()
    cfg.ensure_dirs()  # exist_ok - must not raise
    assert home.is_dir()
