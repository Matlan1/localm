# SPDX-License-Identifier: AGPL-3.0-or-later
"""config.json carries a config_version and an ordered table of rename and
removal steps; a stored key the build does not know warns once and stays."""

import json

import pytest

import localm.config as cfg


class _Recorder:
    def __init__(self):
        self.records = []

    def warning(self, msg, *args):
        self.records.append(msg % args if args else msg)


@pytest.fixture
def config_file(tmp_path, monkeypatch):
    f = tmp_path / "config.json"
    monkeypatch.setattr(cfg, "CONFIG_FILE", f)
    monkeypatch.setattr(cfg, "_warned_unknown_stored_keys", set())
    monkeypatch.setattr(cfg, "_warned_newer_config_version", set())
    return f


@pytest.fixture
def warnings(monkeypatch):
    rec = _Recorder()
    monkeypatch.setattr(cfg, "_logger", lambda: rec)
    return rec.records


def _write(f, data):
    f.write_text(json.dumps(data), encoding="utf-8")


def _stored(f):
    return json.loads(f.read_text(encoding="utf-8"))


def _steps(monkeypatch, *steps):
    monkeypatch.setattr(cfg, "_CONFIG_STEPS", tuple(steps))
    monkeypatch.setattr(cfg, "CURRENT_CONFIG_VERSION", len(steps))


def test_version_zero_file_gets_the_renamed_key_with_the_users_value(
        config_file, monkeypatch, warnings):
    _steps(monkeypatch, cfg._config_rename(("old_name",), ("max_tokens",)))
    _write(config_file, {"old_name": 77, "temperature": 0.5})

    loaded = cfg.load_config()

    assert loaded["max_tokens"] == 77
    assert "old_name" not in loaded
    on_disk = _stored(config_file)
    assert on_disk["max_tokens"] == 77
    assert "old_name" not in on_disk
    assert on_disk["temperature"] == 0.5
    assert on_disk["config_version"] == 1
    assert warnings == []


def test_rename_keeps_a_value_already_set_under_the_new_key(
        config_file, monkeypatch, warnings):
    _steps(monkeypatch, cfg._config_rename(("old_name",), ("max_tokens",)))
    _write(config_file, {"old_name": 77, "max_tokens": 99})

    assert cfg.load_config()["max_tokens"] == 99
    assert "old_name" not in _stored(config_file)


def test_removal_step_drops_the_key_without_a_warning(
        config_file, monkeypatch, warnings):
    _steps(monkeypatch, cfg._config_remove(("gone_key",)))
    _write(config_file, {"gone_key": 1, "temperature": 0.5})

    cfg.load_config()

    assert "gone_key" not in _stored(config_file)
    assert warnings == []


def test_steps_run_in_order_from_the_stored_version(config_file, monkeypatch):
    _steps(monkeypatch,
           cfg._config_rename(("first",), ("second",)),
           cfg._config_rename(("second",), ("max_tokens",)))
    _write(config_file, {"first": 5})
    assert cfg.load_config()["max_tokens"] == 5

    _write(config_file, {"config_version": 1, "first": 5, "second": 6})
    cfg.load_config()
    on_disk = _stored(config_file)
    assert on_disk["max_tokens"] == 6
    assert on_disk["first"] == 5
    assert on_disk["config_version"] == 2


def test_nested_key_path_is_renamed(config_file, monkeypatch):
    _steps(monkeypatch,
           cfg._config_rename(("plugins", "x", "old"), ("plugins", "x", "new")))
    _write(config_file, {"plugins": {"x": {"old": 3}}})

    cfg.load_config()

    assert _stored(config_file)["plugins"]["x"] == {"new": 3}


def test_remote_image_boolean_is_mapped_and_the_file_is_stamped(config_file):
    _write(config_file, {"gui_proxy_remote_images": True})

    assert cfg.load_config()["gui_proxy_remote_images"] == "on"

    on_disk = _stored(config_file)
    assert on_disk["gui_proxy_remote_images"] == "on"
    assert on_disk["config_version"] == cfg.CURRENT_CONFIG_VERSION


def test_unknown_key_warns_once_names_the_key_not_the_value_and_stays(
        config_file, warnings):
    _write(config_file, {"mystery_setting": "s3cr3t-value", "config_version": 1})

    for _ in range(3):
        cfg.load_config()

    mine = [w for w in warnings if "mystery_setting" in w]
    assert len(mine) == 1
    assert "s3cr3t-value" not in mine[0]
    assert _stored(config_file)["mystery_setting"] == "s3cr3t-value"


def test_known_keys_do_not_warn(config_file, warnings):
    _write(config_file, {"temperature": 0.5, "plugins_first_use_done": ["a"],
                         "config_version": cfg.CURRENT_CONFIG_VERSION})

    cfg.load_config()

    assert warnings == []


def test_newer_version_is_read_without_migrating_and_not_rewritten(
        config_file, monkeypatch, warnings):
    _steps(monkeypatch, cfg._config_rename(("old_name",), ("max_tokens",)))
    _write(config_file, {"config_version": 9, "old_name": 77, "temperature": 0.5})
    before_bytes = config_file.read_bytes()
    before_mtime = config_file.stat().st_mtime_ns

    loaded = cfg.load_config()
    cfg.load_config()

    assert loaded["temperature"] == 0.5
    assert config_file.read_bytes() == before_bytes
    assert config_file.stat().st_mtime_ns == before_mtime
    newer = [w for w in warnings if "config_version" in w]
    assert len(newer) == 1


def test_current_version_file_is_not_rewritten_on_load(config_file):
    _write(config_file, {"config_version": cfg.CURRENT_CONFIG_VERSION,
                         "temperature": 0.5})
    before = config_file.stat().st_mtime_ns

    cfg.load_config()

    assert config_file.stat().st_mtime_ns == before


def test_absent_file_is_not_created_by_load(config_file):
    cfg.load_config()

    assert not config_file.exists()


def test_config_version_is_not_a_settable_key(config_file):
    from localm import settings_schema

    assert cfg.CONFIG_VERSION_KEY not in {f.key for f in settings_schema.CORE_FIELDS}
    assert cfg.CONFIG_VERSION_KEY not in cfg.DEFAULT_CONFIG
