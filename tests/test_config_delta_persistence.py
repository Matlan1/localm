# SPDX-License-Identifier: AGPL-3.0-or-later
"""save_config / update_config persist only the user-set delta.

Writing the full defaults-merged dict to config.json would freeze every default
at its then-current value on the user's first save, so a later change to a
DEFAULT_CONFIG value never reaches an existing install. A key equal to the
current default is dropped at save time and reconstructed by load_config(), so
shipped default-value fixes propagate; a differing value is kept.
"""

import json

import pytest

import localm.config as cfg


@pytest.fixture
def config_file(tmp_path, monkeypatch):
    f = tmp_path / "config.json"
    monkeypatch.setattr(cfg, "CONFIG_FILE", f)
    return f


def _stored(config_file) -> dict:
    return json.loads(config_file.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
#  A changed default reaches an existing install                               #
# --------------------------------------------------------------------------- #

def test_changed_default_reaches_config_saved_under_old_default(
        config_file, monkeypatch):
    # An "old release" whose max_tokens default is 1024.
    monkeypatch.setitem(cfg.DEFAULT_CONFIG, "max_tokens", 1024)
    # The user changes an unrelated setting; the config is saved.
    c = cfg.load_config()
    c["temperature"] = 0.5
    cfg.save_config(c)
    # max_tokens equalled the then-current default, so it is not on disk.
    assert "max_tokens" not in _stored(config_file)
    # A "new release" ships max_tokens 4096. It must reach this install.
    monkeypatch.setitem(cfg.DEFAULT_CONFIG, "max_tokens", 4096)
    loaded = cfg.load_config()
    assert loaded["max_tokens"] == 4096       # the changed default propagated
    assert loaded["temperature"] == 0.5       # the user's setting survived


def test_user_set_value_survives_a_default_change(config_file, monkeypatch):
    monkeypatch.setitem(cfg.DEFAULT_CONFIG, "max_tokens", 1024)
    c = cfg.load_config()
    c["max_tokens"] = 2048                    # deliberate user choice
    cfg.save_config(c)
    monkeypatch.setitem(cfg.DEFAULT_CONFIG, "max_tokens", 4096)
    assert cfg.load_config()["max_tokens"] == 2048


def test_old_full_dump_migrates_on_next_save(config_file, monkeypatch):
    """A config.json written by the old full-dump scheme converges to the
    delta on its next save: keys equal to the CURRENT default are dropped
    (safe), a key differing from it is kept - it cannot be told apart from a
    user choice (the documented one-time ambiguity), so the user-choice
    reading wins."""
    full = json.loads(json.dumps(cfg.DEFAULT_CONFIG))  # JSON-clean deep copy
    full["max_tokens"] = 1024                 # frozen OLD default
    full["port"] = 9999                       # genuine user choice
    config_file.write_text(json.dumps(full), encoding="utf-8")

    cfg.update_config(lambda c: c.update({"temperature": 0.5}))

    stored = _stored(config_file)
    assert set(stored) == {"max_tokens", "port", "temperature", "config_version"}
    assert stored["config_version"] == cfg.CURRENT_CONFIG_VERSION
    assert stored["max_tokens"] == 1024       # kept: provenance was destroyed
    assert stored["port"] == 9999
    assert stored["temperature"] == 0.5


# --------------------------------------------------------------------------- #
#  Delta rules                                                                 #
# --------------------------------------------------------------------------- #

def test_default_valued_keys_are_not_written(config_file):
    cfg.save_config(cfg.load_config())        # nothing user-set
    assert _stored(config_file) == {}
    # ... and the merged view is still complete.
    loaded = cfg.load_config()
    for key in cfg.DEFAULT_CONFIG:
        assert key in loaded


def test_explicit_non_default_value_is_preserved(config_file):
    c = cfg.load_config()
    c["port"] = 9999
    c["ctx_auto"] = False                     # falsy user value survives too
    cfg.save_config(c)
    stored = _stored(config_file)
    assert stored == {"port": 9999, "ctx_auto": False}
    assert cfg.load_config()["port"] == 9999
    assert cfg.load_config()["ctx_auto"] is False


def test_unknown_keys_are_kept_verbatim(config_file):
    # Version-scoped state (plugins_first_use_done) and keys from a newer
    # localm version are not in DEFAULT_CONFIG; they must round-trip.
    cfg.update_config(
        lambda c: c.update({"plugins_first_use_done": ["tts"]}))
    cfg.update_config(lambda c: c.update({"temperature": 0.5}))
    stored = _stored(config_file)
    assert stored["plugins_first_use_done"] == ["tts"]
    assert cfg.load_config()["plugins_first_use_done"] == ["tts"]


def test_nested_user_block_is_written_verbatim(config_file):
    # The user owns "plugins": a non-default block is stored as-is, never
    # deep-merged with anything at save time.
    c = cfg.load_config()
    c["plugins"] = {"mine": {"enabled": True}}
    cfg.save_config(c)
    assert _stored(config_file)["plugins"] == {"mine": {"enabled": True}}


def test_update_config_writes_the_delta(config_file):
    cfg.update_config(lambda c: c.update({"n_ctx": 2048}))
    assert _stored(config_file) == {"n_ctx": 2048}
    # The mutator and return value still see the full merged dict.
    result = cfg.update_config(lambda c: None)
    for key in cfg.DEFAULT_CONFIG:
        assert key in result


def test_caller_dict_is_not_mutated(config_file):
    partial = {"n_ctx": 2048}
    cfg.save_config(partial)
    assert partial == {"n_ctx": 2048}


def test_loaded_config_does_not_alias_default_config(config_file):
    # load_config deep-copies the defaults: mutating a loaded nested value
    # must never write through into DEFAULT_CONFIG.
    cfg.save_config({})
    loaded = cfg.load_config()
    loaded["key_presets"].append({"name": "evil", "scopes": []})
    assert all(p.get("name") != "evil" for p in cfg.DEFAULT_CONFIG["key_presets"])


# --------------------------------------------------------------------------- #
#  Migration table, save log and model references                              #
# --------------------------------------------------------------------------- #

@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / "home"
    h.mkdir()
    monkeypatch.setattr(cfg, "HOME_DIR", h)
    monkeypatch.setattr(cfg, "MODELS_DIR", h / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", h / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", h / "registry.json")
    monkeypatch.setattr(cfg, "_warned_unknown_stored_keys", set())
    monkeypatch.setattr(cfg, "_warned_newer_config_version", set())
    return h


def _write(path, data) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


def _saved_events(home) -> list:
    log = home / "logs" / "management.jsonl"
    if not log.exists():
        return []
    events = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    return [e for e in events if e["kind"] == "config_saved"]


@pytest.mark.parametrize("raw, expected", [
    (1, 1), (7, 7), (0, 0), (-1, 0), (True, 0), ("2", 0), (2.0, 0), (None, 0)])
def test_stored_config_version_accepts_only_plain_non_negative_ints(raw, expected):
    assert cfg._stored_config_version({cfg.CONFIG_VERSION_KEY: raw}) == expected


def test_stored_config_version_is_zero_when_absent():
    assert cfg._stored_config_version({"max_tokens": 5}) == 0


def test_config_rename_builds_a_rename_step():
    step = cfg._config_rename(("a",), ("b",))
    assert step == {"kind": "rename", "old": ("a",), "new": ("b",), "value_map": None}


def test_rename_step_moves_the_value_into_a_nested_path():
    stored = {"x": 5, "keep": 1}
    cfg._apply_config_step(stored, cfg._config_rename(("x",), ("a", "b", "c")))
    assert stored == {"keep": 1, "a": {"b": {"c": 5}}}


def test_rename_step_moves_a_nested_value_and_applies_its_value_map():
    stored = {"p": {"q": 3}}
    cfg._apply_config_step(
        stored, cfg._config_rename(("p", "q"), ("p", "r"), lambda v: v * 10))
    assert stored == {"p": {"r": 30}}


def test_rename_step_without_value_map_keeps_the_value():
    stored = {"x": [1, 2]}
    cfg._apply_config_step(stored, cfg._config_rename(("x",), ("y",)))
    assert stored == {"y": [1, 2]}


def test_rename_step_never_overwrites_an_existing_target():
    stored = {"x": 1, "y": 2}
    cfg._apply_config_step(stored, cfg._config_rename(("x",), ("y",)))
    assert stored == {"y": 2}


def test_rename_step_with_absent_source_changes_nothing():
    stored = {"y": 2, "p": "not a dict"}
    cfg._apply_config_step(stored, cfg._config_rename(("x",), ("z",)))
    cfg._apply_config_step(stored, cfg._config_rename(("p", "q"), ("z",)))
    assert stored == {"y": 2, "p": "not a dict"}


def test_remove_step_drops_the_key():
    stored = {"gone": 1, "keep": 2}
    cfg._apply_config_step(stored, cfg._config_remove(("gone",)))
    assert stored == {"keep": 2}


def test_migrations_map_a_stored_boolean_remote_image_value():
    stored = {"gui_proxy_remote_images": True}
    cfg._apply_config_migrations(stored)
    assert stored == {"gui_proxy_remote_images": cfg.REMOTE_IMAGE_ON,
                      cfg.CONFIG_VERSION_KEY: cfg.CURRENT_CONFIG_VERSION}


def test_migrations_at_the_current_version_neither_change_nor_warn(home):
    stored = {"max_tokens": 5, cfg.CONFIG_VERSION_KEY: cfg.CURRENT_CONFIG_VERSION}
    cfg._apply_config_migrations(stored)
    assert stored == {"max_tokens": 5,
                      cfg.CONFIG_VERSION_KEY: cfg.CURRENT_CONFIG_VERSION}
    assert cfg._warned_newer_config_version == set()


def test_migrations_leave_a_newer_version_untouched_and_note_it(home):
    newer = cfg.CURRENT_CONFIG_VERSION + 1
    stored = {"gui_proxy_remote_images": True, cfg.CONFIG_VERSION_KEY: newer}
    cfg._apply_config_migrations(stored)
    assert stored == {"gui_proxy_remote_images": True, cfg.CONFIG_VERSION_KEY: newer}
    assert cfg._warned_newer_config_version == {newer}


def test_config_needs_migration_only_when_a_step_changes_a_setting():
    assert cfg._config_needs_migration({"gui_proxy_remote_images": True}) is True
    assert cfg._config_needs_migration({"max_tokens": 5}) is False
    assert cfg._config_needs_migration(
        {"max_tokens": 5, cfg.CONFIG_VERSION_KEY: 0}) is False
    assert cfg._config_needs_migration({}) is False


def test_config_needs_migration_does_not_mutate_a_nested_stored_block(monkeypatch):
    monkeypatch.setattr(cfg, "_CONFIG_STEPS",
                        (cfg._config_rename(("plugins", "x", "a"), ("plugins", "x", "b")),))
    monkeypatch.setattr(cfg, "CURRENT_CONFIG_VERSION", 1)
    stored = {"plugins": {"x": {"a": 1}}}
    assert cfg._config_needs_migration(stored) is True
    assert stored == {"plugins": {"x": {"a": 1}}}


def test_unknown_stored_keys_are_noted_and_known_ones_are_not(home):
    cfg._warn_unknown_stored_keys({
        cfg.CONFIG_VERSION_KEY: 1, "max_tokens": 5,
        "plugins_first_use_done": ["tts"], "zz_unknown_key": 1})
    assert cfg._warned_unknown_stored_keys == {"zz_unknown_key"}


def test_load_rewrites_a_file_whose_settings_were_migrated(home):
    _write(cfg.CONFIG_FILE, {"gui_proxy_remote_images": True})
    assert cfg.load_config()["gui_proxy_remote_images"] == cfg.REMOTE_IMAGE_ON
    assert _stored(cfg.CONFIG_FILE) == {
        "gui_proxy_remote_images": cfg.REMOTE_IMAGE_ON,
        cfg.CONFIG_VERSION_KEY: cfg.CURRENT_CONFIG_VERSION}


def test_changed_config_keys_names_added_removed_and_changed_keys():
    before = {"same": 1, "a": 1, "b": 2, "y": None, cfg.CONFIG_VERSION_KEY: 0}
    after = {"same": 1, "b": 3, "c": 4, "x": None, cfg.CONFIG_VERSION_KEY: 1}
    assert cfg._changed_config_keys(before, after) == ["a", "b", "c", "x", "y"]


def test_save_logs_only_the_keys_that_changed_on_disk(home):
    _write(cfg.CONFIG_FILE, {"max_tokens": 1234})
    c = cfg.load_config()
    c["port"] = 9911
    cfg.save_config(c)
    (event,) = _saved_events(home)
    assert event["keys"] == ["port"]


def test_save_with_nothing_changed_logs_nothing(home):
    _write(cfg.CONFIG_FILE, {"max_tokens": 1234})
    cfg.save_config(cfg.load_config())
    assert _saved_events(home) == []


def test_update_config_logs_a_change_inside_a_nested_block(home):
    cfg.update_config(lambda c: c["plugins"].setdefault("demo", {}).update(k=1))
    (event,) = _saved_events(home)
    assert event["keys"] == ["plugins"]


def _renamed_model_home(home) -> None:
    _write(cfg.REGISTRY_FILE, {"new-name": {"id": "id-abc", "path": "m.gguf"}})
    _write(cfg.CONFIG_FILE, {"default_model": "old-name",
                             "model_refs": {"old-name": "id-abc"}})


def test_load_follows_a_stored_model_reference_to_the_current_name(home):
    _renamed_model_home(home)
    loaded = cfg.load_config()
    assert loaded["default_model"] == "new-name"
    assert "model_refs" not in loaded


def test_update_config_follows_a_stored_model_reference_to_the_current_name(home):
    _renamed_model_home(home)
    seen = {}
    result = cfg.update_config(lambda c: seen.update(c))
    assert seen["default_model"] == "new-name"
    assert result["default_model"] == "new-name"
    assert "model_refs" not in seen
    assert _stored(cfg.CONFIG_FILE) == {
        "default_model": "new-name", "model_refs": {"new-name": "id-abc"},
        cfg.CONFIG_VERSION_KEY: cfg.CURRENT_CONFIG_VERSION}
