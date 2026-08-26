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
    assert set(stored) == {"max_tokens", "port", "temperature"}
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
