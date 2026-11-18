"""The chat style note is off until the user turns it on."""

from localm import config
from localm.plugins.builtin.chat import plug
from localm.settings_schema import CORE_FIELDS


def test_the_style_note_default_is_off():
    assert config.DEFAULT_CONFIG["chat_style_note"] is False


def test_a_config_without_the_key_leaves_the_note_off(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(config, "HOME_DIR", tmp_path)
    assert plug._style_note_enabled() is False


def test_the_settings_toggle_says_it_is_off_by_default():
    field = next(f for f in CORE_FIELDS if f.key == "chat_style_note")
    assert field.help.startswith("Off by default")
