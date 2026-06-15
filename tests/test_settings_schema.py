"""Tests for the settings schema (localm/settings_schema.py)."""

import json

from localm import settings_schema as ss
from localm.config import DEFAULT_CONFIG


def test_schema_matches_config_keys_exactly():
    """Guards drift: every config key has a field and vice versa."""
    field_keys = {f.key for f in ss.CORE_FIELDS}
    cfg_keys = set(DEFAULT_CONFIG)
    assert field_keys == cfg_keys, (
        f"only-in-config={cfg_keys - field_keys}, "
        f"only-in-schema={field_keys - cfg_keys}")


def test_no_duplicate_field_keys():
    keys = [f.key for f in ss.CORE_FIELDS]
    assert len(keys) == len(set(keys))


def test_select_fields_have_options():
    for f in ss.CORE_FIELDS:
        if f.widget == ss.Widget.SELECT:
            assert f.options, f"{f.key} is SELECT but has no options"


def test_widgets_are_known():
    valid = ss.all_widgets()
    for f in ss.CORE_FIELDS:
        assert f.widget in valid, f"{f.key} has unknown widget {f.widget!r}"


def test_applies_are_known():
    valid = {ss.Applies.LIVE, ss.Applies.NEXT_LOAD, ss.Applies.RESTART}
    for f in ss.CORE_FIELDS:
        assert f.applies in valid


def test_schema_json_serializable_with_defaults():
    js = ss.schema_json()
    json.dumps(js)   # must not raise
    by_key = {f["key"]: f for f in js}
    assert by_key["temperature"]["default"] == DEFAULT_CONFIG["temperature"]
    assert by_key["mode"]["default"] == DEFAULT_CONFIG["mode"]
    assert by_key["mode"]["options"] == ["privacy", "log", "full"]


def test_fields_by_owner_partitions():
    """Every field's owner is 'core' or a known/plugin-name scope."""
    from localm import scopes
    for f in ss.CORE_FIELDS:
        assert f.owner == "core" or scopes.is_valid_scope(f.owner)
    assert ss.fields_by_owner("web"), "expected web-owned network settings"
