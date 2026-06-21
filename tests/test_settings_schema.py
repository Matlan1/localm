# SPDX-License-Identifier: AGPL-3.0-or-later
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


def test_every_visible_field_has_a_description():
    """The settings overhaul requires EVERY rendered field to carry a clear
    description (HIDDEN fields are not rendered, so they are exempt)."""
    missing = [f.key for f in ss.CORE_FIELDS
               if f.widget != ss.Widget.HIDDEN and not (f.help or "").strip()]
    assert not missing, f"fields missing a description: {missing}"


def test_every_field_has_a_label():
    missing = [f.key for f in ss.CORE_FIELDS if not (f.label or "").strip()]
    assert not missing, f"fields missing a label: {missing}"


def test_binary_dir_schema_exposes_auto_resolved_path():
    """Blank binary_dir must surface the auto-detected path so the GUI can show
    it (the 'blank autodetect leaves the field empty' complaint)."""
    by_key = {f["key"]: f for f in ss.schema_json()}
    assert "auto" in by_key["binary_dir"], "binary_dir must carry an 'auto' value"


def test_new_comfy_fields_present_and_owned_by_image():
    by_key = {f.key: f for f in ss.CORE_FIELDS}
    for key in ("comfy_api_url", "comfy_fast_dequant"):
        assert key in by_key, f"{key} missing from the schema"
        assert by_key[key].owner == "image"
        assert by_key[key].group == "ComfyUI"
