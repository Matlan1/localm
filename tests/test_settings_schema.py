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
    for key in ("comfy_api_url", "comfy_fast_dequant", "comfy_delete_outputs"):
        assert key in by_key, f"{key} missing from the schema"
        assert by_key[key].owner == "image"
        assert by_key[key].group == "Media"


# --- owner-only RAG indexing settings (admin_only) -------------------------- #

RAG_OWNER_KEYS = {"rag_indexing_mode", "rag_allowed_roots", "rag_denied_roots"}


def test_rag_indexing_fields_are_owner_only():
    by_key = {f.key: f for f in ss.CORE_FIELDS}
    assert by_key["rag_indexing_mode"].widget == ss.Widget.SELECT
    assert by_key["rag_indexing_mode"].options == ["whitelist", "blacklist"]
    assert by_key["rag_allowed_roots"].widget == ss.Widget.PATHLIST
    assert by_key["rag_denied_roots"].widget == ss.Widget.PATHLIST
    for key in RAG_OWNER_KEYS:
        assert by_key[key].owner == "rag"
        assert by_key[key].admin_only is True, f"{key} must be owner-only"


def test_admin_only_keys_lists_the_owner_only_settings():
    # The rag_* folder keys widen a filesystem-read boundary; net_allow_private
    # disables the SSRF guard (a network trust boundary). Both are owner-only, so
    # a non-owner config:write key cannot flip either (pentest finding LM-PT-001).
    assert ss.admin_only_keys() == RAG_OWNER_KEYS | {"net_allow_private"}


def test_net_allow_private_is_admin_only():
    by_key = {f.key: f for f in ss.CORE_FIELDS}
    assert by_key["net_allow_private"].admin_only is True, \
        "the SSRF-guard-disable toggle must be owner-only"
    # A sibling network field the scoped key legitimately manages stays non-admin.
    assert by_key["net_allow"].admin_only is False


def test_schema_json_hides_admin_only_for_non_owner():
    owner = {f["key"] for f in ss.schema_json(is_owner=True)}
    guest = {f["key"] for f in ss.schema_json(is_owner=False)}
    assert RAG_OWNER_KEYS <= owner, "owner must see the owner-only fields"
    assert not (RAG_OWNER_KEYS & guest), "non-owner must NOT see any of them"
    # A normal (non-admin_only) field is unaffected by the owner filter.
    assert "mode" in owner and "mode" in guest
    # The owner view advertises the admin_only flag so the client/UI can label it.
    by_key = {f["key"]: f for f in ss.schema_json(is_owner=True)}
    assert by_key["rag_indexing_mode"].get("admin_only") is True
    # Default (no is_owner arg) behaves as owner, for the CLI and tests.
    assert RAG_OWNER_KEYS <= {f["key"] for f in ss.schema_json()}
