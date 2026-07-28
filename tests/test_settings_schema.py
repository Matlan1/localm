# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the settings schema (localm/settings_schema.py)."""

import json

import pytest

from localm import settings_schema as ss
from localm.config import DEFAULT_CONFIG


def test_schema_matches_config_keys_exactly():
    """Guards drift: every config key has a field and vice versa."""
    field_keys = {f.key for f in ss.CORE_FIELDS}
    cfg_keys = set(DEFAULT_CONFIG)
    assert field_keys == cfg_keys, (
        f"only-in-config={cfg_keys - field_keys}, "
        f"only-in-schema={field_keys - cfg_keys}")


def test_embedding_pooling_options_match_the_embedder():
    """The schema spells its pooling choices out rather than importing the
    inference stack, so pin them to the one source of truth or they drift."""
    from localm.inference.embedder import POOLING_CHOICES

    by_key = {f.key: f for f in ss.CORE_FIELDS}
    assert set(by_key["embedding_pooling"].options) == set(POOLING_CHOICES)


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


class TestMediaPerPluginAnnotation:
    """schema_json's media_per_plugin annotation: the GUI's Media section skips
    group="Media" fields in the flat form and renders per-plugin-mapped globals
    ONLY in the per-plugin boxes - so it must be able to tell, from the schema
    alone, which Media fields those are. Before this annotation the client
    special-cased two keys by name and every other Media field silently
    rendered NOWHERE (comfy_launch_timeout / comfy_disable_auto_launch /
    comfy_func_shim were GUI-invisible; 2026-07-22 settings-exposure audit).
    MEDIA_PLUGIN_FIELDS is the single source of truth."""

    def test_media_fields_carry_the_annotation(self):
        js = ss.schema_json()
        mapped = {m.global_key for m in ss.MEDIA_PLUGIN_FIELDS}
        for f in js:
            if f.get("group") != "Media":
                assert "media_per_plugin" not in f, (
                    f"{f['key']}: the annotation is Media-only noise elsewhere")
                continue
            assert f.get("media_per_plugin") == (f["key"] in mapped), (
                f"{f['key']}: media_per_plugin must mirror MEDIA_PLUGIN_FIELDS")

    def test_the_previously_orphaned_fields_are_not_per_plugin(self):
        """The three fields the Media section historically dropped: global-only
        reads (media/comfy_client.py), so they must be annotated for the
        SHARED box, never left to the per-plugin boxes that cannot show them."""
        js = {f["key"]: f for f in ss.schema_json()}
        for key in ("comfy_launch_timeout", "comfy_disable_auto_launch",
                    "comfy_func_shim"):
            assert js[key].get("media_per_plugin") is False, key

    def test_per_plugin_mapped_globals_are_annotated_true(self):
        js = {f["key"]: f for f in ss.schema_json()}
        assert js["comfy_workdir"].get("media_per_plugin") is True
        assert js["model_swap_policy"].get("media_per_plugin") is True


class TestComfyFloatTypeGlobalKey:
    """The per-plugin float_type field (MEDIA_PLUGIN_FIELDS) and the media
    backends both fall back to a GLOBAL comfy_float_type key - which did not
    exist in DEFAULT_CONFIG or the schema, so the documented fallback could
    only ever be set by hand-editing config.json (the validated PATCH/CLI
    paths reject unknown keys). Make the fallback real: present, typed, and
    validated with the same options as the per-plugin field."""

    def test_key_exists_with_a_null_default(self):
        assert "comfy_float_type" in DEFAULT_CONFIG
        assert DEFAULT_CONFIG["comfy_float_type"] is None

    def test_options_match_the_per_plugin_field(self):
        by_key = {f.key: f for f in ss.CORE_FIELDS}
        per_plugin = next(m for m in ss.MEDIA_PLUGIN_FIELDS
                          if m.global_key == "comfy_float_type")
        assert by_key["comfy_float_type"].options == per_plugin.options

    def test_validates_like_the_per_plugin_options(self):
        assert ss.validate_update({"comfy_float_type": "fp16"}) == {
            "comfy_float_type": "fp16"}
        assert ss.validate_update({"comfy_float_type": ""}) == {
            "comfy_float_type": None}
        with pytest.raises(ValueError):
            ss.validate_update({"comfy_float_type": "not-a-dtype"})


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


# Outbound-target deployment keys: each names WHERE data goes or comes from, so
# each widens network reach the same way net_allow_private does. They are also
# stored VERBATIM by validate_update (no coercion branch above the HIDDEN tail),
# so HIDDEN was doing no gating at all and a non-owner config:write key could
# re-point the live "Send to maintainer" channel (found sweeping finding X8).
OUTBOUND_OWNER_KEYS = {"bugreport_upload_url", "bugreport_upload_token",
                       "update_url", "update_token"}


def test_admin_only_keys_lists_the_owner_only_settings():
    # The rag_* folder keys widen a filesystem-read boundary; net_allow_private
    # disables the SSRF guard (a network trust boundary). Both are owner-only, so
    # a non-owner config:write key cannot flip either (pentest finding LM-PT-001).
    # The bug-report / update endpoints are the same class of boundary.
    # cors_origins names which browser origins may call the authenticated API -
    # the same class of trust-widening boundary - and must be owner-only too
    # (security-checkup finding 2026-07-23).
    # hf_trust_remote_code lets a downloaded model directory run its OWN Python
    # inside the localm process, i.e. arbitrary code execution on this machine.
    # That is the widest boundary of the lot, so it is owner-only too (CodeQL 49).
    assert ss.admin_only_keys() == (
        RAG_OWNER_KEYS | {"net_allow_private", "cors_origins",
                          "hf_trust_remote_code"} | OUTBOUND_OWNER_KEYS)


def test_outbound_endpoint_keys_are_owner_only():
    """Re-pointing bugreport_upload_url redirects a LIVE channel (it ships with a
    real default) that POSTs collected diagnostics plus whatever the user typed,
    and update_url falls back to it. HIDDEN never gated the write - PATCH
    /v1/config stored these verbatim - so this flag is what makes them owner-only."""
    by_key = {f.key: f for f in ss.CORE_FIELDS}
    for key in OUTBOUND_OWNER_KEYS:
        assert by_key[key].admin_only is True, f"{key} must be owner-only"


def test_net_allow_private_is_admin_only():
    by_key = {f.key: f for f in ss.CORE_FIELDS}
    assert by_key["net_allow_private"].admin_only is True, \
        "the SSRF-guard-disable toggle must be owner-only"
    # A sibling network field the scoped key legitimately manages stays non-admin.
    assert by_key["net_allow"].admin_only is False


def test_cors_origins_is_admin_only():
    by_key = {f.key: f for f in ss.CORE_FIELDS}
    assert by_key["cors_origins"].admin_only is True, \
        "cors_origins widens the browser-origin trust boundary and must be owner-only"


# --- non-finite (NaN / inf) numbers are rejected (fuzzing finding LM-FZ-001) --- #
#
# _to_number coerced with float(val)/int(val) and NO isfinite guard, so a NaN
# slipped past every bounds check (NaN fails all < / > comparisons), an inf
# slipped past any field with no upper bound, and an int field given inf raised
# an uncaught OverflowError. A persisted NaN then 500s every GET/PATCH /v1/config
# (FastAPI renders with allow_nan=False), bricking the Settings page across
# restarts. The coercion must reject non-finite values up front.

@pytest.mark.parametrize("key, bad", [
    ("temperature", float("nan")),     # bounded [0,2]: NaN slips past both bounds
    ("top_p", float("nan")),           # bounded [0,1]: NaN slips past both bounds
    ("repeat_penalty", float("inf")),  # float, no upper bound: inf slips through
    ("max_tokens", float("inf")),      # int field: int(inf) is an OverflowError
    ("top_k", float("inf")),           # int field, no upper bound: OverflowError
    ("main_gpu_index", float("inf")),  # separate hand-rolled int() path (not the
                                       # generic NUMBER branch): int(inf) used to
                                       # leak an uncaught OverflowError -> API 500
])
def test_validate_update_rejects_non_finite_numbers(key, bad):
    with pytest.raises(ValueError):
        ss.validate_update({key: bad})


def test_validate_update_keeps_finite_numbers():
    """The guard must not reject legitimate finite values, edges included."""
    assert ss.validate_update({"temperature": 0.5}) == {"temperature": 0.5}
    assert ss.validate_update({"max_tokens": 2048}) == {"max_tokens": 2048}
    assert ss.validate_update({"top_p": 0.0}) == {"top_p": 0.0}
    assert ss.validate_update({"temperature": 2}) == {"temperature": 2}
    assert ss.validate_update({"main_gpu_index": 1}) == {"main_gpu_index": 1}


def test_to_number_rejects_non_finite_directly():
    """Unit-level: _to_number itself is the guard, independent of field bounds.

    -inf is covered here (with lo=None) because on a real field its lower bound
    would mask it; the isfinite guard must reject it on its own."""
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError):
            ss._to_number("x", bad, want_int=False, lo=None, hi=None)
    # int() of a non-finite float must surface as a clean ValueError, never as
    # an uncaught OverflowError leaking out of validate_update.
    with pytest.raises(ValueError):
        ss._to_number("x", float("inf"), want_int=True, lo=None, hi=None)
    # Finite values pass through untouched.
    assert ss._to_number("x", 0.5, want_int=False, lo=0, hi=2) == 0.5
    assert ss._to_number("x", 5, want_int=True, lo=None, hi=None) == 5


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
