# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the settings schema (localm/settings_schema.py)."""

import json

import pytest

from localm import settings_schema as ss
from localm.config import DEFAULT_CONFIG


# hf_token / civitai_api_key are schema-only BY DESIGN (see their SettingField
# comment in settings_schema.py): they route to model_source_credentials.py's
# owner-only file instead of config.json, and staying OUT of DEFAULT_CONFIG is
# the safety backstop - a caller that ever reaches validate_update with one of
# these keys (the intended interception skipped) gets a loud "unknown config
# key" error instead of a silent write into config.json.
SCHEMA_ONLY_CREDENTIAL_KEYS = {"hf_token", "civitai_api_key"}


def test_schema_matches_config_keys_exactly():
    """Guards drift: every config key has a field and vice versa, except the
    documented schema-only credential keys above."""
    field_keys = {f.key for f in ss.CORE_FIELDS} - SCHEMA_ONLY_CREDENTIAL_KEYS
    cfg_keys = set(DEFAULT_CONFIG)
    assert field_keys == cfg_keys, (
        f"only-in-config={cfg_keys - field_keys}, "
        f"only-in-schema={field_keys - cfg_keys}")


def test_credential_keys_are_deliberately_schema_only():
    """The two model-source credential fields must stay OUT of DEFAULT_CONFIG
    and must be marked secret - if a future edit "fixes" the asymmetry above by
    adding them to DEFAULT_CONFIG, it silently defeats the safety backstop:
    validate_update would then accept and persist a real token into
    config.json instead of rejecting it whenever the interception is skipped."""
    by_key = {f.key: f for f in ss.CORE_FIELDS}
    for key in SCHEMA_ONLY_CREDENTIAL_KEYS:
        assert key not in DEFAULT_CONFIG, f"{key} must not be in DEFAULT_CONFIG"
        assert by_key[key].secret is True
        assert by_key[key].widget == ss.Widget.SECRET
        assert by_key[key].admin_only is True


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


class TestShippedDefaultAnnotation:
    """`default` is the CURRENT value (base=load_config() in the real server
    route), which after a save is the user's own override - so the GUI needs a
    SEPARATE, always-factory value to tell "still shipped default" apart from
    "user set it to this". `shipped_default` is that value: always from
    DEFAULT_CONFIG, regardless of what `values` the caller passed."""

    def test_shipped_default_matches_default_config_regardless_of_override(self):
        overridden = dict(DEFAULT_CONFIG)
        overridden["temperature"] = 1.7   # a real user override, unlike the default 0.8
        js = ss.schema_json(values=overridden)
        by_key = {f["key"]: f for f in js}
        assert by_key["temperature"]["default"] == 1.7, \
            "default still reflects the CURRENT (overridden) value"
        assert by_key["temperature"]["shipped_default"] == DEFAULT_CONFIG["temperature"], \
            "shipped_default must stay the factory value even when overridden"
        assert by_key["temperature"]["default"] != by_key["temperature"]["shipped_default"]

    def test_shipped_default_equals_default_on_a_fresh_install(self):
        js = ss.schema_json()   # values=None -> base is DEFAULT_CONFIG itself
        for f in js:
            if "default" in f:
                assert f.get("shipped_default") == f["default"], (
                    f"{f['key']}: on a fresh install default and shipped_default "
                    f"must agree (both come from DEFAULT_CONFIG)")

    def test_shipped_default_omitted_for_secret_fields(self):
        js = ss.schema_json()
        for f in js:
            if f.get("secret"):
                assert "shipped_default" not in f, \
                    f"{f['key']}: a secret must never carry any default, shipped or not"

    def test_shipped_default_present_for_every_default_bearing_field(self):
        """Every field that gets a `default` (i.e. every non-secret field whose
        key is in DEFAULT_CONFIG) must also get a `shipped_default` - the GUI
        cannot grey a field it has no factory value to compare against."""
        js = ss.schema_json()
        for f in js:
            if "default" in f:
                assert "shipped_default" in f, \
                    f"{f['key']}: has 'default' but no 'shipped_default'"


class TestMediaPerPluginAnnotation:
    """schema_json's media_per_plugin annotation: the GUI's Media section skips
    group="Media" fields in the flat form and renders per-plugin-mapped globals
    ONLY in the per-plugin boxes - so it must be able to tell, from the schema
    alone, which Media fields those are. Without the annotation a client has to
    special-case keys by name, and every other Media field
    (comfy_launch_timeout / comfy_disable_auto_launch / comfy_func_shim) renders
    NOWHERE. MEDIA_PLUGIN_FIELDS is the single source of truth."""

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
        """Three fields the Media section can drop: global-only reads
        (media/comfy_client.py), so they must be annotated for the SHARED box,
        never left to the per-plugin boxes that cannot show them."""
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
    backends both fall back to a GLOBAL comfy_float_type key. Absent from
    DEFAULT_CONFIG and the schema, that fallback can only be set by hand-editing
    config.json, since the validated PATCH/CLI paths reject unknown keys. The
    key must be present, typed, and validated with the same options as the
    per-plugin field."""

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
    """EVERY rendered field must carry a clear description (HIDDEN fields are
    not rendered, so they are exempt)."""
    missing = [f.key for f in ss.CORE_FIELDS
               if f.widget != ss.Widget.HIDDEN and not (f.help or "").strip()]
    assert not missing, f"fields missing a description: {missing}"


def test_every_field_has_a_label():
    missing = [f.key for f in ss.CORE_FIELDS if not (f.label or "").strip()]
    assert not missing, f"fields missing a label: {missing}"


def test_binary_dir_schema_exposes_auto_resolved_path():
    """Blank binary_dir must surface the auto-detected path so the GUI can show
    it rather than leaving the field empty."""
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
# stored VERBATIM by validate_update (no coercion branch above the HIDDEN tail).
OUTBOUND_OWNER_KEYS = {"bugreport_upload_url", "bugreport_upload_token",
                       "update_url", "update_token"}

# --- All CORE_FIELDS, grouped by capability --------------------------------- #
#
# PATCH /v1/config gates on admin_only_keys() | engine_managed_keys() and nothing
# else, so a capability-bearing field carrying neither flag is reachable by any
# delegated key.

# Code or process execution.
EXEC_OWNER_KEYS = {
    "binary_dir",            # -> ctypes.CDLL(): attacker NATIVE code IN-PROCESS
    "comfy_launch_cmd",      # -> shlex.split -> Popen(argv), caller-chosen program
    "comfy_workdir",         # -> launcher auto-discovered inside it -> Popen
    "comfy_api_url",         # -> the render target every media job is sent to
    "coder_reviewer",        # -> selects a cloud/URL/local reviewer backend
    "coder_reviewer_model",  # -> registry.py falls through to Path(name): any GGUF
}

# The privacy contract: whether localm writes the user's content to disk at all.
PRIVACY_OWNER_KEYS = {"mode", "chat_mode", "coder_mode", "keep_diagnostics"}

# Selects a file the server OPENS and parses, with no path confinement:
# embedder.py accepts any caller-chosen path and hands it to llama.cpp's native
# GGUF parser (a UNC path on Windows also makes the probe an outbound SMB/NTLM
# auth).
LOAD_PATH_OWNER_KEYS = {"embedding_model"}

# Controls whose only job is to be restrictive, so CLEARING one widens reach.
GUARD_OWNER_KEYS = {
    "require_auth",                 # latent fail-closed switch (see the field comment)
    "net_deny",                     # clearing it un-blocks every denied host
    "net_search_url",               # where every web search is sent
    "coder_untrusted_provenance",   # indirect-prompt-injection hardening
}

# The GUI-settable server bind. bind_host decides WHICH NETWORK can reach the
# server at all, and the TLS trio decides whether, and with what certificate,
# that traffic is encrypted. All owner-only, so a delegated config:write key
# cannot expose the server or strip its transport encryption.
NETWORK_BIND_OWNER_KEYS = {"bind_host", "tls_enabled", "tls_cert", "tls_key"}

# Optional third-party model-source credentials (ADR-0015 decision 4): each
# names where an outbound Bearer credential goes, the same "where does data
# go" boundary as OUTBOUND_OWNER_KEYS above. Schema-only - see
# SCHEMA_ONLY_CREDENTIAL_KEYS.
MODEL_SOURCE_CREDENTIAL_KEYS = {"hf_token", "civitai_api_key"}


def test_admin_only_keys_lists_the_owner_only_settings():
    # The rag_* folder keys widen a filesystem-read boundary; net_allow_private
    # disables the SSRF guard. The bug-report / update endpoints are the same
    # class of boundary, cors_origins names which browser origins may call the
    # authenticated API, and hf_trust_remote_code lets a downloaded model
    # directory run its OWN Python inside the localm process.
    #
    # EXACT SET EQUALITY, not a subset check. Resolve any conflict here
    # ADDITIVELY, keeping the keys from both sides.
    assert ss.admin_only_keys() == (
        RAG_OWNER_KEYS | OUTBOUND_OWNER_KEYS | EXEC_OWNER_KEYS
        | PRIVACY_OWNER_KEYS | GUARD_OWNER_KEYS | LOAD_PATH_OWNER_KEYS
        | {"hf_trust_remote_code"}
        | {"net_allow_private", "cors_origins"}
        # update_allow_prerelease decides which builds the updater suggests
        # installing.
        | {"update_allow_prerelease"}
        # gui_proxy_remote_images decides whether rendering a reply causes an
        # outbound request at all.
        | {"gui_proxy_remote_images"}
        # update_ignore_net_policy exempts the update channel from net_mode=off.
        | {"update_ignore_net_policy"}
        # net_allow_model_downloads exempts explicit downloads (pull, search,
        # mmproj/voice/embedding fetch) from net_mode=off.
        | {"net_allow_model_downloads"}
        # llama_runtime_pin decides which native build setup-llama downloads and
        # the server then loads in-process; llama_runtime_history is what
        # --rollback reads.
        | {"llama_runtime_pin", "llama_runtime_history"}
        # allow_network_drives decides which host locations the folder picker,
        # folder creation/rename, log export and RAG indexing treat as a normal
        # local folder.
        | {"allow_network_drives"}
        # Optional HF/CivitAI credentials (ADR-0015).
        | MODEL_SOURCE_CREDENTIAL_KEYS
        # The GUI-settable server bind + TLS trio.
        | NETWORK_BIND_OWNER_KEYS)


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


# --- non-finite (NaN / inf) numbers are rejected ---------------------------- #
#
# _to_number rejects non-finite values up front: NaN slips past every bounds
# check, inf slips past any field with no upper bound, and an int field given
# inf raises OverflowError.

@pytest.mark.parametrize("key, bad", [
    ("temperature", float("nan")),     # bounded [0,2]: NaN slips past both bounds
    ("top_p", float("nan")),           # bounded [0,1]: NaN slips past both bounds
    ("repeat_penalty", float("inf")),  # float, no upper bound: inf slips through
    ("max_tokens", float("inf")),      # int field: int(inf) is an OverflowError
    ("top_k", float("inf")),           # int field, no upper bound: OverflowError
    ("main_gpu_index", float("inf")),  # separate hand-rolled int() path (not the
                                       # generic NUMBER branch), where int(inf)
                                       # raises OverflowError
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


def test_max_tokens_zero_is_the_unlimited_sentinel_and_is_accepted():
    """0 means unlimited (both the GGUF and HF backends treat it that way),
    not an invalid runaway-guard value - the min bound must let it through."""
    assert ss.validate_update({"max_tokens": 0}) == {"max_tokens": 0}


def test_max_tokens_negative_still_rejected():
    """Only the exact 0 sentinel is unlimited; a negative value has no
    meaning and stays rejected."""
    with pytest.raises(ValueError, match="max_tokens"):
        ss.validate_update({"max_tokens": -1})


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
    # A normal (non-admin_only) field is unaffected by the owner filter: the
    # over-gating control. `temperature` is a plain sampling knob a delegated key
    # legitimately sets.
    assert "temperature" in owner and "temperature" in guest
    # The owner view advertises the admin_only flag so the client/UI can label it.
    by_key = {f["key"]: f for f in ss.schema_json(is_owner=True)}
    assert by_key["rag_indexing_mode"].get("admin_only") is True
    # Default (no is_owner arg) behaves as owner, for the CLI and tests.
    assert RAG_OWNER_KEYS <= {f["key"] for f in ss.schema_json()}


# --------------------------------------------------------------------------- #
#  Chat avatars: user_avatar / model_avatar_default / model_avatar_overrides  #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("key", ["user_avatar", "model_avatar_default"])
def test_avatar_value_accepts_empty(key):
    """"" clears the field, same contract as chat_system_prompt (also a
    string-default HIDDEN-adjacent field): None is a validation error, not a
    synonym for "" - a client sends the empty string to clear it."""
    assert ss.validate_update({key: ""}) == {key: ""}
    with pytest.raises(ValueError):
        ss.validate_update({key: None})


@pytest.mark.parametrize("key", ["user_avatar", "model_avatar_default"])
def test_avatar_value_accepts_a_short_glyph(key):
    assert ss.validate_update({key: "AB"}) == {key: "AB"}
    assert ss.validate_update({key: "\U0001F600"}) == {key: "\U0001F600"}


@pytest.mark.parametrize("key", ["user_avatar", "model_avatar_default"])
def test_avatar_value_accepts_a_data_uri(key):
    uri = "data:image/png;base64,iVBORw0KGgo="
    assert ss.validate_update({key: uri}) == {key: uri}


@pytest.mark.parametrize("key", ["user_avatar", "model_avatar_default"])
@pytest.mark.parametrize("bad", [
    # Each of these is <= _AVATAR_MAX_GLYPH_LEN chars, so a case here can only
    # go red via the URL/path check itself - the length cap cannot mask it.
    "http://x",
    "https://x",
    "//x",
    "data:img",
    "/etc/pw",
    "a\\b",
])
def test_avatar_value_rejects_url_or_path_short_form(key, bad):
    assert len(bad) <= ss._AVATAR_MAX_GLYPH_LEN
    with pytest.raises(ValueError):
        ss.validate_update({key: bad})


@pytest.mark.parametrize("key", ["user_avatar", "model_avatar_default"])
@pytest.mark.parametrize("bad", [
    "http://example.com/a.png",
    "https://example.com/a.png",
    "//example.com/a.png",
    "data:image/svg+xml;base64,PHN2Zz48L3N2Zz4=",
    "/etc/passwd",
    "C:\\Users\\me\\avatar.png",
    "a" * 17,
])
def test_avatar_value_rejects_url_or_path(key, bad):
    with pytest.raises(ValueError):
        ss.validate_update({key: bad})


def test_avatar_value_rejects_oversized_data_uri():
    huge = "data:image/png;base64," + ("A" * ss._AVATAR_MAX_DATA_URI_LEN)
    with pytest.raises(ValueError):
        ss.validate_update({"user_avatar": huge})


def test_model_avatar_overrides_accepts_a_valid_map():
    val = {"qwen3-coder-30b": "\U0001F916", "llama-3-8b": ""}
    # An empty per-model value clears that entry rather than storing "".
    assert ss.validate_update({"model_avatar_overrides": val}) == {
        "model_avatar_overrides": {"qwen3-coder-30b": "\U0001F916"}}


def test_model_avatar_overrides_rejects_non_dict():
    with pytest.raises(ValueError):
        ss.validate_update({"model_avatar_overrides": ["not", "a", "dict"]})


def test_model_avatar_overrides_rejects_a_bad_entry():
    with pytest.raises(ValueError, match="qwen3-coder-30b"):
        ss.validate_update({
            "model_avatar_overrides": {"qwen3-coder-30b": "http://evil.example/x.png"}})


# --------------------------------------------------------------------------- #
#  chat_background (wallpaper)                                                #
# --------------------------------------------------------------------------- #

def test_background_value_accepts_empty():
    assert ss.validate_update({"chat_background": ""}) == {"chat_background": ""}
    with pytest.raises(ValueError):
        ss.validate_update({"chat_background": None})


def test_background_value_accepts_a_data_uri():
    uri = "data:image/jpeg;base64,iVBORw0KGgo="
    assert ss.validate_update({"chat_background": uri}) == {"chat_background": uri}


@pytest.mark.parametrize("bad", [
    "http://example.com/a.jpg",
    "https://example.com/a.jpg",
    "//example.com/a.jpg",
    "data:image/svg+xml;base64,PHN2Zz48L3N2Zz4=",
    "/etc/passwd",
    "C:\\Users\\me\\wallpaper.jpg",
    "AB",   # unlike an avatar, a background has no short-glyph fallback
])
def test_background_value_rejects_anything_but_a_data_uri(bad):
    with pytest.raises(ValueError):
        ss.validate_update({"chat_background": bad})


def test_background_value_rejects_oversized_data_uri():
    huge = "data:image/jpeg;base64," + ("A" * ss._BACKGROUND_MAX_DATA_URI_LEN)
    with pytest.raises(ValueError):
        ss.validate_update({"chat_background": huge})


def test_background_value_rejects_a_huge_garbage_string_without_reflecting_it():
    """A value that fails the data-URI match entirely (never reaches the size
    check below) must not have its raised error message grow with the input -
    unlike the oversized-but-matching case, this branch has no other bound."""
    huge_garbage = "x" * 5_000_000
    with pytest.raises(ValueError) as exc:
        ss.validate_update({"chat_background": huge_garbage})
    assert len(str(exc.value)) < 1000
    assert huge_garbage not in str(exc.value)


def test_background_value_accepts_up_to_the_cap():
    prefix = "data:image/jpeg;base64,"
    at_cap = prefix + ("A" * (ss._BACKGROUND_MAX_DATA_URI_LEN - len(prefix)))
    assert len(at_cap) == ss._BACKGROUND_MAX_DATA_URI_LEN
    assert ss.validate_update({"chat_background": at_cap}) == {"chat_background": at_cap}


# --------------------------------------------------------------------------- #
#  user_name                                                                  #
# --------------------------------------------------------------------------- #

def test_user_name_accepts_empty():
    """Same contract as user_avatar: "" clears the field; None is a validation
    error rather than a synonym for "" (a client sends "" to clear it)."""
    assert ss.validate_update({"user_name": ""}) == {"user_name": ""}
    with pytest.raises(ValueError):
        ss.validate_update({"user_name": None})


def test_user_name_accepts_a_plain_string_and_strips_it():
    assert ss.validate_update({"user_name": "  Matt  "}) == {"user_name": "Matt"}


def test_user_name_accepts_up_to_the_cap():
    at_cap = "a" * ss._USER_NAME_MAX_LEN
    assert ss.validate_update({"user_name": at_cap}) == {"user_name": at_cap}


def test_user_name_rejects_a_too_long_value():
    with pytest.raises(ValueError):
        ss.validate_update({"user_name": "a" * (ss._USER_NAME_MAX_LEN + 1)})
