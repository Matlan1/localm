# SPDX-License-Identifier: AGPL-3.0-or-later
"""PATCH /v1/config must not be a back door into plugin state (finding X8).

`plugins` and `plugins_enabled` are HIDDEN, engine-managed container keys:
validate_update has no schema for what is inside them, so the HIDDEN branch
stores them verbatim. Their REAL write surfaces validate the value and enforce a
stronger permission:

  - POST /v1/tts/config          - owner (ADMIN) required for library/wasm_paths,
                                   the script every browser imports (#793),
  - POST /v1/media/config/<name> - owner required for launch_cmd/api_url,
  - POST /api/plugins/<n>/enable - the plugins:admin scope.

Without a gate on the generic route, a non-owner `config:write` key writes the
same state directly and skips all of that - so the generic route outranks the
specific one. That is the escalation X8 describes: set
config["plugins"]["tts"]["library"] to a remote URL and tts.js does
`await import(...)` on it in the GUI origin.

These tests pin both directions: the scoped key is refused (and nothing is
written), the owner is not, and the tts route's OWN behavior is untouched.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from localm.inference.http_server import create_app

OWNER_KEY = "owner-admin-key-x8-0001"

# The X8 payload, verbatim from the review: an absolute URL discards the base in
# `new URL(cfg.library, import.meta.url)`, so tts.js would import remote code.
EVIL_LIBRARY = "https://evil.example/x.js"


@pytest.fixture
def app_env(tmp_path, monkeypatch):
    """Protected-mode app: an owner (ADMIN) key via env + a scoped
    config:read/config:write key, both under an isolated data dir."""
    import localm.config as cfg
    home = tmp_path / ".localm"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.setattr(cfg, "HOME_DIR", home)
    monkeypatch.setattr(cfg, "MODELS_DIR", home / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", home / "registry.json")
    monkeypatch.setenv("LOCALM_API_KEY", OWNER_KEY)      # owner key -> {ADMIN}
    from localm import auth
    scoped = auth.create_key(
        "dev", ["config:read", "config:write"], allow_privileged=True)["key"]
    app = create_app(None)
    with TestClient(app) as c:
        yield c, scoped


def _owner():
    return {"Authorization": f"Bearer {OWNER_KEY}"}


def _scoped(key):
    return {"Authorization": f"Bearer {key}"}


def _stored(c, key):
    """The value as the OWNER sees it (the scoped key can read it too; reading is
    deliberately not gated - X8 is an escalation on WRITE)."""
    return c.get("/v1/config", headers=_owner()).json().get(key)


# --------------------------------------------------------------------------- #
#  Refused for a non-owner config:write key - and nothing is written           #
# --------------------------------------------------------------------------- #

def test_scoped_key_cannot_write_the_plugins_subtree(app_env):
    """The X8 chain, at the route: a config:write key without ADMIN must not be
    able to set the tts script URL through the generic settings route."""
    c, scoped = app_env
    before = _stored(c, "plugins")
    denied = c.patch("/v1/config", headers=_scoped(scoped),
                     json={"plugins": {"tts": {"library": EVIL_LIBRARY}}})
    assert denied.status_code == 403, denied.text
    assert "owner" in denied.text.lower()
    # Refused means NOT WRITTEN - a 403 that still persisted the value would be
    # the same vulnerability with a worse error message. Compared against the
    # value BEFORE the attempt rather than against empty, so this keeps testing
    # the real property even if something else later populates the block.
    after = _stored(c, "plugins")
    assert after == before, "a refused write must leave the plugin block untouched"
    assert EVIL_LIBRARY not in str(after), \
        "the remote script URL must not have reached the stored config"


def test_scoped_key_cannot_toggle_plugins_enabled(app_env):
    """Same mechanism, second key: enabling a plugin is plugins:admin
    (POST /api/plugins/<name>/enable), a scope a config:write key need not hold,
    and the engine reads config["plugins_enabled"] as its source of truth."""
    c, scoped = app_env
    # Compare against the value BEFORE the attempt, not against empty: startup
    # enables the always-on chat plugin, so this key is legitimately non-empty in
    # a running app. "Refused" means UNCHANGED, which is the property that matters.
    before = _stored(c, "plugins_enabled")
    denied = c.patch("/v1/config", headers=_scoped(scoped),
                     json={"plugins_enabled": ["chat", "coder"]})
    assert denied.status_code == 403, denied.text
    assert "owner" in denied.text.lower()
    after = _stored(c, "plugins_enabled")
    assert after == before, "a refused write must leave the enabled set untouched"
    assert "coder" not in (after or []), \
        "the plugin the caller tried to enable must NOT have been enabled"


def test_the_refusal_names_the_key_and_points_at_the_real_surface(app_env):
    """AGENTS.md rule 5: a refusal must say what was refused and where to go, not
    fail opaquely (this is the error a legitimate integrator will hit)."""
    c, scoped = app_env
    denied = c.patch("/v1/config", headers=_scoped(scoped),
                     json={"plugins": {"tts": {"voice": "am_onyx"}}})
    assert denied.status_code == 403
    body = denied.json()["detail"]
    assert "plugins" in body
    assert "/v1/tts/config" in body, "the refusal should point at the real surface"


# --------------------------------------------------------------------------- #
#  Still allowed for the owner, and not a blanket block                        #
# --------------------------------------------------------------------------- #

def test_owner_can_still_write_plugin_state(app_env):
    """The gate is about WHO, not a removal of the capability."""
    c, _ = app_env
    ok = c.patch("/v1/config", headers=_owner(),
                 json={"plugins": {"image": {"comfy": {"output_dir": "x"}}}})
    assert ok.status_code == 200, ok.text
    assert _stored(c, "plugins")["image"]["comfy"]["output_dir"] == "x"

    ok2 = c.patch("/v1/config", headers=_owner(),
                  json={"plugins_enabled": ["chat"]})
    assert ok2.status_code == 200, ok2.text
    assert _stored(c, "plugins_enabled") == ["chat"]


def test_an_ordinary_setting_still_works_for_the_scoped_key(app_env):
    """The gate is specific to engine-managed keys, not a blanket block on
    config:write (the same shape the admin_only gate is tested for)."""
    c, scoped = app_env
    ok = c.patch("/v1/config", headers=_scoped(scoped), json={"n_ctx": 8192})
    assert ok.status_code == 200, ok.text
    assert _stored(c, "n_ctx") == 8192


def test_scoped_key_cannot_repoint_the_bug_report_upload_url(app_env):
    """Found sweeping X8 rather than in the review, and the sharper half of it.

    `bugreport_upload_url` is HIDDEN, but HIDDEN is not a gate: it has no coercion
    branch, so validate_update stored whatever it was handed. It also ships with a
    REAL default, so it is a live channel - "Send to maintainer" POSTs the
    collected diagnostics plus whatever the user typed to it. A non-owner
    config:write key re-pointing it would exfiltrate the next bug report. Now
    admin_only, so it is refused on the same gate as the rag_* roots."""
    c, scoped = app_env
    denied = c.patch("/v1/config", headers=_scoped(scoped),
                     json={"bugreport_upload_url": "https://evil.example/collect"})
    assert denied.status_code == 403, denied.text
    got = c.get("/v1/config", headers=_owner()).json()
    assert "evil.example" not in str(got.get("bugreport_upload_url")), \
        "the refused endpoint must not have been persisted"
    # The owner can still configure the channel (this is a WHO gate, not a removal).
    ok = c.patch("/v1/config", headers=_owner(),
                 json={"bugreport_upload_url": "https://reports.example/api"})
    assert ok.status_code == 200, ok.text


def test_reading_plugin_state_is_not_gated(app_env):
    """X8 is an escalation on WRITE. The read side is deliberately unchanged, so
    a config:read caller still sees the values (nothing reads them off
    GET /v1/config today, but silently narrowing reads would be a wider change
    than the finding)."""
    c, scoped = app_env
    assert c.patch("/v1/config", headers=_owner(),
                   json={"plugins_enabled": ["chat"]}).status_code == 200
    got = c.get("/v1/config", headers=_scoped(scoped)).json()
    assert got["plugins_enabled"] == ["chat"]


# --------------------------------------------------------------------------- #
#  The tts route's own behavior is untouched (#793 still holds)                #
# --------------------------------------------------------------------------- #

def test_tts_route_gate_is_unchanged_by_the_config_gate(app_env):
    """Closing the back door must not change the front one: /v1/tts/config still
    refuses library to the scoped key, still accepts an ordinary field from it,
    and still accepts a VALID relative library from the owner (the
    _tts_relative_asset validator is untouched)."""
    c, scoped = app_env
    # an ordinary tts field is still fine for the scoped key ...
    ok = c.post("/v1/tts/config", headers=_scoped(scoped), json={"voice": "am_onyx"})
    assert ok.status_code == 200, ok.text
    # ... the script URL is still owner-only, even with a VALID value
    denied = c.post("/v1/tts/config", headers=_scoped(scoped),
                    json={"library": "vendor/kokoro.min.js"})
    assert denied.status_code == 403, denied.text
    # ... and the owner can still set it
    allowed = c.post("/v1/tts/config", headers=_owner(),
                     json={"library": "vendor/kokoro.min.js"})
    assert allowed.status_code == 200, allowed.text


def test_the_validator_still_rejects_a_remote_library_for_the_owner(app_env):
    """The owner gate is not a licence to store anything: _tts_relative_asset
    still confines the value to the plugin's own static folder, so even an ADMIN
    cannot point every browser at remote code THROUGH THE TTS ROUTE."""
    c, _ = app_env
    r = c.post("/v1/tts/config", headers=_owner(), json={"library": EVIL_LIBRARY})
    assert r.status_code == 400, r.text


# --------------------------------------------------------------------------- #
#  The class cannot silently grow                                             #
# --------------------------------------------------------------------------- #

def _verbatim_hidden_keys():
    """Every HIDDEN core key that hands a hostile value straight back.

    PROBED, not inferred. The obvious guard - "HIDDEN with a list/dict default" -
    is WRONG, and quietly so: the tail of the HIDDEN branch in _validate_one is a
    bare `return val`, reached by ANY hidden field that has no dedicated coercion
    branch above it, whatever its default's type. bugreport_upload_url and
    update_url have str/None defaults and still return a dict unchanged. Asking
    the validator what it actually does is the only guard that cannot drift."""
    from localm import settings_schema as ss
    hostile = {"evil": "payload"}
    out = set()
    for f in ss.CORE_FIELDS:
        if f.widget != ss.Widget.HIDDEN:
            continue
        try:
            got = ss.validate_update({f.key: hostile})
        except ValueError:
            continue                      # rejected outright - not a passthrough
        if got.get(f.key) is hostile:     # identity: stored exactly as supplied
            out.add(f.key)
    return out


def test_no_key_is_stored_verbatim_without_an_owner_gate():
    """The invariant that actually matters, and the guard against a FUTURE key
    joining the passthrough unnoticed.

    If PATCH /v1/config will store a value it never inspected, a non-owner
    config:write key decides that value - so every such key must be owner-gated,
    by either flag. If this fails, a new key reached the passthrough: gate it, or
    give it a real validator. Do not delete the assertion."""
    from localm import settings_schema as ss
    gated = ss.engine_managed_keys() | ss.admin_only_keys()
    ungated = _verbatim_hidden_keys() - gated
    assert not ungated, (
        "these keys are stored VERBATIM from a PATCH body but need no owner: "
        f"{sorted(ungated)}")


def test_the_verbatim_set_is_exactly_what_we_think_it_is():
    """Pin the membership too, so a key LEAVING the set (someone gives it a real
    validator) is also visible, not just one joining it."""
    assert _verbatim_hidden_keys() == {
        "plugins",                  # engine_managed: per-plugin state
        "bugreport_upload_url",     # admin_only: where bug reports are POSTed
        "bugreport_upload_token",
        "update_url",               # admin_only: the update channel base
        "update_token",
    }


def test_key_presets_is_not_a_passthrough():
    """It LOOKS like the class (HIDDEN, list-of-dicts default) but is intercepted
    by _validate_key_presets, which rejects unknown scopes - so it needs no owner
    gate. Pinned because 'looks like the class' is what made the first version of
    this guard wrong."""
    from localm import settings_schema as ss
    assert "key_presets" not in _verbatim_hidden_keys()
    # The rejected value must be something is_valid_scope really rejects. A bare
    # identifier like "not-a-scope" is ACCEPTED on purpose (a plugin owns the
    # scope named after it, and "-" normalises to "_"), so it proves nothing here.
    with pytest.raises(ValueError, match="unknown scope"):
        ss.validate_update({"key_presets": [{"name": "x", "scopes": ["not a scope"]}]})
    # ... and a legitimate preset still round-trips.
    ok = ss.validate_update({"key_presets": [{"name": "Minimal", "scopes": ["chat"]}]})
    assert ok["key_presets"] == [{"name": "Minimal", "scopes": ["chat"]}]


def test_engine_managed_keys_are_exactly_the_plugin_state_keys():
    from localm import settings_schema as ss
    assert ss.engine_managed_keys() == {"plugins", "plugins_enabled"}


def test_engine_managed_is_not_admin_only():
    """The two flags are different gates: admin_only also HIDES the value from a
    non-owner (schema + GET), engine_managed only refuses the WRITE. Conflating
    them would silently narrow reads."""
    from localm import settings_schema as ss
    assert ss.engine_managed_keys() & ss.admin_only_keys() == set()
