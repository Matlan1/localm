# SPDX-License-Identifier: AGPL-3.0-or-later
"""The tts plugin's server-side settings write surface: GET/POST /v1/tts/config
plus the settings_schema helpers behind them.

config["plugins"]["tts"] holds engine, model, device, dtype, voice, speed,
library and wasm_paths.

Shape mirrors the per-plugin media config (validate_media_block + POST
/v1/media/config/{name}): validated in settings_schema, deep-merged into the
plugin's own block, blank clears an override back to the shipped template
default. The two fields that become a script/wasm URL in EVERY browser client
(library, wasm_paths) additionally require an owner (ADMIN) principal, as
launch_cmd/api_url do for a media backend.
"""

import pytest
from fastapi.testclient import TestClient

from localm import settings_schema as ss


# --------------------------------------------------------------------------- #
#  Unit: schema + validator                                                    #
# --------------------------------------------------------------------------- #

def test_tts_defaults_come_from_the_shipped_template():
    d = ss.tts_defaults()
    assert d["engine"] == "kokoro"
    assert d["model"].startswith("onnx-community/Kokoro")
    assert d["voice"] == "af_heart"
    assert "_comment" not in d              # documentation keys stripped


def test_known_voices_loaded_from_the_vendored_voice_list():
    voices = ss.known_tts_voices()
    assert voices, "the vendored voices.json should provide the voice ids"
    assert "af_heart" in voices and "am_onyx" in voices


def test_schema_exposes_the_user_facing_fields_and_hides_the_deployment_ones():
    fields = {f["key"]: f for f in ss.tts_schema_json({})}
    # every stored key has a validated write path
    assert set(fields) == {"engine", "model", "device", "dtype", "voice",
                           "speed", "library", "wasm_paths"}
    # user-facing ones render in the GUI ...
    for key in ("voice", "speed", "model", "device", "dtype"):
        assert fields[key]["gui"] is True, key
    # ... the single-engine key and the two script-URL ones do not
    for key in ("engine", "library", "wasm_paths"):
        assert fields[key]["gui"] is False, key
    # device/dtype are secondary (an "Advanced" box), voice/speed/model primary
    assert fields["device"]["advanced"] and fields["dtype"]["advanced"]
    assert not fields["voice"]["advanced"] and not fields["speed"]["advanced"]
    # the voice control offers the real vendored voices
    assert "af_heart" in fields["voice"]["options"]


def test_schema_flags_the_admin_only_script_url_fields():
    assert ss.tts_admin_only_fields() == {"library", "wasm_paths"}
    fields = {f["key"]: f for f in ss.tts_schema_json({})}
    assert fields["library"]["admin_only"] is True
    assert fields["wasm_paths"]["admin_only"] is True
    assert fields["voice"]["admin_only"] is False


def test_schema_hides_admin_only_fields_for_non_owner():
    # library/wasm_paths become a script/wasm URL every browser loads, so a
    # non-owner must not see their resolved value either.
    owner_fields = {f["key"] for f in ss.tts_schema_json({})}
    scoped_fields = {f["key"] for f in ss.tts_schema_json({}, is_owner=False)}
    assert {"library", "wasm_paths"} <= owner_fields, "owner sees everything"
    assert not ({"library", "wasm_paths"} & scoped_fields), \
        "a non-owner must not see library/wasm_paths"
    assert "voice" in scoped_fields, "an ordinary field stays visible"


def test_schema_resolves_block_over_template_default():
    fields = {f["key"]: f for f in ss.tts_schema_json({"voice": "am_onyx"})}
    assert fields["voice"]["value"] == "am_onyx"
    assert fields["voice"]["is_override"] is True
    # no override -> the shipped template default, not flagged
    assert fields["model"]["value"].startswith("onnx-community/Kokoro")
    assert fields["model"]["is_override"] is False


def test_validate_accepts_the_user_facing_values():
    assert ss.validate_tts_block({"voice": "am_onyx"}) == {"voice": "am_onyx"}
    assert ss.validate_tts_block({"speed": "1.25"}) == {"speed": 1.25}
    assert ss.validate_tts_block({"model": "owner/Some-Model.v2_x"}) == \
        {"model": "owner/Some-Model.v2_x"}
    assert ss.validate_tts_block({"device": "wasm", "dtype": "q8"}) == \
        {"device": "wasm", "dtype": "q8"}


def test_validate_rejects_an_unknown_field():
    with pytest.raises(ValueError):
        ss.validate_tts_block({"not_a_field": 1})


def test_validate_rejects_an_unknown_voice():
    # tts.js falls back to the FIRST voice for an unknown id, so a typo is
    # rejected at set time.
    with pytest.raises(ValueError):
        ss.validate_tts_block({"voice": "af_nosuchvoice"})


def test_validate_rejects_out_of_range_or_non_numeric_speed():
    for bad in ("0", "0.1", "9", "fast", [1]):     # blank is "clear", covered below
        with pytest.raises(ValueError):
            ss.validate_tts_block({"speed": bad})


def test_validate_rejects_a_model_that_is_not_a_repo_id():
    # the browser downloads this id from huggingface.co: a URL or a path is not
    # a repo id, and would fail opaquely at load time.
    for bad in ("https://evil.example/model", "no-slash", "a/b/c", "owner/ ",
                "../etc/passwd", "owner/model?x=1"):
        with pytest.raises(ValueError):
            ss.validate_tts_block({"model": bad})


def test_validate_rejects_a_bad_device_or_dtype():
    with pytest.raises(ValueError):
        ss.validate_tts_block({"device": "cuda"})
    with pytest.raises(ValueError):
        ss.validate_tts_block({"dtype": "bf16"})


def test_validate_rejects_an_unsupported_engine():
    assert ss.validate_tts_block({"engine": "kokoro"}) == {"engine": "kokoro"}
    with pytest.raises(ValueError):
        ss.validate_tts_block({"engine": "elevenlabs"})


def test_validate_confines_library_and_wasm_paths_to_the_plugin_assets():
    # These become a script / wasm URL that EVERY browser client loads, so only
    # a relative in-plugin path is accepted.
    for bad in ("https://evil.example/x.js", "//evil.example/x.js",
                "/etc/passwd", "Z:/windows/x.js", "..\\..\\x.js",
                "../../../secret.js", "vendor\\kokoro.min.js",
                "javascript:alert(1)", "data:text/javascript,alert(1)"):
        with pytest.raises(ValueError):
            ss.validate_tts_block({"library": bad})
        with pytest.raises(ValueError):
            ss.validate_tts_block({"wasm_paths": bad})
    assert ss.validate_tts_block({"library": "vendor/kokoro.min.js"}) == \
        {"library": "vendor/kokoro.min.js"}
    assert ss.validate_tts_block({"wasm_paths": "vendor/"}) == \
        {"wasm_paths": "vendor/"}


def test_validate_rejects_asset_paths_that_would_404_in_the_browser():
    """Existing on disk is not enough: the browser fetches these over HTTP."""
    # A file that is simply not there (a typo) is caught at set time.
    with pytest.raises(ValueError):
        ss.validate_tts_block({"library": "vendor/nope.js"})
    # library must be a FILE: a folder imports as nothing.
    with pytest.raises(ValueError):
        ss.validate_tts_block({"library": "vendor"})
    # An empty segment ("a//b") resolves on disk but is a different URL path.
    with pytest.raises(ValueError):
        ss.validate_tts_block({"library": "vendor//kokoro.min.js"})
    # Windows/macOS filesystems are case-insensitive and the HTTP path is not,
    # so the rejection REASON differs by platform; only the refusal itself is
    # asserted here.
    with pytest.raises(ValueError):
        ss.validate_tts_block({"library": "VENDOR/kokoro.min.js"})
    # Same canonical-spelling rule, platform-independently: "./x" is a different
    # URL path from "x", and the error names the spelling to use.
    with pytest.raises(ValueError, match="vendor/kokoro.min.js"):
        ss.validate_tts_block({"library": "./vendor/kokoro.min.js"})


def test_validate_blank_clears_an_override():
    assert ss.validate_tts_block({"voice": ""}) == {"voice": None}
    assert ss.validate_tts_block({"speed": ""}) == {"speed": None}
    assert ss.validate_tts_block({"model": ""}) == {"model": None}


# --------------------------------------------------------------------------- #
#  Endpoint                                                                    #
# --------------------------------------------------------------------------- #

@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path))
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.delenv("LOCALM_REQUIRE_AUTH", raising=False)
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", tmp_path)
    monkeypatch.setattr(cfg, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", tmp_path / "registry.json")
    return tmp_path


@pytest.fixture
def client(env):
    from localm.inference.http_server import create_app
    app = create_app(None)
    with TestClient(
            app, headers={"Authorization": f"Bearer {app.state.shell_token}"}) as c:
        yield c


def _fields(resp):
    assert resp.status_code == 200, resp.text
    return {f["key"]: f for f in resp.json()["fields"]}


def test_get_returns_the_resolved_template_defaults(client):
    r = client.get("/v1/tts/config")
    fields = _fields(r)
    assert r.json()["plugin"] == "tts"
    assert fields["voice"]["value"] == "af_heart"
    assert fields["voice"]["is_override"] is False
    assert "active" in r.json()            # so the GUI can skip a dead section


def test_post_persists_into_the_plugins_tts_block(client):
    r = client.post("/v1/tts/config", json={"voice": "am_onyx", "speed": 1.25})
    assert r.status_code == 200, r.text
    from localm.config import load_config
    block = load_config()["plugins"]["tts"]
    assert block["voice"] == "am_onyx"
    assert block["speed"] == 1.25
    # the response reflects the new state
    assert _fields(r)["voice"]["value"] == "am_onyx"
    assert _fields(r)["voice"]["is_override"] is True


def test_post_merges_and_leaves_other_keys_and_other_plugins_alone(client):
    from localm.config import load_config, update_config
    update_config(lambda c: c.setdefault("plugins", {}).__setitem__(
        "tts", {"voice": "am_onyx", "wasm_paths": "vendor/"}))
    update_config(lambda c: c["plugins"].__setitem__("image", {"comfy": {"workdir": "/x"}}))

    assert client.post("/v1/tts/config", json={"speed": 1.5}).status_code == 200
    cfg = load_config()
    assert cfg["plugins"]["tts"] == {"voice": "am_onyx", "wasm_paths": "vendor/",
                                     "speed": 1.5}
    assert cfg["plugins"]["image"] == {"comfy": {"workdir": "/x"}}


def test_post_blank_clears_the_override_back_to_the_template_default(client):
    assert client.post("/v1/tts/config", json={"voice": "am_onyx"}).status_code == 200
    r = client.post("/v1/tts/config", json={"voice": ""})
    fields = _fields(r)
    assert fields["voice"]["value"] == "af_heart"        # shipped default
    assert fields["voice"]["is_override"] is False
    from localm.config import load_config
    assert load_config()["plugins"]["tts"]["voice"] is None


def test_post_rejects_unknown_field_and_bad_value(client):
    assert client.post("/v1/tts/config", json={"bogus": 1}).status_code == 400
    assert client.post("/v1/tts/config", json={"voice": "nope"}).status_code == 400
    assert client.post("/v1/tts/config", json={"speed": 99}).status_code == 400
    assert client.post("/v1/tts/config",
                       json={"model": "https://evil.example/m"}).status_code == 400


def test_the_plugin_serves_what_the_settings_endpoint_wrote(env):
    """END TO END: the write surface actually changes what the browser gets.

    The plugin's own /api/tts/config is the resolved config tts.js loads, so a
    save through the settings endpoint must show up there."""
    from types import SimpleNamespace

    from fastapi import FastAPI
    from localm.plugins.engine import attach_engine
    from localm.inference.routes import config as config_routes

    app = FastAPI()
    attach_engine(app)                      # real builtin plugins
    # Only the tts routes are exercised here; they do not read ctx (the mode is
    # for GET /v1/config, which this test does not call).
    config_routes.register(app, SimpleNamespace(mode=None))

    with TestClient(app) as c:
        assert c.post("/api/plugins/tts/install").status_code == 200
        assert c.get("/api/tts/config").json()["voice"] == "af_heart"

        r = c.post("/v1/tts/config", json={"voice": "am_onyx", "speed": 1.25})
        assert r.status_code == 200, r.text
        assert r.json()["active"] is True    # installed + enabled

        served = c.get("/api/tts/config").json()
        assert served["voice"] == "am_onyx"
        assert served["speed"] == 1.25
        # untouched keys still resolve to the shipped defaults
        assert served["model"].startswith("onnx-community/Kokoro")


def test_write_requires_config_write_scope(client, monkeypatch):
    with monkeypatch.context() as m:
        m.setenv("LOCALM_API_KEY", "tts-cfg-key")
        assert client.post("/v1/tts/config",
                           json={"voice": "am_onyx"}).status_code == 401


def test_a_tts_capability_key_cannot_read_or_write_these_settings(env):
    """These routes are core, not on the plugin's own router.

    Routes mounted by a plugin are auto-scoped to that plugin's capability, so a
    key that merely grants "may use text-to-speech" would be able to rewrite the
    voice model id and the script URL every browser loads. Settings cost
    config:read / config:write instead."""
    from localm import auth, scopes
    from localm.inference.http_server import create_app

    auth.set_api_key("owner-secret-key-123")                 # protected mode
    tts_only = auth.create_key("speaker", ["tts"])["key"]    # a plain TTS user
    client = TestClient(create_app(None))
    hdr = {"Authorization": f"Bearer {tts_only}"}

    assert client.get("/v1/tts/config", headers=hdr).status_code == 403
    assert client.post("/v1/tts/config", json={"voice": "am_onyx"},
                       headers=hdr).status_code == 403
    # ... and nothing was written
    from localm.config import load_config
    assert not (load_config().get("plugins") or {}).get("tts")

    # A config:read key may READ but still not write.
    reader = auth.create_key("reader", [scopes.CONFIG_READ])["key"]
    rhdr = {"Authorization": f"Bearer {reader}"}
    assert client.get("/v1/tts/config", headers=rhdr).status_code == 200
    assert client.post("/v1/tts/config", json={"voice": "am_onyx"},
                       headers=rhdr).status_code == 403


def test_active_is_false_when_the_plugin_is_not_installed(client):
    """The GUI hides the section on this flag, so it must not lie."""
    assert client.get("/v1/tts/config").json()["active"] is False


def test_script_url_fields_require_an_owner_admin_key(env):
    """library / wasm_paths are loaded as a script + wasm base by EVERY browser
    client, so a non-owner config:write key must not be able to set them, the
    same as a media backend's launch_cmd / api_url."""
    from localm import auth, scopes
    from localm.inference.http_server import create_app

    auth.set_api_key("owner-secret-key-123")                 # protected mode
    writer = auth.create_key("cfgwriter", [scopes.CONFIG_WRITE],
                             allow_privileged=True)["key"]
    client = TestClient(create_app(None))
    owner_hdr = {"Authorization": "Bearer owner-secret-key-123"}
    writer_hdr = {"Authorization": f"Bearer {writer}"}

    # an ordinary field is fine for the scoped key ...
    ok = client.post("/v1/tts/config", json={"voice": "am_onyx"}, headers=writer_hdr)
    assert ok.status_code == 200, ok.text

    # ... the two script-URL fields are not, even with a VALID value
    for key in ("library", "wasm_paths"):
        denied = client.post("/v1/tts/config", json={key: "vendor/kokoro.min.js"},
                             headers=writer_hdr)
        assert denied.status_code == 403, denied.text

    allowed = client.post("/v1/tts/config", json={"library": "vendor/kokoro.min.js"},
                          headers=owner_hdr)
    assert allowed.status_code == 200, allowed.text


def test_get_hides_library_and_wasm_paths_from_a_config_read_only_key(env):
    """A config:read-scoped, non-owner key must not learn library/wasm_paths
    (the script/wasm URL every browser loads) from GET /v1/tts/config, even
    though it may legitimately read every other tts setting."""
    from localm import auth, scopes
    from localm.inference.http_server import create_app

    auth.set_api_key("owner-secret-key-tts-789")               # protected mode
    reader = auth.create_key("reader", [scopes.CONFIG_READ])["key"]
    client = TestClient(create_app(None))
    owner_hdr = {"Authorization": "Bearer owner-secret-key-tts-789"}
    reader_hdr = {"Authorization": f"Bearer {reader}"}

    owner_fields = _fields(client.get("/v1/tts/config", headers=owner_hdr))
    reader_fields = _fields(client.get("/v1/tts/config", headers=reader_hdr))
    assert {"library", "wasm_paths"} <= set(owner_fields), "owner sees everything"
    assert not ({"library", "wasm_paths"} & set(reader_fields)), \
        "a non-owner must not see library/wasm_paths"
    assert "voice" in reader_fields, "an ordinary field stays visible"


def test_post_response_hides_library_and_wasm_paths_from_a_scoped_writer(env):
    """The write endpoint's own response echoes the plugin's resolved fields
    back too (e.g. after saving voice), so a config:write key that is not an
    owner must not have library/wasm_paths' value echoed back, even for a save
    that never touched either field."""
    from localm import auth, scopes
    from localm.inference.http_server import create_app

    auth.set_api_key("owner-secret-key-tts-999")
    writer = auth.create_key("writer", [scopes.CONFIG_WRITE],
                             allow_privileged=True)["key"]
    client = TestClient(create_app(None))
    writer_hdr = {"Authorization": f"Bearer {writer}"}

    r = client.post("/v1/tts/config", json={"voice": "am_onyx"}, headers=writer_hdr)
    assert r.status_code == 200, r.text
    fields = _fields(r)
    assert not ({"library", "wasm_paths"} & set(fields))
    assert "voice" in fields


def test_script_url_fields_allowed_in_open_mode(client):
    # Open mode = the trusted local owner (already gated by the origin / shell
    # token guard), so caller_scopes is None and the admin check does not block.
    r = client.post("/v1/tts/config", json={"library": "vendor/kokoro.min.js"})
    assert r.status_code == 200, r.text
