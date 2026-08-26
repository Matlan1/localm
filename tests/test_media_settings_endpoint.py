# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the per-plugin media config: GET/POST /v1/media/config + the
settings_schema helpers behind them.

Each media plugin (image/music/video) keeps its OWN config block under
config["plugins"][name], so the three are configured INDEPENDENTLY. The GUI
"Media" section reads the resolved values (block value, else the shared global
comfy_* fallback) and writes a plugin's block deep-merged (other fields and the
other plugins untouched). A blank field clears the override.
"""

import os

import pytest
from fastapi.testclient import TestClient

from localm import settings_schema as ss
from localm.inference.http_server import create_app


# --------------------------------------------------------------------------- #
#  Unit: schema + validator                                                    #
# --------------------------------------------------------------------------- #

def test_media_schema_image_has_fast_dequant_others_do_not():
    cfg = {"comfy_workdir": "/global/comfy"}
    img = {f["key"] for f in ss.media_schema_json("image", {}, cfg)}
    mus = {f["key"] for f in ss.media_schema_json("music", {}, cfg)}
    assert "fast_dequant" in img            # Flux-only, image gets it
    assert "fast_dequant" not in mus        # music/video do not
    # the common fields are present for all three
    for key in ("workdir", "launch_cmd", "api_url", "output_dir",
                "delete_outputs", "reload_after", "swap_policy"):
        assert key in img and key in mus


def test_media_schema_admin_only_fields_hidden_for_non_owner():
    # launch_cmd (a shell command) and api_url (a render target) are admin_only,
    # mirroring the write-side gate, so a non-owner caller does not see their
    # resolved value either.
    cfg = {"comfy_workdir": "/global/comfy"}
    owner_fields = {f["key"] for f in ss.media_schema_json("image", {}, cfg)}
    scoped_fields = {f["key"] for f in
                     ss.media_schema_json("image", {}, cfg, is_owner=False)}
    assert {"launch_cmd", "api_url", "workdir"} <= owner_fields, "owner sees everything"
    assert not ({"launch_cmd", "api_url", "workdir"} & scoped_fields), \
        "a non-owner must not see launch_cmd/api_url/workdir"
    # An ordinary media field stays visible to a non-owner, so the filter is
    # SELECTIVE rather than hiding the whole block.
    assert "output_dir" in scoped_fields, "an ordinary field stays visible"


def test_media_admin_only_fields_is_the_single_source_of_truth():
    # workdir is admin_only too: the per-plugin value wins over the global
    # comfy_workdir, and with a blank launch_cmd the launcher is auto-discovered
    # inside that folder and run via Popen, so it reaches execution the same way
    # launch_cmd does.
    assert ss.media_admin_only_fields() == {"launch_cmd", "api_url", "workdir"}


def test_media_schema_resolves_block_over_global():
    cfg = {"comfy_workdir": "/global/comfy", "comfy_delete_outputs": False}
    block = {"comfy": {"workdir": "/image/own"}}
    fields = {f["key"]: f for f in ss.media_schema_json("image", block, cfg)}
    # block override wins and is flagged
    assert fields["workdir"]["value"] == "/image/own"
    assert fields["workdir"]["is_override"] is True
    # no override -> falls back to the shared global, not flagged
    assert fields["delete_outputs"]["value"] is False
    assert fields["delete_outputs"]["is_override"] is False


def test_validate_media_block_shapes_nested_merge():
    merge = ss.validate_media_block("image", {
        "workdir": "/x", "delete_outputs": True, "swap_policy": "always",
        "reload_after": False,
    })
    assert merge == {
        "comfy": {"workdir": "/x", "delete_outputs": True},
        "model_swap_policy": "always",
        "reload_llm_after_generate": False,
    }


def test_validate_media_block_rejects_unknown_and_bad_values():
    with pytest.raises(ValueError):
        ss.validate_media_block("nope", {"workdir": "/x"})
    with pytest.raises(ValueError):
        ss.validate_media_block("image", {"not_a_field": 1})
    with pytest.raises(ValueError):
        ss.validate_media_block("image", {"swap_policy": "sometimes"})
    # fast_dequant is image-only: rejected for music
    with pytest.raises(ValueError):
        ss.validate_media_block("music", {"fast_dequant": True})


def test_validate_media_block_blank_clears_override():
    merge = ss.validate_media_block("image", {"workdir": ""})
    assert merge == {"comfy": {"workdir": None}}


def test_validate_media_block_rejects_malformed_api_url():
    # A non-URL or a scheme-less / hostless value is rejected at SET time (surfaced
    # as a ValueError -> 400 by the route) instead of being stored and silently
    # dropped to the loopback default at READ time by sanitize_comfy_url.
    for bad in ("not a url", "127.0.0.1:8188", "localhost:8188",
                "ftp://host", "://nohost", "http://"):
        with pytest.raises(ValueError):
            ss.validate_media_block("image", {"api_url": bad})


def test_validate_media_block_accepts_valid_api_url_and_blank_clears():
    assert ss.validate_media_block("image", {"api_url": "http://127.0.0.1:8188"}) == \
        {"comfy": {"api_url": "http://127.0.0.1:8188"}}
    # LAN host + https, on a different plugin, is fine too
    assert ss.validate_media_block("music", {"api_url": "https://box.lan:8188"}) == \
        {"comfy": {"api_url": "https://box.lan:8188"}}
    # blank still clears the override without a URL check
    assert ss.validate_media_block("image", {"api_url": ""}) == \
        {"comfy": {"api_url": None}}


# --------------------------------------------------------------------------- #
#  Endpoint                                                                    #
# --------------------------------------------------------------------------- #

@pytest.fixture
def client(tmp_path, monkeypatch):
    os.environ.pop("LOCALM_API_KEY", None)
    import localm.config as cfg
    home = tmp_path / ".localm"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.setattr(cfg, "HOME_DIR", home)
    monkeypatch.setattr(cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", home / "registry.json")
    app = create_app(None)
    with TestClient(
        app, headers={"Authorization": f"Bearer {app.state.shell_token}"}) as c:
        yield c


def _media(client):
    r = client.get("/v1/media/config")
    assert r.status_code == 200, r.text
    return {p["plugin"]: p for p in r.json()["plugins"]}


def test_get_returns_three_independent_plugins(client):
    plugins = _media(client)
    assert set(plugins) == {"image", "music", "video"}
    assert plugins["image"]["label"] == "Image"
    assert all(p["fields"] for p in plugins.values())


def test_get_resolves_global_fallback(client):
    # set a global comfy_workdir, then every plugin with no override shows it
    assert client.patch("/v1/config",
                        json={"comfy_workdir": "/shared/comfy"}).status_code == 200
    plugins = _media(client)
    for name in ("image", "music", "video"):
        wf = {f["key"]: f for f in plugins[name]["fields"]}["workdir"]
        assert wf["value"] == "/shared/comfy"
        assert wf["is_override"] is False


def test_post_sets_override_and_persists_and_deep_merges(client):
    # set two fields in separate calls; both must persist (deep merge)
    assert client.post("/v1/media/config/image",
                       json={"workdir": "/img/own"}).status_code == 200
    r = client.post("/v1/media/config/image", json={"delete_outputs": True})
    assert r.status_code == 200, r.text

    from localm.config import load_config
    block = load_config()["plugins"]["image"]
    assert block["comfy"]["workdir"] == "/img/own"        # not clobbered
    assert block["comfy"]["delete_outputs"] is True

    # and the image backend reads the saved block
    from localm.plugins.builtin.image import backend as img_backend
    s = img_backend.settings(load_config())
    assert s["workdir"] == "/img/own"
    assert s["delete_outputs"] is True


def test_post_is_per_plugin_independent(client):
    assert client.post("/v1/media/config/music",
                       json={"workdir": "/music/own"}).status_code == 200
    from localm.config import load_config
    cfg = load_config()
    assert cfg["plugins"]["music"]["comfy"]["workdir"] == "/music/own"
    # image + video have no block from this
    assert "image" not in cfg.get("plugins", {}) or \
        not cfg["plugins"].get("image", {}).get("comfy", {}).get("workdir")


def test_post_blank_clears_override_back_to_global(client):
    client.patch("/v1/config", json={"comfy_workdir": "/shared/comfy"})
    client.post("/v1/media/config/image", json={"workdir": "/img/own"})
    # now clear it
    r = client.post("/v1/media/config/image", json={"workdir": ""})
    assert r.status_code == 200
    wf = {f["key"]: f for f in r.json()["fields"]}["workdir"]
    assert wf["value"] == "/shared/comfy"      # back to the shared default
    assert wf["is_override"] is False


def test_post_block_level_fields(client):
    r = client.post("/v1/media/config/video",
                    json={"swap_policy": "always", "reload_after": False})
    assert r.status_code == 200, r.text
    from localm.config import load_config
    block = load_config()["plugins"]["video"]
    assert block["model_swap_policy"] == "always"
    assert block["reload_llm_after_generate"] is False


def test_post_unknown_plugin_404_and_bad_field_400(client):
    assert client.post("/v1/media/config/nope", json={"workdir": "/x"}).status_code == 404
    assert client.post("/v1/media/config/image",
                       json={"bogus": 1}).status_code == 400
    assert client.post("/v1/media/config/image",
                       json={"swap_policy": "sometimes"}).status_code == 400


def test_post_rejects_malformed_api_url(client):
    # Open-mode owner (shell token) -> the api_url admin gate is bypassed, so the
    # value reaches validation: a malformed URL is a 400, a well-formed one persists.
    assert client.post("/v1/media/config/image",
                       json={"api_url": "not a url"}).status_code == 400
    r = client.post("/v1/media/config/image",
                    json={"api_url": "http://127.0.0.1:8188"})
    assert r.status_code == 200, r.text
    from localm.config import load_config
    assert load_config()["plugins"]["image"]["comfy"]["api_url"] == "http://127.0.0.1:8188"


def test_write_requires_config_write_scope(client, monkeypatch):
    with monkeypatch.context() as m:
        m.setenv("LOCALM_API_KEY", "media-cfg-key")
        denied = client.post("/v1/media/config/image", json={"workdir": "/x"})
        assert denied.status_code == 401


# --------------------------------------------------------------------------- #
#  Owner vs. non-owner: launch_cmd/api_url must not leak                      #
# --------------------------------------------------------------------------- #

@pytest.fixture
def env(tmp_path, monkeypatch):
    """Same isolated data dir as `client`, but WITHOUT a pre-built app/client,
    so a test can put the server in protected mode (LOCALM_API_KEY) itself."""
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path))
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.delenv("LOCALM_REQUIRE_AUTH", raising=False)
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", tmp_path)
    monkeypatch.setattr(cfg, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", tmp_path / "registry.json")
    return tmp_path


def test_get_hides_launch_cmd_and_api_url_from_a_config_read_only_key(env):
    """A config:read-scoped, non-owner key (part of the app's own suggested
    'Full' key preset) must not learn a media backend's launch_cmd (a shell
    command) or api_url (a render target) from GET /v1/media/config, even though
    it may legitimately read every other field. The write side already refuses
    this key the ability to SET either field; the read side must match."""
    from localm import auth, scopes

    auth.set_api_key("owner-secret-media-123")                # protected mode
    reader = auth.create_key("reader", [scopes.CONFIG_READ])["key"]
    client = TestClient(create_app(None))
    owner_hdr = {"Authorization": "Bearer owner-secret-media-123"}
    reader_hdr = {"Authorization": f"Bearer {reader}"}

    owner_resp = client.get("/v1/media/config", headers=owner_hdr)
    reader_resp = client.get("/v1/media/config", headers=reader_hdr)
    assert owner_resp.status_code == 200, owner_resp.text
    assert reader_resp.status_code == 200, reader_resp.text

    for plugin in owner_resp.json()["plugins"]:
        owner_keys = {f["key"] for f in plugin["fields"]}
        assert {"launch_cmd", "api_url"} <= owner_keys, "owner sees everything"

    for plugin in reader_resp.json()["plugins"]:
        reader_keys = {f["key"] for f in plugin["fields"]}
        assert not ({"launch_cmd", "api_url", "workdir"} & reader_keys), \
            f"{plugin['plugin']}: launch_cmd/api_url/workdir must be hidden " \
            "from a scoped key"
        # The over-gating control: an ordinary field (output_dir) stays visible.
        assert "output_dir" in reader_keys, "an ordinary field stays visible"


def test_post_response_hides_launch_cmd_and_api_url_from_a_scoped_writer(env):
    """The write endpoint's own response echoes the plugin's resolved fields
    back too (e.g. after saving workdir) - same leak, same fix: a config:write
    key that is not an owner must not have launch_cmd/api_url's value echoed
    back even for a save that never touched either field."""
    from localm import auth, scopes

    auth.set_api_key("owner-secret-media-456")
    writer = auth.create_key("writer", [scopes.CONFIG_WRITE],
                             allow_privileged=True)["key"]
    client = TestClient(create_app(None))
    writer_hdr = {"Authorization": f"Bearer {writer}"}

    # The saved field must be one this key may actually SET: workdir sits on the
    # owner gate, so posting it here would 403.
    r = client.post("/v1/media/config/image", json={"output_dir": "/x"},
                    headers=writer_hdr)
    assert r.status_code == 200, r.text
    fields = {f["key"] for f in r.json()["fields"]}
    assert not ({"launch_cmd", "api_url", "workdir"} & fields)
    assert "output_dir" in fields


def test_generic_config_get_does_not_leak_per_plugin_media_secrets(env):
    """The owner gate on launch_cmd/api_url/workdir must not be defeated by
    reading them off the GENERIC route.

    GET /v1/media/config hides those three from a non-owner, but the media
    plugins keep their OWN copy at cfg["plugins"][<plugin>]["comfy"][...]. That
    subtree lives under the `plugins` key, which is engine_managed - a WRITE
    gate only, never popped from a read - so without the scrub a plain
    config:read key could fetch from /v1/config exactly what /v1/media/config
    refuses it.
    """
    from localm import auth, scopes
    from localm.config import update_config

    auth.set_api_key("owner-secret-media-789")
    reader = auth.create_key("reader", [scopes.CONFIG_READ])["key"]
    update_config(lambda c: c.__setitem__("plugins", {
        "image": {"comfy": {"launch_cmd": "attacker.bat",
                            "api_url": "http://attacker.invalid",
                            "workdir": "/attacker/dir",
                            "output_dir": "/ordinary/dir"}}}))
    client = TestClient(create_app(None))

    owner = client.get("/v1/config",
                       headers={"Authorization": "Bearer owner-secret-media-789"})
    scoped = client.get("/v1/config",
                        headers={"Authorization": f"Bearer {reader}"})
    assert owner.status_code == 200 and scoped.status_code == 200, scoped.text

    owner_comfy = owner.json()["plugins"]["image"]["comfy"]
    scoped_comfy = scoped.json()["plugins"]["image"]["comfy"]

    # Both halves: the owner still sees the hidden value, and an ordinary
    # per-plugin field stays readable.
    assert owner_comfy.get("launch_cmd") == "attacker.bat", \
        "precondition: the owner must still see the value being hidden"
    assert scoped_comfy.get("output_dir") == "/ordinary/dir", \
        "the scrub must be TARGETED - an ordinary per-plugin field stays readable"

    leaked = {"launch_cmd", "api_url", "workdir"} & set(scoped_comfy)
    assert not leaked, f"non-owner read the owner-only media values: {sorted(leaked)}"
