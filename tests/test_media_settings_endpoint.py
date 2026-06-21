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


def test_write_requires_config_write_scope(client, monkeypatch):
    with monkeypatch.context() as m:
        m.setenv("LOCALM_API_KEY", "media-cfg-key")
        denied = client.post("/v1/media/config/image", json={"workdir": "/x"})
        assert denied.status_code == 401
