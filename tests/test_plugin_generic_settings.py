# SPDX-License-Identifier: AGPL-3.0-or-later
"""The generic settings-contribution seam: host.add_settings(), and its
GET/POST /v1/plugins/<name>/settings render/save path.

The path: field-shape validation at register() time, aggregation across active
plugins (PluginManager.get_all_plugin_settings), resolved-value rendering, and
validated persistence into config["plugins"][<name>] - generic over widget
rather than tied to a fixed field list the way the tts/media blocks are.
"""

import textwrap

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


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


def _make_settings_plugin(root, name, *, toml_extra=""):
    """A synthetic plugin that contributes a small, varied set of fields:
    one of every widget family this generic path handles, plus one
    admin_only/secret pair."""
    pdir = root / name
    pdir.mkdir(parents=True, exist_ok=True)
    (pdir / "plugin.toml").write_text(
        f'[plugin]\nname = "{name}"\nscope = "{name}"\nregister = "plug"\n{toml_extra}',
        encoding="utf-8")
    (pdir / "plug.py").write_text(textwrap.dedent('''
        from localm.plugins.contract import PluginSettingField
        from localm.settings_schema import Widget

        def register(host):
            host.add_settings([
                PluginSettingField("greeting", Widget.TEXT, "Greeting",
                                   "Shown to the user.", default="hi"),
                PluginSettingField("max_results", Widget.NUMBER, "Max results",
                                   default=5, min=1, max=50),
                PluginSettingField("enabled", Widget.TOGGLE, "Enabled", default=True),
                PluginSettingField("mode", Widget.SELECT, "Mode",
                                   options=["fast", "accurate"], default="fast"),
                PluginSettingField("api_key", Widget.SECRET, "API key",
                                   admin_only=True),
            ])

        def unregister():
            pass
    '''), encoding="utf-8")
    return pdir


def _bad_plugin(root, name, body):
    pdir = root / name
    pdir.mkdir(parents=True, exist_ok=True)
    (pdir / "plugin.toml").write_text(
        f'[plugin]\nname = "{name}"\nscope = "{name}"\nregister = "plug"\n',
        encoding="utf-8")
    (pdir / "plug.py").write_text(textwrap.dedent(body), encoding="utf-8")
    return pdir


def _install_real_plugin(env, name="myplug"):
    """Install the synthetic settings plugin under the REAL installed-plugins
    dir (plugins_dir(), inside the patched LOCALM_HOME) and mark it enabled in
    config - what create_app(None)'s own PluginManager (default roots) needs
    to load it at construction time. Used only by tests that need the real
    auth/scopes machinery create_app(None) wires up; the plain functional tests
    above use a local PluginManager pointed at a throwaway dir instead."""
    import shutil

    from localm.config import update_config
    from localm.plugins.loader import plugins_dir

    tmp = env / "_settings_plugin_src"
    _make_settings_plugin(tmp, name)
    real_dir = plugins_dir()
    real_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(tmp / name, real_dir / name)
    update_config(lambda c: c.setdefault("plugins_enabled", []).append(name))


# --------------------------------------------------------------------------- #
#  Unit: PluginHost.add_settings shape validation                             #
# --------------------------------------------------------------------------- #

def test_add_settings_rejects_a_non_field_object(env):
    from unittest.mock import MagicMock
    from localm.plugins.engine import PluginHost

    spec = MagicMock()
    spec.name = "alpha"
    host = PluginHost(MagicMock(), MagicMock(), spec)
    with pytest.raises(TypeError):
        host.add_settings([{"key": "x", "widget": "text", "label": "X"}])
    assert host.settings == []          # rejected shape never lands


def test_add_settings_rejects_an_unknown_widget(env):
    from unittest.mock import MagicMock
    from localm.plugins.contract import PluginSettingField
    from localm.plugins.engine import PluginHost

    spec = MagicMock()
    spec.name = "alpha"
    host = PluginHost(MagicMock(), MagicMock(), spec)
    with pytest.raises(ValueError):
        host.add_settings([PluginSettingField("x", "not-a-real-widget", "X")])
    assert host.settings == []


def test_add_settings_accepts_valid_fields(env):
    from unittest.mock import MagicMock
    from localm.plugins.contract import PluginSettingField
    from localm.plugins.engine import PluginHost
    from localm.settings_schema import Widget

    spec = MagicMock()
    spec.name = "alpha"
    host = PluginHost(MagicMock(), MagicMock(), spec)
    fields = [PluginSettingField("greeting", Widget.TEXT, "Greeting", default="hi")]
    host.add_settings(fields)
    assert host.settings == fields
    # A second call ADDS rather than replaces.
    more = [PluginSettingField("mode", Widget.SELECT, "Mode", options=["a", "b"])]
    host.add_settings(more)
    assert host.settings == fields + more


# --------------------------------------------------------------------------- #
#  Unit: settings_schema generic helpers                                      #
# --------------------------------------------------------------------------- #

def _field_list():
    from localm.plugins.contract import PluginSettingField
    from localm.settings_schema import Widget
    return [
        PluginSettingField("greeting", Widget.TEXT, "Greeting", default="hi"),
        PluginSettingField("max_results", Widget.NUMBER, "Max results",
                           default=5, min=1, max=50),
        PluginSettingField("enabled", Widget.TOGGLE, "Enabled", default=True),
        PluginSettingField("mode", Widget.SELECT, "Mode",
                           options=["fast", "accurate"], default="fast"),
        PluginSettingField("api_key", Widget.SECRET, "API key", admin_only=True),
    ]


def test_schema_json_resolves_block_over_default():
    from localm import settings_schema as ss
    fields = {f["key"]: f for f in ss.plugin_settings_schema_json(_field_list(), {})}
    assert fields["greeting"]["value"] == "hi"
    assert fields["greeting"]["is_override"] is False
    assert fields["max_results"]["value"] == 5

    overridden = {f["key"]: f for f in
                 ss.plugin_settings_schema_json(_field_list(), {"greeting": "yo"})}
    assert overridden["greeting"]["value"] == "yo"
    assert overridden["greeting"]["is_override"] is True


def test_schema_json_never_returns_a_secret_value(env):
    from localm import settings_schema as ss
    fields = {f["key"]: f for f in
             ss.plugin_settings_schema_json(_field_list(), {"api_key": "sk-real-secret"})}
    assert "value" not in fields["api_key"]
    assert "default" not in fields["api_key"]
    # is_override is still reported (so the GUI can show "configured" without
    # leaking the value), same shape as SECRET fields elsewhere in the schema.
    assert fields["api_key"]["is_override"] is True


def test_schema_json_hides_admin_only_for_non_owner():
    from localm import settings_schema as ss
    owner = {f["key"] for f in ss.plugin_settings_schema_json(_field_list(), {})}
    scoped = {f["key"] for f in
             ss.plugin_settings_schema_json(_field_list(), {}, is_owner=False)}
    assert "api_key" in owner and "api_key" not in scoped
    assert "greeting" in scoped


def test_admin_only_fields_helper():
    from localm import settings_schema as ss
    assert ss.plugin_settings_admin_only_fields(_field_list()) == {"api_key"}


def test_validate_coerces_each_widget_kind():
    from localm import settings_schema as ss
    merged = ss.validate_plugin_settings_update(_field_list(), {
        "greeting": "hey", "max_results": "12", "enabled": "false", "mode": "accurate",
    })
    assert merged == {"greeting": "hey", "max_results": 12.0,
                      "enabled": False, "mode": "accurate"}


def test_validate_rejects_unknown_key():
    from localm import settings_schema as ss
    with pytest.raises(ValueError):
        ss.validate_plugin_settings_update(_field_list(), {"bogus": 1})


def test_validate_rejects_out_of_range_number():
    from localm import settings_schema as ss
    with pytest.raises(ValueError):
        ss.validate_plugin_settings_update(_field_list(), {"max_results": 999})


def test_validate_rejects_value_outside_select_options():
    from localm import settings_schema as ss
    with pytest.raises(ValueError):
        ss.validate_plugin_settings_update(_field_list(), {"mode": "nonexistent"})


def test_validate_blank_clears_to_none():
    from localm import settings_schema as ss
    assert ss.validate_plugin_settings_update(_field_list(), {"greeting": ""}) == \
        {"greeting": None}


# --------------------------------------------------------------------------- #
#  PluginManager.get_all_plugin_settings aggregation                          #
# --------------------------------------------------------------------------- #

def test_manager_aggregates_settings_from_loaded_plugins_only(env):
    from localm.plugins.engine import PluginManager
    plugins = env / "plugins"
    _make_settings_plugin(plugins, "myplug")
    app = FastAPI()
    mgr = PluginManager(app, external_root=plugins, builtin_root=None)
    mgr.discover()

    assert mgr.get_all_plugin_settings() == []      # not installed yet

    mgr.install("myplug")
    sections = mgr.get_all_plugin_settings()
    assert len(sections) == 1
    assert sections[0]["plugin"] == "myplug"
    keys = {f.key for f in sections[0]["fields"]}
    assert keys == {"greeting", "max_results", "enabled", "mode", "api_key"}

    mgr.disable("myplug")
    assert mgr.get_all_plugin_settings() == []       # disabled -> gone, like its routes

    mgr.enable("myplug")
    assert len(mgr.get_all_plugin_settings()) == 1   # re-enable re-registers them


def test_a_plugin_with_no_add_settings_call_contributes_nothing(env):
    from localm.plugins.engine import PluginManager
    plugins = env / "plugins"
    pdir = plugins / "quiet"
    pdir.mkdir(parents=True)
    (pdir / "plugin.toml").write_text(
        '[plugin]\nname = "quiet"\nscope = "quiet"\nregister = "plug"\n', encoding="utf-8")
    (pdir / "plug.py").write_text(
        "def register(host):\n    pass\ndef unregister():\n    pass\n", encoding="utf-8")
    app = FastAPI()
    mgr = PluginManager(app, external_root=plugins, builtin_root=None)
    mgr.discover()
    mgr.install("quiet")
    assert mgr.get_all_plugin_settings() == []


def test_a_bad_add_settings_call_fails_the_load_and_is_surfaced(env):
    """A wrong-shaped field fails the load at register() time and is recorded,
    not silently dropped."""
    from localm.plugins.engine import PluginManager
    plugins = env / "plugins"
    _bad_plugin(plugins, "broken", '''
        def register(host):
            host.add_settings([{"key": "x", "widget": "text", "label": "X"}])
        def unregister():
            pass
    ''')
    app = FastAPI()
    mgr = PluginManager(app, external_root=plugins, builtin_root=None)
    mgr.discover()
    with pytest.raises(TypeError):
        mgr.install("broken")
    assert mgr.get_all_plugin_settings() == []


# --------------------------------------------------------------------------- #
#  Endpoint: GET/POST /v1/plugins/<name>/settings                             #
# --------------------------------------------------------------------------- #

@pytest.fixture
def client_with_plugin(env):
    from localm.plugins.engine import attach_engine
    from localm.inference.routes import config as config_routes
    from types import SimpleNamespace

    plugins = env / "plugins"
    _make_settings_plugin(plugins, "myplug")
    app = FastAPI()
    manager = attach_engine(app)
    manager._installed_root = plugins
    manager._store_root = None
    config_routes.register(app, SimpleNamespace(mode=None))
    with TestClient(app) as c:
        assert c.post("/api/plugins/myplug/install").status_code == 200
        yield c


def _fields(resp):
    assert resp.status_code == 200, resp.text
    return {f["key"]: f for f in resp.json()["fields"]}


def test_get_returns_no_sections_when_nothing_is_active(env):
    # In open mode every metadata GET under /v1/* additionally requires the
    # shell token (or an equivalent local-process credential) as a CORS
    # defense-in-depth (http_server.py's is_metadata_get gate), independent of
    # require_scope's own open-mode pass-through.
    from localm.inference.http_server import create_app
    app = create_app(None)
    with TestClient(
            app, headers={"Authorization": f"Bearer {app.state.shell_token}"}) as c:
        r = c.get("/v1/plugins/settings")
        assert r.status_code == 200
        assert r.json() == {"plugins": []}


def test_get_lists_the_active_plugins_section(client_with_plugin):
    r = client_with_plugin.get("/v1/plugins/settings")
    assert r.status_code == 200
    body = r.json()
    assert len(body["plugins"]) == 1
    sec = body["plugins"][0]
    assert sec["plugin"] == "myplug"
    fields = {f["key"]: f for f in sec["fields"]}
    assert fields["greeting"]["value"] == "hi"
    assert fields["greeting"]["is_override"] is False
    assert "api_key" in fields          # open mode = owner


def test_post_persists_into_the_plugins_own_block(client_with_plugin):
    r = client_with_plugin.post("/v1/plugins/myplug/settings",
                                json={"greeting": "hey", "max_results": 10})
    assert r.status_code == 200, r.text
    from localm.config import load_config
    block = load_config()["plugins"]["myplug"]
    assert block["greeting"] == "hey"
    assert block["max_results"] == 10.0
    fields = _fields(r)
    assert fields["greeting"]["value"] == "hey"
    assert fields["greeting"]["is_override"] is True


def test_post_merges_and_leaves_other_keys_and_other_plugins_alone(client_with_plugin):
    from localm.config import load_config, update_config
    update_config(lambda c: c.setdefault("plugins", {}).__setitem__(
        "myplug", {"greeting": "yo", "mode": "accurate"}))
    update_config(lambda c: c["plugins"].__setitem__("other", {"x": 1}))

    assert client_with_plugin.post(
        "/v1/plugins/myplug/settings", json={"max_results": 9}).status_code == 200
    cfg = load_config()
    assert cfg["plugins"]["myplug"] == {"greeting": "yo", "mode": "accurate",
                                        "max_results": 9.0}
    assert cfg["plugins"]["other"] == {"x": 1}


def test_post_blank_clears_the_override_back_to_the_default(client_with_plugin):
    assert client_with_plugin.post(
        "/v1/plugins/myplug/settings", json={"greeting": "hey"}).status_code == 200
    r = client_with_plugin.post("/v1/plugins/myplug/settings", json={"greeting": ""})
    fields = _fields(r)
    assert fields["greeting"]["value"] == "hi"        # declared default
    assert fields["greeting"]["is_override"] is False
    from localm.config import load_config
    assert load_config()["plugins"]["myplug"]["greeting"] is None


def test_post_rejects_unknown_field_and_bad_value(client_with_plugin):
    assert client_with_plugin.post(
        "/v1/plugins/myplug/settings", json={"bogus": 1}).status_code == 400
    assert client_with_plugin.post(
        "/v1/plugins/myplug/settings", json={"mode": "nope"}).status_code == 400
    assert client_with_plugin.post(
        "/v1/plugins/myplug/settings", json={"max_results": 9999}).status_code == 400


def test_post_404s_for_an_unknown_or_inactive_plugin(client_with_plugin):
    r = client_with_plugin.post("/v1/plugins/nosuchplugin/settings", json={"x": 1})
    assert r.status_code == 404


def _bulk_fields(resp, plugin="myplug"):
    """Extract {key: field} for one plugin's section out of a GET
    /v1/plugins/settings response (the bulk list shape - distinct from
    _field_list(), which reads a single-plugin POST response)."""
    assert resp.status_code == 200, resp.text
    sections = {s["plugin"]: s for s in resp.json()["plugins"]}
    return {f["key"]: f for f in sections[plugin]["fields"]} if plugin in sections else {}


def test_get_and_post_require_config_scopes_not_the_plugins_own_scope(env):
    """These live on the core routes, not the plugin's own router, so a key that
    merely grants 'may use myplug' cannot rewrite its settings - the same gate
    the tts/media settings routes use."""
    from localm import auth
    from localm.inference.http_server import create_app

    _install_real_plugin(env)
    auth.set_api_key("owner-secret-key-plugsettings")
    myplug_only = auth.create_key("user", ["myplug"])["key"]
    client = TestClient(create_app(None))
    hdr = {"Authorization": f"Bearer {myplug_only}"}
    assert client.get("/v1/plugins/settings", headers=hdr).status_code == 403
    assert client.post("/v1/plugins/myplug/settings", json={"greeting": "x"},
                       headers=hdr).status_code == 403


def test_admin_only_field_requires_an_owner_key(env):
    from localm import auth, scopes
    from localm.inference.http_server import create_app

    _install_real_plugin(env)
    auth.set_api_key("owner-secret-key-admin-only")
    writer = auth.create_key("writer", [scopes.CONFIG_WRITE],
                             allow_privileged=True)["key"]
    client = TestClient(create_app(None))
    writer_hdr = {"Authorization": f"Bearer {writer}"}
    owner_hdr = {"Authorization": "Bearer owner-secret-key-admin-only"}

    ok = client.post("/v1/plugins/myplug/settings", json={"greeting": "hi there"},
                     headers=writer_hdr)
    assert ok.status_code == 200, ok.text

    denied = client.post("/v1/plugins/myplug/settings", json={"api_key": "sk-x"},
                         headers=writer_hdr)
    assert denied.status_code == 403, denied.text

    allowed = client.post("/v1/plugins/myplug/settings", json={"api_key": "sk-x"},
                          headers=owner_hdr)
    assert allowed.status_code == 200, allowed.text


def test_get_hides_admin_only_value_from_a_config_read_only_key(env):
    from localm import auth, scopes
    from localm.inference.http_server import create_app

    _install_real_plugin(env)
    auth.set_api_key("owner-secret-key-read-only")
    reader = auth.create_key("reader", [scopes.CONFIG_READ])["key"]
    client = TestClient(create_app(None))
    owner_hdr = {"Authorization": "Bearer owner-secret-key-read-only"}
    reader_hdr = {"Authorization": f"Bearer {reader}"}

    owner_fields = _bulk_fields(client.get("/v1/plugins/settings", headers=owner_hdr))
    reader_fields = _bulk_fields(client.get("/v1/plugins/settings", headers=reader_hdr))
    assert "api_key" in owner_fields, "owner sees everything"
    assert "api_key" not in reader_fields, \
        "a non-owner must not see the admin_only field"
    assert "greeting" in reader_fields, "an ordinary field stays visible"
