# SPDX-License-Identifier: AGPL-3.0-or-later
"""`localm plugin config <name> [<key> [<value>]]` - the terminal's reach into a
plugin's OWN settings block.

Three HTTP routes edit those blocks (POST /v1/media/config/<name>,
POST /v1/tts/config, POST /v1/plugins/<name>/settings). `localm config plugins
'<json>'` cannot stand in for this command: settings_schema requires a real
dict there and click always passes a str.

The three do not share a source for their field list. media and tts are module
constants this process can read offline; a host.add_settings() block is supplied
at register() time and therefore exists only inside a process that has LOADED
that plugin, which the CLI never is. These tests pin BOTH paths, and pin that
the several "nothing to show" states stay distinguishable rather than collapsing
into one empty answer.
"""

import json

import pytest
from click.testing import CliRunner


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path))
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.delenv("LOCALM_URL", raising=False)
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", tmp_path)
    monkeypatch.setattr(cfg, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", tmp_path / "registry.json")
    return tmp_path


def _run(*args):
    from localm.cli import main
    return CliRunner().invoke(main, ["plugin", "config", *args])


def _blocks(env):
    f = env / "config.json"
    return json.loads(f.read_text(encoding="utf-8")).get("plugins", {}) if f.is_file() else {}


# --------------------------------------------------------------------------- #
#  The discriminator: which source can describe this plugin's block           #
# --------------------------------------------------------------------------- #

def test_plugin_config_kind_splits_static_from_runtime():
    from localm import settings_schema as ss
    assert ss.plugin_config_kind("image") == "media"
    assert ss.plugin_config_kind("music") == "media"
    assert ss.plugin_config_kind("video") == "media"
    assert ss.plugin_config_kind("tts") == "tts"
    # Anything else is only knowable from a process that loaded the plugin.
    assert ss.plugin_config_kind("widget") == "runtime"


def test_local_helpers_refuse_a_runtime_plugin():
    """They RAISE rather than returning [], which would read as 'this plugin has
    no settings' - a different answer."""
    from localm import settings_schema as ss
    with pytest.raises(ValueError):
        ss.local_plugin_config_fields("widget", {})
    with pytest.raises(ValueError):
        ss.local_plugin_config_keys("widget")
    with pytest.raises(ValueError):
        ss.apply_local_plugin_config("widget", "greeting", "hi")


# --------------------------------------------------------------------------- #
#  Media: the per-plugin DIFFERENTIATION the global comfy_* keys cannot express #
# --------------------------------------------------------------------------- #

def test_two_media_plugins_hold_different_values_for_the_same_field(env):
    """`localm config comfy_api_url` sets ONE value for all three plugins, so it
    cannot express image and music pointing at different ComfyUI installs; the
    per-plugin block can."""
    assert _run("image", "api_url", "http://127.0.0.1:9999").exit_code == 0
    assert _run("music", "api_url", "http://127.0.0.1:8188").exit_code == 0
    blocks = _blocks(env)
    assert blocks["image"]["comfy"]["api_url"] == "http://127.0.0.1:9999"
    assert blocks["music"]["comfy"]["api_url"] == "http://127.0.0.1:8188"
    # And reading one back shows its OWN value, not the other's.
    assert "9999" in _run("image", "api_url").output


def test_a_blank_value_clears_back_to_the_global_default(env):
    from localm.config import update_config
    update_config(lambda c: c.__setitem__("comfy_api_url", "http://global:8188"))
    _run("image", "api_url", "http://127.0.0.1:9999")
    res = _run("image", "api_url", "")
    assert res.exit_code == 0
    assert "cleared" in res.output
    # Resolved value falls back to the shared global, and is reported as such.
    assert "http://global:8188" in _run("image", "api_url").output


def test_a_write_leaves_the_other_fields_and_the_other_plugins_alone(env):
    _run("image", "api_url", "http://127.0.0.1:9999")
    _run("image", "swap_policy", "always")
    _run("music", "api_url", "http://127.0.0.1:8188")
    blocks = _blocks(env)
    assert blocks["image"]["comfy"]["api_url"] == "http://127.0.0.1:9999"
    assert blocks["image"]["model_swap_policy"] == "always"
    assert blocks["music"]["comfy"]["api_url"] == "http://127.0.0.1:8188"


def test_a_bad_value_is_refused_with_the_validators_own_message(env):
    res = _run("image", "api_url", "not-a-url")
    assert res.exit_code != 0
    assert "valid http(s) URL" in res.output
    assert "image" not in _blocks(env)          # and nothing was written


def test_an_unknown_key_is_refused_and_lists_the_settable_ones(env):
    res = _run("image", "nope", "1")
    assert res.exit_code != 0
    assert "nope" in res.output
    for expected in ("api_url", "swap_policy", "workflow", "use_config_from"):
        assert expected in res.output


def test_image_only_and_plugin_restricted_fields_follow_media_fields_for(env):
    """fast_dequant is image-only and float_type is music/video-only, so the
    settable key list has to come from media_fields_for, not the whole list."""
    from localm import settings_schema as ss
    assert "fast_dequant" in ss.local_plugin_config_keys("image")
    assert "fast_dequant" not in ss.local_plugin_config_keys("music")
    assert "float_type" in ss.local_plugin_config_keys("video")
    assert "float_type" not in ss.local_plugin_config_keys("image")
    assert _run("image", "float_type", "fp16").exit_code != 0


# --------------------------------------------------------------------------- #
#  use_config_from: cycle prevention                                          #
# --------------------------------------------------------------------------- #

def test_share_config_pointer_round_trips(env):
    assert _run("music", "use_config_from", "image").exit_code == 0
    assert _blocks(env)["music"]["use_config_from"] == "image"
    res = _run("music", "use_config_from", "")
    assert res.exit_code == 0
    assert "use_config_from" not in _blocks(env)["music"]


def test_a_share_config_cycle_is_refused(env):
    assert _run("music", "use_config_from", "image").exit_code == 0
    res = _run("image", "use_config_from", "music")
    assert res.exit_code != 0
    assert "cycle" in res.output
    assert "use_config_from" not in _blocks(env).get("image", {})


def test_share_config_rejects_a_non_media_target(env):
    assert _run("image", "use_config_from", "tts").exit_code != 0
    assert _run("image", "use_config_from", "image").exit_code != 0


# --------------------------------------------------------------------------- #
#  workflow: dispatched to select_workflow, which owns the rule               #
# --------------------------------------------------------------------------- #

def _save_workflow(name="mine"):
    from localm import media_workflows
    return media_workflows.save_workflow(
        "image", name, json.dumps({"1": {"class_type": "KSampler"}}).encode())


def test_workflow_is_refused_when_the_file_does_not_exist(env):
    res = _run("image", "workflow", "ghost.json")
    assert res.exit_code != 0
    assert "no such workflow" in res.output


def test_workflow_selects_an_existing_file_and_clearing_pops_the_key(env):
    saved = _save_workflow()
    assert _run("image", "workflow", saved).exit_code == 0
    assert _blocks(env)["image"]["workflow"] == saved
    from localm import media_workflows
    assert media_workflows.selected_name("image") == saved
    # Cleared the way the GUI's own selection route clears it: the key is
    # POPPED, not written as None.
    assert _run("image", "workflow", "").exit_code == 0
    assert "workflow" not in _blocks(env)["image"]


# --------------------------------------------------------------------------- #
#  tts                                                                        #
# --------------------------------------------------------------------------- #

def test_tts_block_round_trips_and_validates(env):
    assert _run("tts", "speed", "1.25").exit_code == 0
    assert _blocks(env)["tts"]["speed"] == 1.25
    over = _run("tts", "speed", "9")
    assert over.exit_code != 0 and "at most" in over.output
    bad = _run("tts", "voice", "not_a_voice")
    assert bad.exit_code != 0


def test_listing_a_static_plugin_needs_no_server_and_no_install(env):
    """The tts/media write path is NOT gated on the plugin being active, so the
    block can be prepared before it is enabled, same as the routes. The listing
    says which state it is in rather than refusing."""
    res = _run("tts")
    assert res.exit_code == 0
    assert "voice" in res.output and "speed" in res.output


# --------------------------------------------------------------------------- #
#  The runtime path: the states that must NOT collapse into one empty answer  #
# --------------------------------------------------------------------------- #

def _install_widget(env, *, enable):
    import textwrap

    from localm.plugins.engine import PluginManager
    src = env / "_src" / "widget"
    src.mkdir(parents=True, exist_ok=True)
    (src / "plugin.toml").write_text(
        '[plugin]\nname = "widget"\nscope = "widget"\nregister = "plug"\n',
        encoding="utf-8")
    (src / "plug.py").write_text(textwrap.dedent('''
        from localm.plugins.contract import PluginSettingField
        from localm.settings_schema import Widget

        def register(host):
            host.add_settings([
                PluginSettingField("greeting", Widget.TEXT, "Greeting", default="hi"),
                PluginSettingField("api_key", Widget.SECRET, "API key", admin_only=True),
            ])

        def unregister():
            pass
    '''), encoding="utf-8")
    PluginManager(None).set_installed_from_dir(src, enable=enable)


def test_a_name_that_is_not_a_plugin_says_no_such_plugin(env):
    res = _run("totally-made-up")
    assert res.exit_code != 0
    assert "No such plugin" in res.output


def test_installed_but_disabled_says_so_rather_than_no_settings(env):
    _install_widget(env, enable=False)
    res = _run("widget")
    assert res.exit_code != 0
    assert "not enabled" in res.output
    assert "plugin enable widget" in res.output


def test_enabled_but_no_server_says_the_plugin_must_be_running(env):
    """A host.add_settings() field list exists only inside a process that loaded
    the plugin, so with no server this command cannot ASK, and says so rather
    than answering as though it had asked and found nothing."""
    _install_widget(env, enable=True)
    res = _run("widget")
    assert res.exit_code != 0
    assert "declared when the plugin loads" in res.output
    assert "running localm" in res.output
    # ... and is distinguishable from both neighbouring states.
    assert "No such plugin" not in res.output
    assert "declares no settings" not in res.output


def test_the_cli_never_loads_a_plugin_to_answer(env, monkeypatch):
    """Reading a field list does not run a plugin's register(). PluginManager._load
    fires the on_first_use lifecycle hook and writes config for it, so a load
    here would turn a read into a side effect, and would fail outright for any
    route-mounting plugin, since the CLI has no app."""
    from localm.plugins.engine import PluginManager
    _install_widget(env, enable=True)
    loaded = []
    real_load = PluginManager._load
    monkeypatch.setattr(PluginManager, "_load",
                        lambda self, spec: (loaded.append(spec.name),
                                            real_load(self, spec))[1])
    _run("widget")
    _run("image")
    _run("tts")
    assert loaded == []


# --------------------------------------------------------------------------- #
#  The runtime path's own reporting, with the HTTP layer controlled            #
#                                                                             #
#  These drive _runtime_plugin_config against a stubbed server rather than a  #
#  live one, so the reporting logic is pinned without a process to start.     #
#  Each asserts the CALL COUNT as well as the outcome.                        #
# --------------------------------------------------------------------------- #

class _Resp:
    def __init__(self, payload, status=200):
        self._payload, self.status_code = payload, status

    def json(self):
        return self._payload


def _stub_server(monkeypatch, *, fields, on_post=None):
    """Point the CLI at a fake server and record what it sends."""
    import requests
    monkeypatch.setenv("LOCALM_URL", "http://127.0.0.1:1")
    calls = {"get": 0, "post": 0, "body": None}

    def fake_get(url, **kw):
        calls["get"] += 1
        return _Resp({"plugins": [{"plugin": "widget", "label": "Widget",
                                   "fields": fields}]})

    def fake_post(url, **kw):
        calls["post"] += 1
        calls["body"] = kw.get("json")
        return on_post(kw.get("json")) if on_post else _Resp(
            {"plugin": "widget", "fields": fields})

    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr(requests, "post", fake_post)
    return calls


def test_setting_a_secret_reports_a_set_not_a_clear(env, monkeypatch):
    """A SECRET field's value is never echoed back by the server, so the reply's
    absent `value` cannot tell a set from a clear. `is_override` is the field
    that distinguishes them."""
    _install_widget(env, enable=True)
    saved = [{"key": "api_key", "widget": "secret", "label": "API key",
              "is_override": True, "admin_only": True}]        # no "value", by design
    calls = _stub_server(monkeypatch, fields=saved,
                         on_post=lambda body: _Resp({"plugin": "widget",
                                                     "fields": saved}))
    res = _run("widget", "api_key", "sk-secret")
    assert calls["post"] == 1                       # the stub really intercepted
    assert calls["body"] == {"api_key": "sk-secret"}
    assert res.exit_code == 0
    assert "set" in res.output
    assert "cleared" not in res.output
    assert "sk-secret" not in res.output            # and never echoes the value


def test_clearing_a_secret_still_reports_a_clear(env, monkeypatch):
    """The other arm: clearing a secret reports a clear, so a report that always
    said "set" would fail here."""
    _install_widget(env, enable=True)
    cleared = [{"key": "api_key", "widget": "secret", "label": "API key",
                "is_override": False, "admin_only": True}]
    calls = _stub_server(monkeypatch, fields=cleared,
                         on_post=lambda body: _Resp({"plugin": "widget",
                                                     "fields": cleared}))
    res = _run("widget", "api_key", "")
    assert calls["post"] == 1
    assert res.exit_code == 0
    assert "cleared" in res.output


def test_a_runtime_plugin_lists_and_sets_through_the_server(env, monkeypatch):
    _install_widget(env, enable=True)
    fields = [{"key": "greeting", "widget": "text", "label": "Greeting",
               "value": "hi", "default": "hi", "is_override": False}]
    calls = _stub_server(monkeypatch, fields=fields)
    listing = _run("widget")
    assert calls["get"] == 1 and listing.exit_code == 0
    assert "greeting" in listing.output

    updated = [dict(fields[0], value="yo", is_override=True)]
    calls = _stub_server(monkeypatch, fields=fields,
                         on_post=lambda body: _Resp({"plugin": "widget",
                                                     "fields": updated}))
    res = _run("widget", "greeting", "yo")
    assert calls["post"] == 1 and calls["body"] == {"greeting": "yo"}
    assert res.exit_code == 0 and "yo" in res.output


def test_the_servers_own_refusal_is_reported_verbatim(env, monkeypatch):
    """The runtime path does not re-implement validation; the plugin's field
    list lives on the server and so does its validator. A refusal reaches the
    user with the server's own reason, not a bare status."""
    _install_widget(env, enable=True)
    fields = [{"key": "greeting", "widget": "text", "label": "Greeting",
               "value": "hi", "default": "hi", "is_override": False}]
    _stub_server(monkeypatch, fields=fields,
                 on_post=lambda body: _Resp({"detail": "greeting: too long"}, 400))
    res = _run("widget", "greeting", "x" * 999)
    assert res.exit_code != 0
    assert "greeting: too long" in res.output


def test_the_cycle_check_reads_the_config_the_write_is_based_on(env, monkeypatch):
    """The cycle check runs INSIDE update_config's mutator, not against a
    load_config() read taken before the write: update_config holds a
    cross-process lock across the whole read-modify-write, and a check outside
    it is check-then-act.

    Telling the two apart needs the pre-read to DISAGREE with what the write is
    based on, which is what patching load_config does here: it returns a
    cycle-free view while the real stored config already has music -> image.
    """
    import localm.config as cfg_mod
    from localm import settings_schema as ss

    assert _run("music", "use_config_from", "image").exit_code == 0
    monkeypatch.setattr(cfg_mod, "load_config", lambda *a, **k: {"plugins": {}})
    with pytest.raises(ValueError, match="cycle"):
        ss.apply_local_plugin_config("image", "use_config_from", "music")
    # ... and the refusal aborted the write rather than leaving it half done.
    assert "use_config_from" not in _blocks(env).get("image", {})


def test_a_remote_target_is_not_judged_by_this_machines_install_set(env, monkeypatch):
    """LOCALM_URL points at an instance that is not this home. Its plugin list
    is the server's, so a plugin absent from THIS machine does not produce a
    "No such plugin" claim about one the target is running."""
    fields = [{"key": "greeting", "widget": "text", "label": "Greeting",
               "value": "hi", "default": "hi", "is_override": False}]
    calls = _stub_server(monkeypatch, fields=fields)      # sets LOCALM_URL
    res = _run("widget")                                   # never installed locally
    assert calls["get"] == 1                               # it ASKED, rather than assuming
    assert res.exit_code == 0
    assert "greeting" in res.output
    assert "No such plugin" not in res.output


def test_a_remote_that_has_no_section_says_what_it_observed(env, monkeypatch):
    """The other arm: absent from the remote's list is reported as the remote's
    answer, not as a claim about installation here."""
    import requests
    monkeypatch.setenv("LOCALM_URL", "http://127.0.0.1:1")
    calls = {"get": 0}

    def fake_get(url, **kw):
        calls["get"] += 1
        return _Resp({"plugins": []})           # the server answered, with nothing

    monkeypatch.setattr(requests, "get", fake_get)
    res = _run("widget")
    assert calls["get"] == 1                    # it asked, and the stub intercepted
    assert res.exit_code != 0
    assert "reports no settings" in res.output
    assert "No such plugin" not in res.output
    assert "not enabled" not in res.output
