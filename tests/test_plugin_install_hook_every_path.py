# SPDX-License-Identifier: AGPL-3.0-or-later
"""A plugin's on_install hook runs on every install path, through one engine
method (``PluginManager._run_install_hook``): the live install route, the
app-free installs (``localm plugin install`` and ``plugin setup``, the MCP
install_plugin tool) and the install-from-directory paths.

The bundled voice plugin's hook prefetches the Whisper model: inline when no
event loop runs (the process may exit as soon as the hook returns), on a thread
on the server's loop, and never while network access is off. The prefetch
itself is stubbed in every test here: no network in a unit test.
"""

import asyncio
import logging
import sys
import textwrap
import threading

import pytest
from click.testing import CliRunner
from fastapi import FastAPI


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path))
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.delenv("LOCALM_REQUIRE_AUTH", raising=False)
    monkeypatch.delenv("LOCALM_NET_MODE", raising=False)
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", tmp_path)
    monkeypatch.setattr(cfg, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", tmp_path / "registry.json")
    return tmp_path


@pytest.fixture
def prefetch(monkeypatch):
    """Stub the Whisper prefetch; records (allow_download, name) per call.
    ``result`` is what the stub returns."""
    import localm.voice as voice
    calls = []
    state = {"result": (True, "")}

    def fake(allow_download=None, name=None):
        calls.append((allow_download, name))
        return state["result"]
    monkeypatch.setattr(voice, "prefetch_stt_model", fake)
    yield calls, state
    for t in threading.enumerate():
        if t.name == voice.PREFETCH_THREAD_NAME:
            t.join(timeout=10)


def _set(key, value):
    from localm.config import update_config
    update_config(lambda c: c.__setitem__(key, value))


_HOOKED = '''
    from pathlib import Path

    def register(host):
        (Path(__file__).parent / "register_called").write_text("yes")

    def unregister():
        pass

    def on_install():
        (Path(__file__).parent / "on_install_called").write_text("yes")
        return "  hooked note  "
'''


def _make_plugin(root, name, body, *, toml_extra=""):
    pdir = root / name
    pdir.mkdir(parents=True, exist_ok=True)
    (pdir / "plugin.toml").write_text(
        f'[plugin]\nname = "{name}"\nscope = "{name}"\nregister = "plug"\n{toml_extra}',
        encoding="utf-8")
    (pdir / "plug.py").write_text(textwrap.dedent(body), encoding="utf-8")
    return pdir


# --------------------------------------------------------------------------- #
#  Engine: the app-free installs run the hook, without register()             #
# --------------------------------------------------------------------------- #

def test_app_free_store_install_runs_on_install_without_register(env):
    from localm.plugins.engine import PluginManager
    store, inst = env / "store", env / "installed"
    _make_plugin(store, "hooked", _HOOKED)
    mgr = PluginManager(None, store_root=store, installed_root=inst)
    mgr.set_installed_state("hooked", True)
    assert (inst / "hooked" / "on_install_called").exists()
    assert not (inst / "hooked" / "register_called").exists()
    assert mgr.pop_install_note("hooked") == "hooked note"
    assert mgr.pop_install_note("hooked") is None          # returned once
    # The module imported for the hook alone is not left behind.
    assert not [m for m in sys.modules if m.startswith("_localm_plugin_hooked")]


def test_app_free_install_runs_the_hook_with_enable_false(env):
    from localm.plugins.engine import PluginManager
    store, inst = env / "store", env / "installed"
    _make_plugin(store, "hooked", _HOOKED)
    mgr = PluginManager(None, store_root=store, installed_root=inst)
    mgr.set_installed_state("hooked", True, enable=False)
    assert not mgr.is_enabled("hooked")
    assert (inst / "hooked" / "on_install_called").exists()


def test_app_free_install_skips_the_hook_while_a_required_plugin_is_missing(env):
    from localm.plugins.engine import PluginManager
    store, inst = env / "store", env / "installed"
    _make_plugin(store, "basedep", "def register(host):\n    pass\n")
    _make_plugin(store, "hooked", _HOOKED, toml_extra='requires = ["basedep"]\n')
    mgr = PluginManager(None, store_root=store, installed_root=inst)
    mgr.set_installed_state("hooked", True)
    assert mgr.is_installed("hooked") and not mgr.is_enabled("hooked")
    assert not (inst / "hooked" / "on_install_called").exists()
    assert mgr.pop_install_note("hooked") is None


def test_app_free_directory_install_runs_on_install(env):
    from localm.plugins.engine import PluginManager
    src = _make_plugin(env / "src", "hooked", _HOOKED)
    inst = env / "installed"
    mgr = PluginManager(None, external_root=env / "noext", installed_root=inst,
                        builtin_root=None)
    spec = mgr.set_installed_from_dir(src)
    assert spec.name == "hooked"
    assert (inst / "hooked" / "on_install_called").exists()
    assert not (src / "on_install_called").exists()          # ran in the copy
    assert mgr.pop_install_note("hooked") == "hooked note"


def test_a_plugin_that_cannot_be_imported_still_installs_and_warns(env, caplog):
    from localm.plugins.engine import PluginManager
    store, inst = env / "store", env / "installed"
    _make_plugin(store, "broken", "raise RuntimeError('import exploded')\n")
    mgr = PluginManager(None, store_root=store, installed_root=inst)
    with caplog.at_level(logging.WARNING, logger="localm.plugins"):
        mgr.set_installed_state("broken", True)
    assert mgr.is_installed("broken") and mgr.is_enabled("broken")
    warned = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("broken" in m and "on_install" in m and "import exploded" in m
               for m in warned), warned


def test_a_failing_hook_on_an_app_free_install_warns_and_the_install_stands(env, caplog):
    from localm.plugins.engine import PluginManager
    store, inst = env / "store", env / "installed"
    _make_plugin(store, "boom", '''
        def register(host):
            pass

        def on_install():
            raise RuntimeError("hook exploded")
    ''')
    mgr = PluginManager(None, store_root=store, installed_root=inst)
    with caplog.at_level(logging.WARNING, logger="localm.plugins"):
        mgr.set_installed_state("boom", True)
    assert mgr.is_installed("boom") and mgr.is_enabled("boom")
    assert mgr.pop_install_note("boom") is None
    warned = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("boom" in m and "on_install" in m and "hook exploded" in m
               for m in warned), warned


# --------------------------------------------------------------------------- #
#  The voice hook                                                             #
# --------------------------------------------------------------------------- #

def _voice_on_install():
    from localm.plugins.builtin.voice import plug
    return plug.on_install()


def test_voice_hook_without_a_loop_downloads_inline(env, prefetch):
    calls, _ = prefetch
    note = _voice_on_install()
    assert calls == [(True, "base")]          # one explicit download of the named model
    assert "Downloaded the Whisper 'base' speech model" in note
    import localm.voice as voice
    assert not [t for t in threading.enumerate() if t.name == voice.PREFETCH_THREAD_NAME]


def test_voice_hook_reports_a_failed_inline_download(env, prefetch):
    calls, state = prefetch
    state["result"] = (False, "Whisper model 'base' download failed (boom)")
    assert _voice_on_install() == "Whisper model 'base' download failed (boom)"


def test_voice_hook_downloads_nothing_while_network_access_is_off(env, prefetch):
    calls, _ = prefetch
    _set("net_mode", "off")
    note = _voice_on_install()
    assert calls == []
    assert "Network access is off" in note and "'base'" in note


def test_voice_hook_under_off_downloads_when_model_downloads_are_allowed(env, prefetch):
    calls, _ = prefetch
    _set("net_mode", "off")
    _set("net_allow_model_downloads", True)
    _voice_on_install()
    assert calls == [(True, "base")]


def test_voice_hook_skips_a_cached_model(env, prefetch, monkeypatch):
    calls, _ = prefetch
    import localm.voice as voice
    monkeypatch.setattr(voice, "stt_model_cached", lambda: (True, "small"))
    assert "already downloaded" in _voice_on_install()
    assert calls == []


def test_voice_hook_on_an_event_loop_downloads_on_a_thread(env, prefetch, monkeypatch):
    calls, _ = prefetch
    import localm.voice as voice
    gate = threading.Event()
    seen = []

    def slow(allow_download=None, name=None):
        gate.wait(10)
        calls.append((allow_download, name))
        return True, ""
    monkeypatch.setattr(voice, "prefetch_stt_model", slow)

    async def on_loop():
        note = _voice_on_install()
        seen.extend(t for t in threading.enumerate()
                    if t.name == voice.PREFETCH_THREAD_NAME)
        return note
    note = asyncio.run(on_loop())
    assert "in the background" in note
    assert seen, "the download did not start on its own thread"
    assert calls == []                         # the hook returned before it finished
    gate.set()
    for t in seen:
        t.join(timeout=10)
    assert calls == [(True, "base")]


# --------------------------------------------------------------------------- #
#  Every entry point for the real voice plugin                                #
# --------------------------------------------------------------------------- #

def test_cli_plugin_install_voice_prefetches_and_says_so(env, prefetch):
    from localm.cli import main
    calls, _ = prefetch
    r = CliRunner().invoke(main, ["plugin", "install", "voice", "--no-deps"])
    assert r.exit_code == 0, r.output
    assert calls == [(True, "base")]
    assert "Downloaded the Whisper 'base' speech model" in r.output


def test_cli_plugin_install_voice_with_network_off_downloads_nothing(env, prefetch):
    from localm.cli import main
    calls, _ = prefetch
    _set("net_mode", "off")
    r = CliRunner().invoke(main, ["plugin", "install", "voice", "--no-deps"])
    assert r.exit_code == 0, r.output
    assert calls == []
    assert "Network access is off" in " ".join(r.output.split())


def test_plugin_setup_runs_the_voice_hook(env, prefetch):
    from localm.cli import main
    calls, _ = prefetch
    r = CliRunner().invoke(main, ["plugin", "setup", "--no-deps", "--plugins", "voice"])
    assert r.exit_code == 0, r.output
    assert calls == [(True, "base")]
    assert "Downloaded the Whisper 'base' speech model" in r.output


def test_mcp_install_plugin_runs_the_voice_hook_and_reports_its_note(env, prefetch):
    from localm.plugins.mcpserver.tools.plugin_admin import build
    calls, _ = prefetch
    res = build()["install_plugin"]["handler"]({"plugin": "voice", "with_deps": False})
    assert res.get("isError") is not True, res
    text = res["content"][0]["text"]
    assert calls == [(True, "base")]
    assert "successfully installed" in text
    assert "Downloaded the Whisper 'base' speech model" in text


def test_live_install_route_reports_the_note(env, prefetch):
    from fastapi.testclient import TestClient

    from localm.plugins.engine import attach_engine
    calls, _ = prefetch
    _set("net_mode", "off")
    app = FastAPI()
    attach_engine(app)
    with TestClient(app) as c:
        r = c.post("/api/plugins/voice/install")
    assert r.status_code == 200, r.text
    assert calls == []
    assert "Network access is off" in r.json()["note"]
