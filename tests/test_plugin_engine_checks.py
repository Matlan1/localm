# SPDX-License-Identifier: AGPL-3.0-or-later
"""Plugin engine checks that must hold for third-party plugins: the install
route's path and access gate, flags a third-party manifest cannot set, scope
checks at discovery, the route-path rule, stop callbacks, the folder picker
gate, and the trust notice at every install site."""

import json
import textwrap
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent


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


def _write(root, name, *, toml_extra="", body=None, scope=None):
    pdir = root / name
    pdir.mkdir(parents=True, exist_ok=True)
    scope = name if scope is None else scope
    (pdir / "plugin.toml").write_text(
        f'[plugin]\nname = "{name}"\nscope = "{scope}"\nregister = "plug"\n{toml_extra}',
        encoding="utf-8")
    (pdir / "plug.py").write_text(textwrap.dedent(body or '''
        def register(host):
            pass

        def unregister():
            pass
    '''), encoding="utf-8")
    return pdir


def _manager(env, app=None):
    from localm.plugins.engine import PluginManager
    return PluginManager(app, store_root=env / "store", installed_root=env / "installed")


# ---- install-external: path screen and access gate ----------------------------

def _client(env, monkeypatch, *, fs, bind_host):
    from localm.plugins import engine
    app = FastAPI()
    app.state.bind_host = bind_host
    monkeypatch.setattr("localm.inference.http_server.effective_fs_access",
                        lambda request: fs)
    engine.attach_engine(app)
    return TestClient(app)


def test_install_external_refuses_a_unc_path_before_any_filesystem_call(env, monkeypatch):
    calls = []
    real = Path.is_dir

    def spy(self, *a, **k):
        calls.append(str(self))
        return real(self, *a, **k)

    monkeypatch.setattr(Path, "is_dir", spy)
    with _client(env, monkeypatch, fs="host", bind_host="127.0.0.1") as c:
        r = c.post("/api/plugins/install-external",
                   json={"source": "\\\\attacker\\share\\p"})
    assert r.status_code == 400
    assert "UNC" in r.json()["detail"]
    assert not [x for x in calls if "attacker" in x]


def test_install_external_needs_host_access_or_a_loopback_bind(env, monkeypatch):
    src = _write(env / "src", "gated")
    with _client(env, monkeypatch, fs="none", bind_host="0.0.0.0") as c:
        r = c.post("/api/plugins/install-external", json={"source": str(src)})
    assert r.status_code == 403
    assert not (env / "plugins" / "gated").exists()


def test_install_external_allowed_for_a_host_fs_key_on_a_network_bind(env, monkeypatch):
    src = _write(env / "src", "hostkey")
    with _client(env, monkeypatch, fs="host", bind_host="0.0.0.0") as c:
        r = c.post("/api/plugins/install-external", json={"source": str(src)})
    assert r.status_code == 200, r.text


def test_install_external_allowed_for_a_restricted_key_on_a_loopback_bind(env, monkeypatch):
    src = _write(env / "src", "loopkey")
    with _client(env, monkeypatch, fs="none", bind_host="127.0.0.1") as c:
        r = c.post("/api/plugins/install-external", json={"source": str(src)})
    assert r.status_code == 200, r.text


# ---- flags a third-party manifest cannot set -----------------------------------

def test_third_party_manifest_cannot_mark_itself_protected_or_default_enabled(env):
    from localm.plugins.engine import parse_spec
    d = _write(env / "src", "greedy", toml_extra="protected = true\ndefault_enabled = true\n")
    spec = parse_spec(d)
    assert spec.protected is False and spec.default_enabled is False


def test_a_third_party_plugin_that_asked_for_protection_can_still_be_disabled_and_removed(env):
    src = _write(env / "src", "sticky", toml_extra="protected = true\n")
    mgr = _manager(env)
    mgr.set_installed_from_dir(src)
    assert mgr._is_protected("sticky") is False
    mgr.disable("sticky")
    assert mgr.uninstall("sticky") is True


def test_a_first_party_manifest_keeps_its_flags():
    from localm.plugins.engine import parse_spec
    spec = parse_spec(ROOT / "localm" / "plugins" / "builtin" / "chat")
    assert spec.protected is True and spec.default_enabled is True


# ---- scope checks at discovery --------------------------------------------------

def test_discover_drops_a_hand_placed_plugin_with_a_reserved_scope(env):
    _write(env / "installed", "handmade", scope="chat")
    mgr = _manager(env)
    assert "handmade" not in mgr.discover()
    assert "reserved localm scope" in mgr._discover_errors["handmade"]


def test_discover_drops_the_second_plugin_claiming_a_taken_scope(env):
    _write(env / "installed", "aaa", scope="shared")
    _write(env / "installed", "bbb", scope="shared")
    mgr = _manager(env)
    specs = mgr.discover()
    assert "aaa" in specs and "bbb" not in specs
    assert "already used by installed plugin 'aaa'" in mgr._discover_errors["bbb"]


def test_a_dropped_plugin_is_not_loaded_when_enabled(env):
    _write(env / "installed", "handmade2", scope="rag",
           body='''
        import pathlib
        pathlib.Path(__file__).with_name("RAN").write_text("x")

        def register(host):
            pass
    ''')
    mgr = _manager(env)
    mgr._set_enabled("handmade2", True)
    mgr.load_enabled()
    assert "handmade2" not in mgr._loaded
    assert not (env / "installed" / "handmade2" / "RAN").exists()


# ---- mount_router path rule ------------------------------------------------------

def _router_plugin(env, name, path):
    return _write(env / "installed", name, body=f'''
        from fastapi import APIRouter
        _r = APIRouter()

        @_r.get("{path}")
        def h():
            return {{"ok": True}}

        def register(host):
            host.mount_router(_r)

        def unregister():
            pass
    ''')


@pytest.mark.parametrize("path", ["/myplugin/data", "/", "/apix/x", "/plugins/x/data"])
def test_mount_router_refuses_a_route_outside_api_and_v1(env, path):
    _router_plugin(env, "outside", path)
    app = FastAPI()
    mgr = _manager(env, app)
    mgr._set_enabled("outside", True)
    mgr.load_enabled()
    assert "outside" not in mgr._loaded
    assert "outside" in mgr._errors and "/api/ and /v1/" in mgr._errors["outside"]
    with TestClient(app) as c:
        assert c.get(path).status_code == 404


@pytest.mark.parametrize("path", ["/api/inside/x", "/v1/inside/x"])
def test_mount_router_accepts_api_and_v1_routes(env, path):
    _router_plugin(env, "inside", path)
    app = FastAPI()
    mgr = _manager(env, app)
    mgr._set_enabled("inside", True)
    mgr.load_enabled()
    assert "inside" in mgr._loaded
    with TestClient(app) as c:
        assert c.get(path).json() == {"ok": True}


# ---- Host.track -------------------------------------------------------------------

def test_unload_runs_every_tracked_stop_callable_and_survives_a_raising_one(env):
    _write(env / "installed", "stopper", body='''
        import pathlib
        _here = pathlib.Path(__file__).parent

        def _first():
            (_here / "first").write_text("x")

        def _boom():
            raise RuntimeError("stop failed")

        def _last():
            (_here / "last").write_text("x")

        def register(host):
            host.track(_first)
            host.track(_boom)
            host.track(_last)

        def unregister():
            pass
    ''')
    mgr = _manager(env, FastAPI())
    mgr._set_enabled("stopper", True)
    mgr.load_enabled()
    assert "stopper" in mgr._loaded
    d = env / "installed" / "stopper"
    assert not (d / "first").exists()
    mgr.disable("stopper")
    assert (d / "first").exists() and (d / "last").exists()


def test_a_failed_load_runs_the_stop_callables_it_registered(env):
    _write(env / "installed", "failing", body='''
        import pathlib
        _here = pathlib.Path(__file__).parent

        def _stop():
            (_here / "stopped").write_text("x")

        def register(host):
            host.track(_stop)
            raise RuntimeError("register failed")
    ''')
    mgr = _manager(env, FastAPI())
    mgr._set_enabled("failing", True)
    mgr.load_enabled()
    assert (env / "installed" / "failing" / "stopped").exists()


def test_track_rejects_a_non_callable(env):
    from localm.plugins.engine import PluginHost, parse_spec
    spec = parse_spec(_write(env / "src", "tt"))
    host = PluginHost(None, _manager(env), spec)
    with pytest.raises(ValueError):
        host.track("not callable")


def test_memory_plugin_stops_scheduling_background_passes_on_unload(monkeypatch):
    from localm.plugins.builtin.memory import plug as mem

    class FakeHost:
        def __init__(self):
            self.stoppers = []

        def track(self, fn):
            self.stoppers.append(fn)

        def mount_router(self, router):
            pass

        def register_chat_hook(self, *a, **k):
            pass

    host = FakeHost()
    mem.register(host)
    try:
        assert not mem._stop.is_set()
        assert mem._request_stop in host.stoppers
        host.stoppers[0]()
        assert mem._stop.is_set()
        started = []
        monkeypatch.setattr(mem._threading, "Thread",
                            lambda *a, **k: started.append(1))
        monkeypatch.setattr(mem, "_live_engine",
                            lambda: type("E", (), {"loaded": True})())
        mem._maybe_auto_consolidate("p")
        assert started == []
    finally:
        mem._stop.clear()


# ---- Host.browse_dirs -------------------------------------------------------------

def test_browse_dirs_needs_host_filesystem_access(env, monkeypatch, tmp_path):
    from localm.plugins.engine import PluginHost, parse_spec
    (tmp_path / "sub").mkdir()
    spec = parse_spec(_write(env / "src", "bd"))
    host = PluginHost(None, _manager(env), spec)
    monkeypatch.setattr("localm.inference.http_server.effective_fs_access",
                        lambda request: "none")
    with pytest.raises(PermissionError):
        host.browse_dirs(str(tmp_path), object())
    monkeypatch.setattr("localm.inference.http_server.effective_fs_access",
                        lambda request: "shared")
    with pytest.raises(PermissionError):
        host.browse_dirs(str(tmp_path), object())
    monkeypatch.setattr("localm.inference.http_server.effective_fs_access",
                        lambda request: "host")
    assert "sub" in host.browse_dirs(str(tmp_path), object())["dirs"]


# ---- one loader ---------------------------------------------------------------------

def test_the_second_manifest_parser_is_gone():
    from localm.plugins import loader
    for gone in ("parse_manifest", "discover_plugins", "import_plugin_module",
                 "PluginManifest"):
        assert not hasattr(loader, gone)
    assert loader.plugins_dir and loader._RESERVED_NAMES


# ---- trust notice at every install site ------------------------------------------------

def _flat(text):
    return " ".join(text.split())


def test_the_trust_sentence_is_at_every_install_site():
    from localm.plugins.contract import UNSANDBOXED_NOTICE
    assert UNSANDBOXED_NOTICE == (
        "Third-party plugins run unsandboxed in-process, so install only code you trust.")
    assert "run unsandboxed in-process, so install only code you trust" in _flat(
        _install_route_doc())
    for rel in ("SECURITY.md", "docs/plugins.md", "docs/architecture.md"):
        text = _flat((ROOT / rel).read_text(encoding="utf-8"))
        assert "unsandboxed in-process, so install only code you trust" in text, rel
    cli = _flat((ROOT / "localm" / "cli" / "plugins.py").read_text(encoding="utf-8"))
    assert "UNSANDBOXED_NOTICE" in cli and "click.confirm" in cli


def _install_route_doc():
    app = FastAPI()
    from localm.plugins import engine
    engine.attach_engine(app)
    route = next(r for r in app.router.routes
                 if getattr(r, "path", "") == "/api/plugins/install-external")
    return route.endpoint.__doc__


def test_the_gui_plugin_page_states_the_posture_and_no_longer_names_the_home_folder():
    static = ROOT / "localm" / "plugins" / "gui" / "static"
    en = (static / "app" / "i18n-en.js").read_text(encoding="utf-8")
    start = en.index('"plugins.externalIntro"')
    entry = en[start:en.index('",', start)]
    assert "unsandboxed in-process" in entry and "~/.localm" not in entry
    html = _flat((static / "index.html").read_text(encoding="utf-8"))
    assert "unsandboxed in-process, so install only code you trust" in html
    for lang in ("de", "es", "fr"):
        data = json.loads((static / "i18n" / f"{lang}.json").read_text(encoding="utf-8"))
        assert "~/.localm" not in data["plugins.externalIntro"], lang


def test_chat_hooks_are_documented_as_transforms_not_controls():
    from localm.plugins.contract import Host
    assert "transforms, not controls" in _flat(Host.register_chat_hook.__doc__)
    doc = _flat((ROOT / "docs" / "plugins.md").read_text(encoding="utf-8"))
    assert "transforms, not controls" in doc
    assert "[redacted]" not in doc
    for path in ("localm run", "forwarded to a peer", "jobs", "MCP chat tool",
                 "background generation"):
        assert path in doc, path
