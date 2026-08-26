# SPDX-License-Identifier: AGPL-3.0-or-later
"""The client-asset hook (Surface.client_entry + Host.mount_static) and the
real `tts` plugin: static serving, /api/tts/config resolution, and the
no-server-traces property.

Open-mode (no API key) so the mounted, auto-scoped routes are reachable.
"""

import textwrap

import pytest
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
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


def _make_client_plugin(root, name):
    """A synthetic plugin that ships a static client module and serves it."""
    pdir = root / name
    (pdir / "static").mkdir(parents=True, exist_ok=True)
    (pdir / "plugin.toml").write_text(
        f'[plugin]\nname = "{name}"\nscope = "{name}"\nregister = "plug"\n'
        f'[surface]\nassets_dir = "static"\nclient_entry = "{name}.js"\n',
        encoding="utf-8")
    (pdir / "static" / f"{name}.js").write_text(
        "export function register(ctx) { ctx.toast('hi'); }\n", encoding="utf-8")
    (pdir / "plug.py").write_text(textwrap.dedent('''
        def register(host):
            host.mount_static("static")
        def unregister():
            pass
    '''), encoding="utf-8")
    return pdir


def _make_autoclient_plugin(root, name):
    """A client_entry plugin that declares assets_dir but whose register() does
    NOT call mount_static - the engine must auto-mount the assets so the client
    module is still served (otherwise a future plugin like this silently 404s)."""
    pdir = root / name
    (pdir / "static").mkdir(parents=True, exist_ok=True)
    (pdir / "plugin.toml").write_text(
        f'[plugin]\nname = "{name}"\nscope = "{name}"\nregister = "plug"\n'
        f'[surface]\nassets_dir = "static"\nclient_entry = "{name}.js"\n',
        encoding="utf-8")
    (pdir / "static" / f"{name}.js").write_text(
        "export function register(ctx) { ctx.toast('auto'); }\n", encoding="utf-8")
    (pdir / "plug.py").write_text(textwrap.dedent('''
        def register(host):
            pass                 # deliberately does NOT mount_static
        def unregister():
            pass
    '''), encoding="utf-8")
    return pdir


# --------------------------- the generic hook ----------------------------- #

def test_parse_spec_reads_client_entry(env):
    from localm.plugins.engine import parse_spec
    pdir = _make_client_plugin(env / "plugins", "widget")
    spec = parse_spec(pdir)
    assert spec.surface.assets_dir == "static"
    assert spec.surface.client_entry == "widget.js"


def test_api_state_exposes_client_entry_and_assets_base(env):
    from localm.plugins.engine import PluginManager
    plugins = env / "plugins"
    _make_client_plugin(plugins, "widget")
    mgr = PluginManager(FastAPI(), external_root=plugins, builtin_root=None)
    mgr.install("widget")
    st = {p["name"]: p for p in mgr.api_state()["plugins"]}
    assert st["widget"]["client_entry"] == "widget.js"
    assert st["widget"]["assets_base"] == "/plugins/widget"


def test_mount_static_serves_module_then_unmounts(env, tmp_path):
    """An active plugin's client module is served at /plugins/<name>/, coexists
    with the SPA "/" catch-all, and 404s once the plugin is disabled."""
    from localm.plugins.engine import PluginManager
    plugins = env / "plugins"
    _make_client_plugin(plugins, "widget")

    static = tmp_path / "spa"
    static.mkdir()
    (static / "index.html").write_text("INDEX", encoding="utf-8")
    app = FastAPI()
    app.mount("/", StaticFiles(directory=str(static), html=True), name="gui")

    mgr = PluginManager(app, external_root=plugins, builtin_root=None)
    mgr.install("widget")                      # mounted AFTER the catch-all "/"
    with TestClient(app) as c:
        r = c.get("/plugins/widget/widget.js")
        assert r.status_code == 200
        assert "export function register" in r.text
        assert r.headers["content-type"].startswith("text/javascript")
        assert c.get("/").text == "INDEX"      # SPA still served

    mgr.disable("widget")
    with TestClient(app) as c:
        assert c.get("/plugins/widget/widget.js").status_code == 404


def test_engine_auto_mounts_surface_assets(env, tmp_path):
    """A client_entry plugin that declares assets_dir but never calls
    mount_static is still served: the engine auto-mounts the surface's assets
    after register(), so the SPA's import() of the client module never 404s.
    The auto-mount unmounts cleanly on disable, like an explicit mount."""
    from localm.plugins.engine import PluginManager
    plugins = env / "plugins"
    _make_autoclient_plugin(plugins, "autowidget")

    static = tmp_path / "spa"
    static.mkdir()
    (static / "index.html").write_text("INDEX", encoding="utf-8")
    app = FastAPI()
    app.mount("/", StaticFiles(directory=str(static), html=True), name="gui")

    mgr = PluginManager(app, external_root=plugins, builtin_root=None)
    mgr.install("autowidget")
    with TestClient(app) as c:
        r = c.get("/plugins/autowidget/autowidget.js")
        assert r.status_code == 200
        assert "export function register" in r.text
        assert r.headers["content-type"].startswith("text/javascript")
        assert c.get("/").text == "INDEX"      # SPA catch-all still served

    mgr.disable("autowidget")
    with TestClient(app) as c:
        assert c.get("/plugins/autowidget/autowidget.js").status_code == 404


def test_explicit_mount_static_does_not_double_mount(env):
    """mount_static is idempotent on its prefix: a plugin that mounts its assets
    explicitly AND gets the engine's auto-mount ends up with exactly one mount
    for /plugins/<name> (the explicit call wins; auto-mount short-circuits)."""
    from starlette.routing import Mount

    from localm.plugins.engine import PluginManager
    plugins = env / "plugins"
    _make_client_plugin(plugins, "widget")     # register() DOES call mount_static
    app = FastAPI()
    mgr = PluginManager(app, external_root=plugins, builtin_root=None)
    mgr.install("widget")
    mounts = [r for r in app.router.routes
              if isinstance(r, Mount) and r.path == "/plugins/widget"]
    assert len(mounts) == 1                     # not two


def test_mount_static_missing_dir_raises(env):
    from localm.plugins.engine import PluginManager, parse_spec, PluginHost
    pdir = env / "plugins" / "nostatic"
    pdir.mkdir(parents=True)
    (pdir / "plugin.toml").write_text(
        '[plugin]\nname = "nostatic"\nregister = "plug"\n', encoding="utf-8")
    spec = parse_spec(pdir)
    host = PluginHost(FastAPI(), PluginManager(FastAPI(), external_root=env / "x",
                                               builtin_root=None), spec)
    with pytest.raises(ValueError):
        host.mount_static("static")


# ------------------------------ the tts plugin ---------------------------- #

def test_tts_plugin_serves_client_and_config(env):
    """The real builtin tts plugin: client module served, config = template
    defaults, and the user override under plugins.tts wins."""
    from localm.config import load_config, save_config
    from localm.plugins.engine import attach_engine
    app = FastAPI()
    attach_engine(app)                         # default roots = real builtins
    with TestClient(app) as c:
        assert c.post("/api/plugins/tts/install").status_code == 200

        js = c.get("/plugins/tts/tts.js")
        assert js.status_code == 200
        assert js.headers["content-type"].startswith("text/javascript")

        cfg = c.get("/api/tts/config").json()
        assert cfg["engine"] == "kokoro"
        assert cfg["model"].startswith("onnx-community/Kokoro")
        assert "_comment" not in cfg          # doc keys stripped
        assert cfg["voice"] == "af_heart"

        # user override wins over the shipped template default
        full = load_config()
        full.setdefault("plugins", {})["tts"] = {"voice": "am_onyx"}
        save_config(full)
        assert c.get("/api/tts/config").json()["voice"] == "am_onyx"

        # disable -> the client module is no longer served
        assert c.post("/api/plugins/tts/disable").status_code == 200
        assert c.get("/plugins/tts/tts.js").status_code == 404


def test_tts_config_reports_live_net_mode(env, monkeypatch):
    """R-NET: the Kokoro model is fetched by the BROWSER directly, so the
    client needs net_mode from /api/tts/config to gate that fetch itself -
    netpolicy.py's own server-side enforcement never sees a browser-originated
    request. The field must reflect the CURRENT policy, not a stale value."""
    from localm.plugins.engine import attach_engine
    app = FastAPI()
    attach_engine(app)
    with TestClient(app) as c:
        c.post("/api/plugins/tts/install")

        monkeypatch.setenv("LOCALM_NET_MODE", "off")
        assert c.get("/api/tts/config").json()["net_mode"] == "off"

        monkeypatch.setenv("LOCALM_NET_MODE", "ask")
        assert c.get("/api/tts/config").json()["net_mode"] == "ask"

        monkeypatch.setenv("LOCALM_NET_MODE", "allow")
        assert c.get("/api/tts/config").json()["net_mode"] == "allow"


def test_tts_plugin_writes_no_traces(env):
    """The tts plugin renders in the browser: it declares no data dir and its
    routes are read-only, so privacy mode stays trace-free with no gating."""
    from localm.plugins.engine import parse_spec
    from pathlib import Path
    import localm.plugins.engine as eng
    builtin = Path(eng.__file__).resolve().parent / "builtin" / "tts"
    spec = parse_spec(builtin, builtin=True)
    assert spec.data_subdir == ""              # nothing is persisted

    app = FastAPI()
    eng.attach_engine(app)
    with TestClient(app) as c:
        c.post("/api/plugins/tts/install")
        before = {p for p in (env / "plugins" / "tts").rglob("*")}
        c.get("/api/tts/config")
        c.get("/api/tts/status")
        after = {p for p in (env / "plugins" / "tts").rglob("*")}
        assert before == after                 # config/status requests wrote nothing
