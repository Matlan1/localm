# SPDX-License-Identifier: AGPL-3.0-or-later
"""A plugin can declare `requires = [...]` other plugins. api_state carries
`missing_requires` per entry (the declared requirements not currently installed)
alongside the raw `requires` list, so the GUI can warn "requires X (missing)" and
offer a one-click "Install requirements".
"""

import textwrap

import pytest
from fastapi import FastAPI


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


def _ping(name):
    return textwrap.dedent(f'''
        from fastapi import APIRouter
        _r = APIRouter()

        @_r.get("/api/{name}/ping")
        def ping():
            return {{"pong": True}}

        def register(host):
            host.mount_router(_r)

        def unregister():
            pass
    ''')


def _make_plugin(root, name, *, requires=None):
    pdir = root / name
    pdir.mkdir(parents=True, exist_ok=True)
    req = f"requires = {list(requires)!r}\n" if requires else ""
    (pdir / "plugin.toml").write_text(
        f'[plugin]\nname = "{name}"\nscope = "{name}"\nregister = "plug"\n{req}',
        encoding="utf-8")
    (pdir / "plug.py").write_text(_ping(name), encoding="utf-8")
    return pdir


def _state(mgr):
    return {p["name"]: p for p in mgr.api_state()["plugins"]}


def test_missing_requires_lists_uninstalled_deps(env):
    from localm.plugins.engine import PluginManager
    store, installed = env / "store", env / "installed"
    _make_plugin(store, "alpha")                       # available, NOT installed
    _make_plugin(installed, "beta", requires=["alpha"])  # installed, needs alpha
    mgr = PluginManager(FastAPI(), store_root=store, installed_root=installed)

    by_name = _state(mgr)
    assert by_name["beta"]["requires"] == ["alpha"]
    assert by_name["beta"]["missing_requires"] == ["alpha"]
    # alpha itself requires nothing.
    assert by_name["alpha"]["missing_requires"] == []


def test_missing_requires_empty_once_dep_installed(env):
    from localm.plugins.engine import PluginManager
    store, installed = env / "store", env / "installed"
    _make_plugin(store, "alpha")
    _make_plugin(installed, "beta", requires=["alpha"])
    mgr = PluginManager(FastAPI(), store_root=store, installed_root=installed)

    assert _state(mgr)["beta"]["missing_requires"] == ["alpha"]
    mgr.install("alpha")                                # provision alpha
    assert _state(mgr)["beta"]["missing_requires"] == []


def test_no_requires_means_no_missing(env):
    from localm.plugins.engine import PluginManager
    plugins = env / "plugins"
    _make_plugin(plugins, "solo")
    mgr = PluginManager(FastAPI(), external_root=plugins, builtin_root=None)
    assert _state(mgr)["solo"]["missing_requires"] == []


def test_preinstalled_dep_self_heals_before_headless_cli_check(env, monkeypatch):
    """A data dir where the server/GUI has never started - only headless CLI
    commands like `localm plugin install jobs` have run - must still see a
    preinstalled/protected dependency (chat) as installed. `_ensure_preinstalled`
    (the self-heal that provisions chat onto disk) must therefore also run from a
    bare `discover()` call, not only from `load_enabled()` (the server-start
    path), or `missing_requires` falsely reports chat as missing on a fresh home
    even though chat is protected + default_enabled and always present."""
    from localm.plugins import catalog as _cat
    from localm.plugins.engine import PluginManager

    store, installed = env / "store", env / "installed"
    _make_plugin(store, "chat")                          # available in the store...
    _make_plugin(installed, "jobs", requires=["chat"])    # ...but NOT yet installed
    assert not (installed / "chat").exists()

    monkeypatch.setattr(_cat, "preinstalled", lambda: ("chat",))
    mgr = PluginManager(FastAPI(), store_root=store, installed_root=installed)

    by_name = _state(mgr)                                 # triggers discover()
    assert by_name["jobs"]["missing_requires"] == []
    assert (installed / "chat" / "plugin.toml").is_file()
