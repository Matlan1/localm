# SPDX-License-Identifier: AGPL-3.0-or-later
"""Plugins load sequentially at server startup, and the load must not be silent:
without a per-plugin record there is no trace of what loaded, in what order, or
what failed, only the final api_state errors dict. load_enabled reports each
plugin as it loads through an on_event callback (and logs a line per plugin), so
the server surfaces the sequential load and the GUI can show progress.
"""

import logging
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


def _make_plugin(root, name, body):
    pdir = root / name
    pdir.mkdir(parents=True, exist_ok=True)
    (pdir / "plugin.toml").write_text(
        f'[plugin]\nname = "{name}"\nscope = "{name}"\nregister = "plug"\n',
        encoding="utf-8")
    (pdir / "plug.py").write_text(textwrap.dedent(body), encoding="utf-8")
    return pdir


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


def _enable(names):
    from localm.config import load_config, save_config
    cfg = load_config()
    cfg["plugins_enabled"] = list(names)
    save_config(cfg)


def test_on_event_reports_each_plugin_sequentially(env):
    from localm.plugins.engine import PluginManager
    plugins = env / "plugins"
    _make_plugin(plugins, "good", _ping("good"))
    _make_plugin(plugins, "bad",
                 'def register(host):\n    raise RuntimeError("boom")\ndef unregister():\n    pass\n')
    _enable(["good", "bad"])

    events = []
    mgr = PluginManager(FastAPI(), external_root=plugins, builtin_root=None)
    mgr.load_enabled(on_event=lambda name, status, error=None: events.append((name, status)))

    # Loaded in sorted order; each reported exactly once with its outcome.
    assert events == [("bad", "failed"), ("good", "loaded")]


def test_on_event_carries_the_error_on_failure(env):
    from localm.plugins.engine import PluginManager
    plugins = env / "plugins"
    _make_plugin(plugins, "bad",
                 'def register(host):\n    raise RuntimeError("kaboom")\ndef unregister():\n    pass\n')
    _enable(["bad"])

    seen = {}
    mgr = PluginManager(FastAPI(), external_root=plugins, builtin_root=None)
    mgr.load_enabled(on_event=lambda name, status, error=None: seen.update({name: (status, error)}))
    status, error = seen["bad"]
    assert status == "failed"
    assert error and "kaboom" in error


def test_load_enabled_logs_each_plugin(env, caplog):
    from localm.plugins.engine import PluginManager
    plugins = env / "plugins"
    _make_plugin(plugins, "good", _ping("good"))
    _enable(["good"])
    mgr = PluginManager(FastAPI(), external_root=plugins, builtin_root=None)
    with caplog.at_level(logging.INFO, logger="localm.plugins"):
        mgr.load_enabled()
    assert any("good" in r.getMessage() for r in caplog.records), \
        "no per-plugin load line was logged"


def test_on_event_optional_and_back_compat(env):
    # No callback -> still loads fine (the old call site keeps working).
    from localm.plugins.engine import PluginManager
    plugins = env / "plugins"
    _make_plugin(plugins, "good", _ping("good"))
    _enable(["good"])
    mgr = PluginManager(FastAPI(), external_root=plugins, builtin_root=None)
    mgr.load_enabled()                       # no on_event
    state = {p["name"]: p for p in mgr.api_state()["plugins"]}
    assert state["good"]["loaded"] is True
