# SPDX-License-Identifier: AGPL-3.0-or-later
"""PluginHost.mount_static must confine the served directory to the plugin dir.

A plugin's ``assets_dir`` (manifest) or an explicit ``mount_static`` call is
untrusted for a third-party plugin: a value like ``"../secret"`` or an absolute
path would otherwise serve any directory on disk as public static assets.
"""

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


def _make_plugin_dir(root, name="victim"):
    pdir = root / name
    (pdir / "static").mkdir(parents=True, exist_ok=True)
    (pdir / "static" / "ok.js").write_text("export const x = 1;\n", encoding="utf-8")
    (pdir / "plugin.toml").write_text(
        f'[plugin]\nname = "{name}"\nregister = "plug"\n', encoding="utf-8")
    return pdir


def _host_for(env, pdir):
    from localm.plugins.engine import PluginHost, PluginManager, parse_spec
    spec = parse_spec(pdir)
    return PluginHost(
        FastAPI(),
        PluginManager(FastAPI(), external_root=env / "x", builtin_root=None),
        spec,
    )


def test_mount_static_relative_traversal_escape_raises(env, tmp_path):
    """NEGATIVE: ``../secret`` escapes the plugin dir and is refused with
    ValueError."""
    plugins = env / "plugins"
    pdir = _make_plugin_dir(plugins)
    secret = plugins / "secret"
    secret.mkdir(parents=True, exist_ok=True)
    (secret / "leak.txt").write_text("TOP SECRET", encoding="utf-8")
    host = _host_for(env, pdir)
    with pytest.raises(ValueError):
        host.mount_static("../secret")


def test_mount_static_absolute_path_escape_raises(env, tmp_path):
    """NEGATIVE: an absolute directory outside the plugin dir must be refused."""
    plugins = env / "plugins"
    pdir = _make_plugin_dir(plugins)
    outside = tmp_path / "outside_secret"
    outside.mkdir(parents=True, exist_ok=True)
    (outside / "leak.txt").write_text("TOP SECRET", encoding="utf-8")
    host = _host_for(env, pdir)
    with pytest.raises(ValueError):
        host.mount_static(str(outside))


def test_mount_static_in_dir_assets_still_served(env):
    """REGRESSION: a normal in-dir assets dir mounts fine and returns its prefix."""
    plugins = env / "plugins"
    pdir = _make_plugin_dir(plugins)
    host = _host_for(env, pdir)
    prefix = host.mount_static("static")
    assert prefix == "/plugins/victim"


def test_mount_static_plugin_root_itself_allowed(env):
    """EDGE: ``.`` resolves to the plugin dir itself (d == base) and is allowed."""
    plugins = env / "plugins"
    pdir = _make_plugin_dir(plugins)
    host = _host_for(env, pdir)
    # Must not raise: the plugin's own dir is within (equal to) the plugin dir.
    prefix = host.mount_static(".")
    assert prefix == "/plugins/victim"
