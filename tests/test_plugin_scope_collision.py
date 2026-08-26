# SPDX-License-Identifier: AGPL-3.0-or-later
"""Installing a THIRD-PARTY plugin whose (possibly default) scope collides with a
kernel capability, a first-party plugin's scope, a privileged scope, or another
already-installed plugin's scope must be rejected.

``parse_spec`` copies a manifest's ``scope`` verbatim, defaulting to the
plugin's own NAME when omitted (PluginSpec.__post_init__), and ``mount_router``
gates every route the plugin registers on that string. Without a collision
check, a manifest reusing "chat" - or naming itself "rag"/"web"/"voice", which
defaults its scope to the same string as the shipped first-party plugin -
widens what every pre-existing key holding that scope can reach.
"""

import pytest
from fastapi import FastAPI


def _write_plugin(root, name, *, scope=None):
    pdir = root / name
    pdir.mkdir(parents=True, exist_ok=True)
    scope_line = f'scope = "{scope}"\n' if scope is not None else ""
    (pdir / "plugin.toml").write_text(
        f'[plugin]\nname = "{name}"\n{scope_line}register = "plug"\n',
        encoding="utf-8")
    (pdir / "plug.py").write_text(
        "def register(host):\n    pass\n\n\ndef unregister():\n    pass\n",
        encoding="utf-8")
    return pdir


def _manager(tmp_path):
    from localm.plugins.engine import PluginManager
    installed = tmp_path / "installed"
    installed.mkdir(parents=True, exist_ok=True)
    return PluginManager(FastAPI(), external_root=installed, builtin_root=None)


def test_rejects_explicit_scope_matching_builtin_plugin(tmp_path):
    # NEGATIVE: this would install "sneaky" with scope="chat", so every
    # chat-only-scoped key would reach its routes.
    src = _write_plugin(tmp_path / "src", "sneaky", scope="chat")
    mgr = _manager(tmp_path)
    with pytest.raises(ValueError, match="reserved localm scope"):
        mgr.set_installed_from_dir(src)
    assert "sneaky" not in mgr.discover()


def test_rejects_default_scope_matching_builtin_plugin_name(tmp_path):
    # A plugin literally named "rag" (no explicit scope line) defaults its scope
    # to "rag" too.
    src = _write_plugin(tmp_path / "src", "rag")
    mgr = _manager(tmp_path)
    with pytest.raises(ValueError, match="reserved localm scope"):
        mgr.set_installed_from_dir(src)
    assert "rag" not in mgr.discover()


@pytest.mark.parametrize("scope", ["admin", "config:write", "keys:admin", "coder:full"])
def test_rejects_kernel_and_privileged_scopes(tmp_path, scope):
    src = _write_plugin(tmp_path / "src", "widget", scope=scope)
    mgr = _manager(tmp_path)
    with pytest.raises(ValueError, match="reserved localm scope"):
        mgr.set_installed_from_dir(src)
    assert "widget" not in mgr.discover()


def test_rejects_scope_matching_another_installed_plugin(tmp_path):
    # Two DIFFERENT plugin names whose scopes collide (one declares the
    # other's scope explicitly) - not covered by the reserved-name checks at
    # all, since neither name is reserved or a first-party plugin.
    mgr = _manager(tmp_path)
    first = _write_plugin(tmp_path / "src1", "alpha")
    mgr.set_installed_from_dir(first)

    second = _write_plugin(tmp_path / "src2", "beta", scope="alpha")
    with pytest.raises(ValueError, match="already used by installed plugin"):
        mgr.set_installed_from_dir(second)
    assert "beta" not in mgr.discover()


def test_force_reinstall_of_the_same_plugin_is_not_a_self_collision(tmp_path):
    mgr = _manager(tmp_path)
    src = _write_plugin(tmp_path / "src", "gamma")
    mgr.set_installed_from_dir(src)
    # Re-installing the identical plugin (same name, same scope) with --force
    # must not trip the "another installed plugin" collision against itself.
    mgr.set_installed_from_dir(src, force=True)
    assert "gamma" in mgr.discover()


def test_normal_third_party_scope_installs(tmp_path):
    src = _write_plugin(tmp_path / "src", "delta")
    mgr = _manager(tmp_path)
    spec = mgr.set_installed_from_dir(src)
    assert spec.scope == "delta"
    assert "delta" in mgr.discover()


def test_install_external_also_rejects_scope_collision(tmp_path):
    # install_external (the app-mounting sibling of set_installed_from_dir)
    # shares _copy_third_party_source, so it must be gated too.
    src = _write_plugin(tmp_path / "src", "sneaky2", scope="chat")
    mgr = _manager(tmp_path)
    with pytest.raises(ValueError, match="reserved localm scope"):
        mgr.install_external(src)
    assert "sneaky2" not in mgr.discover()
