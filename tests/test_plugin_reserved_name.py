# SPDX-License-Identifier: AGPL-3.0-or-later
"""Installing a THIRD-PARTY plugin whose name shadows a built-in command must be
rejected.

The guard lives on the arbitrary-source install paths (install_external /
set_installed_from_dir), not on parse_spec itself: built-in plugins legitimately
carry names like "run"/"serve"/"coder" and install via the trusted store path
(install()), and they are parsed through parse_spec too. Both the legacy CLI
loader (loader.parse_manifest) and the engine loader - the one the live server
uses - must reject a reserved name, or a third-party plugin installed from an
arbitrary directory shadows a catalog/core command.
"""

import pytest
from fastapi import FastAPI


def _write_plugin(root, name):
    pdir = root / name
    pdir.mkdir(parents=True, exist_ok=True)
    (pdir / "plugin.toml").write_text(
        f'[plugin]\nname = "{name}"\nscope = "{name}"\nregister = "plug"\n',
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


def test_set_installed_from_dir_rejects_reserved_name(tmp_path):
    src = _write_plugin(tmp_path / "src", "run")
    mgr = _manager(tmp_path)
    with pytest.raises(ValueError, match="clashes with a built-in"):
        mgr.set_installed_from_dir(src)
    assert "run" not in mgr.discover()


def test_install_external_rejects_reserved_name(tmp_path):
    src = _write_plugin(tmp_path / "src", "coder")
    mgr = _manager(tmp_path)
    with pytest.raises(ValueError, match="clashes with a built-in"):
        mgr.install_external(src)
    assert "coder" not in mgr.discover()


def test_normal_third_party_name_installs(tmp_path):
    src = _write_plugin(tmp_path / "src", "alpha")
    mgr = _manager(tmp_path)
    mgr.set_installed_from_dir(src)
    assert "alpha" in mgr.discover()


def test_parse_spec_does_not_block_reserved_names(tmp_path):
    # Built-in plugins (coder, mcp, ...) are parsed via parse_spec too, so the
    # guard must NOT live there - only on the third-party install paths.
    from localm.plugins.engine import parse_spec
    pdir = _write_plugin(tmp_path, "coder")
    assert parse_spec(pdir, builtin=False).name == "coder"


def test_engine_and_loader_share_reserved_set():
    from localm.plugins.loader import _RESERVED_NAMES
    assert {"run", "serve", "coder", "config", "gui"} <= _RESERVED_NAMES
