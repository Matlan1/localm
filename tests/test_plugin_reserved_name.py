# SPDX-License-Identifier: AGPL-3.0-or-later
"""Installing a THIRD-PARTY plugin whose name shadows a built-in command must be
rejected.

The guard lives on the arbitrary-source install paths (install_external /
set_installed_from_dir), not on parse_spec itself: built-in plugins legitimately
carry names like "run"/"serve"/"coder" and install via the trusted store path
(install()), and they are parsed through parse_spec too. The engine loader must reject a reserved name, or a third-party plugin installed from an
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


# ---- first-party standing follows install provenance, never the name ---------

def _write_manifest(pdir, name, *, scope=None, extra=""):
    pdir.mkdir(parents=True, exist_ok=True)
    (pdir / "plugin.toml").write_text(
        f'[plugin]\nname = "{name}"\nscope = "{scope or name + "-x"}"\n'
        f'register = "plug"\n{extra}',
        encoding="utf-8")
    (pdir / "plug.py").write_text(
        "def register(host):\n    pass\n\n\ndef unregister():\n    pass\n",
        encoding="utf-8")
    return pdir


def _store_manager(tmp_path):
    from localm.plugins.engine import PluginManager
    installed = tmp_path / "installed"
    installed.mkdir(parents=True, exist_ok=True)
    store = tmp_path / "store"
    _write_manifest(store / "zeta", "zeta", scope="zeta")
    _write_manifest(store / "tts", "tts", scope="tts",
                    extra="protected = true\ndefault_enabled = true\n")
    return PluginManager(FastAPI(), external_root=installed, store_root=store)


def _catalog_names():
    from localm.plugins import catalog
    return list(catalog.names())


@pytest.mark.parametrize("name", _catalog_names())
@pytest.mark.parametrize("force", [False, True])
def test_third_party_install_refuses_every_catalog_name(tmp_path, name, force):
    src = _write_manifest(tmp_path / "src" / name, name,
                          extra="protected = true\n")
    mgr = _manager(tmp_path)
    for install in (mgr.set_installed_from_dir, mgr.install_external):
        with pytest.raises(ValueError,
                           match="reserved for a first-party plugin|clashes with a built-in"):
            install(src, force=force)
    assert not (tmp_path / "installed" / name).exists()


def test_third_party_install_refuses_a_store_plugin_name(tmp_path):
    src = _write_manifest(tmp_path / "src" / "zeta", "zeta", scope="zeta-x")
    mgr = _store_manager(tmp_path)
    with pytest.raises(ValueError, match="reserved for a first-party plugin"):
        mgr.set_installed_from_dir(src)
    assert not (tmp_path / "installed" / "zeta").exists()


def _external_folder(tmp_path, name, **kw):
    from localm.plugins.engine import _write_marker
    pdir = _write_manifest(tmp_path / "installed" / name, name, **kw)
    _write_marker(pdir, "external", "0")
    return pdir


def test_external_folder_under_a_first_party_name_has_third_party_standing(tmp_path):
    _external_folder(tmp_path, "tts",
                     extra="protected = true\ndefault_enabled = true\n")
    mgr = _store_manager(tmp_path)
    spec = mgr.discover()["tts"]
    assert (spec.builtin, spec.protected, spec.default_enabled) == (False, False, False)
    assert mgr._is_protected("tts") is False
    entry = next(p for p in mgr.api_state()["plugins"] if p["name"] == "tts")
    assert (entry["builtin"], entry["protected"]) == (False, False)
    mgr.disable("tts")
    assert mgr.uninstall("tts") is True
    assert not (tmp_path / "installed" / "tts").exists()


def test_external_folder_under_a_first_party_name_gets_the_scope_recheck(tmp_path):
    _external_folder(tmp_path, "tts", scope="chat")
    mgr = _store_manager(tmp_path)
    assert "tts" not in mgr.discover()
    assert "reserved localm scope" in mgr._discover_errors["tts"]


def test_store_install_keeps_first_party_standing(tmp_path):
    from localm.plugins.engine import _write_store_install
    dest = tmp_path / "installed" / "tts"
    mgr = _store_manager(tmp_path)
    dest.mkdir(parents=True)
    _write_store_install(dest, tmp_path / "store" / "tts")
    spec = mgr.discover()["tts"]
    assert (spec.builtin, spec.protected, spec.default_enabled) == (True, True, True)
    assert mgr._is_protected("tts") is True
    entry = next(p for p in mgr.api_state()["plugins"] if p["name"] == "tts")
    assert (entry["builtin"], entry["protected"]) == (True, True)
    with pytest.raises(ValueError, match="protected"):
        mgr.disable("tts")


@pytest.mark.parametrize("body", ["{not json", "[]"])
def test_an_unreadable_marker_under_a_first_party_name_is_not_first_party(
        tmp_path, caplog, body):
    import logging

    from localm.plugins import lookup
    from localm.plugins.lookup import SOURCE_MARKER, installed_from_store
    pdir = _write_manifest(tmp_path / "installed" / "tts", "tts",
                           extra="protected = true\ndefault_enabled = true\n")
    (pdir / SOURCE_MARKER).write_text(body, encoding="utf-8")
    lookup._UNREADABLE_MARKERS_WARNED.discard(str(pdir))
    mgr = _store_manager(tmp_path)
    with caplog.at_level(logging.WARNING, logger="localm"):
        assert installed_from_store(pdir) is False
        spec = mgr.discover()["tts"]
        mgr.discover()
    assert (spec.builtin, spec.protected, spec.default_enabled) == (False, False, False)
    assert mgr._is_protected("tts") is False
    warned = [r for r in caplog.records if "cannot be read" in r.getMessage()]
    assert len(warned) == 1 and str(pdir) in warned[0].getMessage()


def test_a_folder_without_a_marker_still_counts_as_a_store_install(tmp_path):
    from localm.plugins.lookup import installed_from_store
    pdir = _write_manifest(tmp_path / "installed" / "tts", "tts")
    assert installed_from_store(pdir) is True
