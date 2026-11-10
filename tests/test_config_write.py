# SPDX-License-Identifier: AGPL-3.0-or-later
"""A plugin writes a global config key through ``set_global_config_key``, which
applies the schema's owner tier, and no plugin module calls ``update_config``
itself."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import localm
from localm.config_write import OwnerRequired, owner_only_keys, set_global_config_key

_PKG = Path(localm.__file__).resolve().parent

# Plugin code: the bundled plugins, the MCP server's tools and the coder plugin
# implementation. The engine (plugins/engine.py) and the GUI surface are kernel.
_PLUGIN_ROOTS = (
    _PKG / "plugins" / "builtin",
    _PKG / "plugins" / "mcpserver",
    _PKG / "plugins" / "coder",
)


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path))
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", tmp_path)
    monkeypatch.setattr(cfg, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", tmp_path / "registry.json")
    return tmp_path


def _stored(key):
    from localm.config import load_config
    return load_config()[key]


def _ordinary_toggle() -> str:
    from localm.settings_schema import CORE_FIELDS, Widget
    owner_only = owner_only_keys()
    return next(f.key for f in CORE_FIELDS
                if f.widget == Widget.TOGGLE and f.key not in owner_only)


def test_a_non_owner_cannot_write_an_owner_only_key(home):
    before = _stored("embedding_model")
    with pytest.raises(OwnerRequired):
        set_global_config_key("embedding_model", "bge-small-en-v1.5", is_owner=False)
    assert _stored("embedding_model") == before


def test_the_owner_writes_an_owner_only_key(home):
    set_global_config_key("embedding_model", "bge-small-en-v1.5", is_owner=True)
    assert _stored("embedding_model") == "bge-small-en-v1.5"


def test_every_engine_managed_key_is_owner_only(home):
    from localm.settings_schema import engine_managed_keys
    assert engine_managed_keys() <= owner_only_keys()
    for key in engine_managed_keys():
        with pytest.raises(OwnerRequired):
            set_global_config_key(key, {}, is_owner=False)


def test_a_non_owner_writes_an_ordinary_key(home):
    key = _ordinary_toggle()
    flipped = not _stored(key)
    set_global_config_key(key, flipped, is_owner=False)
    assert _stored(key) is flipped


def test_an_unknown_key_is_refused_for_the_owner_too(home):
    with pytest.raises(ValueError):
        set_global_config_key("not_a_setting", 1, is_owner=True)


def test_a_value_the_schema_rejects_is_not_stored(home):
    key = _ordinary_toggle()
    before = _stored(key)
    with pytest.raises(ValueError):
        set_global_config_key(key, "maybe", is_owner=True)
    assert _stored(key) == before


def _update_config_uses(source: str) -> list:
    """Line numbers where *source* imports, names or calls ``update_config``."""
    hits = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom):
            if any(a.name == "update_config" for a in node.names):
                hits.append(node.lineno)
        elif (isinstance(node, ast.Name) and node.id == "update_config") or (
                isinstance(node, ast.Attribute) and node.attr == "update_config"):
            hits.append(node.lineno)
    return sorted(set(hits))


def test_the_scanner_sees_each_way_a_module_reaches_update_config():
    assert _update_config_uses("from localm.config import update_config\n") == [1]
    assert _update_config_uses("import localm.config as c\nc.update_config(f)\n") == [2]
    assert _update_config_uses("update_config(f)\n") == [1]
    assert _update_config_uses("def f():\n    from localm import config\n"
                               "    config.update_config(g)\n") == [3]
    assert _update_config_uses("set_global_config_key('k', 1, is_owner=True)\n") == []


def test_no_plugin_module_writes_global_config_directly():
    offenders = []
    scanned = 0
    for root in _PLUGIN_ROOTS:
        for path in sorted(root.rglob("*.py")):
            scanned += 1
            for line in _update_config_uses(path.read_text(encoding="utf-8")):
                offenders.append(f"{path.relative_to(_PKG.parent).as_posix()}:{line}")
    assert scanned > 50, "the plugin source roots were not found"
    assert not offenders, (
        "plugin modules must write global config through "
        "Host.set_global_config (or config_write.set_global_config_key), not "
        "update_config:\n  " + "\n  ".join(offenders))
