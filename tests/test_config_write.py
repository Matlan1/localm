# SPDX-License-Identifier: AGPL-3.0-or-later
"""A plugin writes a global config key through ``set_global_config_key``, which
applies the schema's owner tier, and no plugin module uses a ``localm.config``
function that writes ``config.json`` itself."""

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


def _host():
    from localm.plugins.contract import PluginSpec
    from localm.plugins.engine import PluginHost
    return PluginHost(None, None, PluginSpec(name="probe"))


def test_the_host_method_applies_the_owner_tier_to_the_requests_caller(
        home, monkeypatch):
    from localm.inference import http_server as hs
    host = _host()
    before = _stored("embedding_model")
    monkeypatch.setattr(hs, "caller_is_operator", lambda request: False)
    with pytest.raises(PermissionError):
        host.set_global_config(object(), "embedding_model", "bge-small-en-v1.5")
    assert _stored("embedding_model") == before
    monkeypatch.setattr(hs, "caller_is_operator", lambda request: True)
    host.set_global_config(object(), "embedding_model", "bge-small-en-v1.5")
    assert _stored("embedding_model") == "bge-small-en-v1.5"


def _config_writers() -> frozenset:
    """Names in ``localm.config`` a plugin module may not use: every function
    whose body writes ``CONFIG_FILE`` through ``_atomic_write_json``, every
    function in the module that calls a public one of those (to a fixed point),
    the write primitive ``_atomic_write_json`` and the ``CONFIG_FILE`` path
    itself. A private writer is denied but does not make its callers writers:
    ``load_config`` persists a stored-config migration through one."""
    tree = ast.parse((_PKG / "config.py").read_text(encoding="utf-8"))
    funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}

    def called(fn):
        return {c.func.id for c in ast.walk(fn)
                if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)}

    def writes_config_file(fn):
        return any(isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
                   and c.func.id == "_atomic_write_json" and c.args
                   and isinstance(c.args[0], ast.Name) and c.args[0].id == "CONFIG_FILE"
                   for c in ast.walk(fn))

    writers = {name for name, fn in funcs.items() if writes_config_file(fn)}
    grew = True
    while grew:
        grew = False
        for name, fn in funcs.items():
            public = {w for w in writers if not w.startswith("_")}
            if name not in writers and called(fn) & public:
                writers.add(name)
                grew = True
    return frozenset(writers | {"_atomic_write_json", "CONFIG_FILE"})


_DENIED = _config_writers()


def _config_writer_uses(source: str, denied=_DENIED) -> list:
    """(line, name) pairs where *source* imports, names or calls a name in
    *denied*."""
    hits = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom):
            hits += [(node.lineno, a.name) for a in node.names if a.name in denied]
        elif isinstance(node, ast.Name) and node.id in denied:
            hits.append((node.lineno, node.id))
        elif isinstance(node, ast.Attribute) and node.attr in denied:
            hits.append((node.lineno, node.attr))
    return sorted(set(hits))


def test_the_denied_set_is_derived_from_every_config_writer():
    assert {"update_config", "save_config", "_atomic_write_json",
            "CONFIG_FILE"} <= _DENIED
    assert "_persist_migrated_config" in _DENIED
    assert not {"load_config", "load_config_checked", "save_registry",
                "update_registry"} & _DENIED


def test_the_scanner_sees_each_way_a_module_reaches_a_config_writer():
    assert _config_writer_uses("from localm.config import update_config\n") == [
        (1, "update_config")]
    assert _config_writer_uses("import localm.config as c\nc.update_config(f)\n") == [
        (2, "update_config")]
    assert _config_writer_uses("update_config(f)\n") == [(1, "update_config")]
    assert _config_writer_uses("def f():\n    from localm import config\n"
                               "    config.update_config(g)\n") == [(3, "update_config")]
    assert _config_writer_uses("from localm.config import save_config\n") == [
        (1, "save_config")]
    assert _config_writer_uses("import localm.config as c\n"
                               "c._atomic_write_json(c.CONFIG_FILE, {})\n") == [
        (2, "CONFIG_FILE"), (2, "_atomic_write_json")]
    assert _config_writer_uses("set_global_config_key('k', 1, is_owner=True)\n") == []
    assert _config_writer_uses("from localm.config import load_config\n") == []


def test_no_plugin_module_writes_global_config_directly():
    offenders = []
    scanned = 0
    for root in _PLUGIN_ROOTS:
        for path in sorted(root.rglob("*.py")):
            scanned += 1
            for line, name in _config_writer_uses(path.read_text(encoding="utf-8")):
                offenders.append(
                    f"{path.relative_to(_PKG.parent).as_posix()}:{line} {name}")
    assert scanned > 50, "the plugin source roots were not found"
    assert not offenders, (
        "plugin modules must write global config through "
        "Host.set_global_config (or config_write.set_global_config_key), not a "
        "config.py writer:\n  " + "\n  ".join(offenders))
