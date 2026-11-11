# SPDX-License-Identifier: AGPL-3.0-or-later
"""First-party plugins load from the localm package; an install is a marker
folder; core reaches a plugin only through ``localm.plugins.lookup``.

Driven through the real PluginManager on the real bundled store, in a throwaway
data dir."""

from __future__ import annotations

import json
import sys

import pytest

from localm.config import load_config
from localm.plugins import lookup
from localm.plugins.engine import PluginManager
from localm.plugins.loader import plugins_dir


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    return tmp_path / "home"


def _manager():
    return PluginManager(None, installed_root=plugins_dir())


def _enabled() -> list:
    return list(load_config().get("plugins_enabled", []))


def test_store_install_writes_a_marker_folder_without_code(home):
    mgr = _manager()
    mgr.set_installed_state("memory", True)
    inst = plugins_dir() / "memory"
    names = sorted(p.name for p in inst.iterdir())
    assert names == [".localm-source.json", "plugin.toml"]
    marker = json.loads((inst / ".localm-source.json").read_text(encoding="utf-8"))
    assert marker == {"source": "store", "package": True}
    mgr.discover()
    spec = mgr._specs["memory"]
    assert spec.builtin is True
    assert (lookup.package_root() / "memory").samefile(spec.path)


def test_disabled_plugin_is_not_reachable_through_the_lookup(home, monkeypatch):
    from localm.plugins.builtin.jobs import runner
    from localm.plugins.builtin.jobs.store import Job
    from localm.plugins.builtin.memory import plug as memory_plug

    calls = []

    def fake_synthesize_memory(complete, **kw):
        calls.append(1)
        return {"status": "ok", "facts": []}

    monkeypatch.setattr(memory_plug, "synthesize_memory", fake_synthesize_memory)
    mgr = _manager()
    mgr.set_installed_state("memory", True)
    mgr.set_enabled_state("memory", False)
    assert "memory" not in _enabled()

    with pytest.raises(lookup.PluginUnavailable):
        lookup.plugin_api("memory", "plug")
    job = Job(name="m", task_kind="memory", prompt="", schedule_kind="interval",
              schedule=60)
    result = runner._run_memory(job, engine=object())
    assert result.startswith("Memory is unavailable")
    assert calls == []

    mgr.set_enabled_state("memory", True)
    assert lookup.plugin_api("memory", "plug") is memory_plug


def test_the_protected_chat_plugin_is_reachable_once_provisioned(home):
    with pytest.raises(lookup.PluginUnavailable):
        lookup.plugin_api("chat", "plug")
    _manager().discover()
    assert "chat" not in _enabled()
    assert lookup.plugin_api("chat", "plug").rename_notes


def test_not_installed_plugin_is_not_reachable_through_the_lookup(home):
    assert not (plugins_dir() / "memory").exists()
    with pytest.raises(lookup.PluginUnavailable):
        lookup.plugin_api("memory")
    assert lookup.is_available("memory") is False


def test_third_party_install_under_a_first_party_name_is_not_the_package(home):
    d = plugins_dir() / "memory"
    d.mkdir(parents=True)
    (d / "plugin.toml").write_text(
        '[plugin]\nname = "memory"\nregister = "plug"\n', encoding="utf-8")
    (d / ".localm-source.json").write_text(
        json.dumps({"source": "external", "src_hash": ""}), encoding="utf-8")
    from localm.config import update_config
    update_config(lambda c: c.__setitem__("plugins_enabled", ["memory"]))
    assert lookup.is_available("memory") is False


def test_one_module_instance_per_first_party_plugin(home):
    from fastapi import FastAPI
    app = FastAPI()
    mgr = PluginManager(app, installed_root=plugins_dir())
    mgr.install("memory")
    entry = mgr._loaded["memory"]
    module, uniq = entry[1], entry[3]
    assert uniq == "localm.plugins.builtin.memory.plug"
    assert sys.modules[uniq] is module
    assert lookup.plugin_api("memory", "plug") is module
    assert not any(k.startswith("_localm_plugin_memory") for k in sys.modules)

    mgr.disable("memory")
    assert sys.modules.get(uniq) is module
    mgr.enable("memory")
    assert mgr._loaded["memory"][1] is module
    mgr.disable("memory")


def test_old_copied_install_is_converted_and_keeps_other_files(home):
    import shutil
    src = lookup.package_root() / "memory"
    dest = plugins_dir() / "memory"
    shutil.copytree(src, dest)
    (dest / "__pycache__").mkdir(exist_ok=True)
    (dest / "__pycache__" / "plug.cpython-312.pyc").write_bytes(b"\0")
    (dest / ".localm-source.json").write_text(
        json.dumps({"source": "store", "src_hash": "old"}), encoding="utf-8")
    (dest / "notes.txt").write_text("mine", encoding="utf-8")
    (dest / "user").mkdir()
    (dest / "user" / "data.json").write_text("{}", encoding="utf-8")
    from localm.config import update_config
    update_config(lambda c: c.__setitem__("plugins", {"memory": {"k": "v"}}))

    mgr = _manager()
    assert mgr.refresh("memory") == ["memory"]

    left = sorted(p.relative_to(dest).as_posix() for p in dest.rglob("*") if p.is_file())
    assert left == [".localm-source.json", "notes.txt", "plugin.toml", "user/data.json"]
    assert (dest / "notes.txt").read_text(encoding="utf-8") == "mine"
    assert json.loads((dest / ".localm-source.json").read_text(encoding="utf-8"))["package"]
    assert load_config()["plugins"]["memory"] == {"k": "v"}
    assert mgr.refresh("memory") == []
    mgr.discover()
    assert (lookup.package_root() / "memory").samefile(mgr._specs["memory"].path)


def test_copied_install_is_never_imported_before_conversion(home):
    import shutil
    src = lookup.package_root() / "memory"
    dest = plugins_dir() / "memory"
    shutil.copytree(src, dest)
    mgr = _manager()
    mgr.discover()
    assert (lookup.package_root() / "memory").samefile(mgr._specs["memory"].path)


def test_shipped_data_reads_only_listed_modules(home):
    assert lookup.shipped_data("tts", "settings").defaults()
    with pytest.raises(lookup.PluginUnavailable):
        lookup.shipped_data("tts", "plug")
