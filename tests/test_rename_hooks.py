"""Model-rename hooks: plugins register a callable with the registry, which
runs it on a rename and never imports the plugin itself."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from localm import model_manager as mm
from localm.model_manager import registry


@pytest.fixture()
def clean_hooks(monkeypatch):
    monkeypatch.setattr(registry, "_RENAME_HOOKS", {})
    return registry._RENAME_HOOKS


def test_a_registered_hook_runs_and_an_unregistered_one_does_not(clean_hooks):
    calls = []

    def kept(old, new):
        calls.append(("kept", old, new))
        return ["kept note"]

    def dropped(old, new):
        calls.append(("dropped", old, new))
        return ["dropped note"]

    mm.register_rename_hook("kept", kept)
    mm.register_rename_hook("dropped", dropped)
    mm.unregister_rename_hook("dropped")
    notes = registry._migrate_model_references("a", "b")
    assert ("kept", "a", "b") in calls
    assert all(c[0] != "dropped" for c in calls)
    assert "kept note" in notes
    assert "dropped note" not in notes


def test_a_failing_hook_is_reported_and_does_not_stop_the_others(clean_hooks):
    def broken(old, new):
        raise RuntimeError("boom")

    mm.register_rename_hook("broken", broken)
    mm.register_rename_hook("fine", lambda o, n: ["fine note"])
    notes = registry._migrate_model_references("a", "b")
    assert any("Could not update broken" in n and "boom" in n for n in notes)
    assert "fine note" in notes


def test_unregistering_an_unknown_hook_is_ignored(clean_hooks):
    mm.unregister_rename_hook("never-registered")


def test_registering_a_name_again_replaces_the_hook(clean_hooks):
    mm.register_rename_hook("x", lambda o, n: ["first"])
    mm.register_rename_hook("x", lambda o, n: ["second"])
    notes = registry._migrate_model_references("a", "b")
    assert "second" in notes and "first" not in notes


class _Host:
    def mount_router(self, router):
        pass

    def mount_static(self, path):
        pass

    def register_chat_hook(self, kind, fn):
        pass


def test_the_chat_plugin_registers_its_hook_on_load_and_removes_it_on_unload(clean_hooks):
    from localm.plugins.builtin.chat import plug
    plug.register(_Host())
    assert "saved chats" in clean_hooks
    plug.unregister()
    assert "saved chats" not in clean_hooks


def test_the_registry_module_names_no_plugin():
    src = Path(registry.__file__).read_text(encoding="utf-8")
    assert "localm.plugins" not in src


def test_a_headless_server_import_loads_nothing_from_the_gui_package():
    code = (
        "import sys, json\n"
        "import localm.inference.http_server\n"
        "import localm.plugins.engine\n"
        "import localm.jobmanager\n"
        "print(json.dumps(sorted(m for m in sys.modules "
        "if m == 'localm.plugins.gui' or m.startswith('localm.plugins.gui.'))))\n"
    )
    root = str(Path(registry.__file__).resolve().parents[2])
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
        cwd=root, env={**__import__("os").environ, "PYTHONPATH": root},
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), timeout=120)
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout.strip().splitlines()[-1]) == []
