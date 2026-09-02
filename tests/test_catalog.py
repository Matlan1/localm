# SPDX-License-Identifier: AGPL-3.0-or-later
"""First-party plugin catalog: command -> plugin lookup and the
'that command needs the X plugin' suggestion, plus the CLI REPL wiring that
turns an unknown command for a known-but-inactive plugin into that hint
(gated on the suggest_plugins config toggle)."""

import io

import pytest
from rich.console import Console

from localm.plugins import catalog


# --------------------------------------------------------------------------- #
#  Pure catalog helpers                                                        #
# --------------------------------------------------------------------------- #

def test_for_command_maps_renamed_media_verbs():
    assert catalog.for_command("generate-image").name == "image"
    assert catalog.for_command("generate-music").name == "music"
    assert catalog.for_command("generate-video").name == "video"
    assert catalog.for_command("coder").name == "coder"
    assert catalog.for_command("nope") is None


def test_commands_map_covers_every_catalog_command():
    cmds = catalog.commands()
    # every command verb declared on an entry resolves back to that entry
    for entry in catalog.CATALOG:
        for verb in entry.commands:
            assert cmds[verb] == entry.name


def test_suggestion_for_known_inactive_plugin_command():
    msg = catalog.suggestion("generate-image")
    assert msg is not None
    assert "image plugin" in msg
    assert "localm plugin install image" in msg


def test_suggestion_none_for_unknown_command():
    assert catalog.suggestion("totally-made-up") is None


def test_suggestion_skips_preinstalled_plugins():
    # chat is preinstalled/protected and owns no commands, but the guard must
    # hold even if a preinstalled plugin ever declared one.
    for entry in catalog.CATALOG:
        if entry.preinstalled:
            for verb in entry.commands:
                assert catalog.suggestion(verb) is None


# --------------------------------------------------------------------------- #
#  CLI REPL wiring (_handle_command)                                           #
# --------------------------------------------------------------------------- #

@pytest.fixture
def cfg_env(tmp_path, monkeypatch):
    """Isolate config to a temp file so suggest_plugins can be toggled."""
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", tmp_path)
    monkeypatch.setattr(cfg, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", tmp_path / "registry.json")
    return cfg


def _run_command(monkeypatch, raw):
    """Invoke the REPL command handler and return what it printed."""
    from localm import cli
    buf = io.StringIO()
    # Wide, non-terminal console so rich does not soft-wrap the hint mid-string.
    monkeypatch.setattr(cli, "console",
                        Console(file=buf, force_terminal=False, width=200))
    cli._handle_command(raw, [], {})
    return buf.getvalue()


# /knowledge (the rag plugin) is a catalogued verb that is NOT a built-in chat
# REPL command, so it exercises the unknown-command -> suggestion wiring.
def test_unknown_plugin_command_suggests_install(cfg_env, monkeypatch):
    out = _run_command(monkeypatch, "/knowledge")
    assert "rag plugin" in out
    assert "localm plugin install rag" in out
    assert "Unknown" not in out


def test_suggest_plugins_off_falls_back_to_unknown(cfg_env, monkeypatch):
    cfg_env.save_config({"suggest_plugins": False})
    out = _run_command(monkeypatch, "/knowledge")
    assert "Unknown" in out
    assert "rag plugin" not in out


def test_genuinely_unknown_command_is_unknown(cfg_env, monkeypatch):
    out = _run_command(monkeypatch, "/wat")
    assert "Unknown" in out

def test_every_bundled_plugin_is_in_the_catalog():
    # `localm plugin setup` and `localm plugin install-deps` build their
    # selection from CATALOG alone, while `plugin status` and the GUI union the
    # bundled store with it. A plugin present in the store but absent here is
    # therefore advertised as available and refused by the installer's own
    # command. The browser plugin shipped in exactly that state.
    from localm.plugins.engine import PluginManager

    store = PluginManager(None).store_catalog()
    names = {s["name"] if isinstance(s, dict) else s for s in store}
    missing = sorted(names - set(catalog.names()))
    assert not missing, (
        f"bundled store plugins missing from CATALOG: {missing}. "
        "Add a CatalogEntry, or `localm plugin setup` can never offer them."
    )
