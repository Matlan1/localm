# SPDX-License-Identifier: AGPL-3.0-or-later
"""First-party plugin catalog: chat command -> plugin lookup, and the terminal
chat, which answers a command it does not have as unknown."""

import io

import pytest
from rich.console import Console

from localm.plugins import catalog


# --------------------------------------------------------------------------- #
#  Pure catalog helpers                                                        #
# --------------------------------------------------------------------------- #

def test_for_command_maps_the_chat_commands_and_their_aliases():
    for verbs, plugin in ((("generate-image", "imagine"), "image"),
                          (("generate-music", "music"), "music"),
                          (("generate-video", "video"), "video"),
                          (("web",), "web"),
                          (("remember", "memory"), "memory")):
        for verb in verbs:
            assert catalog.for_command(verb).name == plugin, verb
    assert catalog.for_command("nope") is None


def test_cli_verbs_are_not_chat_commands():
    for verb in ("coder", "knowledge", "mcp", "job"):
        assert catalog.for_command(verb) is None, verb


def _gui_chat_commands() -> set:
    """The commands the GUI chat composer handles: execChatCommand's cases and the
    CHAT_ALIASES it maps onto one of them."""
    import re
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "localm" / "plugins" / "gui" / "static"
           / "app" / "slash.js").read_text(encoding="utf-8")
    body = src[src.index("export function execChatCommand"):]
    body = body[:body.index("\n}\n")]
    handled = set(re.findall(r'case "([a-z-]+)"', body))
    aliases = re.search(r"export const CHAT_ALIASES = \{([^}]*)\}", src)
    if aliases and "CHAT_ALIASES[cmd]" in body:
        for alias, target in re.findall(r'([a-z-]+):\s*"([a-z-]+)"', aliases.group(1)):
            if target in handled:
                handled.add(alias)
    return handled


def test_every_catalog_command_is_a_command_the_chat_composer_handles():
    handled = _gui_chat_commands()
    assert {"remember", "generate-image"} <= handled, handled
    missing = {verb for verb in catalog.commands() if verb not in handled}
    assert not missing, f"catalog commands the chat composer does not handle: {missing}"


def test_the_coder_entry_names_no_pip_extra_its_manifest_does_not_declare():
    import tomllib
    from pathlib import Path
    store = Path(__file__).resolve().parents[1] / "localm" / "plugins" / "builtin"
    for entry in catalog.CATALOG:
        manifest = store / entry.name / "plugin.toml"
        if not entry.extra or not manifest.is_file():
            continue
        declared = tomllib.loads(manifest.read_text(encoding="utf-8")).get(
            "plugin", {}).get("requires_extras", [])
        assert entry.extra in declared, (entry.name, entry.extra, declared)


def test_commands_map_covers_every_catalog_command():
    cmds = catalog.commands()
    # every command verb declared on an entry resolves back to that entry
    for entry in catalog.CATALOG:
        for verb in entry.commands:
            assert cmds[verb] == entry.name


def test_preinstalled_plugins_own_no_chat_command():
    for entry in catalog.CATALOG:
        if entry.preinstalled:
            assert entry.commands == (), entry.name


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


@pytest.mark.parametrize("raw", ["/knowledge", "/coder", "/remember x", "/web q"])
def test_a_command_the_terminal_chat_does_not_have_is_unknown_not_an_install_hint(
        cfg_env, monkeypatch, raw):
    out = _run_command(monkeypatch, raw)
    assert "Unknown" in out
    assert "plugin" not in out


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
