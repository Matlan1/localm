# SPDX-License-Identifier: AGPL-3.0-or-later
"""`localm plugin setup` - the installer's plugin-selection step (F3).

Out of the box only chat is active; setup lets the user (or the installer, or a
script via --plugins/--all/--defaults) turn on the first-party plugins they want.
Driven through Click's CliRunner against the real bundled store, with a throwaway
LOCALM_HOME so installs land in tmp.
"""

import pytest
from click.testing import CliRunner


@pytest.fixture
def home_env(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path))
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", tmp_path)
    monkeypatch.setattr(cfg, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", tmp_path / "registry.json")
    return tmp_path


def _installed():
    from localm.plugins.engine import PluginManager
    return {p["name"] for p in PluginManager(None).api_state()["plugins"] if p.get("installed")}


def test_setup_installs_named_plugins(home_env):
    from localm.cli import main
    r = CliRunner().invoke(main, ["plugin", "setup", "--plugins", "web,rag"])
    assert r.exit_code == 0, r.output
    inst = _installed()
    assert "web" in inst and "rag" in inst
    assert "image" not in inst        # only the selected ones


def test_setup_defaults_installs_recommended_set(home_env):
    from localm.cli import main
    r = CliRunner().invoke(main, ["plugin", "setup", "--defaults"])
    assert r.exit_code == 0, r.output
    inst = _installed()
    for n in ("coder", "rag", "web", "tts"):
        assert n in inst, f"{n} should be in the default set"
    assert "video" not in inst        # heavy/opt-in plugins are not defaulted


def test_setup_all_installs_every_first_party_plugin(home_env):
    from localm.cli import main
    r = CliRunner().invoke(main, ["plugin", "setup", "--all"])
    assert r.exit_code == 0, r.output
    inst = _installed()
    for n in ("coder", "image", "music", "video", "rag", "web", "voice", "tts", "mcp"):
        assert n in inst, f"--all should install {n}"


def test_setup_non_interactive_shell_skips_without_flags(home_env):
    # CliRunner's stdin is not a tty: with no flags, setup must skip (not hang,
    # not install anything) so a scripted/CI install does not block.
    from localm.cli import main
    r = CliRunner().invoke(main, ["plugin", "setup"])
    assert r.exit_code == 0, r.output
    assert _installed() == set() or "chat" in _installed()  # nothing newly installed
    assert "skipping" in r.output.lower()


def test_setup_unknown_name_is_skipped_not_fatal(home_env):
    from localm.cli import main
    r = CliRunner().invoke(main, ["plugin", "setup", "--plugins", "web,bogusplugin"])
    assert r.exit_code == 0, r.output
    inst = _installed()
    assert "web" in inst
    assert "bogusplugin" not in inst
