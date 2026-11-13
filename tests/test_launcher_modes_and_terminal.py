# SPDX-License-Identifier: AGPL-3.0-or-later
"""The Tk launcher reads the session modes from config.json, the one place
every surface reads them from, writes them back there, and passes no --mode
(which a running server counts as a conflict) unless that write failed. On
POSIX the terminal modes open a terminal window to run in."""

from __future__ import annotations

import importlib.machinery
import importlib.util
from pathlib import Path

import pytest

from tests._tk_root import build_tk_root, collect_tk_garbage  # noqa: F401

_LAUNCHER = Path(__file__).resolve().parents[1] / "launcher.pyw"


def _load():
    loader = importlib.machinery.SourceFileLoader("localm_launcher_modes", str(_LAUNCHER))
    spec = importlib.util.spec_from_file_location("localm_launcher_modes", _LAUNCHER,
                                                  loader=loader)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def mod():
    return _load()


def _set_config(**values):
    from localm.config import update_config

    def _mutate(cfg):
        cfg.update(values)
    update_config(_mutate)


def _app(mod, monkeypatch, tmp_path, saved=None):
    settings = tmp_path / "launcher.json"
    if saved is not None:
        import json
        settings.write_text(json.dumps(saved), encoding="utf-8")
    monkeypatch.setattr(mod, "SETTINGS_FILE", settings)
    monkeypatch.setattr(mod.Launcher, "_refresh_models", lambda self, sync=True: None)
    app = build_tk_root(mod.Launcher)
    app.withdraw()
    return app


# --------------------------------------------------------------------------- #
#  Session modes                                                               #
# --------------------------------------------------------------------------- #

def test_config_privacy_modes_reads_config_json(mod):
    _set_config(mode="full", chat_mode="log", coder_mode=None)
    assert mod.config_privacy_modes() == ("full", "log", mod.USE_GLOBAL)


def test_an_invalid_config_mode_reads_as_privacy(mod):
    _set_config(mode="bogus", chat_mode="nope", coder_mode="full")
    assert mod.config_privacy_modes() == ("privacy", mod.USE_GLOBAL, "full")


def test_the_dropdowns_start_from_config_not_launcher_json(mod, monkeypatch, tmp_path):
    _set_config(mode="full", chat_mode=None, coder_mode="log")
    app = _app(mod, monkeypatch, tmp_path,
               saved={"privacy_global": "privacy", "privacy_chat": "privacy",
                      "privacy_coder": "privacy"})
    try:
        assert app.privacy_global.get() == "full"
        assert app.privacy_chat.get() == mod.USE_GLOBAL
        assert app.privacy_coder.get() == "log"
    finally:
        app.destroy()


def _launch_capture(mod, monkeypatch, app, *, mode="gui"):
    calls = []
    monkeypatch.setattr(mod, "_spawn_detached",
                        lambda cmd, **kw: calls.append((cmd, kw)))
    monkeypatch.setattr(app, "after", lambda *a, **k: None)
    app.mode.set(mode)
    app.model.set(mod.NO_MODEL_LABEL if mode == "gui" else "m")
    app._launch()
    assert calls, app._status_full
    return calls[-1]


def test_a_launch_keeps_the_mode_in_config_and_passes_no_mode_flag(mod, monkeypatch, tmp_path):
    _set_config(mode="full")
    app = _app(mod, monkeypatch, tmp_path)
    try:
        cmd, _ = _launch_capture(mod, monkeypatch, app)
    finally:
        app.destroy()
    assert "--mode" not in cmd
    from localm.config import load_config
    assert load_config().get("mode") == "full"


def test_a_failed_config_write_passes_the_mode_and_says_so(mod, monkeypatch, tmp_path):
    app = _app(mod, monkeypatch, tmp_path)
    try:
        app.privacy_global.set("log")
        app.privacy_chat.set("full")
        monkeypatch.setattr(app, "_save_privacy_config", lambda: "disk full")
        cmd, _ = _launch_capture(mod, monkeypatch, app, mode="chat")
        status = app._status_full
    finally:
        app.destroy()
    assert cmd[cmd.index("--mode") + 1] == "full"
    assert "disk full" in status


def test_terminal_modes_ask_for_a_terminal(mod, monkeypatch, tmp_path):
    app = _app(mod, monkeypatch, tmp_path)
    try:
        _, kw_chat = _launch_capture(mod, monkeypatch, app, mode="chat")
        _, kw_gui = _launch_capture(mod, monkeypatch, app, mode="gui")
    finally:
        app.destroy()
    assert kw_chat.get("terminal") is True
    assert kw_gui.get("terminal") is False


# --------------------------------------------------------------------------- #
#  POSIX terminal window                                                       #
# --------------------------------------------------------------------------- #

def _which(found):
    return lambda name: f"/usr/bin/{name}" if name in found else None


def test_linux_uses_the_first_terminal_found(mod):
    argv = mod.terminal_command(["/c/.venv/bin/python", "-m", "localm", "run", "m"],
                                env={}, platform="linux", which=_which({"xterm"}))
    assert argv[:4] == ["/usr/bin/xterm", "-e", "sh", "-c"]
    assert argv[4].startswith("/c/.venv/bin/python -m localm run m || {")


def test_the_terminal_variable_wins(mod):
    argv = mod.terminal_command(["x"], env={"TERMINAL": "foot"}, platform="linux",
                                which=_which({"foot", "xterm"}))
    assert argv[:2] == ["/usr/bin/foot", "-e"]


def test_gnome_terminal_gets_its_separator(mod):
    argv = mod.terminal_command(["x"], env={}, platform="linux",
                                which=_which({"gnome-terminal"}))
    assert argv[:2] == ["/usr/bin/gnome-terminal", "--"]


def test_no_terminal_found_is_none(mod):
    assert mod.terminal_command(["x"], env={}, platform="linux", which=_which(set())) is None


def test_macos_uses_terminal_app_with_the_command_quoted(mod):
    argv = mod.terminal_command(["/Users/a b/.venv/bin/python", "-m", "localm", "run", 'q"m'],
                                env={}, platform="darwin", which=_which({"osascript"}))
    assert argv[0] == "/usr/bin/osascript"
    script = argv[2]
    assert script.startswith('tell application "Terminal" to do script "')
    assert "'/Users/a b/.venv/bin/python'" in script
    assert '\\"' in script


def test_debug_holds_the_terminal_whatever_happened(mod):
    plain = mod._posix_hold_script(["x"], {})
    debug = mod._posix_hold_script(["x", "--debug"], {})
    env_debug = mod._posix_hold_script(["x"], {"LOCALM_DEBUG": "1"})
    assert plain.startswith("x || {")
    assert debug.startswith("x --debug; ")
    assert env_debug.startswith("x; ")


def test_posix_spawn_without_a_terminal_refuses_before_starting(mod, monkeypatch):
    started = []
    monkeypatch.setattr(mod.subprocess, "Popen", lambda *a, **k: started.append(a))
    monkeypatch.setattr(mod.sys, "platform", "linux")
    monkeypatch.setattr(mod, "terminal_command", lambda cmd, env: None)
    with pytest.raises(RuntimeError, match="no terminal program"):
        mod._spawn_detached(["x"], cwd="/w", terminal=True)
    assert started == []


def test_posix_spawn_with_a_terminal_starts_the_wrapped_command(mod, monkeypatch):
    started = []
    monkeypatch.setattr(mod.subprocess, "Popen", lambda cmd, **k: started.append((cmd, k)))
    monkeypatch.setattr(mod.sys, "platform", "linux")
    monkeypatch.setattr(mod, "terminal_command", lambda cmd, env: ["term", "-e", *cmd])
    mod._spawn_detached(["x"], cwd="/w", terminal=True)
    assert started[0][0] == ["term", "-e", "x"]
    assert started[0][1].get("start_new_session") is True


def test_settings_are_not_saved_when_no_data_folder_is_known(mod, monkeypatch):
    monkeypatch.setattr(mod, "SETTINGS_FILE", None)
    assert mod.load_settings() == {}
    assert "no data folder" in mod.save_settings({"a": 1})
