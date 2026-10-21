# SPDX-License-Identifier: AGPL-3.0-or-later
"""The launcher's status line is clipped to a fixed width, so a long error was
unreadable. A clipped line is clickable and shows the full text, and an error is
also appended to logs/launcher.log."""

import importlib.machinery
import importlib.util
from pathlib import Path

from tests._tk_root import build_tk_root, collect_tk_garbage  # noqa: F401

_LAUNCHER = Path(__file__).resolve().parents[1] / "launcher.pyw"
_LONG = ("Could not check the models folder: AttributeError: 'NoneType' object "
         "has no attribute 'reconfigure'")


def _load_launcher():
    loader = importlib.machinery.SourceFileLoader("localm_launcher_mod", str(_LAUNCHER))
    spec = importlib.util.spec_from_file_location("localm_launcher_mod", _LAUNCHER, loader=loader)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _app(mod, monkeypatch, tmp_path):
    monkeypatch.setattr(mod, "SETTINGS_FILE", tmp_path / "launcher.json")
    monkeypatch.setattr(mod.Launcher, "_refresh_models", lambda self, sync=True: None)
    app = build_tk_root(mod.Launcher)
    app.withdraw()
    return app


def test_a_clipped_status_shows_its_full_text_when_clicked(monkeypatch, tmp_path):
    mod = _load_launcher()
    shown = []
    monkeypatch.setattr(mod.messagebox, "showinfo",
                        lambda title, message, **kw: shown.append(message))
    app = _app(mod, monkeypatch, tmp_path)
    try:
        app.status_msg(_LONG, error=True)
        assert app.status.cget("text") != _LONG
        assert app.status.cget("cursor") == "hand2"
        app._show_full_status()
        assert len(shown) == 1
        assert _LONG in shown[0]
        assert app.clipboard_get() == _LONG
    finally:
        app.destroy()


def test_a_short_status_is_not_clickable(monkeypatch, tmp_path):
    mod = _load_launcher()
    shown = []
    monkeypatch.setattr(mod.messagebox, "showinfo",
                        lambda title, message, **kw: shown.append(message))
    app = _app(mod, monkeypatch, tmp_path)
    try:
        app.status_msg("Up to date (3 models).")
        assert app.status.cget("cursor") == ""
        app._show_full_status()
        assert shown == []
    finally:
        app.destroy()


def test_an_error_status_is_appended_to_the_launcher_log(monkeypatch, tmp_path):
    mod = _load_launcher()
    app = _app(mod, monkeypatch, tmp_path)
    try:
        app.privacy_global.set("log")
        app.status_msg(_LONG, error=True)
        app.status_msg("Up to date (3 models).")
    finally:
        app.destroy()
    log = (tmp_path / "logs" / "launcher.log").read_text(encoding="utf-8")
    assert _LONG in log
    assert "Up to date" not in log


_PATH_ERROR = "Import failed: /private/models/x.gguf is not a GGUF file"


def _error_log(tmp_path):
    return tmp_path / "logs" / "launcher.log"


def test_privacy_mode_writes_no_launcher_log(monkeypatch, tmp_path):
    mod = _load_launcher()
    app = _app(mod, monkeypatch, tmp_path)
    try:
        assert app.privacy_global.get() == "privacy"
        app.keep_diagnostics.set(False)
        app.status_msg(_PATH_ERROR, error=True)
        assert app.status.cget("text").startswith("Import failed")
    finally:
        app.destroy()
    assert not _error_log(tmp_path).exists()
    assert not (tmp_path / "logs").exists()


def test_privacy_mode_with_keep_diagnostics_writes_the_launcher_log(monkeypatch, tmp_path):
    mod = _load_launcher()
    app = _app(mod, monkeypatch, tmp_path)
    try:
        app.privacy_global.set("privacy")
        app.keep_diagnostics.set(True)
        app.status_msg(_PATH_ERROR, error=True)
    finally:
        app.destroy()
    assert _PATH_ERROR in _error_log(tmp_path).read_text(encoding="utf-8")


def test_switching_privacy_off_then_back_on_stops_the_log(monkeypatch, tmp_path):
    mod = _load_launcher()
    app = _app(mod, monkeypatch, tmp_path)
    try:
        app.privacy_global.set("full")
        app.status_msg("first error", error=True)
        app.privacy_global.set("privacy")
        app.keep_diagnostics.set(False)
        app.status_msg("second error", error=True)
    finally:
        app.destroy()
    text = _error_log(tmp_path).read_text(encoding="utf-8")
    assert "first error" in text
    assert "second error" not in text
