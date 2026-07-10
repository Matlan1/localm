# SPDX-License-Identifier: AGPL-3.0-or-later
"""Chat CONTENT must never reach the debug log in privacy mode, even when the
debug log is on (via --debug or the keep_diagnostics toggle). debug_content_enabled()
is the gate the GGUF backend's "raw model output" line and the jobs web-tool arg
line both use; operational lines stay on the looser debug_enabled()."""

from localm import debuglog


def _debug(monkeypatch, on):
    if on:
        monkeypatch.setenv("LOCALM_DEBUG", "1")
    else:
        monkeypatch.delenv("LOCALM_DEBUG", raising=False)


def _mode(monkeypatch, **cfg):
    monkeypatch.delenv("LOCALM_MODE", raising=False)   # config decides
    monkeypatch.setattr("localm.config.load_config", lambda: cfg)


def test_no_content_when_debug_off(monkeypatch):
    _debug(monkeypatch, False)
    _mode(monkeypatch, mode="full")
    assert debuglog.debug_content_enabled() is False   # log off -> no content


def test_no_content_in_privacy_even_with_debug_on(monkeypatch):
    _debug(monkeypatch, True)
    _mode(monkeypatch, mode="privacy")
    assert debuglog.debug_content_enabled() is False   # THE fix


def test_content_allowed_in_log_mode(monkeypatch):
    _debug(monkeypatch, True)
    _mode(monkeypatch, mode="log", chat_mode=None)
    assert debuglog.debug_content_enabled() is True


def test_content_allowed_in_full_mode(monkeypatch):
    _debug(monkeypatch, True)
    _mode(monkeypatch, mode="full", chat_mode=None)
    assert debuglog.debug_content_enabled() is True


def test_suppressed_when_a_per_surface_mode_is_privacy(monkeypatch):
    # Global log but the chat surface is privacy -> content suppressed (the backend
    # is surface-agnostic, so ANY relevant surface being privacy wins).
    _debug(monkeypatch, True)
    _mode(monkeypatch, mode="log", chat_mode="privacy")
    assert debuglog.debug_content_enabled() is False


def test_fails_safe_to_no_content_on_error(monkeypatch):
    _debug(monkeypatch, True)
    monkeypatch.delenv("LOCALM_MODE", raising=False)

    def _boom():
        raise RuntimeError("config unreadable")

    monkeypatch.setattr("localm.config.load_config", _boom)
    assert debuglog.debug_content_enabled() is False   # fail-safe: no content
