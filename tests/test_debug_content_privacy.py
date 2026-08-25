# SPDX-License-Identifier: AGPL-3.0-or-later
"""Chat CONTENT must never reach the debug log in privacy mode, even when the debug log is on (via --debug or the keep_diagnostics toggle). debug_content_enabled() is the gate the GGUF backend's 'raw model output' line and the jobs web-tool arg line both use; operational lines stay on the looser debug_..."""

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


def test_suppressed_when_the_coder_surface_is_privacy(monkeypatch):
    # Regression: debug_content_enabled() used to check only ("server", "chat"),
    # so a coder-only privacy override (coder_mode) never suppressed the raw
    # model output line the shared llama.cpp backend writes for a coder
    # session's own generations - even though the coder surface itself had
    # explicitly opted into privacy. Global log, chat/server unset (fall back
    # to global log), coder alone set to privacy.
    _debug(monkeypatch, True)
    _mode(monkeypatch, mode="log", coder_mode="privacy")
    assert debuglog.debug_content_enabled() is False


def test_content_allowed_when_no_surface_including_coder_is_privacy(monkeypatch):
    # The positive arm of the fix above: an explicit non-privacy coder_mode,
    # with every other surface also non-privacy, must still allow content -
    # adding the coder surface to the check must not turn into an
    # always-suppress.
    _debug(monkeypatch, True)
    _mode(monkeypatch, mode="log", coder_mode="full")
    assert debuglog.debug_content_enabled() is True


def test_fails_safe_to_no_content_on_error(monkeypatch):
    _debug(monkeypatch, True)
    monkeypatch.delenv("LOCALM_MODE", raising=False)

    def _boom():
        raise RuntimeError("config unreadable")

    monkeypatch.setattr("localm.config.load_config", _boom)
    assert debuglog.debug_content_enabled() is False   # fail-safe: no content
