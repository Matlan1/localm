# SPDX-License-Identifier: AGPL-3.0-or-later
"""_confirm_timeout() (localm/plugins/coder/sessions.py) reads the 'coder_confirm_timeout' config key, documented in settings_schema.py as '0 = wait forever'."""

from localm.plugins.coder.sessions import _CONFIRM_TIMEOUT_S, _confirm_timeout


def test_unset_falls_back_to_default(monkeypatch):
    monkeypatch.setattr("localm.config.load_config", lambda: {})
    assert _confirm_timeout() == float(_CONFIRM_TIMEOUT_S)


def test_configured_zero_means_wait_forever(monkeypatch):
    # None (not 0.0) so it can be passed straight to Event.wait(timeout=...)
    # and actually block indefinitely rather than returning instantly.
    monkeypatch.setattr("localm.config.load_config",
                        lambda: {"coder_confirm_timeout": 0})
    assert _confirm_timeout() is None


def test_configured_positive_value_is_used_verbatim(monkeypatch):
    monkeypatch.setattr("localm.config.load_config",
                        lambda: {"coder_confirm_timeout": 30})
    assert _confirm_timeout() == 30.0


def test_config_load_failure_falls_back_to_default(monkeypatch):
    def _boom():
        raise RuntimeError("config unreadable")
    monkeypatch.setattr("localm.config.load_config", _boom)
    assert _confirm_timeout() == float(_CONFIRM_TIMEOUT_S)
