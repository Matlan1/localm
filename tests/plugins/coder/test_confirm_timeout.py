# SPDX-License-Identifier: AGPL-3.0-or-later
"""
_confirm_timeout() (localm/plugins/coder/sessions.py) reads the
"coder_confirm_timeout" config key, documented in settings_schema.py as
"0 = wait forever".

A configured 0 must survive rather than being replaced by the 600s default
(`x or default` treats 0 as falsy), and "wait forever" must reach
`threading.Event.wait` as timeout=None: wait(timeout=0) is a non-blocking
poll, not an indefinite wait.
"""

from localm.plugins.coder.sessions import _CONFIRM_TIMEOUT_S, _confirm_timeout


def test_unset_falls_back_to_default(monkeypatch):
    monkeypatch.setattr("localm.config.load_config", lambda: {})
    assert _confirm_timeout() == float(_CONFIRM_TIMEOUT_S)


def test_configured_zero_means_wait_forever(monkeypatch):
    # None, not 0.0, so Event.wait(timeout=...) blocks indefinitely.
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
