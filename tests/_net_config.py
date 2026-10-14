# SPDX-License-Identifier: AGPL-3.0-or-later
"""Test helper: make config reads return a chosen dict."""


def set_config(monkeypatch, cfg):
    """Make config reads return *cfg* (a dict, or a zero-argument callable that
    returns one) as a config.json that was read successfully. Patches both
    ``load_config`` and ``load_config_checked``, because the network policy and
    require_auth read the config through the checked loader."""
    get = cfg if callable(cfg) else (lambda: cfg)
    monkeypatch.setattr("localm.config.load_config", get)
    monkeypatch.setattr("localm.config.load_config_checked", lambda: (get(), True))
