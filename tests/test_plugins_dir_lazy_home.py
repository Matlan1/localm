# SPDX-License-Identifier: AGPL-3.0-or-later
"""AUD-PLUGINSDIR regression: plugins_dir() must resolve the CURRENT effective
home directory, not the one localm.config froze into the module-level HOME_DIR
at import time (which happens once, long before any test-isolation fixture
runs)."""

from __future__ import annotations


def test_plugins_dir_follows_a_later_localm_home_change(monkeypatch, tmp_path):
    import localm.config as cfg
    from localm.plugins.loader import plugins_dir

    monkeypatch.setenv("LOCALM_HOME", str(tmp_path))

    assert plugins_dir() == tmp_path / "plugins"
    assert plugins_dir() != cfg.HOME_DIR / "plugins"


def test_plugins_dir_matches_the_lazy_home_dir_helper(monkeypatch, tmp_path):
    import localm.config as cfg
    from localm.plugins.loader import plugins_dir

    monkeypatch.setenv("LOCALM_HOME", str(tmp_path))

    assert plugins_dir() == cfg.home_dir() / "plugins"
