# SPDX-License-Identifier: AGPL-3.0-or-later
"""plugins_dir() re-resolves LOCALM_HOME at call time, matching home_dir(); it
must not read the module-level HOME_DIR, which localm.config resolves once at
first import (during test collection, before any per-test override runs)."""

from __future__ import annotations


def test_plugins_dir_tracks_a_later_localm_home_change(monkeypatch, tmp_path):
    from localm.plugins.loader import plugins_dir

    before = plugins_dir()
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path / "repointed"))
    after = plugins_dir()

    assert after == tmp_path / "repointed" / "plugins"
    assert after != before


def test_plugins_dir_matches_the_lazy_home_dir_helper(monkeypatch, tmp_path):
    import localm.config as cfg
    from localm.plugins.loader import plugins_dir

    monkeypatch.setenv("LOCALM_HOME", str(tmp_path / "repointed"))

    assert plugins_dir() == cfg.home_dir() / "plugins"
