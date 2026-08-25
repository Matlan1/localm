# SPDX-License-Identifier: AGPL-3.0-or-later
"""HONESTY (resume audit 2026-07-02): PluginHost.mount_surface_assets swallowed BOTH the 'assets_dir missing on disk' and the 'assets_dir escapes the plugin dir' ValueError into a bare `return None` with a comment that named only the first case and no log."""

import logging

from fastapi import FastAPI

from localm.plugins.contract import PluginSpec, Surface
from localm.plugins.engine import PluginHost


def _host(tmp_path, assets_dir: str) -> PluginHost:
    spec = PluginSpec(name="probe", path=str(tmp_path),
                      surface=Surface(assets_dir=assets_dir, client_entry="x.js"))
    return PluginHost(FastAPI(), None, spec)


def test_missing_assets_dir_is_logged_not_silent(tmp_path, caplog):
    host = _host(tmp_path, "does-not-exist")
    with caplog.at_level(logging.DEBUG, logger="localm.plugins"):
        assert host.mount_surface_assets() is None      # unchanged: best-effort
    assert any("does-not-exist" in r.message and "not mounted" in r.message
               for r in caplog.records), "the missing assets_dir must be logged"


def test_boundary_escape_is_logged_with_its_real_cause(tmp_path, caplog):
    # A manifest typo that escapes the plugin dir: the security guard rejects it
    # (nothing mounted) AND the reason is now surfaced, not misread as 'missing'.
    host = _host(tmp_path, "../../../etc")
    with caplog.at_level(logging.DEBUG, logger="localm.plugins"):
        assert host.mount_surface_assets() is None
    logged = " ".join(r.message for r in caplog.records)
    assert "not mounted" in logged and "escapes" in logged, \
        "a boundary-escape must be logged and distinguished from a missing dir"


def test_present_assets_dir_mounts_and_does_not_warn(tmp_path, caplog):
    (tmp_path / "static").mkdir()
    (tmp_path / "static" / "x.js").write_text("export function register(){}",
                                              encoding="utf-8")
    host = _host(tmp_path, "static")
    with caplog.at_level(logging.DEBUG, logger="localm.plugins"):
        prefix = host.mount_surface_assets()
    assert prefix == "/plugins/probe"                   # happy path unaffected
    assert not any("not mounted" in r.message for r in caplog.records)
