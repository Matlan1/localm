# SPDX-License-Identifier: AGPL-3.0-or-later
"""PluginHost.mount_surface_assets returns None and keeps serving the rest of
the plugin for both of its failure causes - 'assets_dir missing on disk' and
'assets_dir escapes the plugin dir' - and debug-logs which one it was, so a
manifest typo like `assets_dir = '../x'` is not an undiagnosed SPA 404."""

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
        assert host.mount_surface_assets() is None      # best-effort
    assert any("does-not-exist" in r.message and "not mounted" in r.message
               for r in caplog.records), "the missing assets_dir must be logged"


def test_boundary_escape_is_logged_with_its_real_cause(tmp_path, caplog):
    # A manifest typo that escapes the plugin dir: the guard rejects it (nothing
    # mounted) and the log names the escape, distinct from a missing dir.
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
    assert prefix == "/plugins/probe"                   # happy path
    assert not any("not mounted" in r.message for r in caplog.records)
