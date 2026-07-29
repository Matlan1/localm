# SPDX-License-Identifier: AGPL-3.0-or-later
"""POST /api/models/scan: the workdir-override / dry-run guided-import surface.

- SECURITY: scanning is equivalent capability to the host file/folder browser
  (/api/fs/dirs) - it walks a host directory and writes the absolute paths it
  finds into registry.json - so EVERY form of this call requires host filesystem
  access (owner / open mode / a key minted with fs_access="host"). A
  models:write-only key that lacks host fs access must not reach it.
- `dry_run` returns a preview shape and must never register anything.

REVISED CONTRACT (CodeQL WS2). This file used to assert the opposite for the
bodyless form: "the OLD Scan button sends a POST with NO body at all. That call
path must still need only models:write, exactly as before - this is the one
invariant the whole feature must not break." That invariant was the bug. The
route gated on `if workdir: require_fs_host(request)`, so the bodyless form
skipped the check entirely and scanned `get_comfy_workdir()` - and
`comfy_workdir` is a plain Widget.FOLDER with no admin_only, settable by any
config:write key. A key with fs_access="none", the very configuration
require_fs_host exists to constrain, could therefore point the scanner at any
folder on the server and plant arbitrary absolute or UNC paths in registry.json,
which every consumer then re-stats on every launch. Where the folder NAME came
from (a request body or the config file) was never the thing that made the scan
safe; host filesystem reach is. The tests below now encode that.
"""

from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from localm import scopes as S
from localm.plugins.gui.web import attach_gui


@pytest.fixture
def scan_app(tmp_path, monkeypatch):
    """Full stack (engine + GUI) on a throwaway home. No owner key by default,
    so a minted key's own scopes/fs_access govern what it can reach."""
    home = tmp_path / ".localm"
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    import localm.config as _cfg
    monkeypatch.setattr(_cfg, "HOME_DIR", home)
    monkeypatch.setattr(_cfg, "MODELS_DIR", home / "models")
    monkeypatch.setattr(_cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(_cfg, "REGISTRY_FILE", home / "registry.json")
    from localm.plugins.engine import attach_engine
    app = FastAPI()
    attach_engine(app)
    attach_gui(app, self_url="http://127.0.0.1:9/v1",
               switch_model=lambda name: None,
               active_model=lambda: "model-a")
    return app


def _hdr(key):
    return {"Authorization": f"Bearer {key}"}


def _writer_key():
    """models:write only, default fs_access ('none') - must not reach ANY form of
    the scan, configured-workdir or override."""
    from localm import auth
    return auth.create_key("writer", [S.MODELS_WRITE])["key"]


def _host_writer_key():
    """models:write AND host filesystem access - the level a scan requires."""
    from localm import auth
    return auth.create_key("host-writer", [S.MODELS_WRITE], fs_access="host")["key"]


@pytest.fixture
def comfy_tree(tmp_path):
    """A minimal ComfyUI-shaped folder: one unet file, nothing else."""
    d = tmp_path / "comfy"
    unet = d / "models" / "unet"
    unet.mkdir(parents=True)
    (unet / "flux.safetensors").write_bytes(b"UNET")
    return d


class TestBodylessFormAlsoRequiresHostFsAccess:
    """The form the old Scan button sends: POST with no body key at all. It used
    to skip the gate outright; it does not any more."""

    def test_bodyless_post_403s_for_a_models_write_only_key(self, scan_app):
        with TestClient(scan_app) as c:
            r = c.post("/api/models/scan", headers=_hdr(_writer_key()))
            assert r.status_code == 403, r.text

    def test_empty_json_body_is_equivalent(self, scan_app):
        with TestClient(scan_app) as c:
            r = c.post("/api/models/scan", headers=_hdr(_writer_key()), json={})
            assert r.status_code == 403, r.text

    def test_bodyless_post_succeeds_for_a_host_fs_access_key(self, scan_app):
        """The gate is fs_access, not the body: a host-fs key still gets the
        ordinary configured-workdir scan, unchanged."""
        with TestClient(scan_app) as c:
            r = c.post("/api/models/scan", headers=_hdr(_host_writer_key()))
            assert r.status_code == 200, r.text
            body = r.json()
            # No comfy_workdir configured in this throwaway home - a legitimate
            # "none (...)" result, not a failure, and NOT the dry-run shape.
            assert "dry_run" not in body
            assert body["method"].startswith("none (comfy_workdir not configured)")

    def test_bodyless_post_succeeds_in_open_mode(self, scan_app):
        """No key configured at all -> loopback owner -> host access implied, so
        the GUI's own Scan button is unaffected."""
        with TestClient(scan_app) as c:
            assert c.post("/api/models/scan").status_code == 200

    def test_the_scope_gate_still_applies_first(self, scan_app):
        """A key with no models:* at all is refused on scope, as before."""
        from localm import auth
        narrow = auth.create_key("narrow", [S.MCP])["key"]
        with TestClient(scan_app) as c:
            assert c.post("/api/models/scan", headers=_hdr(narrow)).status_code == 403


class TestWorkdirOverrideRequiresHostFsAccess:
    def test_workdir_override_403s_for_a_models_write_only_key(self, scan_app, comfy_tree):
        with TestClient(scan_app) as c:
            r = c.post("/api/models/scan", headers=_hdr(_writer_key()),
                      json={"workdir": str(comfy_tree), "dry_run": True})
            assert r.status_code == 403, r.text

    def test_workdir_override_succeeds_for_a_host_fs_access_key(self, scan_app, comfy_tree):
        with TestClient(scan_app) as c:
            with patch("requests.get", side_effect=Exception("offline")):
                r = c.post("/api/models/scan", headers=_hdr(_host_writer_key()),
                          json={"workdir": str(comfy_tree), "dry_run": True})
            assert r.status_code == 200, r.text
            body = r.json()
            assert body["dry_run"] is True
            assert body["counts"] == {"diffusion-unet": 1}
            assert body["already_registered"] == 0
            assert body["total_new"] == 1

    def test_workdir_override_succeeds_in_open_mode(self, scan_app, comfy_tree):
        """No key configured at all -> loopback owner -> host access implied."""
        with TestClient(scan_app) as c:
            with patch("requests.get", side_effect=Exception("offline")):
                r = c.post("/api/models/scan",
                          json={"workdir": str(comfy_tree), "dry_run": True})
            assert r.status_code == 200, r.text
            assert r.json()["counts"] == {"diffusion-unet": 1}

    def test_workdir_override_succeeds_for_the_owner_key(self, scan_app, comfy_tree, monkeypatch):
        monkeypatch.setenv("LOCALM_API_KEY", "ownersecret")
        with TestClient(scan_app) as c:
            with patch("requests.get", side_effect=Exception("offline")):
                r = c.post("/api/models/scan", headers=_hdr("ownersecret"),
                          json={"workdir": str(comfy_tree), "dry_run": True})
            assert r.status_code == 200, r.text


class TestDryRunNeverRegisters:
    def test_dry_run_leaves_the_registry_empty(self, scan_app, comfy_tree):
        with TestClient(scan_app) as c:
            with patch("requests.get", side_effect=Exception("offline")):
                r = c.post("/api/models/scan", headers=_hdr(_host_writer_key()),
                          json={"workdir": str(comfy_tree), "dry_run": True})
            assert r.status_code == 200, r.text
            # A second, real (non-dry-run) scan of the SAME tree must still find
            # the file unregistered - proving the preview above added nothing.
            with patch("requests.get", side_effect=Exception("offline")):
                r2 = c.post("/api/models/scan", headers=_hdr(_host_writer_key()),
                           json={"workdir": str(comfy_tree), "dry_run": False})
            assert r2.status_code == 200, r2.text
            assert r2.json()["added"] == 1, "the dry run must not have registered it already"
