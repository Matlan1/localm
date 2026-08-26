# SPDX-License-Identifier: AGPL-3.0-or-later
"""POST /api/models/scan: the workdir-override / dry-run guided-import surface.

- SECURITY: scanning is equivalent capability to the host file/folder browser
  (/api/fs/dirs) - it walks a host directory and writes the absolute paths it
  finds into registry.json - so EVERY form of this call requires host filesystem
  access (owner / open mode / a key minted with fs_access="host"). A
  models:write-only key that lacks host fs access must not reach it.
- `dry_run` returns a preview shape and must never register anything.

The BODYLESS form is gated too. A route gating on
`if workdir: require_fs_host(request)` would let the bodyless form skip the
check entirely and scan `get_comfy_workdir()` - and `comfy_workdir` is a plain
Widget.FOLDER with no admin_only, settable by any config:write key. A key with
fs_access="none", the very configuration require_fs_host exists to constrain,
could then point the scanner at any folder on the server and plant arbitrary
absolute or UNC paths in registry.json, which every consumer re-stats on every
launch. Where the folder NAME comes from (a request body or the config file) is
not what makes a scan safe; host filesystem reach is.
"""

import time
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from localm import scopes as S
from localm.plugins.gui.web import attach_gui


def _wait_job(app, job_id, timeout=10.0):
    """Poll a real (non-dry-run) scan's background job until it leaves
    'running'. A real scan is job-based (see gui_scan_models), so a test starts
    the job, waits here, then reads added/skipped/method off its final progress
    event rather than off the POST response."""
    job = app.state.jobs.get(job_id)
    assert job is not None, f"job {job_id} was never registered"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if job.status != "running":
            return job
        time.sleep(0.01)
    raise AssertionError(f"job {job_id} never finished: status={job.status}")


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
    """The form the Scan button sends: POST with no body key at all. It is
    gated on host fs access like every other form."""

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
        ordinary configured-workdir scan. A REAL scan (dry_run absent/false) is
        job-based, so the immediate response is a job id; wait for it and read
        "method" off the final progress event."""
        with TestClient(scan_app) as c:
            r = c.post("/api/models/scan", headers=_hdr(_host_writer_key()))
            assert r.status_code == 200, r.text
            body = r.json()
            assert "dry_run" not in body
            assert "job_id" in body
            job = _wait_job(scan_app, body["job_id"])
            assert job.status == "done", job._history
            # No comfy_workdir configured in this throwaway home - a legitimate
            # "none (...)" result, not a failure.
            assert job._last_progress["method"].startswith("none (comfy_workdir not configured)")
            assert job._last_progress["added"] == 0
            assert job._last_progress["skipped"] == 0

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
            # the file unregistered, proving the preview above added nothing.
            # dry_run=False is job-based: wait for it and read the count off its
            # final progress event.
            with patch("requests.get", side_effect=Exception("offline")):
                r2 = c.post("/api/models/scan", headers=_hdr(_host_writer_key()),
                           json={"workdir": str(comfy_tree), "dry_run": False})
            assert r2.status_code == 200, r2.text
            job = _wait_job(scan_app, r2.json()["job_id"])
            assert job.status == "done", job._history
            assert job._last_progress["added"] == 1, "the dry run must not have registered it already"


@pytest.fixture
def comfy_tree_multi(tmp_path):
    """Three files across three SUBFOLDER_MAPPING conventions - big enough to
    prove a real "N of M" progression, not just a single 1-of-1 tick."""
    d = tmp_path / "comfy"
    for sub, fname in (("unet", "flux.safetensors"),
                       ("clip", "clip_l.safetensors"),
                       ("loras", "style.safetensors")):
        sub_dir = d / "models" / sub
        sub_dir.mkdir(parents=True)
        (sub_dir / fname).write_bytes(b"DATA")
    return d


class TestRealScanReportsProgress:
    """A real (non-dry-run) scan is job-based and must report a real
    "registering model N of M" count as it goes, not just a start/end pair."""

    def test_progress_events_carry_an_increasing_done_of_a_fixed_total(
            self, scan_app, comfy_tree_multi):
        with TestClient(scan_app) as c:
            with patch("requests.get", side_effect=Exception("offline")):
                r = c.post("/api/models/scan", headers=_hdr(_host_writer_key()),
                          json={"workdir": str(comfy_tree_multi), "dry_run": False})
            assert r.status_code == 200, r.text
            job = _wait_job(scan_app, r.json()["job_id"])
            assert job.status == "done", job._history

            registering = [e for e in job._history
                          if e.get("type") == "progress" and e.get("phase") == "registering"]
            assert len(registering) == 3, registering
            # done climbs 1, 2, 3 against a FIXED total of 3, and every event
            # names the file it is registering - the actual "N of M: name"
            # data the GUI renders instead of a silent wait.
            assert [e["done"] for e in registering] == [1, 2, 3]
            assert all(e["total"] == 3 for e in registering)
            assert all(e.get("name") for e in registering)
            assert {e["name"] for e in registering} == {
                "flux.safetensors", "clip_l.safetensors", "style.safetensors"}

    def test_final_progress_event_carries_added_skipped_method(
            self, scan_app, comfy_tree_multi):
        """The final progress event (phase "done") carries added/skipped/method,
        so the GUI can build its scanResultMessage() toast from it."""
        with TestClient(scan_app) as c:
            with patch("requests.get", side_effect=Exception("offline")):
                r = c.post("/api/models/scan", headers=_hdr(_host_writer_key()),
                          json={"workdir": str(comfy_tree_multi), "dry_run": False})
            job = _wait_job(scan_app, r.json()["job_id"])
            assert job.status == "done", job._history
            final = job._last_progress
            assert final["phase"] == "done"
            assert final["added"] == 3
            assert final["skipped"] == 0
            assert final["method"] in ("folder-walk",
                                       "hybrid (folder-walk + /object_info)")

    def test_a_genuine_scan_failure_still_marks_the_job_failed(self, scan_app, comfy_tree_multi):
        """An exception during registration must not report a false "done":
        the job fails.

        The scan runs in a JobManager background thread (start_fn), so the patch
        must stay active for the whole wait, not just the POST: the POST only
        spawns the thread and returns immediately, and unpatching right after it
        races the thread."""
        with TestClient(scan_app) as c:
            with patch("requests.get", side_effect=Exception("offline")), \
                 patch("localm.model_manager.scan._existing_registered_paths",
                       side_effect=RuntimeError("boom")):
                r = c.post("/api/models/scan", headers=_hdr(_host_writer_key()),
                          json={"workdir": str(comfy_tree_multi), "dry_run": False})
                assert r.status_code == 200, r.text   # the job itself always starts
                job = _wait_job(scan_app, r.json()["job_id"])
            assert job.status == "failed", job._history
            lines = [e["text"] for e in job._history if e.get("type") == "line"]
            assert any("boom" in t for t in lines), lines
