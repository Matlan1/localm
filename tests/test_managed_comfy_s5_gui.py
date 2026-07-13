# SPDX-License-Identifier: AGPL-3.0-or-later
"""STAGE S5 (GUI-button slice) for the localm-managed ComfyUI feature.

The GUI backend to set up + manage localm's OWN ComfyUI: HTTP routes that dispatch
provisioning as a progress-streamed JOB (never blocking the request), read the
installed status, and remove the managed instance. Provisioning itself (S2/S3) is
NOT exercised here - the heavy `localm comfy setup` CLI is stubbed so the endpoint
test is fast and asserts only the DISPATCH contract (a job id is returned, going to
the existing setup entry point) plus the status read and the removal.

Design + locked decisions: dev-notes/DESIGN-localm-managed-comfyui-2026-07-08.md
(decision 8: opt-in `localm comfy setup` + a GUI button, off by default). Builds on
S1 (#483) helpers in localm/media/managed_comfy.py and the S2 (#486) copy entry
point; this slice adds ONLY the GUI/HTTP surface and calls those entry points.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import localm.config as cfg
from localm.media import managed_comfy as mc
from localm.plugins.gui import jobs as gui_jobs
from localm.plugins.gui.web import attach_gui


# --------------------------------------------------------------------------- #
#  Isolation: a throwaway LOCALM_HOME wired through both the lazy home_dir()   #
#  AND the import-frozen config paths, so load_config and managed_comfy path   #
#  resolution agree on the same tmp dir (see S1's test + memory note "Test     #
#  home isolation (import-time)").                                             #
# --------------------------------------------------------------------------- #
@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / ".localm"
    h.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("LOCALM_HOME", str(h))
    monkeypatch.setattr(cfg, "HOME_DIR", h)
    monkeypatch.setattr(cfg, "MODELS_DIR", h / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", h / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", h / "registry.json")
    monkeypatch.delenv("FLUX_API_URL", raising=False)
    return h


def _install_managed() -> mc.ManagedComfyPaths:
    """Minimal on-disk layout that makes is_managed_comfy_installed() true, using the
    module's OWN path accessors so the test is platform-agnostic (the venv
    interpreter path differs on Windows vs POSIX)."""
    paths = mc.managed_comfy_paths()
    paths.main_py.parent.mkdir(parents=True, exist_ok=True)
    paths.main_py.write_text("# stand-in for ComfyUI main.py\n", encoding="utf-8")
    paths.venv_python.parent.mkdir(parents=True, exist_ok=True)
    paths.venv_python.write_text("", encoding="utf-8")
    assert mc.is_managed_comfy_installed()
    return paths


@pytest.fixture
def app(home):
    """A FastAPI app with the GUI routes attached (open mode = loopback owner, so
    the CONFIG_READ/CONFIG_WRITE gates pass without a key). Depends on `home` so
    LOCALM_HOME is set before attach_gui builds the shared services."""
    a = FastAPI()
    attach_gui(a, self_url="http://127.0.0.1:9/v1",
               switch_model=lambda name: {"status": "loaded", "model": name},
               active_model=lambda: "model-a")
    return a


class _FakeJob:
    def __init__(self, jid: str) -> None:
        self.id = jid


@pytest.fixture
def no_subprocess(monkeypatch):
    """Stub JobManager.start_cli so the setup route DISPATCHES a job without spawning
    a real `python -m localm comfy setup` subprocess (a multi-GB install). Records the
    calls so the test can assert the dispatch targeted the comfy setup entry point.
    Patched on the CLASS so the instance attach_gui already created uses it."""
    calls = []

    def _fake_start_cli(self, kind, cli_args, **kw):
        calls.append({"kind": kind, "args": list(cli_args), "kw": kw})
        return _FakeJob("job123")

    monkeypatch.setattr(gui_jobs.JobManager, "start_cli", _fake_start_cli)
    return calls


# --------------------------------------------------------------------------- #
#  POST /api/comfy/setup : starts a JOB (job id), does NOT block               #
# --------------------------------------------------------------------------- #

def test_setup_dispatches_job_and_does_not_block(home, app, no_subprocess):
    with TestClient(app) as client:
        r = client.post("/api/comfy/setup")
    assert r.status_code == 200, r.text
    assert r.json().get("job_id") == "job123"
    # Dispatched exactly one job, to the EXISTING `localm comfy setup` CLI entry
    # point (not a re-implementation of provisioning here).
    assert len(no_subprocess) == 1, no_subprocess
    args = no_subprocess[0]["args"]
    assert args[:2] == ["comfy", "setup"], args
    # Default is a clean start: the safe non-interactive default is not to copy
    # the user's custom nodes (decision 3).
    assert "--no-custom-nodes" in args


def test_setup_copy_custom_nodes_flag_is_forwarded(home, app, no_subprocess):
    with TestClient(app) as client:
        r = client.post("/api/comfy/setup", params={"copy_custom_nodes": "true"})
    assert r.status_code == 200, r.text
    assert "--copy-custom-nodes" in no_subprocess[0]["args"]


def test_setup_conflicts_when_already_installed(home, app, no_subprocess):
    """A managed instance already exists -> 409, and NO job is dispatched (do not
    silently clobber; the user removes it first, mirroring provision_by_copy)."""
    _install_managed()
    with TestClient(app) as client:
        r = client.post("/api/comfy/setup")
    assert r.status_code == 409, r.text
    assert no_subprocess == []


# --------------------------------------------------------------------------- #
#  GET /api/comfy/managed-status : reflects is_managed_comfy_installed()        #
# --------------------------------------------------------------------------- #

def test_status_not_installed_offers_setup(home, app):
    with TestClient(app) as client:
        r = client.get("/api/comfy/managed-status")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["installed"] is False
    assert body.get("path") in (None, "")
    # Off by default: not routing to a managed instance (comfy_target defaults
    # to "own" but is inert until an instance is actually installed).
    assert body["managed_active"] is False
    assert body["target"] == "own"


def test_status_installed_reports_path(home, app):
    paths = _install_managed()
    with TestClient(app) as client:
        r = client.get("/api/comfy/managed-status")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["installed"] is True
    assert body["path"] == str(paths.root)


# --------------------------------------------------------------------------- #
#  POST /api/comfy/remove : removes the managed dir                            #
# --------------------------------------------------------------------------- #

def test_remove_deletes_the_managed_dir(home, app):
    paths = _install_managed()
    assert mc.is_managed_comfy_installed() is True
    with TestClient(app) as client:
        r = client.post("/api/comfy/remove")
    assert r.status_code == 200, r.text
    assert mc.is_managed_comfy_installed() is False
    assert not paths.root.exists()


def test_remove_is_honest_noop_when_nothing_installed(home, app):
    with TestClient(app) as client:
        r = client.post("/api/comfy/remove")
    assert r.status_code == 200, r.text
    # Honest: report that nothing was there, not a claimed success of a real delete
    # (rule 5: do not dress up a no-op as a completed removal).
    assert r.json().get("status") == "noop"
