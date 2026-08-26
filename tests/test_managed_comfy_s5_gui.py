# SPDX-License-Identifier: AGPL-3.0-or-later
"""STAGE S5 (GUI-button slice) for the localm-managed ComfyUI feature.

The GUI backend to set up + manage localm's OWN ComfyUI: HTTP routes that dispatch
provisioning as a progress-streamed JOB (never blocking the request), read the
installed status, and remove the managed instance. Provisioning itself (S2/S3) is
NOT exercised here - the heavy `localm comfy setup` CLI is stubbed so the endpoint
test is fast and asserts only the DISPATCH contract (a job id is returned, going to
the existing setup entry point) plus the status read and the removal.

The feature is opt-in (`localm comfy setup` plus a GUI button) and off by default.
This slice adds ONLY the GUI/HTTP surface and calls the S1 helpers in
localm/media/managed_comfy.py and the S2 copy entry point.
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
#  resolution agree on the same tmp dir.                                       #
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
    interpreter path differs on Windows vs POSIX). Includes the completion
    marker: main.py + venv alone means "still installing"."""
    from localm.media.managed_comfy_provision import MARKER_FILENAME
    paths = mc.managed_comfy_paths()
    paths.main_py.parent.mkdir(parents=True, exist_ok=True)
    paths.main_py.write_text("# stand-in for ComfyUI main.py\n", encoding="utf-8")
    paths.venv_python.parent.mkdir(parents=True, exist_ok=True)
    paths.venv_python.write_text("", encoding="utf-8")
    (paths.root / MARKER_FILENAME).write_text("{}", encoding="utf-8")
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
    # Default is a clean start: the safe non-interactive default does not copy
    # the user's custom nodes.
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


def test_setup_refuses_a_second_concurrent_setup(home, app, no_subprocess, monkeypatch):
    """Two overlapping "Set up" clicks (e.g. two browser tabs) must not double-launch
    the setup CLI onto the same target checkout. The on-disk existence check alone
    cannot catch this - the checkout directory does not exist yet for the first few
    moments of a fresh setup - so this needs its OWN has_running guard, same shape as
    comfy_update/comfy_repair's existing guards against a concurrent comfy-setup."""
    monkeypatch.setattr(gui_jobs.JobManager, "has_running",
                        lambda self, kind: kind == "comfy-setup")
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
    # Honest: report that nothing was there, not a claimed success of a real
    # delete.
    assert r.json().get("status") == "noop"


# --------------------------------------------------------------------------- #
#  GET /api/comfy/managed-status : the richer state field                      #
#                                                                             #
#  is_managed_comfy_installed()=False does NOT mean the checkout dir is absent: #
#  an abandoned setup attempt leaves one behind, and state tells the two apart. #
# --------------------------------------------------------------------------- #

def test_status_state_not_installed_when_nothing_exists(home, app):
    with TestClient(app) as client:
        r = client.get("/api/comfy/managed-status")
    assert r.json()["state"] == "not_installed"


def test_status_state_installed_when_genuinely_installed(home, app):
    _install_managed()
    with TestClient(app) as client:
        r = client.get("/api/comfy/managed-status")
    assert r.json()["state"] == "installed"


def test_status_state_corrupt_when_checkout_exists_but_incomplete(home, app):
    """The exact dead-end scenario: main.py exists (checkout started) but the
    completion marker never got written (an abandoned attempt) - no setup job
    is running. is_managed_comfy_installed() correctly says False; the new
    state must distinguish this from a clean "not set up"."""
    paths = mc.managed_comfy_paths()
    paths.main_py.parent.mkdir(parents=True, exist_ok=True)
    paths.main_py.write_text("# partial checkout, no marker\n", encoding="utf-8")
    assert mc.is_managed_comfy_installed() is False
    with TestClient(app) as client:
        r = client.get("/api/comfy/managed-status")
    body = r.json()
    assert body["state"] == "corrupt"
    assert body["path"] == str(paths.root)   # surfaced so Repair/logs can reference it


def test_status_state_installing_when_a_setup_job_is_running(home, app, monkeypatch):
    """A checkout in progress (root exists, no marker yet) while its OWN setup
    job is genuinely still running must read as "installing", never "corrupt" -
    the whole point is not to offer Repair out from under a live install."""
    paths = mc.managed_comfy_paths()
    paths.main_py.parent.mkdir(parents=True, exist_ok=True)
    paths.main_py.write_text("# mid-install\n", encoding="utf-8")
    monkeypatch.setattr(gui_jobs.JobManager, "has_running",
                        lambda self, kind: kind == "comfy-setup")
    with TestClient(app) as client:
        r = client.get("/api/comfy/managed-status")
    assert r.json()["state"] == "installing"


# --------------------------------------------------------------------------- #
#  POST /api/comfy/repair : clears an INCOMPLETE install, never a real one     #
# --------------------------------------------------------------------------- #

def test_repair_refuses_when_genuinely_installed(home, app, no_subprocess):
    """Never repair-away a real install - Remove exists for that, deliberately
    a separate, more visible action."""
    _install_managed()
    with TestClient(app) as client:
        r = client.post("/api/comfy/repair")
    assert r.status_code == 409, r.text
    assert no_subprocess == []


def test_repair_refuses_when_nothing_to_repair(home, app, no_subprocess):
    with TestClient(app) as client:
        r = client.post("/api/comfy/repair")
    assert r.status_code == 409, r.text
    assert no_subprocess == []


def test_repair_refuses_when_a_setup_is_already_running(home, app, no_subprocess, monkeypatch):
    paths = mc.managed_comfy_paths()
    paths.main_py.parent.mkdir(parents=True, exist_ok=True)
    paths.main_py.write_text("# mid-install\n", encoding="utf-8")
    monkeypatch.setattr(gui_jobs.JobManager, "has_running",
                        lambda self, kind: kind == "comfy-setup")
    with TestClient(app) as client:
        r = client.post("/api/comfy/repair")
    assert r.status_code == 409, r.text
    assert no_subprocess == []
    assert paths.root.exists(), "must not clear a live install's checkout"


def test_repair_clears_the_incomplete_checkout_and_dispatches_a_fresh_setup(
        home, app, no_subprocess):
    paths = mc.managed_comfy_paths()
    paths.main_py.parent.mkdir(parents=True, exist_ok=True)
    paths.main_py.write_text("# abandoned partial checkout\n", encoding="utf-8")
    assert paths.root.exists()
    with TestClient(app) as client:
        r = client.post("/api/comfy/repair")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body.get("job_id") == "job123"
    assert not paths.root.exists(), "the incomplete checkout must be cleared"
    # Same dispatch contract as a normal Set up.
    assert len(no_subprocess) == 1
    assert no_subprocess[0]["args"][:2] == ["comfy", "setup"]


def test_repair_never_touches_the_managed_models_folder(home, app, no_subprocess):
    """The models dir is a SIBLING of the checkout (comfyui-models vs comfyui),
    never inside it - a repair must leave any already-downloaded models alone."""
    paths = mc.managed_comfy_paths()
    paths.main_py.parent.mkdir(parents=True, exist_ok=True)
    paths.main_py.write_text("# abandoned partial checkout\n", encoding="utf-8")
    paths.models_dir.mkdir(parents=True, exist_ok=True)
    marker = paths.models_dir / "a-real-downloaded-model.safetensors"
    marker.write_text("not actually a model, just a marker file", encoding="utf-8")
    with TestClient(app) as client:
        r = client.post("/api/comfy/repair")
    assert r.status_code == 200, r.text
    assert marker.exists(), "models must survive a repair"


# --------------------------------------------------------------------------- #
#  POST /api/comfy/update                                                      #
# --------------------------------------------------------------------------- #

def _install_managed_at(commit: str, *, git: bool = True) -> mc.ManagedComfyPaths:
    """An installed managed ComfyUI whose marker records *commit*, optionally without
    a .git dir (the non-git copy-fallback install, which cannot take a pinned update)."""
    import json as _json
    from localm.media.managed_comfy_provision import MARKER_FILENAME
    paths = _install_managed()
    (paths.root / MARKER_FILENAME).write_text(
        _json.dumps({"commit": commit, "comfyui_version": "v-installed"}),
        encoding="utf-8")
    if git:
        (paths.root / ".git").mkdir(parents=True, exist_ok=True)
    return paths


def test_update_dispatches_the_cli_job_and_does_not_block(home, app, no_subprocess):
    _install_managed_at("oldcommit", git=True)
    with TestClient(app) as client:
        r = client.post("/api/comfy/update")
    assert r.status_code == 200, r.text
    assert r.json().get("job_id") == "job123"
    assert len(no_subprocess) == 1, no_subprocess
    # Dispatched to the EXISTING CLI entry point, not a re-implementation: the
    # rollback-on-failure contract lives in update_managed_comfy(), which stays
    # its single owner.
    assert no_subprocess[0]["args"] == ["comfy", "update"], no_subprocess[0]["args"]
    assert no_subprocess[0]["kind"] == "comfy-update"


def test_update_forwards_reinstall_requirements(home, app, no_subprocess):
    """Off by default (a partial pip upgrade is not exactly rollback-able), but it MUST
    be reachable: when a pin advances across a dependency change, an update without it
    leaves a checkout that moved without its new deps."""
    _install_managed_at("oldcommit")
    with TestClient(app) as client:
        r = client.post("/api/comfy/update", params={"reinstall_requirements": "true"})
    assert r.status_code == 200, r.text
    assert no_subprocess[0]["args"] == ["comfy", "update", "--reinstall-requirements"]


def test_update_defaults_to_not_reinstalling_requirements(home, app, no_subprocess):
    _install_managed_at("oldcommit")
    with TestClient(app) as client:
        client.post("/api/comfy/update")
    assert "--reinstall-requirements" not in no_subprocess[0]["args"]


def test_update_forwards_commit(home, app, no_subprocess):
    """`--commit` is an advanced/testing knob to update to a specific ComfyUI
    commit instead of the shipped pin. Not validated here - update_managed_comfy()
    owns checking out whatever it is given and reports a bad ref honestly through
    the job's own output, the same split as the non-git refusal above."""
    _install_managed_at("oldcommit")
    with TestClient(app) as client:
        r = client.post("/api/comfy/update", params={"commit": "abc123def"})
    assert r.status_code == 200, r.text
    assert no_subprocess[0]["args"] == ["comfy", "update", "--commit", "abc123def"]


def test_update_defaults_to_the_shipped_pin(home, app, no_subprocess):
    """No commit entered -> no --commit at all, so update_managed_comfy() falls
    back to its own COMFYUI_PINNED_COMMIT default rather than being passed an
    empty string as a target ref."""
    _install_managed_at("oldcommit")
    with TestClient(app) as client:
        client.post("/api/comfy/update")
    assert "--commit" not in no_subprocess[0]["args"]


def test_update_ignores_a_blank_commit(home, app, no_subprocess):
    """A whitespace-only value from the text field must not become `--commit ""`
    on the argv - that would try to check out an empty ref instead of falling
    back to the pin, the exact case the field's placeholder promises."""
    _install_managed_at("oldcommit")
    with TestClient(app) as client:
        client.post("/api/comfy/update", params={"commit": "   "})
    assert "--commit" not in no_subprocess[0]["args"]


def test_update_forwards_commit_and_reinstall_requirements_together(home, app, no_subprocess):
    _install_managed_at("oldcommit")
    with TestClient(app) as client:
        client.post("/api/comfy/update",
                    params={"commit": "abc123def", "reinstall_requirements": "true"})
    assert no_subprocess[0]["args"] == [
        "comfy", "update", "--reinstall-requirements", "--commit", "abc123def"]


def test_update_conflicts_when_nothing_installed(home, app, no_subprocess):
    """Nothing to update -> 409 and NO job, rather than dispatching a job that would
    fail minutes later with the CLI's own 'nothing to update'."""
    with TestClient(app) as client:
        r = client.post("/api/comfy/update")
    assert r.status_code == 409, r.text
    assert no_subprocess == []


def test_update_refuses_while_a_setup_is_running(home, app, no_subprocess, monkeypatch):
    """Two writers on one checkout is the failure an update must never create."""
    _install_managed_at("oldcommit")
    monkeypatch.setattr(gui_jobs.JobManager, "has_running",
                        lambda self, kind: kind == "comfy-setup")
    with TestClient(app) as client:
        r = client.post("/api/comfy/update")
    assert r.status_code == 409, r.text
    assert no_subprocess == []


def test_update_refuses_a_second_concurrent_update(home, app, no_subprocess, monkeypatch):
    _install_managed_at("oldcommit")
    monkeypatch.setattr(gui_jobs.JobManager, "has_running",
                        lambda self, kind: kind == "comfy-update")
    with TestClient(app) as client:
        r = client.post("/api/comfy/update")
    assert r.status_code == 409, r.text
    assert no_subprocess == []


# --------------------------------------------------------------------------- #
#  GET /api/comfy/managed-status : the update-availability half                 #
# --------------------------------------------------------------------------- #

def test_status_reports_an_available_update(home, app):
    from localm.media.managed_comfy_fresh import COMFYUI_PINNED_COMMIT
    _install_managed_at("a-commit-that-is-not-the-pin")
    with TestClient(app) as client:
        body = client.get("/api/comfy/managed-status").json()
    assert body["update_available"] is True
    assert body["pinned_commit"] == COMFYUI_PINNED_COMMIT
    assert body["installed_commit"] == "a-commit-that-is-not-the-pin"
    assert body["updatable"] is True
    assert body["update_blocked_reason"] == ""


def test_status_reports_up_to_date_at_the_shipped_pin(home, app):
    from localm.media.managed_comfy_fresh import COMFYUI_PINNED_COMMIT
    _install_managed_at(COMFYUI_PINNED_COMMIT)
    with TestClient(app) as client:
        body = client.get("/api/comfy/managed-status").json()
    assert body["update_available"] is False


def test_status_unreadable_marker_is_unknown_not_up_to_date(home, app):
    """A marker that cannot be parsed must report UNKNOWN (null), never False.
    Collapsing 'could not look' into 'no update' hides a genuinely available
    update."""
    from localm.media.managed_comfy_provision import MARKER_FILENAME
    paths = _install_managed()
    (paths.root / MARKER_FILENAME).write_text("{not json at all", encoding="utf-8")
    with TestClient(app) as client:
        body = client.get("/api/comfy/managed-status").json()
    assert body["update_available"] is None
    assert body["installed_commit"] is None


def test_status_non_git_install_is_not_updatable_and_says_why(home, app):
    """The S2 non-git copy fallback cannot take a pinned update. update_managed_comfy()
    refuses honestly at run time; this advisory field lets the GUI say so BEFORE the
    user starts a job that would fail. The reason must be present, not just the flag."""
    _install_managed_at("oldcommit", git=False)
    with TestClient(app) as client:
        body = client.get("/api/comfy/managed-status").json()
    assert body["updatable"] is False
    assert "no git history" in body["update_blocked_reason"]
    assert "set it up again" in body["update_blocked_reason"]


def test_status_not_installed_omits_update_fields(home, app):
    """No install -> no update story to tell; the fields simply are not there rather
    than carrying a misleading default."""
    with TestClient(app) as client:
        body = client.get("/api/comfy/managed-status").json()
    assert body["installed"] is False
    assert "update_available" not in body
