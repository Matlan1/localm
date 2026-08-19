# SPDX-License-Identifier: AGPL-3.0-or-later
"""GUI llama.cpp runtime-update routes (localm/plugins/gui/routes/runtime.py):

  GET  /api/runtime/check   - read-only, reflects setup_llama.check_runtime_update()
  POST /api/runtime/update  - dispatches a JOB to the EXISTING `setup-llama` CLI
                              entry point, never blocking the request and never
                              re-implementing provisioning here.

Mirrors test_managed_comfy_s5_gui.py's shape: the heavy CLI is stubbed so these
assert only the DISPATCH contract (job id returned, correct argv, correct 400s
and 409s). A real run is deliberately never made: setup-llama provisions into
the venv's localm_llama_runtime wheel, so an actual dispatch here would replace
this machine's runtime.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import localm.config as cfg
from localm import setup_llama as sl
from localm.plugins.gui import jobs as gui_jobs
from localm.plugins.gui.web import attach_gui


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / ".localm"
    h.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("LOCALM_HOME", str(h))
    monkeypatch.setattr(cfg, "HOME_DIR", h)
    monkeypatch.setattr(cfg, "CONFIG_FILE", h / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", h / "registry.json")
    return h


@pytest.fixture
def runtime_dir(tmp_path, monkeypatch):
    """Redirect setup_llama's notion of "the runtime lib dir" to a throwaway
    tmp dir, so the routes' own installed_backend()/check_runtime_update()
    calls never touch this box's real provisioned runtime."""
    d = tmp_path / "rt"
    d.mkdir()
    monkeypatch.setattr(sl, "_repo_runtime_lib", lambda: d)
    return d


@pytest.fixture
def app(home, runtime_dir):
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
    """Stub JobManager.start_cli so the update route dispatches a job without
    spawning a real setup-llama subprocess. Records calls for assertion."""
    calls = []

    def _fake_start_cli(self, kind, cli_args, **kw):
        calls.append({"kind": kind, "args": list(cli_args), "kw": kw})
        return _FakeJob("job123")

    monkeypatch.setattr(gui_jobs.JobManager, "start_cli", _fake_start_cli)
    return calls


# --------------------------------------------------------------------------- #
#  GET /api/runtime/check                                                     #
# --------------------------------------------------------------------------- #

def test_check_reports_not_installed(app):
    with TestClient(app) as client:
        r = client.get("/api/runtime/check")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["installed"] is False
    assert body["newer"] is False


def test_check_reports_a_newer_build_available(app, runtime_dir, monkeypatch):
    """The card offers the build setup-llama would install - which is the
    CONFIRMED pin, not upstream's newest. If the two disagreed the GUI would
    advertise an update that pressing the button does not deliver, and would go
    on advertising it after every re-provision.

    The _latest_tag stand-in FAILS if reached: this route runs on every card
    render, so a release lookup creeping back here is a network call per poll as
    well as the wrong answer."""
    monkeypatch.setattr(
        sl, "_latest_tag",
        lambda: pytest.fail("the runtime card must not query upstream by default"))
    sl._record_provisioned_backend(runtime_dir, "vulkan", build="b10300")

    with TestClient(app) as client:
        r = client.get("/api/runtime/check")

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["installed"] is True
    assert body["backend"] == "vulkan"
    assert body["current"] == "b10300"
    assert body["target"] == sl._PINNED_TAG
    assert body["newer"] is True


def test_check_reports_up_to_date(app, runtime_dir, monkeypatch):
    sl._record_provisioned_backend(runtime_dir, "vulkan", build=sl._PINNED_TAG)

    with TestClient(app) as client:
        r = client.get("/api/runtime/check")

    assert r.json()["newer"] is False


# --------------------------------------------------------------------------- #
#  POST /api/runtime/update                                                   #
# --------------------------------------------------------------------------- #

def test_update_dispatches_job_with_force_and_yes(app, runtime_dir, no_subprocess):
    """NEW-UPDATE-RUNTIME-CLASS-IS-A-NO-OP applies here too: without --force
    setup-llama's own already-provisioned guard would make this a no-op;
    --yes keeps the unattended job from ever hanging on a prompt."""
    sl._record_provisioned_backend(runtime_dir, "vulkan", build="b10300")

    with TestClient(app) as client:
        r = client.post("/api/runtime/update")

    assert r.status_code == 200, r.text
    assert r.json().get("job_id") == "job123"
    assert len(no_subprocess) == 1, no_subprocess
    args = no_subprocess[0]["args"]
    assert args[:3] == ["setup-llama", "--backend", "vulkan"], args
    assert "--force" in args
    assert "--yes" in args


def test_update_with_nothing_installed_provisions_with_auto_detect(app, no_subprocess):
    """The uninstalled case used to 409 ("nothing to update - run setup
    first"), which made INITIAL PROVISIONING unreachable for a user who has
    only the GUI - and left a box whose runtime failed to provision with no
    route back, since the one surface still working refused the one action
    that would fix it. With no backend named and none installed, this must
    dispatch the same auto-detect a bare `localm setup-llama` performs."""
    with TestClient(app) as client:
        r = client.post("/api/runtime/update")

    assert r.status_code == 200, r.text
    assert len(no_subprocess) == 1, no_subprocess
    assert no_subprocess[0]["args"] == [
        "setup-llama", "--backend", "auto", "--force", "--yes"]


def test_update_switches_backend_when_one_is_named(app, runtime_dir, no_subprocess):
    """--backend, the switch a GUI user could not reach. The NAMED backend
    wins over the installed one; that is the whole point of the option."""
    sl._record_provisioned_backend(runtime_dir, "vulkan", build="b10300")

    with TestClient(app) as client:
        r = client.post("/api/runtime/update", json={"backend": "cuda"})

    assert r.status_code == 200, r.text
    assert no_subprocess[0]["args"] == [
        "setup-llama", "--backend", "cuda", "--force", "--yes"]


def test_update_forwards_a_tag(app, runtime_dir, no_subprocess):
    """--tag installs AND PINS one exact build. Forwarded verbatim so the pin
    is written by the same code path a terminal run uses."""
    sl._record_provisioned_backend(runtime_dir, "vulkan", build="b10300")

    with TestClient(app) as client:
        r = client.post("/api/runtime/update", json={"tag": "b10355"})

    assert r.status_code == 200, r.text
    assert no_subprocess[0]["args"] == [
        "setup-llama", "--backend", "vulkan", "--tag", "b10355",
        "--force", "--yes"]


def test_update_accepts_backend_and_tag_together(app, no_subprocess):
    with TestClient(app) as client:
        r = client.post("/api/runtime/update",
                        json={"backend": "CPU", "tag": " latest "})

    assert r.status_code == 200, r.text
    # Cased/padded input is normalised the way the CLI normalises it, rather
    # than reaching the child as a value click would then refuse.
    assert no_subprocess[0]["args"] == [
        "setup-llama", "--backend", "cpu", "--tag", "latest",
        "--force", "--yes"]


def test_update_treats_blank_fields_as_absent(app, runtime_dir, no_subprocess):
    """An untouched select and an empty text box are the GUI's normal state,
    so they must mean "no request", not a request for the empty string."""
    sl._record_provisioned_backend(runtime_dir, "vulkan", build="b10300")

    with TestClient(app) as client:
        r = client.post("/api/runtime/update", json={"backend": "", "tag": "  "})

    assert r.status_code == 200, r.text
    assert no_subprocess[0]["args"] == [
        "setup-llama", "--backend", "vulkan", "--force", "--yes"]


def test_update_refuses_an_unknown_backend_before_dispatching(app, no_subprocess):
    """400, not a job that starts and fails: the caller gets the reason now,
    and the value set is setup_llama.BACKENDS - the same tuple the CLI's own
    click.Choice is built from, so the two surfaces cannot disagree."""
    with TestClient(app) as client:
        r = client.post("/api/runtime/update", json={"backend": "rocm"})

    assert r.status_code == 400, r.text
    assert "vulkan" in r.json()["detail"]
    assert no_subprocess == []


def test_update_refuses_a_tag_the_cli_would_refuse(app, no_subprocess):
    """The route validates with setup_llama.is_safe_tag, the SAME predicate
    _validated_tag uses, so a tag is judged once. A leading '-' matters beyond
    tidiness: it is the shape that would otherwise reach click as an option."""
    with TestClient(app) as client:
        for bad in ("--force", "b1/../x", "a b"):
            r = client.post("/api/runtime/update", json={"tag": bad})
            assert r.status_code == 400, (bad, r.text)

    assert no_subprocess == []


def test_update_refuses_when_already_running_whatever_was_asked_for(
        app, runtime_dir, no_subprocess, monkeypatch):
    """ONE job kind covers install, switch and update, so a switch cannot be
    started on top of an update already rewriting the same directory."""
    monkeypatch.setattr(gui_jobs.JobManager, "has_running", lambda self, kind: True)

    with TestClient(app) as client:
        r = client.post("/api/runtime/update", json={"backend": "cuda"})

    assert r.status_code == 409, r.text
    assert no_subprocess == []


def test_update_refuses_when_already_running(app, runtime_dir, monkeypatch):
    sl._record_provisioned_backend(runtime_dir, "vulkan", build="b10300")
    monkeypatch.setattr(gui_jobs.JobManager, "has_running", lambda self, kind: True)

    with TestClient(app) as client:
        r = client.post("/api/runtime/update")

    assert r.status_code == 409, r.text
