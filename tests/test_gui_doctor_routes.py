# SPDX-License-Identifier: AGPL-3.0-or-later
"""GUI diagnostics routes (localm/plugins/gui/routes/doctor.py):

  GET  /api/doctor      - the last report; runs nothing
  POST /api/doctor/run  - starts the five ACTIVE self-checks as a background job

Mirrors test_gui_runtime_update_routes.py's shape: the heavy work is stubbed so
these assert the DISPATCH and REPORTING contract, never re-run the real probes.
The probes themselves are covered by tests/test_diagnostics_core.py (including
one real child-process run) and by the eight test_doctor_*.py files.
"""

from __future__ import annotations

import threading
import time

import pytest
from fastapi import FastAPI
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

import localm.config as cfg
from localm import diagnostics as d
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
def app(home):
    a = FastAPI()
    attach_gui(a, self_url="http://127.0.0.1:9/v1",
               switch_model=lambda name: {"status": "loaded", "model": name},
               active_model=lambda: "model-a")
    return a


def _report(*statuses):
    checks = []
    for key, status in zip(d.CHECK_KEYS, statuses):
        checks.append(d.CheckResult(key=key, label=d.CHECK_LABELS[key],
                                    status=status, summary=f"{key} says {status}",
                                    findings=()))
    return d.build_report(checks)


@pytest.fixture
def stub_run(monkeypatch):
    """Replace the isolated run with a canned report. Patched on localm.
    diagnostics, which is where the route resolves the name from."""
    def _set(report, *, block: threading.Event | None = None,
             progress: list | None = None):
        def _fake(*, timeout=360.0, on_progress=None):
            for ev in (progress or []):
                if on_progress:
                    on_progress(*ev)
            if block is not None:
                block.wait(10)
            return report
        monkeypatch.setattr(d, "run_report_isolated", _fake)
    return _set


def _wait_until_idle(client, deadline_s=10.0):
    """Poll the GET until the background job has finished. The work runs in a
    worker thread (JobManager.start_fn), so a test that read once would be
    racing it."""
    end = time.monotonic() + deadline_s
    while time.monotonic() < end:
        body = client.get("/api/doctor").json()
        if not body["running"] and body["report"] is not None:
            return body
        time.sleep(0.02)
    pytest.fail(f"the diagnostics job never finished: {client.get('/api/doctor').json()}")


# --------------------------------------------------------------------------- #
#  GET before anything has run                                                #
# --------------------------------------------------------------------------- #

def test_before_any_run_the_report_is_null_and_nothing_is_running(app):
    with TestClient(app) as client:
        body = client.get("/api/doctor").json()
    assert body["running"] is False
    assert body["report"] is None
    assert body["started_at"] is None
    assert body["progress"] is None


def test_the_endpoint_names_the_five_checks_it_covers_before_it_runs_them(app):
    """The card has to be able to say WHICH checks this is, both to render
    placeholder rows and so its verdict is never read as a claim about the whole
    machine. That list has to come from the core, not be retyped in the UI."""
    with TestClient(app) as client:
        body = client.get("/api/doctor").json()
    assert [c["key"] for c in body["covers"]] == list(d.CHECK_KEYS)
    assert [c["label"] for c in body["covers"]] == [
        d.CHECK_LABELS[k] for k in d.CHECK_KEYS]


# --------------------------------------------------------------------------- #
#  Running                                                                     #
# --------------------------------------------------------------------------- #

def test_a_run_stores_the_report_and_reports_the_verdict(app, stub_run):
    stub_run(_report(d.OK, d.OK, d.OK, d.OK, d.SKIPPED))
    with TestClient(app) as client:
        started = client.post("/api/doctor/run")
        assert started.status_code == 200
        assert started.json()["job_id"]
        body = _wait_until_idle(client)
    assert body["report"]["verdict"] == d.OK
    assert [c["key"] for c in body["report"]["checks"]] == list(d.CHECK_KEYS)
    assert body["finished_at"] is not None
    assert body["job_id"] == started.json()["job_id"]


def test_a_failing_check_makes_the_verdict_fail_without_failing_the_run(app, stub_run):
    """"We could not check" and "we checked and it is broken" are different
    facts. A report that found a real fault is a SUCCESSFUL run, so the job must
    not be marked failed - otherwise the activity list says the diagnostics
    broke when what actually happened is that they worked."""
    stub_run(_report(d.FAIL, d.SKIPPED, d.OK, d.OK, d.SKIPPED))
    with TestClient(app) as client:
        job_id = client.post("/api/doctor/run").json()["job_id"]
        body = _wait_until_idle(client)
        # /api/activity, NOT /api/jobs: that prefix belongs to the
        # scheduled-jobs plugin (recurring task DEFINITIONS, a different id
        # space); the in-flight listing is /api/activity.operations.
        activity = client.get("/api/activity").json()
    assert body["report"]["verdict"] == d.FAIL
    row = next(j for j in activity["operations"] if j["id"] == job_id)
    assert row["status"] == "done", row


def test_a_run_that_could_not_complete_is_reported_as_an_error(app, stub_run):
    """An unrunnable diagnostic must not render as a clean bill of health: the
    verdict must be ERROR, the reason must survive, and the JOB must be marked
    failed."""
    stub_run(d.DiagnosticsReport(checks=(), verdict=d.ERROR,
                                 error="the diagnostics run did not finish within 360s"))
    with TestClient(app) as client:
        job_id = client.post("/api/doctor/run").json()["job_id"]
        body = _wait_until_idle(client)
        activity = client.get("/api/activity").json()
    assert body["report"]["verdict"] == d.ERROR
    assert "did not finish" in body["report"]["error"]
    assert body["report"]["checks"] == []
    row = next(j for j in activity["operations"] if j["id"] == job_id)
    assert row["status"] == "failed", row


def test_a_second_run_is_refused_while_one_is_in_flight(app, stub_run):
    """Two concurrent runs would each build a throwaway venv and each spawn
    workers, to answer one question."""
    gate = threading.Event()
    stub_run(_report(d.OK, d.OK, d.OK, d.OK, d.OK), block=gate)
    try:
        with TestClient(app) as client:
            assert client.post("/api/doctor/run").status_code == 200
            # The worker thread is inside the blocked stub by the time
            # has_running() can see it; poll rather than sleep a fixed amount.
            end = time.monotonic() + 5
            while time.monotonic() < end and not client.get("/api/doctor").json()["running"]:
                time.sleep(0.01)
            second = client.post("/api/doctor/run")
            assert second.status_code == 409
            assert "already in progress" in second.json()["detail"]
    finally:
        gate.set()


def test_progress_names_the_check_currently_running(app, stub_run):
    """A run is ~20s on a healthy box and minutes at worst. Reporting which
    check is in flight is the difference between progress and a spinner."""
    gate = threading.Event()
    stub_run(_report(d.OK, d.OK, d.OK, d.OK, d.OK), block=gate,
             progress=[("llama_lib", "llama.cpp library", 0, 5),
                       ("venv", "Nested venv creation", 3, 5)])
    try:
        with TestClient(app) as client:
            client.post("/api/doctor/run")
            end = time.monotonic() + 5
            seen = None
            while time.monotonic() < end:
                body = client.get("/api/doctor").json()
                if body["running"] and body["progress"]["done"] == 3:
                    seen = body["progress"]
                    break
                time.sleep(0.01)
            assert seen == {"phase": "Nested venv creation", "done": 3, "total": 5}
    finally:
        gate.set()


def test_progress_is_absent_once_the_run_has_finished(app, stub_run):
    """A phase left over from a finished run reads as a check still going."""
    stub_run(_report(d.OK, d.OK, d.OK, d.OK, d.OK),
             progress=[("venv", "Nested venv creation", 3, 5)])
    with TestClient(app) as client:
        client.post("/api/doctor/run")
        body = _wait_until_idle(client)
    assert body["progress"] is None


# --------------------------------------------------------------------------- #
#  The job log line                                                            #
# --------------------------------------------------------------------------- #

def test_the_job_line_names_the_checks_that_need_attention():
    """A job log is often the only thing quoted in a bug report, so "2 problems"
    sends the reader back to a UI they may not have."""
    from localm.plugins.gui.routes import doctor as route

    line = route._job_line(_report(d.FAIL, d.OK, d.OK, d.WARN, d.SKIPPED))
    assert d.CHECK_LABELS["llama_lib"] in line
    assert d.CHECK_LABELS["venv"] in line
    assert d.CHECK_LABELS["worker_spawn"] not in line


def test_the_all_clear_job_line_does_not_claim_more_than_it_checked():
    from localm.plugins.gui.routes import doctor as route

    line = route._job_line(_report(*([d.OK] * 5)))
    assert "active checks passed" in line
    assert "not the whole system" in line


# --------------------------------------------------------------------------- #
#  Gating                                                                      #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("method,path", [
    ("GET", "/api/doctor"),
    ("POST", "/api/doctor/run"),
])
def test_both_routes_are_scope_gated(app, method, path):
    """Starting a run spawns processes and builds a throwaway venv, so neither
    of these may be reachable without a checked key. Walks the live app the way
    test_kernel_routes_scope_contract.py does, so removing the dependency fails
    here rather than in review."""
    route = next(r for r in app.routes
                 if isinstance(r, APIRoute) and r.path == path
                 and method in r.methods)
    quals = {getattr(dep.call, "__qualname__", "")
             for dep in route.dependant.dependencies}
    assert "require_scope.<locals>.dep" in quals, quals
