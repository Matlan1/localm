# SPDX-License-Identifier: AGPL-3.0-or-later
"""POST /api/models/pull refuses a second download of a spec already running.

The check is ADVISORY. Reading the job list and starting a job are two steps,
so two requests can both find nothing running and both proceed. What keeps two
downloads from writing one file is the cross-process lock the pull itself
takes, which also covers the contender this check cannot see at all: a
``localm pull`` a user ran in a terminal. What is pinned here is only that a
duplicate is refused promptly instead of becoming a job that starts and
immediately gives up.

The route matches a running job by its LABEL, and the label reaches the
manager under a different parameter name (``host_label``); the first pair of
tests pins those two to the same string.

No test here starts a real download: the route's job is either stubbed or
pre-seeded.
"""

from __future__ import annotations

import threading
import time

from fastapi.testclient import TestClient

from tests.test_gui import _FAKE_REGISTRY, gui_app  # noqa: F401


class _FakeJob:
    id = "job-stub"


def _no_real_downloads(monkeypatch):
    """Stub start_cli so a 200 from the route spawns nothing.

    Returns a dict that stays empty unless the route actually started a job,
    and otherwise carries the job's ``kind`` and ``host_label``.
    """
    started = {}

    def fake_start_cli(self, kind, cli_args, **kw):
        started["kind"] = kind
        started["host_label"] = kw.get("host_label")
        return _FakeJob()

    monkeypatch.setattr(
        "localm.plugins.gui.jobs.JobManager.start_cli", fake_start_cli)
    return started


def _running_pull(app, label: str):
    """A REAL job in the manager, kind 'pull', status running, holding *label*.

    Built through the manager's own start_fn, so the status/label/kind are
    the ones the manager produces. Returns ``(job, release_event)``; setting
    the event lets the job finish.
    """
    release = threading.Event()
    job = app.state.jobs.start_fn("pull", lambda *a, **k: release.wait(30),
                                  label=label)
    return job, release


def test_start_cli_host_label_is_reported_as_label_in_the_listing(gui_app):  # noqa: F811
    """start_cli takes `host_label`; the listing reports `label`.

    The duplicate check matches only while those two are the same string.
    Runs `--version`, not a pull.
    """
    app, _ = gui_app
    job = app.state.jobs.start_cli("pull", ["--version"],
                                   host_label="Model pull owner/repo")
    rows = [j for j in app.state.jobs.snapshot() if j["id"] == job.id]
    assert rows, "start_cli produced no listing row"
    assert rows[0]["label"] == "Model pull owner/repo", rows[0]
    assert rows[0]["kind"] == "pull", rows[0]


def test_the_route_labels_its_job_with_the_spec(gui_app, monkeypatch):  # noqa: F811
    """The route passes the spec-derived label straight to start_cli."""
    app, _ = gui_app
    seen = _no_real_downloads(monkeypatch)
    with TestClient(app) as client:
        r = client.post("/api/models/pull", json={"spec": "owner/repo"})
    assert r.status_code == 200, r.text
    assert seen["kind"] == "pull"
    assert seen["host_label"] == "Model pull owner/repo", seen


def test_a_second_pull_of_the_same_spec_is_refused(gui_app):  # noqa: F811
    app, _ = gui_app
    job, release = _running_pull(app, "Model pull owner/repo")
    try:
        # The injection took: the route matches on this kind, status and label.
        row = next(j for j in app.state.jobs.snapshot() if j["id"] == job.id)
        assert row["kind"] == "pull"
        assert row["status"] == "running"
        assert row["label"] == "Model pull owner/repo"

        with TestClient(app) as client:
            r = client.post("/api/models/pull", json={"spec": "owner/repo"})
        assert r.status_code == 409, r.text
        assert "Already downloading" in r.json()["detail"]
    finally:
        release.set()


def test_a_pull_of_a_DIFFERENT_spec_still_starts(gui_app, monkeypatch):  # noqa: F811
    """The check is per SPEC, not per kind: a pull of another spec still
    starts while one is running."""
    app, _ = gui_app
    started = _no_real_downloads(monkeypatch)
    job, release = _running_pull(app, "Model pull owner/repo")
    try:
        with TestClient(app) as client:
            r = client.post("/api/models/pull", json={"spec": "other/model"})
        assert r.status_code == 200, r.text
        assert started.get("host_label") == "Model pull other/model", (
            "the route answered 200 without actually starting the download")
    finally:
        release.set()


def test_a_finished_pull_does_not_block_a_retry(gui_app, monkeypatch):  # noqa: F811
    """Finished jobs stay queryable for an hour; the status half of the match
    is what lets a retry of a finished pull start."""
    app, _ = gui_app
    started = _no_real_downloads(monkeypatch)
    job, release = _running_pull(app, "Model pull owner/repo")
    release.set()

    deadline = time.time() + 30
    while (app.state.jobs.get(job.id).status == "running"
           and time.time() < deadline):
        time.sleep(0.02)
    assert app.state.jobs.get(job.id).status != "running", (
        "the job never finished, so this test would pass for the wrong reason")

    with TestClient(app) as client:
        r = client.post("/api/models/pull", json={"spec": "owner/repo"})
    assert r.status_code == 200, r.text
    assert started.get("host_label") == "Model pull owner/repo", (
        "the route answered 200 without actually starting the retry")
