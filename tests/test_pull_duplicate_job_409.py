# SPDX-License-Identifier: AGPL-3.0-or-later
"""POST /api/models/pull refuses a second download of a spec already running.

ADVISORY, and the tests say so, because the distinction decides what a reader
does with a failure here. Reading the job list and then starting a job are two
steps, so two requests can both find nothing running and both proceed. The
guarantee that two downloads never write one file is the cross-process lock the
pull itself takes (tests/test_pull_part_lock.py), which also covers the
contender this check cannot see at all: a ``localm pull`` a user ran in a
terminal.

So what is pinned here is only that a duplicate is refused promptly instead of
becoming a job that starts and immediately gives up. Nothing here should ever
be cited as the reason concurrent downloads are safe.

The first PAIR of tests exists because of a specific way this could have
shipped broken: the route matches a running job by its LABEL, and the label
reaches the manager under a different parameter name (``host_label``). If those
two ever stop being the same string, the match silently never fires and the
route reverts to its old behaviour with no test noticing - a guard that cannot
fire, wearing the signature of one that checked and found nothing.

NOTHING HERE STARTS A REAL DOWNLOAD. The route's job is either stubbed or
pre-seeded, because a test that spawned `localm pull` to observe a status code
would make an unbounded network call on every run - and would pass or fail on
whether a model repository happens to exist today.
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

    Returns a dict that stays empty unless the route actually started a job, so
    a test can tell "the route proceeded" from "the route refused" without
    reading the status code twice.
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

    Uses the manager's own start_fn rather than a hand-built record, so the
    status/label/kind this test matches on are the ones the manager actually
    produces.
    """
    release = threading.Event()
    job = app.state.jobs.start_fn("pull", lambda *a, **k: release.wait(30),
                                  label=label)
    return job, release


def test_start_cli_host_label_is_reported_as_label_in_the_listing(gui_app):  # noqa: F811
    """Half one of the link the route's match depends on.

    start_cli takes `host_label`; the listing reports `label`. If those ever
    stop being the same string the duplicate check silently never matches.

    Runs `--version`, not a pull: this is about label plumbing, and starting a
    real download to observe a string would make an uncontrolled network call
    on every test run.
    """
    app, _ = gui_app
    job = app.state.jobs.start_cli("pull", ["--version"],
                                   host_label="Model pull owner/repo")
    rows = [j for j in app.state.jobs.snapshot() if j["id"] == job.id]
    assert rows, "start_cli produced no listing row"
    assert rows[0]["label"] == "Model pull owner/repo", rows[0]
    assert rows[0]["kind"] == "pull", rows[0]


def test_the_route_labels_its_job_with_the_spec(gui_app, monkeypatch):  # noqa: F811
    """Half two: the route really passes that exact string.

    Together with the test above this closes the loop - the route's label and
    the string the duplicate check compares against are proven to be the same
    value - without either test having to download anything.
    """
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
        # The injection took: a job with this exact kind/label/status is what
        # the route will look for, and a mismatch on any of the three would
        # make the refusal below unreachable.
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
    """The check is per SPEC, not per kind.

    Without this, refusing every pull while any pull runs would satisfy the
    test above while making the page unusable during a long download - and
    `has_running("pull")`, the nearest existing helper, would do exactly that.
    """
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
    """Finished jobs stay queryable for an hour, so matching on label alone -
    without the status check - would refuse every retry of a download that
    already failed."""
    app, _ = gui_app
    started = _no_real_downloads(monkeypatch)
    job, release = _running_pull(app, "Model pull owner/repo")
    release.set()

    deadline = time.time() + 30
    while (app.state.jobs.get(job.id).status == "running"
           and time.time() < deadline):
        time.sleep(0.02)
    # Waited for the SIGNAL, not a fixed sleep: on a loaded box a fixed wait
    # would leave the job running and this test would pass as a duplicate of
    # the refusal case instead of testing the retry.
    assert app.state.jobs.get(job.id).status != "running", (
        "the job never finished, so this test would pass for the wrong reason")

    with TestClient(app) as client:
        r = client.post("/api/models/pull", json={"spec": "owner/repo"})
    assert r.status_code == 200, r.text
    assert started.get("host_label") == "Model pull owner/repo", (
        "the route answered 200 without actually starting the retry")
