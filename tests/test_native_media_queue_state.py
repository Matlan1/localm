# SPDX-License-Identifier: AGPL-3.0-or-later
"""A native image, upscale or video job waiting on the shared sd.cpp worker lock
reports phase "queued" in its activity listing row and loses it the moment it
takes the lock or is cancelled."""

from __future__ import annotations

import threading
import time

import pytest

from localm.media.sdcpp import shared
from localm.plugins.builtin.image.backends import native as image_native
from tests.test_image_native_backend import (  # noqa: F401  (fixtures)
    _wait_job as _wait_image_job,
    fake_native,
    native_app,
    no_comfy,
)
from tests.test_native_image_qol import (  # noqa: F401  (fixtures)
    _generate_one,
    _png,
    _register,
    qol,
    qol_app,
)
from tests.test_video_native_backend import (  # noqa: F401  (fixtures)
    _wait_job as _wait_video_job,
    fake_video,
    video_app,
)


def _wait_for(predicate, what, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError(f"timed out waiting for {what}")


def _phases(job):
    return [e.get("phase") for e in job._history if e.get("type") == "progress"]


# --------------------------------------------------------------------------- #
#  shared.held                                                                #
# --------------------------------------------------------------------------- #

def test_held_calls_on_state_for_a_wait_only():
    states = []
    with shared.held(lambda t: None, None, states.append) as acquired:
        assert acquired and shared.lock.locked()
    assert states == []


def test_held_reports_queued_then_running_once_the_lock_is_taken():
    states = []
    held_when_running = []

    def on_state(state):
        states.append(state)
        held_when_running.append(shared.lock.locked())

    def body():
        with shared.held(lambda t: None, None, on_state) as acquired:
            assert acquired

    worker = threading.Thread(target=body)
    with shared.lock:
        worker.start()
        _wait_for(lambda: states == ["queued"], "the queued state")
        assert states == ["queued"]
    worker.join(5.0)
    assert not worker.is_alive()
    assert states == ["queued", "running"]
    assert held_when_running == [True, True]
    assert not shared.lock.locked()


def test_held_reports_cancelled_when_the_wait_is_cancelled():
    states = []
    cancel = threading.Event()
    result = []

    def body():
        with shared.held(lambda t: None, cancel.is_set, states.append) as acquired:
            result.append(acquired)

    worker = threading.Thread(target=body)
    with shared.lock:
        worker.start()
        _wait_for(lambda: states == ["queued"], "the queued state")
        cancel.set()
        worker.join(5.0)
        assert not worker.is_alive()
        assert shared.lock.locked()
    assert result == [False]
    assert states == ["queued", "cancelled"]


def test_held_clears_queued_when_the_cancel_check_itself_raises():
    states = []

    def explode():
        raise RuntimeError("cancel check failed")

    outcome = []

    def body():
        try:
            with shared.held(lambda t: None, explode, states.append):
                outcome.append("entered")
        except RuntimeError as e:
            outcome.append(str(e))

    worker = threading.Thread(target=body)
    with shared.lock:
        worker.start()
        worker.join(5.0)
        assert not worker.is_alive()
    assert outcome == ["cancel check failed"]
    assert states == ["queued", "cancelled"]
    assert shared.lock.acquire(blocking=False)
    shared.lock.release()


# --------------------------------------------------------------------------- #
#  backends                                                                   #
# --------------------------------------------------------------------------- #

def _queued_then(call, act=None, until=None):
    """Run *call(on_queue_state)* in a thread while this thread holds the lock.
    Once it has queued, call *act()* and keep holding the lock until *until()*
    is true. Returns (states, result)."""
    states = []
    result = []
    worker = threading.Thread(target=lambda: result.append(call(states.append)))
    with shared.lock:
        worker.start()
        _wait_for(lambda: states == ["queued"], "the queued state")
        if act is not None:
            act()
        if until is not None:
            _wait_for(lambda: until(states), "the state change")
    worker.join(10.0)
    assert not worker.is_alive()
    return states, result


def test_an_upscale_reports_queued_then_running(qol, tmp_path):  # noqa: F811
    runner, s = qol
    up = tmp_path / "up.safetensors"
    up.write_bytes(b"x")
    _register("up4x", up, "upscaler")
    src = _png(tmp_path / "src.png")
    states, result = _queued_then(
        lambda cb: image_native.upscale(s, src, "up4x", tmp_path / "o.png", on_queue_state=cb))
    assert states == ["queued", "running"]
    assert result[0][0] is True


def test_an_upscale_cancelled_while_queued_clears_queued(qol, tmp_path):  # noqa: F811
    runner, s = qol
    up = tmp_path / "up.safetensors"
    up.write_bytes(b"x")
    _register("up4x", up, "upscaler")
    src = _png(tmp_path / "src.png")
    cancel = threading.Event()
    states, result = _queued_then(
        lambda cb: image_native.upscale(s, src, "up4x", tmp_path / "o.png",
                                        cancel_check=cancel.is_set, on_queue_state=cb),
        cancel.set, lambda states: states == ["queued", "cancelled"])
    assert states == ["queued", "cancelled"]
    assert result == [(False, "Cancelled.")]


# --------------------------------------------------------------------------- #
#  the job's activity row                                                     #
# --------------------------------------------------------------------------- #

def _queued_job_summary(app, post):
    """POST while this thread holds the lock; return the job once its row says
    queued."""
    from fastapi.testclient import TestClient
    client = TestClient(app)
    client.__enter__()
    r = post(client)
    assert r.status_code == 200, r.text
    job = app.state.jobs.get(r.json()["job_id"])
    _wait_for(lambda: job.summary().get("phase") == "queued", "the queued row")
    return client, job


@pytest.mark.parametrize("cancel", [False, True])
def test_an_image_job_row_is_queued_only_while_it_waits(native_app, cancel):  # noqa: F811
    app, _runner = native_app
    with shared.lock:
        client, job = _queued_job_summary(
            app, lambda c: c.post("/api/imagine", json={"prompt": "a fox"}))
        assert job.summary()["status"] == "running"
        if cancel:
            job.cancel()
            _wait_for(lambda: job.summary().get("phase") == "cancelled", "the cancelled row")
    try:
        _wait_image_job(app, job.id)
    finally:
        client.__exit__(None, None, None)
    row = job.summary()
    assert row.get("phase") != "queued"
    assert row.get("phase") == ("cancelled" if cancel else "running")
    assert _phases(job)[0] == "queued"


@pytest.mark.parametrize("cancel", [False, True])
def test_a_video_job_row_is_queued_only_while_it_waits(video_app, cancel):  # noqa: F811
    app, _runner = video_app
    body = {"prompt": "a fox", "seconds": 1, "fps": 8, "width": 96, "height": 64}
    with shared.lock:
        client, job = _queued_job_summary(app, lambda c: c.post("/api/video", json=body))
        assert job.summary()["status"] == "running"
        if cancel:
            job.cancel()
            _wait_for(lambda: job.summary().get("phase") == "cancelled", "the cancelled row")
    try:
        _wait_video_job(app, job.id)
    finally:
        client.__exit__(None, None, None)
    row = job.summary()
    assert row.get("phase") != "queued"
    assert row.get("phase") == ("cancelled" if cancel else "running")
    assert _phases(job)[0] == "queued"


def test_a_job_that_never_waits_never_reports_queued(native_app):  # noqa: F811
    from fastapi.testclient import TestClient
    app, _runner = native_app
    with TestClient(app) as c:
        r = c.post("/api/imagine", json={"prompt": "a fox"})
        assert r.status_code == 200, r.text
        job = _wait_image_job(app, r.json()["job_id"])
    assert job.status == "done"
    assert "queued" not in _phases(job)
    assert "phase" not in job.summary()


@pytest.mark.parametrize("cancel", [False, True])
def test_an_upscale_job_row_is_queued_only_while_it_waits(qol_app, tmp_path, cancel):  # noqa: F811
    from fastapi.testclient import TestClient
    app, _runner = qol_app
    up = tmp_path / "up.safetensors"
    up.write_bytes(b"x")
    _register("up4x", up, "upscaler")
    with TestClient(app) as client:
        name = _generate_one(client, app)
        with shared.lock:
            r = client.post(f"/api/imagine/file/{name}/upscale", json={"upscaler": "up4x"})
            assert r.status_code == 200, r.text
            job = app.state.jobs.get(r.json()["job_id"])
            _wait_for(lambda: job.summary().get("phase") == "queued", "the queued row")
            if cancel:
                job.cancel()
                _wait_for(lambda: job.summary().get("phase") == "cancelled",
                          "the cancelled row")
        _wait_image_job(app, job.id)
    assert job.summary().get("phase") == ("cancelled" if cancel else "running")
    assert _phases(job)[0] == "queued"
