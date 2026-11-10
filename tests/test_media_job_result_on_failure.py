# SPDX-License-Identifier: AGPL-3.0-or-later
"""A media job that saved nothing must not name a result file, and the video
route refuses a size the native backend cannot make before it starts a job."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from localm import jobmanager as gj


def _end_event(job, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        ends = [e for e in list(job._history) if e.get("type") == "end"]
        if ends:
            return ends[-1]
        time.sleep(0.01)
    raise AssertionError("no end event")


def _run(fn):
    job = gj.JobManager().start_fn("video", fn, result_path="out.mp4")
    return job, _end_event(job)


def test_a_done_job_names_its_result_file():
    job, end = _run(lambda j: True)
    assert end["status"] == "done"
    assert end["result"] == "out.mp4"


def test_a_failed_job_reports_no_result_file():
    job, end = _run(lambda j: False)
    assert end["status"] == "failed"
    assert end["result"] is None
    assert job.result is None


def test_a_raising_job_reports_no_result_file():
    def boom(j):
        raise RuntimeError("x")

    job, end = _run(boom)
    assert end["status"] == "failed"
    assert end["result"] is None


def test_a_cancelled_job_reports_no_result_file():
    def wait_for_cancel(j):
        j.cancel_event.wait(5)
        return False

    job = gj.JobManager().start_fn("video", wait_for_cancel, result_path="out.mp4")
    job.cancel()
    end = _end_event(job)
    assert end["status"] == "cancelled"
    assert end["result"] is None


def _video_request_for(backend, width, height, monkeypatch):
    from localm.plugins.builtin.video import plug
    monkeypatch.setattr(plug._backend, "settings", lambda cfg: {"backend": backend})
    monkeypatch.setattr("localm.media.backend_choice.refine_auto", lambda s, label: s)
    started = []

    class _Jobs:
        def start_fn(self, kind, fn, result_path=None, owner=None):
            started.append(kind)
            return MagicMock(id="job1")

    request = MagicMock()
    request.app.state.jobs = _Jobs()
    request.app.state.self_url = "http://127.0.0.1:8642/v1"
    request.headers = {}
    request.cookies = {}
    req = plug.VideoRequest(prompt="waves", width=width, height=height)
    return plug, req, request, started


@pytest.mark.parametrize("w,h", [(250, 256), (256, 250), (32, 256), (2048, 256)])
def test_native_video_route_refuses_a_bad_size_before_a_job_exists(monkeypatch, w, h):
    plug, req, request, started = _video_request_for("native", w, h, monkeypatch)
    with pytest.raises(HTTPException) as err:
        asyncio.run(plug.video(req, request))
    assert err.value.status_code == 400
    assert "multiple of 16" in err.value.detail
    assert started == []


def test_native_video_route_accepts_a_valid_size(monkeypatch):
    plug, req, request, started = _video_request_for("native", 256, 256, monkeypatch)
    asyncio.run(plug.video(req, request))
    assert started == ["video"]


def test_the_comfy_backend_keeps_its_own_size_rules(monkeypatch):
    plug, req, request, started = _video_request_for("comfy", 250, 256, monkeypatch)
    asyncio.run(plug.video(req, request))
    assert started == ["video"]


def _music_request(backend, monkeypatch, **fields):
    from localm.plugins.builtin.music import plug
    monkeypatch.setattr(plug._backend, "settings", lambda cfg: {"backend": backend})
    monkeypatch.setattr("localm.media.backend_choice.refine_auto", lambda s, label: s)
    started = []

    class _Jobs:
        def start_fn(self, kind, fn, result_path=None, owner=None):
            started.append(kind)
            return MagicMock(id="job1")

    request = MagicMock()
    request.app.state.jobs = _Jobs()
    request.app.state.self_url = "http://127.0.0.1:8642/v1"
    request.headers = {}
    request.cookies = {}
    return plug, plug.MusicRequest(tags="lofi", **fields), request, started


@pytest.mark.parametrize("field,value", [("sampler_name", "euler"),
                                         ("scheduler", "karras"),
                                         ("lyrics_strength", 0.5)])
def test_native_music_route_refuses_comfy_only_settings_before_a_job(
        monkeypatch, field, value):
    plug, req, request, started = _music_request("native", monkeypatch, **{field: value})
    with pytest.raises(HTTPException) as err:
        asyncio.run(plug.music(req, request))
    assert err.value.status_code == 400
    assert "ComfyUI workflow only" in err.value.detail
    assert started == []


def test_comfy_music_route_accepts_a_sampler(monkeypatch):
    plug, req, request, started = _music_request("comfy", monkeypatch, sampler_name="euler")
    asyncio.run(plug.music(req, request))
    assert started == ["music"]


def test_native_music_route_without_those_settings_starts(monkeypatch):
    plug, req, request, started = _music_request("native", monkeypatch)
    asyncio.run(plug.music(req, request))
    assert started == ["music"]
