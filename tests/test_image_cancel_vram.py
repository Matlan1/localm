# SPDX-License-Identifier: AGPL-3.0-or-later
"""Cancelling an image generation must still restore VRAM.

Stop unloads the chat model to free VRAM for the image model, so the chat-model
reload must run on the cancel path as well as on success.
"""

import asyncio
import time
from unittest.mock import MagicMock

import pytest

from localm.plugins.builtin.image import plug


class _FakeJob:
    def __init__(self, cancelled: bool):
        self.cancel_requested = cancelled
        self.lines = []
        self.result = None
        self.outcomes = []

    def push(self, ev):
        self.lines.append(ev)

    def mark_outcome(self, status):
        # _generate calls this once its real deliverable is decided, before the
        # VRAM-reload tail. Recorded rather than a no-op stub so the tests below
        # can assert it fires at the right point.
        self.outcomes.append(status)


def _run_generate(monkeypatch, *, gen_result, cancelled):
    """Drive the plug's inner _generate closure and return (reload_called, job)."""
    # api_url is read unconditionally by the placement resolver
    # (resolve_media_placement(_cfg, s["api_url"])) before the reload logic under
    # test runs, so this fake must supply it or the closure dies on KeyError.
    # Placement itself stays off (no comfy_gpu_placement config).
    s = {"reload_after": True, "warning": "", "api_url": "http://127.0.0.1:8188"}
    monkeypatch.setattr(plug._backend, "settings", lambda cfg: s)
    monkeypatch.setattr(plug._backend, "ensure_available",
                        lambda s, on_progress=None: (True, "ComfyUI is up."))
    monkeypatch.setattr(plug._backend, "generate",
                        lambda *a, **k: gen_result)
    # A swap IS needed -> chat model gets unloaded, so the reload is mandatory.
    monkeypatch.setattr("localm.vram.decide_media_swap", lambda s: True)
    monkeypatch.setattr("localm.vram.unload_chat_for_media",
                        lambda job, url, label, instance_token=None: True)
    reload_calls = []
    monkeypatch.setattr(
        "localm.vram.reload_chat_after_media",
        lambda job, url, s, backend, label, instance_token=None:
            reload_calls.append(True))

    captured = {}

    class _FakeJobs:
        def start_fn(self, kind, fn, result_path=None, owner=None):
            captured["fn"] = fn
            return MagicMock(id="job1")

    request = MagicMock()
    request.app.state.jobs = _FakeJobs()
    request.app.state.self_url = "http://127.0.0.1:8642/v1"
    # imagine() stamps the job owner via principal_id(request), which reads real
    # headers/cookies - give the mock empty ones so it resolves to an anonymous
    # (None) principal instead of hashing a MagicMock attribute.
    request.headers = {}
    request.cookies = {}

    req = plug.ImagineRequest(prompt="a cat")
    asyncio.run(plug.imagine(req, request))

    job = _FakeJob(cancelled)
    captured["fn"](job)
    return bool(reload_calls), job


@pytest.mark.parametrize(
    "gen_message,cancelled",
    [
        ("Generation cancelled.", True),
        ("ComfyUI rejected the workflow", False),
    ],
    ids=["cancel", "failure"],
)
def test_reload_runs_on_cancel_or_failure(monkeypatch, gen_message, cancelled):
    reloaded, job = _run_generate(
        monkeypatch, gen_result=(False, gen_message), cancelled=cancelled)
    assert reloaded, "chat model must be reloaded after a cancelled or failed generation"
    assert job.outcomes == ["failed"]


def test_reload_runs_on_success(monkeypatch):
    reloaded, job = _run_generate(
        monkeypatch, gen_result=(True, "Image saved to out.png (seed 1)"),
        cancelled=False)
    assert reloaded
    assert job.result is not None       # success still records the artifact
    assert job.outcomes == ["done"]


def _wait_for_terminal(job, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if job.status != "running":
            return
        time.sleep(0.01)
    raise AssertionError(f"job never left 'running': {job.status}")


def test_generate_reports_done_when_reload_raises_after_success(monkeypatch):
    """A successful generation whose VRAM handover then raises (e.g. a non-comfy
    backend's free_vram()) must still land on job.status == "done", not
    "failed".

    Drives the REAL JobManager.start_fn (unlike the fake-job tests above), so
    the interaction between _generate's mark_outcome call and jobs.py's own
    except handler is exercised for real, not just _generate's return value in
    isolation."""
    s = {"reload_after": True, "warning": "", "api_url": "http://127.0.0.1:8188"}
    monkeypatch.setattr(plug._backend, "settings", lambda cfg: s)
    monkeypatch.setattr(plug._backend, "ensure_available",
                        lambda s, on_progress=None: (True, "ComfyUI is up."))
    monkeypatch.setattr(plug._backend, "generate",
                        lambda *a, **k: (True, "Image saved to out.png (seed 1)"))
    monkeypatch.setattr("localm.vram.decide_media_swap", lambda s: True)
    monkeypatch.setattr("localm.vram.unload_chat_for_media",
                        lambda job, url, label, instance_token=None: True)

    def _raising_reload(job, url, s, backend, label, instance_token=None):
        raise RuntimeError("a non-comfy backend's free_vram() blew up")

    monkeypatch.setattr("localm.vram.reload_chat_after_media", _raising_reload)

    captured = {}

    class _FakeJobs:
        def start_fn(self, kind, fn, result_path=None, owner=None):
            captured["fn"] = fn
            return MagicMock(id="job1")

    request = MagicMock()
    request.app.state.jobs = _FakeJobs()
    request.app.state.self_url = "http://127.0.0.1:8642/v1"
    request.headers = {}
    request.cookies = {}

    req = plug.ImagineRequest(prompt="a cat")
    asyncio.run(plug.imagine(req, request))

    from localm.plugins.gui.jobs import JobManager
    real_job = JobManager().start_fn("imagine", captured["fn"])
    _wait_for_terminal(real_job)

    assert real_job.status == "done", (
        f"a successful generation whose VRAM reload then raised must still "
        f"report done, not {real_job.status!r}")
    assert real_job.result is not None, "the generated artifact is still recorded"
    lines = [e.get("text", "") for e in real_job._history if e.get("type") == "line"]
    assert any("cleanup after success failed" in t for t in lines), lines
