# SPDX-License-Identifier: AGPL-3.0-or-later
"""Cancelling an image generation must still restore VRAM.

Regression: Stop unloaded the chat model (to free VRAM for the image model) but,
because the chat-model reload ran only on success, a cancel left the chat model
unloaded AND ComfyUI's model resident - the reported "fails unloading the image
model, loading chat". The reload must run on the cancel path too.
"""

import asyncio
from unittest.mock import MagicMock

from localm.plugins.builtin.image import plug


class _FakeJob:
    def __init__(self, cancelled: bool):
        self.cancel_requested = cancelled
        self.lines = []
        self.result = None

    def push(self, ev):
        self.lines.append(ev)


def _run_generate(monkeypatch, *, gen_result, cancelled):
    """Drive the plug's inner _generate closure and return (reload_called, job)."""
    s = {"reload_after": True, "warning": ""}
    monkeypatch.setattr(plug._backend, "settings", lambda cfg: s)
    monkeypatch.setattr(plug._backend, "ensure_available",
                        lambda s, on_progress=None: (True, "ComfyUI is up."))
    monkeypatch.setattr(plug._backend, "generate",
                        lambda *a, **k: gen_result)
    # A swap IS needed -> chat model gets unloaded, so the reload is mandatory.
    monkeypatch.setattr("localm.vram.decide_media_swap", lambda s: True)
    monkeypatch.setattr(plug, "_unload_chat", lambda job, url: True)
    reload_calls = []
    monkeypatch.setattr(plug, "_reload_llm",
                        lambda job, url, s: reload_calls.append(True))

    captured = {}

    class _FakeJobs:
        def start_fn(self, kind, fn, result_path=None):
            captured["fn"] = fn
            return MagicMock(id="job1")

    request = MagicMock()
    request.app.state.jobs = _FakeJobs()
    request.app.state.self_url = "http://127.0.0.1:8642/v1"

    req = plug.ImagineRequest(prompt="a cat")
    asyncio.run(plug.imagine(req, request))

    job = _FakeJob(cancelled)
    captured["fn"](job)
    return bool(reload_calls), job


def test_reload_runs_on_cancel(monkeypatch):
    reloaded, _ = _run_generate(
        monkeypatch, gen_result=(False, "Generation cancelled."), cancelled=True)
    assert reloaded, "chat model must be reloaded after a cancelled generation"


def test_reload_runs_on_failure(monkeypatch):
    reloaded, _ = _run_generate(
        monkeypatch, gen_result=(False, "ComfyUI rejected the workflow"),
        cancelled=False)
    assert reloaded, "chat model must be reloaded after a failed generation"


def test_reload_runs_on_success(monkeypatch):
    reloaded, job = _run_generate(
        monkeypatch, gen_result=(True, "Image saved to out.png (seed 1)"),
        cancelled=False)
    assert reloaded
    assert job.result is not None       # success still records the artifact
