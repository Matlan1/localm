# SPDX-License-Identifier: AGPL-3.0-or-later
"""A media generation says what it is waiting for: its place in the ComfyUI
queue while other jobs run ahead of it, the elapsed rendering time once it
runs, and every preflight notice (a substituted model, ComfyUI starting) on
the job stream instead of the server console."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from localm.image_gen import comfy
from localm.media import comfy_client


@pytest.fixture
def comfy_queue():
    """A local HTTP server answering ``GET /queue`` with ``state["queue"]``."""
    state = {"queue": {"queue_running": [], "queue_pending": []}, "status": 200}

    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps(state["queue"]).encode()
            self.send_response(state["status"])
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    server = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", state
    finally:
        server.shutdown()
        server.server_close()


def _job(number, prompt_id):
    return [number, prompt_id, {}, {}, []]


class TestJobsAhead:
    def test_running_job_has_none_ahead(self, comfy_queue):
        url, state = comfy_queue
        state["queue"] = {"queue_running": [_job(7, "mine")], "queue_pending": []}
        assert comfy_client.comfy_jobs_ahead(url, "mine") == 0

    def test_queued_job_counts_running_and_earlier_pending(self, comfy_queue):
        url, state = comfy_queue
        state["queue"] = {"queue_running": [_job(5, "other")],
                          "queue_pending": [_job(9, "later"), _job(8, "mine"),
                                            _job(6, "earlier")]}
        assert comfy_client.comfy_jobs_ahead(url, "mine") == 2

    def test_unlisted_or_unreadable_is_unknown(self, comfy_queue):
        url, state = comfy_queue
        assert comfy_client.comfy_jobs_ahead(url, "mine") is None
        state["status"] = 500
        assert comfy_client.comfy_jobs_ahead(url, "mine") is None


class TestHeartbeat:
    def test_queue_position_is_said_at_once_then_rendering(self, comfy_queue):
        url, state = comfy_queue
        said = []
        tick = comfy_client.comfy_wait_heartbeat(url, "mine", said.append)
        state["queue"] = {"queue_running": [_job(1, "other")],
                          "queue_pending": [_job(2, "mine")]}
        tick(0.0)
        tick(2.0)
        state["queue"] = {"queue_running": [_job(2, "mine")], "queue_pending": []}
        tick(4.0)
        tick(6.0)
        tick(19.5)
        assert said == ["Waiting for ComfyUI to finish 1 other job...",
                        "Rendering… (4s elapsed)",
                        "Rendering… (19s elapsed)"]

    def test_a_job_running_from_the_start_keeps_the_15s_cadence(self, comfy_queue):
        url, state = comfy_queue
        state["queue"] = {"queue_running": [_job(1, "mine")], "queue_pending": []}
        said = []
        tick = comfy_client.comfy_wait_heartbeat(url, "mine", said.append)
        for t in (0.0, 2.0, 14.0, 15.0, 17.0):
            tick(t)
        assert said == ["Rendering… (15s elapsed)"]

    def test_plural_and_count_changes_are_said(self, comfy_queue):
        url, state = comfy_queue
        said = []
        tick = comfy_client.comfy_wait_heartbeat(url, "mine", said.append)
        state["queue"] = {"queue_running": [_job(1, "a")],
                          "queue_pending": [_job(2, "b"), _job(3, "mine")]}
        tick(0.0)
        state["queue"] = {"queue_running": [_job(2, "b")],
                          "queue_pending": [_job(3, "mine")]}
        tick(2.0)
        assert said == ["Waiting for ComfyUI to finish 2 other jobs...",
                        "Waiting for ComfyUI to finish 1 other job..."]


class TestImageNoticesReachTheJob:
    def test_comfy_start_and_preflight_notices_go_to_on_progress(
            self, tmp_path, monkeypatch):
        wf = tmp_path / "wf.json"
        wf.write_text(json.dumps({"3": {"class_type": "KSampler", "inputs": {}}}),
                      encoding="utf-8")
        monkeypatch.setattr(comfy, "workflow_path", lambda: wf)

        def _ensure(api_url, on_progress=None, **kw):
            on_progress("Starting ComfyUI...")
            return True, ""

        def _preflight(workflow, api_url, on_progress=None):
            on_progress("Model 'a.gguf' is not installed; substituting 'b.gguf'")
            return False, "stop here"

        monkeypatch.setattr(comfy, "ensure_comfy", _ensure)
        monkeypatch.setattr(comfy, "_build_image_workflow",
                            lambda workflow, **kw: (True, "", None))
        monkeypatch.setattr(comfy, "preflight_models", _preflight)
        lines = []
        ok, msg = comfy.generate_image("a cat", tmp_path / "out.png",
                                       on_progress=lines.append)
        assert (ok, msg) == (False, "stop here")
        assert lines == ["Starting ComfyUI...",
                         "Model 'a.gguf' is not installed; substituting 'b.gguf'"]

    def test_the_fast_dequant_note_is_said(self, monkeypatch):
        monkeypatch.setattr(comfy, "apply_fast_dequant", lambda wf: True)
        said = []
        ok, _msg, _up = comfy._build_image_workflow(
            {"6": {"class_type": "CLIPTextEncode", "inputs": {"text": ""}}},
            prompt="p", api_url="http://127.0.0.1:1", guidance=None,
            negative_prompt=None, cfg=None, seed=1, clip_name1=None,
            clip_name2=None, lora_name=None, lora_strength_model=1.0,
            lora_strength_clip=0.5, input_image=None, denoise=None,
            fast_dequant=True, say=said.append)
        assert any("fast fp16 GGUF dequant" in s for s in said)
