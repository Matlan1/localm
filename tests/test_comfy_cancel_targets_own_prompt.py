# SPDX-License-Identifier: AGPL-3.0-or-later
"""Cancelling or timing out a ComfyUI job stops that job only: other jobs keep
running and keep their place in the queue. Driven against a real HTTP server
that answers like ComfyUI's /api/jobs/{id}/cancel, /queue, /interrupt and
/history routes (the jobs route can be switched off, as on an older ComfyUI)."""

from __future__ import annotations

import inspect
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from localm.media import comfy_client


class _FakeComfy:
    def __init__(self, running, pending, *, jobs_api=True, starts_on_delete=False):
        self.running = list(running)
        self.pending = list(pending)
        self.jobs_api = jobs_api
        self.starts_on_delete = starts_on_delete
        self.posts = []
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def _reply(self, body, status=200):
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path == "/queue":
                    self._reply({
                        "queue_running": [[i, pid, {}, {}, []] for i, pid in enumerate(fake.running)],
                        "queue_pending": [[10 + i, pid, {}, {}, []] for i, pid in enumerate(fake.pending)],
                    })
                else:
                    self._reply({})

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}")
                fake.posts.append((self.path, body))
                if self.path.startswith("/api/jobs/") and self.path.endswith("/cancel"):
                    if not fake.jobs_api:
                        self._reply({"error": "not found"}, status=404)
                        return
                    pid = self.path[len("/api/jobs/"):-len("/cancel")]
                    cancelled = pid in fake.running or pid in fake.pending
                    fake.running = [p for p in fake.running if p != pid]
                    fake.pending = [p for p in fake.pending if p != pid]
                    self._reply({"cancelled": cancelled})
                    return
                if self.path == "/queue":
                    if body.get("clear"):
                        fake.pending.clear()
                    for pid in body.get("delete") or []:
                        if pid in fake.pending:
                            fake.pending.remove(pid)
                            if fake.starts_on_delete:
                                fake.running.append(pid)
                elif self.path == "/interrupt":
                    target = body.get("prompt_id")
                    fake.running = [p for p in fake.running if target not in (None, p)]
                self._reply({})

            def log_message(self, *a):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def comfy_factory():
    made = []

    def make(running, pending, **kw):
        fake = _FakeComfy(running, pending, **kw)
        made.append(fake)
        return fake

    yield make
    for fake in made:
        fake.close()


def _poll(fake, prompt_id, **kw):
    return comfy_client.comfy_poll_until_done(
        fake.url, prompt_id, sleep_seconds=0.01, history_timeout=2.0, **kw)


@pytest.mark.parametrize("jobs_api", [True, False])
def test_cancel_of_a_queued_prompt_leaves_the_running_job_alone(comfy_factory, jobs_api):
    fake = comfy_factory(running=["other"], pending=["mine", "later"], jobs_api=jobs_api)
    status, _ = _poll(fake, "mine", max_poll_seconds=30, cancel_check=lambda: True)
    assert status == comfy_client.POLL_CANCELLED
    assert fake.running == ["other"]
    assert fake.pending == ["later"]
    assert not any(path == "/interrupt" for path, _ in fake.posts)


@pytest.mark.parametrize("jobs_api", [True, False])
def test_cancel_of_the_running_prompt_stops_only_it(comfy_factory, jobs_api):
    fake = comfy_factory(running=["mine"], pending=["later"], jobs_api=jobs_api)
    status, _ = _poll(fake, "mine", max_poll_seconds=30, cancel_check=lambda: True)
    assert status == comfy_client.POLL_CANCELLED
    assert fake.running == [] and fake.pending == ["later"]
    if not jobs_api:
        assert ("/interrupt", {"prompt_id": "mine"}) in fake.posts


@pytest.mark.parametrize("jobs_api", [True, False])
def test_a_timeout_stops_the_prompt_it_gave_up_on(comfy_factory, jobs_api):
    fake = comfy_factory(running=["mine"], pending=["later"], jobs_api=jobs_api)
    status, _ = _poll(fake, "mine", max_poll_seconds=0.2)
    assert status == comfy_client.POLL_TIMEOUT
    assert fake.running == [] and fake.pending == ["later"]


@pytest.mark.parametrize("jobs_api", [True, False])
def test_a_timeout_while_still_queued_removes_it_from_the_queue(comfy_factory, jobs_api):
    fake = comfy_factory(running=["other"], pending=["mine"], jobs_api=jobs_api)
    status, _ = _poll(fake, "mine", max_poll_seconds=0.2)
    assert status == comfy_client.POLL_TIMEOUT
    assert fake.running == ["other"] and fake.pending == []


def test_the_atomic_jobs_route_is_used_when_comfyui_has_it(comfy_factory):
    fake = comfy_factory(running=["other"], pending=["mine"])
    assert comfy_client.cancel_comfy_prompt(fake.url, "mine") is True
    assert [path for path, _ in fake.posts] == ["/api/jobs/mine/cancel"]


def test_a_prompt_that_starts_while_being_dequeued_is_interrupted(comfy_factory):
    fake = comfy_factory(running=[], pending=["mine"], jobs_api=False, starts_on_delete=True)
    assert comfy_client.cancel_comfy_prompt(fake.url, "mine") is True
    assert ("/interrupt", {"prompt_id": "mine"}) in fake.posts
    assert fake.running == [] and fake.pending == []


def test_a_finished_prompt_sends_no_stop_request(comfy_factory):
    fake = comfy_factory(running=["other"], pending=["later"], jobs_api=False)
    assert comfy_client.cancel_comfy_prompt(fake.url, "mine") is False
    assert [path for path, _ in fake.posts] == ["/api/jobs/mine/cancel"]
    assert fake.running == ["other"] and fake.pending == ["later"]


def test_an_unreadable_queue_cancels_nothing(comfy_factory):
    fake = comfy_factory(running=["other"], pending=["mine"])
    fake.close()
    assert comfy_client.cancel_comfy_prompt(fake.url, "mine") is False


def test_an_image_waits_up_to_30_minutes_for_comfyui():
    from localm.image_gen import comfy
    default = inspect.signature(comfy.generate_image).parameters["max_poll_seconds"].default
    assert default == 1800
