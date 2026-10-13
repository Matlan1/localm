# SPDX-License-Identifier: AGPL-3.0-or-later
"""Cancelling or timing out a ComfyUI job stops that job only: other jobs keep
running and keep their place in the queue. Driven against a real HTTP server
that answers like ComfyUI's /queue, /interrupt and /history routes."""

from __future__ import annotations

import inspect
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from localm.media import comfy_client


class _FakeComfy:
    def __init__(self, running, pending):
        self.running = list(running)
        self.pending = list(pending)
        self.posts = []
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def _reply(self, body):
                data = json.dumps(body).encode()
                self.send_response(200)
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
                if self.path == "/queue":
                    if body.get("clear"):
                        fake.pending.clear()
                    for pid in body.get("delete") or []:
                        if pid in fake.pending:
                            fake.pending.remove(pid)
                elif self.path == "/interrupt":
                    target = body.get("prompt_id")
                    if target is None or target in fake.running:
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

    def make(running, pending):
        fake = _FakeComfy(running, pending)
        made.append(fake)
        return fake

    yield make
    for fake in made:
        fake.close()


def _poll(fake, prompt_id, **kw):
    return comfy_client.comfy_poll_until_done(
        fake.url, prompt_id, sleep_seconds=0.01, history_timeout=2.0, **kw)


def test_cancel_of_a_queued_prompt_leaves_the_running_job_alone(comfy_factory):
    fake = comfy_factory(running=["other"], pending=["mine", "later"])
    status, _ = _poll(fake, "mine", max_poll_seconds=30, cancel_check=lambda: True)
    assert status == comfy_client.POLL_CANCELLED
    assert fake.running == ["other"]
    assert fake.pending == ["later"]
    assert not any(path == "/interrupt" for path, _ in fake.posts)


def test_cancel_of_the_running_prompt_interrupts_only_it(comfy_factory):
    fake = comfy_factory(running=["mine"], pending=["later"])
    status, _ = _poll(fake, "mine", max_poll_seconds=30, cancel_check=lambda: True)
    assert status == comfy_client.POLL_CANCELLED
    assert ("/interrupt", {"prompt_id": "mine"}) in fake.posts
    assert fake.running == [] and fake.pending == ["later"]


def test_a_timeout_stops_the_prompt_it_gave_up_on(comfy_factory):
    fake = comfy_factory(running=["mine"], pending=["later"])
    status, _ = _poll(fake, "mine", max_poll_seconds=0.2)
    assert status == comfy_client.POLL_TIMEOUT
    assert ("/interrupt", {"prompt_id": "mine"}) in fake.posts
    assert fake.running == [] and fake.pending == ["later"]


def test_a_timeout_while_still_queued_removes_it_from_the_queue(comfy_factory):
    fake = comfy_factory(running=["other"], pending=["mine"])
    status, _ = _poll(fake, "mine", max_poll_seconds=0.2)
    assert status == comfy_client.POLL_TIMEOUT
    assert fake.running == ["other"] and fake.pending == []


def test_an_unreadable_queue_cancels_nothing(comfy_factory):
    fake = comfy_factory(running=["other"], pending=["mine"])
    fake.close()
    assert comfy_client.cancel_comfy_prompt(fake.url, "mine") is False


def test_an_image_waits_up_to_30_minutes_for_comfyui():
    from localm.image_gen import comfy
    default = inspect.signature(comfy.generate_image).parameters["max_poll_seconds"].default
    assert default == 1800
