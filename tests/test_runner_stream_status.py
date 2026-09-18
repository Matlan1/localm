# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for ModelRunner.chat_stream status envelope handling."""

import multiprocessing as mp
import threading
from typing import List

from localm.inference.backends.llamacpp._runner import ModelRunner


class _FakeProc:
    def __init__(self):
        self.terminated = False
        self.exitcode = 0

    def is_alive(self):
        return not self.terminated

    def terminate(self):
        self.terminated = True

    def join(self, timeout=None):
        return None


def _make_runner() -> ModelRunner:
    ctx = mp.get_context("spawn")
    r = ModelRunner()
    r._req_q, r._resp_q, r._ctrl_q = ctx.Queue(), ctx.Queue(), ctx.Queue()
    r._proc = _FakeProc()
    return r


def _fake_child_with_status(r, stop, *, statuses: List[str], tokens: List[str]):
    while not stop.is_set():
        try:
            cmd = r._req_q.get(timeout=0.05)
        except Exception:
            continue
        if cmd[0] != "chat_stream":
            continue
        for s in statuses:
            r._resp_q.put(("status", s))
        for t in tokens:
            r._resp_q.put(("chunk", t))
        r._resp_q.put(("done", {"finish_reason": "stop"}))
        return


def test_model_runner_chat_stream_relays_status():
    r = _make_runner()
    stop = threading.Event()
    child = threading.Thread(
        target=_fake_child_with_status,
        args=(r, stop),
        kwargs=dict(
            statuses=["Processing prompt...", "Encoding image (GPU)..."],
            tokens=["Hello", " world"],
        ),
        daemon=True,
    )
    child.start()
    received_statuses = []
    try:
        tokens = list(r.chat_stream(
            messages=[],
            on_status=received_statuses.append,
        ))
    finally:
        stop.set()
        child.join(2)

    assert tokens == ["Hello", " world"]
    assert received_statuses == [
        "Processing prompt...",
        "Encoding image (GPU)...",
    ]


def test_model_runner_on_status_exception_does_not_abort_stream():
    r = _make_runner()
    stop = threading.Event()
    child = threading.Thread(
        target=_fake_child_with_status,
        args=(r, stop),
        kwargs=dict(
            statuses=["Processing prompt..."],
            tokens=["token1"],
        ),
        daemon=True,
    )
    child.start()

    def _exploding_callback(s):
        raise ValueError("callback exploded")

    try:
        tokens = list(r.chat_stream(
            messages=[],
            on_status=_exploding_callback,
        ))
    finally:
        stop.set()
        child.join(2)

    assert tokens == ["token1"]
