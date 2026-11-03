# SPDX-License-Identifier: AGPL-3.0-or-later
"""``stop_comfy`` and ``restart_comfy`` against a ComfyUI port held by a process
that accepts connections and never answers, and that localm did not launch.

Both refuse at once with the held-port message, like ``ensure_comfy``, instead
of sending the abort and free-VRAM requests that each wait out their own
timeout on a listener that never replies. The holder is a REAL listening socket
in this process, so the whole path runs against a live port."""

from __future__ import annotations

import os
import socket
import subprocess
import threading
import time

import pytest

from localm.media import comfy_client

_FAST_REFUSAL_S = 20.0


@pytest.fixture
def hung_url():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen(16)
    held: list = []
    stop = threading.Event()

    def accept_and_hold():
        srv.settimeout(0.2)
        while not stop.is_set():
            try:
                conn, _ = srv.accept()
            except OSError:
                continue
            held.append(conn)

    t = threading.Thread(target=accept_and_hold, daemon=True)
    t.start()
    url = f"http://127.0.0.1:{srv.getsockname()[1]}"
    yield url
    stop.set()
    t.join(2)
    for c in held:
        c.close()
    srv.close()


@pytest.fixture
def no_spawn(monkeypatch):
    spawned: list = []

    def refuse(*a, **k):
        spawned.append(a)
        raise AssertionError("spawned a process against a held port")

    monkeypatch.setattr(subprocess, "Popen", refuse)
    return spawned


def _assert_names_the_holder(message: str, url: str) -> None:
    assert url.rsplit(":", 1)[1] in message
    assert str(os.getpid()) in message
    assert "will not stop it" in message


def test_stop_refuses_a_foreign_hung_port_at_once(hung_url, no_spawn):
    started = time.monotonic()
    ok, message = comfy_client.stop_comfy(hung_url)
    elapsed = time.monotonic() - started
    assert ok is False
    _assert_names_the_holder(message, hung_url)
    assert elapsed < _FAST_REFUSAL_S
    assert no_spawn == []


def test_restart_refuses_a_foreign_hung_port_at_once(hung_url, no_spawn, tmp_path):
    bat = tmp_path / "launch.bat"
    bat.write_text("echo hi\n", encoding="utf-8")
    started = time.monotonic()
    ok, message = comfy_client.restart_comfy(
        api_url=hung_url, launch_cmd=str(bat), workdir=str(tmp_path),
        wait_seconds=30)
    elapsed = time.monotonic() - started
    assert ok is False
    _assert_names_the_holder(message, hung_url)
    assert elapsed < _FAST_REFUSAL_S
    assert no_spawn == []


def test_the_refusal_budget_is_far_inside_the_abort_requests_it_skips():
    abort_requests_worst_case_s = 10.0 + 10.0 + 30.0
    assert _FAST_REFUSAL_S * 2 < abort_requests_worst_case_s


def test_stop_of_a_free_port_still_reports_not_running():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    ok, message = comfy_client.stop_comfy(f"http://127.0.0.1:{port}")
    assert ok is True
    assert message == "ComfyUI is not running."
