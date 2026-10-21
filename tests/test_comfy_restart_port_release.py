# SPDX-License-Identifier: AGPL-3.0-or-later
"""Restarting a ComfyUI that localm launched waits for the killed process to
release its port before the port check, so the old listener is never named as
a foreign holder. The listener is a REAL socket that closes a moment after the
kill returns, standing in for a process that outlives ``taskkill``."""

from __future__ import annotations

import socket
import subprocess
import threading

import pytest

from localm.media import comfy_client

CLOSE_AFTER_S = 1.2


class _Proc:
    pid = 4242
    returncode = 0

    def poll(self):
        return None


@pytest.fixture
def lingering_listener(monkeypatch):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen(8)
    port = srv.getsockname()[1]
    srv.settimeout(0.1)
    state = {"open": True, "timer": None}

    def serve():
        while state["open"]:
            try:
                conn, _ = srv.accept()
            except OSError:
                continue
            conn.close()
        srv.close()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()

    def close():
        state["open"] = False

    def kill_returns_before_the_listener_is_gone(proc):
        state["timer"] = threading.Timer(CLOSE_AFTER_S, close)
        state["timer"].start()

    monkeypatch.setattr(comfy_client, "_kill_process_tree",
                        kill_returns_before_the_listener_is_gone)
    yield port
    state["open"] = False
    if state["timer"] is not None:
        state["timer"].cancel()
    thread.join(2)


def test_restart_waits_for_the_killed_listener_instead_of_refusing(
        lingering_listener, monkeypatch, tmp_path):
    url = f"http://127.0.0.1:{lingering_listener}"
    spawned: list = []
    monkeypatch.setitem(comfy_client._spawned_procs, url, _Proc())
    monkeypatch.setattr(subprocess, "Popen",
                        lambda *a, **k: spawned.append(a) or _Proc())
    monkeypatch.setattr(comfy_client, "_remember_spawned", lambda *a, **k: None)
    monkeypatch.setattr(comfy_client, "interrupt_comfy", lambda *a, **k: None)
    monkeypatch.setattr(comfy_client, "free_comfy_vram", lambda *a, **k: None)
    monkeypatch.setattr(comfy_client, "mark_comfy_dead", lambda *a, **k: None)
    monkeypatch.setattr(comfy_client, "_comfy_alive", lambda *a, **k: bool(spawned))
    monkeypatch.setattr(comfy_client, "is_comfy_confirmed", lambda *a, **k: False)
    bat = tmp_path / "launch.bat"
    bat.write_text("echo hi\n", encoding="utf-8")
    ok, message = comfy_client.restart_comfy(
        api_url=url, launch_cmd=str(bat), workdir=str(tmp_path), wait_seconds=30)
    assert "will not stop it" not in message, message
    assert ok is True, message
    assert len(spawned) == 1


def test_a_port_that_never_frees_is_reported_after_the_wait(lingering_listener):
    url = f"http://127.0.0.1:{lingering_listener}"
    assert comfy_client._wait_for_port_release(url, seconds=0.3) is False


def test_the_port_wait_fits_inside_the_restart_routes_slack():
    assert comfy_client._PORT_RELEASE_WAIT_S < 30.0
