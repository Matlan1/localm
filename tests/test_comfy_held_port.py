# SPDX-License-Identifier: AGPL-3.0-or-later
"""A ComfyUI port held by a process that accepts connections but does not
answer (a ComfyUI hung after a GPU crash) is named at once: restart/launch
neither relaunch into the held port nor wait out the launch budget, and the
process is never signalled.

The holder is a REAL listening socket in this process that accepts and never
replies, so ``comfy_port_holder`` is exercised against a live port and the
expected pid is this process's own."""

from __future__ import annotations

import os
import socket
import threading

import pytest
from fastapi.testclient import TestClient

from localm.media import comfy_client


@pytest.fixture
def hung_listener():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen(8)
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
    yield srv.getsockname()[1]
    stop.set()
    t.join(2)
    for c in held:
        c.close()
    srv.close()


@pytest.fixture
def free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_holder_of_a_live_port_is_named_by_pid(hung_listener):
    holder = comfy_client.comfy_port_holder(f"http://127.0.0.1:{hung_listener}")
    assert holder is not None
    assert holder["port"] == hung_listener
    assert holder["pid"] == os.getpid()
    assert holder["name"]


def test_a_free_port_has_no_holder(free_port):
    assert comfy_client.comfy_port_holder(f"http://127.0.0.1:{free_port}") is None


def test_a_remote_address_is_never_probed(monkeypatch):
    import socket as _socket

    def boom(*a, **k):
        raise AssertionError("connected to a non-loopback address")
    monkeypatch.setattr(_socket, "create_connection", boom)
    assert comfy_client.comfy_port_holder("http://203.0.113.9:8188") is None


def test_launch_refuses_a_held_port_at_once_and_spawns_nothing(hung_listener, monkeypatch,
                                                               tmp_path):
    import subprocess
    spawned: list = []
    monkeypatch.setattr(subprocess, "Popen",
                        lambda *a, **k: spawned.append(a) or (_ for _ in ()).throw(
                            AssertionError("launched into a held port")))
    monkeypatch.setattr(comfy_client, "_comfy_alive", lambda *a, **k: False)
    monkeypatch.setattr(comfy_client, "spawned_pid", lambda *a, **k: None)
    bat = tmp_path / "launch.bat"
    bat.write_text("echo hi\n", encoding="utf-8")
    ok, message = comfy_client.ensure_comfy(
        api_url=f"http://127.0.0.1:{hung_listener}", launch_cmd=str(bat),
        workdir=str(tmp_path), wait_seconds=30)
    assert ok is False
    assert str(os.getpid()) in message
    assert str(hung_listener) in message
    assert "will not stop it" in message
    assert spawned == []


def test_launch_proceeds_when_the_port_is_free(free_port, monkeypatch, tmp_path):
    import subprocess
    spawned: list = []

    class _Proc:
        pid = 1
        returncode = 0

        def poll(self):
            return None

    monkeypatch.setattr(subprocess, "Popen",
                        lambda *a, **k: spawned.append(a) or _Proc())
    monkeypatch.setattr(comfy_client, "_remember_spawned", lambda *a, **k: None)
    monkeypatch.setattr(comfy_client.time, "sleep", lambda *_: None)
    bat = tmp_path / "launch.bat"
    bat.write_text("echo hi\n", encoding="utf-8")
    monkeypatch.setattr(comfy_client, "_comfy_alive", lambda *a, **k: bool(spawned))
    monkeypatch.setattr(comfy_client, "is_comfy_confirmed", lambda *a, **k: False)
    ok, _ = comfy_client.ensure_comfy(
        api_url=f"http://127.0.0.1:{free_port}", launch_cmd=str(bat),
        workdir=str(tmp_path), wait_seconds=30)
    assert ok is True
    assert len(spawned) == 1


def test_status_route_names_the_holder_when_the_port_does_not_answer(
        hung_listener, tmp_path, monkeypatch):
    import localm.config as cfg
    from localm.image_gen import comfy as ic
    home = tmp_path / ".localm"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.setattr(cfg, "HOME_DIR", home)
    monkeypatch.setattr(cfg, "MODELS_DIR", home / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", home / "registry.json")
    url = f"http://127.0.0.1:{hung_listener}"
    monkeypatch.setattr(ic, "default_api_url", lambda: url)
    monkeypatch.setattr(ic, "_comfy_alive", lambda u, timeout=1.0: False)
    from localm.inference.http_server import create_app
    app = create_app(None)
    client = TestClient(app)
    r = client.get("/v1/comfy/status",
                   headers={"Authorization": f"Bearer {app.state.shell_token}"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["alive"] is False
    assert body["port_held"]["pid"] == os.getpid()
    assert str(os.getpid()) in body["port_held"]["message"]
