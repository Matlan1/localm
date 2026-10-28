# SPDX-License-Identifier: AGPL-3.0-or-later
"""localm's own ComfyUI is restarted, never duplicated, after a GPU fault.

Every ComfyUI here is a REAL process: a small stand-in HTTP server launched
through ``ensure_comfy`` with the test interpreter, so the handle, the process
tree (a venv ``python.exe`` is a launcher whose child listens on the port), the
port, the kill and the relaunch are all real. Each stand-in records its own
start in a state folder, and a stand-in that cannot bind its port records that
too, so a second instance started into a held port is visible. The stand-in
binds the way ComfyUI's aiohttp server does: SO_REUSEADDR on POSIX (asyncio's
default there), exclusive use on Windows; neither lets a second listener bind
while the first one is alive.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import textwrap
import time
import urllib.request

import psutil
import pytest
from fastapi.testclient import TestClient

from localm.media import comfy_client as cc

_STANDIN = textwrap.dedent('''
    import http.server
    import json
    import os
    import socket
    import sys

    port = int(sys.argv[1])
    state = sys.argv[2]
    broken = {"on": False}


    class Server(http.server.ThreadingHTTPServer):
        allow_reuse_address = sys.platform != "win32"

        def server_bind(self):
            if sys.platform == "win32":
                self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            super().server_bind()


    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body):
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/system_stats":
                if broken["on"]:
                    self._send(500, {"error": "HIP error: unspecified launch failure"})
                else:
                    self._send(200, {"system": {}, "devices": []})
            elif self.path == "/crash":
                broken["on"] = True
                self._send(200, {})
            elif self.path.startswith("/history/"):
                pid = self.path.rsplit("/", 1)[1]
                self._send(200, {pid: {"outputs": {}, "status": {
                    "status_str": "error",
                    "messages": [["execution_error", {
                        "node_type": "VAEDecode",
                        "exception_message":
                            "HIP error: unspecified launch failure\\n"
                            "Search for hipErrorLaunchFailure for more information."}]]}}})
            else:
                self._send(200, {})

        def do_POST(self):
            self._send(200, {})


    try:
        srv = Server(("127.0.0.1", port), Handler)
    except OSError:
        open(os.path.join(state, f"bindfail-{os.getpid()}"), "w").close()
        sys.exit(3)
    open(os.path.join(state, f"up-{os.getpid()}"), "w").close()
    srv.serve_forever()
''')


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _records(state, prefix):
    """The pids that wrote a *prefix* record, oldest record first."""
    names = [n for n in os.listdir(state) if n.startswith(prefix + "-")]
    names.sort(key=lambda n: os.stat(os.path.join(state, n)).st_mtime_ns)
    return [int(n.split("-", 1)[1]) for n in names]


def _get(url, timeout=5):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.status


@pytest.fixture
def standin(tmp_path):
    script = tmp_path / "standin_comfy.py"
    script.write_text(_STANDIN, encoding="utf-8")
    state = tmp_path / "state"
    state.mkdir()
    port = _free_port()
    url = f"http://127.0.0.1:{port}"
    launch_cmd = f'"{sys.executable}" "{script}" {port} "{state}"'
    own: list = []
    cc.clear_comfy_device_error(url)
    cc.mark_comfy_dead(url)
    yield {"url": url, "port": port, "state": str(state), "launch_cmd": launch_cmd,
           "script": str(script), "workdir": str(tmp_path), "own": own}
    proc = cc._take_spawned(url)
    if proc is not None and proc.poll() is None:
        cc._kill_process_tree(proc)
    for p in own:
        if p.poll() is None:
            cc._kill_process_tree(p)
    cc.clear_comfy_device_error(url)
    cc.mark_comfy_dead(url)


def _ensure(s, lines):
    return cc.ensure_comfy(api_url=s["url"], on_progress=lines.append,
                           launch_cmd=s["launch_cmd"], workdir=s["workdir"],
                           wait_seconds=30)


def _wait_gone(pid, seconds=10.0):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if not psutil.pid_exists(pid):
            return True
        time.sleep(0.1)
    return not psutil.pid_exists(pid)


def test_own_comfy_answering_500_is_restarted_not_duplicated(standin, monkeypatch):
    s = standin
    monkeypatch.setattr(cc, "_CONFIRM_TTL_SECONDS", 0.0)
    lines: list = []
    ok, msg = _ensure(s, lines)
    assert ok, msg
    first_listener = cc.comfy_port_holder(s["url"])["pid"]
    assert _records(s["state"], "up") == [first_listener]

    assert _get(s["url"] + "/crash") == 200
    assert cc.comfy_failure_detail(s["url"]) == "it answers HTTP 500"

    spent: list = []
    real_launch = cc._launch_and_wait

    def spy(*a, **k):
        spent.append(k.get("already_spent", 0.0))
        return real_launch(*a, **k)

    monkeypatch.setattr(cc, "_launch_and_wait", spy)
    lines.clear()
    ok, msg = _ensure(s, lines)

    assert ok, msg
    assert len(spent) == 1 and 0.0 < spent[0] < 20.0, spent
    assert _records(s["state"], "bindfail") == []
    ups = _records(s["state"], "up")
    assert len(ups) == 2, ups
    assert _wait_gone(first_listener)
    assert cc.comfy_port_holder(s["url"])["pid"] == ups[1] != first_listener
    restart_lines = [ln for ln in lines if "restarting it" in ln]
    assert restart_lines == [
        "The ComfyUI localm launched stopped working (it answers HTTP 500); "
        "restarting it."]
    assert _get(s["url"] + "/system_stats") == 200


def test_device_error_in_a_job_restarts_own_comfy_before_the_next_job(standin):
    s = standin
    lines: list = []
    ok, msg = _ensure(s, lines)
    assert ok, msg
    first_listener = cc.comfy_port_holder(s["url"])["pid"]

    status, detail = cc.comfy_poll_until_done(
        s["url"], "job1", max_poll_seconds=10, sleep_seconds=0.2)
    assert status == cc.POLL_EXEC_ERROR
    assert "HIP error: unspecified launch failure" in detail
    assert cc.comfy_device_error(s["url"]) == "HIP error: unspecified launch failure"
    assert _get(s["url"] + "/system_stats") == 200

    lines.clear()
    ok, msg = _ensure(s, lines)

    assert ok, msg
    assert _records(s["state"], "bindfail") == []
    assert len(_records(s["state"], "up")) == 2
    assert _wait_gone(first_listener)
    assert [ln for ln in lines if "restarting it" in ln] == [
        "The ComfyUI localm launched stopped working (a job hit a GPU error: "
        "HIP error: unspecified launch failure); restarting it."]
    assert cc.comfy_device_error(s["url"]) is None


def test_foreign_comfy_answering_500_is_left_alone_with_a_clear_message(standin):
    s = standin
    foreign = subprocess.Popen(
        [sys.executable, s["script"], str(s["port"]), s["state"]],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    s["own"].append(foreign)
    deadline = time.monotonic() + 20
    while not _records(s["state"], "up") and time.monotonic() < deadline:
        time.sleep(0.1)
    listener = _records(s["state"], "up")[0]
    assert _get(s["url"] + "/crash") == 200

    holder = cc.comfy_port_holder(s["url"])
    assert holder["pid"] == listener
    assert holder["localm_launched"] is False

    lines: list = []
    ok, msg = _ensure(s, lines)

    assert ok is False
    assert f"process {listener}" in msg
    assert "localm did not start it and will not stop it" in msg
    assert not [ln for ln in lines if "restarting it" in ln]
    assert _records(s["state"], "up") == [listener]
    assert _records(s["state"], "bindfail") == []
    assert psutil.pid_exists(listener)
    assert foreign.poll() is None


def test_ownership_matches_between_the_job_path_status_and_stop(standin, monkeypatch):
    s = standin
    lines: list = []
    ok, msg = _ensure(s, lines)
    assert ok, msg
    root = cc.spawned_pid(s["url"])
    listener = cc.comfy_port_holder(s["url"])["pid"]
    assert _get(s["url"] + "/crash") == 200

    holder = cc.comfy_port_holder(s["url"])
    assert holder["pid"] == listener
    assert holder["localm_launched"] is True
    assert cc.is_own_comfy_process(s["url"], listener)
    tree = {root} | {c.pid for c in psutil.Process(root).children(recursive=True)}
    assert listener in tree
    held_msg = cc.comfy_port_held_message(s["url"], holder)
    assert "did not start it" not in held_msg
    assert "the ComfyUI localm launched" in held_msg

    from localm.image_gen import comfy as ic
    monkeypatch.setattr(ic, "default_api_url", lambda: s["url"])
    from localm.inference.http_server import create_app
    app = create_app(None)
    client = TestClient(app)
    r = client.get("/v1/comfy/status",
                   headers={"Authorization": f"Bearer {app.state.shell_token}"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["alive"] is False
    assert body["launched_by_localm"] is True
    assert "port_held" not in body

    ok, stop_msg = cc.stop_comfy(s["url"])
    assert ok
    assert stop_msg == "Stopped the ComfyUI that localm launched."
    assert _wait_gone(listener)
    assert cc._wait_for_port_release(s["url"])


def test_stop_does_not_claim_a_stop_for_an_already_exited_handle(standin, monkeypatch):
    s = standin
    monkeypatch.setattr(cc, "interrupt_comfy", lambda url: False)
    monkeypatch.setattr(cc, "free_comfy_vram", lambda url=None: False)
    done = subprocess.Popen([sys.executable, "-c", "pass"])
    done.wait(timeout=30)
    cc._remember_spawned(s["url"], done)
    ok, msg = cc.stop_comfy(s["url"])
    assert ok
    assert msg == "The ComfyUI localm launched had already exited."


def test_launch_wait_is_shortened_by_the_time_recovery_spent(tmp_path, monkeypatch):
    """ensure_comfy's bounded callers allow comfy_launch_wait_seconds plus a
    buffer; time spent stopping the old ComfyUI comes off the launch wait."""
    port = _free_port()
    url = f"http://127.0.0.1:{port}"
    never_up = tmp_path / "never_up.py"
    never_up.write_text("import time\ntime.sleep(60)\n", encoding="utf-8")
    monkeypatch.setattr(cc, "_comfy_alive", lambda *a, **k: False)
    lines: list = []
    t0 = time.monotonic()
    try:
        ok, msg = cc._launch_and_wait(
            url, f'"{sys.executable}" "{never_up}"', str(tmp_path), 30,
            lines.append, already_spent=28.5)
        elapsed = time.monotonic() - t0
    finally:
        proc = cc._take_spawned(url)
        if proc is not None and proc.poll() is None:
            cc._kill_process_tree(proc)
    assert ok is False
    assert "did not come up" in msg
    assert elapsed < 10.0, elapsed


@pytest.mark.parametrize("detail, expected", [
    ("HIP error: unspecified launch failure", True),
    ("torch.AcceleratorError: CUDA error: an illegal memory access was encountered", True),
    ("HIP out of memory. Tried to allocate 2.00 GiB", False),
    ("CUDA error: out of memory", False),
    ("Value not in list: ckpt_name", False),
    ("", False),
])
def test_device_error_classification(detail, expected):
    assert cc.is_comfy_device_error(detail) is expected
