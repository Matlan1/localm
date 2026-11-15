# SPDX-License-Identifier: AGPL-3.0-or-later
"""`localm comfy stop` without a localm server, and the `localm image` hint.

`localm image` launches ComfyUI in its own process and leaves it running when
that process exits, so no live process holds its handle. `_launch_and_wait`
writes a launch record (`comfy_launch_record`), and `comfy stop` stops the
recorded process once its pid, start time, executable and command line still
match.

The stop tests use a real sleeping child as the recorded "ComfyUI" and assert
on whether that child is alive afterwards, because a recording double would
also pass against a kill that never reaches the process.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
import requests
from click.testing import CliRunner

import localm.cli.comfy as comfy_cli
import localm.cli.media as media_cli
from localm.media import comfy_client as cc
from localm.media import comfy_launch_record as rec

psutil = pytest.importorskip("psutil")

URL = "http://127.0.0.1:8189"
LAUNCH_URL = "http://127.0.0.1:8196"


def _spawn_sleeper(seconds: int = 120) -> subprocess.Popen:
    """A real child in its own process group, the way _launch_and_wait spawns
    ComfyUI, so _kill_process_tree's group signal never reaches the runner."""
    popen_kw: dict = {}
    if sys.platform == "win32":
        popen_kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen_kw["start_new_session"] = True
    return subprocess.Popen([sys.executable, "-c",
                             f"import time; time.sleep({seconds})"], **popen_kw)


def _alive(proc: subprocess.Popen) -> bool:
    return proc.poll() is None


def _dead(proc: subprocess.Popen, timeout: float = 15.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            return True
        time.sleep(0.05)
    return proc.poll() is not None


@pytest.fixture
def sleeper():
    proc = _spawn_sleeper()
    yield proc
    try:
        for child in psutil.Process(proc.pid).children(recursive=True):
            child.kill()
    except psutil.Error:
        pass
    if proc.poll() is None:
        proc.kill()
    proc.wait(timeout=15)


@pytest.fixture(autouse=True)
def _offline_comfy(monkeypatch):
    """No ComfyUI HTTP traffic, a free port, a fresh handle table, URL as the
    targeted ComfyUI, and an existing data dir for the records."""
    from localm.config import home_dir
    home_dir().mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(cc, "_spawned_procs", {})
    monkeypatch.setattr(cc, "interrupt_comfy", lambda *a, **k: False)
    monkeypatch.setattr(cc, "free_comfy_vram", lambda *a, **k: False)
    monkeypatch.setattr(cc, "_comfy_alive", lambda *a, **k: False)
    monkeypatch.setattr(cc, "comfy_port_holder", lambda *a, **k: None)
    monkeypatch.setattr(cc, "default_api_url", lambda: URL)


def _no_server(monkeypatch):
    calls = []
    monkeypatch.setattr(requests, "request",
                        lambda *a, **k: calls.append(a) or pytest.fail("no request expected"))
    monkeypatch.setattr(comfy_cli, "running_server", lambda **kw: None)
    return calls


def _record(api_url: str = URL) -> dict:
    return json.loads(rec.record_path(api_url).read_text(encoding="utf-8"))


def _rewrite(api_url: str, **changes) -> None:
    data = _record(api_url)
    data.update(changes)
    rec.record_path(api_url).write_text(json.dumps(data), encoding="utf-8")


def _stop():
    result = CliRunner().invoke(comfy_cli.comfy_stop, [])
    result.flat = " ".join(result.output.split())
    return result


# --------------------------------------------------------------------------
#  The record is written by the launch path
# --------------------------------------------------------------------------

def test_a_launch_writes_a_record_naming_the_launched_process(monkeypatch, sleeper, tmp_path):
    """Break: delete the write_record call in _launch_and_wait."""
    from tests._popen_scope import ignore_background_threads

    calls = {"n": 0}

    def _comfy_up(api_url, timeout=3.0):
        calls["n"] += 1
        return calls["n"] > 2

    monkeypatch.setattr(cc, "_comfy_alive", _comfy_up)
    monkeypatch.setattr(time, "sleep", lambda s: None)

    class _Proc:
        pid = sleeper.pid
        returncode = None

        def poll(self):
            return None

    monkeypatch.setattr(subprocess, "Popen", ignore_background_threads(lambda *a, **k: _Proc()))
    monkeypatch.setattr("localm.config.load_config", lambda *a, **k: {"comfy_launch_timeout": 45})
    monkeypatch.setattr(cc, "_confirmed_alive", {})
    ok, msg = cc.ensure_comfy(LAUNCH_URL, launch_cmd="echo hi", workdir=str(tmp_path))
    assert ok is True, msg
    data = _record(LAUNCH_URL)
    assert data["pid"] == sleeper.pid
    assert data["cmdline"] == psutil.Process(sleeper.pid).cmdline()
    assert data["owner_pid"] == os.getpid()


# --------------------------------------------------------------------------
#  (a) a recorded localm-launched ComfyUI is stopped
# --------------------------------------------------------------------------

def test_stop_without_a_server_ends_the_recorded_comfyui(monkeypatch, sleeper):
    """Break: drop `_remember_spawned(api_url, handle)` in stop_recorded_comfy;
    stop_comfy then finds no handle and the sleeper survives."""
    calls = _no_server(monkeypatch)
    assert rec.write_record(URL, sleeper.pid)
    result = _stop()
    assert _dead(sleeper), result.output
    assert result.exit_code == 0, result.output
    assert "Stopped the ComfyUI that localm launched" in result.flat, result.output
    assert not rec.record_path(URL).exists()
    assert calls == []


def test_a_stop_whose_process_survives_is_a_failure(monkeypatch, sleeper):
    """Break: drop the gone_within check in stop_recorded_comfy; the kill
    no-op then reports "Stopped" with exit 0."""
    _no_server(monkeypatch)
    monkeypatch.setattr(cc, "_kill_process_tree", lambda proc: None)
    monkeypatch.setattr(cc, "_ADOPTED_EXIT_WAIT_S", 0.3)
    assert rec.write_record(URL, sleeper.pid)
    result = _stop()
    assert _alive(sleeper)
    assert result.exit_code == 1, result.output
    assert "still running" in result.flat, result.output
    assert rec.record_path(URL).exists()


# --------------------------------------------------------------------------
#  (b) a process that does not match the record is never touched
# --------------------------------------------------------------------------

@pytest.mark.parametrize("change", [
    {"cmdline": ["comfyui", "main.py"]},
    {"create_time": 1_000_000.0},
    {"exe": str(Path("not") / "the" / "same.exe")},
], ids=["cmdline", "start-time", "exe"])
def test_a_pid_whose_identity_differs_is_left_alone(monkeypatch, sleeper, change):
    """Break (per case): remove the matching comparison from adopt()'s
    identity check (`cmdline != record["cmdline"]`, `_same_start(...)`,
    `exe != record["exe"]`); the sleeper is then killed."""
    _no_server(monkeypatch)
    assert rec.write_record(URL, sleeper.pid)
    _rewrite(URL, **change)
    result = _stop()
    time.sleep(0.5)
    assert _alive(sleeper), result.output
    assert result.exit_code == 0, result.output
    assert "left alone" in result.flat, result.output
    assert not rec.record_path(URL).exists()


def test_a_record_that_cannot_be_checked_stops_nothing_and_fails(monkeypatch, sleeper):
    """Break: return ok=True from the UNVERIFIABLE branch of
    stop_recorded_comfy; the exit code then reads 0."""
    _no_server(monkeypatch)
    assert rec.write_record(URL, sleeper.pid)
    monkeypatch.setattr(rec, "adopt", lambda record: (rec.UNVERIFIABLE, None))
    result = _stop()
    assert _alive(sleeper)
    assert result.exit_code == 1, result.output
    assert "nothing was stopped" in result.flat, result.output
    assert rec.record_path(URL).exists()


# --------------------------------------------------------------------------
#  (c) nothing running
# --------------------------------------------------------------------------

def test_nothing_running_says_so_and_exits_zero(monkeypatch):
    """Break: restore `no_server_message(...); raise SystemExit(1)` for the
    no-server case in comfy_stop."""
    calls = _no_server(monkeypatch)
    result = _stop()
    assert result.exit_code == 0, result.output
    assert "ComfyUI is not running." in result.flat, result.output
    assert "No running localm server" not in result.flat, result.output
    assert calls == []


def test_a_record_of_an_exited_process_is_dropped(monkeypatch):
    """Break: remove `rec.drop(record)` from the GONE branch."""
    _no_server(monkeypatch)
    rec.record_path(URL).write_text(json.dumps({
        "api_url": URL, "pid": 999999, "create_time": 1.0, "exe": "x",
        "cmdline": ["x"], "owner_pid": 1, "owner_create_time": 1.0}), encoding="utf-8")
    monkeypatch.setattr(rec, "adopt", lambda record: (rec.GONE, None))
    result = _stop()
    assert result.exit_code == 0, result.output
    assert "ComfyUI is not running." in result.flat, result.output
    assert not rec.record_path(URL).exists()


# --------------------------------------------------------------------------
#  With a server: only a ComfyUI whose launching process has exited
# --------------------------------------------------------------------------

class _Resp:
    status_code = 200
    ok = True
    text = "{}"

    def json(self):
        return {"ok": True, "message": "ComfyUI is not running."}


def _with_server(monkeypatch):
    paths = []

    def _request(method, url, **kw):
        paths.append(url.split("127.0.0.1:9999", 1)[-1])
        return _Resp()

    monkeypatch.setattr(requests, "request", _request)
    monkeypatch.setattr(comfy_cli, "running_server",
                        lambda **kw: ("http://127.0.0.1:9999", {}))
    return paths


def test_with_a_server_an_orphaned_comfyui_is_stopped(monkeypatch, sleeper):
    """Break: remove the `_stop_orphaned()` call from comfy_stop."""
    paths = _with_server(monkeypatch)
    assert rec.write_record(URL, sleeper.pid)
    _rewrite(URL, owner_create_time=1_000_000.0)
    result = _stop()
    assert _dead(sleeper), result.output
    assert result.exit_code == 0, result.output
    assert paths == ["/v1/comfy/stop"]


def test_with_a_server_a_comfyui_whose_launcher_lives_is_left_to_it(monkeypatch, sleeper):
    """Break: call stop_recorded_comfy() without skip_live_owner=True in
    _stop_orphaned; the sleeper (owned by this live test process) is killed."""
    paths = _with_server(monkeypatch)
    assert rec.write_record(URL, sleeper.pid)
    result = _stop()
    time.sleep(0.5)
    assert _alive(sleeper), result.output
    assert result.exit_code == 0, result.output
    assert paths == ["/v1/comfy/stop"]
    assert rec.record_path(URL).exists()


# --------------------------------------------------------------------------
#  (d) the `localm image` hint
# --------------------------------------------------------------------------

class _Backend:
    def __init__(self, needs_comfy=True):
        self.needs_comfy = needs_comfy

    def supports(self, s, capability):
        return capability == "NEEDS_COMFY_SERVER" and self.needs_comfy

    def free_vram(self, s):
        pass


def _generate(monkeypatch, tmp_path, *, launched: bool, needs_comfy=True):
    from localm.plugins import media_config
    out = tmp_path / "out.png"
    monkeypatch.setattr(media_config, "generate_media",
                        lambda *a, **k: (True, f"Saved {out.name}", out))
    monkeypatch.setattr(media_cli, "_is_interactive", lambda: False)
    monkeypatch.setattr(cc, "spawned_pid", lambda *a, **k: 4242 if launched else None)
    runner_out = []
    monkeypatch.setattr(media_cli.console, "print",
                        lambda *a, **k: runner_out.append(" ".join(str(x) for x in a)))
    media_cli._via_backend(_Backend(needs_comfy), {"api_url": URL}, "a fox", out, "image")
    return "\n".join(runner_out)


def test_the_hint_appears_when_this_run_launched_comfyui(monkeypatch, tmp_path):
    """Break: delete the `_say_comfy_left_running(...)` call in _via_backend."""
    out = _generate(monkeypatch, tmp_path, launched=True)
    assert "ComfyUI keeps running for the next image" in out
    assert "localm comfy stop" in out


def test_no_hint_when_comfyui_was_already_running(monkeypatch, tmp_path):
    """Break: remove the `spawned_pid(api_url) is None` early return in
    _say_comfy_left_running."""
    out = _generate(monkeypatch, tmp_path, launched=False)
    assert "keeps running" not in out


def test_no_hint_for_a_backend_without_comfyui(monkeypatch, tmp_path):
    """Break: call `_say_comfy_left_running(s["api_url"], what)` outside the
    NEEDS_COMFY_SERVER branch of _via_backend."""
    out = _generate(monkeypatch, tmp_path, launched=True, needs_comfy=False)
    assert "keeps running" not in out
