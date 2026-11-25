# SPDX-License-Identifier: AGPL-3.0-or-later
"""`localm comfy stop` without a localm server, and the `localm image` hint.

`localm image` launches ComfyUI in its own process and leaves it running when
that process exits, so no live process holds its handle. `_launch_and_wait`
writes a launch record (`comfy_launch_record`), and `comfy stop` stops the
recorded process once its pid, start time, executable and command line still
match.

The stop tests use a real sleeping child as the recorded "ComfyUI" and assert
on whether that child is alive afterwards, because a recording double would
also pass against a kill that never reaches the process. Those need psutil;
the tests that do not are left to run without it.
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

try:
    import psutil
except ImportError:
    psutil = None

needs_psutil = pytest.mark.skipif(psutil is None, reason="psutil is not installed")

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


def _reap(proc: subprocess.Popen) -> None:
    if psutil is not None:
        try:
            for child in psutil.Process(proc.pid).children(recursive=True):
                child.kill()
        except psutil.Error:
            pass
    if proc.poll() is None:
        proc.kill()
    proc.wait(timeout=15)


@pytest.fixture
def sleeper():
    proc = _spawn_sleeper()
    yield proc
    _reap(proc)


@pytest.fixture
def second_sleeper():
    proc = _spawn_sleeper()
    yield proc
    _reap(proc)


@pytest.fixture(autouse=True)
def _offline_comfy(monkeypatch):
    """No ComfyUI HTTP traffic, a free port, a fresh handle table, URL as the
    targeted ComfyUI, and an existing data dir for the records."""
    from localm.config import home_dir
    home_dir().mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(cc, "_spawned_procs", {})
    monkeypatch.setattr(cc, "_record_outcomes", {})
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


def _hand_written_record(api_url: str = URL, pid: int = 999999) -> None:
    rec.record_path(api_url).write_text(json.dumps({
        "api_url": api_url, "pid": pid, "create_time": 1.0, "exe": "x",
        "cmdline": ["x"], "owner_pid": 1, "owner_create_time": 1.0}), encoding="utf-8")


def _stop():
    result = CliRunner().invoke(comfy_cli.comfy_stop, [])
    result.flat = " ".join(result.output.split())
    return result


def _fake_launch(monkeypatch, pid: int, *, exits_at_once: bool = False):
    """Make ensure_comfy's Popen return a handle on *pid*, with ComfyUI coming
    up on the third liveness probe."""
    from tests._popen_scope import ignore_background_threads

    calls = {"n": 0}

    def _comfy_up(api_url, timeout=3.0):
        calls["n"] += 1
        return calls["n"] > 2

    monkeypatch.setattr(cc, "_comfy_alive", _comfy_up)
    monkeypatch.setattr(time, "sleep", lambda s: None)

    class _Proc:
        returncode = 1 if exits_at_once else None

        def __init__(self):
            self.pid = pid

        def poll(self):
            return 1 if exits_at_once else None

    monkeypatch.setattr(subprocess, "Popen", ignore_background_threads(lambda *a, **k: _Proc()))
    monkeypatch.setattr("localm.config.load_config", lambda *a, **k: {"comfy_launch_timeout": 45})
    monkeypatch.setattr(cc, "_confirmed_alive", {})


# --------------------------------------------------------------------------
#  The record: written by the launch path, removed when localm ends it
# --------------------------------------------------------------------------

@needs_psutil
def test_a_launch_writes_a_record_naming_the_launched_process(monkeypatch, sleeper, tmp_path):
    """Break: delete the write_record call in _launch_and_wait."""
    _fake_launch(monkeypatch, sleeper.pid)
    ok, msg = cc.ensure_comfy(LAUNCH_URL, launch_cmd="echo hi", workdir=str(tmp_path))
    assert ok is True, msg
    data = _record(LAUNCH_URL)
    assert data["pid"] == sleeper.pid
    assert data["cmdline"] == psutil.Process(sleeper.pid).cmdline()
    assert data["owner_pid"] == os.getpid()
    assert cc.launch_record_outcome(LAUNCH_URL) == (True, "")


@needs_psutil
def test_a_launch_that_exits_at_once_leaves_no_record(monkeypatch, sleeper, tmp_path):
    """Break: remove `comfy_launch_record.drop_pid(api_url, _proc_pid)` from
    the immediate-exit branch of _launch_and_wait."""
    _fake_launch(monkeypatch, sleeper.pid, exits_at_once=True)
    ok, msg = cc.ensure_comfy(LAUNCH_URL, launch_cmd="echo hi", workdir=str(tmp_path))
    assert ok is False, msg
    assert "exited immediately" in msg
    assert not rec.record_path(LAUNCH_URL).exists()


@needs_psutil
def test_stop_comfy_of_its_own_comfyui_removes_the_record(sleeper):
    """The path /v1/comfy/stop and stop_all_spawned_comfy take. Break: remove
    the `comfy_launch_record.drop_pid(...)` after `_kill_process_tree(proc)`
    in stop_comfy."""
    assert rec.write_record(URL, sleeper.pid) == (True, "")
    cc._remember_spawned(URL, sleeper)
    ok, message = cc.stop_comfy(URL)
    assert ok, message
    assert _dead(sleeper)
    assert not rec.record_path(URL).exists()


@needs_psutil
def test_a_second_launch_keeps_the_record_of_a_still_running_first(sleeper, second_sleeper):
    """Break: remove the `adopt(existing)[0] == MATCH` check in write_record;
    the second write then replaces the first record."""
    assert rec.write_record(URL, sleeper.pid) == (True, "")
    written, reason = rec.write_record(URL, second_sleeper.pid)
    assert written is False
    assert str(sleeper.pid) in reason
    assert _record()["pid"] == sleeper.pid


@needs_psutil
def test_a_record_whose_process_no_longer_matches_is_replaced(sleeper, second_sleeper):
    """Break: keep any existing record in write_record, without asking adopt()
    whether its process still runs."""
    assert rec.write_record(URL, sleeper.pid) == (True, "")
    _rewrite(URL, create_time=1_000_000.0)
    assert rec.write_record(URL, second_sleeper.pid) == (True, "")
    assert _record()["pid"] == second_sleeper.pid


def test_without_psutil_nothing_is_recorded_and_the_reason_says_so(monkeypatch):
    """Break: remove the `except ImportError` branch of write_record; the
    reason then reads as the raw import error."""
    monkeypatch.setitem(sys.modules, "psutil", None)
    written, reason = rec.write_record(URL, os.getpid())
    assert written is False
    assert reason == "psutil is not installed"
    assert not rec.record_path(URL).exists()


# --------------------------------------------------------------------------
#  adopt(): the start time is compared first
# --------------------------------------------------------------------------

class _FakePsProcess:
    def __init__(self, create_time, exe=None, cmdline=None):
        self._create_time, self._exe, self._cmdline = create_time, exe, cmdline

    def create_time(self):
        return self._create_time

    def exe(self):
        if self._exe is None:
            raise psutil.AccessDenied(4242)
        return self._exe

    def cmdline(self):
        if self._cmdline is None:
            raise psutil.AccessDenied(4242)
        return self._cmdline


_RECORD = {"api_url": URL, "pid": 4242, "create_time": 100.0, "exe": "python",
           "cmdline": ["python", "main.py"], "owner_pid": 1, "owner_create_time": 1.0}


@needs_psutil
def test_a_different_start_time_is_a_mismatch_even_when_exe_is_unreadable(monkeypatch):
    """Break: read exe() and cmdline() before comparing the start time (any
    AccessDenied then answers UNVERIFIABLE)."""
    monkeypatch.setattr(psutil, "Process", lambda pid: _FakePsProcess(5000.0))
    assert rec.adopt(dict(_RECORD)) == (rec.MISMATCH, None)


@needs_psutil
def test_a_matching_start_time_with_unreadable_cmdline_is_unverifiable(monkeypatch):
    """Break: return MISMATCH from the second `except psutil.Error` in adopt()."""
    monkeypatch.setattr(psutil, "Process",
                        lambda pid: _FakePsProcess(100.0, exe="python", cmdline=None))
    assert rec.adopt(dict(_RECORD)) == (rec.UNVERIFIABLE, None)


# --------------------------------------------------------------------------
#  (a) a recorded localm-launched ComfyUI is stopped
# --------------------------------------------------------------------------

@needs_psutil
def test_stop_without_a_server_ends_the_recorded_comfyui(monkeypatch, sleeper):
    """Break: drop `_remember_spawned(api_url, handle)` in stop_recorded_comfy;
    stop_comfy then finds no handle and the sleeper survives."""
    calls = _no_server(monkeypatch)
    assert rec.write_record(URL, sleeper.pid) == (True, "")
    result = _stop()
    assert _dead(sleeper), result.output
    assert result.exit_code == 0, result.output
    assert "Stopped the ComfyUI that localm launched" in result.flat, result.output
    assert not rec.record_path(URL).exists()
    assert calls == []


@needs_psutil
def test_a_stop_whose_process_survives_is_a_failure_and_keeps_the_record(monkeypatch, sleeper):
    """Break: drop the gone_within check in stop_recorded_comfy (exit 0), or
    the `rec.restore(record)` (the record is gone)."""
    _no_server(monkeypatch)
    monkeypatch.setattr(cc, "_kill_process_tree", lambda proc: None)
    monkeypatch.setattr(cc, "_ADOPTED_EXIT_WAIT_S", 0.3)
    assert rec.write_record(URL, sleeper.pid) == (True, "")
    result = _stop()
    assert _alive(sleeper)
    assert result.exit_code == 1, result.output
    assert "still running" in result.flat, result.output
    assert _record()["pid"] == sleeper.pid


# --------------------------------------------------------------------------
#  (b) a process that does not match the record is never touched
# --------------------------------------------------------------------------

@needs_psutil
@pytest.mark.parametrize("change", [
    {"cmdline": ["comfyui", "main.py"]},
    {"create_time": 1_000_000.0},
    {"exe": str(Path("not") / "the" / "same.exe")},
], ids=["cmdline", "start-time", "exe"])
def test_a_pid_whose_identity_differs_is_left_alone(monkeypatch, sleeper, change):
    """Break (per case): remove the matching comparison from adopt()
    (`cmdline != record["cmdline"]`, `_same_start(...)`,
    `exe != record["exe"]`); the sleeper is then killed. The target's own
    abort-and-free still runs ("ComfyUI is not running."): break that by
    returning covered=True for a MISMATCH in stop_recorded_comfy."""
    _no_server(monkeypatch)
    assert rec.write_record(URL, sleeper.pid) == (True, "")
    _rewrite(URL, **change)
    result = _stop()
    time.sleep(0.5)
    assert _alive(sleeper), result.output
    assert result.exit_code == 0, result.output
    assert "left alone" in result.flat, result.output
    assert "ComfyUI is not running." in result.flat, result.output
    assert not rec.record_path(URL).exists()


def test_a_record_that_cannot_be_checked_stops_nothing_and_fails(monkeypatch):
    """Break: return ok=True from the UNVERIFIABLE branch of
    stop_recorded_comfy; the exit code then reads 0."""
    _no_server(monkeypatch)
    _hand_written_record()
    monkeypatch.setattr(rec, "adopt", lambda record: (rec.UNVERIFIABLE, None))
    result = _stop()
    assert result.exit_code == 1, result.output
    assert "nothing was stopped" in result.flat, result.output
    assert rec.record_path(URL).exists()


# --------------------------------------------------------------------------
#  (c) nothing running, and no record of a running one
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
    _hand_written_record()
    monkeypatch.setattr(rec, "adopt", lambda record: (rec.GONE, None))
    result = _stop()
    assert result.exit_code == 0, result.output
    assert "ComfyUI is not running." in result.flat, result.output
    assert not rec.record_path(URL).exists()


def test_a_running_comfyui_with_no_record_is_not_called_one_localm_did_not_launch(monkeypatch):
    """Break: drop `ownership_unknown=True` from the stop_comfy call in
    _stop_without_server."""
    _no_server(monkeypatch)
    monkeypatch.setattr(cc, "_comfy_alive", lambda *a, **k: True)
    monkeypatch.setattr(cc, "interrupt_comfy", lambda *a, **k: True)
    result = _stop()
    assert result.exit_code == 0, result.output
    assert "no record of launching this ComfyUI" in result.flat, result.output
    assert "did not launch" not in result.flat, result.output


def test_a_held_port_with_no_record_is_not_called_one_localm_did_not_start(monkeypatch):
    """Break: drop the `ownership_unknown=ownership_unknown` argument from the
    comfy_port_held_message call in stop_comfy."""
    _no_server(monkeypatch)
    monkeypatch.setattr(cc, "comfy_port_holder",
                        lambda *a, **k: {"port": 8189, "pid": 77, "name": "python",
                                         "localm_launched": False})
    result = _stop()
    assert result.exit_code == 1, result.output
    assert "no record of starting it" in result.flat, result.output
    assert "did not start it" not in result.flat, result.output


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


@needs_psutil
def test_with_a_server_an_orphaned_comfyui_is_stopped(monkeypatch, sleeper):
    """Break: remove the `_stop_orphaned()` call from comfy_stop."""
    paths = _with_server(monkeypatch)
    assert rec.write_record(URL, sleeper.pid) == (True, "")
    _rewrite(URL, owner_create_time=1_000_000.0)
    result = _stop()
    assert _dead(sleeper), result.output
    assert result.exit_code == 0, result.output
    assert paths == ["/v1/comfy/stop"]


@needs_psutil
def test_with_a_server_a_comfyui_whose_launcher_lives_is_left_to_it(monkeypatch, sleeper):
    """Break: call stop_recorded_comfy() without skip_live_owner=True in
    _stop_orphaned; the sleeper (owned by this live test process) is killed."""
    paths = _with_server(monkeypatch)
    assert rec.write_record(URL, sleeper.pid) == (True, "")
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


def _generate(monkeypatch, tmp_path, *, outcome, needs_comfy=True):
    """Run _via_backend with *outcome* as this process's launch record result
    (None: this process did not launch ComfyUI); return what it printed."""
    from localm.plugins import media_config
    out = tmp_path / "out.png"
    monkeypatch.setattr(media_config, "generate_media",
                        lambda *a, **k: (True, f"Saved {out.name}", out))
    monkeypatch.setattr(media_cli, "_is_interactive", lambda: False)
    monkeypatch.setattr(cc, "launch_record_outcome", lambda *a, **k: outcome)
    monkeypatch.setattr(cc, "spawned_pid", lambda *a, **k: None if outcome is None else 4242)
    printed = []
    monkeypatch.setattr(media_cli.console, "print",
                        lambda *a, **k: printed.append(" ".join(str(x) for x in a)))
    media_cli._via_backend(_Backend(needs_comfy), {"api_url": URL}, "a fox", out, "image",
                           explicit=True)
    return " ".join(" ".join(printed).split())


def test_the_hint_appears_when_this_run_launched_and_recorded_comfyui(monkeypatch, tmp_path):
    """Break: delete the `_say_comfy_left_running(...)` call in _via_backend."""
    out = _generate(monkeypatch, tmp_path, outcome=(True, ""))
    assert "ComfyUI keeps running for the next image" in out
    assert "localm comfy stop[dim] stops it" in out
    assert "could not record" not in out


def test_an_unrecorded_launch_says_so_instead_of_promising_comfy_stop(monkeypatch, tmp_path):
    """Break: print the `localm comfy stop` hint without checking `written`
    in _say_comfy_left_running."""
    out = _generate(monkeypatch, tmp_path, outcome=(False, "psutil is not installed"))
    assert "could not record it" in out
    assert "psutil is not installed" in out
    assert "End process 4242" in out
    assert "stops it and frees its VRAM" not in out


def test_no_hint_when_comfyui_was_already_running(monkeypatch, tmp_path):
    """Break: remove the `if outcome is None: return` early return in
    _say_comfy_left_running."""
    out = _generate(monkeypatch, tmp_path, outcome=None)
    assert "keeps running" not in out


def test_no_hint_for_a_backend_without_comfyui(monkeypatch, tmp_path):
    """Break: call `_say_comfy_left_running(s["api_url"], what)` outside the
    NEEDS_COMFY_SERVER branch of _via_backend."""
    out = _generate(monkeypatch, tmp_path, outcome=(True, ""), needs_comfy=False)
    assert "keeps running" not in out
