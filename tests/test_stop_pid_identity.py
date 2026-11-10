# SPDX-License-Identifier: AGPL-3.0-or-later
"""`localm stop` and POST /api/instances/{id}/stop end a process only when its
start identity matches the one its registry entry recorded.

A registry entry whose server died without removing it can name a pid the OS
has since given to an unrelated program. Each test here starts a harmless
child process and writes an entry with the child's pid, then checks what each
stop path does to it.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys

import pytest
from click.testing import CliRunner
from fastapi import FastAPI
from fastapi.testclient import TestClient

from localm import instances
from localm.cli import main


def _spawn_child():
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"],
                            creationflags=flags)


@pytest.fixture
def child():
    proc = _spawn_child()
    try:
        yield proc
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=10)


def _dead_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _other_identity(identity: dict) -> dict:
    """A start identity of the same shape as *identity* naming another process."""
    if "ticks" in identity:
        return {"boot": identity["boot"], "ticks": identity["ticks"] - 1}
    return {"created": identity["created"] - 100.0}


def _write_entry(home, *, iid, pid, identity="match", space="own"):
    """Write a registry entry for *pid*. *identity*: ``"match"`` records the
    live process's own start identity, ``"other"`` one naming another process,
    ``None`` leaves both identity fields out (an entry from an earlier localm).
    *space*: ``"own"`` records this machine's pid space, any other string is
    written as is."""
    entry = {"instance_id": iid, "pid": pid, "port": _dead_port(),
             "host": "127.0.0.1", "scheme": "http", "root_dir": f"/proj/{iid}",
             "mode": "api", "version": "test",
             "started": "2026-11-10T00:00:00+00:00",
             "token": instances.new_token()}
    if identity is not None:
        current = instances.process_start_identity(pid)
        assert current is not None, "test needs a readable start identity"
        entry["start_identity"] = (current if identity == "match"
                                   else _other_identity(current))
        entry["pid_space"] = (instances.pid_space_id() if space == "own"
                              else space)
    path = instances.registry_path(home, iid)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entry, indent=2), encoding="utf-8")
    return path


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / ".localm"
    monkeypatch.setenv("LOCALM_HOME", str(h))
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    return h


# --------------------------------------------------------------------------- #
#  The registry entry and entry_process_state                                 #
# --------------------------------------------------------------------------- #

class TestEntryRecordsIdentity:
    def test_register_instance_records_this_process_start_identity(self, home):
        path = instances.register_instance(
            home, instance_id="ownentry0000001", port=1, host="127.0.0.1",
            root_dir="/proj/own", mode="api", token="t")
        entry = json.loads(path.read_text(encoding="utf-8"))
        assert entry["pid"] == os.getpid()
        assert entry["pid_space"] == instances.pid_space_id()
        assert instances.start_identity_matches(
            entry["start_identity"], instances.process_start_identity(os.getpid()))
        assert instances.entry_process_state(entry) == ("same", "")


class TestEntryProcessState:
    def test_matching_identity_is_same(self, home, child):
        p = _write_entry(home, iid="same000000000001", pid=child.pid)
        assert instances.entry_process_state(instances.read_entry(p))[0] == "same"

    def test_differing_identity_is_stale(self, home, child):
        p = _write_entry(home, iid="stale00000000001", pid=child.pid,
                         identity="other")
        state, reason = instances.entry_process_state(instances.read_entry(p))
        assert state == "stale"
        assert f"pid {child.pid} now belongs to another program" in reason

    def test_entry_without_identity_is_unknown(self, home, child):
        p = _write_entry(home, iid="old0000000000001", pid=child.pid, identity=None)
        state, reason = instances.entry_process_state(instances.read_entry(p))
        assert state == "unknown"
        assert "earlier localm" in reason

    def test_entry_from_another_pid_space_is_unknown(self, home, child):
        p = _write_entry(home, iid="space00000000001", pid=child.pid,
                         identity="other", space="0123456789abcdef")
        state, reason = instances.entry_process_state(instances.read_entry(p))
        assert state == "unknown"
        assert "another machine" in reason

    def test_dead_pid_is_gone(self, home):
        proc = _spawn_child()
        p = _write_entry(home, iid="gone000000000001", pid=proc.pid)
        proc.kill()
        proc.wait(timeout=10)
        assert instances.entry_process_state(instances.read_entry(p))[0] == "gone"


class TestReapStale:
    def test_reaps_stale_keeps_same_and_unknown(self, home, child):
        stale = _write_entry(home, iid="stale00000000002", pid=child.pid,
                             identity="other")
        same = _write_entry(home, iid="same000000000002", pid=child.pid)
        old = _write_entry(home, iid="old0000000000002", pid=child.pid,
                           identity=None)
        removed = instances.reap_stale(home)
        assert removed == ["stale00000000002"]
        assert not stale.exists()
        assert same.exists() and old.exists()
        assert child.poll() is None


class TestKillPidIdentity:
    def test_a_mismatching_identity_does_not_end_the_process(self, child):
        current = instances.process_start_identity(child.pid)
        assert instances.kill_pid(child.pid, timeout=5,
                                  start_identity=_other_identity(current)) is False
        assert child.poll() is None

    def test_a_matching_identity_ends_the_process(self, child):
        current = instances.process_start_identity(child.pid)
        assert instances.kill_pid(child.pid, timeout=10,
                                  start_identity=current) is True
        child.wait(timeout=10)


# --------------------------------------------------------------------------- #
#  localm stop                                                                #
# --------------------------------------------------------------------------- #

class TestCliStop:
    def test_a_stale_entry_is_removed_and_nothing_is_killed(self, home, child):
        p = _write_entry(home, iid="clistale00000001", pid=child.pid,
                         identity="other")
        res = CliRunner().invoke(main, ["stop", "clistale"])
        assert child.poll() is None, "the unrelated process must still run"
        assert res.exit_code == 0, res.output
        assert "this entry was stale" in res.output
        assert "stopped nothing" in res.output
        assert not p.exists()

    def test_an_entry_without_identity_is_not_killed(self, home, child):
        p = _write_entry(home, iid="cliold0000000001", pid=child.pid, identity=None)
        res = CliRunner().invoke(main, ["stop", "cliold"])
        assert child.poll() is None, "an unproven pid must not be ended"
        assert res.exit_code == 1, res.output
        assert f"did not end pid {child.pid}" in res.output
        assert "earlier localm" in res.output
        assert p.exists()

    def test_a_matching_entry_is_still_stopped(self, home, child):
        p = _write_entry(home, iid="clisame000000001", pid=child.pid)
        res = CliRunner().invoke(main, ["stop", "clisame", "--timeout", "5"])
        assert res.exit_code == 0, res.output
        assert "stopped" in res.output
        child.wait(timeout=10)
        assert not p.exists()

    def test_all_reports_a_stale_entry_and_stops_the_real_one(self, home, child):
        real = _spawn_child()
        try:
            _write_entry(home, iid="allstale00000001", pid=child.pid,
                         identity="other")
            _write_entry(home, iid="allsame000000001", pid=real.pid)
            res = CliRunner().invoke(main, ["stop", "--all", "--timeout", "5"])
            assert child.poll() is None
            assert res.exit_code == 0, res.output
            assert "this entry was stale" in res.output
            real.wait(timeout=10)
        finally:
            if real.poll() is None:
                real.kill()
            real.wait(timeout=10)


# --------------------------------------------------------------------------- #
#  POST /api/instances/{id}/stop                                              #
# --------------------------------------------------------------------------- #

@pytest.fixture
def client(home):
    from localm.plugins.engine import attach_engine
    from localm.plugins.gui.web import attach_gui
    app = FastAPI()
    attach_engine(app)
    attach_gui(app, self_url="http://127.0.0.1:9/v1",
               switch_model=lambda name: None, active_model=lambda: "")
    with TestClient(app) as c:
        yield c


class TestGuiStop:
    def test_a_stale_entry_is_removed_and_nothing_is_killed(self, home, client, child):
        p = _write_entry(home, iid="guistale00000001", pid=child.pid,
                         identity="other")
        r = client.post("/api/instances/guistale00000001/stop")
        assert child.poll() is None, "the unrelated process must still run"
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["status"] == "stale"
        assert body["pid"] == child.pid
        assert "This entry was stale" in body["message"]
        assert not p.exists()

    def test_an_entry_without_identity_is_not_killed(self, home, client, child):
        p = _write_entry(home, iid="guiold0000000001", pid=child.pid, identity=None)
        r = client.post("/api/instances/guiold0000000001/stop")
        assert child.poll() is None, "an unproven pid must not be ended"
        assert r.status_code == 409, r.text
        assert f"Did not end pid {child.pid}" in r.json()["detail"]
        assert p.exists()

    def test_a_matching_entry_is_still_stopped(self, home, client, child):
        p = _write_entry(home, iid="guisame000000001", pid=child.pid)
        r = client.post("/api/instances/guisame000000001/stop")
        assert r.status_code == 200, r.text
        assert r.json()["status"] == "stopped"
        child.wait(timeout=10)
        assert not p.exists()


# --------------------------------------------------------------------------- #
#  Further paths                                                              #
# --------------------------------------------------------------------------- #

def _fake_resp(status):
    from types import SimpleNamespace
    return SimpleNamespace(status_code=status, ok=status == 200,
                           json=lambda: {}, text="")


class TestUnreadableIdentity:
    def test_entry_with_pid_space_but_no_identity_names_the_cause(self, home, child):
        p = _write_entry(home, iid="noid000000000001", pid=child.pid)
        entry = instances.read_entry(p)
        entry["start_identity"] = None
        state, reason = instances.entry_process_state(entry)
        assert state == "unknown"
        assert "could not read its own start time" in reason
        assert "earlier localm" not in reason


class TestKillCarriesIdentity:
    """Both stop paths hand the recorded identity to kill_pid."""

    def _spy(self, monkeypatch):
        calls = []
        real = instances.kill_pid

        def spy(pid, *, timeout=10.0, start_identity=None):
            calls.append(start_identity)
            return real(pid, timeout=timeout, start_identity=start_identity)

        monkeypatch.setattr(instances, "kill_pid", spy)
        return calls

    def test_cli(self, home, child, monkeypatch):
        calls = self._spy(monkeypatch)
        p = _write_entry(home, iid="clikill000000001", pid=child.pid)
        recorded = instances.read_entry(p)["start_identity"]
        res = CliRunner().invoke(main, ["stop", "clikill", "--timeout", "5"])
        assert res.exit_code == 0, res.output
        assert calls == [recorded]

    def test_gui(self, home, client, child, monkeypatch):
        calls = self._spy(monkeypatch)
        p = _write_entry(home, iid="guikill000000001", pid=child.pid)
        recorded = instances.read_entry(p)["start_identity"]
        r = client.post("/api/instances/guikill000000001/stop")
        assert r.status_code == 200, r.text
        assert calls == [recorded]


class TestFindAttachableReapsStale:
    def test_a_stale_same_dir_entry_is_removed_and_nothing_attached(self, home, child):
        p = _write_entry(home, iid="attstale00000001", pid=child.pid,
                         identity="other")
        root = instances.read_entry(p)["root_dir"]
        assert instances.find_attachable(home, root, probe=lambda e: False) is None
        assert not p.exists()
        assert child.poll() is None

    def test_an_unknown_same_dir_entry_is_kept(self, home, child):
        p = _write_entry(home, iid="attold0000000001", pid=child.pid, identity=None)
        root = instances.read_entry(p)["root_dir"]
        assert instances.find_attachable(home, root, probe=lambda e: False) is None
        assert p.exists()


class TestCliNoArgumentStop:
    def test_a_stale_entry_for_this_directory_is_reported(self, home, child,
                                                          monkeypatch):
        p = _write_entry(home, iid="dirstale00000001", pid=child.pid,
                         identity="other")
        root = instances.read_entry(p)["root_dir"]
        monkeypatch.setattr(instances, "resolve_root_dir", lambda *a, **k: root)
        res = CliRunner().invoke(main, ["stop"])
        assert child.poll() is None
        assert "this entry was stale" in res.output
        assert not p.exists()


class TestCliAfterGracefulAnswer:
    def test_a_declined_shutdown_of_an_unproven_entry_kills_nothing(
            self, home, child):
        from unittest.mock import patch
        p = _write_entry(home, iid="decl000000000001", pid=child.pid, identity=None)
        with patch("requests.post", return_value=_fake_resp(403)), \
             patch("localm.tls.requests_verify", return_value=True):
            res = CliRunner().invoke(main, ["stop", "decl"])
        assert child.poll() is None
        assert res.exit_code == 1, res.output
        assert f"did not end pid {child.pid}" in res.output
        assert "LOCALM_API_KEY" not in res.output
        assert p.exists()

    def test_an_accepted_shutdown_whose_pid_lives_on_is_ended_by_identity(
            self, home, child):
        from unittest.mock import patch
        _write_entry(home, iid="acpt000000000001", pid=child.pid)
        with patch("requests.post", return_value=_fake_resp(200)), \
             patch("localm.tls.requests_verify", return_value=True):
            res = CliRunner().invoke(main, ["stop", "acpt", "--timeout", "1"])
        assert res.exit_code == 0, res.output
        child.wait(timeout=10)

    def test_an_accepted_shutdown_whose_pid_now_names_another_process(
            self, home, child):
        from unittest.mock import patch
        p = _write_entry(home, iid="acst000000000001", pid=child.pid)
        real_state = instances.entry_process_state
        seen = []

        def flip(entry):
            seen.append(1)
            if len(seen) == 1:
                return real_state(entry)
            return "stale", f"pid {child.pid} now belongs to another program."

        with patch("requests.post", return_value=_fake_resp(200)), \
             patch("localm.tls.requests_verify", return_value=True), \
             patch.object(instances, "entry_process_state", side_effect=flip):
            res = CliRunner().invoke(main, ["stop", "acst", "--timeout", "1"])
        assert child.poll() is None
        assert "this entry was stale" in res.output
        assert not p.exists()
