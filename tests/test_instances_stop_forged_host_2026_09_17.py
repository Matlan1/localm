# SPDX-License-Identifier: AGPL-3.0-or-later
"""The two instance-stop paths (POST /api/instances/{id}/stop and `localm
stop`) only ever dial an address THIS machine holds before attempting a
graceful shutdown.

Both read a per-install registry entry (`<LOCALM_HOME>/run/<id>.json`), which
is untrusted input: a forged `host` field would otherwise make the owner's own
stop request POST /server/shutdown, carrying that entry's own token, to an
outside address. `bindhost.is_own_address` is the same gate PR #1881 put in
front of `instances.fetch_whoami`; these tests pin it in front of the two
remaining sites that build a URL from a registry entry directly, without a
/whoami probe ahead of it.

A forged host must never be dialed, and the pid-based `instances.kill_pid`
fallback must still run so the process is stopped anyway. A loopback entry
must be unaffected: the graceful HTTP attempt still happens exactly as
before.
"""

from __future__ import annotations

import json
from unittest.mock import Mock, patch

import pytest
import requests
from click.testing import CliRunner
from fastapi import FastAPI
from fastapi.testclient import TestClient

from localm import instances
from localm.cli import main

FORGED_HOST = "attacker.example"
UNREACHABLE_PID = 999999999


# --------------------------------------------------------------------------- #
#  POST /api/instances/{id}/stop                                              #
# --------------------------------------------------------------------------- #

@pytest.fixture
def instances_app(tmp_path, monkeypatch):
    """Full stack (kernel + GUI) on a throwaway home, same shape as
    test_instances_gui_route_2026_08_20.py."""
    home = tmp_path / ".localm"
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    from localm.plugins.engine import attach_engine
    from localm.plugins.gui.web import attach_gui
    app = FastAPI()
    attach_engine(app)
    attach_gui(app, self_url="http://127.0.0.1:9/v1",
               switch_model=lambda name: None,
               active_model=lambda: "")
    return app, home


def _write_entry(home, *, instance_id, host, pid=UNREACHABLE_PID,
                 port=59970, scheme="http", token=None):
    entry = dict(instance_id=instance_id, pid=pid, port=port, host=host,
                scheme=scheme, root_dir="/proj/one", mode="api",
                version="test", token=token or instances.new_token(),
                started="2026-09-17T00:00:00+00:00")
    path = instances.registry_path(home, instance_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entry), encoding="utf-8")
    return path


class TestGuiStopRefusesAForgedHost:
    def test_forged_host_is_never_dialed_and_falls_back_to_kill(
            self, instances_app, monkeypatch, caplog):
        app, home = instances_app
        monkeypatch.setattr(instances, "reap_stale", lambda *a, **k: [])
        iid = "forgedguistop0001"
        _write_entry(home, instance_id=iid, host=FORGED_HOST)
        kill_calls = []
        monkeypatch.setattr(
            instances, "kill_pid",
            lambda pid, timeout=None: kill_calls.append(pid) or True)
        request_spy = Mock()
        with patch("requests.request", request_spy), \
             caplog.at_level("WARNING", logger="localm"):
            with TestClient(app) as c:
                r = c.post(f"/api/instances/{iid}/stop")   # open mode
        request_spy.assert_not_called()
        assert kill_calls == [UNREACHABLE_PID]
        assert "not an address this machine holds" in caplog.text
        assert r.status_code == 200, r.text
        assert r.json()["graceful_denied"] is False

    def test_a_loopback_host_still_attempts_the_graceful_shutdown(
            self, instances_app, monkeypatch):
        """The positive control: an ordinary loopback entry is unaffected by
        the guard - the graceful POST is still attempted, same as before."""
        app, home = instances_app
        monkeypatch.setattr(instances, "reap_stale", lambda *a, **k: [])
        iid = "loopbackguistop01"
        _write_entry(home, instance_id=iid, host="127.0.0.1")
        request_spy = Mock(side_effect=requests.ConnectionError("refused"))
        monkeypatch.setattr(instances, "kill_pid", lambda *a, **k: True)
        with patch("requests.request", request_spy):
            with TestClient(app) as c:
                r = c.post(f"/api/instances/{iid}/stop")
        request_spy.assert_called_once()
        assert r.status_code == 200, r.text


# --------------------------------------------------------------------------- #
#  `localm stop`                                                              #
# --------------------------------------------------------------------------- #

def _cli_entry(iid, host, pid=UNREACHABLE_PID, port=59971, token=None):
    return {"instance_id": iid, "pid": pid, "root_dir": "/proj/demo",
            "scheme": "http", "host": host, "port": port,
            "_path": f"/run/{iid}.json", "token": token or "cli-attach-token"}


class TestCliStopRefusesAForgedHost:
    def test_forged_host_is_never_dialed_and_falls_back_to_kill(
            self, monkeypatch, caplog):
        monkeypatch.delenv("LOCALM_API_KEY", raising=False)
        entry = _cli_entry("forgedclistop0001", FORGED_HOST)
        monkeypatch.setattr(instances, "reap_stale", lambda *a, **k: [])
        monkeypatch.setattr(instances, "list_entries", lambda *a, **k: [entry])
        kill_calls = []
        monkeypatch.setattr(
            instances, "kill_pid",
            lambda pid, timeout=None: kill_calls.append(pid) or True)
        post_spy = Mock()
        with patch("requests.post", post_spy), \
             caplog.at_level("WARNING", logger="localm"):
            res = CliRunner().invoke(main, ["stop", "forgedclistop0001"])
        post_spy.assert_not_called()
        assert kill_calls == [UNREACHABLE_PID]
        assert "not an address this machine holds" in caplog.text
        assert res.exit_code == 0, res.output

    def test_a_loopback_host_still_attempts_the_graceful_shutdown(
            self, monkeypatch):
        """The positive control: an ordinary loopback entry is unaffected -
        the graceful POST is still attempted, same as before."""
        monkeypatch.delenv("LOCALM_API_KEY", raising=False)
        entry = _cli_entry("loopbackclistop01", "127.0.0.1")
        monkeypatch.setattr(instances, "reap_stale", lambda *a, **k: [])
        monkeypatch.setattr(instances, "list_entries", lambda *a, **k: [entry])
        monkeypatch.setattr(instances, "kill_pid", lambda *a, **k: True)
        post_spy = Mock(side_effect=requests.ConnectionError("refused"))
        with patch("requests.post", post_spy), \
             patch("localm.tls.requests_verify", return_value=True):
            res = CliRunner().invoke(main, ["stop", "loopbackclistop01"])
        post_spy.assert_called_once()
        assert res.exit_code == 0, res.output
