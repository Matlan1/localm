# SPDX-License-Identifier: AGPL-3.0-or-later
"""`localm ps` (running per-directory instances) and `localm status` (the
server serving this directory), built on the instance discovery registry.
Also `localm stop`: id/prefix/--all target resolution, the graceful
/v1/server/shutdown HTTP path, and the direct-kill fallback."""

from types import SimpleNamespace
from unittest.mock import patch

import requests
from click.testing import CliRunner

from localm import bugreport, instances
from localm.cli import main


def _register(home, *, port, root, iid):
    return instances.register_instance(
        home, instance_id=iid, port=port, host="127.0.0.1",
        root_dir=root, mode="full", token="secret-" + iid, scheme="http")


def test_snapshot_marks_alive_and_strips_token(tmp_path):
    _register(tmp_path, port=8642, root=str(tmp_path / "a"), iid="aaaa1111")
    _register(tmp_path, port=8643, root=str(tmp_path / "b"), iid="bbbb2222")
    rows = instances.snapshot(tmp_path, probe=lambda e: e["port"] == 8642, reap=False)
    assert len(rows) == 2
    by_port = {r["port"]: r for r in rows}
    assert by_port[8642]["alive"] is True
    assert by_port[8643]["alive"] is False
    # A listing must never surface the attach token.
    assert all("token" not in r for r in rows)


def test_snapshot_empty(tmp_path):
    assert instances.snapshot(tmp_path, reap=False) == []


def test_snapshot_include_token_keeps_it(tmp_path):
    """An INTERNAL caller (the MCP server_activity tool) needs the attach
    token to authenticate its own request to a genuinely open instance - the
    opt-in companion to the strip-by-default case above. Never used for
    anything a human reads."""
    _register(tmp_path, port=8642, root=str(tmp_path / "a"), iid="aaaa1111")
    rows = instances.snapshot(tmp_path, reap=False, include_token=True)
    assert len(rows) == 1
    assert rows[0]["token"] == "secret-aaaa1111"


def test_ps_empty(monkeypatch):
    monkeypatch.setattr(instances, "snapshot", lambda *a, **k: [])
    res = CliRunner().invoke(main, ["ps"])
    assert res.exit_code == 0
    assert "No running localm instances" in res.output


def test_ps_lists_instances(monkeypatch):
    monkeypatch.setattr(instances, "snapshot", lambda *a, **k: [
        {"instance_id": "abcd1234ef", "alive": True, "mode": "full",
         "scheme": "http", "host": "127.0.0.1", "port": 8642, "pid": 4321,
         "root_dir": "/proj/demo"},
    ])
    res = CliRunner().invoke(main, ["ps"])
    assert res.exit_code == 0
    assert "abcd1234" in res.output      # id shown (truncated)
    assert "8642" in res.output          # address
    assert "demo" in res.output          # directory


def test_status_none(monkeypatch):
    monkeypatch.setattr(instances, "find_attachable", lambda *a, **k: None)
    res = CliRunner().invoke(main, ["status"])
    assert res.exit_code == 0
    assert "No localm server is serving" in res.output


def test_status_found(monkeypatch):
    monkeypatch.setattr(instances, "find_attachable", lambda *a, **k: {
        "scheme": "http", "host": "127.0.0.1", "port": 8642, "mode": "full",
        "pid": 4321, "version": "0.1.0"})
    res = CliRunner().invoke(main, ["status"])
    assert res.exit_code == 0
    assert "8642" in res.output
    assert "full" in res.output


def test_status_passes_the_registry_token_to_activity(monkeypatch):
    """`localm status` must thread the instance's own attach token
    through to read_activity, or a genuinely open server has no way to prove
    this is a local process and 403s the activity read."""
    monkeypatch.setattr(instances, "find_attachable", lambda *a, **k: {
        "scheme": "http", "host": "127.0.0.1", "port": 8642, "mode": "full",
        "pid": 4321, "version": "0.1.0", "token": "the-attach-token"})
    captured = {}

    def _fake_read_activity(scheme, port, instance_token=None):
        captured["instance_token"] = instance_token
        return "ok", {"now": 1.0, "operations": []}

    monkeypatch.setattr("localm.cli.models.read_activity", _fake_read_activity)
    res = CliRunner().invoke(main, ["status"])
    assert res.exit_code == 0
    assert captured["instance_token"] == "the-attach-token"


# ------------------------------------------------------------------ #
#  stop                                                               #
# ------------------------------------------------------------------ #

def _entry(iid, pid, root, port=8642):
    return {"instance_id": iid, "pid": pid, "root_dir": root, "scheme": "http",
            "host": "127.0.0.1", "port": port, "_path": f"/run/{iid}.json"}


def _resp(status_code):
    return SimpleNamespace(status_code=status_code)


def test_stop_all_no_instances(monkeypatch):
    monkeypatch.setattr(instances, "reap_stale", lambda *a, **k: [])
    monkeypatch.setattr(instances, "list_entries", lambda *a, **k: [])
    res = CliRunner().invoke(main, ["stop", "--all"])
    assert res.exit_code == 0
    assert "No running localm instances" in res.output


def test_stop_no_server_for_current_directory(monkeypatch):
    monkeypatch.setattr(instances, "reap_stale", lambda *a, **k: [])
    monkeypatch.setattr(instances, "list_entries", lambda *a, **k: [])
    monkeypatch.setattr(instances, "find_attachable", lambda *a, **k: None)
    res = CliRunner().invoke(main, ["stop"])
    assert res.exit_code == 1
    assert "No localm server is serving" in res.output


def test_stop_id_and_all_together_rejected():
    res = CliRunner().invoke(main, ["stop", "abcd1234", "--all"])
    assert res.exit_code == 1
    assert "not both" in res.output


def test_stop_unknown_id(monkeypatch):
    monkeypatch.setattr(instances, "reap_stale", lambda *a, **k: [])
    monkeypatch.setattr(instances, "list_entries",
                         lambda *a, **k: [_entry("abcd1234ef", 111, "/proj/demo")])
    res = CliRunner().invoke(main, ["stop", "zzzz"])
    assert res.exit_code == 1
    assert "No running instance matches" in res.output


def test_stop_ambiguous_id_lists_matches(monkeypatch):
    entries = [_entry("abcd1111", 111, "/proj/a"), _entry("abcd2222", 222, "/proj/b")]
    monkeypatch.setattr(instances, "reap_stale", lambda *a, **k: [])
    monkeypatch.setattr(instances, "list_entries", lambda *a, **k: entries)
    res = CliRunner().invoke(main, ["stop", "abcd"])
    assert res.exit_code == 1
    assert "matches 2 instances" in res.output
    assert "abcd1111" in res.output and "abcd2222" in res.output


def test_stop_by_id_graceful_shutdown_success(monkeypatch):
    entry = _entry("abcd1234ef", 999999, "/proj/demo")
    monkeypatch.setattr(instances, "reap_stale", lambda *a, **k: [])
    monkeypatch.setattr(instances, "list_entries", lambda *a, **k: [entry])
    # The server answered 200 and the pid is confirmed gone right away, so the
    # graceful path suffices and kill_pid is never reached.
    monkeypatch.setattr(instances, "pid_alive", lambda pid: False)
    monkeypatch.setattr(instances, "kill_pid",
                         lambda *a, **k: (_ for _ in ()).throw(
                             AssertionError("kill_pid must not run on the happy path")))
    captured = {}

    def fake_post(url, headers=None, timeout=None, verify=None):
        captured["url"] = url
        captured["headers"] = headers
        return _resp(200)

    with patch("requests.post", side_effect=fake_post), \
         patch("localm.tls.requests_verify", return_value=True):
        res = CliRunner().invoke(main, ["stop", "abcd1234"])

    assert res.exit_code == 0
    assert captured["url"] == "http://127.0.0.1:8642/v1/server/shutdown"
    assert "stopped" in res.output
    assert "abcd1234" in res.output


def test_stop_falls_back_to_kill_pid_when_graceful_shutdown_denied(monkeypatch):
    # A plain local instance with no LOCALM_API_KEY set refuses an unauthenticated
    # shutdown request (the open-mode management gate), the same 403 `localm unload`
    # hits unauthenticated, so `stop` falls back to killing the process directly.
    entry = _entry("abcd1234ef", 111, "/proj/demo")
    monkeypatch.setattr(instances, "reap_stale", lambda *a, **k: [])
    monkeypatch.setattr(instances, "list_entries", lambda *a, **k: [entry])
    killed = {}

    def fake_kill_pid(pid, timeout=10.0):
        killed["pid"] = pid
        return True

    monkeypatch.setattr(instances, "kill_pid", fake_kill_pid)

    with patch("requests.post", return_value=_resp(403)), \
         patch("localm.tls.requests_verify", return_value=True):
        res = CliRunner().invoke(main, ["stop", "abcd1234"])

    assert res.exit_code == 0
    assert killed == {"pid": 111}
    assert "LOCALM_API_KEY" in res.output   # names the fallback reason
    assert "stopped" in res.output


def test_stop_disarms_crash_guard_after_a_confirmed_direct_kill(monkeypatch):
    # A direct kill bypasses the server's own clean-shutdown code, which is what
    # normally clears the crash marker, so `stop` disarms the marker itself.
    entry = _entry("abcd1234ef", 111, "/proj/demo")
    monkeypatch.setattr(instances, "reap_stale", lambda *a, **k: [])
    monkeypatch.setattr(instances, "list_entries", lambda *a, **k: [entry])
    monkeypatch.setattr(instances, "kill_pid", lambda *a, **k: True)
    disarmed = {}
    monkeypatch.setattr(
        bugreport, "disarm_crash_guard",
        lambda home, instance_id=None: disarmed.update(
            home=home, instance_id=instance_id))

    with patch("requests.post", return_value=_resp(403)), \
         patch("localm.tls.requests_verify", return_value=True):
        res = CliRunner().invoke(main, ["stop", "abcd1234"])

    assert res.exit_code == 0
    assert "home" in disarmed
    # The direct-kill disarm targets THIS entry's own marker (its instance_id),
    # never an unscoped or shared one, so a live sibling instance sharing the same
    # LOCALM_HOME keeps its marker.
    assert disarmed["instance_id"] == "abcd1234ef"


def test_stop_does_not_disarm_crash_guard_when_kill_fails(monkeypatch):
    # If the kill did NOT confirm the process is gone, the marker survives.
    entry = _entry("abcd1234ef", 111, "/proj/demo")
    monkeypatch.setattr(instances, "reap_stale", lambda *a, **k: [])
    monkeypatch.setattr(instances, "list_entries", lambda *a, **k: [entry])
    monkeypatch.setattr(instances, "kill_pid", lambda *a, **k: False)
    disarm_calls = []
    monkeypatch.setattr(
        bugreport, "disarm_crash_guard",
        lambda home, instance_id=None: disarm_calls.append(home))

    with patch("requests.post", return_value=_resp(403)), \
         patch("localm.tls.requests_verify", return_value=True):
        res = CliRunner().invoke(main, ["stop", "abcd1234"])

    assert res.exit_code == 1
    assert disarm_calls == []


def test_stop_falls_back_to_kill_pid_when_server_unreachable(monkeypatch):
    entry = _entry("abcd1234ef", 4321, "/proj/demo")
    monkeypatch.setattr(instances, "reap_stale", lambda *a, **k: [])
    monkeypatch.setattr(instances, "list_entries", lambda *a, **k: [entry])
    killed = {}

    def fake_kill_pid(pid, timeout=10.0):
        killed["pid"] = pid
        killed["timeout"] = timeout
        return True

    monkeypatch.setattr(instances, "kill_pid", fake_kill_pid)

    with patch("requests.post", side_effect=requests.RequestException("refused")), \
         patch("localm.tls.requests_verify", return_value=True):
        res = CliRunner().invoke(main, ["stop", "abcd1234", "--timeout", "3"])

    assert res.exit_code == 0
    assert killed == {"pid": 4321, "timeout": 3.0}
    assert "stopped" in res.output


def test_stop_reports_failure_when_kill_cannot_confirm(monkeypatch):
    entry = _entry("abcd1234ef", 4321, "/proj/demo")
    monkeypatch.setattr(instances, "reap_stale", lambda *a, **k: [])
    monkeypatch.setattr(instances, "list_entries", lambda *a, **k: [entry])
    monkeypatch.setattr(instances, "kill_pid", lambda *a, **k: False)

    with patch("requests.post", side_effect=requests.RequestException("refused")), \
         patch("localm.tls.requests_verify", return_value=True):
        res = CliRunner().invoke(main, ["stop", "abcd1234"])

    assert res.exit_code == 1
    assert "could not confirm" in res.output
