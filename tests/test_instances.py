# SPDX-License-Identifier: AGPL-3.0-or-later
"""Instance discovery registry (H6 phase 3, advertise-only): localm/instances.py
+ the GET /whoami endpoint.

Covers root-dir resolution, atomic register/unregister, stale reaping (dead PID
removed, live + self kept, corrupt removed), the identity payload (no token/pid
leak), and the advertise() context manager lifecycle.
"""

import json
import os

import pytest

from localm import instances


# ------------------------------------------------------------------ #
#  root_dir resolution (decision 2)                                  #
# ------------------------------------------------------------------ #

def test_resolve_root_dir_walks_to_marker(tmp_path):
    proj = tmp_path / "proj"
    (proj / ".git").mkdir(parents=True)
    sub = proj / "a" / "b"
    sub.mkdir(parents=True)
    assert instances.resolve_root_dir(start=str(sub)) == str(proj.resolve())


def test_resolve_root_dir_localcoder_marker(tmp_path):
    proj = tmp_path / "proj"
    (proj / ".localcoder").mkdir(parents=True)
    assert instances.resolve_root_dir(start=str(proj)) == str(proj.resolve())


def test_resolve_root_dir_falls_back_to_start(tmp_path):
    plain = tmp_path / "no-marker"
    plain.mkdir()
    assert instances.resolve_root_dir(start=str(plain)) == str(plain.resolve())


def test_resolve_root_dir_override_wins(tmp_path):
    other = tmp_path / "explicit"
    other.mkdir()
    assert instances.resolve_root_dir(start=str(tmp_path),
                                      override=str(other)) == str(other.resolve())


# ------------------------------------------------------------------ #
#  register / list / unregister                                      #
# ------------------------------------------------------------------ #

def test_register_writes_full_schema(tmp_path):
    iid = instances.new_instance_id()
    path = instances.register_instance(
        tmp_path, instance_id=iid, port=8642, host="0.0.0.0",
        root_dir=str(tmp_path), mode="full", token="tok-secret")
    assert path.exists()
    entry = json.loads(path.read_text(encoding="utf-8"))
    for key in ("instance_id", "pid", "port", "host", "root_dir", "mode",
                "version", "started", "token"):
        assert key in entry, f"missing {key}"
    assert entry["instance_id"] == iid
    assert entry["pid"] == os.getpid()
    assert entry["mode"] == "full"
    assert entry["token"] == "tok-secret"


def test_list_entries_and_unregister(tmp_path):
    iid = instances.new_instance_id()
    path = instances.register_instance(
        tmp_path, instance_id=iid, port=9000, host="127.0.0.1",
        root_dir=str(tmp_path), mode="api", token="t")
    entries = instances.list_entries(tmp_path)
    assert len(entries) == 1 and entries[0]["instance_id"] == iid
    instances.unregister_instance(path)
    assert not path.exists()
    assert instances.list_entries(tmp_path) == []


def test_list_entries_skips_corrupt(tmp_path):
    instances.run_dir(tmp_path).mkdir(parents=True)
    (instances.run_dir(tmp_path) / "bad.json").write_text("{not json", encoding="utf-8")
    assert instances.list_entries(tmp_path) == []


@pytest.mark.skipif(__import__("sys").platform == "win32", reason="POSIX modes only")
def test_registry_file_is_owner_only_on_posix(tmp_path):
    import stat
    iid = instances.new_instance_id()
    path = instances.register_instance(
        tmp_path, instance_id=iid, port=1, host="127.0.0.1",
        root_dir=str(tmp_path), mode="api", token="t")
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600


# ------------------------------------------------------------------ #
#  reaping                                                            #
# ------------------------------------------------------------------ #

def _raw(tmp_path, iid, pid):
    d = instances.run_dir(tmp_path)
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{iid}.json"
    p.write_text(json.dumps({"instance_id": iid, "pid": pid, "port": 1,
                             "host": "127.0.0.1", "root_dir": str(tmp_path),
                             "mode": "api", "version": "0", "started": "0",
                             "token": "t"}), encoding="utf-8")
    return p


def test_reap_removes_dead_keeps_live_and_self(tmp_path):
    dead = _raw(tmp_path, "dead0001", 111)
    live = _raw(tmp_path, "live0001", 222)
    mine = _raw(tmp_path, "self0001", 333)
    alive_pids = {222, 333}
    removed = instances.reap_stale(
        tmp_path, self_id="self0001",
        is_alive=lambda e: int(e["pid"]) in alive_pids)
    assert "dead0001" in removed
    assert not dead.exists()
    assert live.exists(), "a live instance must never be reaped"
    assert mine.exists(), "self must never be reaped"


def test_reap_removes_corrupt(tmp_path):
    instances.run_dir(tmp_path).mkdir(parents=True)
    bad = instances.run_dir(tmp_path) / "corrupt.json"
    bad.write_text("xxx", encoding="utf-8")
    instances.reap_stale(tmp_path, is_alive=lambda e: True)
    assert not bad.exists()


def test_reap_keeps_all_on_probe_error(tmp_path):
    e = _raw(tmp_path, "keep0001", 444)

    def boom(_entry):
        raise RuntimeError("probe failed")

    instances.reap_stale(tmp_path, is_alive=boom)
    assert e.exists(), "an errored probe must not delete the entry"


def test_pid_alive_self_and_invalid():
    assert instances.pid_alive(os.getpid()) is True
    assert instances.pid_alive(-1) is False
    assert instances.pid_alive(0) is False


# ------------------------------------------------------------------ #
#  whoami payload + advertise lifecycle                              #
# ------------------------------------------------------------------ #

def test_whoami_payload_has_identity_no_secrets():
    p = instances.whoami_payload("abc123", "/home/x/proj", "full")
    assert p["app"] == "localm"
    assert p["instance_id"] == "abc123"
    assert p["root_dir"] == "/home/x/proj"
    assert p["mode"] == "full"
    assert "version" in p
    assert "token" not in p and "pid" not in p


class _FakeApp:
    class _State:
        pass

    def __init__(self):
        self.state = _FakeApp._State()


def test_advertise_registers_sets_state_and_cleans_up(tmp_path):
    app = _FakeApp()
    assert instances.list_entries(tmp_path) == []
    with instances.advertise(app, tmp_path, host="0.0.0.0", port=8642, mode="full") as info:
        entries = instances.list_entries(tmp_path)
        assert len(entries) == 1
        assert entries[0]["instance_id"] == info["instance_id"]
        # app.state is wired for /whoami
        assert app.state.instance_id == info["instance_id"]
        assert app.state.instance_mode == "full"
        assert app.state.root_dir == info["root_dir"]
        assert app.state.instance_token  # token set, not exposed via whoami
    # cleaned up on exit (only our file)
    assert instances.list_entries(tmp_path) == []


def test_advertise_project_override(tmp_path):
    app = _FakeApp()
    proj = tmp_path / "explicit-root"
    proj.mkdir()
    with instances.advertise(app, tmp_path, host="127.0.0.1", port=1,
                             mode="api", project=str(proj)) as info:
        assert info["root_dir"] == str(proj.resolve())


# ------------------------------------------------------------------ #
#  GET /whoami endpoint                                              #
# ------------------------------------------------------------------ #

def _mock_engine():
    from unittest.mock import MagicMock
    e = MagicMock()
    e.display_name = "test-model"
    type(e).loaded = property(lambda self: True)
    return e


def test_whoami_endpoint_reports_state_unauthenticated(tmp_path):
    from fastapi.testclient import TestClient
    from localm.inference.http_server import create_app

    app = create_app(_mock_engine())
    app.state.instance_id = "iid-xyz"
    app.state.root_dir = str(tmp_path)
    app.state.instance_mode = "full"

    client = TestClient(app)
    r = client.get("/whoami")          # no Authorization header
    assert r.status_code == 200
    body = r.json()
    assert body == {
        "app": "localm",
        "version": body["version"],
        "instance_id": "iid-xyz",
        "root_dir": str(tmp_path),
        "mode": "full",
    }
    assert "token" not in body and "pid" not in body


def test_whoami_endpoint_before_wiring_returns_nulls():
    from fastapi.testclient import TestClient
    from localm.inference.http_server import create_app

    client = TestClient(create_app(_mock_engine()))
    body = client.get("/whoami").json()
    assert body["app"] == "localm"
    assert body["instance_id"] is None and body["mode"] is None
