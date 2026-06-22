# SPDX-License-Identifier: AGPL-3.0-or-later
"""B4: `localm ps` (running per-directory instances) and `localm status` (the
server serving this directory), built on the instance discovery registry."""

from click.testing import CliRunner

from localm import instances
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
