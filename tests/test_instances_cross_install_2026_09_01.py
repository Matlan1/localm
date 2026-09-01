# SPDX-License-Identifier: AGPL-3.0-or-later
"""Cross-install instance discovery: `instances.list_machine_peers`, and the
`same_install` flag GET /api/instances puts on every row.

An install's own registry lives under its LOCALM_HOME, so two installs with
different homes are invisible to each other there. These tests pin BOTH halves
of the resulting contract:

  * `instances.snapshot` STAYS home-scoped. `selfclient.read_model_file_hold`
    treats a registry miss as proof a model file is not held elsewhere, which is
    sound only while every server it reaches shares one registry - so widening
    snapshot would turn a safety refusal into a false all-clear.
  * `instances.list_machine_peers` covers the OTHER installs, read from the
    machine-wide coordination registry that every non-isolated server writes
    whatever its LOCALM_HOME.

The /whoami probes here run against a REAL loopback HTTP server rather than a
patched `requests`, so the identity handshake gating every listed peer is
exercised for real - including the impostor cases, where the responder is a
genuine HTTP server answering with the wrong instance_id or a non-localm app
name.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from localm import gpu_registry, instances

_COORD_TOKEN = "coordination-token-that-must-never-be-returned"


class _WhoamiHandler(BaseHTTPRequestHandler):
    payload: dict = {}

    def do_GET(self):
        if self.path != "/whoami":
            self.send_response(404)
            self.end_headers()
            return
        body = json.dumps(self.payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        return


@contextmanager
def whoami_server(payload: dict):
    """A real HTTP server answering GET /whoami with *payload*, on a throwaway
    loopback port. Yields the port."""
    handler = type("_Bound", (_WhoamiHandler,), {"payload": payload})
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        yield srv.server_address[1]
    finally:
        srv.shutdown()
        srv.server_close()
        thread.join(timeout=5)


def write_gpu_entry(gpu_dir, *, instance_id, port, pid,
                    host="127.0.0.1", scheme="http"):
    """A machine-wide coordination entry, matching gpu_registry.write_entry's
    schema.

    *pid* is required and must be a live process OTHER than this one: an entry
    is dropped when its pid is gone, and also when its pid is the caller's own
    (that entry is the caller, not a peer). Passing os.getpid() here makes every
    peer vanish, which every negative assertion in this file would still pass
    on. Use the `foreign_pid` fixture."""
    gpu_dir.mkdir(parents=True, exist_ok=True)
    entry = {
        "instance_id": instance_id,
        "pid": pid,
        "port": port,
        "host": host,
        "scheme": scheme,
        "model": None,
        "vram_estimate_bytes": None,
        "gpu_index": 0,
        "updated_at": "2026-09-01T00:00:00+00:00",
        "coordination_token": _COORD_TOKEN,
    }
    (gpu_dir / f"{instance_id}.json").write_text(json.dumps(entry),
                                                 encoding="utf-8")
    return instance_id


def write_home_entry(home, *, instance_id, port=59999, pid=None,
                     root_dir="/proj/mine", mode="api"):
    """An entry in one install's OWN registry, matching register_instance's
    schema."""
    entry = dict(instance_id=instance_id, pid=os.getpid() if pid is None else pid,
                 port=port, host="127.0.0.1", scheme="http", root_dir=root_dir,
                 mode=mode, version="test", token=instances.new_token(),
                 started="2026-09-01T00:00:00+00:00")
    path = instances.registry_path(home, instance_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entry), encoding="utf-8")
    return instance_id


@pytest.fixture
def foreign_pid():
    """A REAL live process that is not this one, for a peer entry's pid."""
    proc = subprocess.Popen([sys.executable, "-c",
                             "import time; time.sleep(120)"])
    try:
        assert proc.pid != os.getpid()
        yield proc.pid
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=15)


@pytest.fixture
def gpu_dir(tmp_path, monkeypatch):
    """Point the machine-wide registry at a throwaway dir, so a test never reads
    or writes the real one shared by every localm on the box."""
    d = tmp_path / "machine-registry"
    d.mkdir()
    monkeypatch.setattr(gpu_registry, "registry_dir", lambda: d)
    return d


def a_dead_pid():
    """A pid that has genuinely exited."""
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait(timeout=30)
    return proc.pid


# --------------------------------------------------------------------------- #
#  The boundary snapshot() keeps                                              #
# --------------------------------------------------------------------------- #

class TestSnapshotStaysHomeScoped:
    def test_snapshot_does_not_include_another_homes_entry(self, tmp_path):
        home_a = tmp_path / "homeA"
        home_b = tmp_path / "homeB"
        write_home_entry(home_a, instance_id="aaaa000000000001")
        write_home_entry(home_b, instance_id="bbbb000000000002")

        ids = {r["instance_id"] for r in
               instances.snapshot(home_a, probe=lambda e: True, reap=False)}

        assert ids == {"aaaa000000000001"}, (
            "snapshot must stay scoped to its own home: "
            "selfclient.read_model_file_hold reads a registry miss as proof a "
            "model file is not held elsewhere")

    def test_snapshot_ignores_the_machine_wide_registry(self, tmp_path, gpu_dir,
                                                       foreign_pid):
        home = tmp_path / "homeA"
        home.mkdir()
        with whoami_server({"app": "localm", "instance_id": "cccc000000000003",
                            "root_dir": "/proj/other", "mode": "full",
                            "version": "9.9.9"}) as port:
            write_gpu_entry(gpu_dir, instance_id="cccc000000000003", port=port,
                            pid=foreign_pid)
            assert instances.snapshot(home) == []


# --------------------------------------------------------------------------- #
#  list_machine_peers                                                         #
# --------------------------------------------------------------------------- #

class TestListMachinePeers:
    def test_finds_a_peer_registered_under_another_home(self, tmp_path, gpu_dir,
                                                        foreign_pid):
        home = tmp_path / "homeA"
        home.mkdir()
        with whoami_server({"app": "localm", "instance_id": "cccc000000000003",
                            "root_dir": "/proj/other", "mode": "full",
                            "version": "9.9.9"}) as port:
            write_gpu_entry(gpu_dir, instance_id="cccc000000000003", port=port,
                            pid=foreign_pid)

            peers = instances.list_machine_peers(home)

        assert len(peers) == 1
        peer = peers[0]
        assert peer["instance_id"] == "cccc000000000003"
        assert peer["same_install"] is False
        assert peer["alive"] is True
        assert peer["port"] == port
        # root_dir/mode/version come from the handshake: the coordination entry
        # records none of them.
        assert peer["root_dir"] == "/proj/other"
        assert peer["mode"] == "full"
        assert peer["version"] == "9.9.9"

    def test_excludes_an_instance_this_home_already_lists(self, tmp_path, gpu_dir,
                                                          foreign_pid):
        home = tmp_path / "homeA"
        with whoami_server({"app": "localm", "instance_id": "dddd000000000004",
                            "root_dir": "/proj/mine", "mode": "api"}) as port:
            write_home_entry(home, instance_id="dddd000000000004", port=port)
            write_gpu_entry(gpu_dir, instance_id="dddd000000000004", port=port,
                            pid=foreign_pid)

            assert instances.list_machine_peers(home) == [], (
                "an instance of THIS install must not be listed twice")

    def test_rejects_a_responder_whose_instance_id_does_not_match(self, tmp_path,
                                                                  gpu_dir,
                                                                  foreign_pid):
        home = tmp_path / "homeA"
        home.mkdir()
        with whoami_server({"app": "localm", "instance_id": "not-the-same-id",
                            "root_dir": "/proj/other", "mode": "full"}) as port:
            write_gpu_entry(gpu_dir, instance_id="eeee000000000005", port=port,
                            pid=foreign_pid)

            assert instances.list_machine_peers(home) == [], (
                "a reused port answering with a different identity is not the "
                "registered instance")

    def test_rejects_a_responder_that_is_not_localm(self, tmp_path, gpu_dir,
                                                    foreign_pid):
        home = tmp_path / "homeA"
        home.mkdir()
        with whoami_server({"app": "something-else",
                            "instance_id": "ffff000000000006"}) as port:
            write_gpu_entry(gpu_dir, instance_id="ffff000000000006", port=port,
                            pid=foreign_pid)

            assert instances.list_machine_peers(home) == []

    def test_a_dead_pid_is_never_listed(self, tmp_path, gpu_dir):
        home = tmp_path / "homeA"
        home.mkdir()
        with whoami_server({"app": "localm", "instance_id": "aaaa000000000007",
                            "root_dir": "/proj/other", "mode": "full"}) as port:
            write_gpu_entry(gpu_dir, instance_id="aaaa000000000007", port=port,
                            pid=a_dead_pid())

            assert instances.list_machine_peers(home) == []

    def test_never_returns_the_coordination_token(self, tmp_path, gpu_dir,
                                                  foreign_pid):
        home = tmp_path / "homeA"
        home.mkdir()
        with whoami_server({"app": "localm", "instance_id": "bbbb000000000008",
                            "root_dir": "/proj/other", "mode": "full"}) as port:
            write_gpu_entry(gpu_dir, instance_id="bbbb000000000008", port=port,
                            pid=foreign_pid)

            peers = instances.list_machine_peers(home)

        assert len(peers) == 1
        assert "coordination_token" not in peers[0]
        assert "token" not in peers[0]
        assert _COORD_TOKEN not in json.dumps(peers[0])

    def test_an_unreadable_machine_registry_is_not_an_error(self, tmp_path,
                                                            monkeypatch):
        home = tmp_path / "homeA"
        home.mkdir()
        monkeypatch.setattr(gpu_registry, "registry_dir",
                            lambda: tmp_path / "nope" / "missing")
        assert instances.list_machine_peers(home) == []


# --------------------------------------------------------------------------- #
#  GET /api/instances + POST /api/instances/{id}/stop                         #
# --------------------------------------------------------------------------- #

@pytest.fixture
def instances_app(tmp_path, monkeypatch):
    """The GUI stack on a throwaway home, standing in for a REAL advertised
    server: `instance_id` set and not isolated.

    Both route halves read the machine-wide registry only under exactly that
    condition, mirroring the gate http_server.py puts on WRITING the same
    registry. A bare app leaves instance_id unset, which is what keeps every
    other GUI test in the suite from probing whatever localm happens to be
    running on the box - see TestIsolationGate."""
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
    app.state.instance_id = "5e1f000000000000"
    app.state.instance_isolated = False
    return app, home


class TestRouteSpansInstalls:
    def test_lists_a_peer_from_another_install_flagged_same_install_false(
            self, instances_app, gpu_dir, foreign_pid):
        app, home = instances_app
        with whoami_server({"app": "localm", "instance_id": "cccc000000000009",
                            "root_dir": "/proj/other", "mode": "full",
                            "version": "9.9.9"}) as port:
            write_gpu_entry(gpu_dir, instance_id="cccc000000000009", port=port,
                            pid=foreign_pid)
            write_home_entry(home, instance_id="aaaa00000000000a")

            with TestClient(app) as c:
                body = c.get("/api/instances").json()

        rows = {r["instance_id"]: r for r in body["instances"]}
        assert "cccc000000000009" in rows, (
            "an instance of another install must be listed, or the card's "
            "'every one running on this machine' promise is false")
        assert rows["cccc000000000009"]["same_install"] is False
        assert rows["cccc000000000009"]["alive"] is True
        assert rows["cccc000000000009"]["root_dir"] == "/proj/other"
        assert rows["aaaa00000000000a"]["same_install"] is True

    def test_a_peer_row_leaks_no_token_and_no_registry_path(self, instances_app,
                                                            gpu_dir,
                                                            foreign_pid):
        app, home = instances_app
        with whoami_server({"app": "localm", "instance_id": "cccc00000000000b",
                            "root_dir": "/proj/other", "mode": "full"}) as port:
            write_gpu_entry(gpu_dir, instance_id="cccc00000000000b", port=port,
                            pid=foreign_pid)
            with TestClient(app) as c:
                body = c.get("/api/instances").json()

        blob = json.dumps(body)
        assert _COORD_TOKEN not in blob
        assert "_path" not in blob
        assert "token" not in blob

    def test_stopping_another_installs_instance_is_refused_with_a_reason(
            self, instances_app, gpu_dir, foreign_pid):
        app, home = instances_app
        with whoami_server({"app": "localm", "instance_id": "cccc00000000000c",
                            "root_dir": "/proj/other", "mode": "full"}) as port:
            write_gpu_entry(gpu_dir, instance_id="cccc00000000000c", port=port,
                            pid=foreign_pid)
            with TestClient(app) as c:
                resp = c.post("/api/instances/cccc00000000000c/stop")

        assert resp.status_code == 409, (
            "a cross-install id is a known instance this server will not stop, "
            "not an unknown one")
        detail = resp.json()["detail"]
        assert "different localm install" in detail
        assert "crash" in detail

    def test_an_unknown_id_is_still_a_404(self, instances_app, gpu_dir):
        app, home = instances_app
        with TestClient(app) as c:
            resp = c.post("/api/instances/no-such-instance/stop")
        assert resp.status_code == 404


class TestIsolationGate:
    """A server that registers in NO machine-wide registry must not read one.

    `--isolated` is documented as invisible to discovery, and a bare test app
    never advertises at all; both would otherwise start listing and probing
    every localm running on the box. Caught for real: before this gate existed,
    two pre-existing tests in test_instances_gui_route_2026_08_20.py went red
    because the test app picked up live servers belonging to this machine.
    """

    def _app_listing(self, app, gpu_dir, foreign_pid, home):
        with whoami_server({"app": "localm", "instance_id": "cccc00000000000d",
                            "root_dir": "/proj/other", "mode": "full"}) as port:
            write_gpu_entry(gpu_dir, instance_id="cccc00000000000d", port=port,
                            pid=foreign_pid)
            with TestClient(app) as c:
                return c.get("/api/instances").json()["instances"]

    def test_an_isolated_server_lists_no_machine_peers(self, instances_app,
                                                       gpu_dir, foreign_pid):
        app, home = instances_app
        app.state.instance_isolated = True
        assert self._app_listing(app, gpu_dir, foreign_pid, home) == []

    def test_an_unadvertised_app_lists_no_machine_peers(self, instances_app,
                                                        gpu_dir, foreign_pid):
        app, home = instances_app
        app.state.instance_id = None
        assert self._app_listing(app, gpu_dir, foreign_pid, home) == []

    def test_an_isolated_server_does_not_explain_a_cross_install_id(
            self, instances_app, gpu_dir, foreign_pid):
        app, home = instances_app
        app.state.instance_isolated = True
        with whoami_server({"app": "localm", "instance_id": "cccc00000000000e",
                            "root_dir": "/proj/other", "mode": "full"}) as port:
            write_gpu_entry(gpu_dir, instance_id="cccc00000000000e", port=port,
                            pid=foreign_pid)
            with TestClient(app) as c:
                resp = c.post("/api/instances/cccc00000000000e/stop")
        assert resp.status_code == 404
