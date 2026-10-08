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
  * `instances.list_machine_peers` covers the OTHER installs, found by detecting
    the running instances directly (listening ports in localm's range that answer
    /whoami and /v1/instances/status), whatever their LOCALM_HOME.

The peers here are REAL loopback HTTP servers (tests/_peer_servers.py) rather than
a patched `requests`, so the identity handshake gating every listed peer is
exercised for real - including the impostor cases, where the responder is a
genuine HTTP server answering with a non-localm app name or no status.
"""

from __future__ import annotations

import json
import os

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from localm import gpu_registry, instances
from tests._peer_servers import enable_detection, localm_whoami, peer_server


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

    def test_snapshot_ignores_instances_detected_on_the_machine(self, tmp_path,
                                                                monkeypatch):
        home = tmp_path / "homeA"
        home.mkdir()
        with peer_server(localm_whoami("cccc000000000003", root_dir="/proj/other",
                                       version="9.9.9")) as srv:
            enable_detection(monkeypatch, srv.port)
            assert instances.snapshot(home) == []


# --------------------------------------------------------------------------- #
#  list_machine_peers                                                         #
# --------------------------------------------------------------------------- #

class TestListMachinePeers:
    def test_finds_a_running_instance_of_another_install(self, tmp_path, monkeypatch):
        home = tmp_path / "homeA"
        home.mkdir()
        with peer_server(localm_whoami("cccc000000000003", root_dir="/proj/other",
                                       version="9.9.9")) as srv:
            enable_detection(monkeypatch, srv.port)

            peers = instances.list_machine_peers(home)

        assert len(peers) == 1
        peer = peers[0]
        assert peer["instance_id"] == "cccc000000000003"
        assert peer["same_install"] is False
        assert peer["alive"] is True
        assert peer["port"] == srv.port
        # root_dir/mode/version come from the instance's own /whoami answer.
        assert peer["root_dir"] == "/proj/other"
        assert peer["mode"] == "full"
        assert peer["version"] == "9.9.9"

    def test_only_ever_dials_loopback(self, tmp_path, monkeypatch):
        """Detection dials addresses on this machine's loopback only, never a host
        named by anything a peer says. Spies on the real requests.get so a
        genuine loopback peer is still exercised for real in the same test."""
        import requests as requests_module
        home = tmp_path / "homeA"
        home.mkdir()
        real_get = requests_module.get
        calls = []

        def spy_get(url, *a, **kw):
            calls.append(url)
            return real_get(url, *a, **kw)

        monkeypatch.setattr(requests_module, "get", spy_get)
        with peer_server(localm_whoami("cccc00000000000f", root_dir="/proj/other",
                                       host="attacker.example")) as srv:
            enable_detection(monkeypatch, srv.port)

            peers = instances.list_machine_peers(home)

        assert calls, "no probe was sent"
        assert all(u.startswith(("http://127.0.0.1:", "https://127.0.0.1:",
                                 "http://[::1]:", "https://[::1]:")) for u in calls), calls
        assert len(peers) == 1 and peers[0]["host"] == "127.0.0.1"

    def test_excludes_an_instance_this_home_already_lists(self, tmp_path, monkeypatch):
        home = tmp_path / "homeA"
        with peer_server(localm_whoami("dddd000000000004", root_dir="/proj/mine",
                                       mode="api")) as srv:
            write_home_entry(home, instance_id="dddd000000000004", port=srv.port)
            enable_detection(monkeypatch, srv.port)

            assert instances.list_machine_peers(home) == [], (
                "an instance of THIS install must not be listed twice")

    def test_rejects_a_responder_that_is_not_localm(self, tmp_path, monkeypatch):
        home = tmp_path / "homeA"
        home.mkdir()
        with peer_server({"app": "something-else",
                          "instance_id": "ffff000000000006"}) as srv:
            enable_detection(monkeypatch, srv.port)

            assert instances.list_machine_peers(home) == []

    def test_rejects_a_responder_whose_status_names_another_instance(
            self, tmp_path, monkeypatch):
        home = tmp_path / "homeA"
        home.mkdir()
        with peer_server(localm_whoami("eeee000000000005"),
                         status={"instance_id": "not-the-same-id", "pid": 1}) as srv:
            enable_detection(monkeypatch, srv.port)

            assert instances.list_machine_peers(home) == [], (
                "a whoami and a status that disagree are not one instance")

    def test_an_instance_that_does_not_coordinate_is_not_listed(self, tmp_path,
                                                                monkeypatch):
        """An isolated (or network-bound) instance answers /whoami but serves no
        coordination status; it is invisible to discovery."""
        home = tmp_path / "homeA"
        home.mkdir()
        with peer_server(localm_whoami("aaaa000000000009"), status=None) as srv:
            enable_detection(monkeypatch, srv.port)

            assert instances.list_machine_peers(home) == []

    def test_an_instance_that_has_stopped_is_not_listed(self, tmp_path, monkeypatch):
        home = tmp_path / "homeA"
        home.mkdir()
        with peer_server(localm_whoami("aaaa000000000007")) as srv:
            port = srv.port
        enable_detection(monkeypatch, port)

        assert instances.list_machine_peers(home) == []

    def test_never_returns_a_credential(self, tmp_path, monkeypatch):
        home = tmp_path / "homeA"
        home.mkdir()
        with peer_server(localm_whoami("bbbb000000000008", root_dir="/proj/other")) as srv:
            enable_detection(monkeypatch, srv.port)

            peers = instances.list_machine_peers(home)

        assert len(peers) == 1
        assert not any("token" in k for k in peers[0])

    def test_a_failing_lookup_is_not_an_error(self, tmp_path, monkeypatch):
        home = tmp_path / "homeA"
        home.mkdir()

        def boom(**kw):
            raise RuntimeError("socket table unreadable")

        monkeypatch.setattr(gpu_registry, "list_gpu_peers", boom)
        assert instances.list_machine_peers(home) == []


# --------------------------------------------------------------------------- #
#  GET /api/instances + POST /api/instances/{id}/stop                         #
# --------------------------------------------------------------------------- #

@pytest.fixture
def instances_app(tmp_path, monkeypatch):
    """The GUI stack on a throwaway home, standing in for a REAL advertised
    server: `instance_id` set and not isolated.

    Both route halves look for other installs' instances only under exactly that
    condition. A bare app leaves instance_id unset, which is what keeps every
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
            self, instances_app, monkeypatch):
        app, home = instances_app
        with peer_server(localm_whoami("cccc000000000009", root_dir="/proj/other",
                                       version="9.9.9")) as srv:
            enable_detection(monkeypatch, srv.port)
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
                                                            monkeypatch):
        app, home = instances_app
        with peer_server(localm_whoami("cccc00000000000b", root_dir="/proj/other")) as srv:
            enable_detection(monkeypatch, srv.port)
            with TestClient(app) as c:
                body = c.get("/api/instances").json()

        blob = json.dumps(body)
        assert "_path" not in blob
        assert "token" not in blob

    def test_stopping_another_installs_instance_is_refused_with_a_reason(
            self, instances_app, monkeypatch):
        app, home = instances_app
        with peer_server(localm_whoami("cccc00000000000c", root_dir="/proj/other")) as srv:
            enable_detection(monkeypatch, srv.port)
            with TestClient(app) as c:
                resp = c.post("/api/instances/cccc00000000000c/stop")

        assert resp.status_code == 409, (
            "a cross-install id is a known instance this server will not stop, "
            "not an unknown one")
        detail = resp.json()["detail"]
        assert "different localm install" in detail
        assert "crash" in detail

    def test_an_unknown_id_is_still_a_404(self, instances_app):
        app, home = instances_app
        with TestClient(app) as c:
            resp = c.post("/api/instances/no-such-instance/stop")
        assert resp.status_code == 404


class TestIsolationGate:
    """A server that is invisible to discovery must not look for other instances.

    `--isolated` is documented as invisible to discovery, and a bare test app
    never advertises at all; both would otherwise start listing and probing
    every localm running on the box. Caught for real: before this gate existed,
    two pre-existing tests in test_instances_gui_route_2026_08_20.py went red
    because the test app picked up live servers belonging to this machine.
    """

    def _app_listing(self, app, monkeypatch):
        with peer_server(localm_whoami("cccc00000000000d", root_dir="/proj/other")) as srv:
            enable_detection(monkeypatch, srv.port)
            with TestClient(app) as c:
                return c.get("/api/instances").json()["instances"]

    def test_an_isolated_server_lists_no_machine_peers(self, instances_app, monkeypatch):
        app, home = instances_app
        app.state.instance_isolated = True
        assert self._app_listing(app, monkeypatch) == []

    def test_an_unadvertised_app_lists_no_machine_peers(self, instances_app, monkeypatch):
        app, home = instances_app
        app.state.instance_id = None
        assert self._app_listing(app, monkeypatch) == []

    def test_an_isolated_server_does_not_explain_a_cross_install_id(
            self, instances_app, monkeypatch):
        app, home = instances_app
        app.state.instance_isolated = True
        with peer_server(localm_whoami("cccc00000000000e", root_dir="/proj/other")) as srv:
            enable_detection(monkeypatch, srv.port)
            with TestClient(app) as c:
                resp = c.post("/api/instances/cccc00000000000e/stop")
        assert resp.status_code == 404
