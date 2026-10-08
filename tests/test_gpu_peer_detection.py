# SPDX-License-Identifier: AGPL-3.0-or-later
"""Live detection of, and communication with, other running localm instances.

Nothing is written to disk: running instances are found from the OS's table of
listening ports (localm/listeners.py), identified over HTTP (``/whoami`` then
``/v1/instances/status``), and asked to release VRAM over HTTP
(``/v1/instances/cooperate-unload``), which the receiver honours only after the
requester confirms the request on ``/v1/instances/vouch``.

The peers are REAL loopback HTTP servers (tests/_peer_servers.py), so the socket
table, the identity handshake and the request / vouch exchange all run for real.
"""

from __future__ import annotations

import ast
import inspect
import os
import socket
import struct
import sys
import time

import pytest
from fastapi.testclient import TestClient

from localm import gpu_registry, listeners
from localm.inference import http_server as hs
from localm.inference.http_server import create_app
from tests._peer_servers import (FOREIGN_PID, default_status, enable_detection,
                                 localm_whoami, peer_server)


@pytest.fixture(autouse=True)
def _clean_coordination_state():
    """No cross-test leakage of coordination state, pending requests or the local
    status provider."""
    hs._gpu_coord = None
    gpu_registry.set_local_status_provider(None)
    gpu_registry._pending.clear()
    yield
    hs._gpu_coord = None
    gpu_registry.set_local_status_provider(None)
    gpu_registry._pending.clear()


# ------------------------------------------------------------------ #
#  The OS socket table                                               #
# ------------------------------------------------------------------ #

_PROC_TCP = """\
  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode
   0: 0100007F:21D2 00000000:0000 0A 00000000:00000000 00:00000000 00000000  1000        0 11111 1
   1: 00000000:21D3 00000000:0000 0A 00000000:00000000 00:00000000 00000000  1000        0 22222 1
   2: 0100007F:9C40 0100007F:21D2 01 00000000:00000000 00:00000000 00000000  1000        0 33333 1
"""

_PROC_TCP6 = """\
  sl  local_address                         rem_address                            st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode
   0: 00000000000000000000000001000000:21D4 00000000000000000000000000000000:0000 0A 00000000:00000000 00:00000000 00000000  1000        0 44444 1
   1: 00000000000000000000000000000000:21D5 00000000000000000000000000000000:0000 0A 00000000:00000000 00:00000000 00000000  1000        0 55555 1
"""

_NETSTAT = """\
Active Internet connections (including servers)
Proto Recv-Q Send-Q  Local Address          Foreign Address        (state)
tcp4       0      0  127.0.0.1.8642         *.*                    LISTEN
tcp46      0      0  *.8643                 *.*                    LISTEN
tcp4       0      0  127.0.0.1.50000        127.0.0.1.8642         ESTABLISHED
"""


class TestSocketTableParsing:
    def test_proc_net_tcp_keeps_only_listening_rows(self):
        got = listeners.parse_proc_net_tcp(_PROC_TCP, socket.AF_INET)
        assert got == [("127.0.0.1", 0x21D2), ("0.0.0.0", 0x21D3)]

    def test_proc_net_tcp6_reads_loopback_and_wildcard(self):
        got = listeners.parse_proc_net_tcp(_PROC_TCP6, socket.AF_INET6)
        assert got == [("::1", 0x21D4), ("::", 0x21D5)]

    def test_netstat_keeps_only_listening_rows(self):
        assert listeners.parse_netstat(_NETSTAT) == [("127.0.0.1", 8642), ("0.0.0.0", 8643)]

    def test_windows_ipv4_rows(self):
        # MIB_TCPROW_OWNER_PID: state, local addr, local port (network order in
        # the low 16 bits), remote addr, remote port, pid.
        def row(addr_bytes, port):
            net_port = ((port & 0xFF) << 8) | (port >> 8)
            return struct.pack("<6I", 2, struct.unpack("<I", bytes(addr_bytes))[0],
                               net_port, 0, 0, 1234)
        buf = struct.pack("<I", 2) + row([127, 0, 0, 1], 8642) + row([0, 0, 0, 0], 8700)
        got = listeners.parse_windows_table(buf, 2)
        assert got == [("127.0.0.1", 8642), ("0.0.0.0", 8700)]

    def test_windows_table_truncated_mid_row_keeps_only_complete_rows(self):
        count = struct.pack("<I", 3)
        assert listeners.parse_windows_table(count + b"\x00" * 20, 2) == []
        assert listeners.parse_windows_table(count + b"\x00" * 30, 2) == [("0.0.0.0", 0)]

    def test_an_empty_buffer_is_empty(self):
        assert listeners.parse_windows_table(b"", 2) == []


class TestRealSocketTable:
    def test_a_listening_socket_is_found_and_a_closed_port_is_not(self):
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        s.listen()
        port = s.getsockname()[1]
        try:
            found = listeners.listening_ports_in(port, port)
            if found is None:
                pytest.skip("this platform's socket table is unreadable here")
            assert found == [port]
        finally:
            s.close()
        assert listeners.listening_ports_in(port, port) == []

    def test_an_ipv6_listener_is_found(self):
        if not socket.has_ipv6:
            pytest.skip("no IPv6")
        s = socket.socket(socket.AF_INET6)
        try:
            s.bind(("::1", 0))
        except OSError:
            s.close()
            pytest.skip("cannot bind ::1")
        s.listen()
        port = s.getsockname()[1]
        try:
            found = listeners.listening_ports_in(port, port)
            if found is None:
                pytest.skip("this platform's socket table is unreadable here")
            assert found == [port]
        finally:
            s.close()

    def test_an_unsupported_platform_reports_none_not_empty(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "plan9")
        assert listeners.listening_endpoints() is None
        assert listeners.listening_ports_in(1, 65535) is None


# ------------------------------------------------------------------ #
#  Candidate ports                                                   #
# ------------------------------------------------------------------ #

class TestCandidateEndpoints:
    def test_detection_off_finds_nothing(self, monkeypatch):
        with peer_server(localm_whoami("a1")) as srv:
            enable_detection(monkeypatch, srv.port)
            monkeypatch.setenv("LOCALM_PEER_DETECTION", "off")
            assert gpu_registry.candidate_endpoints() == []

    def test_only_listeners_inside_the_claimed_range(self, monkeypatch):
        outside = None
        with peer_server(localm_whoami("a1")) as inside:
            for offset in range(2, 400):
                s = socket.socket()
                try:
                    s.bind(("127.0.0.1", inside.port + offset))
                    s.listen()
                except OSError:
                    s.close()
                    continue
                outside = s
                break
            assert outside is not None, "no free port near the test server"
            try:
                outside_port = outside.getsockname()[1]
                enable_detection(monkeypatch, inside.port)
                monkeypatch.setattr("localm.config.PORT_RANGE",
                                    (inside.port, inside.port))
                ports = [p for _a, p in gpu_registry.candidate_endpoints()]
                # The control: the same listener IS found once the range covers it.
                monkeypatch.setattr("localm.config.PORT_RANGE",
                                    (inside.port, outside_port))
                widened = [p for _a, p in gpu_registry.candidate_endpoints()]
            finally:
                outside.close()
        assert ports == [inside.port]
        assert outside_port in widened

    def test_a_wildcard_listener_is_dialled_on_loopback(self, monkeypatch):
        monkeypatch.setenv("LOCALM_PEER_DETECTION", "on")
        monkeypatch.setattr("localm.config.PORT_RANGE", (8642, 8642))
        monkeypatch.setattr(listeners, "listening_endpoints",
                            lambda: [("0.0.0.0", 8642), ("::", 8642)])
        assert gpu_registry.candidate_endpoints() == [("127.0.0.1", 8642)]

    def test_a_non_loopback_only_listener_is_skipped(self, monkeypatch):
        monkeypatch.setenv("LOCALM_PEER_DETECTION", "on")
        monkeypatch.setattr("localm.config.PORT_RANGE", (8642, 8642))
        monkeypatch.setattr(listeners, "listening_endpoints",
                            lambda: [("192.168.1.20", 8642)])
        assert gpu_registry.candidate_endpoints() == []

    def test_an_unreadable_socket_table_falls_back_to_probing_the_range(
            self, monkeypatch):
        with peer_server(localm_whoami("a1")) as srv:
            enable_detection(monkeypatch, srv.port)
            monkeypatch.setattr(listeners, "listening_endpoints", lambda: None)
            assert gpu_registry.candidate_endpoints() == [("127.0.0.1", srv.port)]


# ------------------------------------------------------------------ #
#  list_gpu_peers                                                    #
# ------------------------------------------------------------------ #

class TestListGpuPeers:
    def test_a_running_instance_is_found_with_what_it_holds(self, monkeypatch):
        who = localm_whoami("peer-1", root_dir="/proj/x", version="9.9.9")
        status = default_status(who, model="big", vram_estimate_bytes=123, gpu_index=1,
                                models=[{"name": "big"}])
        with peer_server(who, status=status) as srv:
            enable_detection(monkeypatch, srv.port)

            peers = gpu_registry.list_gpu_peers()

        assert len(peers) == 1
        p = peers[0]
        assert (p["instance_id"], p["model"], p["vram_estimate_bytes"], p["gpu_index"]) == \
            ("peer-1", "big", 123, 1)
        assert (p["host"], p["scheme"], p["port"]) == ("127.0.0.1", "http", srv.port)
        assert (p["root_dir"], p["version"], p["mode"]) == ("/proj/x", "9.9.9", "full")
        assert p["pid"] == FOREIGN_PID

    def test_the_requesting_instance_is_excluded_by_id(self, monkeypatch):
        with peer_server(localm_whoami("me")) as srv:
            enable_detection(monkeypatch, srv.port)
            assert gpu_registry.list_gpu_peers(exclude_self_id="me") == []
            assert len(gpu_registry.list_gpu_peers()) == 1

    def test_this_process_is_excluded_by_pid_even_without_an_id(self, monkeypatch):
        who = localm_whoami("looks-like-a-peer")
        with peer_server(who, status=default_status(who, pid=os.getpid())) as srv:
            enable_detection(monkeypatch, srv.port)
            assert gpu_registry.list_gpu_peers() == []

    def test_a_listener_that_is_not_localm_is_not_a_peer(self, monkeypatch):
        with peer_server({"app": "something-else", "instance_id": "x"}) as srv:
            enable_detection(monkeypatch, srv.port)
            assert gpu_registry.list_gpu_peers() == []

    def test_an_instance_serving_no_status_is_not_a_peer(self, monkeypatch):
        with peer_server(localm_whoami("iso"), status=None) as srv:
            enable_detection(monkeypatch, srv.port)
            assert gpu_registry.list_gpu_peers() == []

    def test_whoami_and_status_must_name_the_same_instance(self, monkeypatch):
        with peer_server(localm_whoami("one"),
                         status={"instance_id": "two", "pid": 1}) as srv:
            enable_detection(monkeypatch, srv.port)
            assert gpu_registry.list_gpu_peers() == []

    def test_a_stopped_instance_is_gone(self, monkeypatch):
        with peer_server(localm_whoami("p")) as srv:
            port = srv.port
            enable_detection(monkeypatch, port)
            assert len(gpu_registry.list_gpu_peers()) == 1
        assert gpu_registry.list_gpu_peers() == []

    def test_several_peers_come_back_sorted_by_instance_id(self, monkeypatch):
        with peer_server(localm_whoami("bbb")) as b, peer_server(localm_whoami("aaa")) as a:
            enable_detection(monkeypatch, a.port, b.port)
            ids = [p["instance_id"] for p in gpu_registry.list_gpu_peers()]
        assert ids == ["aaa", "bbb"]

    def test_a_failing_probe_does_not_hide_the_other_peers(self, monkeypatch):
        with peer_server(localm_whoami("good")) as good, \
                peer_server(localm_whoami("bad")) as bad:
            enable_detection(monkeypatch, good.port, bad.port)
            real = gpu_registry.fetch_status

            def flaky(scheme, port, timeout, dial="127.0.0.1"):
                if port == bad.port:
                    raise RuntimeError("probe exploded")
                return real(scheme, port, timeout, dial)

            monkeypatch.setattr(gpu_registry, "fetch_status", flaky)
            ids = [p["instance_id"] for p in gpu_registry.list_gpu_peers()]
        assert ids == ["good"]

    def test_an_unreadable_candidate_list_is_not_an_error(self, monkeypatch):
        def boom():
            raise RuntimeError("socket table exploded")
        monkeypatch.setattr(gpu_registry, "candidate_endpoints", boom)
        assert gpu_registry.list_gpu_peers() == []


# ------------------------------------------------------------------ #
#  Asking a peer to release its VRAM                                 #
# ------------------------------------------------------------------ #

def _me(monkeypatch, instance_id="me-1", port=18642):
    gpu_registry.set_local_status_provider(
        lambda: {"instance_id": instance_id, "port": port, "scheme": "http"})


def _peer_dict(srv, instance_id="peer-1", **kw):
    return {"instance_id": instance_id, "port": srv.port, "host": "127.0.0.1",
            "scheme": "http", **kw}


class TestRequestCooperativeUnload:
    def test_a_vouched_request_succeeds_and_names_this_instance(self, monkeypatch):
        _me(monkeypatch)
        seen = {}

        def unload(body):
            rid = body.get("request_id")
            seen["body"] = body
            # What the real receiver does: ask the requester to vouch for it.
            seen["vouched"] = gpu_registry.vouch_for(rid, "peer-1")
            return 200, {"status": "unloaded"}

        with peer_server(localm_whoami("peer-1"),
                         post={"/v1/instances/cooperate-unload": unload}) as srv:
            assert gpu_registry.request_cooperative_unload(_peer_dict(srv)) is True

        assert seen["vouched"] is True
        assert seen["body"]["requester"] == {"instance_id": "me-1", "port": 18642,
                                              "scheme": "http"}
        assert len(seen["body"]["request_id"]) >= 24

    def test_the_request_id_is_forgotten_once_the_call_returns(self, monkeypatch):
        _me(monkeypatch)
        ids = []

        def unload(body):
            ids.append(body["request_id"])
            return 200, {"status": "already_unloaded"}

        with peer_server(localm_whoami("peer-1"),
                         post={"/v1/instances/cooperate-unload": unload}) as srv:
            assert gpu_registry.request_cooperative_unload(_peer_dict(srv)) is True
        assert gpu_registry.vouch_for(ids[0], "peer-1") is False

    def test_no_own_status_sends_nothing(self, monkeypatch):
        with peer_server(localm_whoami("peer-1")) as srv:
            assert gpu_registry.request_cooperative_unload(_peer_dict(srv)) is False
            assert srv.requests == []

    def test_a_non_ok_answer_is_false(self, monkeypatch):
        _me(monkeypatch)
        with peer_server(localm_whoami("peer-1"),
                         post={"/v1/instances/cooperate-unload":
                               lambda b: (403, {"detail": "no"})}) as srv:
            assert gpu_registry.request_cooperative_unload(_peer_dict(srv)) is False

    def test_an_unexpected_status_value_is_false(self, monkeypatch):
        _me(monkeypatch)
        with peer_server(localm_whoami("peer-1"),
                         post={"/v1/instances/cooperate-unload":
                               lambda b: (200, {"status": "in_use"})}) as srv:
            assert gpu_registry.request_cooperative_unload(_peer_dict(srv)) is False

    def test_a_dead_peer_is_false(self, monkeypatch):
        _me(monkeypatch)
        with peer_server(localm_whoami("peer-1")) as srv:
            peer = _peer_dict(srv)
        assert gpu_registry.request_cooperative_unload(peer, timeout=1.0) is False

    def test_missing_port_or_id_is_false(self, monkeypatch):
        _me(monkeypatch)
        assert gpu_registry.request_cooperative_unload({}) is False
        assert gpu_registry.request_cooperative_unload({"port": 1}) is False
        assert gpu_registry.request_cooperative_unload({"instance_id": "x"}) is False

    def test_a_non_loopback_host_is_refused_without_sending(self, monkeypatch):
        _me(monkeypatch)
        import requests
        sent = []
        monkeypatch.setattr(requests, "post", lambda *a, **k: sent.append(a) or None)
        peer = {"instance_id": "peer-1", "port": 8642, "host": "203.0.113.5",
                "scheme": "http"}
        assert gpu_registry.request_cooperative_unload(peer) is False
        assert sent == []

    def test_a_scheme_smuggling_an_authority_is_refused_without_sending(self, monkeypatch):
        _me(monkeypatch)
        import requests
        sent = []
        monkeypatch.setattr(requests, "post", lambda *a, **k: sent.append(a) or None)
        peer = {"instance_id": "peer-1", "port": 8642, "host": "127.0.0.1",
                "scheme": "http://attacker.example/x?"}
        assert gpu_registry.request_cooperative_unload(peer) is False
        assert sent == []


class TestVouching:
    def test_a_pending_request_is_confirmed_once_for_the_right_asker(self):
        gpu_registry._remember_request("rid-1", "peer-1")
        assert gpu_registry.vouch_for("rid-1", "peer-1") is True
        assert gpu_registry.vouch_for("rid-1", "peer-1") is False

    def test_the_wrong_asker_gets_no_confirmation_and_the_request_survives(self):
        gpu_registry._remember_request("rid-2", "peer-1")
        assert gpu_registry.vouch_for("rid-2", "someone-else") is False
        assert gpu_registry.vouch_for("rid-2", "peer-1") is True

    def test_an_unknown_id_gets_no_confirmation(self):
        assert gpu_registry.vouch_for("never-sent", "peer-1") is False

    def test_an_expired_request_gets_no_confirmation(self, monkeypatch):
        gpu_registry._remember_request("rid-3", "peer-1")
        real = time.monotonic
        monkeypatch.setattr(time, "monotonic",
                            lambda: real() + gpu_registry.REQUEST_TTL_S + 1)
        assert gpu_registry.vouch_for("rid-3", "peer-1") is False

    @pytest.mark.parametrize("rid, asker", [(None, "p"), ("r", None), (1, 2), ({}, [])])
    def test_malformed_inputs_get_no_confirmation(self, rid, asker):
        assert gpu_registry.vouch_for(rid, asker) is False


class TestVerifyRequester:
    """The receiving side: it calls the requester back before acting."""

    def _requester(self, srv, instance_id="req-1"):
        return {"instance_id": instance_id, "port": srv.port, "scheme": "http"}

    def test_a_requester_that_vouches_is_verified_and_sees_the_request_id(self):
        calls = []

        def vouch(body):
            calls.append(body)
            return 200, {"vouched": True}

        with peer_server(localm_whoami("req-1"),
                         post={"/v1/instances/vouch": vouch}) as srv:
            ok = gpu_registry.verify_requester(self._requester(srv), "rid-9", "me-2")

        assert ok is True
        assert calls == [{"request_id": "rid-9", "asker_instance_id": "me-2"}]

    def test_a_requester_that_does_not_vouch_is_refused(self):
        with peer_server(localm_whoami("req-1"),
                         post={"/v1/instances/vouch":
                               lambda b: (403, {"detail": "no"})}) as srv:
            assert gpu_registry.verify_requester(self._requester(srv), "r", "me") is False

    def test_a_vouched_false_body_is_refused(self):
        with peer_server(localm_whoami("req-1"),
                         post={"/v1/instances/vouch":
                               lambda b: (200, {"vouched": False})}) as srv:
            assert gpu_registry.verify_requester(self._requester(srv), "r", "me") is False

    def test_a_listener_that_is_not_the_named_instance_is_refused_before_any_post(self):
        with peer_server(localm_whoami("someone-else"),
                         post={"/v1/instances/vouch":
                               lambda b: (200, {"vouched": True})}) as srv:
            assert gpu_registry.verify_requester(self._requester(srv), "r", "me") is False
            assert all(m == "GET" for m, _p, _b in srv.requests)

    def test_naming_this_instance_as_the_requester_is_refused(self):
        with peer_server(localm_whoami("me")) as srv:
            assert gpu_registry.verify_requester(
                self._requester(srv, "me"), "r", "me") is False
            assert srv.requests == []

    @pytest.mark.parametrize("requester", [
        None, "text", {}, {"instance_id": "x"}, {"instance_id": "x", "port": "8642"},
        {"instance_id": "x", "port": True}, {"instance_id": "x", "port": 0},
        {"instance_id": "x", "port": 70000}, {"instance_id": "", "port": 8642},
        {"instance_id": "x", "port": 8642, "scheme": "ftp"},
        {"instance_id": "x", "port": 8642, "scheme": "http://evil/"},
    ])
    def test_a_malformed_requester_is_refused_without_dialling(self, requester,
                                                               monkeypatch):
        import requests
        sent = []
        monkeypatch.setattr(requests, "get", lambda *a, **k: sent.append(a))
        monkeypatch.setattr(requests, "post", lambda *a, **k: sent.append(a))
        assert gpu_registry.verify_requester(requester, "rid", "me") is False
        assert sent == []

    @pytest.mark.parametrize("rid", [None, "", 5])
    def test_a_missing_request_id_is_refused_without_dialling(self, rid):
        with peer_server(localm_whoami("req-1")) as srv:
            assert gpu_registry.verify_requester(self._requester(srv), rid, "me") is False
            assert srv.requests == []


# ------------------------------------------------------------------ #
#  The three routes                                                  #
# ------------------------------------------------------------------ #

def _coordinating_app(bind_host="127.0.0.1"):
    app = create_app(None)
    app.state.bind_host = bind_host
    hs._gpu_coord = {"instance_id": "self-1", "port": 8700, "host": bind_host,
                     "scheme": "http"}
    return app


class TestStatusRoute:
    def test_reports_this_instances_live_status(self):
        client = TestClient(_coordinating_app())
        r = client.get("/v1/instances/status")
        assert r.status_code == 200
        body = r.json()
        assert body["instance_id"] == "self-1"
        assert body["pid"] == os.getpid()
        assert body["port"] == 8700 and body["scheme"] == "http"
        assert set(body) >= {"model", "models", "vram_estimate_bytes", "gpu_index"}

    def test_a_non_coordinating_instance_does_not_have_it(self):
        assert TestClient(create_app(None)).get("/v1/instances/status").status_code == 404

    def test_a_network_bound_instance_does_not_have_it(self):
        client = TestClient(_coordinating_app(bind_host="0.0.0.0"))
        assert client.get("/v1/instances/status").status_code == 404


class TestVouchRoute:
    def test_confirms_a_request_this_instance_sent_exactly_once(self):
        client = TestClient(_coordinating_app())
        gpu_registry._remember_request("rid-a", "peer-9")
        body = {"request_id": "rid-a", "asker_instance_id": "peer-9"}
        first = client.post("/v1/instances/vouch", json=body)
        second = client.post("/v1/instances/vouch", json=body)
        assert (first.status_code, first.json()) == (200, {"vouched": True})
        assert second.status_code == 403

    def test_refuses_a_request_it_never_sent(self):
        client = TestClient(_coordinating_app())
        r = client.post("/v1/instances/vouch",
                        json={"request_id": "forged", "asker_instance_id": "peer-9"})
        assert r.status_code == 403

    def test_refuses_garbage_bodies(self):
        client = TestClient(_coordinating_app())
        assert client.post("/v1/instances/vouch", content=b"not json").status_code == 403
        assert client.post("/v1/instances/vouch", json=[1, 2]).status_code == 403

    def test_a_non_coordinating_instance_refuses(self):
        client = TestClient(create_app(None))
        gpu_registry._remember_request("rid-b", "peer-9")
        r = client.post("/v1/instances/vouch",
                        json={"request_id": "rid-b", "asker_instance_id": "peer-9"})
        assert r.status_code == 403


class TestCooperateUnloadRoute:
    def _spy_unload(self, monkeypatch):
        calls = []

        async def unload_all_models():
            calls.append(1)
            return {"status": "unloaded"}

        monkeypatch.setattr(hs, "unload_all_models", unload_all_models)
        return calls

    def test_unloads_once_the_requester_is_verified(self, monkeypatch):
        calls = self._spy_unload(monkeypatch)
        seen = {}

        def fake_verify(requester, request_id, self_id, **k):
            seen.update(requester=requester, request_id=request_id, self_id=self_id)
            return True

        monkeypatch.setattr(gpu_registry, "verify_requester", fake_verify)
        client = TestClient(_coordinating_app())
        r = client.post("/v1/instances/cooperate-unload",
                        json={"requester": {"instance_id": "req", "port": 1},
                              "request_id": "rid"})
        assert (r.status_code, r.json()) == (200, {"status": "unloaded"})
        assert calls == [1]
        assert seen == {"requester": {"instance_id": "req", "port": 1},
                        "request_id": "rid", "self_id": "self-1"}

    def test_does_not_unload_when_the_requester_is_not_verified(self, monkeypatch):
        calls = self._spy_unload(monkeypatch)
        monkeypatch.setattr(gpu_registry, "verify_requester", lambda *a, **k: False)
        client = TestClient(_coordinating_app())
        r = client.post("/v1/instances/cooperate-unload",
                        json={"requester": {"instance_id": "req", "port": 1},
                              "request_id": "rid"})
        assert r.status_code == 403
        assert calls == []

    def test_a_real_api_key_alone_does_not_grant_it(self, monkeypatch):
        calls = self._spy_unload(monkeypatch)
        client = TestClient(_coordinating_app())
        r = client.post("/v1/instances/cooperate-unload",
                        headers={"Authorization": "Bearer anything"}, json={})
        assert r.status_code == 403
        assert calls == []

    def test_a_non_coordinating_instance_refuses_even_a_verified_request(
            self, monkeypatch):
        calls = self._spy_unload(monkeypatch)
        monkeypatch.setattr(gpu_registry, "verify_requester", lambda *a, **k: True)
        client = TestClient(create_app(None))
        r = client.post("/v1/instances/cooperate-unload",
                        json={"requester": {"instance_id": "req", "port": 1},
                              "request_id": "rid"})
        assert r.status_code == 403
        assert calls == []

    def test_a_network_bound_instance_refuses_even_a_verified_request(self, monkeypatch):
        calls = self._spy_unload(monkeypatch)
        monkeypatch.setattr(gpu_registry, "verify_requester", lambda *a, **k: True)
        client = TestClient(_coordinating_app(bind_host="0.0.0.0"))
        r = client.post("/v1/instances/cooperate-unload",
                        json={"requester": {"instance_id": "req", "port": 1},
                              "request_id": "rid"})
        assert r.status_code == 403
        assert calls == []

    def test_refusals_look_identical_whatever_the_reason(self):
        a = TestClient(create_app(None)).post("/v1/instances/cooperate-unload", json={})
        b = TestClient(_coordinating_app()).post("/v1/instances/cooperate-unload", json={})
        assert (a.status_code, a.json()) == (b.status_code, b.json())


# ------------------------------------------------------------------ #
#  Lifespan: publish live status, write nothing                       #
# ------------------------------------------------------------------ #

class TestLifespanPublishesLiveStatus:
    def _advertised_app(self, *, isolated=False):
        app = create_app(None)
        app.state.instance_id = "iid-gpu-lifespan"
        app.state.instance_port = 18642
        app.state.instance_scheme = "http"
        app.state.bind_host = "127.0.0.1"
        app.state.instance_isolated = isolated
        return app

    def test_a_real_instance_publishes_status_while_running_and_stops_after(self):
        app = self._advertised_app()
        with TestClient(app):
            status = gpu_registry.own_status()
            assert status is not None
            assert status["instance_id"] == "iid-gpu-lifespan"
            assert status["port"] == 18642
            assert hs._gpu_coord is not None
        assert gpu_registry.own_status() is None
        assert hs._gpu_coord is None

    def test_an_isolated_instance_publishes_nothing(self):
        with TestClient(self._advertised_app(isolated=True)):
            assert gpu_registry.own_status() is None
            assert hs._gpu_coord is None

    def test_a_plain_app_publishes_nothing(self):
        with TestClient(create_app(None)):
            assert gpu_registry.own_status() is None
            assert hs._gpu_coord is None

    def test_starting_and_stopping_creates_no_registry_file(self, tmp_path):
        from localm.config import home_dir
        before = {p for p in home_dir().rglob("*")}
        with TestClient(self._advertised_app()):
            gpu_registry.list_gpu_peers()
        after = {p for p in home_dir().rglob("*")}
        assert not [p for p in after - before if "gpu" in p.parts or p.suffix == ".json"
                    and p.parent.name == "gpu"], after - before


class TestThereIsNoRegistryFile:
    """gpu_registry coordinates by talking to running instances. These pin that
    no file machinery crept back in."""

    def test_the_module_has_no_file_or_directory_writes(self):
        tree = ast.parse(inspect.getsource(gpu_registry))
        offenders = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            f = node.func
            name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
            if name in {"open", "write_text", "write_bytes", "mkdir", "unlink", "replace",
                        "rename", "makedirs", "mkstemp", "gettempdir", "chmod"}:
                offenders.append((name, node.lineno))
        assert offenders == []

    def test_the_old_file_registry_api_is_gone(self):
        for name in ("registry_dir", "write_entry", "remove_entry", "list_entries",
                     "reap_stale", "entry_path", "own_entry", "new_coordination_token"):
            assert not hasattr(gpu_registry, name), name


class TestEndToEndOverRealSockets:
    """Two real HTTP servers: the receiver is this process's real app, the
    requester is a fake instance that answers the vouch callback."""

    def test_a_real_receiver_unloads_after_calling_the_requester_back(self, monkeypatch):
        import threading
        import uvicorn

        calls = []

        async def unload_all_models():
            calls.append(1)
            return {"status": "unloaded"}

        monkeypatch.setattr(hs, "unload_all_models", unload_all_models)
        app = _coordinating_app()
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()
        server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port,
                                               log_level="error"))
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        try:
            deadline = time.time() + 20
            while not server.started and time.time() < deadline:
                time.sleep(0.05)
            assert server.started

            # The requester: a fake instance whose vouch endpoint defers to the
            # real pending-request table, exactly as the real route does.
            def vouch(body):
                ok = gpu_registry.vouch_for(body.get("request_id"),
                                            body.get("asker_instance_id"))
                return (200, {"vouched": True}) if ok else (403, {"detail": "no"})

            with peer_server(localm_whoami("requester-1"),
                             post={"/v1/instances/vouch": vouch}) as req:
                gpu_registry.set_local_status_provider(
                    lambda: {"instance_id": "requester-1", "port": req.port,
                             "scheme": "http"})
                peer = {"instance_id": "self-1", "port": port, "host": "127.0.0.1",
                        "scheme": "http"}
                assert gpu_registry.request_cooperative_unload(peer, timeout=10) is True
                assert calls == [1]

                # A caller that never received a request id is refused.
                import requests
                r = requests.post(
                    f"http://127.0.0.1:{port}/v1/instances/cooperate-unload",
                    json={"requester": {"instance_id": "requester-1", "port": req.port,
                                        "scheme": "http"},
                          "request_id": "guessed"}, timeout=10)
                assert r.status_code == 403
                assert calls == [1]
        finally:
            server.should_exit = True
            thread.join(timeout=20)
