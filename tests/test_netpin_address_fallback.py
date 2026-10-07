# SPDX-License-Identifier: AGPL-3.0-or-later
"""The pinned transport dials every validated address of a host in turn.

``netpolicy._resolve_pinned_all`` resolves a host once and returns the
validated addresses in dial order (families interleaved, capped, blocked ones
excluded). ``netpin`` dials them in order, moving on only when a TCP connect
fails, and never looks the host up again.
"""

import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
import requests
import urllib3.connection

from localm import netpin, netpolicy
from localm.netpolicy import NetworkPolicyError


@pytest.fixture(autouse=True)
def _allow(monkeypatch):
    monkeypatch.setenv("LOCALM_NET_MODE", "allow")
    monkeypatch.setattr("localm.config.load_config", lambda: {})


def _infos(*ips):
    """A ``getaddrinfo`` double: every host resolves to *ips*. A literal IP
    resolves to itself (urllib3 calls getaddrinfo on the pinned literal before
    connecting) with the requested port."""
    def fake(host, port=None, *a, **k):
        port = port if isinstance(port, int) else 0
        targets = [host] if host in ips else ips
        out = []
        for ip in targets:
            fam = socket.AF_INET6 if ":" in ip else socket.AF_INET
            out.append((fam, socket.SOCK_STREAM, 6, "", (ip, port)))
        return out
    return fake


# --------------------------------------------------------------------------- #
#  _resolve_pinned_all: the validated dial list                                #
# --------------------------------------------------------------------------- #

class TestResolvePinnedAll:
    def test_families_interleaved_starting_with_the_first(self, monkeypatch):
        monkeypatch.setattr("socket.getaddrinfo", _infos(
            "2606:50c0:8000::154", "2606:50c0:8001::154", "2606:50c0:8002::154",
            "185.199.108.133", "185.199.109.133"))
        assert netpolicy._resolve_pinned_all("raw.example") == [
            "2606:50c0:8000::154", "185.199.108.133",
            "2606:50c0:8001::154", "185.199.109.133"]

    def test_duplicates_collapsed_and_capped(self, monkeypatch):
        monkeypatch.setattr("socket.getaddrinfo", _infos(
            "93.184.216.34", "93.184.216.34", "93.184.216.35", "93.184.216.36",
            "93.184.216.37", "93.184.216.38", "93.184.216.39"))
        got = netpolicy._resolve_pinned_all("many.example")
        assert got == ["93.184.216.34", "93.184.216.35", "93.184.216.36",
                       "93.184.216.37"]
        assert len(got) == netpolicy._MAX_PINNED_ADDRESSES

    @pytest.mark.parametrize("blocked", [
        pytest.param("127.0.0.1", id="loopback"),
        pytest.param("169.254.169.254", id="cloud-metadata"),
        pytest.param("10.0.0.5", id="rfc1918"),
        pytest.param("::1", id="ipv6-loopback"),
    ])
    def test_blocked_address_never_in_the_dial_list(self, blocked, monkeypatch):
        monkeypatch.setattr("socket.getaddrinfo",
                            _infos("93.184.216.34", blocked, "93.184.216.35"))
        got = netpolicy._resolve_pinned_all("mixed.example")
        assert blocked not in got
        assert got == ["93.184.216.34", "93.184.216.35"]

    def test_all_blocked_is_refused(self, monkeypatch):
        monkeypatch.setattr("socket.getaddrinfo",
                            _infos("127.0.0.1", "169.254.169.254"))
        with pytest.raises(NetworkPolicyError, match="non-public"):
            netpolicy._resolve_pinned_all("internal.example")

    def test_private_kept_when_configured(self, monkeypatch):
        monkeypatch.setattr("localm.config.load_config",
                            lambda: {"net_allow_private": True})
        monkeypatch.setattr("socket.getaddrinfo",
                            _infos("127.0.0.1", "93.184.216.34"))
        assert netpolicy._resolve_pinned_all("lan.example") == [
            "127.0.0.1", "93.184.216.34"]

    def test_unresolvable_is_empty(self, monkeypatch):
        def boom(*a, **k):
            raise socket.gaierror("nxdomain")
        monkeypatch.setattr("socket.getaddrinfo", boom)
        assert netpolicy._resolve_pinned_all("ghost.example") == []

    def test_resolve_pinned_is_the_first_entry(self, monkeypatch):
        monkeypatch.setattr("socket.getaddrinfo", _infos(
            "10.0.0.5", "2606:50c0:8000::154", "185.199.108.133"))
        assert netpolicy._resolve_pinned("x.example") == "2606:50c0:8000::154"

    def test_session_for_pins_the_whole_list(self, monkeypatch):
        monkeypatch.setattr("socket.getaddrinfo", _infos(
            "2606:50c0:8000::154", "185.199.108.133"))
        seen = {}
        monkeypatch.setattr(netpin, "pinned_session",
                            lambda ips: seen.setdefault("ips", list(ips)))
        netpolicy._session_for("https://raw.example/x")
        assert seen["ips"] == ["2606:50c0:8000::154", "185.199.108.133"]


# --------------------------------------------------------------------------- #
#  netpin: dial in order, move on only on a TCP connect failure                #
# --------------------------------------------------------------------------- #

class _Echo(BaseHTTPRequestHandler):
    seen: dict = {}

    def do_GET(self):
        type(self).seen["host"] = self.headers.get("Host")
        body = b"fallback-ok"
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


@pytest.fixture
def echo_server():
    _Echo.seen = {}
    srv = HTTPServer(("127.0.0.1", 0), _Echo)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        yield srv.server_address[1]
    finally:
        srv.shutdown()
        srv.server_close()


@pytest.fixture
def dial_log(monkeypatch):
    """Record every (host, port) urllib3 dials. Addresses in ``refuse`` raise
    ``ConnectionRefusedError`` without touching the network; every other one is
    dialled for real."""
    real = urllib3.connection.connection.create_connection
    state = {"dialled": [], "refuse": set()}

    def recording(address, *a, **k):
        state["dialled"].append(address[0])
        if address[0] in state["refuse"]:
            raise ConnectionRefusedError(10061, "refused")
        return real(address, *a, **k)

    monkeypatch.setattr(urllib3.connection.connection, "create_connection",
                        recording)
    return state


def test_connect_failure_moves_to_the_next_validated_address(echo_server,
                                                            dial_log):
    dial_log["refuse"] = {"192.0.2.10"}
    with netpin.pinned_session(["192.0.2.10", "127.0.0.1"]) as session:
        resp = session.get(f"http://vhost.test:{echo_server}/",
                           headers={"Host": f"vhost.test:{echo_server}"},
                           timeout=5)
        body = resp.content
    assert body == b"fallback-ok"
    assert dial_log["dialled"] == ["192.0.2.10", "127.0.0.1"]
    assert _Echo.seen["host"] == f"vhost.test:{echo_server}"


def test_first_address_reachable_dials_only_the_first(echo_server, dial_log):
    with netpin.pinned_session(["127.0.0.1", "192.0.2.10"]) as session:
        resp = session.get(f"http://vhost.test:{echo_server}/",
                           headers={"Host": f"vhost.test:{echo_server}"},
                           timeout=5)
        assert resp.content == b"fallback-ok"
    assert dial_log["dialled"] == ["127.0.0.1"]


def test_every_address_failing_raises_connection_error(dial_log):
    dial_log["refuse"] = {"192.0.2.10", "192.0.2.11"}
    with netpin.pinned_session(["192.0.2.10", "192.0.2.11"]) as session:
        with pytest.raises(requests.ConnectionError) as info:
            session.get("http://vhost.test:8/", timeout=5)
    assert dial_log["dialled"] == ["192.0.2.10", "192.0.2.11"]
    assert not isinstance(info.value, requests.exceptions.SSLError)


def test_no_address_is_looked_up_again(echo_server, dial_log, monkeypatch):
    def no_dns(*a, **k):
        raise AssertionError("the pinned transport must not resolve the host")
    dial_log["refuse"] = {"192.0.2.10"}
    with netpin.pinned_session(["192.0.2.10", "127.0.0.1"]) as session:
        monkeypatch.setattr("socket.getaddrinfo", no_dns)
        monkeypatch.setattr(
            urllib3.connection.connection.socket, "getaddrinfo",
            lambda host, *a, **k: [(socket.AF_INET, socket.SOCK_STREAM, 6, "",
                                    (host, a[0] if a else 0))]
            if host in ("127.0.0.1", "192.0.2.10") else no_dns())
        resp = session.get(f"http://vhost.test:{echo_server}/",
                           headers={"Host": f"vhost.test:{echo_server}"},
                           timeout=5)
        assert resp.content == b"fallback-ok"
    assert dial_log["dialled"] == ["192.0.2.10", "127.0.0.1"]


def test_full_fetch_falls_back_across_resolved_addresses(echo_server, dial_log,
                                                         monkeypatch):
    """End to end through safe_fetch_bytes: the host resolves to an
    unreachable address first and a reachable one second; the fetch succeeds
    on the second, and the blocked address in the same answer is never dialled."""
    monkeypatch.setattr("localm.config.load_config",
                        lambda: {"net_allow_private": True})
    monkeypatch.setattr("socket.getaddrinfo",
                        _infos("192.0.2.10", "127.0.0.1"))
    dial_log["refuse"] = {"192.0.2.10"}
    final, ctype, body = netpolicy.safe_fetch_bytes(
        f"http://vhost.test:{echo_server}/")
    assert body == b"fallback-ok"
    assert dial_log["dialled"] == ["192.0.2.10", "127.0.0.1"]


def test_empty_address_list_rejected():
    with pytest.raises(ValueError):
        netpin.PinnedIPAdapter([])
