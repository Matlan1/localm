# SPDX-License-Identifier: AGPL-3.0-or-later
"""The DNS-rebinding TOCTOU is closed by pinning the validated IP.

netpolicy.check_url resolves+validates a host, but a plain requests.get
re-resolves at connect time - a TTL-0 attacker can answer public for the check
and internal for the connect. These tests pin the fix: the host is resolved
ONCE, the exact address dialled is validated, and the socket connects to that IP
while the original hostname is presented for SNI / cert / Host.
"""

import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import MagicMock, patch

import pytest
import requests

from localm import netpolicy
from localm.netpolicy import NetworkPolicyError


@pytest.fixture(autouse=True)
def _allow(monkeypatch):
    monkeypatch.setenv("LOCALM_NET_MODE", "allow")
    monkeypatch.setattr("localm.config.load_config", lambda: {})


def _dns(ip, port=0):
    return lambda host, *a, **k: [(2, 1, 6, "", (ip, port))]


# --------------------------------------------------------------------------- #
#  _resolve_pinned: resolve once, validate the exact IP                        #
# --------------------------------------------------------------------------- #

class TestResolvePinned:
    def test_public_host_returns_its_ip(self, monkeypatch):
        monkeypatch.setattr("socket.getaddrinfo", _dns("93.184.216.34"))
        assert netpolicy._resolve_pinned("example.com") == "93.184.216.34"

    def test_ip_literal_pinned_directly(self, monkeypatch):
        # No DNS needed for a literal; if getaddrinfo were called it would raise.
        monkeypatch.setattr("socket.getaddrinfo",
                            lambda *a, **k: (_ for _ in ()).throw(AssertionError("no DNS")))
        assert netpolicy._resolve_pinned("93.184.216.34") == "93.184.216.34"

    def test_unresolvable_returns_none(self, monkeypatch):
        import socket as _socket
        monkeypatch.setattr("socket.getaddrinfo",
                            lambda *a, **k: (_ for _ in ()).throw(_socket.gaierror("nope")))
        assert netpolicy._resolve_pinned("does-not-exist.example") is None

    @pytest.mark.parametrize("ip", [
        pytest.param("127.0.0.1", id="loopback"),
        pytest.param("169.254.169.254", id="cloud-metadata"),
    ])
    def test_private_resolution_refused_by_default(self, ip, monkeypatch):
        monkeypatch.setattr("socket.getaddrinfo", _dns(ip))
        with pytest.raises(NetworkPolicyError, match="non-public"):
            netpolicy._resolve_pinned("sneaky.example")

    def test_private_allowed_when_configured(self, monkeypatch):
        monkeypatch.setattr("localm.config.load_config",
                            lambda: {"net_allow_private": True})
        monkeypatch.setattr("socket.getaddrinfo", _dns("127.0.0.1"))
        assert netpolicy._resolve_pinned("localhost") == "127.0.0.1"

    def test_first_public_ip_chosen_over_a_blocked_sibling(self, monkeypatch):
        # A host advertising both a private and a public A record: pin the public
        # one (do not reject a legit host), never the private one.
        monkeypatch.setattr("socket.getaddrinfo", lambda *a, **k: [
            (2, 1, 6, "", ("10.0.0.5", 0)),
            (2, 1, 6, "", ("93.184.216.34", 0)),
        ])
        assert netpolicy._resolve_pinned("dual.example") == "93.184.216.34"


# --------------------------------------------------------------------------- #
#  Host header helper                                                          #
# --------------------------------------------------------------------------- #

class TestHostHeader:
    @pytest.mark.parametrize("url,expected", [
        ("https://example.com/x", "example.com"),          # default 443 omitted
        ("http://example.com/x", "example.com"),           # default 80 omitted
        ("https://example.com:8443/x", "example.com:8443"),  # non-default kept
        ("http://example.com:8080/x", "example.com:8080"),
    ])
    def test_host_header(self, url, expected):
        import urllib.parse
        assert netpolicy._host_header(urllib.parse.urlparse(url)) == expected


# --------------------------------------------------------------------------- #
#  The TOCTOU killer: public at check, internal at connect -> refused          #
# --------------------------------------------------------------------------- #

def test_rebind_between_check_and_connect_is_refused(monkeypatch):
    """check_url sees a public IP (passes); the pin re-resolves to loopback and
    refuses - the classic DNS-rebinding flip cannot reach an internal service."""
    calls = {"n": 0}

    def flip(host, *a, **k):
        calls["n"] += 1
        ip = "93.184.216.34" if calls["n"] == 1 else "127.0.0.1"   # public, then loopback
        return [(2, 1, 6, "", (ip, 0))]

    monkeypatch.setattr("socket.getaddrinfo", flip)
    with pytest.raises(NetworkPolicyError, match="non-public"):
        netpolicy.safe_fetch("https://rebind.example/secret")
    assert calls["n"] >= 2, "both the validation lookup AND the pin lookup must run"


def test_unresolvable_host_fails_closed(monkeypatch):
    """check_url lets an unresolvable host through (it would DNS-fail anyway), but
    the pin fails CLOSED rather than handing off to a re-resolving session - else a
    host that is NXDOMAIN at validation and a private A record at connect (TTL-0
    rebinding) would reach an internal service unpinned."""
    import socket as _socket

    monkeypatch.setattr(
        "socket.getaddrinfo",
        lambda *a, **k: (_ for _ in ()).throw(_socket.gaierror("nxdomain")))
    with pytest.raises(NetworkPolicyError, match="[Cc]ould not resolve"):
        netpolicy.safe_fetch("https://ghost.example/x")


# --------------------------------------------------------------------------- #
#  Real socket: the pin dials the IP and preserves the Host header (hermetic)  #
# --------------------------------------------------------------------------- #

def test_full_fetch_pins_socket_and_preserves_host(monkeypatch):
    seen = {}

    class _Echo(BaseHTTPRequestHandler):
        def do_GET(self):
            seen["host"] = self.headers.get("Host")
            seen["peer"] = self.client_address[0]
            body = b"pinned-ok"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), _Echo)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        monkeypatch.setattr("localm.config.load_config",
                            lambda: {"net_allow_private": True})
        # The made-up vhost "resolves" to loopback; the pin must dial 127.0.0.1
        # and still send Host: vhost.test:<port>.
        monkeypatch.setattr("socket.getaddrinfo", _dns("127.0.0.1", port))
        final, ctype, body = netpolicy.safe_fetch_bytes(f"http://vhost.test:{port}/")
    finally:
        srv.shutdown()

    assert body == b"pinned-ok"
    assert seen["peer"] == "127.0.0.1"                    # socket reached the pinned IP
    assert seen["host"] == f"vhost.test:{port}"           # Host preserved despite the pin


# --------------------------------------------------------------------------- #
#  HTTPS pool preserves SNI / cert hostname while repointing the socket        #
# --------------------------------------------------------------------------- #

def test_https_pool_preserves_sni_and_repoints_socket():
    from urllib3.connectionpool import HTTPSConnectionPool

    from localm.netpin import _PinnedHTTPSConnectionPool

    pool = _PinnedHTTPSConnectionPool("realhost.example", 443)
    pool._pinned_ip = "93.184.216.34"

    fake_conn = MagicMock()
    fake_conn.host = "realhost.example"
    fake_conn.server_hostname = None
    with patch.object(HTTPSConnectionPool, "_new_conn", return_value=fake_conn):
        conn = pool._new_conn()

    assert conn._dns_host == "93.184.216.34"              # socket dials the pinned IP
    assert conn.server_hostname == "realhost.example"     # SNI + cert keep the real host


# --------------------------------------------------------------------------- #
#  trust_env=False: the pin must not be routable AROUND (env proxy / .netrc)  #
# --------------------------------------------------------------------------- #

def test_pinned_session_disables_trust_env():
    """trust_env=True (the requests default) lets HTTP_PROXY/HTTPS_PROXY
    environment variables route a request through an operator- or
    attacker-controlled proxy INSTEAD of the pinned IP, and lets .netrc on
    disk auto-attach credentials to a request that never asked to
    authenticate. Both defeat the pin from entirely outside this module's own
    logic, so a pinned session must never consult either."""
    from localm.netpin import pinned_session

    assert pinned_session("93.184.216.34").trust_env is False


def test_pinned_session_ignores_an_environment_proxy(monkeypatch):
    """When a proxy is selected for a plain-HTTP request,
    requests.adapters.HTTPAdapter routes it through a DIFFERENT connection pool
    (proxy_manager_for's own plain urllib3.ProxyManager) that never sees
    _pinned_ip at all, so with trust_env=True an HTTP_PROXY env var bypasses the
    pin completely. A "trap" server stands in for the env proxy and a "real"
    server for the pinned target; the trap must never be hit."""
    trap_hits = {"n": 0}
    real_hits = {"n": 0}

    class _Trap(BaseHTTPRequestHandler):
        def do_GET(self):
            trap_hits["n"] += 1
            self.send_response(200)
            self.end_headers()

        def log_message(self, *a):
            pass

    class _Real(BaseHTTPRequestHandler):
        def do_GET(self):
            real_hits["n"] += 1
            body = b"real-ok"
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    trap = HTTPServer(("127.0.0.1", 0), _Trap)
    real = HTTPServer(("127.0.0.1", 0), _Real)
    trap_port = trap.server_address[1]
    real_port = real.server_address[1]
    threading.Thread(target=trap.serve_forever, daemon=True).start()
    threading.Thread(target=real.serve_forever, daemon=True).start()
    try:
        monkeypatch.setenv("HTTP_PROXY", f"http://127.0.0.1:{trap_port}")
        monkeypatch.setenv("http_proxy", f"http://127.0.0.1:{trap_port}")

        from localm.netpin import pinned_session
        session = pinned_session("127.0.0.1")
        resp = session.get(f"http://vhost.test:{real_port}/",
                           headers={"Host": f"vhost.test:{real_port}"})
        body = resp.content
        resp.close()
        session.close()
    finally:
        trap.shutdown()
        real.shutdown()

    assert trap_hits["n"] == 0, (
        "the environment HTTP_PROXY was consulted and the request was routed "
        "through the proxy instead of the pinned IP - the SSRF pin is "
        "bypassable via the environment")
    assert real_hits["n"] == 1, "the pinned request never reached the real target"
    assert body == b"real-ok"


def test_pinned_session_ignores_netrc_credentials(monkeypatch, tmp_path):
    """trust_env=True (the requests default) auto-attaches .netrc credentials
    to any request whose host has a matching entry, even though the caller
    never asked to authenticate - a pinned session must never do this."""
    netrc_path = tmp_path / "netrc"
    netrc_path.write_text(
        "machine vhost.test login leaked-user password leaked-pass\n")
    monkeypatch.setenv("NETRC", str(netrc_path))

    from localm.netpin import pinned_session
    session = pinned_session("127.0.0.1")
    prepared = session.prepare_request(requests.Request("GET", "http://vhost.test/"))
    assert "Authorization" not in prepared.headers, (
        "a .netrc entry was auto-attached to a pinned request as Basic auth - "
        "trust_env must be False")
