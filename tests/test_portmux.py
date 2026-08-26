# SPDX-License-Identifier: AGPL-3.0-or-later
"""A TLS bind must catch a plain-http request on the same port with an https
redirect (not a bare connection reset), while still serving HTTPS normally.

Spins up localm.portmux in a real subprocess (so the asyncio demux + internal
uvicorn run exactly as in production) and checks both schemes on the one port.
"""
import http.client
import socket
import ssl
import subprocess
import sys
import time
from pathlib import Path

import pytest

from localm import tls

HERE = Path(__file__).resolve().parent
WORKER = HERE / "_portmux_server.py"


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _wait_listening(port: int, timeout: float = 20.0) -> None:
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return
        except OSError as e:   # not up yet
            last = e
            time.sleep(0.1)
    raise AssertionError(f"portmux server never listened on {port}: {last}")


@pytest.fixture
def tls_server(tmp_path):
    cert, key = tls.ensure_cert(tmp_path, hostnames=["127.0.0.1"])
    ca = str(tls.ca_cert_path(tmp_path))
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, str(WORKER), str(port), cert, key],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    try:
        _wait_listening(port)
        yield port, ca
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


@pytest.fixture
def plain_server():
    """A PLAIN-HTTP portmux bind (no cert) - the loopback default. R45: it must
    front uvicorn with the first-byte peek so a TLS-on-the-HTTP-port connection is
    handled at the socket layer instead of flooding uvicorn's HTTP parser."""
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, str(WORKER), str(port)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    try:
        _wait_listening(port)
        yield port
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def test_plain_http_request_is_served(plain_server):
    """A normal HTTP request on the plain bind is relayed to the internal uvicorn
    and answered (the peek layer is transparent to real HTTP)."""
    port = plain_server
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=15)
    conn.request("GET", "/")
    resp = conn.getresponse()
    body = resp.read().decode("utf-8", "replace")
    conn.close()
    assert resp.status == 200
    assert body == "hello-localm"


def test_tls_on_plain_port_is_handled_and_server_survives(plain_server):
    """R45 root cause: a client that opens TLS/HTTPS on the plain-HTTP port (a
    browser with HSTS / HTTPS-First) must be handled at the socket layer, NOT fed
    into uvicorn's HTTP parser. The handshake fails cleanly (the port speaks HTTP)
    and - crucially - a normal request right after still works, proving the server
    neither crashed nor wedged and there was no 'Invalid HTTP request' to log."""
    port = plain_server
    raw = socket.create_connection(("127.0.0.1", port), timeout=15)
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    with pytest.raises((ssl.SSLError, OSError, ConnectionError)):
        tls_sock = ctx.wrap_socket(raw, server_hostname="127.0.0.1")
        tls_sock.do_handshake()
    try:
        raw.close()
    except OSError:
        pass

    # The server is unharmed: a normal HTTP request still succeeds.
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=15)
    conn.request("GET", "/after-tls")
    resp = conn.getresponse()
    body = resp.read().decode("utf-8", "replace")
    conn.close()
    assert resp.status == 200
    assert body == "hello-localm"


def test_plain_http_gets_308_to_https(tls_server):
    port, _ca = tls_server
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=15)
    conn.request("GET", "/some/path")
    resp = conn.getresponse()
    body = resp.read().decode("utf-8", "replace")
    conn.close()
    assert resp.status == 308
    loc = resp.getheader("Location")
    assert loc and loc.startswith("https://")
    assert loc.endswith(":%d/some/path" % port)   # same port, path preserved
    assert "secure connection" in body            # catch-page fallback body


def test_https_still_serves_the_app(tls_server):
    port, ca = tls_server
    ctx = ssl.create_default_context(cafile=ca)
    conn = http.client.HTTPSConnection("127.0.0.1", port, context=ctx, timeout=15)
    conn.request("GET", "/")
    resp = conn.getresponse()
    body = resp.read().decode("utf-8", "replace")
    conn.close()
    assert resp.status == 200
    assert body == "hello-localm"


def test_http_host_without_port_redirect_keeps_the_listening_port(tls_server):
    # A (non-compliant) client that sends a bare host with no port must still be
    # redirected to the actual listening port, not the https default :443.
    port, _ca = tls_server
    raw = socket.create_connection(("127.0.0.1", port), timeout=15)
    raw.sendall(b"GET /x HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n")
    data = b""
    raw.settimeout(15)
    try:
        while b"\r\n\r\n" not in data:
            chunk = raw.recv(4096)
            if not chunk:
                break
            data += chunk
    finally:
        raw.close()
    text = data.decode("latin-1", "replace")
    assert "308" in text.split("\r\n")[0]
    assert ("Location: https://127.0.0.1:%d/x" % port) in text


def test_http_without_host_header_explains_instead_of_resetting(tls_server):
    # HTTP/1.0 with no Host header: we cannot build a redirect target, but the
    # user must still get a helpful page, never a connection reset.
    port, _ca = tls_server
    raw = socket.create_connection(("127.0.0.1", port), timeout=15)
    raw.sendall(b"GET / HTTP/1.0\r\n\r\n")
    data = b""
    raw.settimeout(15)
    try:
        while b"\r\n\r\n" not in data:
            chunk = raw.recv(4096)
            if not chunk:
                break
            data += chunk
    finally:
        raw.close()
    text = data.decode("latin-1", "replace")
    assert text.startswith("HTTP/1.1 ")
    assert "secure connection" in text
