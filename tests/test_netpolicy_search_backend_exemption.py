# SPDX-License-Identifier: AGPL-3.0-or-later
"""A SearXNG instance on a LAN or loopback address is reachable from the search
provider without ``net_allow_private``, and only from it: the exemption covers
the configured ``net_search_url`` origin (scheme, host and port), never a
link-local address, and never a model-driven fetch of the same URL."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from localm import netpolicy
from localm.config import update_config
from localm.netpolicy import NetworkPolicyError, check_url


def _configure(**values):
    def _set(c):
        c["net_mode"] = "allow"
        c["net_allow_private"] = False
        c.update(values)
    update_config(_set)


@pytest.fixture
def searx_server():
    hits = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            hits.append(self.path)
            body = json.dumps({"results": [
                {"title": "Hit", "url": "https://example.org/a", "content": "snippet"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", hits
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_search_provider_reaches_a_loopback_searxng_without_net_allow_private(searx_server):
    from localm.web_retrieval.providers import SearXNGProvider
    base, hits = searx_server
    _configure(net_search_url=base)
    results = SearXNGProvider(base).search("hello", 3)
    assert [r.url for r in results] == ["https://example.org/a"]
    assert hits and hits[0].startswith("/search?")


def test_a_model_fetch_of_the_same_origin_is_still_refused(searx_server):
    base, hits = searx_server
    _configure(net_search_url=base)
    with pytest.raises(NetworkPolicyError, match="non-public"):
        netpolicy.safe_fetch_bytes(base + "/search?q=x")
    assert hits == []


@pytest.mark.parametrize("configured,requested", [
    ("http://192.168.1.10:8080", "http://192.168.1.10:8080/search?q=x"),
    ("http://192.168.1.10:8080/search", "http://192.168.1.10:8080/search?q=x"),
    ("http://127.0.0.1", "http://127.0.0.1:80/search?q=x"),
    ("https://LOCALHOST.:8443/", "https://localhost:8443/search?q=x"),
])
def test_the_configured_origin_is_exempt_for_the_search_provider(configured, requested):
    _configure(net_search_url=configured)
    check_url(requested, search_backend=True)
    with pytest.raises(NetworkPolicyError):
        check_url(requested)


@pytest.mark.parametrize("configured,requested", [
    ("http://192.168.1.10:8080", "http://192.168.1.10:8081/search"),
    ("http://192.168.1.10:8080", "https://192.168.1.10:8080/search"),
    ("http://192.168.1.10:8080", "http://192.168.1.11:8080/search"),
    ("http://192.168.1.10:8080", "http://127.0.0.1:8080/search"),
    ("", "http://192.168.1.10:8080/search"),
    ("not a url", "http://192.168.1.10:8080/search"),
])
def test_any_other_origin_keeps_the_address_check(configured, requested):
    _configure(net_search_url=configured)
    with pytest.raises(NetworkPolicyError, match="non-public"):
        check_url(requested, search_backend=True)


@pytest.mark.parametrize("configured", ["http://169.254.169.254", "http://[fe80::1]:8080"])
def test_a_link_local_search_backend_is_refused(configured):
    _configure(net_search_url=configured)
    with pytest.raises(NetworkPolicyError, match="link-local"):
        check_url(configured + "/search?q=x", search_backend=True)


def test_pin_time_check_permits_private_but_never_link_local():
    assert netpolicy._resolve_pinned_all("127.0.0.1", search_backend=True) == ["127.0.0.1"]
    assert netpolicy._resolve_pinned_all("192.168.1.10", search_backend=True) == ["192.168.1.10"]
    for host in ("169.254.169.254", "::ffff:169.254.169.254", "fe80::1"):
        with pytest.raises(NetworkPolicyError):
            netpolicy._resolve_pinned_all(host, search_backend=True)
    with pytest.raises(NetworkPolicyError):
        netpolicy._resolve_pinned_all("127.0.0.1")


def test_session_exemption_follows_the_configured_origin():
    _configure(net_search_url="http://127.0.0.1:8080")
    with netpolicy._session_for("http://127.0.0.1:8080/search?q=x", search_backend=True):
        pass
    with pytest.raises(NetworkPolicyError):
        netpolicy._session_for("http://127.0.0.1:9090/search?q=x", search_backend=True)
    with pytest.raises(NetworkPolicyError):
        netpolicy._session_for("http://127.0.0.1:8080/search?q=x")


def test_net_allow_private_still_opens_everything():
    _configure(net_allow_private=True, net_search_url="")
    check_url("http://192.168.1.10:8080/x")
    assert netpolicy._resolve_pinned_all("169.254.169.254") == ["169.254.169.254"]
