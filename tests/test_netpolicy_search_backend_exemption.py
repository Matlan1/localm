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


_METADATA_FORMS = ["169.254.169.254", "::ffff:169.254.169.254", "fe80::1",
                   "64:ff9b::a9fe:a9fe", "::a9fe:a9fe", "2002:a9fe:a9fe::1",
                   "fd00:ec2::254", "0.0.0.0"]


def _url_host(ip: str) -> str:
    return f"[{ip}]" if ":" in ip else ip


@pytest.mark.parametrize("ip", _METADATA_FORMS)
def test_a_link_local_or_metadata_search_backend_is_refused(ip):
    configured = f"http://{_url_host(ip)}:8080"
    _configure(net_search_url=configured)
    with pytest.raises(NetworkPolicyError, match="search backend"):
        check_url(configured + "/search?q=x", search_backend=True)


@pytest.mark.parametrize("ip", _METADATA_FORMS)
def test_pin_time_refuses_link_local_and_metadata_in_every_form(ip):
    with pytest.raises(NetworkPolicyError):
        netpolicy._resolve_pinned_all(ip, search_backend=True)


@pytest.mark.parametrize("ip", ["127.0.0.1", "192.168.1.10", "10.0.0.5", "172.16.3.4",
                                "::1", "fd12:3456::7", "::ffff:192.168.1.10", "8.8.8.8"])
def test_pin_time_permits_loopback_private_ula_and_public(ip):
    assert netpolicy._resolve_pinned_all(ip, search_backend=True)


def test_pin_time_without_the_exemption_refuses_loopback():
    with pytest.raises(NetworkPolicyError):
        netpolicy._resolve_pinned_all("127.0.0.1")


@pytest.mark.parametrize("configured", [
    "https://searx.example.org", "http://search.example.com:8080"])
def test_a_public_host_name_gets_no_exemption(configured, monkeypatch):
    import socket
    _configure(net_search_url=configured)
    monkeypatch.setattr(socket, "getaddrinfo",
                        lambda *a, **k: [(2, 1, 6, "", ("127.0.0.1", 0))])
    assert netpolicy.is_search_backend_origin(configured + "/search", netpolicy._config()) is False
    with pytest.raises(NetworkPolicyError, match="non-public"):
        check_url(configured + "/search?q=x", search_backend=True)


@pytest.mark.parametrize("configured", [
    "http://searx:8080", "http://localhost:8888", "http://nas.local:8080",
    "http://searx.lan", "http://box.home.arpa", "http://s.internal:1"])
def test_a_local_host_name_is_exempt(configured, monkeypatch):
    import socket
    _configure(net_search_url=configured)
    monkeypatch.setattr(socket, "getaddrinfo",
                        lambda *a, **k: [(2, 1, 6, "", ("192.168.1.10", 0))])
    check_url(configured + "/search?q=x", search_backend=True)


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


def test_a_unicode_and_an_a_label_spelling_are_the_same_origin(monkeypatch):
    import socket
    _configure(net_search_url="http://bücher.local:8080")
    monkeypatch.setattr(socket, "getaddrinfo",
                        lambda *a, **k: [(2, 1, 6, "", ("192.168.1.10", 0))])
    assert netpolicy.is_search_backend_origin(
        "http://xn--bcher-kva.local:8080/search", netpolicy._config()) is True
    check_url("http://xn--bcher-kva.local:8080/search?q=x", search_backend=True)
