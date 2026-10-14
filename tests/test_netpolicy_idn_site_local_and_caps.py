# SPDX-License-Identifier: AGPL-3.0-or-later
"""Network policy: internationalised host names are matched in their IDNA
A-label form, the IPv6 site-local range is refused, and a capped
``pinned_request`` (used by the search providers) refuses a body longer than
its cap. The capped-read tests run a real HTTP server on loopback."""

from __future__ import annotations

import gzip
import http.server
import ipaddress
import threading

import pytest

from localm import netpolicy
from localm.netpolicy import NetworkPolicyError, ResponseTooLarge, check_url
from localm.web_retrieval import providers
from tests._net_config import set_config
from tests._web_retrieval_fixtures import (
    DDG_ENDPOINT,
    LITE_ENDPOINT,
    FakeResponse,
    Transport,
    allow_public,
)


def _cfg(monkeypatch, **cfg):
    monkeypatch.delenv("LOCALM_NET_MODE", raising=False)
    set_config(monkeypatch, {"net_mode": "allow", **cfg})


def _public_dns(monkeypatch, ip="93.184.216.34"):
    monkeypatch.setattr("socket.getaddrinfo",
                        lambda *a, **k: [(2, 1, 6, "", (ip, 0))])


# --------------------------------------------------------------------------- #
#  IPv6 site-local fec0::/10                                                   #
# --------------------------------------------------------------------------- #

class TestSiteLocalBlocked:
    @pytest.mark.parametrize("addr", ["fec0::1", "fec0::a9fe:a9fe", "feff::1"])
    def test_site_local_addresses_are_blocked(self, addr):
        assert netpolicy._is_blocked_ip(ipaddress.ip_address(addr)) is True

    def test_public_ipv6_is_not_blocked(self):
        assert netpolicy._is_blocked_ip(
            ipaddress.ip_address("2606:4700:4700::1111")) is False

    def test_host_resolving_to_site_local_is_refused(self, monkeypatch):
        _cfg(monkeypatch)
        monkeypatch.setattr("socket.getaddrinfo",
                            lambda *a, **k: [(23, 1, 6, "", ("fec0::1", 0, 0, 0))])
        with pytest.raises(NetworkPolicyError, match="non-public address fec0::1"):
            check_url("https://intranet.example/")

    def test_site_local_literal_is_refused_at_pin_time(self, monkeypatch):
        _cfg(monkeypatch)
        with pytest.raises(NetworkPolicyError, match="non-public"):
            netpolicy._resolve_pinned_all("fec0::1")


# --------------------------------------------------------------------------- #
#  IDNA canonicalisation                                                       #
# --------------------------------------------------------------------------- #

UNICODE_HOST = "bücher.example"
ALABEL_HOST = "xn--bcher-kva.example"


def _refused(url: str) -> bool:
    try:
        check_url(url)
    except NetworkPolicyError:
        return True
    return False


class TestIdnaHostMatching:
    @pytest.mark.parametrize("entry", [UNICODE_HOST, ALABEL_HOST])
    @pytest.mark.parametrize("host", [UNICODE_HOST, ALABEL_HOST,
                                      "BÜCHER.example", "api." + ALABEL_HOST])
    def test_deny_entry_matches_both_spellings(self, monkeypatch, entry, host):
        _cfg(monkeypatch, net_deny=[entry])
        _public_dns(monkeypatch)
        assert _refused(f"https://{host}/") is True

    @pytest.mark.parametrize("entry", [UNICODE_HOST, ALABEL_HOST])
    @pytest.mark.parametrize("host", [UNICODE_HOST, ALABEL_HOST])
    def test_allow_entry_admits_both_spellings(self, monkeypatch, entry, host):
        _cfg(monkeypatch, net_allow=[entry])
        _public_dns(monkeypatch)
        assert _refused(f"https://{host}/") is False

    def test_unrelated_host_still_passes_a_deny_list(self, monkeypatch):
        _cfg(monkeypatch, net_deny=[UNICODE_HOST])
        _public_dns(monkeypatch)
        assert _refused("https://example.org/") is False

    def test_comma_separated_deny_string_is_canonicalised(self, monkeypatch):
        _cfg(monkeypatch, net_deny=f"doubleclick.net, {UNICODE_HOST}")
        _public_dns(monkeypatch)
        assert _refused(f"https://{ALABEL_HOST}/") is True

    def test_eszett_uses_the_encoding_requests_dials(self, monkeypatch):
        import requests
        dialled = requests.Request("GET", "http://straße.example/").prepare().url
        assert "xn--strae-oqa.example" in dialled
        _cfg(monkeypatch, net_deny=["xn--strae-oqa.example"])
        _public_dns(monkeypatch)
        assert _refused("http://straße.example/") is True

    def test_shape_returns_the_a_label(self):
        assert netpolicy.check_url_shape(f"https://{UNICODE_HOST}/x") == ALABEL_HOST

    def test_shape_leaves_ascii_hosts_alone(self):
        assert netpolicy.check_url_shape("https://API.Example.COM./") == "api.example.com"
        assert netpolicy.check_url_shape("http://[::1]:8080/") == "::1"

    def test_invalid_internationalised_host_is_refused(self):
        with pytest.raises(NetworkPolicyError, match="internationalised"):
            netpolicy.check_url_shape("https://☃.example/")

    def test_pin_and_host_header_use_the_a_label(self, monkeypatch):
        _cfg(monkeypatch)
        resolved = []

        def fake_resolve(host):
            resolved.append(host)
            return ["93.184.216.34"]
        monkeypatch.setattr(netpolicy, "_resolve_pinned_all", fake_resolve)
        session = netpolicy._session_for(f"https://{UNICODE_HOST}/")
        session.close()
        assert resolved == [ALABEL_HOST]
        import urllib.parse
        assert netpolicy._host_header(
            urllib.parse.urlparse(f"https://{UNICODE_HOST}:8443/")) == f"{ALABEL_HOST}:8443"


# --------------------------------------------------------------------------- #
#  pinned_request(max_bytes=...) against a real loopback server                #
# --------------------------------------------------------------------------- #

class _Server:
    def __init__(self, body: bytes, headers: dict | None = None):
        self.hits = 0
        self.seen_accept_encoding = []
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def _serve(self):
                outer.hits += 1
                outer.seen_accept_encoding.append(self.headers.get("Accept-Encoding"))
                self.send_response(200)
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                try:
                    self.wfile.write(body)
                except (ConnectionError, OSError):
                    pass

            do_GET = _serve
            do_POST = _serve

            def log_message(self, *a):
                pass

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(5)


@pytest.fixture()
def loopback_allowed(monkeypatch):
    _cfg(monkeypatch, net_allow_private=True)


class TestCappedPinnedRequest:
    def test_body_at_the_cap_is_read(self, loopback_allowed):
        body = b"x" * 4096
        with _Server(body) as srv:
            resp = netpolicy.pinned_request("GET", srv.url + "/", max_bytes=4096,
                                            timeout=10)
        assert resp.content == body
        assert resp.text == body.decode()

    def test_body_over_the_cap_is_refused(self, loopback_allowed):
        with _Server(b"x" * 4097) as srv:
            with pytest.raises(ResponseTooLarge, match="larger than 4096 bytes"):
                netpolicy.pinned_request("GET", srv.url + "/", max_bytes=4096,
                                         timeout=10)

    def test_decompressed_size_is_capped(self, loopback_allowed):
        bomb = gzip.compress(b"\0" * (20 * 1024 * 1024))
        with _Server(bomb, {"Content-Encoding": "gzip"}) as srv:
            with pytest.raises(ResponseTooLarge):
                netpolicy.pinned_request("GET", srv.url + "/", max_bytes=1024 * 1024,
                                         timeout=10)
            assert srv.seen_accept_encoding == [netpolicy.PER_READ_ACCEPT_ENCODING]
        assert len(bomb) < 1024 * 1024

    def test_without_a_cap_the_response_is_left_to_the_caller(self, loopback_allowed):
        body = b"y" * 10000
        with _Server(body) as srv:
            resp = netpolicy.pinned_request("GET", srv.url + "/", stream=True,
                                            timeout=10)
            try:
                assert resp.raw.read() == body
            finally:
                resp.close()


# --------------------------------------------------------------------------- #
#  The search providers read through the cap                                   #
# --------------------------------------------------------------------------- #

class TestSearchProvidersCapped:
    def test_searxng_refuses_an_oversized_answer_without_retrying(
            self, loopback_allowed, monkeypatch):
        monkeypatch.setattr(providers, "_MAX_RESPONSE_BYTES", 64 * 1024)
        body = b'{"results": [' + b'{"title": "t", "url": "https://a.example", "content": "c"},' * 2000 + b'{}]}'
        assert len(body) > 64 * 1024
        with _Server(body, {"Content-Type": "application/json"}) as srv:
            with pytest.raises(ResponseTooLarge):
                providers.SearXNGProvider(srv.url).search("q", 5)
            assert srv.hits == 1

    def test_searxng_reads_an_answer_under_the_cap(self, loopback_allowed):
        body = b'{"results": [{"title": "T", "url": "https://ok.example", "content": "c"}]}'
        with _Server(body, {"Content-Type": "application/json"}) as srv:
            results = providers.SearXNGProvider(srv.url).search("q", 5)
        assert [r.url for r in results] == ["https://ok.example"]

    def test_default_chain_names_an_oversized_answer(self, monkeypatch):
        allow_public(monkeypatch)
        monkeypatch.setattr(providers, "_MAX_RESPONSE_BYTES", 1000)
        big = FakeResponse(headers={"Content-Type": "text/html"}, body=b"a" * 5000)
        transport = Transport().install(monkeypatch)
        transport.route("POST", DDG_ENDPOINT, big)
        transport.route("POST", LITE_ENDPOINT, big)
        transport.route("GET", "https://search.brave.com/search*", big)
        with pytest.raises(providers.SearchProviderError) as info:
            providers.DefaultSearchProvider().search("q", 5)
        assert "larger than localm reads" in str(info.value)
        assert len(transport.urls()) == 3
