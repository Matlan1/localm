# SPDX-License-Identifier: AGPL-3.0-or-later
"""Pins the exact observable behaviour of the netpolicy helpers that the other
netpolicy tests only exercise through a broader outcome: canonical host
forms, the Host header, the SSRF address predicate, the search-backend flag
defaults, the byte-capped request and the URL shape guard."""

import ipaddress
import urllib.parse

import pytest

import localm.netpolicy as netpolicy
from localm.netpolicy import NetworkPolicyError
from tests._net_config import set_config


class TestCanonicalHost:
    def test_every_trailing_dot_is_removed_and_a_leading_dot_is_kept(self):
        assert netpolicy._canonical_host("Example.COM..") == "example.com"
        assert netpolicy._canonical_host(".Example.com") == ".example.com"

    def test_a_trailing_dot_is_removed_not_only_whitespace(self):
        assert netpolicy._canonical_host("example.com.") == "example.com"


class TestDomainList:
    def test_a_non_ascii_entry_is_lowercased_before_its_a_label_form_is_taken(self):
        assert netpolicy._domain_list(["Straße.DE"]) == ["xn--strae-oqa.de"]

    def test_blank_entries_are_dropped(self):
        assert netpolicy._domain_list(["", "  ", "A.com", " *.B.com "]) == [
            "a.com", "b.com"]
        assert netpolicy._domain_list("a.com, ,b.com") == ["a.com", "b.com"]


class TestOrigin:
    def test_the_default_port_follows_the_scheme(self):
        assert netpolicy._origin("http://A.com/x") == ("http", "a.com", 80)
        assert netpolicy._origin("https://A.com/x") == ("https", "a.com", 443)

    def test_an_explicit_port_is_kept(self):
        assert netpolicy._origin("http://a.com:81/") == ("http", "a.com", 81)
        assert netpolicy._origin("https://a.com:80/") == ("https", "a.com", 80)

    def test_a_url_without_a_host_has_no_origin(self):
        assert netpolicy._origin("http:///path") is None
        assert netpolicy._origin("https://") is None

    def test_a_scheme_other_than_http_or_https_has_no_origin_even_with_a_host(self):
        assert netpolicy._origin("ftp://a.com/") is None
        assert netpolicy._origin("file://a.com/x") is None

    def test_the_scheme_is_matched_case_insensitively(self):
        assert netpolicy._origin("HTTP://a.com/") == ("http", "a.com", 80)
        assert netpolicy._origin("HTTPS://a.com/") == ("https", "a.com", 443)


class TestHostHeader:
    @staticmethod
    def _header(url):
        return netpolicy._host_header(urllib.parse.urlparse(url))

    @pytest.mark.parametrize("url, expected", [
        ("http://Example.com/", "example.com"),
        ("http://example.com:80/", "example.com"),
        ("https://example.com:443/", "example.com"),
        ("http://example.com:8080/", "example.com:8080"),
        ("http://example.com:81/", "example.com:81"),
        ("http://example.com:443/", "example.com:443"),
        ("https://example.com:80/", "example.com:80"),
        ("https://example.com:444/", "example.com:444"),
        ("https://example.com:8443/", "example.com:8443"),
    ])
    def test_the_port_is_sent_only_when_it_is_not_the_scheme_default(self, url, expected):
        assert self._header(url) == expected

    def test_a_url_without_a_host_gives_an_empty_header(self):
        assert self._header("http:///x") == ""


class TestIsBlockedIp:
    @pytest.mark.parametrize("address", [
        "100.64.0.1",
        "192.88.99.1",
        "fec0::1",
        "224.0.0.1",
        "ff0e::1",
        "::127.0.0.1",
        "64:ff9b::7f00:1",
    ])
    def test_a_non_public_address_each_predicate_alone_catches_is_blocked(self, address):
        assert netpolicy._is_blocked_ip(ipaddress.ip_address(address)) is True

    @pytest.mark.parametrize("address", [
        "93.184.216.34", "2606:4700::1111", "8.8.8.8", "::ffff:8.8.8.8"])
    def test_a_public_address_is_not_blocked(self, address):
        assert netpolicy._is_blocked_ip(ipaddress.ip_address(address)) is False

    @pytest.mark.parametrize("address", [
        "127.0.0.1", "10.1.2.3", "172.16.0.1", "192.168.1.1", "169.254.169.254",
        "::1", "fe80::1", "fc00::1", "0.0.0.0", "::", "240.0.0.1", "2002::1"])
    def test_loopback_private_link_local_and_unspecified_addresses_are_blocked(
            self, address):
        assert netpolicy._is_blocked_ip(ipaddress.ip_address(address)) is True

    def test_the_global_flag_already_covers_private_loopback_and_link_local(self):
        for address in ["127.0.0.1", "10.1.2.3", "172.16.0.1", "192.168.1.1",
                        "169.254.169.254", "::1", "fe80::1", "fc00::1",
                        "0.0.0.0", "::"]:
            assert ipaddress.ip_address(address).is_global is False, address


_SEARX = "http://searx.lan:8080"


@pytest.fixture
def private_search_backend(monkeypatch):
    monkeypatch.delenv("LOCALM_NET_MODE", raising=False)
    set_config(monkeypatch, lambda: {
        "net_mode": "allow", "net_search_url": _SEARX})
    monkeypatch.setattr(
        "socket.getaddrinfo",
        lambda host, port, *a, **k: [(2, 1, 6, "", ("10.0.0.5", 0))])


class TestSearchBackendFlagDefaults:
    def test_check_url_refuses_the_search_origin_unless_the_flag_is_passed(
            self, private_search_backend):
        with pytest.raises(NetworkPolicyError):
            netpolicy.check_url(_SEARX + "/search")
        with pytest.raises(NetworkPolicyError):
            netpolicy.check_url(_SEARX + "/search", search_backend=False)

    def test_check_url_allows_the_search_origin_when_the_flag_is_passed(
            self, private_search_backend):
        netpolicy.check_url(_SEARX + "/search", search_backend=True)

    def test_the_flag_exempts_only_the_configured_origin(self, private_search_backend):
        with pytest.raises(NetworkPolicyError):
            netpolicy.check_url("http://other.lan:8080/search", search_backend=True)

    def test_resolve_pinned_all_refuses_a_private_address_by_default(
            self, private_search_backend):
        with pytest.raises(NetworkPolicyError):
            netpolicy._resolve_pinned_all("searx.lan")
        assert netpolicy._resolve_pinned_all("searx.lan", search_backend=True) == [
            "10.0.0.5"]

    def test_session_for_refuses_the_search_origin_by_default(
            self, private_search_backend):
        with pytest.raises(NetworkPolicyError):
            netpolicy._session_for(_SEARX + "/search")

    def test_session_for_pins_the_search_origin_when_the_flag_is_passed(
            self, private_search_backend, monkeypatch):
        pinned = []
        monkeypatch.setattr("localm.netpin.pinned_session",
                            lambda ips: pinned.append(list(ips)) or object())
        netpolicy._session_for(_SEARX + "/search", search_backend=True)
        assert pinned == [["10.0.0.5"]]


class TestCheckUrlShape:
    def test_the_backslash_refusal_message_is_exact(self):
        with pytest.raises(NetworkPolicyError) as exc:
            netpolicy.check_url_shape("http://127.0.0.1\\@public.com/")
        assert str(exc.value) == (
            "URL contains a backslash or control character; refusing it "
            "(possible SSRF parser differential).")

    @pytest.mark.parametrize("char", ["\x00", "\x1f", "\x7f", "\n"])
    def test_a_control_character_is_refused(self, char):
        with pytest.raises(NetworkPolicyError, match="backslash or control"):
            netpolicy.check_url_shape(f"http://example.com/a{char}b")

    def test_a_space_and_the_first_printable_character_are_not_control_characters(self):
        assert netpolicy.check_url_shape("http://example.com/a b") == "example.com"
        assert netpolicy.check_url_shape("http://example.com/a!b") == "example.com"
        assert netpolicy.check_url_shape("http://example.com/a\x80b") == "example.com"

    def test_a_host_of_only_dots_is_a_missing_host(self):
        with pytest.raises(NetworkPolicyError, match="URL has no host"):
            netpolicy.check_url_shape("http://./")

    def test_a_url_without_a_hostname_is_a_missing_host(self):
        with pytest.raises(NetworkPolicyError, match="URL has no host"):
            netpolicy.check_url_shape("http:///path")

    def test_only_trailing_dots_are_removed_from_the_host(self):
        assert netpolicy.check_url_shape("http://.a.com./") == ".a.com"

    def test_a_trailing_dot_and_case_are_normalised(self):
        assert netpolicy.check_url_shape("http://Example.COM./x") == "example.com"

    def test_a_non_ascii_host_is_lowercased_before_its_a_label_form_is_taken(self):
        assert netpolicy.check_url_shape("http://Straße.de/") == "xn--strae-oqa.de"


class _RecordingSession:
    def __init__(self):
        self.requests = []
        self.closed = False

    def request(self, method, url, **kw):
        self.requests.append((method, url, kw))
        return _Response()

    def close(self):
        self.closed = True


class _Response:
    def __init__(self):
        self.closed = False
        self.headers = {}

    def close(self):
        self.closed = True


class TestPinnedRequest:
    def _spy_session(self, monkeypatch):
        calls = []
        session = _RecordingSession()

        def fake(*args, **kwargs):
            calls.append((args, kwargs))
            return session

        monkeypatch.setattr(netpolicy, "_session_for", fake)
        return calls, session

    def test_the_default_asks_for_a_session_without_the_search_backend_flag(
            self, monkeypatch):
        calls, _ = self._spy_session(monkeypatch)
        netpolicy.pinned_request("GET", "https://example.com/x")
        assert calls == [(("https://example.com/x",), {})]

    def test_search_backend_asks_for_the_flagged_session_for_the_url(self, monkeypatch):
        calls, _ = self._spy_session(monkeypatch)
        netpolicy.pinned_request("GET", "https://example.com/x", search_backend=True)
        assert calls == [(("https://example.com/x",), {"search_backend": True})]

    def test_the_host_header_names_the_url_and_keeps_the_callers_headers(
            self, monkeypatch):
        _, session = self._spy_session(monkeypatch)
        resp = netpolicy.pinned_request(
            "POST", "http://Example.com:8080/x", headers={"X-A": "1"}, timeout=3)
        (method, url, kw), = session.requests
        assert (method, url) == ("POST", "http://Example.com:8080/x")
        assert kw == {"headers": {"X-A": "1", "Host": "example.com:8080"},
                      "timeout": 3}
        assert resp._localm_pin_session is session

    def test_a_call_without_headers_still_sends_the_host_header(self, monkeypatch):
        _, session = self._spy_session(monkeypatch)
        netpolicy.pinned_request("GET", "https://example.com/")
        assert session.requests[0][2]["headers"] == {"Host": "example.com"}

    def test_without_max_bytes_the_body_is_not_read_here(self, monkeypatch):
        _, session = self._spy_session(monkeypatch)
        resp = netpolicy.pinned_request("GET", "https://example.com/")
        assert resp.closed is False
        assert session.closed is False

    def test_with_max_bytes_the_response_and_session_are_closed(self, monkeypatch):
        _, session = self._spy_session(monkeypatch)
        monkeypatch.setattr(netpolicy, "_body_chunks",
                            lambda resp, per_read, limit: iter([b"ab"]))
        resp = netpolicy.pinned_request("GET", "https://example.com/", max_bytes=10)
        assert resp.closed is True
        assert session.closed is True


class TestCappedRequest:
    def _run(self, monkeypatch, chunks, max_bytes=10, headers=None, kwargs=None):
        session = _RecordingSession()
        seen = []

        def spy(resp, per_read, limit):
            seen.append((per_read, limit))
            return iter(chunks)

        monkeypatch.setattr(netpolicy, "_body_chunks", spy)
        resp = netpolicy._capped_request(
            session, "GET", "https://example.com/", headers or {"Host": "h"},
            max_bytes, kwargs or {})
        return resp, session, seen

    def test_the_request_advertises_only_the_per_read_encodings(self, monkeypatch):
        _, session, _ = self._run(monkeypatch, [b"a"])
        (_, _, kw), = session.requests
        assert kw["headers"] == {
            "Host": "h", "Accept-Encoding": netpolicy.PER_READ_ACCEPT_ENCODING}

    def test_the_request_is_streamed_and_keeps_the_callers_other_arguments(
            self, monkeypatch):
        _, session, _ = self._run(monkeypatch, [b"a"], kwargs={"timeout": 7})
        (_, _, kw), = session.requests
        assert kw == {"headers": kw["headers"], "timeout": 7, "stream": True}

    def test_a_streamed_argument_from_the_caller_is_overridden(self, monkeypatch):
        _, session, _ = self._run(monkeypatch, [b"a"], kwargs={"stream": False})
        assert session.requests[0][2]["stream"] is True

    def test_the_body_is_read_per_network_read_with_one_byte_over_the_cap(
            self, monkeypatch):
        _, _, seen = self._run(monkeypatch, [b"a"], max_bytes=10)
        assert seen == [(True, 11)]

    def test_chunks_are_joined_into_the_content_without_a_separator(self, monkeypatch):
        resp, _, _ = self._run(monkeypatch, [b"abc", b"def"])
        assert resp._content == b"abcdef"
        assert resp._content_consumed is True

    def test_a_body_of_exactly_the_cap_is_accepted(self, monkeypatch):
        resp, _, _ = self._run(monkeypatch, [b"12345"], max_bytes=5)
        assert resp._content == b"12345"

    def test_a_body_one_byte_over_the_cap_is_refused_naming_the_host_and_cap(
            self, monkeypatch):
        with pytest.raises(netpolicy.ResponseTooLarge) as exc:
            self._run(monkeypatch, [b"123456"], max_bytes=5)
        assert str(exc.value) == (
            "The response from example.com is larger than 5 bytes; "
            "refusing to read it.")

    def test_a_body_that_crosses_the_cap_over_several_chunks_is_refused(
            self, monkeypatch):
        with pytest.raises(netpolicy.ResponseTooLarge):
            self._run(monkeypatch, [b"123", b"456"], max_bytes=5)

    def test_the_response_and_the_session_are_closed_on_success(self, monkeypatch):
        resp, session, _ = self._run(monkeypatch, [b"a"])
        assert resp.closed is True and session.closed is True

    def test_the_session_is_closed_when_the_body_is_refused(self, monkeypatch):
        session = _RecordingSession()
        monkeypatch.setattr(netpolicy, "_body_chunks",
                            lambda resp, per_read, limit: iter([b"123456"]))
        with pytest.raises(netpolicy.ResponseTooLarge):
            netpolicy._capped_request(
                session, "GET", "https://example.com/", {}, 5, {})
        assert session.closed is True


class TestUnsetValuesFallBackToTheDefaults:
    @pytest.fixture(autouse=True)
    def _public(self, monkeypatch):
        monkeypatch.delenv("LOCALM_NET_MODE", raising=False)
        monkeypatch.setattr(
            "socket.getaddrinfo",
            lambda host, port, *a, **k: [(2, 1, 6, "", ("93.184.216.34", 0))])

    def test_the_mode_names_are_exactly_off_ask_and_allow(self):
        assert netpolicy.NET_MODES == ("off", "ask", "allow")

    @pytest.mark.parametrize("cfg", [{}, {"net_mode": "bogus"}, {"net_mode": None}])
    def test_a_missing_or_unknown_mode_does_not_disable_the_network(
            self, monkeypatch, cfg):
        set_config(monkeypatch, lambda: dict(cfg))
        netpolicy.check_url("https://example.com/")

    def test_the_off_mode_still_refuses_without_the_model_download_setting(
            self, monkeypatch):
        set_config(monkeypatch, lambda: {"net_mode": "off"})
        with pytest.raises(NetworkPolicyError) as exc:
            netpolicy.check_url("https://example.com/", allow_when_off=True)
        assert exc.value.off is True

    def test_a_missing_private_address_setting_blocks_private_addresses(
            self, monkeypatch):
        set_config(monkeypatch, lambda: {"net_mode": "allow"})
        monkeypatch.setattr(
            "socket.getaddrinfo",
            lambda host, port, *a, **k: [(2, 1, 6, "", ("10.0.0.5", 0))])
        with pytest.raises(NetworkPolicyError):
            netpolicy.check_url("https://example.com/")
