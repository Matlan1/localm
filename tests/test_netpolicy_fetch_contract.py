# SPDX-License-Identifier: AGPL-3.0-or-later
"""Contract tests for the fetch entry points in localm.netpolicy:
safe_fetch_bytes, safe_fetch, _body_chunks and web_search.

Each test pins one observable argument or return value that the other
netpolicy tests leave unchecked (what reaches the transport, what reaches the
policy check, what is decoded)."""

import time

import pytest
import requests

import localm.netpolicy as netpolicy
from localm.netpin import _CONNECT_BUDGET_FACTOR, ReadBudgetExceeded
from localm.netpolicy import NetworkPolicyError, safe_fetch, safe_fetch_bytes


@pytest.fixture(autouse=True)
def _public_allow_mode(monkeypatch):
    monkeypatch.delenv("LOCALM_NET_MODE", raising=False)
    monkeypatch.setattr("localm.config.load_config", lambda: {"net_mode": "allow"})
    monkeypatch.setattr(
        "socket.getaddrinfo",
        lambda host, port, *a, **k: [(2, 1, 6, "", ("93.184.216.34", 0))])


class _Resp:
    def __init__(self, *, status=200, headers=None, body=b""):
        self.status_code = status
        self.headers = dict(headers or {})
        self._body = body

    @property
    def is_redirect(self):
        return 300 <= self.status_code < 400 and "Location" in self.headers

    is_permanent_redirect = False

    def iter_content(self, chunk_size=65536):
        for i in range(0, len(self._body), chunk_size):
            yield self._body[i:i + chunk_size]

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def close(self):
        pass


class _Session:
    def __init__(self, calls, responder):
        self._calls = calls
        self._responder = responder

    def get(self, url, **kw):
        self._calls.append((url, kw))
        return self._responder(url, kw)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _transport(monkeypatch, responder):
    calls: list = []
    monkeypatch.setattr(netpolicy, "_session_for",
                        lambda url: _Session(calls, responder))
    return calls


def _ok(url, kw):
    return _Resp(headers={"Content-Type": "text/plain"}, body=b"ok")


class TestPolicyCheckArguments:
    def test_off_mode_refuses_by_default_even_when_downloads_are_allowed(self, monkeypatch):
        monkeypatch.setattr("localm.config.load_config", lambda: {
            "net_mode": "off", "net_allow_model_downloads": True})
        _transport(monkeypatch, _ok)
        with pytest.raises(NetworkPolicyError):
            safe_fetch_bytes("https://example.com/")

    def test_allow_when_off_reaches_the_policy_check_unchanged(self, monkeypatch):
        seen = []
        monkeypatch.setattr(netpolicy, "check_url",
                            lambda url, **kw: seen.append(kw))
        _transport(monkeypatch, _ok)
        safe_fetch_bytes("https://example.com/", allow_when_off=True)
        safe_fetch_bytes("https://example.com/", allow_when_off=False)
        safe_fetch_bytes("https://example.com/")
        assert seen == [{"allow_when_off": True}, {"allow_when_off": False},
                        {"allow_when_off": False}]

    def test_allow_when_off_lifts_the_off_floor_for_a_download(self, monkeypatch):
        monkeypatch.setattr("localm.config.load_config", lambda: {
            "net_mode": "off", "net_allow_model_downloads": True})
        _transport(monkeypatch, _ok)
        assert safe_fetch_bytes("https://example.com/", allow_when_off=True)[2] == b"ok"


class TestRequestShape:
    def test_each_hop_requests_the_current_url_streamed_without_following_redirects(
            self, monkeypatch):
        calls = _transport(monkeypatch, _ok)
        safe_fetch_bytes("https://example.com/a?q=1")
        (url, kw), = calls
        assert url == "https://example.com/a?q=1"
        assert kw["stream"] is True
        assert kw["allow_redirects"] is False
        assert kw["timeout"] == netpolicy._DEFAULT_TIMEOUT
        assert "Accept-Encoding" not in kw["headers"]

    def test_extra_headers_are_sent_on_the_original_host(self, monkeypatch):
        calls = _transport(monkeypatch, _ok)
        safe_fetch_bytes("https://example.com/", extra_headers={"X-Token": "t"})
        assert calls[0][1]["headers"]["X-Token"] == "t"
        assert calls[0][1]["headers"]["User-Agent"] == netpolicy._USER_AGENT

    def test_a_relative_redirect_is_resolved_against_the_current_url(self, monkeypatch):
        def responder(url, kw):
            if url.endswith("/start"):
                return _Resp(status=302, headers={"Location": "/next/page"})
            return _Resp(headers={"Content-Type": "text/plain"}, body=b"x")
        calls = _transport(monkeypatch, responder)
        final, _, _ = safe_fetch_bytes("https://example.com/dir/start")
        assert [c[0] for c in calls] == [
            "https://example.com/dir/start", "https://example.com/next/page"]
        assert final == "https://example.com/next/page"

    def test_a_redirect_without_a_location_is_refused(self, monkeypatch):
        class _Bare(_Resp):
            is_redirect = True
        _transport(monkeypatch, lambda url, kw: _Bare(status=302))
        with pytest.raises(NetworkPolicyError, match="without a Location"):
            safe_fetch_bytes("https://example.com/")

    def test_a_missing_content_type_is_an_empty_string(self, monkeypatch):
        _transport(monkeypatch, lambda url, kw: _Resp(body=b"x"))
        assert safe_fetch_bytes("https://example.com/")[1] == ""


class TestRedirectLimit:
    def _chain(self, monkeypatch, hops):
        def responder(url, kw):
            n = int(url.rsplit("/", 1)[1])
            if n < hops:
                return _Resp(status=302, headers={"Location": f"/{n + 1}"})
            return _Resp(headers={"Content-Type": "text/plain"}, body=b"done")
        return _transport(monkeypatch, responder)

    def test_exactly_the_maximum_number_of_redirects_is_followed(self, monkeypatch):
        calls = self._chain(monkeypatch, netpolicy._MAX_REDIRECTS)
        assert safe_fetch_bytes("https://example.com/0")[2] == b"done"
        assert len(calls) == netpolicy._MAX_REDIRECTS + 1

    def test_one_redirect_more_than_the_maximum_is_refused_after_the_last_allowed_request(
            self, monkeypatch):
        calls = self._chain(monkeypatch, netpolicy._MAX_REDIRECTS + 1)
        with pytest.raises(NetworkPolicyError, match="Too many redirects"):
            safe_fetch_bytes("https://example.com/0")
        assert len(calls) == netpolicy._MAX_REDIRECTS + 1


class _Clock:
    def __init__(self, now=1000.0):
        self.now = now

    def __call__(self):
        return self.now


class TestTotalTimeout:
    def test_each_hop_gets_a_connect_and_a_read_budget_from_the_time_left(
            self, monkeypatch):
        monkeypatch.setattr(time, "monotonic", _Clock())
        calls = _transport(monkeypatch, _ok)
        safe_fetch_bytes("https://example.com/", timeout=15, total_timeout=8)
        assert calls[0][1]["timeout"] == (8 / _CONNECT_BUDGET_FACTOR, 8)

    def test_the_per_wait_timeout_caps_both_budgets(self, monkeypatch):
        monkeypatch.setattr(time, "monotonic", _Clock())
        calls = _transport(monkeypatch, _ok)
        safe_fetch_bytes("https://example.com/", timeout=3, total_timeout=100)
        assert calls[0][1]["timeout"] == (3, 3)

    def test_a_timed_request_advertises_only_the_per_read_encodings(self, monkeypatch):
        monkeypatch.setattr(time, "monotonic", _Clock())
        calls = _transport(monkeypatch, _ok)
        safe_fetch_bytes("https://example.com/", total_timeout=5)
        assert calls[0][1]["headers"]["Accept-Encoding"] == \
            netpolicy.PER_READ_ACCEPT_ENCODING

    def test_a_budget_under_one_second_is_still_usable(self, monkeypatch):
        monkeypatch.setattr(time, "monotonic", _Clock())
        calls = _transport(monkeypatch, _ok)
        assert safe_fetch_bytes("https://example.com/", total_timeout=0.5)[2] == b"ok"
        assert calls[0][1]["timeout"] == (0.25, 0.5)

    def test_an_exhausted_budget_raises_before_any_request_and_reports_the_allowance(
            self, monkeypatch):
        monkeypatch.setattr(time, "monotonic", _Clock())
        calls = _transport(monkeypatch, _ok)
        with pytest.raises(ReadBudgetExceeded) as ei:
            safe_fetch_bytes("https://example.com/x", total_timeout=0)
        assert ei.value.seconds == 0
        assert ei.value.url == "https://example.com/x"
        assert calls == []

    def test_a_budget_spent_on_the_first_hop_stops_the_redirect_chain(self, monkeypatch):
        clock = _Clock()
        monkeypatch.setattr(time, "monotonic", clock)

        def responder(url, kw):
            clock.now += 10
            return _Resp(status=302, headers={"Location": "/again"})
        calls = _transport(monkeypatch, responder)
        with pytest.raises(ReadBudgetExceeded) as ei:
            safe_fetch_bytes("https://example.com/", total_timeout=5)
        assert len(calls) == 1
        assert ei.value.seconds == 5
        assert ei.value.url == "https://example.com/"


class TestSafeFetchForwarding:
    def test_the_caller_limits_reach_safe_fetch_bytes(self, monkeypatch):
        seen = []

        def fake(url, **kw):
            seen.append((url, kw))
            return url, "text/plain", b"x"
        monkeypatch.setattr(netpolicy, "safe_fetch_bytes", fake)
        safe_fetch("https://example.com/", max_bytes=77, timeout=7, total_timeout=9)
        assert seen == [("https://example.com/",
                         {"max_bytes": 77, "timeout": 7, "total_timeout": 9})]

    def test_an_omitted_total_timeout_is_forwarded_as_none(self, monkeypatch):
        seen = []

        def fake(url, **kw):
            seen.append(kw)
            return url, "text/plain", b"x"
        monkeypatch.setattr(netpolicy, "safe_fetch_bytes", fake)
        safe_fetch("https://example.com/")
        assert seen[0]["total_timeout"] is None
        assert seen[0]["timeout"] == netpolicy._DEFAULT_TIMEOUT
        assert seen[0]["max_bytes"] == netpolicy._DEFAULT_MAX_BYTES

    def test_a_declared_charset_that_cannot_decode_the_body_replaces_instead_of_falling_back(
            self, monkeypatch):
        body = "café".encode()
        monkeypatch.setattr(netpolicy, "safe_fetch_bytes",
                            lambda url, **kw: (url, "text/plain; charset=ascii", body))
        assert safe_fetch("https://example.com/")[2] == "caf��"

    def test_an_undeclared_body_with_invalid_utf8_decodes_with_replacement(self, monkeypatch):
        monkeypatch.setattr(netpolicy, "safe_fetch_bytes",
                            lambda url, **kw: (url, "text/plain", b"a\xffb"))
        assert safe_fetch("https://example.com/")[2] == "a�b"


class _Raw:
    def __init__(self, chunks):
        self._chunks = list(chunks)
        self.calls = []

    def read1(self, amt, decode_content=True):
        self.calls.append((amt, decode_content))
        return self._chunks.pop(0) if self._chunks else b""


class _ChunkResp:
    def __init__(self, raw, headers=None):
        self.raw = raw
        self.headers = headers or {}

    def iter_content(self, chunk_size=None):
        raise AssertionError("iter_content must not be used")


class TestBodyChunks:
    def test_per_read_mode_reads_the_raw_stream_one_read_at_a_time(self):
        raw = _Raw([b"ab", b"cd"])
        out = list(netpolicy._body_chunks(_ChunkResp(raw), True, 1000))
        assert out == [b"ab", b"cd"]
        assert raw.calls == [(netpolicy._BODY_CHUNK, False)] * 3

    def test_per_read_mode_decodes_the_declared_content_encoding(self):
        import gzip
        raw = _Raw([gzip.compress(b"hello")])
        resp = _ChunkResp(raw, {"Content-Encoding": "gzip"})
        assert b"".join(netpolicy._body_chunks(resp, True, 1000)) == b"hello"

    def test_a_response_without_a_raw_stream_falls_back_to_iter_content(self):
        class _NoRaw:
            headers: dict = {}

            def iter_content(self, chunk_size=None):
                yield b"x"
        assert list(netpolicy._body_chunks(_NoRaw(), True, 10)) == [b"x"]

    def test_without_per_read_the_raw_stream_is_ignored(self):
        class _Both(_ChunkResp):
            def iter_content(self, chunk_size=None):
                yield b"via-iter"
        raw = _Raw([b"via-raw"])
        assert list(netpolicy._body_chunks(_Both(raw), False, 10)) == [b"via-iter"]
        assert raw.calls == []

    def test_a_read_timeout_surfaces_as_a_requests_connection_error(self):
        from urllib3.exceptions import ReadTimeoutError

        class _Slow:
            def read1(self, amt, decode_content=True):
                raise ReadTimeoutError(None, "u", "slow")
        with pytest.raises(requests.exceptions.ConnectionError):
            list(netpolicy._body_chunks(_ChunkResp(_Slow()), True, 10))


class TestWebSearchDefault:
    def test_the_default_result_count_is_five(self, monkeypatch):
        seen = []

        def fake(query, max_results, *a, **k):
            seen.append((query, max_results))
            return []
        monkeypatch.setattr("localm.web_retrieval.providers.search", fake)
        netpolicy.web_search("hello")
        assert seen == [("hello", 5)]
