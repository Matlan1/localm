# SPDX-License-Identifier: AGPL-3.0-or-later
"""Contract tests for the fetch entry points in localm.netpolicy:
safe_fetch_bytes, safe_fetch, _body_chunks and web_search.

Each test pins one observable argument or return value that the other
netpolicy tests leave unchecked (what reaches the transport, what reaches the
policy check, what is decoded)."""

import ipaddress
import socket
import time
import urllib.parse

import pytest
import requests

import localm.netpolicy as netpolicy
from localm.netpin import _CONNECT_BUDGET_FACTOR, ReadBudgetExceeded
from localm.netpolicy import NetworkPolicyError, safe_fetch, safe_fetch_bytes
from tests._net_config import set_config


@pytest.fixture(autouse=True)
def _public_allow_mode(monkeypatch):
    monkeypatch.delenv("LOCALM_NET_MODE", raising=False)
    set_config(monkeypatch, lambda: {"net_mode": "allow"})
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
        set_config(monkeypatch, lambda: {
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
        set_config(monkeypatch, lambda: {
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


class _Counting:
    def __init__(self, chunks):
        self.chunks = chunks
        self.consumed = 0

    def __iter__(self):
        for c in self.chunks:
            self.consumed += 1
            yield c


class _ChunkedResp(_Resp):
    def __init__(self, chunks, **kw):
        super().__init__(**kw)
        self.stream = _Counting(chunks)

    def iter_content(self, chunk_size=65536):
        return iter(self.stream)


class TestBodyReadLoop:
    def test_reading_stops_once_the_cap_is_reached_not_after_it(self, monkeypatch):
        resp = _ChunkedResp([b"a" * 10] * 5)
        _transport(monkeypatch, lambda url, kw: resp)
        body = safe_fetch_bytes("https://example.com/", max_bytes=20)[2]
        assert body == b"a" * 20
        assert resp.stream.consumed == 2

    def test_the_size_counts_every_chunk_read(self, monkeypatch):
        resp = _ChunkedResp([b"a" * 10] * 5)
        _transport(monkeypatch, lambda url, kw: resp)
        assert safe_fetch_bytes("https://example.com/", max_bytes=25)[2] == b"a" * 25
        assert resp.stream.consumed == 3

    def test_the_size_starts_at_zero(self, monkeypatch):
        resp = _ChunkedResp([b"a" * 5, b"b" * 4, b"c" * 3, b"d" * 3])
        _transport(monkeypatch, lambda url, kw: resp)
        body = safe_fetch_bytes("https://example.com/", max_bytes=10)[2]
        assert body == b"aaaaabbbbc"
        assert resp.stream.consumed == 3

    def test_chunks_are_joined_without_a_separator(self, monkeypatch):
        resp = _ChunkedResp([b"ab", b"cd"])
        _transport(monkeypatch, lambda url, kw: resp)
        assert safe_fetch_bytes("https://example.com/")[2] == b"abcd"

    def test_the_transport_session_is_opened_for_the_current_url(self, monkeypatch):
        opened = []

        def session_for(url):
            opened.append(url)
            return _Session([], _ok)
        monkeypatch.setattr(netpolicy, "_session_for", session_for)
        safe_fetch_bytes("https://example.com/p")
        assert opened == ["https://example.com/p"]

    def test_a_body_still_arriving_at_the_deadline_instant_is_kept(self, monkeypatch):
        clock = _Clock()
        monkeypatch.setattr(time, "monotonic", clock)

        class _Tick(_ChunkedResp):
            def iter_content(self, chunk_size=65536):
                for c in self.stream.chunks:
                    clock.now = 1005.0
                    yield c
        resp = _Tick([b"ab", b"cd"])
        _transport(monkeypatch, lambda url, kw: resp)
        assert safe_fetch_bytes("https://example.com/", total_timeout=5)[2] == b"abcd"

    def test_a_body_still_arriving_after_the_deadline_names_the_url(self, monkeypatch):
        clock = _Clock()
        monkeypatch.setattr(time, "monotonic", clock)

        class _Late(_ChunkedResp):
            def iter_content(self, chunk_size=65536):
                for c in self.stream.chunks:
                    clock.now = 1006.0
                    yield c
        resp = _Late([b"ab", b"cd"])
        _transport(monkeypatch, lambda url, kw: resp)
        with pytest.raises(ReadBudgetExceeded) as ei:
            safe_fetch_bytes("https://example.com/slow", total_timeout=5)
        assert ei.value.url == "https://example.com/slow"
        assert ei.value.seconds == 5

    def test_a_timed_request_reads_the_raw_stream_and_decodes_gzip(self, monkeypatch):
        import gzip
        monkeypatch.setattr(time, "monotonic", _Clock())

        class _Raw1(_Resp):
            def __init__(self):
                super().__init__(headers={"Content-Encoding": "gzip"})
                self.raw = _Raw([gzip.compress(b"hello" * 10)])

            def iter_content(self, chunk_size=65536):
                raise AssertionError("iter_content must not be used")
        _transport(monkeypatch, lambda url, kw: _Raw1())
        assert safe_fetch_bytes("https://example.com/", total_timeout=5)[2] == b"hello" * 10

    def test_an_untimed_request_reads_through_iter_content(self, monkeypatch):
        class _NoRead1(_Resp):
            def __init__(self):
                super().__init__(body=b"via-iter")
                self.raw = _Raw([b"via-raw"])
        _transport(monkeypatch, lambda url, kw: _NoRead1())
        assert safe_fetch_bytes("https://example.com/")[2] == b"via-iter"


class TestBodyChunksErrors:
    def test_a_read_timeout_keeps_its_message(self):
        from urllib3.exceptions import ReadTimeoutError

        class _Slow:
            def read1(self, amt, decode_content=True):
                raise ReadTimeoutError(None, "u", "slow read")
        with pytest.raises(requests.exceptions.ConnectionError, match="slow read"):
            list(netpolicy._body_chunks(_ChunkResp(_Slow()), True, 10))

    def test_an_ssl_failure_keeps_its_message(self):
        from urllib3.exceptions import SSLError

        class _Bad:
            def read1(self, amt, decode_content=True):
                raise SSLError("handshake broke")
        with pytest.raises(requests.exceptions.SSLError, match="handshake broke"):
            list(netpolicy._body_chunks(_ChunkResp(_Bad()), True, 16))

    def test_data_left_in_the_decoder_after_the_last_read_is_yielded(self):
        import gzip
        payload = bytes(range(256)) * 8
        raw = _Raw([gzip.compress(payload)])
        resp = _ChunkResp(raw, {"Content-Encoding": "gzip"})
        assert b"".join(netpolicy._body_chunks(resp, True, 16)) == payload


class TestBodyDecoder:
    @staticmethod
    def _gz(data):
        import gzip
        return gzip.compress(data)

    @pytest.mark.parametrize("encoding", ["", None, "identity", "IDENTITY", " Identity "])
    def test_identity_passes_bytes_through(self, encoding):
        dec = netpolicy._BodyDecoder(encoding)
        assert dec.decode(b"plain", 100) == b"plain"
        assert dec.flush() == b""

    @pytest.mark.parametrize("encoding", ["gzip", "x-gzip", "GZIP", "X-Gzip"])
    def test_gzip_names_are_decoded(self, encoding):
        out = netpolicy._BodyDecoder(encoding).decode(self._gz(b"hi" * 20), 1000)
        assert out == b"hi" * 20

    def test_zlib_wrapped_and_raw_deflate_are_both_decoded(self):
        import zlib
        wrapped = zlib.compress(b"deflate me" * 5)
        co = zlib.compressobj(9, zlib.DEFLATED, -zlib.MAX_WBITS)
        raw = co.compress(b"deflate me" * 5) + co.flush()
        assert netpolicy._BodyDecoder("deflate").decode(wrapped, 1000) == b"deflate me" * 5
        assert netpolicy._BodyDecoder("deflate").decode(raw, 1000) == b"deflate me" * 5

    def test_raw_deflate_split_over_two_reads_is_decoded(self):
        import zlib
        co = zlib.compressobj(9, zlib.DEFLATED, -zlib.MAX_WBITS)
        raw = co.compress(b"0123456789" * 20) + co.flush()
        dec = netpolicy._BodyDecoder("deflate")
        assert dec.decode(raw[:10], 1000) + dec.decode(raw[10:], 1000) == b"0123456789" * 20

    def test_a_comma_separated_list_with_one_encoding_is_accepted(self):
        import zlib
        assert netpolicy._BodyDecoder("deflate , ").decode(zlib.compress(b"x"), 10) == b"x"

    @pytest.mark.parametrize("encoding", ["br", "gzip, gzip", "gzip,deflate", "compress"])
    def test_other_encodings_are_refused(self, encoding):
        with pytest.raises(requests.exceptions.ContentDecodingError, match="unsupported"):
            netpolicy._BodyDecoder(encoding)

    def test_corrupt_gzip_data_is_refused_naming_the_mode(self):
        with pytest.raises(requests.exceptions.ContentDecodingError, match="corrupt gzip body"):
            netpolicy._BodyDecoder("gzip").decode(b"this is not gzip", 100)

    def test_corrupt_deflate_data_is_refused_naming_the_mode(self):
        with pytest.raises(requests.exceptions.ContentDecodingError, match="corrupt deflate body"):
            netpolicy._BodyDecoder("deflate").decode(b"\x00\x01\x02garbage", 100)

    def test_a_gzip_declaration_does_not_accept_zlib_framing(self):
        import zlib
        with pytest.raises(requests.exceptions.ContentDecodingError, match="corrupt gzip body"):
            netpolicy._BodyDecoder("gzip").decode(zlib.compress(b"x" * 20), 100)

    def test_each_step_yields_at_most_the_limit(self):
        assert len(netpolicy._BodyDecoder("gzip").decode(self._gz(b"a" * 1000), 10)) == 10

    def test_deflate_steps_yield_at_most_the_limit(self):
        import zlib
        co = zlib.compressobj(9, zlib.DEFLATED, -zlib.MAX_WBITS)
        raw = co.compress(b"z" * 1000) + co.flush()
        for stream in (zlib.compress(b"z" * 1000), raw):
            assert len(netpolicy._BodyDecoder("deflate").decode(stream, 10)) == 10

    def test_a_second_deflate_chunk_is_decoded_by_the_same_stream(self):
        import zlib
        wrapped = zlib.compress(b"0123456789" * 30)
        dec = netpolicy._BodyDecoder("deflate")
        out = dec.decode(wrapped[:20], 1000) + dec.decode(wrapped[20:], 1000)
        assert out == b"0123456789" * 30

    def test_corrupt_data_after_a_good_deflate_chunk_is_refused(self):
        import zlib
        wrapped = zlib.compress(b"0123456789" * 30)
        dec = netpolicy._BodyDecoder("deflate")
        dec.decode(wrapped[:20], 1000)
        with pytest.raises(requests.exceptions.ContentDecodingError, match="corrupt deflate body"):
            dec.decode(bytes(range(250, 256)) + b" garbage garbage", 1000)

    def test_the_raw_deflate_retry_is_only_for_the_first_chunk(self):
        import zlib
        wrapped = zlib.compress(b"0123456789" * 30)
        stored_block = bytes([1, 5, 0, 0xFA, 0xFF]) + b"hello"
        dec = netpolicy._BodyDecoder("deflate")
        dec.decode(wrapped[:6], 1000)
        with pytest.raises(requests.exceptions.ContentDecodingError, match="corrupt deflate body"):
            dec.decode(stored_block, 1000)

    def test_a_trailing_member_must_be_gzip_framed(self):
        import zlib
        data = self._gz(b"first") + zlib.compress(b"second")
        with pytest.raises(requests.exceptions.ContentDecodingError, match="corrupt gzip body"):
            netpolicy._BodyDecoder("gzip").decode(data, 1000)

    def test_a_trailing_member_is_cut_at_the_room_left_under_the_limit(self):
        data = self._gz(b"ab") + self._gz(b"cdefghij")
        assert netpolicy._BodyDecoder("gzip").decode(data, 5) == b"abcde"

    def test_several_gzip_members_are_decoded_in_one_step(self):
        data = self._gz(b"first-") + self._gz(b"second")
        assert netpolicy._BodyDecoder("gzip").decode(data, 1000) == b"first-second"

    def test_members_after_the_limit_is_reached_are_not_decoded(self):
        data = self._gz(b"abcde") + self._gz(b"fghij")
        assert netpolicy._BodyDecoder("gzip").decode(data, 5) == b"abcde"


class TestCheckUrlMessages:
    def test_an_unreadable_config_refuses_with_the_retry_message_and_logs_the_cause(
            self, monkeypatch):
        import logging

        def boom():
            raise OSError("disk gone")
        monkeypatch.setattr("localm.config.load_config_checked", boom)
        records = []

        class _Grab(logging.Handler):
            def emit(self, record):
                records.append(record.getMessage())
        handler = _Grab(level=logging.WARNING)
        netpolicy.logger.addHandler(handler)
        try:
            with pytest.raises(NetworkPolicyError) as ei:
                netpolicy.check_url("https://example.com/")
        finally:
            netpolicy.logger.removeHandler(handler)
        assert str(ei.value) == (
            "Network policy configuration could not be read; refusing this "
            "request as a precaution. Retry once the config is readable.")
        assert ei.value.off is False
        assert records == [
            "netpolicy: could not load config (disk gone); refusing this request "
            "(fail-safe) rather than resolving net_mode and net_deny/"
            "net_allow from different reads"]

    def test_the_off_refusal_names_the_remedy_and_is_marked_off(self, monkeypatch):
        set_config(monkeypatch, lambda: {"net_mode": "off"})
        with pytest.raises(NetworkPolicyError) as ei:
            netpolicy.check_url("https://example.com/")
        assert str(ei.value) == ("Network access is disabled (net_mode=off). Enable it "
                                 "with:  localm config net_mode ask")
        assert ei.value.off is True

    @pytest.mark.parametrize("env", ["off", "OFF", " Off "])
    def test_the_environment_mode_overrides_the_config_mode(self, monkeypatch, env):
        monkeypatch.setenv("LOCALM_NET_MODE", env)
        set_config(monkeypatch, lambda: {"net_mode": "allow"})
        with pytest.raises(NetworkPolicyError, match="net_mode=off"):
            netpolicy.check_url("https://example.com/")

    def test_the_off_exemption_needs_both_the_caller_flag_and_the_setting(self, monkeypatch):
        for cfg, flag in (({"net_mode": "off", "net_allow_model_downloads": True}, False),
                          ({"net_mode": "off", "net_allow_model_downloads": False}, True),
                          ({"net_mode": "off"}, True)):
            set_config(monkeypatch, lambda cfg=cfg: cfg)
            with pytest.raises(NetworkPolicyError, match="net_mode=off"):
                netpolicy.check_url("https://example.com/", allow_when_off=flag)

    def test_the_allow_list_refusal_lists_the_allowed_hosts_and_the_new_one(self, monkeypatch):
        set_config(monkeypatch, lambda: {
            "net_mode": "allow", "net_allow": ["a.example", "b.example"]})
        with pytest.raises(NetworkPolicyError) as ei:
            netpolicy.check_url("https://c.example/")
        assert str(ei.value) == (
            "'c.example' is not on the allow list (net_allow). Add it with:  "
            "localm config net_allow \"a.example, b.example, c.example\"")


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


def _addrinfo(*addresses):
    return lambda host, port, *a, **k: [(2, 1, 6, "", (ip, 0)) for ip in addresses]


def _unresolvable(host, port, *a, **k):
    raise socket.gaierror("no such host")


class TestCanonicalHostMapping:
    def test_a_full_width_name_is_mapped_to_its_ascii_form(self):
        name = "ｅｘａｍｐｌｅ.com"
        assert netpolicy._canonical_host(name) == "example.com"


class TestSearchBackendAddressCheck:
    def test_a_dotless_decimal_link_local_literal_is_refused(self, monkeypatch):
        monkeypatch.setattr("socket.getaddrinfo", _unresolvable)
        with pytest.raises(NetworkPolicyError):
            netpolicy._check_search_backend_address("2852039166")

    def test_a_link_local_ipv6_literal_is_refused(self, monkeypatch):
        monkeypatch.setattr("socket.getaddrinfo", _unresolvable)
        with pytest.raises(NetworkPolicyError):
            netpolicy._check_search_backend_address("fe80::1")

    def test_a_loopback_ipv6_literal_is_accepted(self, monkeypatch):
        monkeypatch.setattr("socket.getaddrinfo", _unresolvable)
        netpolicy._check_search_backend_address("::1")

    def test_a_host_name_resolving_to_a_metadata_address_is_refused(self, monkeypatch):
        seen = []

        def resolve(host, port, *a, **k):
            seen.append(host)
            return [(2, 1, 6, "", ("169.254.169.254", 0))]

        monkeypatch.setattr("socket.getaddrinfo", resolve)
        with pytest.raises(NetworkPolicyError, match="169.254.169.254"):
            netpolicy._check_search_backend_address("searx.lan")
        assert seen == ["searx.lan"]

    def test_an_unresolvable_host_is_accepted(self, monkeypatch):
        monkeypatch.setattr("socket.getaddrinfo", _unresolvable)
        netpolicy._check_search_backend_address("searx.lan")

    def test_an_unparseable_resolver_answer_is_skipped(self, monkeypatch):
        monkeypatch.setattr("socket.getaddrinfo", lambda *a, **k: [
            (2, 1, 6, "", ("not-an-ip", 0)), (2, 1, 6, "", ("10.0.0.5", 0))])
        netpolicy._check_search_backend_address("searx.lan")


class TestSearchBackendAddressOk:
    @pytest.mark.parametrize("address, expected", [
        ("93.184.216.34", True), ("127.0.0.1", True), ("10.0.0.5", True),
        ("192.168.1.1", True), ("::1", True), ("fd12::1", True),
        ("::ffff:10.0.0.5", True), ("::ffff:127.0.0.1", True),
        ("169.254.169.254", False), ("0.0.0.0", False), ("fe80::1", False),
        ("fd00:ec2::254", False), ("::ffff:169.254.169.254", False),
        ("100.64.0.1", False), ("224.0.0.1", False),
    ])
    def test_which_addresses_the_search_backend_may_use(self, address, expected):
        assert netpolicy._search_backend_address_ok(
            ipaddress.ip_address(address)) is expected


class TestInterleaveFamilies:
    def test_a_longer_second_family_is_appended_in_order(self):
        v4 = ipaddress.ip_address("1.1.1.1")
        v6 = [ipaddress.ip_address(f"2606:4700::{n}") for n in (1, 2, 3)]
        assert netpolicy._interleave_families([v4, *v6]) == [v4, v6[0], v6[1], v6[2]]

    def test_a_longer_first_family_is_appended_in_order(self):
        v4 = [ipaddress.ip_address(f"1.1.1.{n}") for n in (1, 2, 3)]
        v6 = ipaddress.ip_address("2606:4700::1")
        assert netpolicy._interleave_families([*v4, v6]) == [v4[0], v6, v4[1], v4[2]]

    def test_an_empty_list_stays_empty(self):
        assert netpolicy._interleave_families([]) == []


class TestIsLocalNameOrLiteral:
    @pytest.mark.parametrize("host, expected", [
        ("localhost", True), ("box", True), ("searx.lan", True), ("nas.local", True),
        ("a.home.arpa", True), ("10.0.0.5", True), ("2130706433", True),
        ("::1", True), ("::ffff:8.8.8.8", True), ("fe80::1", True),
        ("example.com", False), ("www.example.org", False), ("lan.example.com", False),
    ])
    def test_which_hosts_are_local_by_their_form(self, host, expected):
        assert netpolicy._is_local_name_or_literal(host) is expected


class TestResolvePinnedAll:
    @pytest.fixture(autouse=True)
    def _config(self, monkeypatch):
        set_config(monkeypatch, lambda: {"net_mode": "allow"})

    def test_a_numeric_loopback_literal_is_refused(self, monkeypatch):
        monkeypatch.setattr("socket.getaddrinfo", _unresolvable)
        with pytest.raises(NetworkPolicyError):
            netpolicy._resolve_pinned_all("2130706433")

    def test_a_link_local_ipv6_literal_is_refused_and_a_public_one_is_returned(
            self, monkeypatch):
        monkeypatch.setattr("socket.getaddrinfo", _unresolvable)
        with pytest.raises(NetworkPolicyError):
            netpolicy._resolve_pinned_all("fe80::1")
        assert netpolicy._resolve_pinned_all("2606:4700::1111") == ["2606:4700::1111"]

    def test_the_host_name_is_what_the_resolver_is_asked_for(self, monkeypatch):
        seen = []

        def resolve(host, port, *a, **k):
            seen.append(host)
            return [(2, 1, 6, "", ("93.184.216.34", 0))]

        monkeypatch.setattr("socket.getaddrinfo", resolve)
        assert netpolicy._resolve_pinned_all("example.com") == ["93.184.216.34"]
        assert seen == ["example.com"]

    def test_net_allow_private_lets_a_private_address_through(self, monkeypatch):
        set_config(monkeypatch, lambda: {"net_allow_private": True})
        monkeypatch.setattr("socket.getaddrinfo", _addrinfo("10.0.0.5"))
        assert netpolicy._resolve_pinned_all("box.lan") == ["10.0.0.5"]
        assert netpolicy._resolve_pinned_all("10.0.0.5") == ["10.0.0.5"]

    def test_a_blocked_address_does_not_hide_a_later_public_one(self, monkeypatch):
        monkeypatch.setattr("socket.getaddrinfo", _addrinfo("10.0.0.5", "93.184.216.34"))
        assert netpolicy._resolve_pinned_all("example.com") == ["93.184.216.34"]

    def test_only_blocked_addresses_are_refused_with_the_exact_message(self, monkeypatch):
        monkeypatch.setattr("socket.getaddrinfo", _addrinfo("10.0.0.5"))
        with pytest.raises(NetworkPolicyError) as exc:
            netpolicy._resolve_pinned_all("example.com")
        assert str(exc.value) == (
            "'example.com' resolves to the non-public address 10.0.0.5. "
            "Requests to local/private networks are blocked "
            "(set net_allow_private true to permit them).")

    def test_an_answer_with_no_parseable_address_is_unresolvable_not_blocked(
            self, monkeypatch):
        monkeypatch.setattr("socket.getaddrinfo", lambda *a, **k: [
            (2, 1, 6, "", ("not-an-ip", 0))])
        assert netpolicy._resolve_pinned_all("example.com") == []

    def test_an_unresolvable_host_gives_no_addresses(self, monkeypatch):
        monkeypatch.setattr("socket.getaddrinfo", _unresolvable)
        assert netpolicy._resolve_pinned_all("example.com") == []


class TestSessionFor:
    @pytest.fixture(autouse=True)
    def _config(self, monkeypatch):
        set_config(monkeypatch, lambda: {"net_mode": "allow"})

    def test_the_session_is_pinned_to_the_address_of_the_urls_host(self, monkeypatch):
        seen = []

        def resolve(host, port, *a, **k):
            seen.append(host)
            return [(2, 1, 6, "", ("93.184.216.34", 0))]

        pinned = []
        monkeypatch.setattr("socket.getaddrinfo", resolve)
        monkeypatch.setattr("localm.netpin.pinned_session",
                            lambda ips: pinned.append(list(ips)) or object())
        netpolicy._session_for("https://Example.com/x")
        assert seen == ["example.com"]
        assert pinned == [["93.184.216.34"]]

    def test_a_url_without_a_host_is_unresolvable(self, monkeypatch):
        def resolve(host, port, *a, **k):
            if host == "":
                raise socket.gaierror("empty host")
            return [(2, 1, 6, "", ("93.184.216.34", 0))]

        monkeypatch.setattr("socket.getaddrinfo", resolve)
        with pytest.raises(netpolicy.UnresolvableHost):
            netpolicy._session_for("http:///x")
