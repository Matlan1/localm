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
        monkeypatch.setattr("localm.config.load_config", boom)
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
        monkeypatch.setattr("localm.config.load_config", lambda: {"net_mode": "off"})
        with pytest.raises(NetworkPolicyError) as ei:
            netpolicy.check_url("https://example.com/")
        assert str(ei.value) == ("Network access is disabled (net_mode=off). Enable it "
                                 "with:  localm config net_mode ask")
        assert ei.value.off is True

    @pytest.mark.parametrize("env", ["off", "OFF", " Off "])
    def test_the_environment_mode_overrides_the_config_mode(self, monkeypatch, env):
        monkeypatch.setenv("LOCALM_NET_MODE", env)
        monkeypatch.setattr("localm.config.load_config", lambda: {"net_mode": "allow"})
        with pytest.raises(NetworkPolicyError, match="net_mode=off"):
            netpolicy.check_url("https://example.com/")

    def test_the_off_exemption_needs_both_the_caller_flag_and_the_setting(self, monkeypatch):
        for cfg, flag in (({"net_mode": "off", "net_allow_model_downloads": True}, False),
                          ({"net_mode": "off", "net_allow_model_downloads": False}, True),
                          ({"net_mode": "off"}, True)):
            monkeypatch.setattr("localm.config.load_config", lambda cfg=cfg: cfg)
            with pytest.raises(NetworkPolicyError, match="net_mode=off"):
                netpolicy.check_url("https://example.com/", allow_when_off=flag)

    def test_the_allow_list_refusal_lists_the_allowed_hosts_and_the_new_one(self, monkeypatch):
        monkeypatch.setattr("localm.config.load_config", lambda: {
            "net_mode": "allow", "net_allow": ["a.example", "b.example"]})
        with pytest.raises(NetworkPolicyError) as ei:
            netpolicy.check_url("https://c.example/")
        assert str(ei.value) == (
            "'c.example' is not on the allow list (net_allow). Add it with:  "
            "localm config net_allow \"a.example, b.example, c.example\"")
