# SPDX-License-Identifier: AGPL-3.0-or-later
"""Readable failure reasons, transient-failure retries and the total read
budget, driven over real local sockets through the pinned transport.

A raw TCP server on 127.0.0.1 plays the remote host: it can reset a
connection, never answer, drip a body slowly, or answer with a status. Hosts
"resolve" to 127.0.0.1 with ``net_allow_private`` on, so every request runs
the real policy check, pin, requests/urllib3 stack and exception wrapping.
"""

from __future__ import annotations

import gzip
import socket
import struct
import threading
import time
import zlib
from typing import Callable

import pytest
import requests
import urllib3.exceptions

from localm import netpolicy
from localm.netpin import ReadBudgetExceeded
from localm.web_retrieval import errors, providers
from localm.web_retrieval.errors import describe_failure, failure_kind, is_transient
from localm.web_retrieval.providers import SearXNGProvider
from localm.web_retrieval.retrieve import retrieve

from tests._web_retrieval_fixtures import no_sleep


class RawServer:
    """A TCP server on 127.0.0.1 running ``handlers[n]`` for the n-th
    connection (the last handler repeats). Each handler gets the accepted
    socket after the request bytes were read; those bytes are appended to
    ``requests``."""

    def __init__(self, handlers: list[Callable[[socket.socket], None]]):
        self.handlers = handlers
        self.connections = 0
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(8)
        self.port = self.sock.getsockname()[1]
        self._stop = False
        self.requests: list[bytes] = []
        self._threads: list[threading.Thread] = []
        self._accept = threading.Thread(target=self._loop, daemon=True)
        self._accept.start()

    def _loop(self):
        while not self._stop:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return
            handler = self.handlers[min(self.connections,
                                        len(self.handlers) - 1)]
            self.connections += 1
            t = threading.Thread(target=self._run, args=(handler, conn),
                                 daemon=True)
            t.start()
            self._threads.append(t)

    def _run(self, handler, conn):
        try:
            conn.settimeout(5)
            self.requests.append(conn.recv(65536))
            handler(conn)
        except OSError:
            pass
        finally:
            try:
                conn.close()
            except OSError:
                pass

    def close(self):
        self._stop = True
        self.sock.close()
        for t in self._threads:
            t.join(timeout=5)


def reset(conn: socket.socket) -> None:
    conn.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER,
                    struct.pack("ii", 1, 0))


def respond(status: int, body: bytes,
            ctype: str = "application/json") -> Callable:
    def handler(conn: socket.socket) -> None:
        reason = {200: "OK", 403: "Forbidden", 503: "Service Unavailable"}.get(
            status, "X")
        conn.sendall(
            f"HTTP/1.1 {status} {reason}\r\nContent-Type: {ctype}\r\n"
            f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n"
            .encode() + body)
    return handler


def silent(seconds: float) -> Callable:
    def handler(conn: socket.socket) -> None:
        time.sleep(seconds)
    return handler


def drip(interval: float, count: int) -> Callable:
    def handler(conn: socket.socket) -> None:
        conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n"
                     b"Content-Length: %d\r\nConnection: close\r\n\r\n" % count)
        for _ in range(count):
            time.sleep(interval)
            conn.sendall(b"x")
    return handler


@pytest.fixture
def local_net(monkeypatch):
    """``net_mode=allow`` with private targets allowed; every host resolves to
    127.0.0.1 with the requested port; provider sleeps recorded, not slept."""
    cfg: dict = {"net_mode": "allow", "net_allow_private": True}
    monkeypatch.delenv("LOCALM_NET_MODE", raising=False)
    monkeypatch.setattr("localm.config.load_config", lambda: cfg)

    def loopback(host, port=None, *a, **k):
        port = port if isinstance(port, int) else 0
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "",
                 ("127.0.0.1", port))]
    monkeypatch.setattr("socket.getaddrinfo", loopback)
    slept = no_sleep(monkeypatch)
    return cfg, slept


@pytest.fixture
def server():
    made: list[RawServer] = []

    def make(*handlers):
        srv = RawServer(list(handlers))
        made.append(srv)
        return srv
    yield make
    for srv in made:
        srv.close()


_SEARX_OK = (b'{"results": [{"title": "Linz", "url": "https://linz.example/",'
             b' "content": "Linz is a city."}]}')


# --------------------------------------------------------------------------- #
#  Search: retry a reset, report a plain sentence when it never recovers      #
# --------------------------------------------------------------------------- #

class TestSearchRetries:
    def test_reset_then_success_returns_results(self, local_net, server):
        _cfg, slept = local_net
        srv = server(reset, respond(200, _SEARX_OK))
        out = SearXNGProvider(f"http://searx.test:{srv.port}").search("linz", 5)
        assert [r.url for r in out] == ["https://linz.example/"]
        assert srv.connections == 2
        assert slept == [1.0]

    def test_reset_every_time_gives_a_plain_sentence(self, local_net, server):
        cfg, slept = local_net
        srv = server(reset)
        cfg["net_search_url"] = f"http://searx.test:{srv.port}"
        b = retrieve("linz", fetch_top=0)
        assert b.search_status == "failed"
        assert b.search_error.startswith(
            "searx.test closed the connection before answering. Check that")
        assert "ConnectionResetError" not in b.search_error
        assert srv.connections == 3
        assert slept == [1.0, 2.0]

    def test_read_timeout_is_retried_once_only(self, local_net, server,
                                               monkeypatch):
        _cfg, slept = local_net
        monkeypatch.setattr(providers, "_SEARCH_TIMEOUT", 0.3)
        srv = server(silent(2.0))
        with pytest.raises(requests.exceptions.ReadTimeout) as info:
            SearXNGProvider(f"http://searx.test:{srv.port}").search("q", 5)
        assert srv.connections == 2
        assert slept == [1.0]
        assert describe_failure(info.value) == "searx.test did not respond in time"

    def test_http_status_is_not_retried(self, local_net, server):
        _cfg, slept = local_net
        srv = server(respond(503, b"{}"))
        with pytest.raises(requests.HTTPError) as info:
            SearXNGProvider(f"http://searx.test:{srv.port}").search("q", 5)
        assert srv.connections == 1 and slept == []
        assert describe_failure(info.value) == \
            "searx.test had a server error, HTTP 503"

    def test_policy_refusal_is_not_retried(self, local_net, server):
        cfg, slept = local_net
        cfg["net_deny"] = ["searx.test"]
        srv = server(respond(200, _SEARX_OK))
        with pytest.raises(netpolicy.NetworkPolicyError):
            SearXNGProvider(f"http://searx.test:{srv.port}").search("q", 5)
        assert srv.connections == 0 and slept == []


# --------------------------------------------------------------------------- #
#  Page reads: real 403, reset retried once, slow drip capped                  #
# --------------------------------------------------------------------------- #

class TestPageReadFailures:
    def test_real_403_reads_as_a_sentence(self, local_net, server):
        srv = server(respond(403, b"blocked", "text/html"))
        with pytest.raises(requests.HTTPError) as info:
            netpolicy.safe_fetch(f"http://blocked.test:{srv.port}/q/1")
        assert describe_failure(info.value) == (
            "blocked.test refused access, HTTP 403; the site may block "
            "automated readers")
        assert not is_transient(info.value)

    def test_real_reset_is_transient_and_plain(self, local_net, server):
        srv = server(reset)
        with pytest.raises(requests.ConnectionError) as info:
            netpolicy.safe_fetch(f"http://flaky.test:{srv.port}/")
        assert failure_kind(info.value) == "reset"
        assert is_transient(info.value)
        assert describe_failure(info.value) == \
            "flaky.test closed the connection before answering"

    def test_slow_drip_is_cut_at_the_total_budget(self, local_net, server):
        srv = server(drip(0.1, 200))
        started = time.monotonic()
        with pytest.raises(ReadBudgetExceeded) as info:
            netpolicy.safe_fetch(f"http://drip.test:{srv.port}/", timeout=5,
                                 total_timeout=0.6)
        elapsed = time.monotonic() - started
        assert elapsed < 2.0
        assert info.value.seconds == 0.6
        assert describe_failure(info.value, f"http://drip.test:{srv.port}/") == \
            "drip.test was too slow to send the page (over 0.6s)"
        assert not is_transient(info.value)

    def test_without_a_budget_the_drip_completes(self, local_net, server):
        srv = server(drip(0.01, 30))
        _final, _ctype, body = netpolicy.safe_fetch(
            f"http://drip.test:{srv.port}/", timeout=5)
        assert body == "x" * 30

    def test_retrieve_reports_a_hung_page_and_keeps_the_others(
            self, local_net, server):
        cfg, _ = local_net
        hung = server(silent(3.0))
        ok = server(respond(200, b"<html><body><main><p>" + b"Linz facts. " * 40
                            + b"</p></main></body></html>", "text/html"))
        searx = server(respond(200, (
            '{"results": ['
            f'{{"title": "Hung", "url": "http://hung.test:{hung.port}/", '
            '"content": "hung snippet"}, '
            f'{{"title": "Ok", "url": "http://ok.test:{ok.port}/", '
            '"content": "ok snippet"}]}').encode()))
        cfg["net_search_url"] = f"http://searx.test:{searx.port}"
        started = time.monotonic()
        b = retrieve("linz facts", fetch_timeout=5, deadline_seconds=1.0)
        assert time.monotonic() - started < 3.0
        assert b.sources[0].retrieval_status == "failed"
        assert b.sources[0].error == "hung.test did not finish loading within 1s"
        assert b.sources[1].grounding == "page-backed"


# --------------------------------------------------------------------------- #
#  describe_failure / is_transient on the exception shapes requests raises     #
# --------------------------------------------------------------------------- #

def _prepared(url: str):
    return requests.Request("GET", url).prepare()


class TestDescribeFailure:
    def test_the_reported_reset_repr_becomes_a_sentence(self):
        import urllib3.exceptions
        inner = ConnectionResetError(
            10054, "An existing connection was forcibly closed by the remote "
            "host", None, 10054, None)
        exc = requests.ConnectionError(
            urllib3.exceptions.ProtocolError("Connection aborted.", inner),
            request=_prepared("https://html.duckduckgo.com/html/"))
        assert describe_failure(exc) == \
            "html.duckduckgo.com closed the connection before answering"
        assert is_transient(exc)

    def test_read_timeout_inside_a_streamed_body(self):
        import urllib3.exceptions
        exc = requests.ConnectionError(urllib3.exceptions.ReadTimeoutError(
            None, "https://www.accuweather.com/x", "Read timed out."))
        assert describe_failure(exc, "https://www.accuweather.com/x") == \
            "www.accuweather.com did not respond in time"

    def test_tls_failure_is_named_and_not_transient(self):
        exc = requests.exceptions.SSLError(
            "certificate verify failed",
            request=_prepared("https://bad-cert.example/"))
        assert describe_failure(exc).startswith(
            "bad-cert.example failed the secure-connection check")
        assert not is_transient(exc)

    def test_connect_timeout(self):
        exc = requests.exceptions.ConnectTimeout(
            "timed out", request=_prepared("https://slow.example/"))
        assert describe_failure(exc) == "could not connect to slow.example in time"
        assert is_transient(exc)

    @pytest.mark.parametrize("code,text", [
        (404, "page not found on site.example, HTTP 404"),
        (429, "site.example is rate-limiting requests, HTTP 429"),
        (418, "site.example answered HTTP 418"),
    ])
    def test_http_statuses(self, code, text):
        resp = requests.Response()
        resp.status_code = code
        resp.url = "https://site.example/p"
        exc = requests.HTTPError(f"{code} Client Error", response=resp)
        assert describe_failure(exc) == text
        assert not is_transient(exc)

    def test_policy_and_provider_messages_kept(self):
        from localm.web_retrieval.contracts import SearchProviderError
        assert describe_failure(netpolicy.NetworkPolicyError(
            "'x' is on the deny list (net_deny). ")) == \
            "'x' is on the deny list (net_deny)."
        assert describe_failure(SearchProviderError("no results")) == "no results"
        assert not is_transient(netpolicy.NetworkPolicyError("off"))

    def test_unknown_exception_uses_its_message_capped(self):
        assert describe_failure(RuntimeError("boom  \n  here")) == "boom here"
        assert describe_failure(RuntimeError()) == "RuntimeError"
        assert len(describe_failure(RuntimeError("x" * 900))) == 300
        assert errors.failure_kind(RuntimeError("x")) == "other"


# --------------------------------------------------------------------------- #
#  The capped read path decodes gzip/deflate itself, one network read at a    #
#  time, so an encoded slow drip is cut at its budget too                      #
# --------------------------------------------------------------------------- #

def gzip_comment_drip(interval: float, count: int) -> Callable:
    def handler(conn: socket.socket) -> None:
        conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n"
                     b"Content-Encoding: gzip\r\nConnection: close\r\n\r\n"
                     b"\x1f\x8b\x08\x10\x00\x00\x00\x00\x00\xff")
        for _ in range(count):
            time.sleep(interval)
            conn.sendall(b"a")
    return handler


def encoded(body: bytes, encoding: str) -> Callable:
    def handler(conn: socket.socket) -> None:
        conn.sendall(
            b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n"
            + f"Content-Encoding: {encoding}\r\n".encode()
            + f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n"
            .encode() + body)
    return handler


def _raw_deflate(data: bytes) -> bytes:
    comp = zlib.compressobj(9, zlib.DEFLATED, -15)
    return comp.compress(data) + comp.flush()


class TestEncodedBodies:
    def test_gzip_slow_drip_is_cut_at_the_total_budget(self, local_net, server):
        srv = server(gzip_comment_drip(0.1, 60))
        started = time.monotonic()
        with pytest.raises(ReadBudgetExceeded):
            netpolicy.safe_fetch(f"http://gzdrip.test:{srv.port}/", timeout=5,
                                 total_timeout=0.6)
        assert time.monotonic() - started < 2.0

    @pytest.mark.parametrize("encoding,make", [
        ("gzip", gzip.compress),
        ("x-gzip", gzip.compress),
        ("deflate", zlib.compress),
        ("deflate", _raw_deflate),
        ("identity", lambda b: b),
    ], ids=["gzip", "x-gzip", "deflate-zlib", "deflate-raw", "identity"])
    def test_capped_path_decodes(self, local_net, server, encoding, make):
        text = ("Linz facts. " * 400).encode()
        srv = server(encoded(make(text), encoding))
        _final, _ctype, body = netpolicy.safe_fetch(
            f"http://enc.test:{srv.port}/", timeout=5, total_timeout=5)
        assert body.encode() == text

    def test_multi_member_gzip_decodes(self, local_net, server):
        srv = server(encoded(gzip.compress(b"first part, ")
                             + gzip.compress(b"second part"), "gzip"))
        _final, _ctype, body = netpolicy.safe_fetch(
            f"http://enc.test:{srv.port}/", timeout=5, total_timeout=5)
        assert body == "first part, second part"

    def test_unsupported_encoding_is_a_plain_failure(self, local_net, server):
        srv = server(encoded(b"\x00\x01\x02", "br"))
        url = f"http://br.test:{srv.port}/"
        with pytest.raises(requests.exceptions.ContentDecodingError) as info:
            netpolicy.safe_fetch(url, timeout=5, total_timeout=5)
        assert describe_failure(info.value, url) == \
            "br.test sent an incomplete or undecodable response"

    def test_corrupt_gzip_is_a_decoding_error(self, local_net, server):
        srv = server(encoded(b"\x1f\x8b\x08\x00garbage-not-deflate", "gzip"))
        with pytest.raises(requests.exceptions.ContentDecodingError):
            netpolicy.safe_fetch(f"http://bad.test:{srv.port}/", timeout=5,
                                 total_timeout=5)

    def test_decompression_is_capped_at_max_bytes(self, local_net, server):
        srv = server(encoded(gzip.compress(b"\x00" * 20_000_000), "gzip"))
        _final, _ctype, body = netpolicy.safe_fetch_bytes(
            f"http://bomb.test:{srv.port}/", timeout=5, total_timeout=10,
            max_bytes=100_000)
        assert len(body) == 100_000

    def test_capped_request_advertises_only_gzip_and_deflate(self, local_net,
                                                             server):
        srv = server(respond(200, b"ok", "text/plain"))
        netpolicy.safe_fetch(f"http://hdr.test:{srv.port}/", timeout=5,
                             total_timeout=5)
        assert b"\r\naccept-encoding: gzip, deflate\r\n" in \
            srv.requests[0].lower()


class _RaisingRaw:
    def __init__(self, exc):
        self.exc = exc

    def read1(self, amt, decode_content=None):
        raise self.exc


class _Resp:
    def __init__(self, exc):
        self.raw = _RaisingRaw(exc)
        self.headers = {}


@pytest.mark.parametrize("make_exc,expected", [
    (lambda: urllib3.exceptions.SSLError("bad record mac"),
     requests.exceptions.SSLError),
    (lambda: urllib3.exceptions.ReadTimeoutError(None, "/", "Read timed out."),
     requests.exceptions.ConnectionError),
    (lambda: urllib3.exceptions.ProtocolError("broken"),
     requests.exceptions.ChunkedEncodingError),
], ids=["ssl", "read-timeout", "protocol"])
def test_per_read_errors_are_requests_exceptions(make_exc, expected):
    with pytest.raises(expected) as info:
        list(netpolicy._body_chunks(_Resp(make_exc()), True, 1000))
    assert isinstance(info.value, requests.RequestException)


def test_mid_body_tls_error_reads_as_tls():
    exc = requests.exceptions.SSLError(urllib3.exceptions.SSLError(
        "[SSL: DECRYPTION_FAILED_OR_BAD_RECORD_MAC]"))
    assert failure_kind(exc) == "tls"
    assert describe_failure(exc, "https://tls.example/").startswith(
        "tls.example failed the secure-connection check")


# --------------------------------------------------------------------------- #
#  Retries on a real refused connection, and the page budget after a slow     #
#  search                                                                     #
# --------------------------------------------------------------------------- #

@pytest.fixture
def refuse_first(monkeypatch):
    """The first ``state["n"]`` TCP connects are refused by the operating
    system; the rest are dialled for real."""
    import urllib3.connection
    real = urllib3.connection.connection.create_connection
    state = {"n": 0, "seen": 0}

    def maybe_refuse(address, *a, **k):
        state["seen"] += 1
        if state["seen"] <= state["n"]:
            raise ConnectionRefusedError(10061, "refused")
        return real(address, *a, **k)
    monkeypatch.setattr(urllib3.connection.connection, "create_connection",
                        maybe_refuse)
    return state


_LINZ_PAGE = (b"<html><body><main><p>" + b"Linz facts. " * 40
              + b"</p></main></body></html>")


class _OneHit:
    name = "stub"

    def __init__(self, url: str, delay: float = 0.0):
        self.url = url
        self.delay = delay

    def search(self, query, max_results):
        from localm.web_retrieval.contracts import SearchResult
        if self.delay:
            time.sleep(self.delay)
        return [SearchResult(title="Ok", url=self.url, snippet="s", rank=1,
                             provider="stub")]


class TestRealRefusedConnections:
    def test_search_refused_twice_succeeds_on_the_third_try(
            self, local_net, server, refuse_first):
        _cfg, slept = local_net
        refuse_first["n"] = 2
        srv = server(respond(200, _SEARX_OK))
        out = SearXNGProvider(f"http://searx.test:{srv.port}").search("linz", 5)
        assert [r.url for r in out] == ["https://linz.example/"]
        assert refuse_first["seen"] == 3
        assert slept == [1.0, 2.0]

    def test_refused_page_read_is_retried_once(self, local_net, server,
                                               refuse_first):
        page = server(respond(200, _LINZ_PAGE, "text/html"))
        refuse_first["n"] = 1
        b = retrieve("linz facts", provider=_OneHit(
            f"http://ok.test:{page.port}/"), deadline_seconds=5)
        assert b.sources[0].grounding == "page-backed", b.sources[0].error
        assert refuse_first["seen"] == 2


def test_slow_search_leaves_page_reads_their_full_budget(local_net, server):
    page = server(respond(200, _LINZ_PAGE, "text/html"))
    b = retrieve("linz facts", provider=_OneHit(
        f"http://ok.test:{page.port}/", delay=1.2), deadline_seconds=0.5)
    assert b.sources[0].retrieval_status == "fetched", b.sources[0].error
    assert b.sources[0].grounding == "page-backed"
