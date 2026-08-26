# SPDX-License-Identifier: AGPL-3.0-or-later
"""In-process, adversarial unit tests for the portmux HTTP->HTTPS layer.

The existing tests/test_portmux.py drives portmux in a SUBPROCESS (real uvicorn),
which exercises the happy paths but (a) registers no coverage in this process and
(b) never probes the security-sensitive parsing adversarially. This module calls
the pure functions DIRECTLY with fake asyncio streams, so it both covers the code
and pins the open-redirect / header-injection / scheme-confusion defences that the
module docstring promises ("a crafted Host cannot turn the redirect into an open
redirect / header smuggle").

No subprocess, no uvicorn: portmux imports uvicorn lazily inside run_server, so the
pure redirect/routing/throttle helpers are importable and testable on their own.
"""
from __future__ import annotations

import asyncio
import io
import logging
import socket
import sys

import pytest

from localm import portmux


# --------------------------------------------------------------------------- #
#  Fakes + driver
# --------------------------------------------------------------------------- #

class _FakeWriter:
    """Minimal asyncio.StreamWriter stand-in that records everything written."""

    def __init__(self) -> None:
        self.data = b""
        self.closed = False

    def write(self, b: bytes) -> None:
        self.data += b

    async def drain(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True

    def can_write_eof(self) -> bool:
        return True

    def write_eof(self) -> None:
        pass

    def get_extra_info(self, *a, **k):
        return None


def _run_redirect(request: bytes, public_port: int = 8443) -> bytes:
    """Drive _redirect_to_https with *request* and return the raw response bytes.
    The first byte is split off (as _handle_conn does) and passed separately; the
    reader is fed the remainder."""
    async def go() -> bytes:
        reader = asyncio.StreamReader()
        reader.feed_data(request[1:])
        reader.feed_eof()
        writer = _FakeWriter()
        await portmux._redirect_to_https(request[:1], reader, writer, public_port)
        return writer.data
    return asyncio.run(go())


def _split(resp: bytes):
    """(status_line, header_lines, body) from a raw HTTP/1.1 response."""
    head, _, body = resp.partition(b"\r\n\r\n")
    lines = head.decode("latin-1").split("\r\n")
    return lines[0], lines[1:], body.decode("utf-8", "replace")


def _location(header_lines):
    for ln in header_lines:
        if ln.lower().startswith("location:"):
            return ln.split(":", 1)[1].strip()
    return None


# --------------------------------------------------------------------------- #
#  _redirect_to_https: happy paths
# --------------------------------------------------------------------------- #

def test_normal_request_redirects_preserving_path_and_port():
    resp = _run_redirect(b"GET /some/path HTTP/1.1\r\nHost: 127.0.0.1:8443\r\n\r\n")
    status, headers, body = _split(resp)
    assert "308" in status
    assert _location(headers) == "https://127.0.0.1:8443/some/path"
    assert "secure connection" in body


def test_bare_host_gets_the_listening_port_appended():
    # A non-compliant client that omits the port must still land on the actual
    # listening port, never the https default :443.
    resp = _run_redirect(b"GET /x HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n", public_port=9111)
    _, headers, _ = _split(resp)
    assert _location(headers) == "https://127.0.0.1:9111/x"


def test_ipv6_host_is_accepted_and_bracketed_port_preserved():
    resp = _run_redirect(b"GET /p HTTP/1.1\r\nHost: [::1]:8443\r\n\r\n")
    _, headers, _ = _split(resp)
    assert _location(headers) == "https://[::1]:8443/p"


# --------------------------------------------------------------------------- #
#  _redirect_to_https: adversarial (open redirect / injection / scheme confusion)
# --------------------------------------------------------------------------- #

def test_absolute_uri_target_is_neutralised_to_root():
    # An absolute-form request target (proxy style) must NOT be reflected into the
    # redirect; the path allowlist requires a leading "/", so it falls back to "/".
    resp = _run_redirect(
        b"GET http://evil.example/steal HTTP/1.1\r\nHost: 127.0.0.1:8443\r\n\r\n")
    _, headers, _ = _split(resp)
    loc = _location(headers)
    assert loc == "https://127.0.0.1:8443/"
    assert "evil.example" not in (loc or "")


def test_protocol_relative_path_stays_anchored_to_the_real_host():
    # "//evil" is a valid path; it must remain a PATH on the real authority, never
    # become the authority (which would be an open redirect).
    resp = _run_redirect(
        b"GET //evil.example/x HTTP/1.1\r\nHost: 127.0.0.1:8443\r\n\r\n")
    _, headers, _ = _split(resp)
    loc = _location(headers)
    assert loc is not None and loc.startswith("https://127.0.0.1:8443/")


@pytest.mark.parametrize("evil_host", [
    "evil.example/path",          # slash -> not a bare authority
    "user@evil.example",          # userinfo
    "evil.example evil2",         # space
    "evil.example\tx",            # tab / control
    "evil.example:80:443",        # double port
    "a_b.example",                # underscore not in the host charset
    "evil.example%2f",            # percent
])
def test_crafted_host_is_rejected_no_open_redirect(evil_host):
    req = ("GET /x HTTP/1.1\r\nHost: " + evil_host + "\r\n\r\n").encode("latin-1")
    resp = _run_redirect(req)
    status, headers, body = _split(resp)
    # A host that fails the allowlist yields the no-redirect explain page, never a
    # Location reflecting attacker-controlled bytes.
    assert "400" in status
    assert _location(headers) is None
    assert evil_host.split("/")[0] not in (_location(headers) or "")
    assert "secure connection" in body


def test_no_header_injection_in_response():
    # Even a crafted-looking host cannot inject extra response headers: the body
    # boundary is a single CRLFCRLF and every header line is one we control.
    resp = _run_redirect(b"GET /x HTTP/1.1\r\nHost: 127.0.0.1:8443\r\n\r\n")
    head, sep, _ = resp.partition(b"\r\n\r\n")
    assert sep == b"\r\n\r\n"
    allowed = ("http/1.1 ", "location:", "content-type:", "content-length:",
               "connection:")
    for ln in head.decode("latin-1").split("\r\n"):
        assert ln.lower().startswith(allowed), f"unexpected header line: {ln!r}"


def test_missing_host_http10_explains_without_a_location():
    resp = _run_redirect(b"GET / HTTP/1.0\r\n\r\n")
    status, headers, body = _split(resp)
    assert "400" in status
    assert _location(headers) is None
    assert "secure connection" in body


def test_garbage_request_line_falls_back_to_root_path():
    resp = _run_redirect(b"GARBAGE\r\nHost: 127.0.0.1:8443\r\n\r\n")
    _, headers, _ = _split(resp)
    assert _location(headers) == "https://127.0.0.1:8443/"


# --------------------------------------------------------------------------- #
#  First-byte routing: _handle_conn (TLS bind) + _handle_conn_plain (HTTP bind)
# --------------------------------------------------------------------------- #

def _drive_handle(handler, first_byte: bytes, monkeypatch, *, plain: bool):
    """Run a connection handler with _relay / _redirect / _note stubbed to record
    which branch fired. Returns the set of branch names that were called."""
    called: set = set()

    async def fake_relay(first, cr, cw, internal_port):
        called.add("relay")

    async def fake_redirect(first, cr, cw, public_port):
        called.add("redirect")

    def fake_note(public_port, state):
        called.add("note")
        state["count"] += 1

    monkeypatch.setattr(portmux, "_relay", fake_relay)
    monkeypatch.setattr(portmux, "_redirect_to_https", fake_redirect)
    monkeypatch.setattr(portmux, "_note_tls_on_http", fake_note)

    async def go():
        reader = asyncio.StreamReader()
        reader.feed_data(first_byte)
        reader.feed_eof()
        writer = _FakeWriter()
        if plain:
            await portmux._handle_conn_plain(reader, writer, 50000, 8443,
                                             {"warned": False, "count": 0})
        else:
            await portmux._handle_conn(reader, writer, 50000, 8443)
        assert writer.closed, "the client writer must always be closed"
        return called
    return asyncio.run(go())


def test_tls_bind_routes_handshake_byte_to_relay(monkeypatch):
    got = _drive_handle(portmux._handle_conn, b"\x16", monkeypatch, plain=False)
    assert got == {"relay"}


def test_tls_bind_routes_plaintext_to_redirect(monkeypatch):
    got = _drive_handle(portmux._handle_conn, b"G", monkeypatch, plain=False)
    assert got == {"redirect"}


def test_plain_bind_notes_tls_byte_and_does_not_relay(monkeypatch):
    got = _drive_handle(portmux._handle_conn_plain, b"\x16", monkeypatch, plain=True)
    assert got == {"note"}


def test_plain_bind_relays_real_http(monkeypatch):
    got = _drive_handle(portmux._handle_conn_plain, b"G", monkeypatch, plain=True)
    assert got == {"relay"}


def test_empty_connection_closes_cleanly_without_routing(monkeypatch):
    # readexactly(1) on an immediately-EOF connection raises IncompleteReadError;
    # the handler must close the writer and route nowhere (no crash).
    called: set = set()
    monkeypatch.setattr(portmux, "_relay",
                        lambda *a, **k: called.add("relay") or _noop())
    monkeypatch.setattr(portmux, "_redirect_to_https",
                        lambda *a, **k: called.add("redirect") or _noop())

    async def go():
        reader = asyncio.StreamReader()
        reader.feed_eof()                      # no bytes at all
        writer = _FakeWriter()
        await portmux._handle_conn(reader, writer, 50000, 8443)
        return writer.closed, called
    closed, called = asyncio.run(go())
    assert closed and called == set()


async def _noop():
    pass


# --------------------------------------------------------------------------- #
#  _note_tls_on_http throttle state machine
# --------------------------------------------------------------------------- #

def test_tls_on_http_notice_fires_once_then_counts_silently(monkeypatch):
    notices: list = []
    monkeypatch.setattr(portmux, "_safe_notice", lambda msg: notices.append(msg))
    state = {"warned": False, "count": 0}
    for _ in range(4):
        portmux._note_tls_on_http(8443, state)
    assert len(notices) == 1, "exactly one prominent notice per run"
    assert state["count"] == 4, "every occurrence is still counted"
    assert state["warned"] is True
    assert "8443" in notices[0]


# --------------------------------------------------------------------------- #
#  Validation regexes: the security contract, stated as a table
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("host,ok", [
    ("127.0.0.1", True), ("a.com", True), ("a.com:8443", True),
    ("sub-domain.example.com:1", True),
    ("a.com/x", False), ("a com", False), ("user@a.com", False),
    ("a.com:bad", False), ("a_b.com", False), ("a.com\r\n", False), ("", False),
])
def test_host_re_contract(host, ok):
    assert bool(portmux._HOST_RE.match(host)) is ok


@pytest.mark.parametrize("host,ok", [
    ("[::1]", True), ("[::1]:8443", True), ("[2001:db8::1]:80", True),
    ("[::1]extra", False), ("::1", False), ("[gggg::1]", False),
])
def test_host6_re_contract(host, ok):
    assert bool(portmux._HOST6_RE.match(host)) is ok


@pytest.mark.parametrize("path,ok", [
    ("/", True), ("/a/b", True), ("/a%20b", True), ("//x", True),
    ("a/b", False), ("/a b", False), ("/a\tb", False), ("", False),
])
def test_path_re_contract(path, ok):
    assert bool(portmux._PATH_RE.match(path)) is ok


# --------------------------------------------------------------------------- #
#  _redirect_to_https: the readuntil exception branch (client disconnects or
#  sends an oversized preamble mid-handshake) - the "partial bytes" contract
# --------------------------------------------------------------------------- #

def test_redirect_uses_partial_bytes_when_client_disconnects_before_crlfcrlf():
    # A client that sends a Host header then dies mid-handshake (EOF before the
    # blank line) still gets a best-effort redirect built from whatever arrived:
    # readuntil raises IncompleteReadError, whose .partial carries the bytes read
    # so far.
    async def go():
        reader = asyncio.StreamReader()
        reader.feed_data(b"ET /x HTTP/1.1\r\nHost: 127.0.0.1:8443\r\n")
        reader.feed_eof()   # EOF strikes before the terminating blank line
        writer = _FakeWriter()
        await portmux._redirect_to_https(b"G", reader, writer, 8443)
        return writer.data
    resp = asyncio.run(go())
    status, headers, _ = _split(resp)
    assert "308" in status
    assert _location(headers) == "https://127.0.0.1:8443/x"


def test_redirect_falls_back_to_explain_page_on_oversized_preamble():
    # A client that floods headers without ever sending the terminating blank
    # line trips asyncio's LimitOverrunError, which carries no .partial attribute
    # at all, so the getattr fallback is exercised. The result is still a clean
    # HTTP response, never an unhandled exception.
    async def go():
        reader = asyncio.StreamReader()
        oversized = b"X-Pad: " + b"a" * 70000 + b"\r\n"
        reader.feed_data(oversized)
        reader.feed_eof()
        writer = _FakeWriter()
        await portmux._redirect_to_https(b"G", reader, writer, 8443)
        return writer.data
    resp = asyncio.run(go())
    status, headers, body = _split(resp)
    # No usable Host header was ever parsed (the flood pre-empted parsing), so
    # this must be the safe explain page, never a Location built from garbage.
    assert "400" in status
    assert _location(headers) is None
    assert "secure connection" in body


# --------------------------------------------------------------------------- #
#  _handle_conn / _handle_conn_plain: outer except after routing raises
#  mid-stream (the client vanished while _relay/_redirect were mid-flight)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("exc", [ConnectionError, OSError, asyncio.IncompleteReadError])
def test_handle_conn_closes_writer_even_when_redirect_raises_midstream(monkeypatch, exc):
    async def boom(*a, **k):
        if exc is asyncio.IncompleteReadError:
            raise exc(partial=b"", expected=1)
        raise exc("peer vanished")
    monkeypatch.setattr(portmux, "_redirect_to_https", boom)

    async def go():
        reader = asyncio.StreamReader()
        reader.feed_data(b"G")
        reader.feed_eof()
        writer = _FakeWriter()
        await portmux._handle_conn(reader, writer, 50000, 8443)  # must not raise
        return writer.closed
    assert asyncio.run(go()) is True


def test_handle_conn_closes_writer_even_when_relay_raises_midstream(monkeypatch):
    async def boom(*a, **k):
        raise OSError("connection reset mid-relay")
    monkeypatch.setattr(portmux, "_relay", boom)

    async def go():
        reader = asyncio.StreamReader()
        reader.feed_data(b"\x16")
        reader.feed_eof()
        writer = _FakeWriter()
        await portmux._handle_conn(reader, writer, 50000, 8443)  # must not raise
        return writer.closed
    assert asyncio.run(go()) is True


def test_handle_conn_plain_closes_writer_even_when_relay_raises_midstream(monkeypatch):
    async def boom(*a, **k):
        raise ConnectionError("peer vanished")
    monkeypatch.setattr(portmux, "_relay", boom)

    async def go():
        reader = asyncio.StreamReader()
        reader.feed_data(b"G")
        reader.feed_eof()
        writer = _FakeWriter()
        await portmux._handle_conn_plain(reader, writer, 50000, 8443,
                                         {"warned": False, "count": 0})
        return writer.closed
    assert asyncio.run(go()) is True   # must not raise


def test_handle_conn_plain_first_byte_read_failure_closes_cleanly(monkeypatch):
    # A raw connection-level failure (not just a graceful IncompleteReadError)
    # while peeking the first byte must still close the writer, never crash.
    class DyingReader:
        async def readexactly(self, n):
            raise ConnectionResetError("reset before first byte")

    async def go():
        writer = _FakeWriter()
        await portmux._handle_conn_plain(DyingReader(), writer, 50000, 8443,
                                         {"warned": False, "count": 0})
        return writer.closed
    assert asyncio.run(go()) is True


# --------------------------------------------------------------------------- #
#  _note_tls_on_http: best-effort debug logging must never crash the caller,
#  on EITHER the first-notice path or the already-warned counting path
# --------------------------------------------------------------------------- #

def test_note_tls_on_http_survives_a_broken_debug_logger_on_first_notice(monkeypatch):
    monkeypatch.setattr(portmux, "_safe_notice", lambda msg: None)

    def broken_debug(*a, **k):
        raise RuntimeError("logger broke")
    monkeypatch.setattr(portmux._log, "debug", broken_debug)

    state = {"warned": False, "count": 0}
    portmux._note_tls_on_http(8443, state)   # must not raise
    assert state["warned"] is True
    assert state["count"] == 1


def test_note_tls_on_http_survives_a_broken_debug_logger_on_repeat(monkeypatch):
    def broken_debug(*a, **k):
        raise RuntimeError("logger broke")
    monkeypatch.setattr(portmux._log, "debug", broken_debug)

    state = {"warned": True, "count": 0}
    portmux._note_tls_on_http(8443, state)   # must not raise
    assert state["count"] == 1


# --------------------------------------------------------------------------- #
#  _pump: the copy loop's failure/teardown branches
# --------------------------------------------------------------------------- #

def test_pump_swallows_a_connection_error_from_the_source():
    class RaisingReader:
        async def read(self, n):
            raise ConnectionResetError("reset")

    async def go():
        writer = _FakeWriter()
        await portmux._pump(RaisingReader(), writer)   # must not raise
        return writer.data
    assert asyncio.run(go()) == b""


def test_pump_swallows_a_write_eof_failure_on_half_close():
    class WriterEofRaises(_FakeWriter):
        def write_eof(self):
            raise OSError("half-close failed")

    async def go():
        reader = asyncio.StreamReader()
        reader.feed_eof()
        writer = WriterEofRaises()
        await portmux._pump(reader, writer)   # must not raise
    asyncio.run(go())


def test_pump_skips_write_eof_when_the_writer_cannot_half_close():
    class NoEofWriter(_FakeWriter):
        def can_write_eof(self):
            return False
        def write_eof(self):
            raise AssertionError("write_eof must not be called when unsupported")

    async def go():
        reader = asyncio.StreamReader()
        reader.feed_eof()
        writer = NoEofWriter()
        await portmux._pump(reader, writer)   # must not raise / assert
    asyncio.run(go())


# --------------------------------------------------------------------------- #
#  _relay: the OSError early-return when the internal backend is unreachable
# --------------------------------------------------------------------------- #

def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_relay_returns_cleanly_when_the_internal_backend_is_unreachable():
    # If the internal uvicorn is not (yet, or no longer) listening, _relay must
    # give up quietly rather than propagate a connection-refused error up into
    # the caller's routing logic.
    async def go():
        reader = asyncio.StreamReader()
        reader.feed_eof()
        writer = _FakeWriter()
        dead_port = _free_port()   # bound then released: nothing listens here
        await portmux._relay(b"X", reader, writer, dead_port)   # must not raise
        return writer.data
    assert asyncio.run(go()) == b""


# --------------------------------------------------------------------------- #
#  _safe_close
# --------------------------------------------------------------------------- #

def test_safe_close_with_none_writer_is_a_noop():
    portmux._safe_close(None)   # must not raise


def test_safe_close_swallows_close_errors():
    class BadWriter:
        def close(self):
            raise OSError("already gone")
    portmux._safe_close(BadWriter())   # must not raise


def test_safe_close_closes_a_real_writer():
    class GoodWriter:
        def __init__(self):
            self.closed = False
        def close(self):
            self.closed = True
    w = GoodWriter()
    portmux._safe_close(w)
    assert w.closed is True


# --------------------------------------------------------------------------- #
#  _get_stable_stream: resolve-once caching
# --------------------------------------------------------------------------- #

@pytest.fixture
def _cold_stable_stream(monkeypatch):
    """Force _get_stable_stream's module-level cache back to its unresolved
    state for the duration of one test, regardless of test order."""
    monkeypatch.setattr(portmux, "_stable_resolved", False)
    monkeypatch.setattr(portmux, "_stable_stream", None)


def test_get_stable_stream_resolves_once_and_caches(monkeypatch, _cold_stable_stream):
    sentinel = object()
    calls = []

    def fake_resolver():
        calls.append(1)
        return sentinel
    monkeypatch.setattr("localm.debuglog._stable_console_stream", fake_resolver)

    first = portmux._get_stable_stream()
    second = portmux._get_stable_stream()
    third = portmux._get_stable_stream()
    assert first is sentinel and second is sentinel and third is sentinel
    assert len(calls) == 1, "a resolved stream must be cached, not re-resolved per call"


def test_get_stable_stream_caches_a_failed_resolution_as_none(monkeypatch, _cold_stable_stream):
    calls = []

    def broken_resolver():
        calls.append(1)
        raise RuntimeError("no duplicable fd")
    monkeypatch.setattr("localm.debuglog._stable_console_stream", broken_resolver)

    first = portmux._get_stable_stream()
    second = portmux._get_stable_stream()
    assert first is None and second is None
    assert len(calls) == 1, "a failed resolution must not be retried on every connection"


# --------------------------------------------------------------------------- #
#  _safe_notice: an operational notice must NEVER be able to crash the caller
# --------------------------------------------------------------------------- #

def test_safe_notice_writes_through_the_stable_stream(monkeypatch):
    written = []

    class FakeStream:
        def write(self, s):
            written.append(s)
        def flush(self):
            pass
    monkeypatch.setattr(portmux, "_get_stable_stream", lambda: FakeStream())

    portmux._safe_notice("something happened")
    assert any("something happened" in s for s in written)


def test_safe_notice_falls_back_to_live_stderr_when_no_stable_stream(monkeypatch, capsys):
    monkeypatch.setattr(portmux, "_get_stable_stream", lambda: None)
    portmux._safe_notice("fallback-path-message")
    captured = capsys.readouterr()
    assert "fallback-path-message" in captured.err


def test_safe_notice_never_raises_even_when_the_stream_write_fails(monkeypatch):
    class BrokenStream:
        def write(self, s):
            raise OSError("[WinError 6] The handle is invalid")
        def flush(self):
            pass
    monkeypatch.setattr(portmux, "_get_stable_stream", lambda: BrokenStream())
    portmux._safe_notice("must not crash the caller")   # must not raise


# --------------------------------------------------------------------------- #
#  _harden_uvicorn_logging
# --------------------------------------------------------------------------- #

def test_harden_uvicorn_logging_is_a_noop_without_a_stable_stream(monkeypatch):
    monkeypatch.setattr(portmux, "_get_stable_stream", lambda: None)
    logger = logging.getLogger("uvicorn")
    handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(handler)
    try:
        portmux._harden_uvicorn_logging()
        assert handler.stream is sys.stdout   # untouched
    finally:
        logger.removeHandler(handler)


def test_harden_uvicorn_logging_redirects_stream_handlers_but_skips_file_handlers(
        monkeypatch, tmp_path):
    fake_stable = io.StringIO()
    monkeypatch.setattr(portmux, "_get_stable_stream", lambda: fake_stable)

    stream_handler = logging.StreamHandler(sys.stdout)
    file_handler = logging.FileHandler(str(tmp_path / "uv.log"))
    logger = logging.getLogger("uvicorn.error")
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    try:
        portmux._harden_uvicorn_logging()
        assert stream_handler.stream is fake_stable
        # FileHandler IS a StreamHandler subclass; the explicit exclusion must
        # hold or a log file would be silently repointed at the console dup.
        assert file_handler.stream is not fake_stable
    finally:
        logger.removeHandler(stream_handler)
        logger.removeHandler(file_handler)
        file_handler.close()


def test_harden_uvicorn_logging_swallows_a_setstream_failure(monkeypatch):
    fake_stable = io.StringIO()
    monkeypatch.setattr(portmux, "_get_stable_stream", lambda: fake_stable)

    class BadHandler(logging.StreamHandler):
        def setStream(self, stream):
            raise RuntimeError("nope")
    handler = BadHandler(sys.stdout)
    logger = logging.getLogger("uvicorn.asgi")
    logger.addHandler(handler)
    try:
        portmux._harden_uvicorn_logging()   # must not raise
    finally:
        logger.removeHandler(handler)
