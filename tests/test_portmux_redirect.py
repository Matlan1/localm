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
