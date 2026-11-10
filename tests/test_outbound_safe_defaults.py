# SPDX-License-Identifier: AGPL-3.0-or-later
"""Outbound request defaults: no redirects, policy check inside, no credential
forwarding across hosts, bounded reads, and one local-host classifier."""

import http.server
import io
import threading
import urllib.request

import pytest
import requests

from localm import bindhost, netpolicy, peer_routing
from localm import http_ssl
from localm.selfclient import local_request


# --------------------------------------------------------------------------- #
#  pinned_request                                                             #
# --------------------------------------------------------------------------- #

class _Session:
    def __init__(self):
        self.requests = []

    def request(self, method, url, **kw):
        self.requests.append((method, url, kw))
        return type("R", (), {})()


@pytest.fixture()
def spy(monkeypatch):
    session = _Session()
    checks = []
    monkeypatch.setattr(netpolicy, "_session_for", lambda *a, **k: session)
    monkeypatch.setattr(netpolicy, "check_url",
                        lambda url, **k: checks.append((url, k)))
    return session, checks


def test_pinned_request_does_not_follow_redirects_by_default(spy):
    session, _ = spy
    netpolicy.pinned_request("GET", "https://example.com/x")
    assert session.requests[0][2]["allow_redirects"] is False


def test_pinned_request_follows_redirects_only_when_asked(spy):
    session, _ = spy
    netpolicy.pinned_request("GET", "https://example.com/x", allow_redirects=True)
    assert session.requests[0][2]["allow_redirects"] is True


def test_pinned_request_runs_the_policy_check_itself(spy):
    _, checks = spy
    netpolicy.pinned_request("GET", "https://example.com/x",
                             allow_when_off=True, search_backend=True)
    assert checks == [("https://example.com/x",
                       {"allow_when_off": True, "search_backend": True})]


def test_a_refused_url_never_reaches_the_session(monkeypatch):
    session = _Session()
    monkeypatch.setattr(netpolicy, "_session_for", lambda *a, **k: session)

    def refuse(url, **k):
        raise netpolicy.NetworkPolicyError("refused")

    monkeypatch.setattr(netpolicy, "check_url", refuse)
    with pytest.raises(netpolicy.NetworkPolicyError):
        netpolicy.pinned_request("GET", "https://example.com/x")
    assert session.requests == []


# --------------------------------------------------------------------------- #
#  HttpsOnlyRedirect                                                          #
# --------------------------------------------------------------------------- #

def _redirect(url, newurl, headers):
    req = urllib.request.Request(url, headers=headers, method="GET")
    return http_ssl.HttpsOnlyRedirect().redirect_request(
        req, io.BytesIO(b""), 302, "Found", {}, newurl)


_CREDS = {"X-Localm-Token": "secret", "Authorization": "Bearer x",
          "Cookie": "a=b", "User-Agent": "localm"}


def test_a_redirect_to_another_host_drops_caller_credentials():
    new = _redirect("https://proxy.example/dl", "https://cdn.example/f", _CREDS)
    assert {k.lower() for k in new.headers} == {"user-agent"}


def test_a_redirect_on_the_same_host_keeps_caller_headers():
    new = _redirect("https://proxy.example/dl", "https://proxy.example/f", _CREDS)
    assert {k.lower() for k in new.headers} == {k.lower() for k in _CREDS}


def test_a_downgrade_is_still_refused():
    with pytest.raises(http_ssl.RedirectDowngradeRefused):
        _redirect("https://proxy.example/dl", "http://cdn.example/f", _CREDS)


# --------------------------------------------------------------------------- #
#  Bounded readers                                                            #
# --------------------------------------------------------------------------- #

def test_read_bounded_returns_a_body_at_the_cap():
    assert http_ssl.read_bounded(io.BytesIO(b"x" * 10), 10) == b"x" * 10


def test_read_bounded_refuses_a_body_over_the_cap():
    with pytest.raises(http_ssl.BodyTooLarge):
        http_ssl.read_bounded(io.BytesIO(b"x" * 11), 10)


def test_read_bounded_reads_no_more_than_the_cap_plus_one():
    class _Resp:
        asked = []

        def read(self, n=-1):
            self.asked.append(n)
            return b"x" * 11

    with pytest.raises(http_ssl.BodyTooLarge):
        http_ssl.read_bounded(_Resp(), 10)
    assert _Resp.asked == [11]


def test_copy_bounded_stops_writing_past_the_cap():
    out = io.BytesIO()
    with pytest.raises(http_ssl.BodyTooLarge):
        http_ssl.copy_bounded(io.BytesIO(b"x" * 100), out, 30, chunk_size=10)
    assert len(out.getvalue()) == 30


def test_copy_bounded_copies_a_body_within_the_cap():
    out = io.BytesIO()
    assert http_ssl.copy_bounded(io.BytesIO(b"abc" * 10), out, 30, chunk_size=7) == 30
    assert out.getvalue() == b"abc" * 10


def test_a_body_too_large_is_an_oserror():
    assert issubclass(http_ssl.BodyTooLarge, OSError)


# --------------------------------------------------------------------------- #
#  A peer answering 307                                                       #
# --------------------------------------------------------------------------- #

class _Stub:
    """Two loopback servers: the first answers every POST with a 307 to the
    second; the second records the body it receives."""

    def __init__(self):
        self.received = []
        outer = self

        class _Target(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                outer.received.append(self.rfile.read(n))
                self.send_response(200)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def log_message(self, *a):
                pass

        self.target = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Target)

        class _Peer(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                self.rfile.read(n)
                self.send_response(307)
                self.send_header(
                    "Location", f"http://127.0.0.1:{outer.target.server_port}/")
                self.send_header("Content-Length", "0")
                self.end_headers()

            def log_message(self, *a):
                pass

        self.peer = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Peer)
        self.threads = [threading.Thread(target=s.serve_forever, daemon=True)
                        for s in (self.target, self.peer)]

    def __enter__(self):
        for t in self.threads:
            t.start()
        self.url = f"http://127.0.0.1:{self.peer.server_port}/v1/chat/completions"
        return self

    def __exit__(self, *a):
        for s in (self.target, self.peer):
            s.shutdown()
            s.server_close()


def test_raw_requests_re_sends_a_307_body_to_the_redirect_target():
    with _Stub() as stub:
        requests.post(stub.url, data=b"conversation", timeout=5)
    assert stub.received == [b"conversation"]


def test_local_request_returns_a_307_without_following_it():
    with _Stub() as stub:
        resp = local_request("POST", stub.url, data=b"conversation", timeout=5)
    assert resp.status_code == 307
    assert stub.received == []


def test_local_request_refuses_to_be_told_to_follow_redirects():
    with _Stub() as stub:
        resp = local_request("POST", stub.url, data=b"x", timeout=5,
                             allow_redirects=True)
    assert resp.status_code == 307
    assert stub.received == []


@pytest.mark.parametrize("code, refused", [(200, False), (301, True), (307, True),
                                           (399, True), (400, False)])
def test_a_peer_redirect_is_a_credential_probe_failure(code, refused):
    resp = type("R", (), {"status_code": code})()
    if refused:
        with pytest.raises(peer_routing.PeerCredentialError):
            peer_routing._refuse_peer_redirect(resp)
    else:
        peer_routing._refuse_peer_redirect(resp)


# --------------------------------------------------------------------------- #
#  One local-host classifier                                                  #
# --------------------------------------------------------------------------- #

# host: (loopback, wildcard, local_dial, peer_routable)
_AGREEMENT = {
    "127.0.0.1": (True, False, True, True),
    "127.0.0.2": (True, False, True, True),
    "::1": (True, False, True, True),
    "[::1]": (True, False, True, True),
    "localhost": (True, False, True, True),
    "LOCALHOST": (True, False, True, True),
    " localhost ": (True, False, True, True),
    "0.0.0.0": (False, True, True, True),
    "::": (False, True, True, True),
    "": (False, False, False, True),
    "localhost.": (False, False, False, False),
    "2130706433": (False, False, False, False),
    "127.1": (False, False, False, False),
    "0x7f.0.0.1": (False, False, False, False),
    "192.168.1.5": (False, False, False, False),
    "example.com": (False, False, False, False),
    "127.0.0.1.evil.example": (False, False, False, False),
}


@pytest.mark.parametrize("host", list(_AGREEMENT))
def test_the_local_host_predicates_agree_on(host):
    loopback, wildcard, local_dial, routable = _AGREEMENT[host]
    assert bindhost.is_loopback_host(host) is loopback
    assert bindhost.is_wildcard_host(host) is wildcard
    assert bindhost.is_local_dial_host(host) is local_dial
    assert peer_routing.is_routable_peer_host(host) is routable


@pytest.mark.parametrize("value", [None, 5, b"127.0.0.1", ["localhost"]])
def test_a_non_string_host_is_never_local(value):
    assert bindhost.is_loopback_host(value) is False
    assert bindhost.is_wildcard_host(value) is False
    assert bindhost.is_local_dial_host(value) is False


@pytest.mark.parametrize("origin, local", [
    ("http://localhost", True), ("https://127.0.0.1:8642", True),
    ("http://localhost:1", True), ("http://localhost.evil.example", False),
    ("http://127.0.0.1.evil.example", False), ("http://[::1]:80", False),
])
def test_the_cors_origin_pattern_admits_only_literal_localhost_and_127_0_0_1(
        origin, local):
    import re
    assert bool(re.fullmatch(bindhost.LOCAL_ORIGIN_PATTERN, origin)) is local


def test_the_voice_and_cors_regexes_use_the_one_pattern():
    from localm.plugins.builtin.voice import transcriptions
    assert transcriptions._LOCAL_ORIGIN_RE.pattern == (
        f"^{bindhost.LOCAL_ORIGIN_PATTERN}$")
