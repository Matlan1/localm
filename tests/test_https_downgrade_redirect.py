# SPDX-License-Identifier: AGPL-3.0-or-later
"""http_ssl.HttpsOnlyRedirect: a verified HTTPS request must not be walked off
HTTPS by a redirect.

Verifying the first hop's certificate says nothing about the hops after it.
urllib's HTTPRedirectHandler follows up to 10 redirects and, in http_error_302,
admits any target whose scheme is in ``('http', 'https', 'ftp', '')``, so a
plain ``http://`` Location IS followed, in cleartext. HttpsOnlyRedirect is
installed by verified_urlopen for every caller, and refuses that downgrade.

THREE real loopback stubs, not one, and the third is what makes the other two
mean anything:

  hop A  https, the redirector: 302s to whatever target the test picks.
  hop B  PLAIN HTTP, the downgrade target: serves distinctive bytes it must
         never get the chance to serve.
  hop C  https, the legitimate second hop: the GitHub -> release-CDN 302 that
         every real download depends on.

The refused redirect points at a stub that WOULD answer: a Location pointing at
something genuinely unreachable cannot tell "refused" apart from "reached for it
and failed". Hop B is never contacted when the guard holds; without it, hop B's
bytes come back to the caller exactly as a real downgrade would deliver them.

THE CONTROL FOR THE CONTROL: test_urllib_default_does_follow_the_downgrade
drives these same stubs through urllib's OWN opener and asserts hop B's bytes DO
arrive, so "hop B was never dialed" cannot also be the reading for a hop A that
stopped redirecting, a cert that stopped verifying, or a broken fixture.

TEST (b) IS THE LOAD-BEARING ONE. Refusing every 3xx would pass (a) and break
setup-llama, the updater and the bug-report upload, all of which legitimately
follow a 302 within https.
"""

from __future__ import annotations

import ssl
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from localm import http_ssl, tls

_HOPB_BYTES = b"HOPB-CLEARTEXT-BYTES-THAT-MUST-NEVER-ARRIVE"
_HOPC_BYTES = b"HOPC-LEGITIMATE-SECOND-HOP-BYTES"

_real_create_default_context = ssl.create_default_context


class _Stub(HTTPServer):
    """A loopback stub that records every path it was actually asked for."""

    def __init__(self, handler, tls_files=None):
        super().__init__(("127.0.0.1", 0), handler)
        self.hit_paths: list = []
        self.redirect_to = ""
        self.body = b""
        self.scheme = "http"
        if tls_files:
            cert, key = tls_files
            ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ctx.load_cert_chain(cert, key)
            self.socket = ctx.wrap_socket(self.socket, server_side=True)
            self.scheme = "https"

    @property
    def base_url(self) -> str:
        return f"{self.scheme}://127.0.0.1:{self.server_address[1]}"


class _Redirector(BaseHTTPRequestHandler):
    def log_message(self, *_a):
        pass

    def do_GET(self):  # noqa: N802
        self.server.hit_paths.append(self.path)
        self.send_response(302)
        self.send_header("Location", self.server.redirect_to)
        self.send_header("Content-Length", "0")
        self.end_headers()


class _Payload(BaseHTTPRequestHandler):
    def log_message(self, *_a):
        pass

    def do_GET(self):  # noqa: N802
        self.server.hit_paths.append(self.path)
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(self.server.body)))
        self.end_headers()
        self.wfile.write(self.server.body)


def _serve(stub):
    t = threading.Thread(target=stub.serve_forever, daemon=True)
    t.start()
    return t


@pytest.fixture
def local_ca(tmp_path, monkeypatch):
    """A throwaway CA + 127.0.0.1 leaf (localm's own tls.py mints them), and
    verified_urlopen taught to trust that CA.

    Patches ssl.create_default_context rather than passing a context in, because
    the code under test builds its own. The certifi fallback builds one the same
    way, so a real certificate failure still takes its normal path."""
    cert, key = tls.ensure_cert(tmp_path)
    ca = str(tls.ca_cert_path(tmp_path))

    def trusting_context(*_a, **_k):
        ctx = _real_create_default_context()
        ctx.load_verify_locations(cafile=ca)
        return ctx

    monkeypatch.setattr(http_ssl.ssl, "create_default_context", trusting_context)
    return (cert, key), trusting_context


@pytest.fixture
def hops(local_ca):
    """(hop_a https redirector, hop_b plain-http target, hop_c https target)."""
    tls_files, _ctx = local_ca
    a = _Stub(_Redirector, tls_files=tls_files)
    b = _Stub(_Payload)
    c = _Stub(_Payload, tls_files=tls_files)
    b.body = _HOPB_BYTES
    c.body = _HOPC_BYTES
    threads = [_serve(a), _serve(b), _serve(c)]
    try:
        yield a, b, c
    finally:
        for s in (a, b, c):
            s.shutdown()
            s.server_close()
        for t in threads:
            t.join(timeout=5)


def test_https_to_http_redirect_is_refused_and_never_dialed(hops):
    """(a) The defect. The downgrade target must not be contacted at all."""
    a, b, _c = hops
    a.redirect_to = f"{b.base_url}/payload"

    req = urllib.request.Request(f"{a.base_url}/start")
    with pytest.raises(http_ssl.RedirectDowngradeRefused) as exc_info:
        http_ssl.verified_urlopen(req, timeout=10)

    assert b.hit_paths == [], "the cleartext hop was dialed"
    assert a.hit_paths == ["/start"], (
        "hop A was retried - a refused downgrade must not be mistaken for a "
        "certificate failure and re-attempted against certifi")
    assert "downgrade" in str(exc_info.value).lower()
    assert b.base_url in str(exc_info.value)


def test_https_to_https_redirect_is_still_followed(hops):
    """(b) The load-bearing one. Refusing all 3xx would pass (a) and break every
    real download: the GitHub -> release-CDN hop is a legitimate 302."""
    a, b, c = hops
    a.redirect_to = f"{c.base_url}/payload"

    req = urllib.request.Request(f"{a.base_url}/start")
    with http_ssl.verified_urlopen(req, timeout=10) as r:
        assert r.read() == _HOPC_BYTES
    assert c.hit_paths == ["/payload"]
    assert b.hit_paths == []


def test_urllib_default_does_follow_the_downgrade(hops, local_ca):
    """The control: the same stubs, driven through urllib's OWN opener, DO hand
    the caller hop B's cleartext bytes. So test (a)'s empty hop_paths is the
    guard refusing, not the fixture failing to offer the redirect."""
    a, b, _c = hops
    _tls_files, trusting_context = local_ca
    a.redirect_to = f"{b.base_url}/payload"

    opener = urllib.request.build_opener(
        urllib.request.HTTPSHandler(context=trusting_context()))
    with opener.open(f"{a.base_url}/start", timeout=10) as r:
        assert r.read() == _HOPB_BYTES
    assert b.hit_paths == ["/payload"]


def test_refusal_is_a_urlerror_so_every_caller_reports_it(hops):
    """RedirectDowngradeRefused is a URLError: each outbound caller funnels
    URLError into its own domain error while interpolating the reason, so a
    refusal surfaces as a real failure everywhere and never reads as a
    success."""
    a, b, _c = hops
    a.redirect_to = f"{b.base_url}/payload"

    req = urllib.request.Request(f"{a.base_url}/start")
    with pytest.raises(urllib.error.URLError) as exc_info:
        http_ssl.verified_urlopen(req, timeout=10)
    assert isinstance(exc_info.value, http_ssl.RedirectDowngradeRefused)
    assert "downgrade" in str(exc_info.value.reason).lower()


def test_setup_llama_download_refuses_and_says_why(hops, tmp_path):
    """End to end at the layer where the consequence lands rather than at the
    guard: setup_llama names the downgrade rather than routing it into its
    generic OSError branch (a URLError IS an OSError), which tells the user this
    "looks like a dropped or flaky connection" and to retry.

    Asserted on the DATA first: whether a file was written is the property, the
    message is the proxy. A runner stops at the first failing assertion, so this
    order decides whether a regression reports the loss or reports a string."""
    from localm import setup_llama as sl

    a, b, _c = hops
    a.redirect_to = f"{b.base_url}/llama-runtime.zip"
    dest = tmp_path / "runtime.zip"

    with pytest.raises(sl.ArtifactError) as exc_info:
        sl._download(f"{a.base_url}/releases/download/x/llama-runtime.zip", dest)

    assert b.hit_paths == [], "the cleartext hop was dialed for a native library"
    assert not dest.exists() or dest.read_bytes() != _HOPB_BYTES, (
        "hop B's bytes were written to disk")
    msg = str(exc_info.value).lower()
    assert "off https" in msg and "cleartext" in msg
    assert "flaky" not in msg, (
        "reported as a transient network fault - the advice would be to retry")


def test_a_plain_http_caller_is_not_broken_by_the_guard(hops):
    """The rule is DOWNGRADE, not https-only: a caller that started on plain
    http (a user-configured http endpoint) still follows an http -> http
    redirect. Hop A is https here, so the plain hop B drives hop C's payload
    instead: b -> c over http."""
    _a, b, _c = hops
    plain_target = _Stub(_Payload)
    plain_target.body = _HOPC_BYTES
    plain_redirector = _Stub(_Redirector)
    plain_redirector.redirect_to = f"{plain_target.base_url}/payload"
    threads = [_serve(plain_target), _serve(plain_redirector)]
    try:
        req = urllib.request.Request(f"{plain_redirector.base_url}/start")
        with http_ssl.verified_urlopen(req, timeout=10) as r:
            assert r.read() == _HOPC_BYTES
        assert plain_target.hit_paths == ["/payload"]
        assert b.hit_paths == []
    finally:
        for s in (plain_target, plain_redirector):
            s.shutdown()
            s.server_close()
        for t in threads:
            t.join(timeout=5)
