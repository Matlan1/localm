# SPDX-License-Identifier: AGPL-3.0-or-later
"""The shared outbound-HTTPS SSL context (localm/http_ssl.py) and a regression
guard that every outbound client actually PRESENTS a verifying context.

Background: setup-llama, `localm update` (proxy check + release-CDN download), the
issues list, and the bug-report upload all did raw urllib HTTPS with no SSL
context, so they verified against the machine's OS cert store. Python's OpenSSL on
Windows does not keep that store current, so a fresh box failed every call with
CERTIFICATE_VERIFY_FAILED. These tests lock in the certifi-backed context and that
each client passes it. (setup-llama's own two call sites are guarded in
tests/test_setup_llama_backends.py.)
"""
from __future__ import annotations

import ssl
import sys
import types
import urllib.request

from localm import http_ssl


class _Resp:
    """A minimal urlopen/opener response: one-shot body, works for a single
    ``read()`` (proxy/bugreport) and for a chunked ``read(n)`` loop (updater)."""
    def __init__(self, status=200, body=b"{}"):
        self.status = status
        self._body = body
        self._done = False

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self, n=-1):
        if self._done:
            return b""
        self._done = True
        return self._body


def test_client_ssl_context_verifies_with_certifi():
    ctx = http_ssl.client_ssl_context()
    assert ctx.verify_mode == ssl.CERT_REQUIRED
    assert ctx.check_hostname is True
    assert ctx.get_ca_certs(), "expected a non-empty CA bundle (certifi)"


def test_client_ssl_context_fallback_never_disables_verification(monkeypatch):
    # If certifi is somehow unavailable, the context must STILL verify (fall back
    # to the stdlib default) - never a downgrade to an unverified handshake.
    broken = types.SimpleNamespace(
        where=lambda: (_ for _ in ()).throw(RuntimeError("no certifi")))
    monkeypatch.setitem(sys.modules, "certifi", broken)
    ctx = http_ssl.client_ssl_context()
    assert ctx.verify_mode == ssl.CERT_REQUIRED
    assert ctx.check_hostname is True


def test_proxy_default_opener_passes_verifying_context(monkeypatch):
    from localm import _proxy
    seen = {}

    def fake_urlopen(req, timeout=None, context=None):
        seen["context"] = context
        return _Resp(200, b"{}")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    status, _raw = _proxy._default_opener("GET", "https://proxy.example/x", None, {}, 5.0)
    assert status == 200
    assert isinstance(seen["context"], ssl.SSLContext)
    assert seen["context"].verify_mode == ssl.CERT_REQUIRED


def test_bugreport_opener_passes_verifying_context(monkeypatch):
    from localm import bugreport
    seen = {}

    def fake_urlopen(req, timeout=None, context=None):
        seen["context"] = context
        return _Resp(200, b"{}")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    bugreport.upload_report("t", "b", url="https://proxy.example/report", token=None)
    assert isinstance(seen["context"], ssl.SSLContext)
    assert seen["context"].verify_mode == ssl.CERT_REQUIRED


def test_updater_download_uses_verifying_https_context(monkeypatch, tmp_path):
    # The update download follows a 302 to the release CDN; the HTTPSHandler it
    # builds MUST carry a verifying context, or a fresh box fails the CDN hop.
    from localm import updater
    monkeypatch.setattr(updater, "endpoint", lambda: ("https://updates.example", None))
    seen = {}

    def capture_https(*a, context=None, **k):
        seen["context"] = context
        return object()          # build_opener is faked below; the handler is unused

    class _FakeOpener:
        def open(self, req, timeout=None):
            return _Resp(200, b"payload")

    monkeypatch.setattr(urllib.request, "HTTPSHandler", capture_https)
    monkeypatch.setattr(urllib.request, "build_opener", lambda *h: _FakeOpener())
    dest = tmp_path / "build.zip"
    updater.download(123, dest, timeout=5.0)
    assert dest.read_bytes() == b"payload"
    assert isinstance(seen["context"], ssl.SSLContext)
    assert seen["context"].verify_mode == ssl.CERT_REQUIRED
