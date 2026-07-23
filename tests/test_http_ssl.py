# SPDX-License-Identifier: AGPL-3.0-or-later
"""The shared outbound-HTTPS opener (localm/http_ssl.py) and a regression guard
that every outbound client actually goes through it.

Background: setup-llama, `localm update` (proxy check + release-CDN download), the
issues list, and the bug-report upload all did raw urllib HTTPS with no explicit SSL
context, so they verified against whatever the machine's OS cert store happened to
have cached. Python's OpenSSL on Windows does not keep that store current, so a
fresh box failed every call with CERTIFICATE_VERIFY_FAILED (#765). The fix then
hardcoded a certifi-only context - which verifies fine on a fresh box, but a
corporate/security-product TLS-intercepting proxy's re-signed certificate is not in
certifi's public root list either, so that DIDN'T help a managed machine behind one
(#825 - confirmed live: uv's own equivalent bundled-root default fails identically
on such a network with "invalid peer certificate: UnknownIssuer").

verified_urlopen() tries the platform's NATIVE certificate store first (the same
trust a browser, or an IT-provisioned proxy root, already has - so a managed
machine behind a proxy like that verifies on the very first attempt, nothing ever
shown) and falls back to certifi ONLY on a certificate-verification failure
specifically (rescuing the original #765 fresh-Windows case). These tests lock in
that ordering, that a non-certificate failure never triggers the fallback, and that
a certificate failure surviving both attempts is never silently swallowed.
(setup-llama's own two call sites are guarded in tests/test_setup_llama_backends.py.)
"""
from __future__ import annotations

import ssl
import sys
import types
import urllib.error
import urllib.request

import pytest

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


def _cert_error(host="example.com"):
    reason = ssl.SSLCertVerificationError(
        f"[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed for {host}")
    return urllib.error.URLError(reason)


def test_verified_urlopen_succeeds_on_native_store_first_try(monkeypatch):
    # The common case (a plain machine, or one behind a proxy whose root IT
    # already provisioned into the OS store): ONE call, no fallback, no error
    # ever surfaced - "do it right from the start", not "fail then recover".
    calls = []

    def fake_urlopen(req, timeout=None, context=None):
        calls.append(context)
        return _Resp(200, b"ok")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    req = urllib.request.Request("https://example.com/")
    with http_ssl.verified_urlopen(req, timeout=5) as r:
        assert r.read() == b"ok"
    assert len(calls) == 1, "must not retry when the first attempt succeeds"
    assert calls[0].verify_mode == ssl.CERT_REQUIRED
    # The native context carries no explicit cafile override - it is exactly
    # ssl.create_default_context(), not certifi's.


def test_verified_urlopen_falls_back_to_certifi_on_cert_failure(monkeypatch):
    # Rescues the original #765 case: a freshly-imaged Windows box whose ROOT
    # store has not cached a legitimate CA chain yet. First (native) attempt
    # fails verification; second (certifi) attempt succeeds.
    contexts = []

    def fake_urlopen(req, timeout=None, context=None):
        contexts.append(context)
        if len(contexts) == 1:
            raise _cert_error()
        return _Resp(200, b"ok-via-certifi")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    req = urllib.request.Request("https://example.com/")
    with http_ssl.verified_urlopen(req, timeout=5) as r:
        assert r.read() == b"ok-via-certifi"
    assert len(contexts) == 2
    assert contexts[0].verify_mode == ssl.CERT_REQUIRED
    assert contexts[1].verify_mode == ssl.CERT_REQUIRED
    assert contexts[1].get_ca_certs(), "the fallback must actually be certifi-backed"


def test_verified_urlopen_never_downgrades_to_unverified_when_certifi_missing(monkeypatch):
    # If certifi is unavailable at the moment the fallback would run, the ORIGINAL
    # certificate failure must propagate - never a silent downgrade to an
    # unverified handshake (a security step that fails must not report success).
    def fake_urlopen(req, timeout=None, context=None):
        raise _cert_error()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    broken = types.SimpleNamespace(
        where=lambda: (_ for _ in ()).throw(RuntimeError("no certifi")))
    monkeypatch.setitem(sys.modules, "certifi", broken)
    req = urllib.request.Request("https://example.com/")
    with pytest.raises(urllib.error.URLError) as exc_info:
        http_ssl.verified_urlopen(req, timeout=5)
    assert isinstance(exc_info.value.reason, ssl.SSLCertVerificationError)


def test_verified_urlopen_certifi_also_fails_propagates_original_cert_error(monkeypatch):
    # A certificate failure that survives BOTH attempts (native and certifi) is a
    # real, non-recoverable failure - it must reach the caller, not vanish.
    calls = {"n": 0}

    def fake_urlopen(req, timeout=None, context=None):
        calls["n"] += 1
        raise _cert_error()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    req = urllib.request.Request("https://example.com/")
    with pytest.raises(urllib.error.URLError) as exc_info:
        http_ssl.verified_urlopen(req, timeout=5)
    assert calls["n"] == 2, "must try native then certifi, not loop or give up early"
    assert isinstance(exc_info.value.reason, ssl.SSLCertVerificationError)


def test_verified_urlopen_non_certificate_failure_never_retries(monkeypatch):
    # A non-certificate failure (DNS, connection refused, timeout, ...) is not
    # fixable by switching CA bundles - retrying would just waste time and could
    # mask what actually went wrong. Must propagate immediately, unmodified.
    calls = {"n": 0}

    def fake_urlopen(req, timeout=None, context=None):
        calls["n"] += 1
        raise urllib.error.URLError(OSError("Name or service not known"))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    req = urllib.request.Request("https://example.com/")
    with pytest.raises(urllib.error.URLError) as exc_info:
        http_ssl.verified_urlopen(req, timeout=5)
    assert calls["n"] == 1, "a non-certificate failure must not trigger a retry"
    assert not isinstance(exc_info.value.reason, ssl.SSLCertVerificationError)


def test_verified_urlopen_http_error_passes_through_unmodified(monkeypatch):
    # HTTPError is a URLError subclass but represents a real server response
    # (e.g. 404/500), not a transport/certificate problem - must not be
    # misidentified as a certificate failure or retried.
    def fake_urlopen(req, timeout=None, context=None):
        raise urllib.error.HTTPError("https://example.com/", 404, "Not Found", {}, None)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    req = urllib.request.Request("https://example.com/")
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        http_ssl.verified_urlopen(req, timeout=5)
    assert exc_info.value.code == 404


def test_verified_urlopen_uses_build_opener_when_handlers_given(monkeypatch):
    # updater.py needs its own https-only-redirect handler installed alongside
    # the HTTPS handler - verify the handler is actually threaded through, and
    # that plain urlopen (which would silently ignore it) is NOT used instead.
    seen = {}

    class _MarkerHandler(urllib.request.BaseHandler):
        pass

    class _FakeOpener:
        def open(self, req, timeout=None):
            return _Resp(200, b"ok")

    def fake_build_opener(*handlers):
        seen["handlers"] = handlers
        return _FakeOpener()

    def fail_if_called(*a, **k):
        raise AssertionError("plain urlopen must not be used when handlers are given")

    monkeypatch.setattr(urllib.request, "build_opener", fake_build_opener)
    monkeypatch.setattr(urllib.request, "urlopen", fail_if_called)
    req = urllib.request.Request("https://example.com/")
    with http_ssl.verified_urlopen(req, timeout=5, handlers=(_MarkerHandler,)) as r:
        assert r.read() == b"ok"
    assert _MarkerHandler in seen["handlers"]
    assert any(isinstance(h, urllib.request.HTTPSHandler) for h in seen["handlers"])


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
