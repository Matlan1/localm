# SPDX-License-Identifier: AGPL-3.0-or-later
"""The shared outbound-HTTPS opener (localm/http_ssl.py) and a regression guard
that every outbound client actually goes through it.

setup-llama, `localm update` (proxy check + release-CDN download), the issues list,
and the bug-report upload all reach HTTPS through verified_urlopen(). It tries the
platform's NATIVE certificate store first (the same trust a browser, or an
IT-provisioned proxy root, already has) and falls back to certifi ONLY on a
certificate-verification failure specifically. A raw urllib call with no explicit
SSL context verifies against whatever the machine's OS cert store happens to have
cached, and Python's OpenSSL on Windows does not keep that store current; a
certifi-only context has the opposite gap, since a corporate TLS-intercepting
proxy's re-signed certificate is not in certifi's public root list.

These tests lock in that ordering, that a non-certificate failure never triggers
the fallback, and that a certificate failure surviving both attempts is never
swallowed.

They also lock the WIRING of the redirect guard: HttpsOnlyRedirect is installed
even when the caller passes no handlers, and a caller's own redirect policy
REPLACES it instead of joining it.

NOTE THE SEAM: verified_urlopen does not call urllib.request.urlopen (it builds an
opener to install a handler), so patching urlopen intercepts nothing and lets the
real network through. Use patch_https_transport.
"""
from __future__ import annotations

import ssl
import sys
import types
import urllib.error
import urllib.request

import pytest

from localm import http_ssl
from tests._fake_https import patch_https_transport


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
    # The common case (a machine whose OS store already carries the chain):
    # ONE call, no fallback, no error surfaced.
    calls = []

    def fake_urlopen(req, timeout=None, context=None):
        calls.append(context)
        return _Resp(200, b"ok")

    patch_https_transport(monkeypatch, fake_urlopen)
    req = urllib.request.Request("https://example.com/")
    with http_ssl.verified_urlopen(req, timeout=5) as r:
        assert r.read() == b"ok"
    assert len(calls) == 1, "must not retry when the first attempt succeeds"
    assert calls[0].verify_mode == ssl.CERT_REQUIRED
    # The native context carries no explicit cafile override - it is exactly
    # ssl.create_default_context(), not certifi's.


def test_verified_urlopen_falls_back_to_certifi_on_cert_failure(monkeypatch):
    # A box whose ROOT store has not cached a legitimate CA chain: the native
    # attempt fails verification and the certifi attempt succeeds.
    contexts = []

    def fake_urlopen(req, timeout=None, context=None):
        contexts.append(context)
        if len(contexts) == 1:
            raise _cert_error()
        return _Resp(200, b"ok-via-certifi")

    seen = patch_https_transport(monkeypatch, fake_urlopen)
    req = urllib.request.Request("https://example.com/")
    with http_ssl.verified_urlopen(req, timeout=5) as r:
        assert r.read() == b"ok-via-certifi"
    assert len(contexts) == 2
    assert contexts[0].verify_mode == ssl.CERT_REQUIRED
    assert contexts[1].verify_mode == ssl.CERT_REQUIRED
    assert contexts[1].get_ca_certs(), "the fallback must actually be certifi-backed"
    # The SECOND build_opener call, so the redirect guard is on the RETRY
    # opener too.
    assert http_ssl.HttpsOnlyRedirect in seen["handlers"]


def test_verified_urlopen_never_downgrades_to_unverified_when_certifi_missing(monkeypatch):
    # If certifi is unavailable at the moment the fallback would run, the
    # ORIGINAL certificate failure propagates; there is no downgrade to an
    # unverified handshake.
    def fake_urlopen(req, timeout=None, context=None):
        raise _cert_error()

    patch_https_transport(monkeypatch, fake_urlopen)
    broken = types.SimpleNamespace(
        where=lambda: (_ for _ in ()).throw(RuntimeError("no certifi")))
    monkeypatch.setitem(sys.modules, "certifi", broken)
    req = urllib.request.Request("https://example.com/")
    with pytest.raises(urllib.error.URLError) as exc_info:
        http_ssl.verified_urlopen(req, timeout=5)
    assert isinstance(exc_info.value.reason, ssl.SSLCertVerificationError)


def test_verified_urlopen_certifi_also_fails_propagates_original_cert_error(monkeypatch):
    # A certificate failure that survives BOTH attempts reaches the caller.
    calls = {"n": 0}

    def fake_urlopen(req, timeout=None, context=None):
        calls["n"] += 1
        raise _cert_error()

    patch_https_transport(monkeypatch, fake_urlopen)
    req = urllib.request.Request("https://example.com/")
    with pytest.raises(urllib.error.URLError) as exc_info:
        http_ssl.verified_urlopen(req, timeout=5)
    assert calls["n"] == 2, "must try native then certifi, not loop or give up early"
    assert isinstance(exc_info.value.reason, ssl.SSLCertVerificationError)


def test_verified_urlopen_non_certificate_failure_never_retries(monkeypatch):
    # A non-certificate failure (DNS, connection refused, timeout, ...)
    # propagates immediately, unmodified.
    calls = {"n": 0}

    def fake_urlopen(req, timeout=None, context=None):
        calls["n"] += 1
        raise urllib.error.URLError(OSError("Name or service not known"))

    patch_https_transport(monkeypatch, fake_urlopen)
    req = urllib.request.Request("https://example.com/")
    with pytest.raises(urllib.error.URLError) as exc_info:
        http_ssl.verified_urlopen(req, timeout=5)
    assert calls["n"] == 1, "a non-certificate failure must not trigger a retry"
    assert not isinstance(exc_info.value.reason, ssl.SSLCertVerificationError)


def test_verified_urlopen_http_error_passes_through_unmodified(monkeypatch):
    # HTTPError is a URLError subclass but represents a real server response
    # (404/500), so it is neither treated as a certificate failure nor retried.
    def fake_urlopen(req, timeout=None, context=None):
        raise urllib.error.HTTPError("https://example.com/", 404, "Not Found", {}, None)

    patch_https_transport(monkeypatch, fake_urlopen)
    req = urllib.request.Request("https://example.com/")
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        http_ssl.verified_urlopen(req, timeout=5)
    assert exc_info.value.code == 404


def test_verified_urlopen_threads_extra_handlers_through(monkeypatch):
    # A caller's own handler is installed alongside the HTTPS handler, and
    # plain urlopen is not used instead.
    class _MarkerHandler(urllib.request.BaseHandler):
        pass

    def fail_if_called(*a, **k):
        raise AssertionError("plain urlopen must never be used")

    monkeypatch.setattr(urllib.request, "urlopen", fail_if_called)
    seen = patch_https_transport(
        monkeypatch, lambda req, timeout=None, context=None: _Resp(200, b"ok"))
    req = urllib.request.Request("https://example.com/")
    with http_ssl.verified_urlopen(req, timeout=5, handlers=(_MarkerHandler,)) as r:
        assert r.read() == b"ok"
    assert _MarkerHandler in seen["handlers"]


def test_verified_urlopen_installs_the_downgrade_guard_by_default(monkeypatch):
    # A caller that passes no handlers at all still gets HttpsOnlyRedirect, not
    # urllib's permissive default.
    seen = patch_https_transport(
        monkeypatch, lambda req, timeout=None, context=None: _Resp(200, b"ok"))
    req = urllib.request.Request("https://example.com/")
    with http_ssl.verified_urlopen(req, timeout=5) as r:
        assert r.read() == b"ok"
    assert http_ssl.HttpsOnlyRedirect in seen["handlers"]


def test_a_caller_supplied_redirect_policy_replaces_the_default(monkeypatch):
    # comfy_client refuses EVERY hop, not just a downgrade, and its handler is
    # the only redirect handler on the opener.
    class _RefuseEverything(urllib.request.HTTPRedirectHandler):
        pass

    seen = patch_https_transport(
        monkeypatch, lambda req, timeout=None, context=None: _Resp(200, b"ok"))
    req = urllib.request.Request("https://example.com/")
    with http_ssl.verified_urlopen(req, timeout=5, handlers=(_RefuseEverything,)) as r:
        assert r.read() == b"ok"
    assert _RefuseEverything in seen["handlers"]
    assert http_ssl.HttpsOnlyRedirect not in seen["handlers"]


def test_proxy_default_opener_passes_verifying_context(monkeypatch):
    from localm import _proxy
    seen = {}

    def fake_urlopen(req, timeout=None, context=None):
        seen["context"] = context
        return _Resp(200, b"{}")

    patch_https_transport(monkeypatch, fake_urlopen)
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

    patch_https_transport(monkeypatch, fake_urlopen)
    bugreport.upload_report("t", "b", url="https://proxy.example/report", token=None)
    assert isinstance(seen["context"], ssl.SSLContext)
    assert seen["context"].verify_mode == ssl.CERT_REQUIRED


def test_updater_download_uses_verifying_https_context(monkeypatch, tmp_path):
    # The update download follows a 302 to the release CDN, so the HTTPSHandler
    # it builds carries a verifying context.
    from localm import updater
    monkeypatch.setattr(updater, "endpoint", lambda: ("https://updates.example", None))
    seen = patch_https_transport(
        monkeypatch, lambda req, timeout=None, context=None: _Resp(200, b"payload"))
    dest = tmp_path / "build.zip"
    updater.download(123, dest, timeout=5.0)
    assert dest.read_bytes() == b"payload"
    assert isinstance(seen["context"], ssl.SSLContext)
    assert seen["context"].verify_mode == ssl.CERT_REQUIRED
    # updater.py has no _HttpsOnlyRedirect of its own; the shared guard is on
    # the opener it builds.
    assert http_ssl.HttpsOnlyRedirect in seen["handlers"]
