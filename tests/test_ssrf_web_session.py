# SPDX-License-Identifier: AGPL-3.0-or-later
"""SSRF hardening for the web/coder/chat surface.

Covers the fixes from the adversarial SSRF audit of the web plugin and the
shared localm.netpolicy guard it delegates to:

  F1  CGNAT 100.64.0.0/10 (RFC 6598) and other non-globally-routable ranges
      pass an explicit is_* list; they are refused via the `not is_global`
      predicate instead. Public hosts still pass.
  N1  Parser-differential SSRF: a backslash / control char in the authority makes
      urllib.parse read a public host while requests/urllib3 connect to loopback,
      defeating the guard AND the net_allow allowlist. Refused up front.
  6to4 The deprecated 6to4 relay anycast prefix (192.88.99.0/24, 2002::/16) is
      is_global=True yet not an ordinary public host; refused.
  N3  The search backends must pass allow_redirects=False and refuse any 3xx,
      rather than following redirects with no per-hop check_url.
  F2  Chat image_url parts are routed through netpolicy.safe_fetch_bytes
      (check_url + per-hop redirect re-validation + size cap), not a raw
      requests.get, and are reachable from a baseline chat turn.

These pin the behavior so a future refactor cannot silently re-open the holes.
"""

import base64
import io
import socket
from unittest.mock import patch

import pytest

from localm import netpolicy
from localm.netpolicy import NetworkPolicyError, check_url


class _FakeSession:
    """Doubles netpolicy._session_for (the pinned-transport seam) so these tests
    exercise the real check_url + redirect logic without a live socket. get/post
    may each be a fixed response object or a responder callable (url, **kw)."""

    def __init__(self, *, get=None, post=None):
        self._get, self._post = get, post

    def _resolve(self, fn, url, **kw):
        if fn is None:
            raise AssertionError(f"unexpected request to {url}")
        return fn(url, **kw) if callable(fn) else fn

    def get(self, url, **kw):
        return self._resolve(self._get, url, **kw)

    def post(self, url, **kw):
        return self._resolve(self._post, url, **kw)

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture(autouse=True)
def _allow_mode(monkeypatch):
    """net_mode=allow (so check_url runs the IP/host gates, not the off bail) and
    a default config (net_allow_private False, no allow/deny lists)."""
    monkeypatch.setenv("LOCALM_NET_MODE", "allow")
    monkeypatch.setattr("localm.config.load_config", lambda: {})


def _cfg(monkeypatch, **cfg):
    monkeypatch.setattr("localm.config.load_config", lambda: cfg)


# --------------------------------------------------------------------------- #
#  CGNAT / non-global ranges blocked; public hosts still allowed               #
# --------------------------------------------------------------------------- #

class TestF1NonGlobalRanges:
    @pytest.mark.parametrize("host", [
        "100.64.0.1",            # CGNAT (RFC 6598) low end
        "100.127.255.254",       # CGNAT high end
        "1681920001",            # dotless-decimal form of a CGNAT address
    ])
    def test_cgnat_literal_blocked(self, host):
        with pytest.raises(NetworkPolicyError):
            check_url(f"http://{host}/")

    def test_cgnat_dotless_is_really_cgnat(self):
        # guard the test data: 1681920001 must actually be inside 100.64.0.0/10
        import ipaddress
        ip = ipaddress.IPv4Address(1681920001)
        assert ip in ipaddress.ip_network("100.64.0.0/10")

    @pytest.mark.parametrize("url", [
        "http://[::ffff:100.64.0.1]/",   # IPv6-mapped CGNAT
        "http://[::ffff:10.0.0.1]/",      # IPv6-mapped RFC1918
        "http://[::ffff:127.0.0.1]/",     # IPv6-mapped loopback
        "http://[::ffff:169.254.169.254]/",  # IPv6-mapped metadata
    ])
    def test_ipv4_mapped_ipv6_blocked(self, url):
        with pytest.raises(NetworkPolicyError):
            check_url(url)

    @pytest.mark.parametrize("url", [
        "http://[::127.0.0.1]/",          # IPv4-compatible (stdlib: is_global True)
        "http://[64:ff9b::7f00:1]/",       # NAT64 embedding 127.0.0.1
        "http://[64:ff9b::a9fe:a9fe]/",    # NAT64 embedding 169.254.169.254
        "http://224.0.0.1/",               # multicast
    ])
    def test_deprecated_and_special_forms_still_blocked(self, url):
        """The `not is_global` union must NOT regress the forms the stdlib marks
        is_global=True but that still route internally - is_reserved/is_multicast
        catch those, so the union is what is asserted, not is_global alone."""
        with pytest.raises(NetworkPolicyError):
            check_url(url)

    @pytest.mark.parametrize("url", [
        "http://8.8.8.8/",
        "http://1.1.1.1/",
    ])
    def test_public_ip_literals_allowed(self, url):
        check_url(url)   # no raise

    def test_public_hostname_allowed(self, monkeypatch):
        monkeypatch.setattr(
            "socket.getaddrinfo",
            lambda host, port, *a, **k: [(2, 1, 6, "", ("93.184.216.34", 0))])
        check_url("https://example.com/")   # no raise

    def test_public_numeric_ip_not_false_blocked(self):
        # 93.184.216.34 in dotless decimal must still pass (no over-block)
        dotless = str(int.from_bytes(socket.inet_aton("93.184.216.34"), "big"))
        check_url(f"https://{dotless}/")


# --------------------------------------------------------------------------- #
#  6to4 relay anycast                                                          #
# --------------------------------------------------------------------------- #

class TestSixToFourAnycast:
    @pytest.mark.parametrize("url", [
        "http://192.88.99.1/",   # 6to4 relay anycast (IPv4)
        "http://[2002::1]/",      # 6to4 (IPv6)
    ])
    def test_6to4_blocked(self, url):
        with pytest.raises(NetworkPolicyError):
            check_url(url)


# --------------------------------------------------------------------------- #
#  Parser-differential (backslash / control char in authority)                #
# --------------------------------------------------------------------------- #

class TestN1ParserDifferential:
    @pytest.mark.parametrize("url", [
        r"http://127.0.0.1\@expected.com/",
        r"http://169.254.169.254\@example.com/",
        r"http://localhost\@example.com/",
        "http://127.0.0.1\t@example.com/",
        "http://127.0.0.1\r\n@example.com/",
        "http://example.com\\../",
    ])
    def test_backslash_or_control_char_refused(self, url):
        with pytest.raises(NetworkPolicyError, match="backslash|control"):
            check_url(url)

    def test_backslash_defeats_net_allow_is_blocked(self, monkeypatch):
        """The PoC variant that also bypassed the allowlist: with net_allow set,
        the backslash trick must STILL be refused (not validated as the allowed
        host while connecting to loopback)."""
        _cfg(monkeypatch, net_allow=["allowed.example"], net_allow_private=False)
        with pytest.raises(NetworkPolicyError, match="backslash|control"):
            check_url(r"http://127.0.0.1\@allowed.example/")

    def test_clean_url_with_no_backslash_passes(self):
        check_url("https://8.8.8.8/path?q=a%5Cb")   # %5C is fine; raw \\ is not


# --------------------------------------------------------------------------- #
#  Search backends refuse redirects (no unchecked per-hop bounce)             #
# --------------------------------------------------------------------------- #

class _FakeResp:
    def __init__(self, *, status=200, location=None, json_body=None, text=""):
        self.status_code = status
        self.is_redirect = location is not None
        self.is_permanent_redirect = False
        self.headers = {"Location": location} if location else {}
        self._json = json_body if json_body is not None else {"results": []}
        self.text = text

    def raise_for_status(self):
        if self.status_code >= 400:
            raise AssertionError(f"HTTP {self.status_code}")

    def json(self):
        return self._json


class TestN3SearchRedirects:
    def test_searxng_redirect_refused(self, monkeypatch):
        _cfg(monkeypatch, net_search_url="https://searx.example", net_mode="allow")
        monkeypatch.setenv("LOCALM_NET_MODE", "allow")
        monkeypatch.setattr(
            "socket.getaddrinfo",
            lambda *a, **k: [(2, 1, 6, "", ("93.184.216.34", 0))])
        with patch("localm.netpolicy._session_for", return_value=_FakeSession(
                get=_FakeResp(status=302, location="http://127.0.0.1/"))):
            with pytest.raises(NetworkPolicyError, match="redirect"):
                netpolicy.web_search("hello")

    def test_ddg_redirect_refused(self, monkeypatch):
        monkeypatch.setenv("LOCALM_NET_MODE", "allow")
        with patch("localm.netpolicy._session_for", return_value=_FakeSession(
                post=_FakeResp(status=302, location="http://127.0.0.1/"))):
            with pytest.raises(NetworkPolicyError, match="redirect"):
                netpolicy.web_search("hello")

    def test_searxng_passes_allow_redirects_false(self, monkeypatch):
        _cfg(monkeypatch, net_search_url="https://searx.example")
        monkeypatch.setenv("LOCALM_NET_MODE", "allow")
        monkeypatch.setattr(
            "socket.getaddrinfo",
            lambda *a, **k: [(2, 1, 6, "", ("93.184.216.34", 0))])
        captured = {}

        def fake_get(url, **kw):
            captured.update(kw)
            return _FakeResp(json_body={"results": [
                {"title": "T", "url": "https://ok.example", "content": "c"}]})

        with patch("localm.netpolicy._session_for",
                   return_value=_FakeSession(get=fake_get)):
            results = netpolicy.web_search("hello")
        assert captured.get("allow_redirects") is False
        assert results and results[0]["url"] == "https://ok.example"


# --------------------------------------------------------------------------- #
#  Chat image_url fetch is policy-checked                                      #
# --------------------------------------------------------------------------- #

def _png_bytes():
    Image = pytest.importorskip("PIL.Image")
    buf = io.BytesIO()
    Image.new("RGB", (2, 2), (255, 0, 0)).save(buf, "PNG")
    return buf.getvalue()


class _FakeBytesResp:
    """Minimal streaming response for safe_fetch_bytes."""
    def __init__(self, body: bytes, content_type="image/png"):
        self._body = body
        self.is_redirect = False
        self.is_permanent_redirect = False
        self.headers = {"Content-Type": content_type}

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size=65536):
        yield self._body

    def close(self):
        pass


class TestF2ImageUrlSSRF:
    @pytest.mark.parametrize("url", [
        "http://127.0.0.1/x.png",
        "http://169.254.169.254/latest/meta-data/",
        "http://100.64.0.1/x.png",
        "file:///etc/passwd",
        r"http://127.0.0.1\@example.com/x.png",
    ])
    def test_internal_image_url_refused(self, url):
        pytest.importorskip("PIL.Image")
        from localm.inference.media import decode_image_url
        with pytest.raises(NetworkPolicyError):
            decode_image_url(url)

    def test_data_uri_still_decodes(self):
        pytest.importorskip("PIL.Image")
        from localm.inference.media import decode_image_url
        uri = "data:image/png;base64," + base64.b64encode(_png_bytes()).decode()
        img = decode_image_url(uri)
        assert img.size == (2, 2)

    def test_public_image_url_fetched_through_policy(self, monkeypatch):
        pytest.importorskip("PIL.Image")
        from localm.inference.media import decode_image_url
        monkeypatch.setattr(
            "socket.getaddrinfo",
            lambda host, port, *a, **k: [(2, 1, 6, "", ("93.184.216.34", 0))])
        with patch("localm.netpolicy._session_for",
                   return_value=_FakeSession(get=_FakeBytesResp(_png_bytes()))):
            img = decode_image_url("https://example.com/pic.png")
        assert img.size == (2, 2)


# --------------------------------------------------------------------------- #
#  safe_fetch_bytes: binary integrity (no UTF-8 corruption)                    #
# --------------------------------------------------------------------------- #

def test_safe_fetch_bytes_returns_raw_bytes(monkeypatch):
    monkeypatch.setenv("LOCALM_NET_MODE", "allow")
    monkeypatch.setattr(
        "socket.getaddrinfo",
        lambda host, port, *a, **k: [(2, 1, 6, "", ("93.184.216.34", 0))])
    raw = bytes(range(256))   # includes non-UTF-8 bytes
    with patch("localm.netpolicy._session_for",
               return_value=_FakeSession(get=_FakeBytesResp(raw, "image/png"))):
        final_url, ctype, body = netpolicy.safe_fetch_bytes("https://example.com/x")
    assert body == raw          # exact bytes, not utf-8 mangled
    assert ctype == "image/png"
