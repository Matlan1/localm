# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for localm.netpolicy - the network policy for model-initiated requests."""


import logging
import urllib.parse

import pytest

from localm.netpolicy import (
    NetworkPolicyError,
    _domain_list,
    _host_matches,
    check_url,
    format_results,
    html_to_text,
    network_mode,
    safe_fetch,
    safe_fetch_bytes,
    web_search,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Tests control the mode explicitly; the machine's env must not leak in."""
    monkeypatch.delenv("LOCALM_NET_MODE", raising=False)


def _with_config(monkeypatch, cfg: dict):
    monkeypatch.setattr("localm.config.load_config", lambda: cfg)


# ------------------------------------------------------------------ #
#  Mode resolution                                                    #
# ------------------------------------------------------------------ #

class TestNetworkMode:
    def test_env_overrides_config(self, monkeypatch):
        _with_config(monkeypatch, {"net_mode": "allow"})
        monkeypatch.setenv("LOCALM_NET_MODE", "off")
        assert network_mode() == "off"

    def test_config_value_used(self, monkeypatch):
        _with_config(monkeypatch, {"net_mode": "allow"})
        assert network_mode() == "allow"

    def test_invalid_values_fall_back_to_ask(self, monkeypatch):
        _with_config(monkeypatch, {"net_mode": "yolo"})
        assert network_mode() == "ask"
        monkeypatch.setenv("LOCALM_NET_MODE", "nonsense")
        assert network_mode() == "ask"   # bad env falls through to config→ask

    def test_default_is_ask(self, monkeypatch):
        _with_config(monkeypatch, {})
        assert network_mode() == "ask"

    def test_config_read_failure_resolves_off(self, monkeypatch, caplog):
        # An unreadable config must NOT downgrade an explicit net_mode="off"
        # kill switch to "ask". It resolves to "off" and surfaces a warning.
        def boom():
            raise OSError("config unreadable")
        monkeypatch.setattr("localm.config.load_config", boom)
        with caplog.at_level(logging.WARNING, logger="localm"):
            assert network_mode() == "off"
        assert any("net_mode" in r.message and "off" in r.message.lower()
                   for r in caplog.records), \
            "config-read failure resolved silently (no warning)"

    def test_config_read_failure_env_override_still_wins(self, monkeypatch):
        # The env var short-circuits before config is read, so an explicit env
        # value is still honored even when the config file is unreadable.
        monkeypatch.setenv("LOCALM_NET_MODE", "allow")

        def boom():
            raise OSError("config unreadable")
        monkeypatch.setattr("localm.config.load_config", boom)
        assert network_mode() == "allow"


# ------------------------------------------------------------------ #
#  Domain rules                                                       #
# ------------------------------------------------------------------ #

class TestDomainRules:
    def test_domain_list_accepts_list_and_csv(self):
        assert _domain_list(["A.com", " b.org "]) == ["a.com", "b.org"]
        assert _domain_list("a.com, b.org") == ["a.com", "b.org"]
        assert _domain_list("*.a.com") == ["a.com"]
        assert _domain_list(None) == []
        assert _domain_list("") == []

    def test_host_matches_suffix(self):
        assert _host_matches("example.com", "example.com")
        assert _host_matches("api.example.com", "example.com")
        assert not _host_matches("notexample.com", "example.com")
        assert not _host_matches("example.com.evil.net", "example.com")

    def test_deny_wins(self, monkeypatch):
        _with_config(monkeypatch, {
            "net_mode": "allow",
            "net_allow": ["example.com"],
            "net_deny": ["example.com"],
        })
        with pytest.raises(NetworkPolicyError, match="deny list"):
            check_url("https://example.com/page")

    def test_allow_list_restricts(self, monkeypatch):
        _with_config(monkeypatch, {
            "net_mode": "allow",
            "net_allow": ["example.com"],
            "net_allow_private": True,   # skip DNS in this test
        })
        check_url("https://api.example.com/x")    # passes
        with pytest.raises(NetworkPolicyError, match="allow list"):
            check_url("https://other.org/x")

    def test_empty_allow_means_any(self, monkeypatch):
        _with_config(monkeypatch, {"net_mode": "allow",
                                   "net_allow_private": True})
        check_url("https://anything.example/x")


# ------------------------------------------------------------------ #
#  check_url: mode, scheme, SSRF                                      #
# ------------------------------------------------------------------ #

class TestCheckUrl:
    def test_mode_off_blocks_everything(self, monkeypatch):
        _with_config(monkeypatch, {"net_mode": "off"})
        with pytest.raises(NetworkPolicyError, match="net_mode=off"):
            check_url("https://example.com")

    @pytest.mark.parametrize("url", [
        "file:///etc/passwd",
        "ftp://example.com/x",
        "data:text/plain;base64,aGk=",
        "javascript:alert(1)",
    ])
    def test_non_http_schemes_rejected(self, monkeypatch, url):
        _with_config(monkeypatch, {"net_mode": "allow"})
        with pytest.raises(NetworkPolicyError, match="http/https"):
            check_url(url)

    @pytest.mark.parametrize("ip", [
        "127.0.0.1",        # loopback
        "10.1.2.3",         # private
        "172.16.0.9",       # private
        "192.168.1.1",      # private (router admin)
        "169.254.169.254",  # link-local / cloud metadata
        "0.0.0.0",          # unspecified
    ])
    def test_private_addresses_blocked(self, monkeypatch, ip):
        _with_config(monkeypatch, {"net_mode": "allow"})
        monkeypatch.setattr(
            "socket.getaddrinfo",
            lambda host, port, *a, **k: [(2, 1, 6, "", (ip, 0))])
        with pytest.raises(NetworkPolicyError, match="non-public"):
            check_url("https://innocent-looking.example/")

    def test_private_allowed_when_configured(self, monkeypatch):
        _with_config(monkeypatch, {"net_mode": "allow",
                                   "net_allow_private": True})
        monkeypatch.setattr(
            "socket.getaddrinfo",
            lambda host, port, *a, **k: [(2, 1, 6, "", ("127.0.0.1", 0))])
        check_url("http://localhost:8188/")   # no raise

    def test_public_address_passes(self, monkeypatch):
        _with_config(monkeypatch, {"net_mode": "allow"})
        monkeypatch.setattr(
            "socket.getaddrinfo",
            lambda host, port, *a, **k: [(2, 1, 6, "", ("93.184.216.34", 0))])
        check_url("https://example.com/")

    def test_unresolvable_host_passes_policy(self, monkeypatch):
        import socket as _socket
        _with_config(monkeypatch, {"net_mode": "allow"})

        def boom(host, port, *a, **k):
            raise _socket.gaierror("no such host")
        monkeypatch.setattr("socket.getaddrinfo", boom)
        check_url("https://does-not-exist.example/")   # fetch will fail later

    def test_config_read_failure_refuses_regardless_of_env_mode(self, monkeypatch):
        # net_mode and net_deny/net_allow come from ONE config read: an env
        # LOCALM_NET_MODE=allow must not short-circuit the mode check before the
        # config is touched, or a config-read failure would resolve to an empty
        # net_deny/net_allow instead of a refusal, letting an explicitly denied
        # host through.
        monkeypatch.setenv("LOCALM_NET_MODE", "allow")
        monkeypatch.setattr(
            "socket.getaddrinfo",
            lambda host, port, *a, **k: [(2, 1, 6, "", ("93.184.216.34", 0))])

        def boom():
            raise OSError("config unreadable")
        monkeypatch.setattr("localm.config.load_config", boom)
        with pytest.raises(NetworkPolicyError):
            check_url("https://denied.example/")


# ------------------------------------------------------------------ #
#  safe_fetch: redirects re-validated, size caps                      #
# ------------------------------------------------------------------ #

class _FakeResponse:
    def __init__(self, *, status=200, headers=None, body=b"", redirect=None):
        self.status_code = status
        self.headers = headers or {}
        if redirect:
            self.headers["Location"] = redirect
        self._body = body

    @property
    def is_redirect(self):
        return self.status_code in (301, 302, 303, 307, 308) and \
            "Location" in self.headers

    @property
    def is_permanent_redirect(self):
        return False

    def iter_content(self, chunk_size=65536):
        for i in range(0, len(self._body), chunk_size):
            yield self._body[i:i + chunk_size]

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def close(self):
        pass


class _FakeSession:
    """Doubles netpolicy._session_for - the pinned-transport seam - so tests
    exercise the real fetch/redirect logic without opening a live socket."""

    def __init__(self, responder):
        self._responder = responder      # (method, url, **kw) -> response

    def get(self, url, **kw):
        return self._responder("GET", url, **kw)

    def post(self, url, **kw):
        return self._responder("POST", url, **kw)

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _patch_session(monkeypatch, *, get=None, post=None):
    """Route netpolicy's pinned session to fake get/post responders."""
    import localm.netpolicy as netpolicy

    def responder(method, url, **kw):
        fn = get if method == "GET" else post
        if fn is None:
            raise AssertionError(f"unexpected {method} to {url}")
        return fn(url, **kw)
    monkeypatch.setattr(netpolicy, "_session_for",
                        lambda url: _FakeSession(responder))


class TestSafeFetch:
    def _public_dns(self, monkeypatch):
        monkeypatch.setattr(
            "socket.getaddrinfo",
            lambda host, port, *a, **k: [(2, 1, 6, "", ("93.184.216.34", 0))])

    def test_basic_fetch(self, monkeypatch):
        _with_config(monkeypatch, {"net_mode": "allow"})
        self._public_dns(monkeypatch)
        _patch_session(monkeypatch, get=lambda url, **kw: _FakeResponse(
            headers={"Content-Type": "text/plain"}, body=b"hello"))
        final, ctype, text = safe_fetch("https://example.com/a")
        assert (final, text) == ("https://example.com/a", "hello")
        assert "text/plain" in ctype

    def test_redirect_to_private_blocked(self, monkeypatch):
        """A public page must not be able to bounce the fetch into LAN/loopback."""
        _with_config(monkeypatch, {"net_mode": "allow"})

        def dns(host, port, *a, **k):
            ip = "127.0.0.1" if host == "localhost" else "93.184.216.34"
            return [(2, 1, 6, "", (ip, 0))]
        monkeypatch.setattr("socket.getaddrinfo", dns)

        def fake_get(url, **kw):
            if "example.com" in url:
                return _FakeResponse(status=302,
                                     redirect="http://localhost:8642/v1/models")
            return _FakeResponse(body=b"secret")
        _patch_session(monkeypatch, get=fake_get)
        with pytest.raises(NetworkPolicyError, match="non-public"):
            safe_fetch("https://example.com/jump")

    def test_too_many_redirects(self, monkeypatch):
        _with_config(monkeypatch, {"net_mode": "allow"})
        self._public_dns(monkeypatch)
        _patch_session(monkeypatch, get=lambda url, **kw: _FakeResponse(
            status=302, redirect="https://example.com/again"))
        with pytest.raises(NetworkPolicyError, match="redirects"):
            safe_fetch("https://example.com/loop")

    def test_body_capped(self, monkeypatch):
        _with_config(monkeypatch, {"net_mode": "allow"})
        self._public_dns(monkeypatch)
        _patch_session(monkeypatch, get=lambda url, **kw: _FakeResponse(
            headers={"Content-Type": "text/plain"}, body=b"x" * 500_000))
        _, _, text = safe_fetch("https://example.com/big", max_bytes=1000)
        assert len(text) <= 65536   # stops after the first chunk crosses the cap


# ------------------------------------------------------------------ #
#  safe_fetch_bytes: extra_headers (ADR-0015's optional bearer tokens) #
# ------------------------------------------------------------------ #

class TestSafeFetchBytesExtraHeaders:
    """extra_headers is how HF/CivitAI credentials (model_source_credentials.py)
    reach the actual outbound request. Every existing caller omits it, so the
    default-None path must still send exactly the two fixed headers it always
    has - these are the regression check for that."""

    def _public_dns(self, monkeypatch):
        monkeypatch.setattr(
            "socket.getaddrinfo",
            lambda host, port, *a, **k: [(2, 1, 6, "", ("93.184.216.34", 0))])

    def test_no_extra_headers_by_default(self, monkeypatch):
        _with_config(monkeypatch, {"net_mode": "allow"})
        self._public_dns(monkeypatch)
        seen = {}

        def fake_get(url, **kw):
            seen.update(kw.get("headers") or {})
            return _FakeResponse(body=b"ok")

        _patch_session(monkeypatch, get=fake_get)
        safe_fetch_bytes("https://example.com/a")
        assert set(seen) == {"User-Agent", "Host"}

    def test_extra_headers_are_sent(self, monkeypatch):
        _with_config(monkeypatch, {"net_mode": "allow"})
        self._public_dns(monkeypatch)
        seen = {}

        def fake_get(url, **kw):
            seen.update(kw.get("headers") or {})
            return _FakeResponse(body=b"ok")

        _patch_session(monkeypatch, get=fake_get)
        safe_fetch_bytes("https://example.com/a",
                         extra_headers={"Authorization": "Bearer secret-token"})
        assert seen["Authorization"] == "Bearer secret-token"
        # The fixed pair is still present alongside it.
        assert "User-Agent" in seen and "Host" in seen

    def test_extra_headers_cannot_override_host_or_user_agent(self, monkeypatch):
        """A caller-supplied header dict must never be able to spoof the
        pinned Host or the User-Agent this module presents to every server -
        both are security-relevant (Host backs the redirect-revalidation
        story; a caller-chosen value here would defeat the whole point of
        pinning it)."""
        _with_config(monkeypatch, {"net_mode": "allow"})
        self._public_dns(monkeypatch)
        seen = {}

        def fake_get(url, **kw):
            seen.update(kw.get("headers") or {})
            return _FakeResponse(body=b"ok")

        _patch_session(monkeypatch, get=fake_get)
        safe_fetch_bytes("https://example.com/a", extra_headers={
            "Host": "evil.example", "User-Agent": "attacker-agent",
            "Authorization": "Bearer secret-token",
        })
        assert seen["Host"] == "example.com"
        assert seen["User-Agent"] != "attacker-agent"
        assert seen["Authorization"] == "Bearer secret-token"

    def test_extra_headers_stripped_on_cross_host_redirect(self, monkeypatch):
        """A redirect to a DIFFERENT host must not carry the caller's
        credential with it - the same protection requests' own redirect
        handling provides by default, which this module's manual
        redirect-revalidation loop must not silently drop."""
        _with_config(monkeypatch, {"net_mode": "allow"})
        self._public_dns(monkeypatch)
        seen_by_host = {}

        def fake_get(url, **kw):
            host = urllib.parse.urlparse(url).hostname
            seen_by_host[host] = dict(kw.get("headers") or {})
            if host == "example.com":
                return _FakeResponse(status=302, redirect="https://other.example/b")
            return _FakeResponse(body=b"ok")

        _patch_session(monkeypatch, get=fake_get)
        safe_fetch_bytes("https://example.com/a",
                         extra_headers={"Authorization": "Bearer secret-token"})
        assert seen_by_host["example.com"].get("Authorization") == "Bearer secret-token"
        assert "Authorization" not in seen_by_host["other.example"], (
            "the Authorization header leaked to a DIFFERENT host across a redirect")

    def test_extra_headers_survive_a_same_host_redirect(self, monkeypatch):
        """Only a HOST CHANGE strips the credential - an ordinary same-host
        redirect (e.g. a trailing-slash normalization) must still carry it,
        or every legitimate HF/CivitAI redirect would silently lose auth."""
        _with_config(monkeypatch, {"net_mode": "allow"})
        self._public_dns(monkeypatch)
        calls = []

        def fake_get(url, **kw):
            calls.append(dict(kw.get("headers") or {}))
            if len(calls) == 1:
                return _FakeResponse(status=302, redirect="https://example.com/b")
            return _FakeResponse(body=b"ok")

        _patch_session(monkeypatch, get=fake_get)
        safe_fetch_bytes("https://example.com/a",
                         extra_headers={"Authorization": "Bearer secret-token"})
        assert calls[1].get("Authorization") == "Bearer secret-token"


# ------------------------------------------------------------------ #
#  HTML stripping                                                     #
# ------------------------------------------------------------------ #

class TestHtmlToText:
    def test_strips_tags_and_script(self):
        text = html_to_text(
            "<html><head><title>t</title><script>evil()</script></head>"
            "<body><h1>Hi</h1><p>one</p><p>two &amp; three</p></body></html>")
        assert "evil" not in text
        assert "Hi" in text
        assert "two & three" in text

    def test_survives_malformed_html(self):
        assert isinstance(html_to_text("<div><p>ok<"), str)

    def test_void_meta_link_in_head_do_not_swallow_body(self):
        # <meta> and <link> are void (no end tag). They must not leave the skip
        # counter stuck > 0, which would drop the entire <body> and make
        # html_to_text return "" for every real HTML page.
        text = html_to_text(
            "<html><head>"
            "<meta charset='utf-8'>"
            "<link rel='icon' href='x'>"
            "<meta name='viewport' content='w'>"
            "<title>PageTitle</title><style>.a{}</style>"
            "</head><body><h1>Heading</h1><p>Hello world</p></body></html>")
        assert "Heading" in text
        assert "Hello world" in text
        assert "PageTitle" not in text   # head content is still skipped


# ------------------------------------------------------------------ #
#  Web search                                                         #
# ------------------------------------------------------------------ #

_DDG_HTML = """
<html><body>
<div class="result">
  <h2 class="result__title">
    <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fdocs&amp;rut=x">Example <b>Docs</b></a>
  </h2>
  <a class="result__snippet" href="...">The documentation for Example.</a>
</div>
<div class="result">
  <h2 class="result__title">
    <a class="result__a" href="https://direct.example.org/page">Direct Result</a>
  </h2>
  <a class="result__snippet" href="...">A direct link snippet.</a>
</div>
</body></html>
"""


class TestWebSearch:
    def test_empty_query_rejected(self, monkeypatch):
        _with_config(monkeypatch, {"net_mode": "allow"})
        with pytest.raises(ValueError):
            web_search("   ")

    def test_ddg_parsing(self, monkeypatch):
        _with_config(monkeypatch, {"net_mode": "allow"})
        monkeypatch.setattr(
            "socket.getaddrinfo",
            lambda host, port, *a, **k: [(2, 1, 6, "", ("93.184.216.34", 0))])

        class _R:
            text = _DDG_HTML
            def raise_for_status(self):
                pass
        _patch_session(monkeypatch, post=lambda url, **kw: _R())

        results = web_search("example docs")
        assert results[0]["url"] == "https://example.com/docs"   # uddg decoded
        assert results[0]["title"] == "Example Docs"
        assert "documentation" in results[0]["snippet"]
        assert results[1]["url"] == "https://direct.example.org/page"

    def test_searxng_backend(self, monkeypatch):
        _with_config(monkeypatch, {
            "net_mode": "allow",
            "net_allow_private": True,
            "net_search_url": "http://127.0.0.1:8080/",
        })
        seen = {}

        class _R:
            def raise_for_status(self):
                pass
            def json(self):
                return {"results": [
                    {"title": "T1", "url": "https://a.example/", "content": "c1"},
                    {"title": "T2", "url": "https://b.example/", "content": "c2"},
                ]}

        def fake_get(url, **kw):
            seen["url"] = url
            return _R()
        _patch_session(monkeypatch, get=fake_get)

        results = web_search("query", max_results=1)
        assert seen["url"].startswith("http://127.0.0.1:8080/search?")
        assert results == [{"title": "T1", "url": "https://a.example/",
                            "snippet": "c1"}]

    def test_no_results_is_an_error(self, monkeypatch):
        _with_config(monkeypatch, {"net_mode": "allow"})
        monkeypatch.setattr(
            "socket.getaddrinfo",
            lambda host, port, *a, **k: [(2, 1, 6, "", ("93.184.216.34", 0))])

        class _R:
            text = "<html><body>captcha?</body></html>"
            def raise_for_status(self):
                pass
        _patch_session(monkeypatch, post=lambda url, **kw: _R())
        with pytest.raises(RuntimeError, match="no parseable results"):
            web_search("anything")

    def test_mode_off_blocks_search(self, monkeypatch):
        _with_config(monkeypatch, {"net_mode": "off"})
        with pytest.raises(NetworkPolicyError):
            web_search("anything")

    def test_format_results(self):
        out = format_results([
            {"title": "A", "url": "https://a/", "snippet": "sa"},
            {"title": "B", "url": "https://b/", "snippet": ""},
        ])
        assert "1. A" in out and "https://a/" in out and "sa" in out
        assert "2. B" in out
