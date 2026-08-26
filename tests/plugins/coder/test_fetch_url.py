# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Tests for tool_fetch_url / tool_web_search in localm.plugins.coder.tools.

Both route through localm.netpolicy. All network calls are mocked - no real
HTTP is made. The policy itself is tested in tests/test_netpolicy.py; here we
test the tool-level behaviour (stripping, truncation, errors, privacy audit,
and that policy refusals surface as tool errors).
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from localm.plugins.coder.tools import tool_fetch_url, tool_web_search


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

class _FakeResponse:
    def __init__(self, body: str, content_type: str = "text/html; charset=utf-8"):
        self.status_code = 200
        self.headers = {"Content-Type": content_type}
        self._body = body.encode("utf-8")

    is_redirect = False
    is_permanent_redirect = False

    def iter_content(self, chunk_size=65536):
        yield self._body

    def raise_for_status(self):
        pass

    def close(self):
        pass


_PUBLIC_DNS = [(2, 1, 6, "", ("93.184.216.34", 80))]
_LOOPBACK_DNS = [(2, 1, 6, "", ("127.0.0.1", 8642))]


class _FakeSession:
    """Doubles netpolicy._session_for (the pinned-transport seam) so the fetch
    path is exercised without a live socket. get may be a fixed response or a
    responder callable (url, **kw)."""

    def __init__(self, get=None):
        self._get = get

    def get(self, url, **kw):
        if self._get is None:
            raise AssertionError(f"unexpected GET to {url}")
        return self._get(url, **kw) if callable(self._get) else self._get

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _session(get=None):
    """patch() context manager routing the pinned session to a fake response."""
    return patch("localm.netpolicy._session_for",
                 return_value=_FakeSession(get=get))


def _raise(exc):
    def _f(url, **kw):
        raise exc
    return _f


@pytest.fixture(autouse=True)
def _policy_env(monkeypatch):
    """Deterministic policy: mode allow, no domain rules, default SSRF guard."""
    monkeypatch.setenv("LOCALM_NET_MODE", "allow")
    monkeypatch.setattr("localm.config.load_config", lambda: {})


def _call(url: str = "http://example.com", max_chars: int = 8000, **kwargs):
    return tool_fetch_url(Path("/tmp"), url, max_chars=max_chars, **kwargs)


def _fetch(html: str, content_type="text/html", url="http://example.com"):
    with patch("socket.getaddrinfo", return_value=_PUBLIC_DNS), \
         _session(get=_FakeResponse(html, content_type)):
        return _call(url)


# ---------------------------------------------------------------------------
#  Happy-path HTML stripping
# ---------------------------------------------------------------------------

class TestHtmlStripping:
    def test_basic_tag_removal(self):
        result = _fetch("<html><body><p>Hello world</p></body></html>")
        assert result.ok
        assert "Hello world" in result.output
        assert "<p>" not in result.output

    def test_script_tags_stripped(self):
        result = _fetch(
            "<html><body><p>Keep me</p>"
            "<script>alert('secret')</script></body></html>"
        )
        assert "Keep me" in result.output
        assert "secret" not in result.output
        assert "alert" not in result.output

    def test_style_tags_stripped(self):
        result = _fetch(
            "<html><head><style>body{color:red}</style></head>"
            "<body><p>Visible</p></body></html>"
        )
        assert "Visible" in result.output
        assert "color:red" not in result.output

    def test_head_content_stripped(self):
        result = _fetch(
            "<html><head><title>Page Title</title></head>"
            "<body><p>Body text</p></body></html>"
        )
        assert "Body text" in result.output
        assert "Page Title" not in result.output

    def test_noscript_stripped(self):
        result = _fetch(
            "<html><body><noscript>JS required</noscript><p>Content</p></body></html>"
        )
        assert "Content" in result.output
        assert "JS required" not in result.output

    def test_html_entities_decoded(self):
        result = _fetch("<p>&amp; &lt; &gt; &quot;</p>")
        assert "& < > \"" in result.output

    def test_excessive_blank_lines_collapsed(self):
        result = _fetch("<p>A</p>\n\n\n\n\n\n<p>B</p>")
        assert "\n\n\n" not in result.output
        assert "A" in result.output
        assert "B" in result.output

    def test_url_and_content_in_output(self):
        result = _fetch("<p>text</p>")
        assert "http://example.com" in result.output
        assert "text" in result.output


# ---------------------------------------------------------------------------
#  Plain-text content type
# ---------------------------------------------------------------------------

class TestPlainText:
    @pytest.mark.parametrize(
        "content, content_type, expected",
        [
            ("line one\nline two\nline three", "text/plain",
             ("line one", "line two")),
            ('{"key": "value"}', "application/json", ('"key"',)),
        ],
        ids=["text_plain", "application_json"],
    )
    def test_non_html_content_type_not_stripped(self, content, content_type, expected):
        result = _fetch(content, content_type)
        assert result.ok
        for substring in expected:
            assert substring in result.output


# ---------------------------------------------------------------------------
#  Truncation
# ---------------------------------------------------------------------------

class TestTruncation:
    def test_long_content_truncated_at_max_chars(self):
        with patch("socket.getaddrinfo", return_value=_PUBLIC_DNS), \
             _session(get=_FakeResponse("<p>" + "x" * 20_000 + "</p>")):
            result = tool_fetch_url(Path("/tmp"), "http://x.com", max_chars=100)
        assert result.ok
        assert result.truncated
        assert "truncated" in result.output.lower()

    def test_short_content_not_truncated(self):
        result = _fetch("<p>short</p>")
        assert not result.truncated

    def test_summary_mentions_truncation(self):
        with patch("socket.getaddrinfo", return_value=_PUBLIC_DNS), \
             _session(get=_FakeResponse("<p>" + "y" * 20_000 + "</p>")):
            result = tool_fetch_url(Path("/tmp"), "http://x.com", max_chars=50)
        assert "truncated" in result.summary


# ---------------------------------------------------------------------------
#  Error handling
# ---------------------------------------------------------------------------

class TestErrors:
    def test_connection_error_returns_error_result(self):
        with patch("socket.getaddrinfo", return_value=_PUBLIC_DNS), \
             _session(get=_raise(ConnectionError("connection refused"))):
            result = _call("http://unreachable.example")
        assert not result.ok
        assert "Could not fetch" in result.output

    def test_generic_exception_returns_error_result(self):
        with patch("socket.getaddrinfo", return_value=_PUBLIC_DNS), \
             _session(get=_raise(Exception("boom"))):
            result = _call()
        assert not result.ok
        assert not result.truncated


# ---------------------------------------------------------------------------
#  User-Agent header
# ---------------------------------------------------------------------------

class TestUserAgent:
    def test_user_agent_set(self):
        sent = {}

        def fake_get(url, **kwargs):
            sent.update(kwargs.get("headers") or {})
            return _FakeResponse("<p>ok</p>")

        with patch("socket.getaddrinfo", return_value=_PUBLIC_DNS), \
             _session(get=fake_get):
            _call()
        assert "localm" in sent.get("User-Agent", "").lower()


# ---------------------------------------------------------------------------
#  Summary field
# ---------------------------------------------------------------------------

class TestSummary:
    def test_summary_contains_url(self):
        result = _fetch("<p>hi</p>", url="http://docs.example.com/page")
        assert "docs.example.com" in result.summary

    def test_summary_contains_char_count(self):
        result = _fetch("<p>hello world</p>")
        assert "chars" in result.summary or any(c.isdigit() for c in result.summary)


# ---------------------------------------------------------------------------
#  Privacy mode audit log
# ---------------------------------------------------------------------------

class TestPrivacyAuditLog:
    def test_prints_url_to_stderr_in_privacy_mode(self, capsys):
        with patch("socket.getaddrinfo", return_value=_PUBLIC_DNS), \
             _session(get=_FakeResponse("<p>ok</p>")):
            result = tool_fetch_url(
                Path("/tmp"), "http://example.com/secret",
                _privacy=True,
            )
        assert result.ok
        err = capsys.readouterr().err
        assert "http://example.com/secret" in err
        assert "privacy" in err.lower() or "fetch_url" in err.lower()

    def test_no_stderr_without_privacy_mode(self, capsys):
        with patch("socket.getaddrinfo", return_value=_PUBLIC_DNS), \
             _session(get=_FakeResponse("<p>ok</p>")):
            tool_fetch_url(Path("/tmp"), "http://example.com/page")
        err = capsys.readouterr().err
        assert "http://example.com/page" not in err

    def test_web_search_prints_query_in_privacy_mode(self, capsys):
        with patch("localm.netpolicy.web_search",
                   return_value=[{"title": "t", "url": "https://u/",
                                  "snippet": "s"}]):
            result = tool_web_search(Path("/tmp"), "secret query", _privacy=True)
        assert result.ok
        assert "secret query" in capsys.readouterr().err


# ---------------------------------------------------------------------------
#  Policy enforcement surfaces as tool errors (no fetch attempted)
# ---------------------------------------------------------------------------

class TestPolicyEnforcement:
    def test_file_scheme_rejected(self):
        with patch("requests.get") as m:
            r = _call("file:///etc/passwd")
        assert not r.ok
        assert "http/https" in r.output
        m.assert_not_called()

    def test_windows_file_scheme_rejected(self):
        with patch("requests.get") as m:
            r = _call("file:///Z:/Users/me/.ssh/id_rsa")
        assert not r.ok
        m.assert_not_called()

    def test_ftp_scheme_rejected(self):
        with patch("requests.get") as m:
            r = _call("ftp://example.com/secret")
        assert not r.ok
        m.assert_not_called()

    def test_data_scheme_rejected(self):
        with patch("requests.get") as m:
            r = _call("data:text/plain;base64,SGVsbG8=")
        assert not r.ok
        m.assert_not_called()

    def test_link_local_metadata_blocked(self):
        # 169.254.169.254 (cloud metadata) must be refused before any fetch.
        with patch("requests.get") as m, \
             patch("socket.getaddrinfo",
                   return_value=[(2, 1, 6, "", ("169.254.169.254", 80))]):
            r = _call("http://metadata.internal/latest/meta-data/")
        assert not r.ok
        assert "non-public" in r.output
        m.assert_not_called()

    def test_normal_http_still_works(self):
        result = _fetch("<p>hi</p>")
        assert result.ok
        assert "hi" in result.output

    def test_localhost_blocked_by_default(self):
        # Loopback and private targets are refused unless net_allow_private is
        # set; the error message names that setting.
        with patch("requests.get") as m, \
             patch("socket.getaddrinfo", return_value=_LOOPBACK_DNS):
            r = _call("http://127.0.0.1:8642/health")
        assert not r.ok
        assert "net_allow_private" in r.output
        m.assert_not_called()

    def test_localhost_reachable_with_net_allow_private(self, monkeypatch):
        monkeypatch.setattr("localm.config.load_config",
                            lambda: {"net_allow_private": True})
        with patch("socket.getaddrinfo", return_value=_LOOPBACK_DNS), \
             _session(get=_FakeResponse("<p>local</p>")):
            r = _call("http://127.0.0.1:8642/health")
        assert r.ok
        assert "local" in r.output

    def test_deny_list_blocks_domain(self, monkeypatch):
        monkeypatch.setattr("localm.config.load_config",
                            lambda: {"net_deny": ["example.com"]})
        with patch("requests.get") as m:
            r = _call("http://example.com/page")
        assert not r.ok
        assert "deny list" in r.output
        m.assert_not_called()

    def test_web_search_policy_refusal_is_tool_error(self, monkeypatch):
        monkeypatch.setenv("LOCALM_NET_MODE", "off")
        r = tool_web_search(Path("/tmp"), "anything")
        assert not r.ok
        assert "net_mode=off" in r.output
