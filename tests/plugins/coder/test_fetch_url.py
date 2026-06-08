"""
Tests for tool_fetch_url in localm.plugins.coder.tools

All network calls are mocked — no real HTTP is made.
"""

import io
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from localm.plugins.coder.tools import tool_fetch_url


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _fake_response(body: str, content_type: str = "text/html; charset=utf-8"):
    """Build a mock urllib response object."""
    resp = MagicMock()
    resp.read.return_value = body.encode("utf-8")
    resp.headers.get.return_value = content_type
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


def _call(url: str = "http://example.com", max_chars: int = 8000, **kwargs):
    return tool_fetch_url(Path("/tmp"), url, max_chars=max_chars, **kwargs)


# ---------------------------------------------------------------------------
#  Happy-path HTML stripping
# ---------------------------------------------------------------------------

class TestHtmlStripping:
    def _fetch(self, html: str, content_type="text/html"):
        with patch("urllib.request.urlopen", return_value=_fake_response(html, content_type)):
            return _call()

    def test_basic_tag_removal(self):
        result = self._fetch("<html><body><p>Hello world</p></body></html>")
        assert result.ok
        assert "Hello world" in result.output
        assert "<p>" not in result.output

    def test_script_tags_stripped(self):
        result = self._fetch(
            "<html><body><p>Keep me</p>"
            "<script>alert('secret')</script></body></html>"
        )
        assert "Keep me" in result.output
        assert "secret" not in result.output
        assert "alert" not in result.output

    def test_style_tags_stripped(self):
        result = self._fetch(
            "<html><head><style>body{color:red}</style></head>"
            "<body><p>Visible</p></body></html>"
        )
        assert "Visible" in result.output
        assert "color:red" not in result.output

    def test_head_content_stripped(self):
        result = self._fetch(
            "<html><head><title>Page Title</title></head>"
            "<body><p>Body text</p></body></html>"
        )
        assert "Body text" in result.output
        assert "Page Title" not in result.output

    def test_noscript_stripped(self):
        result = self._fetch(
            "<html><body><noscript>JS required</noscript><p>Content</p></body></html>"
        )
        assert "Content" in result.output
        assert "JS required" not in result.output

    def test_html_entities_decoded(self):
        result = self._fetch("<p>&amp; &lt; &gt; &quot;</p>")
        assert "& < > \"" in result.output

    def test_excessive_blank_lines_collapsed(self):
        result = self._fetch("<p>A</p>\n\n\n\n\n\n<p>B</p>")
        # Should not have 3+ consecutive newlines
        assert "\n\n\n" not in result.output
        assert "A" in result.output
        assert "B" in result.output

    def test_url_and_content_in_output(self):
        result = self._fetch("<p>text</p>")
        assert "http://example.com" in result.output
        assert "text" in result.output


# ---------------------------------------------------------------------------
#  Plain-text content type
# ---------------------------------------------------------------------------

class TestPlainText:
    def test_plain_text_not_html_stripped(self):
        raw = "line one\nline two\nline three"
        with patch("urllib.request.urlopen",
                   return_value=_fake_response(raw, "text/plain")):
            result = _call()
        assert result.ok
        assert "line one" in result.output
        assert "line two" in result.output

    def test_json_content_type_passthrough(self):
        raw = '{"key": "value"}'
        with patch("urllib.request.urlopen",
                   return_value=_fake_response(raw, "application/json")):
            result = _call()
        assert result.ok
        assert '"key"' in result.output


# ---------------------------------------------------------------------------
#  Truncation
# ---------------------------------------------------------------------------

class TestTruncation:
    def test_long_content_truncated_at_max_chars(self):
        body = "<p>" + "x" * 20_000 + "</p>"
        with patch("urllib.request.urlopen",
                   return_value=_fake_response(body)):
            result = tool_fetch_url(Path("/tmp"), "http://x.com", max_chars=100)
        assert result.ok
        assert result.truncated
        assert "truncated" in result.output.lower()

    def test_short_content_not_truncated(self):
        body = "<p>short</p>"
        with patch("urllib.request.urlopen",
                   return_value=_fake_response(body)):
            result = _call(max_chars=8000)
        assert not result.truncated

    def test_summary_mentions_truncation(self):
        body = "<p>" + "y" * 20_000 + "</p>"
        with patch("urllib.request.urlopen",
                   return_value=_fake_response(body)):
            result = tool_fetch_url(Path("/tmp"), "http://x.com", max_chars=50)
        assert "truncated" in result.summary


# ---------------------------------------------------------------------------
#  Error handling
# ---------------------------------------------------------------------------

class TestErrors:
    def test_url_error_returns_error_result(self):
        import urllib.error
        with patch("urllib.request.urlopen",
                   side_effect=urllib.error.URLError("connection refused")):
            result = _call("http://unreachable.example")
        assert not result.ok
        assert "Could not fetch" in result.output

    def test_generic_exception_returns_error_result(self):
        with patch("urllib.request.urlopen", side_effect=Exception("boom")):
            result = _call()
        assert not result.ok

    def test_error_result_not_truncated(self):
        import urllib.error
        with patch("urllib.request.urlopen",
                   side_effect=urllib.error.URLError("nope")):
            result = _call()
        assert not result.truncated


# ---------------------------------------------------------------------------
#  User-Agent header
# ---------------------------------------------------------------------------

class TestUserAgent:
    def test_user_agent_set(self):
        sent_headers = {}

        def fake_urlopen(req, timeout=None):
            sent_headers.update(req.headers)
            return _fake_response("<p>ok</p>")

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            _call()

        # Headers are title-cased by urllib
        ua = sent_headers.get("User-agent", "")
        assert "localm" in ua.lower()


# ---------------------------------------------------------------------------
#  Summary field
# ---------------------------------------------------------------------------

class TestSummary:
    def test_summary_contains_url(self):
        with patch("urllib.request.urlopen",
                   return_value=_fake_response("<p>hi</p>")):
            result = _call("http://docs.example.com/page")
        assert "docs.example.com" in result.summary

    def test_summary_contains_char_count(self):
        with patch("urllib.request.urlopen",
                   return_value=_fake_response("<p>hello world</p>")):
            result = _call()
        # summary should mention how many chars
        assert "chars" in result.summary or any(c.isdigit() for c in result.summary)
