# SPDX-License-Identifier: AGPL-3.0-or-later
"""The line a browser session logs for a request it refused.

It is written at INFO, so it reaches the debug log and the recent-activity
list in every session mode. A URL's path and query string are content (a search
query, a token), so the line names the host only, unless the debug log may
hold chat content.
"""

from __future__ import annotations

import logging

import pytest

from localm.browser import session as bs

URL = "https://user:hunter2@search.example.com:8443/find?q=my+secret+diagnosis&token=abc123"


def _session():
    s = object.__new__(bs.BrowserSession)
    s.session_id = "s1"
    s.state = bs.SessionState()
    return s


def _refuse(caplog, url=URL) -> str:
    caplog.set_level(logging.INFO, logger=bs.logger.name)
    s = _session()
    s._refuse(url, "host not allowed")
    assert [b.url for b in s.state.blocked] == [url]
    lines = [r.getMessage() for r in caplog.records if "blocked" in r.getMessage()]
    assert len(lines) == 1, lines
    return lines[0]


@pytest.mark.parametrize("debug", [False, True])
def test_privacy_mode_logs_the_host_only(monkeypatch, caplog, debug):
    monkeypatch.setenv("LOCALM_MODE", "privacy")
    if debug:
        monkeypatch.setenv("LOCALM_DEBUG", "1")
    else:
        monkeypatch.delenv("LOCALM_DEBUG", raising=False)
    line = _refuse(caplog)
    assert line == "browser s1 blocked search.example.com:8443: host not allowed"
    for secret in ("secret", "diagnosis", "abc123", "hunter2", "/find"):
        assert secret not in line


def test_full_mode_without_debug_logs_the_host_only(monkeypatch, caplog):
    monkeypatch.setenv("LOCALM_MODE", "full")
    monkeypatch.delenv("LOCALM_DEBUG", raising=False)
    assert "diagnosis" not in _refuse(caplog)


def test_full_mode_with_debug_logs_the_whole_url(monkeypatch, caplog):
    monkeypatch.setenv("LOCALM_MODE", "full")
    monkeypatch.setenv("LOCALM_DEBUG", "1")
    assert URL in _refuse(caplog)


@pytest.mark.parametrize("url,shown", [
    ("data:text/html,<script>secret()</script>", "(no host)"),
    ("about:blank", "(no host)"),
    ("http://[not-an-address/secret", "(no host)"),
    ("wss://chat.example.com/socket?session=secret", "chat.example.com"),
    ("", "(no host)"),
])
def test_urls_without_a_usable_host(monkeypatch, caplog, url, shown):
    monkeypatch.setenv("LOCALM_MODE", "privacy")
    line = _refuse(caplog, url)
    assert line == f"browser s1 blocked {shown}: host not allowed"
    assert "secret" not in line


def _messages(caplog) -> list:
    return [r.getMessage() for r in caplog.records]


def test_a_reason_that_quotes_the_url_is_shortened_too(monkeypatch, caplog):
    monkeypatch.setenv("LOCALM_MODE", "privacy")
    caplog.set_level(logging.INFO, logger=bs.logger.name)
    _session()._refuse(URL, f"URL is not allowed: {URL}")
    assert _messages(caplog)[-1] == (
        "browser s1 blocked search.example.com:8443: "
        "URL is not allowed: search.example.com:8443")


def test_a_failed_gate_logs_the_host_only(monkeypatch, caplog):
    import asyncio
    monkeypatch.setenv("LOCALM_MODE", "privacy")
    caplog.set_level(logging.DEBUG, logger=bs.logger.name)

    class _Request:
        url = URL

    class _Route:
        aborted = False

        async def abort(self):
            self.aborted = True

    async def _raises(route, request):
        raise RuntimeError(f"could not decide {request.url}")

    s = _session()
    s._route = _raises
    route = _Route()
    asyncio.run(s._on_route(route, _Request()))
    assert route.aborted
    blob = "\n".join(_messages(caplog))
    assert "gate failed for search.example.com:8443" in blob
    for secret in ("diagnosis", "abc123", "hunter2", "/find"):
        assert secret not in blob


@pytest.mark.parametrize("mode,full", [("privacy", False), ("full", True)])
def test_a_note_logs_the_host_only_and_keeps_the_console_line(
        monkeypatch, caplog, mode, full):
    monkeypatch.setenv("LOCALM_MODE", mode)
    monkeypatch.setenv("LOCALM_DEBUG", "1")
    caplog.set_level(logging.INFO, logger=bs.logger.name)
    s = _session()
    s._note("the page %s closed itself" % URL, URL)
    assert s.state.console == [{"type": "localm",
                                "text": f"the page {URL} closed itself"}]
    line = _messages(caplog)[-1]
    assert (URL in line) is full
    assert ("diagnosis" in line) is full
    assert "search.example.com:8443" in line
