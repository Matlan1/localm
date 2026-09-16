# SPDX-License-Identifier: AGPL-3.0-or-later
"""Doubles and HTML fixtures shared by the web_retrieval tests.

``Transport`` replaces ``localm.netpolicy._session_for`` (the pinned-transport
seam) with a router keyed by HTTP method and URL, so every test drives the real
policy check, redirect handling, byte cap and charset decoding without a
socket. Page fixtures reproduce the audited failure: thousands of characters of
site chrome before a short article that holds the answer.
"""

from __future__ import annotations

import threading
import urllib.parse
from typing import Callable, Optional

DDG_ENDPOINT = "https://html.duckduckgo.com/html/"
PUBLIC_IP = "93.184.216.34"

QUERY = "Linz museum opening hours"
ANSWER = ("The Linz museum opening hours are 10:00 to 18:00 from Tuesday to "
          "Sunday, and admission is free on the first Sunday of the month.")


def public_dns(host, port, *args, **kwargs):
    return [(2, 1, 6, "", (PUBLIC_IP, 0))]


def allow_public(monkeypatch, **extra_cfg) -> dict:
    """Configure ``net_mode=allow`` plus *extra_cfg*, resolve every host to a
    public address, and clear the env override. Returns the config dict."""
    cfg = {"net_mode": "allow", **extra_cfg}
    monkeypatch.delenv("LOCALM_NET_MODE", raising=False)
    monkeypatch.setattr("localm.config.load_config", lambda: cfg)
    monkeypatch.setattr("socket.getaddrinfo", public_dns)
    return cfg


class FakeResponse:
    def __init__(self, *, status: int = 200, headers: Optional[dict] = None,
                 body: bytes | str = b"", text: Optional[str] = None,
                 json_body=None, redirect: Optional[str] = None):
        self.status_code = status
        self.headers = dict(headers or {})
        if redirect:
            self.headers["Location"] = redirect
        if isinstance(body, str):
            body = body.encode("utf-8")
        self._body = body
        self._text = text
        self._json = json_body

    @property
    def is_redirect(self) -> bool:
        return self.status_code in (301, 302, 303, 307, 308) and \
            "Location" in self.headers

    @property
    def is_permanent_redirect(self) -> bool:
        return False

    @property
    def text(self) -> str:
        if self._text is not None:
            return self._text
        return self._body.decode("utf-8", errors="replace")

    def json(self):
        if self._json is None:
            raise ValueError("no JSON body")
        return self._json

    def iter_content(self, chunk_size: int = 65536):
        for i in range(0, len(self._body), chunk_size):
            yield self._body[i:i + chunk_size]

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def close(self) -> None:
        pass


class FakeSession:
    def __init__(self, responder: Callable):
        self._responder = responder

    def get(self, url, **kw):
        return self._responder("GET", url, **kw)

    def post(self, url, **kw):
        return self._responder("POST", url, **kw)

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class Transport:
    """Route (method, url) to a ``FakeResponse`` or a callable
    ``(url, **kw) -> FakeResponse``. A route URL ending in ``*`` matches by
    prefix. Every request is recorded in ``calls`` as ``(method, url, kw)``;
    an unrouted request raises ``AssertionError``."""

    def __init__(self):
        self.routes: dict[tuple[str, str], object] = {}
        self.calls: list[tuple[str, str, dict]] = []
        self._lock = threading.Lock()

    def route(self, method: str, url: str, handler) -> "Transport":
        self.routes[(method.upper(), url)] = handler
        return self

    def _lookup(self, method: str, url: str):
        exact = self.routes.get((method, url))
        if exact is not None:
            return exact
        for (m, pattern), handler in self.routes.items():
            if m == method and pattern.endswith("*") \
                    and url.startswith(pattern[:-1]):
                return handler
        return None

    def responder(self, method: str, url: str, **kw):
        with self._lock:
            self.calls.append((method, url, kw))
        handler = self._lookup(method, url)
        if handler is None:
            raise AssertionError(f"unexpected {method} {url}")
        if isinstance(handler, FakeResponse):
            return handler
        return handler(url, **kw)

    def install(self, monkeypatch) -> "Transport":
        monkeypatch.setattr("localm.netpolicy._session_for",
                            lambda url: FakeSession(self.responder))
        return self

    def urls(self, method: Optional[str] = None) -> list[str]:
        return [u for m, u, _ in self.calls if method is None or m == method]


def html_page(body: str, *, title: str = "Fixture",
              charset: str = "utf-8") -> str:
    return (f'<!DOCTYPE html><html><head><meta charset="{charset}">'
            f"<title>{title}</title></head><body>{body}</body></html>")


def html_response(markup: str, *, charset: str = "utf-8",
                  content_type: Optional[str] = None,
                  status: int = 200) -> FakeResponse:
    ctype = content_type if content_type is not None \
        else f"text/html; charset={charset}"
    return FakeResponse(status=status, headers={"Content-Type": ctype},
                        body=markup.encode(charset))


def ddg_html(results: list[tuple[str, str, str]]) -> str:
    """DuckDuckGo html endpoint markup for ``(title, url, snippet)`` rows."""
    rows = []
    for title, url, snippet in results:
        target = urllib.parse.quote(url, safe="")
        rows.append(
            '<div class="result results_links">\n'
            '  <h2 class="result__title">\n'
            f'    <a class="result__a" href="//duckduckgo.com/l/?uddg={target}'
            f'&amp;rut=abc">{title}</a>\n'
            "  </h2>\n"
            f'  <a class="result__snippet" href="#">{snippet}</a>\n'
            "</div>\n")
    return "<html><body>\n" + "".join(rows) + "</body></html>"


def searx_json(results: list[tuple[str, str, str]]) -> dict:
    return {"results": [{"title": t, "url": u, "content": s}
                        for t, u, s in results]}


_FILLER = (
    "The committee reviewed the annual budget and adjourned without further "
    "remarks. Several members asked for a written summary of the procurement "
    "changes, which the chair promised for the following week. The minutes "
    "record a short discussion about the parking arrangements near the east "
    "entrance and a vote to repaint the corridor on the second floor. "
)


def filler_paragraph(index: int, chars: int = 400) -> str:
    text = f"Paragraph {index}. " + (_FILLER * 3)
    return text[:chars].rsplit(" ", 1)[0] + "."


def boilerplate_text(chars: int) -> str:
    """Plain, link-free boilerplate (a legal notice) of about *chars*
    characters; contains no word of QUERY."""
    sentence = ("This website stores small text files on your device to keep "
                "you signed in and to remember your language choice. By "
                "continuing you agree to the terms of use and the privacy "
                "notice published by the operator. ")
    out = []
    total = 0
    i = 0
    while total < chars:
        piece = f"Notice {i}: {sentence}"
        out.append(piece)
        total += len(piece)
        i += 1
    return "".join(out)


def menu_items(chars: int) -> str:
    """``<li><a>`` items whose text totals about *chars* characters."""
    items = []
    total = 0
    i = 0
    while total < chars:
        label = f"Section {i} overview page"
        items.append(f'<li><a href="/section/{i}">{label}</a></li>')
        total += len(label)
        i += 1
    return "".join(items)


def article_html(answer: str = ANSWER, paragraphs: int = 4) -> str:
    parts = ["<h1>Visiting the city</h1>"]
    for i in range(paragraphs):
        parts.append(f"<p>{filler_paragraph(i)}</p>")
    parts.append(f"<p>{answer}</p>")
    parts.append(f"<p>{filler_paragraph(paragraphs)}</p>")
    return "".join(parts)


def nav_heavy_page(mode: str = "semantic", *, nav_chars: int = 7000,
                   answer: str = ANSWER) -> str:
    """A page with about *nav_chars* characters of chrome before a short
    article that ends with *answer*.

    ``semantic``: chrome in ``<header>``/``<nav>``/``<aside>``/``<footer>``,
    content in ``<main><article>``.
    ``menus``: chrome as link-dense ``<div class="menu"><ul>`` lists, content
    in a plain ``<div>``; no ``main``/``article``.
    ``plain``: chrome as link-free boilerplate prose in a ``<div>``, content in
    a plain ``<div>``; nothing marks either as chrome or content.
    """
    article = article_html(answer)
    if mode == "semantic":
        body = (
            "<header><a href='/'>Site Name</a> <a href='/login'>Login</a> "
            "<a href='/register'>Register</a></header>"
            f"<nav><ul>{menu_items(nav_chars)}</ul></nav>"
            f"<main><article>{article}</article></main>"
            "<aside><h3>Related</h3><ul><li><a href='/a'>Weather</a></li>"
            "<li><a href='/b'>Traffic</a></li><li><a href='/c'>Events</a></li>"
            "</ul></aside>"
            "<footer><a href='/imprint'>Imprint</a> <a href='/privacy'>"
            "Privacy</a> Copyright 2026</footer>")
    elif mode == "menus":
        body = (
            "<div class='topbar'><a href='/'>Site Name</a> <a href='/login'>"
            "Login</a> <a href='/register'>Register</a></div>"
            f"<div class='menu'><ul>{menu_items(nav_chars)}</ul></div>"
            f"<div class='content'>{article}</div>"
            "<div class='bottom'><a href='/imprint'>Imprint</a> "
            "<a href='/privacy'>Privacy</a> <a href='/contact'>Contact</a></div>")
    elif mode == "plain":
        body = (f"<div class='legal'><p>{boilerplate_text(nav_chars)}</p></div>"
                f"<div class='content'>{article}</div>")
    else:
        raise ValueError(mode)
    return html_page(body, title="City guide")


class StubProvider:
    """A ``SearchProvider`` that returns canned ``(title, url, snippet)`` rows
    (or ``SearchResult`` objects) and records every ``(query, max_results)``
    call. A ``fail`` exception is raised instead of returning."""

    name = "stub"

    def __init__(self, rows, *, fail: Optional[BaseException] = None):
        self.rows = list(rows)
        self.fail = fail
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, max_results: int):
        from localm.web_retrieval import SearchResult
        self.calls.append((query, max_results))
        if self.fail is not None:
            raise self.fail
        out = []
        for i, row in enumerate(self.rows[:max_results], 1):
            if isinstance(row, SearchResult):
                out.append(row)
            else:
                title, url, snippet = row
                out.append(SearchResult(title=title, url=url, snippet=snippet,
                                        rank=i, provider=self.name))
        return out


def stub_retrieval(monkeypatch, rows, pages: Optional[dict] = None, *,
                   fail: Optional[BaseException] = None) -> list[str]:
    """Route ``localm.web_retrieval.retrieve`` through a ``StubProvider`` built
    from *rows* and an in-memory page fetch (``pages`` maps a URL to its HTML
    body; an unmapped URL fails with ``RuntimeError("HTTP 404")``), so the
    real controller, extraction and evidence selection run with no socket.
    Returns the list the retrieve calls' queries are appended to."""
    from localm import web_retrieval

    real_retrieve = web_retrieval.retrieve
    provider = StubProvider(rows, fail=fail)
    queries: list[str] = []

    def fake_fetch(url, *, timeout):
        body = (pages or {}).get(url)
        if body is None:
            raise RuntimeError("HTTP 404")
        return url, "text/html; charset=utf-8", body

    def fake_retrieve(query, **kw):
        queries.append(query)
        kw.setdefault("provider", provider)
        kw.setdefault("fetch", fake_fetch)
        return real_retrieve(query, **kw)

    monkeypatch.setattr("localm.web_retrieval.retrieve", fake_retrieve)
    return queries
