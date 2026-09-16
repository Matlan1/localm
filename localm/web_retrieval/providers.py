# SPDX-License-Identifier: AGPL-3.0-or-later
"""Search providers: DuckDuckGo's no-key HTML endpoint and a SearXNG instance.

Both send exactly one policy-checked request through ``localm.netpolicy``:
``check_url`` on the request URL, then ``netpolicy._session_for`` (the pinned
transport seam) with ``allow_redirects=False``; any 3xx is refused. Every
``netpolicy`` attribute is read from the module at call time.

``provider_from_config`` picks SearXNG when ``net_search_url`` is set and
DuckDuckGo otherwise. A provider never falls back to another provider.
"""

from __future__ import annotations

import html.parser
import urllib.parse
from typing import Optional

from localm import netpolicy

from .contracts import SearchProvider, SearchProviderError, SearchResult

_MAX_RESULTS = 10
_TITLE_CAP = 300
_SNIPPET_CAP = 500

NO_RESULTS_MESSAGE = (
    "The search backend returned no parseable results. It may be "
    "rate-limiting; try again, or set a Search backend URL (SearXNG) with:  "
    "localm config net_search_url http://...")


def _refuse_redirect(resp, backend: str) -> None:
    """Raise ``NetworkPolicyError`` when *resp* is a 3xx. Search requests are
    sent with ``allow_redirects=False`` and their redirect target is never
    policy-checked, so a redirect is refused rather than followed."""
    if getattr(resp, "is_redirect", False) or \
            getattr(resp, "is_permanent_redirect", False):
        raise netpolicy.NetworkPolicyError(
            f"{backend} tried to redirect (to "
            f"{resp.headers.get('Location', '?')!r}); refusing - a search "
            "backend's redirect target is not policy-checked.")


class _DDGParser(html.parser.HTMLParser):
    """Parse DuckDuckGo's html.duckduckgo.com result page.

    Result anchors carry class ``result__a``; snippets ``result__snippet``.
    Anchor hrefs are //duckduckgo.com/l/?uddg=<encoded-target> redirects -
    the real URL is extracted from the uddg parameter."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.results: list[dict] = []
        self._in_title = False
        self._in_snippet = False
        self._current: Optional[dict] = None

    @staticmethod
    def _classes(attrs) -> set:
        return set((dict(attrs).get("class") or "").split())

    @staticmethod
    def _real_url(href: str) -> str:
        if href.startswith("//"):
            href = "https:" + href
        parsed = urllib.parse.urlparse(href)
        if parsed.path.startswith("/l/"):
            qs = urllib.parse.parse_qs(parsed.query)
            target = qs.get("uddg", [""])[0]
            if target:
                return target
        return href

    def handle_starttag(self, tag, attrs):
        classes = self._classes(attrs)
        if tag == "a" and "result__a" in classes:
            href = dict(attrs).get("href", "")
            self._current = {"title": "", "url": self._real_url(href),
                             "snippet": ""}
            self._in_title = True
        elif "result__snippet" in classes and self.results:
            self._in_snippet = True

    def handle_endtag(self, tag):
        if self._in_title and tag == "a":
            self._in_title = False
            if self._current and self._current["url"]:
                self.results.append(self._current)
            self._current = None
        elif self._in_snippet and tag in ("a", "div", "td", "span"):
            self._in_snippet = False

    def handle_data(self, data):
        if self._in_title and self._current is not None:
            self._current["title"] += data
        elif self._in_snippet and self.results:
            self.results[-1]["snippet"] += data


class DuckDuckGoHTMLProvider:
    """DuckDuckGo's html.duckduckgo.com endpoint, queried by POST."""

    name = "duckduckgo-html"
    endpoint = "https://html.duckduckgo.com/html/"

    def search(self, query: str, max_results: int) -> list[SearchResult]:
        url = self.endpoint
        netpolicy.check_url(url)
        parsed = urllib.parse.urlparse(url)
        with netpolicy._session_for(url) as session:
            resp = session.post(
                url,
                data={"q": query},
                timeout=netpolicy._DEFAULT_TIMEOUT,
                allow_redirects=False,
                headers={"User-Agent": netpolicy._USER_AGENT,
                         "Host": netpolicy._host_header(parsed)},
            )
            _refuse_redirect(resp, "The DuckDuckGo search backend")
            resp.raise_for_status()
            text = resp.text
        parser = _DDGParser()
        try:
            parser.feed(text)
        except Exception:
            # Malformed results HTML: keep whatever was parsed so far.
            pass
        out: list[SearchResult] = []
        for rank, item in enumerate(parser.results[:max_results], 1):
            out.append(SearchResult(
                title=item["title"].strip()[:_TITLE_CAP],
                url=item["url"],
                snippet=" ".join(item["snippet"].split())[:_SNIPPET_CAP],
                rank=rank,
                provider=self.name,
            ))
        return out


class SearXNGProvider:
    """A SearXNG instance's JSON API (``/search?q=...&format=json``)."""

    name = "searxng"

    def __init__(self, base_url: str):
        self.base_url = str(base_url).rstrip("/")

    def search(self, query: str, max_results: int) -> list[SearchResult]:
        url = (f"{self.base_url}/search?"
               f"{urllib.parse.urlencode({'q': query, 'format': 'json'})}")
        netpolicy.check_url(url)
        parsed = urllib.parse.urlparse(url)
        with netpolicy._session_for(url) as session:
            resp = session.get(url, timeout=netpolicy._DEFAULT_TIMEOUT,
                               allow_redirects=False,
                               headers={"User-Agent": netpolicy._USER_AGENT,
                                        "Host": netpolicy._host_header(parsed)})
            _refuse_redirect(resp, "The SearXNG search backend")
            resp.raise_for_status()
            payload = resp.json()
        if not isinstance(payload, dict):
            raise SearchProviderError(
                "The SearXNG search backend returned a non-object JSON body.")
        items = payload.get("results", [])
        if not isinstance(items, list):
            raise SearchProviderError(
                "The SearXNG search backend returned a malformed results list.")
        out: list[SearchResult] = []
        for rank, item in enumerate(items[:max_results], 1):
            if not isinstance(item, dict):
                continue
            out.append(SearchResult(
                title=str(item.get("title", ""))[:_TITLE_CAP],
                url=str(item.get("url", "")),
                snippet=str(item.get("content", ""))[:_SNIPPET_CAP],
                rank=rank,
                provider=self.name,
            ))
        return out


def provider_from_config(config: Optional[dict] = None) -> SearchProvider:
    """``SearXNGProvider`` when ``net_search_url`` is set in *config* (default:
    the live config), otherwise ``DuckDuckGoHTMLProvider``."""
    cfg = config if config is not None else netpolicy._config()
    base = cfg.get("net_search_url")
    if base:
        return SearXNGProvider(str(base))
    return DuckDuckGoHTMLProvider()


def search(query: str, max_results: int = 5,
           provider: Optional[SearchProvider] = None) -> list[SearchResult]:
    """Run one search. *query* is stripped and must be non-empty
    (``ValueError``); *max_results* is clamped to 1..10. Raises
    ``SearchProviderError`` when the provider returns no results, and lets
    ``NetworkPolicyError`` and transport exceptions propagate."""
    query = (query or "").strip()
    if not query:
        raise ValueError("Empty search query")
    max_results = max(1, min(int(max_results), _MAX_RESULTS))
    if provider is None:
        provider = provider_from_config()
    results = provider.search(query, max_results)
    if not results:
        raise SearchProviderError(NO_RESULTS_MESSAGE)
    return results
