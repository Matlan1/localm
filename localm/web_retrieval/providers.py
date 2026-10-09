# SPDX-License-Identifier: AGPL-3.0-or-later
"""Search providers: the built-in no-key search chain and a SearXNG instance.

Every request is policy-checked through ``localm.netpolicy``: ``check_url``
on the request URL, then ``netpolicy._session_for`` (the pinned transport
seam) with ``allow_redirects=False``; any 3xx is refused. Every ``netpolicy``
attribute is read from the module at call time. A request that fails with a
transient transport error (``errors.is_transient``: connection reset, connect
failure, timeout, incomplete body) is sent again on a fresh pinned
connection, at most ``_SEARCH_ATTEMPTS`` times in all and at most twice when
the failure is a timeout, with ``_RETRY_BACKOFF`` seconds between attempts.
A policy refusal, an HTTP status and a TLS verification failure are never
retried. Requests to one search service from this process are spaced at
least ``_MIN_INTERVAL`` seconds apart; requests to other services are not
held up by that spacing. A redirect is refused (``RedirectRefusedError``,
a ``NetworkPolicyError``), never followed.

``DefaultSearchProvider`` (no ``net_search_url`` configured) tries, in order,
DuckDuckGo's HTML page, DuckDuckGo's lite page and Brave Search, moving on
when one fails, answers with a bot check (``BotCheckError``) or returns a
page with no results it can read (``UnreadableResultsError``), and returns
the first service's answer. A page with no results counts as an answer only
when it carries the service's own no-results message; that answer (an empty
list) ends the search. When every service failed, a service that answered
with a bot check is asked once more after ``_BOT_CHECK_WAIT`` seconds. No
service, retry or second try starts with less than ``_MIN_TIME_FOR_ROUTE`` of
the ``_SEARCH_BUDGET`` seconds left, and every request's timeout is capped at
the time left. Each failed service is logged at INFO with its
cause (never the query). When every service failed, ``SearchProviderError``
names each one's cause; when every service was refused by the network
policy, the first ``NetworkPolicyError`` is raised.

``SearXNGProvider`` (``net_search_url`` configured) queries only that
instance: the configured URL is normalised (query, fragment and a trailing
``/search`` removed) and an instance that refuses the JSON format (HTTP 403)
is read through its HTML results page instead, with the same no-results
rule. No results while the instance reports search engines that failed
(``unresponsive_engines`` in its JSON, ``response-error`` rows on its HTML
page) is a failure, not an empty answer. The same ``_SEARCH_BUDGET`` applies.
It never falls back to another search service.
"""

from __future__ import annotations

import contextlib
import contextvars
import html.parser
import logging
import re
import threading
import time
import urllib.parse
from typing import Callable, Iterator, Optional

from localm import netpolicy

from .contracts import SearchProvider, SearchProviderError, SearchResult
from .errors import describe_failure, failure_kind, is_transient

logger = logging.getLogger(__name__)

_MAX_RESULTS = 10
_TITLE_CAP = 300
_SNIPPET_CAP = 500

_SEARCH_ATTEMPTS = 3
_TIMEOUT_ATTEMPTS = 2
_RETRY_BACKOFF = (1.0, 2.0)
_SEARCH_TIMEOUT = 10
_SEARCH_BUDGET = 30.0
_MIN_TIME_FOR_ROUTE = 2.0
_MIN_INTERVAL = 1.0
_BOT_CHECK_WAIT = 3.0
_BOT_CHECK_STATUSES = frozenset({202, 403, 418, 429})
_CHALLENGE_RE = re.compile(r"""id\s*=\s*["']?challenge-form\b""",
                           re.IGNORECASE)
_DDG_NO_RESULTS_RE = re.compile(r"""class\s*=\s*["'][^"']*\bno-results\b""",
                                re.IGNORECASE)
_SEARXNG_NO_RESULTS_RE = re.compile(
    r"""class\s*=\s*["'][^"']*\bdialog-error-block\b""", re.IGNORECASE)
_SEARXNG_ENGINE_ERROR_RE = re.compile(
    r"""class\s*=\s*["'][^"']*\bresponse-error\b""", re.IGNORECASE)
_ENGINES_NAMED = 5
_BROWSER_ACCEPT = ("text/html,application/xhtml+xml,application/xml;q=0.9,"
                   "*/*;q=0.8")

NO_RESULTS_MESSAGE = "The search returned no results."

BOT_CHECK_MESSAGE = (
    "DuckDuckGo answered with a bot check instead of results (it limits "
    "automated searches from one network).")

_pace_lock = threading.Lock()
_last_request: dict[str, float] = {}
_sleep = time.sleep
_clock = time.monotonic
_observer: contextvars.ContextVar[Optional[Callable[[str], None]]] = \
    contextvars.ContextVar("search_request_observer", default=None)


@contextlib.contextmanager
def observe_requests(callback: Optional[Callable[[str], None]]
                     ) -> Iterator[None]:
    """While active (in this context), call *callback* with the URL of every
    search request just before it is sent, after its policy check."""
    token = _observer.set(callback)
    try:
        yield
    finally:
        _observer.reset(token)


def _notify(url: str) -> None:
    callback = _observer.get()
    if callback is not None:
        callback(url)


class BotCheckError(SearchProviderError):
    """A search service answered with a bot check (a challenge page or a
    rate-limit status) instead of results."""


class RedirectRefusedError(netpolicy.NetworkPolicyError):
    """A search service answered with a redirect, which is refused."""


class UnreadableResultsError(SearchProviderError):
    """A search service answered with a page that has no results localm can
    read and no no-results message."""


def _checked_results(items: list[dict], text: str, max_results: int,
                     provider: str, label: str,
                     no_results: Optional[re.Pattern],
                     own_hosts: tuple = ()) -> list[SearchResult]:
    """The results parsed from *text* (see ``_results``); an empty list only
    when *no_results* matches *text*, otherwise ``UnreadableResultsError``."""
    out = _results(items, max_results, provider, own_hosts)
    if not out and (no_results is None or not no_results.search(text)):
        raise UnreadableResultsError(
            f"{label} answered with a page localm could not read results "
            "from.")
    return out


def _pace(service: str) -> None:
    """Sleep until at least ``_MIN_INTERVAL`` seconds have passed since the
    previous request to *service* from this process. The lock is not held
    while sleeping, so a request to another service is not held up."""
    with _pace_lock:
        now = _clock()
        start = max(now, _last_request.get(service, 0.0) + _MIN_INTERVAL)
        _last_request[service] = start
    if start > now:
        _sleep(start - now)


def _with_retries(send: Callable[[], object], *,
                  finish_by: Optional[float] = None,
                  attempts: int = _SEARCH_ATTEMPTS):
    """Call *send* until it returns, retrying a transient failure (see the
    module docstring) at most *attempts* times in all. With *finish_by* (a
    ``_clock()`` value), no retry starts when less than
    ``_MIN_TIME_FOR_ROUTE`` seconds would be left after its backoff. The last
    failure is raised unchanged."""
    attempt = 0
    while True:
        attempt += 1
        try:
            return send()
        except netpolicy.NetworkPolicyError:
            raise
        except Exception as exc:
            limit = (min(_TIMEOUT_ATTEMPTS, attempts)
                     if failure_kind(exc) in ("connect-timeout", "read-timeout")
                     else attempts)
            backoff = _RETRY_BACKOFF[min(attempt, len(_RETRY_BACKOFF)) - 1]
            if not is_transient(exc) or attempt >= limit or (
                    finish_by is not None
                    and finish_by - _clock() - backoff < _MIN_TIME_FOR_ROUTE):
                raise
            _sleep(backoff)


def _refuse_redirect(resp, backend: str) -> None:
    """Raise ``RedirectRefusedError`` when *resp* is a 3xx. Search requests are
    sent with ``allow_redirects=False`` and their redirect target is never
    policy-checked, so a redirect is refused rather than followed."""
    if getattr(resp, "is_redirect", False) or \
            getattr(resp, "is_permanent_redirect", False):
        raise RedirectRefusedError(
            f"{backend} tried to redirect (to "
            f"{resp.headers.get('Location', '?')!r}); refusing - a search "
            "backend's redirect target is not policy-checked.")


def _classes(attrs) -> set:
    return set((dict(attrs).get("class") or "").split())


def _real_url(href: str) -> str:
    """*href* with a DuckDuckGo ``/l/?uddg=<target>`` redirect unwrapped and a
    protocol-relative URL made https."""
    if href.startswith("//"):
        href = "https:" + href
    parsed = urllib.parse.urlparse(href)
    if parsed.path.startswith("/l/"):
        target = urllib.parse.parse_qs(parsed.query).get("uddg", [""])[0]
        if target:
            return target
    return href


def _is_result_url(url: str, own_hosts: tuple = ()) -> bool:
    """True for an http(s) URL whose host is not one of *own_hosts* (the
    search service's own hosts, where its ads and internal links point) or a
    subdomain of one."""
    parsed = urllib.parse.urlparse(url)
    host = (parsed.hostname or "").lower()
    if parsed.scheme not in ("http", "https") or not host:
        return False
    return not any(host == d or host.endswith("." + d) for d in own_hosts)


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

    def handle_starttag(self, tag, attrs):
        classes = _classes(attrs)
        if tag == "a" and "result__a" in classes:
            href = dict(attrs).get("href", "")
            self._current = {"title": "", "url": _real_url(href),
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


class _DDGLiteParser(html.parser.HTMLParser):
    """Parse DuckDuckGo's lite.duckduckgo.com result page: anchors with class
    ``result-link`` and table cells with class ``result-snippet``."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.results: list[dict] = []
        self._in_title = False
        self._in_snippet = False

    def handle_starttag(self, tag, attrs):
        classes = _classes(attrs)
        if tag == "a" and "result-link" in classes:
            self.results.append({"title": "",
                                 "url": _real_url(dict(attrs).get("href", "")),
                                 "snippet": ""})
            self._in_title = True
        elif tag == "td" and "result-snippet" in classes and self.results:
            self._in_snippet = True

    def handle_endtag(self, tag):
        if self._in_title and tag == "a":
            self._in_title = False
        elif self._in_snippet and tag == "td":
            self._in_snippet = False

    def handle_data(self, data):
        if self._in_title:
            self.results[-1]["title"] += data
        elif self._in_snippet:
            self.results[-1]["snippet"] += data


class _BraveParser(html.parser.HTMLParser):
    """Parse a Brave Search result page: each ``div[data-type="web"]`` is a
    result whose first link is its URL, whose ``search-snippet-title``
    element's ``title`` attribute is its title and whose
    ``generic-snippet`` element's text is its snippet."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.results: list[dict] = []
        self._depth = 0
        self._snippet_depth: Optional[int] = None

    def handle_starttag(self, tag, attrs):
        attr = dict(attrs)
        if tag == "div":
            self._depth += 1
        if tag == "div" and attr.get("data-type") == "web":
            self.results.append({"title": "", "url": "", "snippet": ""})
            self._snippet_depth = None
            return
        if not self.results:
            return
        current = self.results[-1]
        classes = _classes(attrs)
        if tag == "a" and not current["url"] and attr.get("href"):
            current["url"] = attr["href"]
        if "search-snippet-title" in classes and not current["title"]:
            current["title"] = attr.get("title") or ""
        if tag == "div" and "generic-snippet" in classes \
                and self._snippet_depth is None and not current["snippet"]:
            self._snippet_depth = self._depth

    def handle_endtag(self, tag):
        if tag != "div":
            return
        if self._snippet_depth is not None and self._depth <= self._snippet_depth:
            self._snippet_depth = None
        self._depth -= 1

    def handle_data(self, data):
        if self._snippet_depth is not None and self.results:
            self.results[-1]["snippet"] += data


class _SearXNGHTMLParser(html.parser.HTMLParser):
    """Parse a SearXNG HTML result page: each ``article.result`` holds an
    ``h3 > a[href]`` (URL and title) and a ``p.content`` (snippet)."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.results: list[dict] = []
        self._in_h3 = False
        self._in_title = False
        self._in_content = False

    def handle_starttag(self, tag, attrs):
        classes = _classes(attrs)
        if tag == "article" and "result" in classes:
            self.results.append({"title": "", "url": "", "snippet": ""})
        elif not self.results:
            return
        elif tag == "h3":
            self._in_h3 = True
        elif tag == "a" and self._in_h3 and not self.results[-1]["url"]:
            self.results[-1]["url"] = dict(attrs).get("href", "")
            self._in_title = True
        elif tag == "p" and "content" in classes \
                and "empty_element" not in classes:
            self._in_content = True

    def handle_endtag(self, tag):
        if tag == "a":
            self._in_title = False
        elif tag == "h3":
            self._in_h3 = False
        elif tag == "p":
            self._in_content = False

    def handle_data(self, data):
        if self._in_title:
            self.results[-1]["title"] += data
        elif self._in_content:
            self.results[-1]["snippet"] += data


def _results(items: list[dict], max_results: int, provider: str,
             own_hosts: tuple = ()) -> list[SearchResult]:
    """At most *max_results* ``SearchResult`` from parsed *items*, skipping
    URLs that ``_is_result_url`` rejects for *own_hosts*."""
    out: list[SearchResult] = []
    for item in items:
        url = _real_url(str(item.get("url") or "").strip())
        if not _is_result_url(url, own_hosts):
            continue
        out.append(SearchResult(
            title=" ".join(str(item.get("title") or "").split())[:_TITLE_CAP],
            url=url,
            snippet=" ".join(str(item.get("snippet") or "").split())[:_SNIPPET_CAP],
            rank=len(out) + 1,
            provider=provider,
        ))
        if len(out) >= max_results:
            break
    return out


def _parse(parser: html.parser.HTMLParser, text: str) -> list[dict]:
    try:
        parser.feed(text)
    except Exception:
        # Malformed results HTML: keep whatever was parsed so far.
        pass
    return parser.results


def _request(method: str, url: str, *, service: str, label: str,
             timeout: float, headers: dict, data: Optional[dict] = None,
             bot_statuses: frozenset = frozenset(),
             finish_by: Optional[float] = None):
    """One policy-checked, pinned, paced request; returns the response after
    the redirect refusal, the bot-check status check (``BotCheckError``) and
    ``raise_for_status``. With *finish_by*, *timeout* is capped at the time
    left after pacing (at least one second)."""
    netpolicy.check_url(url)
    parsed = urllib.parse.urlparse(url)
    _pace(service)
    if finish_by is not None:
        timeout = min(timeout, max(finish_by - _clock(), 1.0))
    _notify(url)
    with netpolicy._session_for(url) as session:
        send = session.post if method == "POST" else session.get
        kwargs = {"timeout": timeout, "allow_redirects": False,
                  "headers": {**headers, "User-Agent": netpolicy._USER_AGENT,
                              "Host": netpolicy._host_header(parsed)}}
        if data is not None:
            kwargs["data"] = data
        resp = send(url, **kwargs)
        _refuse_redirect(resp, f"The {label} search backend")
        if getattr(resp, "status_code", None) in bot_statuses:
            raise BotCheckError(
                f"{label} answered with a bot check instead of results")
        resp.raise_for_status()
        return resp


class DuckDuckGoHTMLProvider:
    """DuckDuckGo's html.duckduckgo.com endpoint, queried by POST. A bot
    check raises ``BotCheckError`` with ``BOT_CHECK_MESSAGE``."""

    name = "duckduckgo-html"
    label = "DuckDuckGo"
    service = "duckduckgo"
    endpoint = "https://html.duckduckgo.com/html/"

    def search(self, query: str, max_results: int,
               timeout: float = _SEARCH_TIMEOUT,
               finish_by: Optional[float] = None) -> list[SearchResult]:
        def send() -> str:
            try:
                resp = _request(
                    "POST", self.endpoint, service=self.service,
                    label=self.label, timeout=timeout,
                    data={"q": query, "b": "", "kl": "wt-wt"},
                    headers={"Accept": _BROWSER_ACCEPT,
                             "Accept-Language": "en-US,en;q=0.9",
                             "Referer": "https://html.duckduckgo.com/"},
                    bot_statuses=_BOT_CHECK_STATUSES, finish_by=finish_by)
            except BotCheckError as e:
                raise BotCheckError(BOT_CHECK_MESSAGE) from e
            text = resp.text
            if _CHALLENGE_RE.search(text):
                raise BotCheckError(BOT_CHECK_MESSAGE)
            return text
        text = _with_retries(send, finish_by=finish_by)
        return _checked_results(_parse(_DDGParser(), text), text, max_results,
                                self.name, self.label, _DDG_NO_RESULTS_RE,
                                ("duckduckgo.com",))


class DuckDuckGoLiteProvider:
    """DuckDuckGo's lite.duckduckgo.com endpoint, queried by POST."""

    name = "duckduckgo-lite"
    label = "DuckDuckGo lite"
    service = "duckduckgo"
    endpoint = "https://lite.duckduckgo.com/lite/"

    def search(self, query: str, max_results: int,
               timeout: float = _SEARCH_TIMEOUT,
               finish_by: Optional[float] = None) -> list[SearchResult]:
        def send() -> str:
            resp = _request(
                "POST", self.endpoint, service=self.service, label=self.label,
                timeout=timeout, data={"q": query, "kl": "wt-wt"},
                headers={"Accept": _BROWSER_ACCEPT,
                         "Accept-Language": "en-US,en;q=0.9",
                         "Referer": "https://lite.duckduckgo.com/"},
                bot_statuses=_BOT_CHECK_STATUSES, finish_by=finish_by)
            text = resp.text
            if _CHALLENGE_RE.search(text):
                raise BotCheckError(
                    f"{self.label} answered with a bot check instead of results")
            return text
        text = _with_retries(send, finish_by=finish_by)
        return _checked_results(_parse(_DDGLiteParser(), text), text,
                                max_results, self.name, self.label,
                                _DDG_NO_RESULTS_RE, ("duckduckgo.com",))


class BraveSearchProvider:
    """Brave Search's HTML result page (``search.brave.com/search``)."""

    name = "brave"
    label = "Brave Search"
    service = "brave"
    endpoint = "https://search.brave.com/search"

    def search(self, query: str, max_results: int,
               timeout: float = _SEARCH_TIMEOUT,
               finish_by: Optional[float] = None) -> list[SearchResult]:
        url = (f"{self.endpoint}?"
               f"{urllib.parse.urlencode({'q': query, 'source': 'web'})}")

        def send() -> str:
            resp = _request(
                "GET", url, service=self.service, label=self.label,
                timeout=timeout,
                headers={"Accept": _BROWSER_ACCEPT,
                         "Accept-Language": "en-US,en;q=0.9"},
                bot_statuses=frozenset({403, 429}), finish_by=finish_by)
            return resp.text
        text = _with_retries(send, finish_by=finish_by)
        items = _parse(_BraveParser(), text)
        if not items and "captcha" in text.lower():
            raise BotCheckError(
                f"{self.label} answered with a bot check instead of results")
        return _checked_results(items, text, max_results, self.name,
                                self.label, None, ("search.brave.com",))


def _cause(exc: BaseException, url: str) -> str:
    """One service's failure as a plain clause, without the query."""
    if isinstance(exc, RedirectRefusedError):
        return "it answered with a redirect, which localm does not follow"
    if isinstance(exc, netpolicy.NetworkPolicyError):
        return "blocked by the network policy"
    if isinstance(exc, BotCheckError):
        return "it answered with a bot check instead of results"
    if isinstance(exc, UnreadableResultsError):
        return "it answered with a page localm could not read results from"
    return describe_failure(exc, url)


class DefaultSearchProvider:
    """The built-in no-key search chain (see the module docstring). ``name``
    is the first service's; each result names the service that produced
    it."""

    name = "duckduckgo-html"

    def __init__(self, routes: Optional[list] = None):
        self.routes = routes if routes is not None else [
            DuckDuckGoHTMLProvider(), DuckDuckGoLiteProvider(),
            BraveSearchProvider()]

    def search(self, query: str, max_results: int) -> list[SearchResult]:
        finish_by = _clock() + _SEARCH_BUDGET
        outcomes: dict[int, Optional[BaseException]] = {}

        def attempt(index: int) -> Optional[list[SearchResult]]:
            """The route's answer (an empty list for a no-results answer), or
            None when it failed or less than ``_MIN_TIME_FOR_ROUTE`` was
            left to start it."""
            if finish_by - _clock() < _MIN_TIME_FOR_ROUTE:
                return None
            route = self.routes[index]
            try:
                found = route.search(query, max_results, finish_by=finish_by)
            except Exception as exc:
                outcomes[index] = exc
                logger.info("web search: %s failed: %s", route.label,
                            _cause(exc, getattr(route, "endpoint", "")))
                return None
            outcomes[index] = None
            return found

        for i in range(len(self.routes)):
            found = attempt(i)
            if found is not None:
                return found
        checked = [i for i, exc in outcomes.items()
                   if isinstance(exc, BotCheckError)]
        if checked and (finish_by - _clock()
                        > _BOT_CHECK_WAIT + _MIN_TIME_FOR_ROUTE):
            _sleep(_BOT_CHECK_WAIT)
            for i in checked:
                found = attempt(i)
                if found is not None:
                    return found
        failures = [(self.routes[i], outcomes.get(i))
                    for i in range(len(self.routes))]
        tried = [(r, e) for r, e in failures if e is not None]
        if tried and all(isinstance(e, netpolicy.NetworkPolicyError)
                         for _, e in tried) and len(tried) == len(failures):
            raise tried[0][1]
        parts = []
        for route, exc in failures:
            cause = (_cause(exc, getattr(route, "endpoint", ""))
                     if exc is not None else "not tried: out of time")
            parts.append(f"{route.label}: {cause}")
        message = ("Web search failed on every search service localm tried ("
                   + "; ".join(parts) + ").")
        if any(isinstance(e, BotCheckError) for _, e in failures):
            message += (" These services limit automated searches from one "
                        "network; a self-hosted SearXNG search backend can be "
                        "set under Settings > Network.")
        raise SearchProviderError(message)


def _searxng_base(raw: str) -> str:
    """*raw* without query, fragment, trailing slashes and a trailing
    ``/search`` path segment."""
    parsed = urllib.parse.urlparse(str(raw).strip())
    path = parsed.path.rstrip("/")
    if path.endswith("/search"):
        path = path[: -len("/search")]
    return urllib.parse.urlunparse(
        (parsed.scheme, parsed.netloc, path, "", "", "")).rstrip("/")


class SearXNGProvider:
    """A SearXNG instance: its JSON API (``/search?q=...&format=json``), or
    its HTML results page when the instance refuses the JSON format (HTTP
    403)."""

    name = "searxng"
    label = "SearXNG"

    def __init__(self, base_url: str):
        self.base_url = _searxng_base(base_url)

    def _get(self, url: str, accept: str, finish_by: float):
        netpolicy.check_url(url)
        parsed = urllib.parse.urlparse(url)
        _notify(url)
        timeout = min(_SEARCH_TIMEOUT, max(finish_by - _clock(), 1.0))
        with netpolicy._session_for(url) as session:
            resp = session.get(url, timeout=timeout,
                               allow_redirects=False,
                               headers={"User-Agent": netpolicy._USER_AGENT,
                                        "Accept": accept,
                                        "Host": netpolicy._host_header(parsed)})
            _refuse_redirect(resp, "The SearXNG search backend")
            resp.raise_for_status()
            return resp

    def _label(self) -> str:
        host = urllib.parse.urlparse(self.base_url).hostname or self.base_url
        return f"The search backend set in Settings > Network ({host})"

    def search(self, query: str, max_results: int,
               finish_by: Optional[float] = None) -> list[SearchResult]:
        if finish_by is None:
            finish_by = _clock() + _SEARCH_BUDGET
        json_url = (f"{self.base_url}/search?"
                    f"{urllib.parse.urlencode({'q': query, 'format': 'json'})}")
        try:
            payload = _with_retries(
                lambda: self._get(json_url, "application/json",
                                  finish_by).json(),
                finish_by=finish_by)
        except Exception as exc:
            if failure_kind(exc) != "http" or \
                    getattr(getattr(exc, "response", None), "status_code",
                            None) != 403:
                raise
            return self._search_html(query, max_results, finish_by)
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
        failed = payload.get("unresponsive_engines")
        if not out and isinstance(failed, list) and failed:
            named = []
            for entry in failed[:_ENGINES_NAMED]:
                if isinstance(entry, (list, tuple)) and entry:
                    reason = (f" ({str(entry[1])[:60]})"
                              if len(entry) > 1 and entry[1] else "")
                    named.append(f"{str(entry[0])[:60]}{reason}")
                else:
                    named.append(str(entry)[:60])
            more = (f" and {len(failed) - _ENGINES_NAMED} more"
                    if len(failed) > _ENGINES_NAMED else "")
            raise SearchProviderError(
                f"{self._label()} returned no results, and these of its "
                f"search engines failed: {', '.join(named)}{more}.")
        return out

    def _search_html(self, query: str, max_results: int,
                     finish_by: float) -> list[SearchResult]:
        html_url = (f"{self.base_url}/search?"
                    f"{urllib.parse.urlencode({'q': query})}")
        text = _with_retries(
            lambda: self._get(html_url, _BROWSER_ACCEPT, finish_by).text,
            finish_by=finish_by)
        items = _parse(_SearXNGHTMLParser(), text)
        errors = len(_SEARXNG_ENGINE_ERROR_RE.findall(text))
        if errors and not _results(items, max_results, self.name):
            raise SearchProviderError(
                f"{self._label()} returned no results, and {errors} of its "
                "search engines reported an error.")
        return _checked_results(items, text, max_results, self.name,
                                self._label(), _SEARXNG_NO_RESULTS_RE)


def provider_from_config(config: Optional[dict] = None) -> SearchProvider:
    """``SearXNGProvider`` when ``net_search_url`` is set in *config* (default:
    the live config), otherwise ``DefaultSearchProvider``."""
    cfg = config if config is not None else netpolicy._config()
    base = cfg.get("net_search_url")
    if base:
        return SearXNGProvider(str(base))
    return DefaultSearchProvider()


def search(query: str, max_results: int = 5,
           provider: Optional[SearchProvider] = None) -> list[SearchResult]:
    """Run one search. *query* is stripped and must be non-empty
    (``ValueError``); *max_results* is clamped to 1..10. Raises
    ``SearchProviderError`` when the provider returns no results or fails
    (see the provider), and ``NetworkPolicyError`` on a policy refusal."""
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
