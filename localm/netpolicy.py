"""
Central network-access policy for model-initiated requests.

Every network capability that a *model* can trigger (the coder's fetch_url /
web_search tools, the GUI chat's web access) is routed through this module.
Explicit user actions (``localm pull``, typing ``/web`` in chat) are consent
by definition, but still respect ``net_mode = off`` so one switch really does
kill everything.

Config keys (set via ``localm config``, the GUI Settings page, or /v1/config)
------------------------------------------------------------------------------
net_mode           "off" | "ask" | "allow"   (default "ask")
                   off   - every policy-routed request fails fast
                   ask   - allowed, but surfaces that support confirmation
                           ask first (the coder routes network tools through
                           its destructive-tool approval flow)
                   allow - no confirmation
                   The LOCALM_NET_MODE env var overrides the config value.
net_allow          list of domains (or comma-separated string). Empty = any
                   domain. "example.com" matches example.com and *.example.com.
net_deny           same format; matches are always refused (wins over allow).
net_allow_private  False (default) blocks loopback/private/link-local targets
                   (SSRF guard). True restores access to local dev services.
net_search_url     None = DuckDuckGo HTML (no API key). Or the base URL of a
                   SearXNG instance with the JSON API enabled.

What this module does NOT govern
--------------------------------
Child processes spawned by run_shell (pip, npm, git, …) talk to the network
on their own; the shell-command approval is the gate for those. Model pulls
and online coder providers (OpenAI/Anthropic opt-ins) are explicit user
choices outside this policy.

Limits of the SSRF guard: the hostname is resolved and checked before the
request, but the actual connection re-resolves (a determined DNS-rebinding
attacker could race this). Redirects are re-validated hop by hop.
"""

from __future__ import annotations

import html.parser
import ipaddress
import os
import re
import socket
import urllib.parse
from typing import Optional

NET_MODES = ("off", "ask", "allow")
NET_MODE_ENV_VAR = "LOCALM_NET_MODE"

_DEFAULT_TIMEOUT = 15
_DEFAULT_MAX_BYTES = 1_000_000
_MAX_REDIRECTS = 5
_USER_AGENT = "Mozilla/5.0 (compatible; localm/0.1; +https://github.com/localm)"


class NetworkPolicyError(Exception):
    """A request was refused by the network policy. The message says why
    and how to change the policy - safe to show to the model and the user."""


# ---------------------------------------------------------------------------
#  Policy resolution
# ---------------------------------------------------------------------------

def network_mode() -> str:
    """Resolve the active mode: LOCALM_NET_MODE env > config > "ask"."""
    env = os.environ.get(NET_MODE_ENV_VAR, "").strip().lower()
    if env in NET_MODES:
        return env
    try:
        from localm.config import load_config
        mode = str(load_config().get("net_mode", "ask")).strip().lower()
    except Exception:
        return "ask"
    return mode if mode in NET_MODES else "ask"


def _domain_list(value) -> list[str]:
    """Accept a list or a comma-separated string (CLI sets strings)."""
    if not value:
        return []
    if isinstance(value, str):
        value = value.split(",")
    return [str(d).strip().lower().lstrip("*.") for d in value if str(d).strip()]


def _host_matches(host: str, pattern: str) -> bool:
    """Suffix match: "example.com" covers example.com and api.example.com."""
    host = host.lower().rstrip(".")
    return host == pattern or host.endswith("." + pattern)


def _config() -> dict:
    try:
        from localm.config import load_config
        return load_config()
    except Exception:
        return {}


def check_url(url: str) -> None:
    """
    Validate one URL against the policy. Raises NetworkPolicyError with an
    actionable message when refused; returns silently when allowed.

    Checks, in order: mode, scheme, deny list, allow list, resolved-IP class.
    """
    if network_mode() == "off":
        raise NetworkPolicyError(
            "Network access is disabled (net_mode=off). Enable it with: "
            "localm config net_mode ask")

    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise NetworkPolicyError(
            f"Only http/https URLs are allowed (got '{parsed.scheme}:'). "
            "Reading local files via file:// is not allowed.")
    host = (parsed.hostname or "").lower().rstrip(".")
    if not host:
        raise NetworkPolicyError(f"URL has no host: {url}")

    cfg = _config()
    deny = _domain_list(cfg.get("net_deny"))
    for pattern in deny:
        if _host_matches(host, pattern):
            raise NetworkPolicyError(
                f"'{host}' is on the deny list (net_deny). ")
    allow = _domain_list(cfg.get("net_allow"))
    if allow and not any(_host_matches(host, p) for p in allow):
        raise NetworkPolicyError(
            f"'{host}' is not on the allow list (net_allow). Add it with: "
            f"localm config net_allow \"{', '.join(allow + [host])}\"")

    if not cfg.get("net_allow_private", False):
        _check_public_address(host)


def _is_blocked_ip(ip: ipaddress._BaseAddress) -> bool:
    """True for addresses the SSRF guard refuses (loopback, RFC1918 private,
    link-local incl. 169.254.169.254 cloud metadata, reserved, multicast,
    unspecified)."""
    return bool(ip.is_loopback or ip.is_private or ip.is_link_local
                or ip.is_reserved or ip.is_multicast or ip.is_unspecified)


def _literal_ipv4(host: str) -> Optional[ipaddress.IPv4Address]:
    """If ``host`` is a numeric / short-form IPv4 literal (dotless decimal
    '2130706433', hex '0x7f000001', octal '0177.0.0.1', short '127.1', or the
    plain dotted form), return its canonical IPv4Address. Otherwise None.

    ipaddress.ip_address() refuses the dotless / hex / octal / short forms, and
    socket.getaddrinfo may raise for them, so without this an attacker can hand
    '2130706433' (== 127.0.0.1) to the policy and slip past the public-address
    check (SEC-5). socket.inet_aton parses the historical IPv4 forms; we then
    classify the canonical address it yields. Normal hostnames contain letters
    or dots-with-letters and make inet_aton raise, so they fall through."""
    try:
        packed = socket.inet_aton(host)
    except OSError:
        return None
    try:
        return ipaddress.IPv4Address(packed)
    except ValueError:
        return None


def _check_public_address(host: str) -> None:
    """SSRF guard: refuse hosts that resolve to loopback / private /
    link-local / reserved addresses (cloud metadata, router admin pages,
    the localm API itself…). Unresolvable hosts pass - the fetch will fail
    with a normal DNS error anyway.

    Numeric / short-form IPv4 literals are normalized and classified directly
    (see _literal_ipv4) so they cannot evade the check by being unresolvable or
    unparseable by the ipaddress module."""
    literal = _literal_ipv4(host)
    if literal is not None:
        if _is_blocked_ip(literal):
            raise NetworkPolicyError(
                f"'{host}' is the non-public address {literal}. "
                "Requests to local/private networks are blocked "
                "(set net_allow_private true to permit them).")
        return
    try:
        infos = socket.getaddrinfo(host, None)
    except (socket.gaierror, ValueError):
        return
    for info in infos:
        try:
            ip = ipaddress.ip_address(info[4][0])
        except ValueError:
            continue
        if _is_blocked_ip(ip):
            raise NetworkPolicyError(
                f"'{host}' resolves to the non-public address {ip}. "
                "Requests to local/private networks are blocked "
                "(set net_allow_private true to permit them).")


# ---------------------------------------------------------------------------
#  Fetching
# ---------------------------------------------------------------------------

def safe_fetch(
    url: str,
    *,
    max_bytes: int = _DEFAULT_MAX_BYTES,
    timeout: int = _DEFAULT_TIMEOUT,
) -> tuple[str, str, str]:
    """
    Policy-checked GET. Returns (final_url, content_type, body_text).

    Redirects are followed manually so every hop is re-validated against the
    policy - a public page cannot bounce the agent into 127.0.0.1. The body
    is capped at max_bytes.

    Raises NetworkPolicyError (policy refusal) or requests exceptions.
    """
    import requests

    current = url
    for _ in range(_MAX_REDIRECTS + 1):
        check_url(current)
        resp = requests.get(
            current,
            timeout=timeout,
            stream=True,
            allow_redirects=False,
            headers={"User-Agent": _USER_AGENT},
        )
        try:
            if resp.is_redirect or resp.is_permanent_redirect:
                location = resp.headers.get("Location", "")
                if not location:
                    raise NetworkPolicyError(
                        f"Redirect from {current} without a Location header.")
                current = urllib.parse.urljoin(current, location)
                continue
            resp.raise_for_status()
            content_type = resp.headers.get("Content-Type", "")
            chunks, size = [], 0
            for chunk in resp.iter_content(chunk_size=65536):
                chunks.append(chunk)
                size += len(chunk)
                if size >= max_bytes:
                    break
            body = b"".join(chunks)[:max_bytes]
            return current, content_type, body.decode("utf-8", errors="replace")
        finally:
            resp.close()
    raise NetworkPolicyError(
        f"Too many redirects (>{_MAX_REDIRECTS}) fetching {url}")


class _HTMLStripper(html.parser.HTMLParser):
    """Extract readable text from an HTML document."""

    _SKIP = {"script", "style", "head", "meta", "link", "noscript", "svg",
             "template"}
    # Void elements have no end tag. They must NOT move the skip counter:
    # a <meta>/<link> in <head> would otherwise increment it with no matching
    # decrement, leaving _skip > 0 forever so the whole <body> is dropped and
    # html_to_text returns "" for every normal page.
    _VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input",
             "link", "meta", "param", "source", "track", "wbr"}
    _BLOCK = {"p", "br", "div", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self._buf: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        t = tag.lower()
        if t in self._VOID:
            if t in self._BLOCK:          # <br> -> line break
                self._buf.append("\n")
            return                        # never touch the skip counter
        if t in self._SKIP:
            self._skip += 1
        elif t in self._BLOCK:
            self._buf.append("\n")

    def handle_endtag(self, tag):
        t = tag.lower()
        if t in self._VOID:
            return
        if t in self._SKIP and self._skip:
            self._skip -= 1

    def handle_data(self, data):
        if not self._skip:
            self._buf.append(data)

    def get_text(self) -> str:
        raw = "".join(self._buf)
        raw = re.sub(r"[ \t]+", " ", raw)
        return re.sub(r"\n\s*\n+", "\n\n", raw).strip()


def html_to_text(markup: str) -> str:
    """Best-effort plain text from HTML."""
    stripper = _HTMLStripper()
    try:
        stripper.feed(markup)
    except Exception:
        pass
    return stripper.get_text()


def fetch_text(
    url: str,
    *,
    max_bytes: int = _DEFAULT_MAX_BYTES,
    timeout: int = _DEFAULT_TIMEOUT,
) -> tuple[str, str]:
    """safe_fetch + HTML stripping. Returns (final_url, plain_text)."""
    final_url, content_type, body = safe_fetch(
        url, max_bytes=max_bytes, timeout=timeout)
    if "html" in content_type.lower():
        return final_url, html_to_text(body)
    return final_url, body.strip()


# ---------------------------------------------------------------------------
#  Web search
# ---------------------------------------------------------------------------

def web_search(query: str, max_results: int = 5) -> list[dict]:
    """
    Search the web. Returns [{"title", "url", "snippet"}, ...].

    Backend: a SearXNG instance when net_search_url is configured (its JSON
    API must be enabled), otherwise DuckDuckGo's no-key HTML endpoint.
    Raises NetworkPolicyError when the policy refuses, or RuntimeError when
    the backend yields nothing parseable.
    """
    query = (query or "").strip()
    if not query:
        raise ValueError("Empty search query")
    max_results = max(1, min(int(max_results), 10))

    base = _config().get("net_search_url")
    if base:
        results = _searxng_search(str(base).rstrip("/"), query, max_results)
    else:
        results = _ddg_search(query, max_results)
    if not results:
        raise RuntimeError(
            "The search backend returned no parseable results. It may be "
            "rate-limiting; try again, or configure a SearXNG instance via: "
            "localm config net_search_url http://...")
    return results


def _searxng_search(base: str, query: str, max_results: int) -> list[dict]:
    import requests

    url = f"{base}/search?{urllib.parse.urlencode({'q': query, 'format': 'json'})}"
    check_url(url)
    resp = requests.get(url, timeout=_DEFAULT_TIMEOUT,
                        headers={"User-Agent": _USER_AGENT})
    resp.raise_for_status()
    out = []
    for item in resp.json().get("results", [])[:max_results]:
        out.append({
            "title": str(item.get("title", ""))[:300],
            "url": str(item.get("url", "")),
            "snippet": str(item.get("content", ""))[:500],
        })
    return out


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


def _ddg_search(query: str, max_results: int) -> list[dict]:
    import requests

    check_url("https://html.duckduckgo.com/html/")
    resp = requests.post(   # the HTML endpoint prefers POST for queries
        "https://html.duckduckgo.com/html/",
        data={"q": query},
        timeout=_DEFAULT_TIMEOUT,
        headers={"User-Agent": _USER_AGENT},
    )
    resp.raise_for_status()
    parser = _DDGParser()
    try:
        parser.feed(resp.text)
    except Exception:
        pass
    out = []
    for item in parser.results[:max_results]:
        out.append({
            "title": item["title"].strip()[:300],
            "url": item["url"],
            "snippet": " ".join(item["snippet"].split())[:500],
        })
    return out


def format_results(results: list[dict]) -> str:
    """Stable plain-text rendering shared by the coder tool and the chat."""
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {r['title']}\n   {r['url']}")
        if r.get("snippet"):
            lines.append(f"   {r['snippet']}")
    return "\n".join(lines)
