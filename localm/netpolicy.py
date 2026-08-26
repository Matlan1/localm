# SPDX-License-Identifier: AGPL-3.0-or-later
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

SSRF guard: the hostname is resolved and validated once, then the socket is
pinned to that IP (see netpin.py), so the connection cannot
re-resolve to a rebound address and an unresolvable host fails closed.
Redirects are re-validated hop by hop.
"""

from __future__ import annotations

import html.parser
import ipaddress
import os
import re
import socket
import urllib.parse
from typing import Optional

from localm.debuglog import logger

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
    """Resolve the active mode: LOCALM_NET_MODE env > config > "ask".

    On a config-read failure the mode resolves to "off", NOT "ask": returning
    "ask" would silently RE-ENABLE network access for a user who set
    net_mode="off" as a kill switch - the exact fail-open a safety toggle must
    never do. Failing closed (and warning) keeps the switch honest; a transiently
    unreadable config errs toward no network, never toward more. The valid-config
    path is unchanged: an unset or unrecognised value still resolves to "ask"."""
    env = os.environ.get(NET_MODE_ENV_VAR, "").strip().lower()
    if env in NET_MODES:
        return env
    try:
        from localm.config import load_config
        mode = str(load_config().get("net_mode", "ask")).strip().lower()
    except Exception as exc:
        logger.warning("netpolicy: could not load config (%s); resolving "
                       "net_mode to 'off' (fail-safe) so an unreadable config "
                       "cannot silently re-enable network access", exc)
        return "off"
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
    """Best-effort config read for callers OTHER than check_url.

    check_url reads the config itself, once, up front, and refuses outright
    on a read failure - it must never reach this fallback. The
    remaining callers are _resolve_pinned (net_allow_private) and web_search
    (net_search_url); neither reads net_deny/net_allow, so an unreadable
    config here only means those two settings fall back to their safe
    defaults (False / unset) for this call - never a dropped deny list."""
    try:
        from localm.config import load_config
        return load_config()
    except Exception as exc:
        logger.warning("netpolicy: could not load config (%s); using "
                       "defaults for this call", exc)
        return {}


def check_url(url: str) -> None:
    """
    Validate one URL against the policy. Raises NetworkPolicyError with an
    actionable message when refused; returns silently when allowed.

    Checks, in order: mode, malformed-authority, scheme, deny list, allow list,
    resolved-IP class.

    Reads the config exactly ONCE, up front: net_mode and the
    net_deny/net_allow lists must come from the same snapshot, so a read
    failure has exactly one outcome - refuse - no matter what LOCALM_NET_MODE
    says. Resolving mode and lists from two separate reads would let an env
    override reach past a transient config-read failure and silently drop the
    user's explicit deny list while still letting the request through.
    """
    env = os.environ.get(NET_MODE_ENV_VAR, "").strip().lower()
    try:
        from localm.config import load_config
        cfg = load_config()
    except Exception as exc:
        logger.warning(
            "netpolicy: could not load config (%s); refusing this request "
            "(fail-safe) rather than resolving net_mode and net_deny/"
            "net_allow from different reads", exc)
        raise NetworkPolicyError(
            "Network policy configuration could not be read; refusing this "
            "request as a precaution. Retry once the config is readable.")

    mode = env if env in NET_MODES else str(cfg.get("net_mode", "ask")).strip().lower()
    if mode not in NET_MODES:
        mode = "ask"
    if mode == "off":
        raise NetworkPolicyError(
            "Network access is disabled (net_mode=off). Enable it with:  "
            "localm config net_mode ask")

    # Parser-differential SSRF guard: urllib.parse and requests/urllib3 disagree
    # on backslashes and raw control characters in the authority, so a URL whose
    # userinfo ends in a backslash parses here as the public host but connects to
    # 127.0.0.1. Any raw backslash or control character is refused, anywhere in
    # the URL; a conformant http(s) URL percent-encodes them.
    if "\\" in url or any(ord(c) < 0x20 or ord(c) == 0x7F for c in url):
        raise NetworkPolicyError(
            "URL contains a backslash or control character; refusing it "
            "(possible SSRF parser differential).")

    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise NetworkPolicyError(
            f"Only http/https URLs are allowed (got '{parsed.scheme}:'). "
            "Reading local files via file:// is not allowed.")
    host = (parsed.hostname or "").lower().rstrip(".")
    if not host:
        raise NetworkPolicyError(f"URL has no host: {url}")

    deny = _domain_list(cfg.get("net_deny"))
    for pattern in deny:
        if _host_matches(host, pattern):
            raise NetworkPolicyError(
                f"'{host}' is on the deny list (net_deny). ")
    allow = _domain_list(cfg.get("net_allow"))
    if allow and not any(_host_matches(host, p) for p in allow):
        raise NetworkPolicyError(
            f"'{host}' is not on the allow list (net_allow). Add it with:  "
            f"localm config net_allow "
            f"\"{', '.join(allow + [host])}\"")

    if not cfg.get("net_allow_private", False):
        _check_public_address(host)


# Special-use ranges the stdlib marks is_global=True but which are not ordinary
# public hosts. The deprecated 6to4 relay anycast prefix routes to whatever 6to4
# relay the local network advertises.
_EXTRA_BLOCKED_NETS = (
    ipaddress.ip_network("192.88.99.0/24"),   # 6to4 relay anycast (deprecated)
    ipaddress.ip_network("2002::/16"),         # 6to4
)


def _is_blocked_ip(ip: ipaddress._BaseAddress) -> bool:
    """True for addresses the SSRF guard refuses: anything that is not a
    globally-routable public address.

    ``not ip.is_global`` is the primary predicate. It rejects loopback, RFC1918
    private, link-local (incl. 169.254.169.254 cloud metadata), the CGNAT shared
    space 100.64.0.0/10 (RFC 6598, which the stdlib does NOT mark is_private on
    every version), benchmarking,
    documentation and other special-use ranges in one shot, and it stays correct
    as the stdlib adds new reserved ranges.

    The explicit special-use flags are KEPT as a belt-and-suspenders catch: the
    stdlib quirkily marks a few deprecated IPv6 forms (IPv4-compatible
    ``::127.0.0.1``, NAT64-embedded ``64:ff9b::7f00:1``) is_global=True even
    though they still route to internal IPv4 - is_reserved catches those.
    _EXTRA_BLOCKED_NETS covers the residual special-use ranges (6to4 anycast)
    that is_global=True still misses. A genuine public address (is_global True
    with no special-use flag, outside the extra nets) is the only thing that
    passes."""
    return bool(not ip.is_global
                or ip.is_loopback or ip.is_private or ip.is_link_local
                or ip.is_reserved or ip.is_multicast or ip.is_unspecified
                or any(ip.version == net.version and ip in net
                       for net in _EXTRA_BLOCKED_NETS))


def _literal_ipv4(host: str) -> Optional[ipaddress.IPv4Address]:
    """If ``host`` is a numeric / short-form IPv4 literal (dotless decimal
    '2130706433', hex '0x7f000001', octal '0177.0.0.1', short '127.1', or the
    plain dotted form), return its canonical IPv4Address. Otherwise None.

    ipaddress.ip_address() refuses the dotless / hex / octal / short forms, and
    socket.getaddrinfo may raise for them, so without this an attacker can hand
    '2130706433' (== 127.0.0.1) to the policy and slip past the public-address
    check. socket.inet_aton parses the historical IPv4 forms; the canonical
    address it yields is then classified. Normal hostnames contain letters
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


def _resolve_pinned(host: str) -> Optional[str]:
    """Resolve *host* to ONE IP to pin the connection to, closing the
    check-and-connect DNS-rebinding TOCTOU. ``check_url`` resolves
    and validates the host, but ``requests`` re-resolves at connect time, so a
    TTL-0 attacker can answer 'public' for the check and 'internal' for the
    connect. Here we resolve ONCE, validate the address(es), and return the exact
    IP the socket will dial - there is no second lookup to poison.

    Returns the canonical IP string to pin, or None when the host is unresolvable
    (the caller then lets the request fail with a normal DNS error - nothing
    connects, so there is no race). Numeric/short-form and IPv6 literals are
    pinned directly (already validated by check_url). When net_allow_private is
    False, an address that fails the SSRF class check is refused HERE too, on the
    exact IP to be dialled - this is what catches a rebind that slipped past
    check_url's separate lookup."""
    allow_private = bool(_config().get("net_allow_private", False))

    def _guard(ip_obj) -> None:
        if not allow_private and _is_blocked_ip(ip_obj):
            raise NetworkPolicyError(
                f"'{host}' resolves to the non-public address {ip_obj}. "
                "Requests to local/private networks are blocked "
                "(set net_allow_private true to permit them).")

    literal = _literal_ipv4(host)
    if literal is not None:
        _guard(literal)
        return str(literal)
    try:
        ip_obj = ipaddress.ip_address(host)
        _guard(ip_obj)
        return str(ip_obj)
    except ValueError:
        pass

    try:
        infos = socket.getaddrinfo(host, None)
    except (socket.gaierror, ValueError, OSError):
        return None
    blocked = None
    for info in infos:
        try:
            ip_obj = ipaddress.ip_address(info[4][0])
        except ValueError:
            continue
        if not allow_private and _is_blocked_ip(ip_obj):
            blocked = ip_obj          # remember, keep scanning for a usable one
            continue
        return str(ip_obj)            # first usable (and now pinned) address
    if blocked is not None:
        raise NetworkPolicyError(
            f"'{host}' resolves to the non-public address {blocked}. "
            "Requests to local/private networks are blocked "
            "(set net_allow_private true to permit them).")
    return None


def _host_header(parsed) -> str:
    """The Host header value for a pinned request: the original hostname (so
    virtual-host routing survives the IP pin), with the port only when it is
    non-default for the scheme."""
    host = parsed.hostname or ""
    port = parsed.port
    if port and not ((parsed.scheme == "http" and port == 80)
                     or (parsed.scheme == "https" and port == 443)):
        return f"{host}:{port}"
    return host


def _session_for(url: str):
    """A ``requests.Session`` whose socket is pinned to *url*'s pre-validated IP
    ``check_url`` MUST already have passed on *url*. This is the
    single network-transport seam: production pins here, tests double it here.
    The caller sends ``_host_header(url)`` as the Host header and closes the
    session (use it as a context manager).

    Fails CLOSED when the host cannot be resolved to a validated address: we do
    NOT fall back to a re-resolving session. Otherwise a host that is NXDOMAIN at
    validation time (check_url lets unresolvable hosts through) but flips to a
    private A record at connect time (TTL-0 DNS rebinding) would reach an internal
    service unvalidated - the exact hole this closes. A genuinely unresolvable
    host cannot be connected to anyway, so refusing costs nothing legitimate."""
    from localm import netpin
    parsed = urllib.parse.urlparse(url)
    ip = _resolve_pinned(parsed.hostname or "")
    if not ip:
        raise NetworkPolicyError(
            f"Could not resolve '{parsed.hostname}' to an address; refusing the "
            "request rather than connecting through an unvalidated re-resolution.")
    return netpin.pinned_session(ip)


def pinned_request(method: str, url: str, **kwargs):
    """A single policy-pinned HTTP request for callers that manage
    their own response (streamed downloads, HEAD probes) instead of going through
    safe_fetch_bytes. ``check_url`` MUST already have passed on *url*.

    The socket is pinned to the pre-validated IP and the original hostname is sent
    as the ``Host`` header. The pinned session is attached to the returned response
    (``resp._localm_pin_session``) so a streamed body stays usable until the caller
    is done with the response; it is released when the response is GC'd, matching
    how requests' own streamed responses are managed. Raises NetworkPolicyError
    when the host cannot be resolved to a validated address (fail-closed)."""
    session = _session_for(url)
    headers = {**(kwargs.pop("headers", None) or {}),
               "Host": _host_header(urllib.parse.urlparse(url))}
    resp = session.request(method, url, headers=headers, **kwargs)
    resp._localm_pin_session = session   # tie the session lifetime to the response
    return resp


# ---------------------------------------------------------------------------
#  Fetching
# ---------------------------------------------------------------------------

def safe_fetch_bytes(
    url: str,
    *,
    max_bytes: int = _DEFAULT_MAX_BYTES,
    timeout: int = _DEFAULT_TIMEOUT,
) -> tuple[str, str, bytes]:
    """
    Policy-checked GET returning RAW bytes. Returns (final_url, content_type,
    body_bytes).

    Same protections as safe_fetch - redirects are followed manually so every
    hop is re-validated against the policy (a public page cannot bounce the
    fetch into 127.0.0.1), and the body is capped at max_bytes - but the body is
    NOT decoded, so this is the entry point for binary payloads (images fetched
    for vision input). Text callers go through safe_fetch / fetch_text.

    Raises NetworkPolicyError (policy refusal) or requests exceptions.
    """
    current = url
    for _ in range(_MAX_REDIRECTS + 1):
        check_url(current)
        parsed = urllib.parse.urlparse(current)
        # Pin the socket to the just-validated IP for this hop;
        # each redirect target is independently re-checked and re-pinned.
        with _session_for(current) as session:
            resp = session.get(
                current,
                timeout=timeout,
                stream=True,
                allow_redirects=False,
                headers={"User-Agent": _USER_AGENT, "Host": _host_header(parsed)},
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
                return current, content_type, b"".join(chunks)[:max_bytes]
            finally:
                resp.close()
    raise NetworkPolicyError(
        f"Too many redirects (>{_MAX_REDIRECTS}) fetching {url}")


def safe_fetch(
    url: str,
    *,
    max_bytes: int = _DEFAULT_MAX_BYTES,
    timeout: int = _DEFAULT_TIMEOUT,
) -> tuple[str, str, str]:
    """
    Policy-checked GET. Returns (final_url, content_type, body_text).

    Thin text wrapper over safe_fetch_bytes (which does the policy check,
    per-hop redirect re-validation and size cap); the body is decoded as UTF-8.

    Raises NetworkPolicyError (policy refusal) or requests exceptions.
    """
    final_url, content_type, body = safe_fetch_bytes(
        url, max_bytes=max_bytes, timeout=timeout)
    return final_url, content_type, body.decode("utf-8", errors="replace")


class _HTMLStripper(html.parser.HTMLParser):
    """Extract readable text from an HTML document."""

    _SKIP = {"script", "style", "head", "meta", "link", "noscript", "svg",
             "template"}
    # Void elements have no end tag and must not move the skip counter: an
    # increment with no matching decrement would leave _skip > 0 forever and drop
    # the whole body.
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
        # Best-effort: HTMLParser can choke on malformed markup, so whatever was
        # parsed before the error is returned.
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
            "rate-limiting; try again, or set a Search backend URL (SearXNG) with:  "
            "localm config net_search_url http://...")
    return results


def _refuse_redirect(resp, backend: str) -> None:
    """The search backends call check_url ONCE on the request URL, so - unlike
    safe_fetch - they cannot re-validate a redirect target per hop. requests
    follows redirects by default, which would let a 3xx from the search host
    bounce the GET into 127.0.0.1 / 169.254.169.254 / an RFC1918 service with no
    policy check (SSRF). So the callers pass allow_redirects=False and we refuse
    any 3xx outright (surfacing it rather than silently following an unchecked
    hop) - the search backend is expected to answer directly."""
    # getattr default: a real requests.Response always exposes these properties;
    # the default only applies to minimal test doubles standing in for a 200.
    if getattr(resp, "is_redirect", False) or \
            getattr(resp, "is_permanent_redirect", False):
        raise NetworkPolicyError(
            f"{backend} tried to redirect (to "
            f"{resp.headers.get('Location', '?')!r}); refusing - a search "
            "backend's redirect target is not policy-checked.")


def _searxng_search(base: str, query: str, max_results: int) -> list[dict]:
    url = f"{base}/search?{urllib.parse.urlencode({'q': query, 'format': 'json'})}"
    check_url(url)
    parsed = urllib.parse.urlparse(url)
    with _session_for(url) as session:   # pinned to the validated IP
        resp = session.get(url, timeout=_DEFAULT_TIMEOUT, allow_redirects=False,
                           headers={"User-Agent": _USER_AGENT,
                                    "Host": _host_header(parsed)})
        _refuse_redirect(resp, "The SearXNG search backend")
        resp.raise_for_status()
        items = resp.json().get("results", [])[:max_results]
    out = []
    for item in items:
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
    url = "https://html.duckduckgo.com/html/"
    check_url(url)
    parsed = urllib.parse.urlparse(url)
    with _session_for(url) as session:   # pinned to the validated IP
        resp = session.post(   # the HTML endpoint prefers POST for queries
            url,
            data={"q": query},
            timeout=_DEFAULT_TIMEOUT,
            allow_redirects=False,
            headers={"User-Agent": _USER_AGENT, "Host": _host_header(parsed)},
        )
        _refuse_redirect(resp, "The DuckDuckGo search backend")
        resp.raise_for_status()
        text = resp.text
    parser = _DDGParser()
    try:
        parser.feed(text)
    except Exception:
        # Best-effort: on malformed results HTML, return whatever parsed. The HTTP
        # status was already checked above, so this guards only the scrape.
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
