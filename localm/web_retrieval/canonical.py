# SPDX-License-Identifier: AGPL-3.0-or-later
"""URL canonicalization and duplicate removal for search candidates.

The canonical form is a comparison key. Retrieval fetches the provider's URL
as given; only the fragment-free, tracking-parameter-free, case-normalised
form is used to decide that two candidates are the same page.
"""

from __future__ import annotations

import urllib.parse
from typing import Iterable

from .contracts import SearchResult

_HTTP_SCHEMES = ("http", "https")
_DEFAULT_PORTS = {"http": 80, "https": 443}
_TRACKING_PREFIXES = ("utm_",)
_TRACKING_PARAMS = frozenset({
    "fbclid", "gclid", "gclsrc", "dclid", "msclkid", "mc_cid", "mc_eid",
    "yclid", "igshid", "_ga", "_gl", "mkt_tok", "oly_anon_id", "oly_enc_id",
    "vero_id", "wickedid",
})


def _is_tracking_param(key: str) -> bool:
    k = key.lower()
    return k in _TRACKING_PARAMS or k.startswith(_TRACKING_PREFIXES)


def _split(url: str):
    try:
        return urllib.parse.urlsplit((url or "").strip())
    except ValueError:
        return None


def is_fetchable(url: str) -> bool:
    """True when *url* has an http or https scheme and a host."""
    parts = _split(url)
    return bool(parts and parts.scheme.lower() in _HTTP_SCHEMES
                and parts.hostname)


def canonicalize_url(url: str) -> str:
    """Canonical form of an http(s) URL: lower-case scheme and host, trailing
    dot and default port dropped, empty path made ``/``, a trailing slash on a
    non-root path dropped, tracking query parameters (``utm_*``, ``fbclid``,
    ``gclid``, ...) removed, remaining query pairs sorted, fragment dropped.
    Userinfo is kept. A URL that is not http(s) or does not parse is returned
    stripped of surrounding whitespace, otherwise unchanged."""
    raw = (url or "").strip()
    parts = _split(raw)
    if parts is None or parts.scheme.lower() not in _HTTP_SCHEMES:
        return raw
    scheme = parts.scheme.lower()
    host = (parts.hostname or "").lower().rstrip(".")
    try:
        port = parts.port
    except ValueError:
        port = None
    netloc = host
    if port and port != _DEFAULT_PORTS[scheme]:
        netloc = f"{host}:{port}"
    if parts.username is not None:
        userinfo = parts.username
        if parts.password is not None:
            userinfo += f":{parts.password}"
        netloc = f"{userinfo}@{netloc}"
    path = parts.path or "/"
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/") or "/"
    pairs = [(k, v) for k, v in
             urllib.parse.parse_qsl(parts.query, keep_blank_values=True)
             if not _is_tracking_param(k)]
    pairs.sort()
    query = urllib.parse.urlencode(pairs)
    return urllib.parse.urlunsplit((scheme, netloc, path, query, ""))


def dedup_key(url: str) -> str:
    """The canonical form without scheme and without a leading ``www.`` on the
    host, so ``http://www.a.example/x`` and ``https://a.example/x/`` compare
    equal."""
    parts = _split(canonicalize_url(url))
    if parts is None or parts.scheme.lower() not in _HTTP_SCHEMES:
        return (url or "").strip()
    netloc = parts.netloc
    host_start = netloc.rfind("@") + 1
    if netloc[host_start:].startswith("www."):
        netloc = netloc[:host_start] + netloc[host_start + 4:]
    return urllib.parse.urlunsplit(("", netloc, parts.path, parts.query, ""))


def dedup_results(results: Iterable[SearchResult]) -> list[SearchResult]:
    """Results with duplicates removed, in the order given. The first result
    for a ``dedup_key`` wins, so a survivor keeps its original ``rank``.
    Results whose URL is not fetchable (no http(s) scheme or no host) are
    dropped."""
    seen: set[str] = set()
    out: list[SearchResult] = []
    for r in results:
        if not is_fetchable(r.url):
            continue
        key = dedup_key(r.url)
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out
