# SPDX-License-Identifier: AGPL-3.0-or-later
"""Content endpoints for sites whose HTML page is a poor or blocked read.

``read_url`` reads a URL through a ``fetch`` callable. For a recognised URL it
first tries the site's own content endpoint and falls back to the page itself:

- ``github.com/<owner>/<repo>``: the README from
  ``raw.githubusercontent.com/<owner>/<repo>/HEAD/README.md``, then from the
  GitHub REST ``/repos/<owner>/<repo>/readme`` endpoint (any README name).
- ``github.com/<owner>/<repo>/tree/<ref>[/<dir>]``: that directory's README
  the same way, at *ref*; the path segment after ``tree`` is taken as the
  whole ref.
- ``github.com/<owner>/<repo>/blob/<ref>/<path>``: the raw file from
  ``raw.githubusercontent.com``.
- a Stack Overflow / Stack Exchange question URL: the question and its top
  answers from the Stack Exchange API (``api.stackexchange.com``).

The original URL must pass ``netpolicy.check_url`` before any content
endpoint is tried, and every request goes through *fetch*, so with the default
fetch every endpoint URL is policy-checked and pinned by ``localm.netpolicy``
as well. A failing content
endpoint (policy refusal included) moves on to the next one and finally to
the page itself; only the page's own failure is raised.
"""

from __future__ import annotations

import base64
import binascii
import html
import json
import re
import urllib.parse
from dataclasses import dataclass
from typing import Callable, Optional

from localm.debuglog import logger

#: ``fetch(url, timeout=...) -> (final_url, content_type, text)``.
Fetcher = Callable[..., tuple[str, str, str]]

CONTENT_TYPE_TEXT = "text/plain; charset=utf-8"

_GITHUB_HOSTS = frozenset({"github.com", "www.github.com"})
#: First path segments of github.com that are site pages, not owners.
_GITHUB_RESERVED = frozenset({
    "about", "apps", "codespaces", "collections", "contact", "customer-stories",
    "enterprise", "events", "explore", "features", "issues", "login", "logout",
    "marketplace", "new", "notifications", "orgs", "organizations", "pricing",
    "pulls", "search", "security", "settings", "site", "sponsors", "team",
    "topics", "trending", "users",
})
_GITHUB_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]{1,100}$")

#: Stack Exchange hosts with an API site name other than their first label.
_STACKEXCHANGE_SITES = {
    "stackoverflow.com": "stackoverflow",
    "superuser.com": "superuser",
    "serverfault.com": "serverfault",
    "askubuntu.com": "askubuntu",
    "mathoverflow.net": "mathoverflow.net",
    "stackapps.com": "stackapps",
}
_STACKEXCHANGE_API = "https://api.stackexchange.com/2.3"
_QUESTION_PATH_RE = re.compile(r"^/(?:questions|q)/(\d{1,12})(?:/|$)")
_SE_ANSWERS = 3
_SE_SITE_RE = re.compile(r"^[a-z0-9.-]{1,60}$")


@dataclass(frozen=True)
class SiteRead:
    """Text read from a site's content endpoint. *url* is the address the
    text is reported under."""

    url: str
    text: str


Reader = Callable[[Fetcher, float], Optional[SiteRead]]


def _strip_www(host: str) -> str:
    return host[4:] if host.startswith("www.") else host


def _github_parts(parsed) -> Optional[list[str]]:
    host = (parsed.hostname or "").lower()
    if host not in _GITHUB_HOSTS or parsed.scheme not in ("http", "https"):
        return None
    if parsed.port not in (None, 443, 80):
        return None
    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) < 2 or parts[0].lower() in _GITHUB_RESERVED:
        return None
    owner, repo = parts[0], parts[1]
    if repo.endswith(".git"):
        repo = repo[:-4]
    if not (_GITHUB_NAME_RE.match(owner) and _GITHUB_NAME_RE.match(repo)):
        return None
    if repo in (".", ".."):
        return None
    return [owner, repo, *parts[2:]]


def _quote_path(segments: list[str]) -> str:
    return "/".join(urllib.parse.quote(urllib.parse.unquote(s), safe="")
                    for s in segments)


def _decode_github_readme(body: str) -> Optional[str]:
    """The README text inside a GitHub REST ``readme`` JSON body (base64
    ``content``), or None when the body is not that shape."""
    try:
        payload = json.loads(body)
    except ValueError:
        return None
    if not isinstance(payload, dict) or payload.get("encoding") != "base64":
        return None
    content = payload.get("content")
    if not isinstance(content, str):
        return None
    try:
        raw = base64.b64decode(content, validate=False)
    except (binascii.Error, ValueError):
        return None
    return raw.decode("utf-8", errors="replace")


def _github_reader(url: str) -> Optional[Reader]:
    parsed = urllib.parse.urlparse(url)
    parts = _github_parts(parsed)
    if parts is None:
        return None
    owner, repo, rest = parts[0], parts[1], parts[2:]
    quoted = _quote_path([owner, repo])

    if not rest or (rest[0] == "tree" and len(rest) >= 2):
        if rest:
            ref = urllib.parse.quote(urllib.parse.unquote(rest[1]), safe="")
            sub = _quote_path(rest[2:]) if rest[2:] else ""
            raw_url = (f"https://raw.githubusercontent.com/{quoted}/{ref}/"
                       + (f"{sub}/" if sub else "") + "README.md")
            api_url = (f"https://api.github.com/repos/{quoted}/readme"
                       + (f"/{sub}" if sub else "") + f"?ref={ref}")
        else:
            raw_url = f"https://raw.githubusercontent.com/{quoted}/HEAD/README.md"
            api_url = f"https://api.github.com/repos/{quoted}/readme"

        def read_readme(fetch: Fetcher, timeout: float) -> Optional[SiteRead]:
            text = _try_text(fetch, raw_url, timeout)
            if text:
                return SiteRead(raw_url, text)
            body = _try_text(fetch, api_url, timeout)
            readme = _decode_github_readme(body) if body else None
            if readme and readme.strip():
                return SiteRead(api_url, readme)
            return None

        return read_readme

    if rest[0] == "blob" and len(rest) >= 3:
        raw_url = (f"https://raw.githubusercontent.com/{quoted}/"
                   f"{_quote_path(rest[1:])}")

        def read_blob(fetch: Fetcher, timeout: float) -> Optional[SiteRead]:
            text = _try_text(fetch, raw_url, timeout)
            return SiteRead(raw_url, text) if text else None

        return read_blob
    return None


def _stackexchange_site(host: str) -> Optional[str]:
    host = _strip_www(host.lower())
    if host in _STACKEXCHANGE_SITES:
        return _STACKEXCHANGE_SITES[host]
    if host.endswith(".stackexchange.com"):
        name = host[: -len(".stackexchange.com")]
        if name and name not in ("api", "chat") and \
                _SE_SITE_RE.match(name):
            return name
    if host.endswith(".stackoverflow.com"):
        name = host[: -len(".stackoverflow.com")]
        if name and "." not in name and _SE_SITE_RE.match(name):
            return f"{name}.stackoverflow"
    return None


def _html_fragment_text(fragment: str) -> str:
    from localm.netpolicy import html_to_text
    return html_to_text(fragment or "")


def _render_stackexchange(question: dict, answers: list) -> Optional[str]:
    title = html.unescape(str(question.get("title") or "")).strip()
    body = _html_fragment_text(str(question.get("body") or ""))
    if not (title or body):
        return None
    lines = []
    if title:
        lines.append(f"Question: {title}")
    if body:
        lines.append(body)
    for answer in answers[:_SE_ANSWERS]:
        if not isinstance(answer, dict):
            continue
        text = _html_fragment_text(str(answer.get("body") or ""))
        if not text:
            continue
        label = "Accepted answer" if answer.get("is_accepted") else "Answer"
        score = answer.get("score")
        if isinstance(score, int):
            label += f" (score {score})"
        lines.append(f"{label}:\n{text}")
    return "\n\n".join(lines)


def _items(body: Optional[str]) -> list:
    if not body:
        return []
    try:
        payload = json.loads(body)
    except ValueError:
        return []
    items = payload.get("items") if isinstance(payload, dict) else None
    return items if isinstance(items, list) else []


def _stackexchange_reader(url: str) -> Optional[Reader]:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https") or \
            parsed.port not in (None, 443, 80):
        return None
    site = _stackexchange_site(parsed.hostname or "")
    if site is None:
        return None
    m = _QUESTION_PATH_RE.match(parsed.path)
    if not m:
        return None
    qid = m.group(1)
    common = urllib.parse.urlencode({"site": site, "filter": "withbody"})
    question_url = f"{_STACKEXCHANGE_API}/questions/{qid}?{common}"
    answers_url = (f"{_STACKEXCHANGE_API}/questions/{qid}/answers?{common}&"
                   + urllib.parse.urlencode({"sort": "votes", "order": "desc",
                                             "pagesize": _SE_ANSWERS}))

    def read_question(fetch: Fetcher, timeout: float) -> Optional[SiteRead]:
        questions = _items(_try_text(fetch, question_url, timeout))
        if not questions or not isinstance(questions[0], dict):
            return None
        answers = _items(_try_text(fetch, answers_url, timeout))
        text = _render_stackexchange(questions[0], answers)
        return SiteRead(url, text) if text else None

    return read_question


def _try_text(fetch: Fetcher, url: str, timeout: float) -> Optional[str]:
    """*url*'s body through *fetch*, or None when the read fails or is
    blank. A failure is logged at debug level by exception type only."""
    try:
        _final, _ctype, body = fetch(url, timeout=timeout)
    except Exception as exc:
        logger.debug("web retrieval: content endpoint read failed (%s)",
                     type(exc).__name__)
        return None
    return body if body and body.strip() else None


def site_reader(url: str) -> Optional[Reader]:
    """The content-endpoint reader for *url*, or None when *url* is not a
    recognised GitHub repository/tree/blob or Stack Exchange question URL."""
    for make in (_github_reader, _stackexchange_reader):
        try:
            reader = make(url)
        except ValueError:
            reader = None
        if reader is not None:
            return reader
    return None


def read_url(url: str, fetch: Fetcher, *, timeout: float
             ) -> tuple[str, str, str]:
    """Read *url* through *fetch* as ``(final_url, content_type, text)``.

    A recognised URL (see the module docstring) is first checked against the
    network policy with ``netpolicy.check_url`` (a refusal raises
    ``NetworkPolicyError`` before any request), then answered from its site's
    content endpoint when that yields text, reported as plain text under the
    endpoint's URL (a Stack Exchange question keeps its own URL). Otherwise,
    or when every endpoint fails, *url* itself is fetched and its failure, if
    any, is raised."""
    reader = site_reader(url)
    if reader is not None:
        from localm import netpolicy
        netpolicy.check_url(url)
        got = reader(fetch, timeout)
        if got is not None:
            return got.url, CONTENT_TYPE_TEXT, got.text
    return fetch(url, timeout=timeout)
