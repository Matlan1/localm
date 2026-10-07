# SPDX-License-Identifier: AGPL-3.0-or-later
"""Plain-language reasons for failed web requests.

``describe_failure`` turns a transport or HTTP exception into a short clause
that names the host and what went wrong, for the GUI card, the toast, the
coder and jobs tool results and the text a model reads. ``failure_kind``
classifies an exception; ``is_transient`` says whether repeating the identical
request can help.
"""

from __future__ import annotations

import errno
import http.client
import urllib.parse
from typing import Optional

import requests
import urllib3.exceptions

from localm.netpin import ReadBudgetExceeded

_TEXT_CAP = 300

_RESET_ERRNOS = frozenset({
    errno.ECONNRESET, errno.ECONNABORTED, errno.EPIPE, 10053, 10054,
})
_REFUSED_ERRNOS = frozenset({errno.ECONNREFUSED, 10061})
_UNREACHABLE_ERRNOS = frozenset({
    errno.ENETUNREACH, errno.EHOSTUNREACH, errno.ENETDOWN, errno.EADDRNOTAVAIL,
    10049, 10050, 10051, 10065,
})


def _host_of(url: str) -> str:
    try:
        host = urllib.parse.urlparse(url).hostname or ""
    except ValueError:
        host = ""
    return host


def _exc_url(exc: BaseException) -> str:
    url = getattr(exc, "url", "") or ""
    if url:
        return url
    request = getattr(exc, "request", None)
    if request is not None and getattr(request, "url", None):
        return request.url
    response = getattr(exc, "response", None)
    if response is not None and getattr(response, "url", None):
        return response.url
    return ""


def _chain(exc: BaseException):
    """*exc*, then its ``__cause__``/``__context__`` chain and the reasons
    urllib3 and requests nest in ``args`` and ``reason``, at most 12 deep."""
    seen: set[int] = set()
    todo: list[BaseException] = [exc]
    while todo and len(seen) < 12:
        cur = todo.pop(0)
        if id(cur) in seen:
            continue
        seen.add(id(cur))
        yield cur
        for nxt in (getattr(cur, "reason", None), cur.__cause__,
                    cur.__context__, *getattr(cur, "args", ())):
            if isinstance(nxt, BaseException):
                todo.append(nxt)


def _http_status(exc: BaseException) -> Optional[int]:
    response = getattr(exc, "response", None)
    code = getattr(response, "status_code", None)
    return code if isinstance(code, int) else None


def failure_kind(exc: BaseException) -> str:
    """The kind of failure *exc* is: ``tls``, ``connect-timeout``,
    ``read-timeout``, ``budget`` (``ReadBudgetExceeded``), ``reset``,
    ``refused``, ``unreachable``, ``incomplete``, ``http`` (an ``HTTPError``
    carrying a status) or ``other``."""
    if isinstance(exc, ReadBudgetExceeded):
        return "budget"
    if isinstance(exc, requests.exceptions.SSLError):
        return "tls"
    if isinstance(exc, requests.exceptions.ConnectTimeout):
        return "connect-timeout"
    if isinstance(exc, requests.exceptions.HTTPError) and \
            _http_status(exc) is not None:
        return "http"
    if isinstance(exc, (requests.exceptions.ChunkedEncodingError,
                        requests.exceptions.ContentDecodingError)):
        return "incomplete"
    for cur in _chain(exc):
        if isinstance(cur, ReadBudgetExceeded):
            return "budget"
        if isinstance(cur, urllib3.exceptions.ConnectTimeoutError):
            return "connect-timeout"
        if isinstance(cur, urllib3.exceptions.ReadTimeoutError):
            return "read-timeout"
        if isinstance(cur, (ConnectionResetError, ConnectionAbortedError,
                            BrokenPipeError, http.client.RemoteDisconnected)):
            return "reset"
        if isinstance(cur, ConnectionRefusedError):
            return "refused"
        if isinstance(cur, OSError) and cur.errno is not None:
            if cur.errno in _RESET_ERRNOS:
                return "reset"
            if cur.errno in _REFUSED_ERRNOS:
                return "refused"
            if cur.errno in _UNREACHABLE_ERRNOS:
                return "unreachable"
    if isinstance(exc, (requests.exceptions.ReadTimeout, TimeoutError)):
        return "read-timeout"
    if isinstance(exc, requests.exceptions.Timeout):
        return "connect-timeout"
    if isinstance(exc, (requests.exceptions.ConnectionError, ConnectionError)):
        text = str(exc).lower()
        if "connection aborted" in text or "remote end closed" in text:
            return "reset"
        return "unreachable"
    return "other"


def is_transient(exc: BaseException) -> bool:
    """True when the identical request may succeed if sent again: the
    connection was reset or closed early, could not be opened, timed out, or
    the body arrived incomplete. False for a policy refusal, a TLS
    verification failure, any HTTP status, a read that exceeded its total
    time allowance, and anything unrecognised."""
    from localm.netpolicy import NetworkPolicyError
    if isinstance(exc, NetworkPolicyError):
        return False
    return failure_kind(exc) in ("reset", "refused", "unreachable",
                              "connect-timeout", "read-timeout", "incomplete")


def describe_failure(exc: BaseException, url: str = "") -> str:
    """A plain-language clause (no trailing period) saying why a request to
    *url* failed, naming its host. *url* defaults to the URL the exception
    carries. A ``NetworkPolicyError`` or ``SearchProviderError`` keeps its own
    message; an unrecognised exception yields its message, or its type name
    when the message is empty. At most 300 characters."""
    from localm.netpolicy import NetworkPolicyError

    from .contracts import SearchProviderError

    if isinstance(exc, (NetworkPolicyError, SearchProviderError)):
        return (str(exc).strip() or type(exc).__name__)[:_TEXT_CAP]
    host = _host_of(url or _exc_url(exc)) or "the site"
    kind = failure_kind(exc)
    if kind == "tls":
        text = (f"{host} failed the secure-connection check "
                "(TLS certificate or handshake error)")
    elif kind == "connect-timeout":
        text = f"could not connect to {host} in time"
    elif kind == "read-timeout":
        text = f"{host} did not respond in time"
    elif kind == "budget":
        seconds = next((c.seconds for c in _chain(exc)
                        if isinstance(c, ReadBudgetExceeded)), None)
        text = (f"{host} was too slow to send the page"
                + (f" (over {seconds:g}s)" if seconds else ""))
    elif kind == "reset":
        text = f"{host} closed the connection before answering"
    elif kind == "refused":
        text = f"{host} refused the connection"
    elif kind == "unreachable":
        text = f"could not connect to {host}"
    elif kind == "incomplete":
        text = f"{host} sent an incomplete or corrupted response"
    elif kind == "http":
        code = _http_status(exc)
        if code in (401, 403):
            text = (f"{host} refused access, HTTP {code}; the site may block "
                    "automated readers")
        elif code in (404, 410):
            text = f"page not found on {host}, HTTP {code}"
        elif code == 429:
            text = f"{host} is rate-limiting requests, HTTP 429"
        elif code is not None and code >= 500:
            text = f"{host} had a server error, HTTP {code}"
        else:
            text = f"{host} answered HTTP {code}"
    else:
        text = " ".join(str(exc).split()) or type(exc).__name__
    return text[:_TEXT_CAP]
