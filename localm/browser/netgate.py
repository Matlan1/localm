# SPDX-License-Identifier: AGPL-3.0-or-later
"""Per-request network gate for the automated browser.

A browser issues far more requests than the caller asks for: the top-level
navigation, then every subresource the loaded page wants (images, scripts,
stylesheets, fonts, XHR/fetch, websockets), plus redirects. Each one is a
separate destination, so each one is decided here.

``decide()`` is pure policy and takes no browser objects, so it is testable
without a browser. ``localm.netpolicy.check_url`` makes the network decision;
this module only adds the scheme triage a browser needs and an OPTIONAL
browser-specific narrowing that can refuse more but never permit more.

``check_url`` resolves DNS, so ``decide()`` blocks. Call it off the event loop
that owns the browser session (see ``decide_async``).

KNOWN LIMIT, and it is not closable at this seam: netpolicy's own HTTP client
pins the resolved IP onto the socket (``pinned_request``), so a check-then-
connect DNS rebind cannot retarget it. Chromium resolves and connects on its
own, so a request allowed here is re-resolved by the browser before it
connects. The window is narrow and the check is still worth making, but this
path is NOT rebinding-proof the way the Python client is.
"""

from __future__ import annotations

import asyncio
from typing import Iterable, Optional

from localm import netpolicy

#: Schemes a page uses that never reach the network or the filesystem. Allowed
#: without consulting netpolicy, whose check_url only understands http/https.
INERT_SCHEMES = frozenset({"about", "data", "blob", "javascript"})

#: Schemes decided by netpolicy.
NETWORK_SCHEMES = frozenset({"http", "https"})


def _scheme_of(url: str) -> str:
    head, sep, _ = url.partition(":")
    return head.lower() if sep else ""


def decide(url: str, *, extra_deny: Iterable[str] = (),
           extra_allow: Iterable[str] = ()) -> Optional[str]:
    """Return None to allow *url*, or a string reason to refuse it.

    Order: scheme triage, then the global network policy, then the optional
    browser-specific narrowing. The narrowing runs LAST and only ever refuses,
    so no browser setting can reach a destination ``netpolicy`` already denied.

    *extra_deny* refuses any host it matches. *extra_allow*, when non-empty,
    refuses every host it does NOT match. Both use netpolicy's own host
    matching, so they behave exactly like net_deny / net_allow.
    """
    scheme = _scheme_of(url)
    if scheme in INERT_SCHEMES:
        return None
    if scheme not in NETWORK_SCHEMES:
        return (f"scheme '{scheme or '(none)'}' is not allowed in the "
                "automated browser")

    try:
        netpolicy.check_url(url)
    except netpolicy.NetworkPolicyError as exc:
        return str(exc)
    except Exception as exc:                       # noqa: BLE001
        return f"network policy could not be evaluated: {exc}"

    host = netpolicy.check_url_shape(url)
    for pattern in netpolicy._domain_list(list(extra_deny)):
        if netpolicy._host_matches(host, pattern):
            return f"'{host}' is on the browser deny list"
    allow = netpolicy._domain_list(list(extra_allow))
    if allow and not any(netpolicy._host_matches(host, p) for p in allow):
        return f"'{host}' is not on the browser allow list"
    return None


async def decide_async(url: str, *, extra_deny: Iterable[str] = (),
                       extra_allow: Iterable[str] = ()) -> Optional[str]:
    """``decide`` on a worker thread, so its DNS lookup never stalls the event
    loop driving the browser."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None, lambda: decide(url, extra_deny=tuple(extra_deny),
                             extra_allow=tuple(extra_allow)))
