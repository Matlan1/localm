# SPDX-License-Identifier: AGPL-3.0-or-later
"""Web plugin: search and page fetch for the chat surface.

Routes (mounted by the engine, auto-scoped to the ``web`` capability):
  POST /api/web/search  - run a web search, return ranked results
  POST /api/web/fetch   - fetch a URL and return readable text

Every request is enforced by ``localm.netpolicy`` (net_mode, net_allow/
net_deny, and the private-address SSRF guard). "off" blocks; "allow" permits;
"ask" means each MODEL-INITIATED request must be approved by the user first.
That per-request approval is interactive, so it lives in the chat front end:
under net_mode=ask the GUI prompts before it calls these endpoints (WEB-ask).
A request that reaches here is therefore treated as already-consented (an
explicit ``/search-web`` command, the per-conversation toggle, or a
GUI-approved model request); these endpoints do not re-prompt. Domain rules and
the private-address guard always apply.

Search results and fetched page text are UNTRUSTED content: the caller approved
the REQUEST, never the bytes a remote page returns, and both callers here (the
GUI chat and the scheduled-job web tool, ``jobs/webtool.py``) splice this text
straight into the model's message list. Both backends tokenise with
special-token parsing on, so a literal chat-template control token in a
page/snippet is parsed as a REAL role delimiter and can forge a turn.
``neutralise()`` defangs that here, at the boundary, so every consumer gets
defanged content. ``jobs/webtool.py`` calls ``localm.netpolicy`` directly rather
than these HTTP endpoints, so it neutralises its own copy at that boundary.
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from localm.inference.errors import route_errors
from localm.netpolicy import NetworkPolicyError
from localm.executor import get_plugin_executor
from localm.textguard import neutralise

_router = APIRouter()


def _neutralise_results(results: list) -> list:
    """Defang chat control / frame tokens in each search result's title/snippet
    before it leaves this boundary. A search result is UNTRUSTED content: a page
    author can embed a control token (``<|im_start|>system ...``) or a frame
    marker in the title/snippet, which both backends' tokenizers parse as a real
    role delimiter once spliced into the prompt. ``url`` is a locator, not prose,
    and is left untouched."""
    for r in results:
        if isinstance(r, dict):
            if isinstance(r.get("title"), str):
                r["title"] = neutralise(r["title"])
            if isinstance(r.get("snippet"), str):
                r["snippet"] = neutralise(r["snippet"])
    return results


class WebSearchRequest(BaseModel):
    query: str
    max_results: int = 5


class WebFetchRequest(BaseModel):
    url: str
    max_chars: int = 8000


@_router.post("/api/web/search")
@route_errors({
    NetworkPolicyError: 403,
    Exception: lambda e: (502, f"Search failed: {e}"),
})
async def web_search_endpoint(req: WebSearchRequest):
    from localm.netpolicy import web_search
    if not req.query.strip():
        raise HTTPException(400, "Empty query")
    loop = asyncio.get_running_loop()
    # Defanging runs INSIDE the executor with the search itself: it is unbounded
    # CPU over remote-controlled text and must not run on the event loop.
    results = await loop.run_in_executor(
        get_plugin_executor(),
        lambda: _neutralise_results(
            web_search(req.query, max_results=req.max_results)))
    return {"query": req.query, "results": results}


@_router.post("/api/web/fetch")
@route_errors({
    NetworkPolicyError: 403,
    Exception: lambda e: (502, f"Fetch failed: {e}"),
})
async def web_fetch_endpoint(req: WebFetchRequest):
    from localm.netpolicy import fetch_text
    max_chars = max(500, min(req.max_chars, 60_000))

    def _fetch_and_defang():
        # neutralise() runs in the SAME executor call as the fetch: both the URL
        # and the bytes are attacker-controlled, and defanging is unbounded CPU
        # over that text.
        final_url, text = fetch_text(req.url)
        return final_url, neutralise(text[:max_chars]), len(text) > max_chars

    loop = asyncio.get_running_loop()
    final_url, text, truncated = await loop.run_in_executor(
        get_plugin_executor(), _fetch_and_defang)
    return {"url": final_url, "text": text, "truncated": truncated}


def register(host) -> None:
    host.mount_router(_router)


def unregister() -> None:
    pass
