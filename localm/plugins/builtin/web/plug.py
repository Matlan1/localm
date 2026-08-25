# SPDX-License-Identifier: AGPL-3.0-or-later
"""Web plugin: search and page fetch for the chat surface."""

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
    """Defang chat control / frame tokens in each search result's title/snippet before it leaves this boundary (LM-DA-014, indirect prompt injection - the same bug class LM-DA-SEC-03 fixed for RAG's retrieved chunks, mirrored here via ``rag/plug.py._neutralise_hits``)."""
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
    # Defanging runs INSIDE the executor with the search itself. It is pure CPU
    # over remote-controlled text, so on the event loop it is a whole-server
    # stall, not a slow request - see the fetch handler below.
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
        # neutralise() belongs in the SAME executor call as the fetch. Both the
        # URL and the bytes are attacker-controlled (under net_mode=allow the
        # model picks the URL, so a poisoned page can chain into a fetch of the
        # attacker's own), and defanging is unbounded CPU over that text - on the
        # event loop it stalls every other request, not just this one.
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
