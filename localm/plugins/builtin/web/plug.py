"""Web plugin: search and page fetch for the chat surface.

Routes (mounted by the engine, auto-scoped to the ``web`` capability):
  POST /api/web/search  - run a web search, return ranked results
  POST /api/web/fetch   - fetch a URL and return readable text

Every request is enforced by ``localm.netpolicy`` (net_mode, net_allow/
net_deny, and the private-address SSRF guard). Callers are the user's explicit
``/search-web`` command or the per-conversation web-access toggle, both direct
consent, so "ask" mode does not re-prompt here; only "off" blocks. Domain rules
and the private-address guard always apply.
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

_router = APIRouter()


class WebSearchRequest(BaseModel):
    query: str
    max_results: int = 5


class WebFetchRequest(BaseModel):
    url: str
    max_chars: int = 8000


@_router.post("/api/web/search")
async def web_search_endpoint(req: WebSearchRequest):
    from localm.netpolicy import NetworkPolicyError, web_search
    if not req.query.strip():
        raise HTTPException(400, "Empty query")
    loop = asyncio.get_running_loop()
    try:
        results = await loop.run_in_executor(
            None, lambda: web_search(req.query, max_results=req.max_results))
    except NetworkPolicyError as e:
        raise HTTPException(403, str(e))
    except Exception as e:
        raise HTTPException(502, f"Search failed: {e}")
    return {"query": req.query, "results": results}


@_router.post("/api/web/fetch")
async def web_fetch_endpoint(req: WebFetchRequest):
    from localm.netpolicy import NetworkPolicyError, fetch_text
    loop = asyncio.get_running_loop()
    try:
        final_url, text = await loop.run_in_executor(
            None, lambda: fetch_text(req.url))
    except NetworkPolicyError as e:
        raise HTTPException(403, str(e))
    except Exception as e:
        raise HTTPException(502, f"Fetch failed: {e}")
    max_chars = max(500, min(req.max_chars, 60_000))
    return {"url": final_url,
            "text": text[:max_chars],
            "truncated": len(text) > max_chars}


def register(host) -> None:
    host.mount_router(_router)


def unregister() -> None:
    pass
