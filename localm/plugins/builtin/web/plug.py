# SPDX-License-Identifier: AGPL-3.0-or-later
"""Web plugin: search and page fetch for the chat surface.

Routes (mounted by the engine, auto-scoped to the ``web`` capability):
  POST /api/web/retrieve - search, read the top result pages and return an
                           evidence bundle (``localm.web_retrieval``)
  POST /api/web/search   - run a web search, return ranked results
  POST /api/web/fetch    - fetch a URL and return readable text

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
defanged content. ``jobs/webtool.py`` and the coder's ``tools/web.py`` call
``localm.web_retrieval`` and ``localm.netpolicy`` directly rather than these
HTTP endpoints, so each neutralises its own copy at that boundary.

Every response also carries ``untrusted_fields``, naming the fields whose value is
wholly remote-controlled. A caller that splices one of them into a prompt uses
that to set ``Message.untrusted_spans`` over the range it landed on, so the
backend tokenises it with special-token parsing off. Defanging is unconditional
and does not depend on a caller reading this field.
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


# Response fields whose value is wholly remote-controlled text. A caller that
# splices one into a prompt must mark the range it lands on as untrusted (see
# Message.untrusted_spans). Named per response shape rather than as character
# ranges because every character of these fields is untrusted, so a caller only
# needs the offset it spliced them at.
_UNTRUSTED_SEARCH_FIELDS = ["title", "snippet"]
_UNTRUSTED_FETCH_FIELDS = ["text"]
_UNTRUSTED_RETRIEVE_FIELDS = ["prompt_text", "search_error"]
_UNTRUSTED_SOURCE_FIELDS = ["title", "snippet", "error"]
_UNTRUSTED_CHUNK_FIELDS = ["text"]


def _neutralise_fields(record: dict, fields: list) -> dict:
    """Defang chat control / frame tokens in each of *fields* that *record*
    carries as a string, and set ``record["untrusted_fields"]`` to the fields
    it carries."""
    for f in fields:
        if isinstance(record.get(f), str):
            record[f] = neutralise(record[f])
    record["untrusted_fields"] = [f for f in fields
                                  if isinstance(record.get(f), str)]
    return record


def _neutralise_results(results: list) -> list:
    """Defang chat control / frame tokens in each search result's title/snippet
    before it leaves this boundary, and declare those fields untrusted.

    A search result is UNTRUSTED content: a page author can embed a control token
    (``<|im_start|>system ...``) or a frame marker in the title/snippet, which
    both backends' tokenizers parse as a real role delimiter once spliced into
    the prompt. ``url`` is a locator, not prose, and is left untouched.

    Each result gains ``untrusted_fields``, naming the fields a caller must mark
    as an untrusted range when it splices them into a prompt.
    """
    for r in results:
        if isinstance(r, dict):
            _neutralise_fields(r, _UNTRUSTED_SEARCH_FIELDS)
    return results


def _neutralise_bundle(bundle) -> dict:
    """Serialise an ``EvidenceBundle`` with every prose field defanged: each
    source's title/snippet/error, each chunk's text, ``search_error`` and the
    ``prompt_text`` rendering, all declared in ``untrusted_fields`` at their
    level. URL fields are locators, not prose, and are left untouched.
    ``grounding_summary`` is built from states and counts only and stays
    trusted."""
    data = bundle.to_dict()
    data["prompt_text"] = bundle.to_prompt_text()
    for src in data.get("sources", []):
        _neutralise_fields(src, _UNTRUSTED_SOURCE_FIELDS)
    for chunk in data.get("chunks", []):
        _neutralise_fields(chunk, _UNTRUSTED_CHUNK_FIELDS)
    _neutralise_fields(data, _UNTRUSTED_RETRIEVE_FIELDS)
    return data


class WebSearchRequest(BaseModel):
    query: str
    max_results: int = 5


class WebFetchRequest(BaseModel):
    url: str
    max_chars: int = 8000


class WebRetrieveRequest(BaseModel):
    query: str


@_router.post("/api/web/retrieve")
@route_errors({
    NetworkPolicyError: 403,
    Exception: lambda e: (502, f"Retrieval failed: {e}"),
})
async def web_retrieve_endpoint(req: WebRetrieveRequest):
    """Search *query*, read the top result pages and return the evidence
    bundle (``EvidenceBundle.to_dict()`` plus ``prompt_text``), every prose
    field defanged and declared in ``untrusted_fields``. A provider failure
    is reported inside the bundle (``search_status``, ``grounding``
    ``failed``); a policy refusal is 403."""
    from localm.debuglog import logger
    from localm.web_retrieval import retrieve
    if not req.query.strip():
        raise HTTPException(400, "Empty query")
    logger.info("web retrieve: query=%r", req.query)
    loop = asyncio.get_running_loop()
    # The retrieval and the defanging of its text both run in the executor.
    bundle_dict = await loop.run_in_executor(
        get_plugin_executor(),
        lambda: _neutralise_bundle(retrieve(req.query)))
    logger.info("web retrieve: status=%s, sources=%d, chunks=%d",
                bundle_dict.get("search_status"),
                len(bundle_dict.get("sources", [])),
                len(bundle_dict.get("chunks", [])))
    return bundle_dict


@_router.post("/api/web/search")
@route_errors({
    NetworkPolicyError: 403,
    Exception: lambda e: (502, f"Search failed: {e}"),
})
async def web_search_endpoint(req: WebSearchRequest):
    from localm.debuglog import logger
    from localm.netpolicy import web_search
    if not req.query.strip():
        raise HTTPException(400, "Empty query")
    logger.info("web search: query=%r (max_results=%d)", req.query, req.max_results)
    loop = asyncio.get_running_loop()
    # Defanging runs INSIDE the executor with the search itself: it is unbounded
    # CPU over remote-controlled text and must not run on the event loop.
    results = await loop.run_in_executor(
        get_plugin_executor(),
        lambda: _neutralise_results(
            web_search(req.query, max_results=req.max_results)))
    logger.info("web search: returned %d result(s)", len(results))
    return {"query": req.query, "results": results}


@_router.post("/api/web/fetch")
@route_errors({
    NetworkPolicyError: 403,
    Exception: lambda e: (502, f"Fetch failed: {e}"),
})
async def web_fetch_endpoint(req: WebFetchRequest):
    from localm.debuglog import logger
    from localm.netpolicy import fetch_text
    max_chars = max(500, min(req.max_chars, 60_000))
    logger.info("web fetch: url=%r (max_chars=%d)", req.url, max_chars)

    def _fetch_and_defang():
        # neutralise() runs in the SAME executor call as the fetch: both the URL
        # and the bytes are attacker-controlled, and defanging is unbounded CPU
        # over that text.
        final_url, text = fetch_text(req.url)
        return final_url, neutralise(text[:max_chars]), len(text) > max_chars

    loop = asyncio.get_running_loop()
    final_url, text, truncated = await loop.run_in_executor(
        get_plugin_executor(), _fetch_and_defang)
    logger.info("web fetch: retrieved %d chars (truncated=%s)", len(text), truncated)
    return {"url": final_url, "text": text, "truncated": truncated,
            "untrusted_fields": list(_UNTRUSTED_FETCH_FIELDS)}


def register(host) -> None:
    host.mount_router(_router)


def unregister() -> None:
    pass
