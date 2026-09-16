# SPDX-License-Identifier: AGPL-3.0-or-later
"""The retrieval controller: search, deduplicate, read the top pages
concurrently through ``localm.netpolicy``, extract, select evidence.

``retrieve`` returns an ``EvidenceBundle`` for every outcome except a policy
refusal or an empty query. A failing provider is reported in
``bundle.search_status`` / ``bundle.search_error`` and is never replaced by
another provider. A failing page read is reported on its ``Source`` and never
fails the retrieval.
"""

from __future__ import annotations

import concurrent.futures
from typing import Callable, Optional

from localm import netpolicy

from .canonical import canonicalize_url, dedup_key, dedup_results
from .chunking import select_evidence
from .contracts import (
    CHUNK_PAGE,
    CHUNK_SNIPPET,
    EVIDENCE_BUDGET_CHARS,
    FETCH_TOP,
    GROUNDING_FAILED,
    GROUNDING_PAGE_BACKED,
    GROUNDING_SNIPPET_ONLY,
    PER_SOURCE_CAP_CHARS,
    SEARCH_CANDIDATES,
    SEARCH_EMPTY,
    SEARCH_FAILED,
    SEARCH_OK,
    STATUS_DUPLICATE,
    STATUS_FETCHED,
    STATUS_SKIPPED,
    EvidenceBundle,
    PageDocument,
    SearchProvider,
    Source,
)
from .extract import extract_page
from .providers import provider_from_config

#: ``fetch(url, timeout=...) -> (final_url, content_type, text)``.
Fetcher = Callable[..., tuple[str, str, str]]

_MAX_SEARCH_CANDIDATES = 10
_ERROR_TEXT_CAP = 300
_HTML_SNIFF_BYTES = 1024


def _describe(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}".strip()
    return text[:_ERROR_TEXT_CAP]


def _default_fetch(url: str, *, timeout: int) -> tuple[str, str, str]:
    return netpolicy.safe_fetch(url, timeout=timeout)


def _looks_like_html(content_type: str, body: str) -> bool:
    if "html" in (content_type or "").lower():
        return True
    head = (body or "")[:_HTML_SNIFF_BYTES].lower()
    return "<html" in head or "<!doctype html" in head


def _read_page(source: Source, fetch: Fetcher, timeout: int) -> PageDocument:
    final_url, content_type, body = fetch(source.url, timeout=timeout)
    if _looks_like_html(content_type, body):
        page = extract_page(body)
        return PageDocument(url=source.url, final_url=final_url,
                            title=page.title, text=page.text,
                            content_type=content_type, region=page.region)
    return PageDocument(url=source.url, final_url=final_url, title="",
                        text=(body or "").strip(), content_type=content_type,
                        region="text")


def _record_page(source: Source, page: PageDocument,
                 pages: dict[str, PageDocument]) -> None:
    source.retrieval_status = STATUS_FETCHED
    source.final_url = page.final_url
    source.region = page.region
    source.text_chars = len(page.text)
    if not source.title.strip() and page.title:
        source.title = page.title
    if page.text.strip():
        source.grounding = GROUNDING_PAGE_BACKED
        pages[source.id] = page
    else:
        source.grounding = (GROUNDING_SNIPPET_ONLY if source.snippet.strip()
                            else GROUNDING_FAILED)
        source.error = "page had no extractable text"


def _read_pages(to_fetch: list[Source], fetch: Fetcher, timeout: int,
                deadline: float) -> dict[str, PageDocument]:
    """Read every source in *to_fetch* concurrently. A read still running at
    *deadline* seconds is recorded as failed; its worker finishes on its own
    transport timeout."""
    pages: dict[str, PageDocument] = {}
    pool = concurrent.futures.ThreadPoolExecutor(
        max_workers=len(to_fetch), thread_name_prefix="web-retrieval")
    try:
        futures = {pool.submit(_read_page, s, fetch, timeout): s
                   for s in to_fetch}
        done, pending = concurrent.futures.wait(futures, timeout=deadline)
        for fut in pending:
            futures[fut].mark_failed(f"timed out after {deadline:g}s")
            fut.cancel()
        for fut in done:
            source = futures[fut]
            try:
                page = fut.result()
            except netpolicy.NetworkPolicyError as exc:
                source.mark_failed(f"refused by policy: {exc}"[:_ERROR_TEXT_CAP])
            except Exception as exc:
                source.mark_failed(_describe(exc))
            else:
                _record_page(source, page, pages)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    return pages


def _drop_duplicate_pages(fetched: list[Source],
                          pages: dict[str, PageDocument]) -> None:
    """A fetched source whose final URL has the same ``dedup_key`` as a
    higher-ranked fetched source loses its page and becomes ``duplicate``."""
    seen: dict[str, str] = {}
    for source in fetched:
        if source.id not in pages:
            continue
        key = dedup_key(source.final_url or source.url)
        if key in seen:
            del pages[source.id]
            source.retrieval_status = STATUS_DUPLICATE
            source.grounding = (GROUNDING_SNIPPET_ONLY if source.snippet.strip()
                                else GROUNDING_FAILED)
            source.error = f"same page as {seen[key]}"
        else:
            seen[key] = source.id


def retrieve(
    query: str,
    *,
    provider: Optional[SearchProvider] = None,
    search_candidates: int = SEARCH_CANDIDATES,
    fetch_top: int = FETCH_TOP,
    budget_chars: int = EVIDENCE_BUDGET_CHARS,
    per_source_cap_chars: int = PER_SOURCE_CAP_CHARS,
    fetch: Optional[Fetcher] = None,
    fetch_timeout: Optional[int] = None,
    deadline_seconds: Optional[float] = None,
) -> EvidenceBundle:
    """Search *query*, read the top pages and return an ``EvidenceBundle``.

    *provider* defaults to ``provider_from_config()``. *search_candidates* is
    clamped to 1..10 and *fetch_top* to 0..*search_candidates*. The provider
    is asked for twice *search_candidates* results (at most 10); after
    duplicate removal the first *search_candidates* become sources. *fetch*
    defaults to ``netpolicy.safe_fetch``; *fetch_timeout* to netpolicy's
    default; *deadline_seconds* (the wait for all page reads together) to
    twice *fetch_timeout*.

    Raises ``ValueError`` for an empty query and ``NetworkPolicyError`` when
    the policy refuses the search request. Every other search failure is
    recorded in the bundle (``search_status`` ``failed``); every page-read
    failure is recorded on its source.
    """
    query = (query or "").strip()
    if not query:
        raise ValueError("Empty search query")
    search_candidates = max(1, min(int(search_candidates),
                                   _MAX_SEARCH_CANDIDATES))
    fetch_top = max(0, min(int(fetch_top), search_candidates))
    if provider is None:
        provider = provider_from_config()
    if fetch is None:
        fetch = _default_fetch
    timeout = int(fetch_timeout if fetch_timeout is not None
                  else netpolicy._DEFAULT_TIMEOUT)
    deadline = float(deadline_seconds if deadline_seconds is not None
                     else 2 * timeout)

    bundle = EvidenceBundle(query=query, provider=provider.name,
                            budget_chars=budget_chars,
                            per_source_cap_chars=per_source_cap_chars)
    requested = min(2 * search_candidates, _MAX_SEARCH_CANDIDATES)
    try:
        results = provider.search(query, requested)
    except netpolicy.NetworkPolicyError:
        raise
    except Exception as exc:
        bundle.search_status = SEARCH_FAILED
        bundle.search_error = _describe(exc)
        return bundle

    results = dedup_results(results)[:search_candidates]
    if not results:
        bundle.search_status = SEARCH_EMPTY
        bundle.search_error = "the search backend returned no usable results"
        return bundle
    bundle.search_status = SEARCH_OK

    sources = [
        Source(id=f"S{i}", url=r.url, canonical_url=canonicalize_url(r.url),
               title=r.title, snippet=r.snippet, provider_rank=r.rank,
               retrieval_status=STATUS_SKIPPED,
               grounding=(GROUNDING_SNIPPET_ONLY if r.snippet.strip()
                          else GROUNDING_FAILED))
        for i, r in enumerate(results, 1)
    ]
    bundle.sources = sources

    to_fetch = sources[:fetch_top]
    pages: dict[str, PageDocument] = {}
    if to_fetch:
        pages = _read_pages(to_fetch, fetch, timeout, deadline)
        _drop_duplicate_pages(to_fetch, pages)

    inputs: list[tuple[str, str, str]] = []
    for s in sources:
        if s.id in pages:
            inputs.append((s.id, CHUNK_PAGE, pages[s.id].text))
        elif s.snippet.strip():
            inputs.append((s.id, CHUNK_SNIPPET, s.snippet))
    bundle.chunks = select_evidence(inputs, query, budget=budget_chars,
                                    per_source=per_source_cap_chars)
    return bundle
