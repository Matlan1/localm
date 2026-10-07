# SPDX-License-Identifier: AGPL-3.0-or-later
"""The retrieval controller: search, deduplicate, read the top pages
concurrently through ``localm.netpolicy``, extract, select evidence.

``retrieve`` returns an ``EvidenceBundle`` for every outcome except a policy
refusal or an empty query. A failing provider is reported in
``bundle.search_status`` / ``bundle.search_error`` and is never replaced by
another provider. A failing page read is reported on its ``Source`` and never
fails the retrieval. Both error texts are plain-language sentences from
``errors.describe_failure``.

A page read goes through ``sites.read_url`` (GitHub and Stack Exchange
content endpoints first, the page itself last) and is sent once more when it
fails with a reset, refused or unreachable connection, or an incomplete body.
"""

from __future__ import annotations

import concurrent.futures
import functools
import hashlib
import time
import urllib.parse
from typing import Callable, Optional

from localm import netpolicy
from localm.debuglog import logger
from localm.netpin import ReadBudgetExceeded

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
    PAGE_READ_DEADLINE,
    PAGE_READ_TIMEOUT,
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
from .errors import describe_failure, failure_kind
from .extract import extract_page
from .providers import observe_requests, provider_from_config
from .sites import read_url

#: ``fetch(url, timeout=...) -> (final_url, content_type, text)``.
Fetcher = Callable[..., tuple[str, str, str]]

_MAX_SEARCH_CANDIDATES = 10
_ERROR_TEXT_CAP = 300
_HTML_SNIFF_BYTES = 1024
_PAGE_RETRY_KINDS = frozenset({"reset", "refused", "unreachable", "incomplete"})
_WORKER_GRACE = 1.0


def _default_fetch(url: str, *, timeout: float,
                   finish_by: Optional[float] = None
                   ) -> tuple[str, str, str]:
    """``netpolicy.safe_fetch``. With *finish_by* (a ``time.monotonic()``
    value) the call gets only the time left until then, and raises
    ``ReadBudgetExceeded`` when none is left."""
    if finish_by is None:
        return netpolicy.safe_fetch(url, timeout=timeout)
    left = finish_by - time.monotonic()
    if left <= 0:
        raise ReadBudgetExceeded(0.0, url)
    return netpolicy.safe_fetch(url, timeout=min(timeout, left),
                                total_timeout=left)


def search_failure_text(exc: BaseException,
                        provider: Optional[SearchProvider] = None) -> str:
    """The ``search_error`` sentence for a failed search by *provider*
    (default: ``provider_from_config()``): the specific cause, and for a
    configured SearXNG instance which backend it was. A
    ``SearchProviderError`` keeps its own message."""
    from .contracts import SearchProviderError
    if isinstance(exc, SearchProviderError):
        return str(exc).strip()[:_ERROR_TEXT_CAP * 3] or "the search failed"
    if provider is None:
        provider = provider_from_config()
    url = (getattr(provider, "endpoint", "")
           or getattr(provider, "base_url", "") or "")
    reason = describe_failure(exc, url)
    if getattr(provider, "name", "") == "searxng":
        return f"The search backend set in Settings > Network failed: {reason}."
    host = urllib.parse.urlparse(url).hostname or ""
    if not (host and reason.startswith(host)):
        reason = reason[:1].upper() + reason[1:]
    return f"{reason}."


def _looks_like_html(content_type: str, body: str) -> bool:
    if "html" in (content_type or "").lower():
        return True
    head = (body or "")[:_HTML_SNIFF_BYTES].lower()
    return "<html" in head or "<!doctype html" in head


def _read_page(source: Source, fetch: Fetcher, timeout: float,
               on_endpoint: Optional[Callable[[str], None]] = None
               ) -> PageDocument:
    try:
        final_url, content_type, body = read_url(
            source.url, fetch, timeout=timeout, on_endpoint=on_endpoint)
    except netpolicy.NetworkPolicyError:
        raise
    except Exception as exc:
        if failure_kind(exc) not in _PAGE_RETRY_KINDS:
            raise
        logger.debug("web retrieval: page read retried after %s",
                     type(exc).__name__)
        final_url, content_type, body = read_url(
            source.url, fetch, timeout=timeout, on_endpoint=on_endpoint)
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


def _read_pages(to_fetch: list[Source], fetch: Fetcher, timeout: float,
                deadline: float,
                on_endpoint: Optional[Callable[[str], None]] = None
                ) -> dict[str, PageDocument]:
    """Read every source in *to_fetch* concurrently. A read still running at
    *deadline* seconds is recorded as failed; its worker finishes on its own
    transport timeout."""
    pages: dict[str, PageDocument] = {}
    pool = concurrent.futures.ThreadPoolExecutor(
        max_workers=len(to_fetch), thread_name_prefix="web-retrieval")
    try:
        futures = {pool.submit(_read_page, s, fetch, timeout, on_endpoint): s
                   for s in to_fetch}
        done, pending = concurrent.futures.wait(futures, timeout=deadline)
        for fut in pending:
            source = futures[fut]
            host = urllib.parse.urlparse(source.url).hostname or "the site"
            source.mark_failed(
                f"{host} did not finish loading within {deadline:g}s")
            fut.cancel()
        for fut in done:
            source = futures[fut]
            try:
                page = fut.result()
            except netpolicy.NetworkPolicyError as exc:
                source.mark_failed(f"refused by policy: {exc}"[:_ERROR_TEXT_CAP])
            except Exception as exc:
                logger.debug("web retrieval: page read failed (%s)",
                             type(exc).__name__)
                source.mark_failed(describe_failure(exc, source.url))
            else:
                _record_page(source, page, pages)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    return pages


def _drop_duplicate_pages(fetched: list[Source],
                          pages: dict[str, PageDocument]) -> None:
    """A fetched source whose final URL has the same ``dedup_key`` as a
    higher-ranked fetched source, or whose page text is the same apart from
    whitespace, loses its page and becomes ``duplicate``."""
    seen: dict[str, str] = {}
    for source in fetched:
        if source.id not in pages:
            continue
        normalized = " ".join(pages[source.id].text.split())
        keys = (dedup_key(source.final_url or source.url),
                "text:" + hashlib.sha256(normalized.encode("utf-8")).hexdigest())
        first = next((seen[k] for k in keys if k in seen), None)
        if first is not None:
            del pages[source.id]
            source.retrieval_status = STATUS_DUPLICATE
            source.grounding = (GROUNDING_SNIPPET_ONLY if source.snippet.strip()
                                else GROUNDING_FAILED)
            source.error = f"same page as {first}"
        else:
            for k in keys:
                seen[k] = source.id


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
    on_endpoint: Optional[Callable[[str], None]] = None,
) -> EvidenceBundle:
    """Search *query*, read the top pages and return an ``EvidenceBundle``.

    *provider* defaults to ``provider_from_config()``. *search_candidates* is
    clamped to 1..10 and *fetch_top* to 0..*search_candidates*. The provider
    is asked for twice *search_candidates* results (at most 10); after
    duplicate removal the first *search_candidates* become sources. *fetch*
    defaults to ``netpolicy.safe_fetch``, every call of it (content endpoints
    and retries included) given only the time left until one second after
    the page-read deadline, which starts when the page reads start;
    *fetch_timeout* (each connect attempt and each wait for data) to
    ``PAGE_READ_TIMEOUT``; *deadline_seconds* (the wait for all
    page reads together) to ``PAGE_READ_DEADLINE``, or twice *fetch_timeout*
    when only *fetch_timeout* is given. *on_endpoint*, when given, is
    called with every search request URL (``providers.observe_requests``)
    and every site content-endpoint URL (``sites.read_url``) before it is
    requested, the latter from the page-read worker threads.

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
    timeout = (fetch_timeout if fetch_timeout is not None
               else PAGE_READ_TIMEOUT)
    if deadline_seconds is not None:
        deadline = float(deadline_seconds)
    elif fetch_timeout is not None:
        deadline = float(2 * timeout)
    else:
        deadline = float(PAGE_READ_DEADLINE)
    default_fetch = fetch is None

    bundle = EvidenceBundle(query=query, provider=provider.name,
                            budget_chars=budget_chars,
                            per_source_cap_chars=per_source_cap_chars)
    requested = min(2 * search_candidates, _MAX_SEARCH_CANDIDATES)
    try:
        with observe_requests(on_endpoint):
            results = provider.search(query, requested)
    except netpolicy.NetworkPolicyError:
        raise
    except Exception as exc:
        logger.debug("web retrieval: search failed (%s)", type(exc).__name__)
        bundle.search_status = SEARCH_FAILED
        bundle.search_error = search_failure_text(exc, provider)
        return bundle

    results = dedup_results(results)[:search_candidates]
    if not results:
        bundle.search_status = SEARCH_EMPTY
        bundle.search_error = "the search backend returned no usable results"
        return bundle
    bundle.search_status = SEARCH_OK
    bundle.provider = results[0].provider or bundle.provider

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
        if default_fetch:
            fetch = functools.partial(
                _default_fetch,
                finish_by=time.monotonic() + deadline + _WORKER_GRACE)
        pages = _read_pages(to_fetch, fetch, timeout, deadline, on_endpoint)
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
