# SPDX-License-Identifier: AGPL-3.0-or-later
"""Shared web retrieval for chat, scheduled jobs and the coder.

Owns provider adaptation (DuckDuckGo HTML, SearXNG), normalised search
results, URL canonicalization and duplicate removal, page acquisition through
``localm.netpolicy``, main-content extraction, query-aware chunk selection and
the evidence bundle those produce. ``localm.netpolicy`` remains the only
outbound transport and policy layer; nothing in this package opens a socket
of its own.

Entry points: ``retrieve`` (search plus bounded page reads into an
``EvidenceBundle``), ``search`` (search only), ``extract_page`` and
``select_evidence`` (the extraction and selection steps on their own).
"""

from .canonical import canonicalize_url, dedup_key, dedup_results
from .chunking import make_chunks, query_terms, rank_chunks, select_evidence
from .contracts import (
    EVIDENCE_BUDGET_CHARS,
    FETCH_TOP,
    GROUNDING_FAILED,
    GROUNDING_PAGE_BACKED,
    GROUNDING_SNIPPET_ONLY,
    GROUNDING_STATES,
    PER_SOURCE_CAP_CHARS,
    SEARCH_CANDIDATES,
    EvidenceBundle,
    EvidenceChunk,
    PageDocument,
    SearchProvider,
    SearchProviderError,
    SearchResult,
    Source,
)
from .extract import ExtractedPage, extract_page, html_to_main_text
from .providers import (
    DuckDuckGoHTMLProvider,
    SearXNGProvider,
    provider_from_config,
    search,
)
from .retrieve import retrieve

__all__ = [
    "EVIDENCE_BUDGET_CHARS",
    "FETCH_TOP",
    "GROUNDING_FAILED",
    "GROUNDING_PAGE_BACKED",
    "GROUNDING_SNIPPET_ONLY",
    "GROUNDING_STATES",
    "PER_SOURCE_CAP_CHARS",
    "SEARCH_CANDIDATES",
    "DuckDuckGoHTMLProvider",
    "EvidenceBundle",
    "EvidenceChunk",
    "ExtractedPage",
    "PageDocument",
    "SearXNGProvider",
    "SearchProvider",
    "SearchProviderError",
    "SearchResult",
    "Source",
    "canonicalize_url",
    "dedup_key",
    "dedup_results",
    "extract_page",
    "html_to_main_text",
    "make_chunks",
    "provider_from_config",
    "query_terms",
    "rank_chunks",
    "retrieve",
    "search",
    "select_evidence",
]
