# SPDX-License-Identifier: AGPL-3.0-or-later
"""Typed contracts for web retrieval.

Search results, the provider interface, page documents, evidence chunks and the
evidence bundle a retrieval returns. Every string that came from the network
(titles, snippets, page text, error text quoting a response) is untrusted and is
NOT neutralised here; the consumer that puts it in front of a model or a user
applies its own neutralisation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional, Protocol, runtime_checkable

#: Search results requested from the provider per retrieval.
SEARCH_CANDIDATES = 5
#: Distinct candidates whose pages are read, highest provider rank first.
FETCH_TOP = 3
#: Total characters of evidence one bundle may carry.
EVIDENCE_BUDGET_CHARS = 12_000
#: Characters of evidence one source may contribute.
PER_SOURCE_CAP_CHARS = 4_000

GROUNDING_PAGE_BACKED = "page-backed"
GROUNDING_SNIPPET_ONLY = "snippet-only"
GROUNDING_FAILED = "failed"
GROUNDING_STATES = (GROUNDING_PAGE_BACKED, GROUNDING_SNIPPET_ONLY,
                    GROUNDING_FAILED)

STATUS_FETCHED = "fetched"
STATUS_FAILED = "failed"
STATUS_SKIPPED = "skipped"
STATUS_DUPLICATE = "duplicate"
RETRIEVAL_STATUSES = (STATUS_FETCHED, STATUS_FAILED, STATUS_SKIPPED,
                      STATUS_DUPLICATE)

SEARCH_OK = "ok"
SEARCH_EMPTY = "empty"
SEARCH_FAILED = "failed"

CHUNK_PAGE = "page"
CHUNK_SNIPPET = "snippet"


class SearchProviderError(RuntimeError):
    """The search backend answered but yielded nothing parseable, or its
    response was malformed. A policy refusal is a ``NetworkPolicyError``,
    never this."""


@dataclass(frozen=True)
class SearchResult:
    """One provider hit. ``rank`` is the 1-based position in the provider's
    own result order; ``provider`` is the provider's ``name``."""

    title: str
    url: str
    snippet: str
    rank: int
    provider: str

    def to_legacy(self) -> dict:
        """The ``{"title", "url", "snippet"}`` dict ``netpolicy.web_search``
        returns."""
        return {"title": self.title, "url": self.url, "snippet": self.snippet}


@runtime_checkable
class SearchProvider(Protocol):
    """A search backend. ``search`` returns at most ``max_results`` results in
    provider order with ranks starting at 1. Raises ``NetworkPolicyError``
    when the policy refuses the request and ``SearchProviderError`` (or any
    transport exception) when the backend fails. Never falls back to another
    provider."""

    name: str

    def search(self, query: str, max_results: int) -> list[SearchResult]: ...


@dataclass(frozen=True)
class PageDocument:
    """A fetched and extracted page. ``region`` names the part of the document
    the text came from: ``main``, ``article``, ``body`` (whole document with
    chrome removed) or ``text`` (a non-HTML response used verbatim)."""

    url: str
    final_url: str
    title: str
    text: str
    content_type: str
    region: str

    @property
    def char_count(self) -> int:
        return len(self.text)


@dataclass(frozen=True)
class EvidenceChunk:
    """A span of evidence attributed to ``source_id``. ``offset`` is the
    character offset of the span in the source's extracted text (0 for a
    snippet); ``kind`` is ``page`` or ``snippet``."""

    source_id: str
    text: str
    score: float
    offset: int
    kind: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Source:
    """One candidate from the search, identified by a request-local ``id``
    (``S1``, ``S2``, ...) that is stable for the life of the bundle.

    ``retrieval_status`` says what happened to the read: ``fetched``, ``failed``,
    ``skipped`` (not among the top candidates read) or ``duplicate`` (its page
    resolved to the same final URL as a higher-ranked fetched source).

    ``grounding`` says what evidence backs the source: ``page-backed`` when page
    text was extracted; ``failed`` when a read was attempted and failed, or when
    the source has neither page text nor a snippet; ``snippet-only`` otherwise
    (not attempted, skipped, duplicate, or the page had no extractable text)
    while the provider snippet is the only evidence.
    """

    id: str
    url: str
    canonical_url: str
    title: str
    snippet: str
    provider_rank: int
    retrieval_status: str = STATUS_SKIPPED
    grounding: str = GROUNDING_SNIPPET_ONLY
    final_url: Optional[str] = None
    error: Optional[str] = None
    region: Optional[str] = None
    text_chars: int = 0

    def mark_failed(self, error: str) -> None:
        self.retrieval_status = STATUS_FAILED
        self.grounding = GROUNDING_FAILED
        self.error = error

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EvidenceBundle:
    """What one retrieval produced.

    ``search_status`` is ``ok``, ``empty`` (the provider returned no usable
    result) or ``failed`` (the provider raised; ``search_error`` carries the
    reason and ``provider`` names the backend that failed). ``sources`` are in
    provider rank order after duplicate removal; ``chunks`` are grouped by
    source in that order and, within a source, by offset.
    """

    query: str
    provider: str
    budget_chars: int = EVIDENCE_BUDGET_CHARS
    per_source_cap_chars: int = PER_SOURCE_CAP_CHARS
    search_status: str = SEARCH_FAILED
    search_error: Optional[str] = None
    sources: list[Source] = field(default_factory=list)
    chunks: list[EvidenceChunk] = field(default_factory=list)

    @property
    def grounding(self) -> str:
        """``page-backed`` when the bundle carries at least one page chunk,
        else ``snippet-only`` when any evidence chunk exists, else ``failed``.
        A page-backed source that contributed no chunk (budget exhausted) does
        not make the bundle page-backed."""
        if any(c.kind == CHUNK_PAGE for c in self.chunks):
            return GROUNDING_PAGE_BACKED
        if self.chunks:
            return GROUNDING_SNIPPET_ONLY
        return GROUNDING_FAILED

    @property
    def page_backed(self) -> bool:
        return self.grounding == GROUNDING_PAGE_BACKED

    @property
    def total_chars(self) -> int:
        return sum(len(c.text) for c in self.chunks)

    def source(self, source_id: str) -> Optional[Source]:
        for s in self.sources:
            if s.id == source_id:
                return s
        return None

    def chunks_for(self, source_id: str) -> list[EvidenceChunk]:
        return [c for c in self.chunks if c.source_id == source_id]

    def evidence_text(self) -> str:
        """Every chunk's text joined with blank lines, in bundle order."""
        return "\n\n".join(c.text for c in self.chunks)

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "provider": self.provider,
            "search_status": self.search_status,
            "search_error": self.search_error,
            "grounding": self.grounding,
            "budget_chars": self.budget_chars,
            "per_source_cap_chars": self.per_source_cap_chars,
            "total_chars": self.total_chars,
            "sources": [s.to_dict() for s in self.sources],
            "chunks": [c.to_dict() for c in self.chunks],
        }

    def to_prompt_text(self) -> str:
        """A plain-text rendering: a source list with id, title, URL and
        grounding, then every chunk prefixed with its source id. Untrusted
        text is included verbatim."""
        lines: list[str] = []
        if self.search_status != SEARCH_OK:
            lines.append(f"Search {self.search_status}"
                         + (f": {self.search_error}" if self.search_error else ""))
        if self.sources:
            lines.append("Sources:")
            for s in self.sources:
                shown = s.final_url or s.url
                detail = s.grounding
                if s.error:
                    detail += f", {s.error}"
                title = s.title.strip() or "(untitled)"
                lines.append(f"[{s.id}] {title} - {shown} ({detail})")
        if self.chunks:
            lines.append("")
            lines.append("Evidence:")
            for c in self.chunks:
                tag = f"[{c.source_id}]" if c.kind == CHUNK_PAGE \
                    else f"[{c.source_id} snippet]"
                lines.append(f"{tag} {c.text}")
        return "\n".join(lines)
