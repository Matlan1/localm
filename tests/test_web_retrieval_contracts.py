# SPDX-License-Identifier: AGPL-3.0-or-later
"""EvidenceBundle's grounding claim (localm.web_retrieval.contracts): one
dedicated invariant test, not scattered incidental assertions.

Every consumer of a retrieval - the GUI's tool-event card and toast, the
prompt text injected back into the model, the scheduled-jobs and coder loops,
and an exported transcript - derives its "were pages actually read" claim from
``EvidenceBundle.grounding`` or the values built from it (``.page_backed``,
``to_dict()["grounding"]``, the first line of ``to_prompt_text()``). That
property is already exercised incidentally through ``retrieve()`` in
tests/test_web_retrieval_retrieve.py and through the jobs/GUI surfaces in
tests/test_jobs_web_search.py and tests-js/web.test.mjs. Per this repo's
diff-review-discipline.md item 35, a dedicated test bound to the contract
itself is what stays comprehensive as new call sites are added - scattered
coverage elsewhere goes stale silently the moment a new consumer is wired in
without anyone remembering to extend it.

This file constructs ``EvidenceBundle``/``Source``/``EvidenceChunk`` directly
(no ``retrieve()``, no network), so every scenario here is a pure test of the
contract every consumer actually reads.
"""

from __future__ import annotations

from localm.web_retrieval import (
    GROUNDING_FAILED,
    GROUNDING_PAGE_BACKED,
    GROUNDING_SNIPPET_ONLY,
    EvidenceBundle,
    EvidenceChunk,
    Source,
)

QUERY = "example query"


def _bundle(sources, chunks, *, search_status="ok"):
    b = EvidenceBundle(query=QUERY, provider="stub", search_status=search_status)
    b.sources = sources
    b.chunks = chunks
    return b


def _assert_never_claims_page_backed(b: EvidenceBundle) -> None:
    """The one assertion every scenario in this file makes: nothing a consumer
    reads to decide "were pages read" says page-backed.

    The BUNDLE-level claim is the property/dict value and the FIRST line of
    ``to_prompt_text()`` (what the GUI, jobs and coder actually inject as
    "Grounding: ..."). A LATER line may still legitimately carry a per-SOURCE
    "(page-backed)" label for a page that was read but lost the evidence
    budget race - see TestAdversarialPageReadButUnselected below - so only the
    first line is checked here; that per-source case gets its own explicit
    assertion where it is relevant.
    """
    assert b.grounding != GROUNDING_PAGE_BACKED
    assert b.page_backed is False
    assert b.to_dict()["grounding"] != GROUNDING_PAGE_BACKED
    first_line = b.to_prompt_text().splitlines()[0]
    assert first_line.startswith("Grounding: ")
    assert "page-backed" not in first_line


class TestSnippetOnlyBundleNeverClaimsPageBacked:
    def test_one_snippet_source(self):
        source = Source(id="S1", url="https://a.example/",
                        canonical_url="https://a.example/", title="A",
                        snippet="a snippet", provider_rank=1,
                        retrieval_status="skipped",
                        grounding=GROUNDING_SNIPPET_ONLY)
        chunk = EvidenceChunk(source_id="S1", text="a snippet", score=0.0,
                              offset=0, kind="snippet")
        b = _bundle([source], [chunk])
        assert b.grounding == GROUNDING_SNIPPET_ONLY
        _assert_never_claims_page_backed(b)

    def test_several_snippet_sources(self):
        sources = [Source(id=f"S{i}", url=f"https://{i}.example/",
                          canonical_url=f"https://{i}.example/", title=f"T{i}",
                          snippet=f"snippet {i}", provider_rank=i,
                          retrieval_status="skipped",
                          grounding=GROUNDING_SNIPPET_ONLY)
                  for i in range(1, 4)]
        chunks = [EvidenceChunk(source_id=s.id, text=s.snippet, score=0.0,
                                offset=0, kind="snippet") for s in sources]
        b = _bundle(sources, chunks)
        assert b.grounding == GROUNDING_SNIPPET_ONLY
        _assert_never_claims_page_backed(b)


class TestFailedBundleNeverClaimsPageBacked:
    def test_no_sources_no_chunks_search_failed(self):
        b = _bundle([], [], search_status="failed")
        b.search_error = "RuntimeError: rate-limited"
        assert b.grounding == GROUNDING_FAILED
        _assert_never_claims_page_backed(b)

    def test_sources_present_but_every_read_failed(self):
        sources = [Source(id="S1", url="https://a.example/",
                          canonical_url="https://a.example/", title="A",
                          snippet="", provider_rank=1, retrieval_status="failed",
                          grounding=GROUNDING_FAILED, error="RuntimeError: HTTP 500")]
        b = _bundle(sources, [])
        assert b.grounding == GROUNDING_FAILED
        _assert_never_claims_page_backed(b)


class TestAdversarialPageReadButUnselected:
    """The trap this dedicated test exists for: a source's OWN grounding is
    legitimately "page-backed" (its page really was fetched and had text),
    but its chunk lost the evidence-budget race, so the bundle never actually
    shows the model any page text for it. An implementation that derived the
    bundle's headline claim from a source's status instead of strictly from
    the chunks it actually carries would wrongly report this bundle as
    page-backed."""

    def test_source_marked_page_backed_but_no_chunk_survived(self):
        page_source = Source(id="S1", url="https://a.example/",
                             canonical_url="https://a.example/", title="A",
                             snippet="", provider_rank=1, retrieval_status="fetched",
                             grounding=GROUNDING_PAGE_BACKED,
                             final_url="https://a.example/")
        snippet_source = Source(id="S2", url="https://b.example/",
                                canonical_url="https://b.example/", title="B",
                                snippet="a real snippet", provider_rank=2,
                                retrieval_status="skipped",
                                grounding=GROUNDING_SNIPPET_ONLY)
        chunk = EvidenceChunk(source_id="S2", text="a real snippet", score=0.0,
                              offset=0, kind="snippet")
        b = _bundle([page_source, snippet_source], [chunk])

        # The trap, made explicit: a PER-SOURCE label does legitimately say
        # page-backed here, and it is allowed to appear later in the text.
        assert page_source.grounding == GROUNDING_PAGE_BACKED
        assert "(page-backed)" in b.to_prompt_text()

        # And yet the BUNDLE never claims it - this is the property every
        # consumer actually relies on to decide what to show or tell the model.
        assert b.grounding == GROUNDING_SNIPPET_ONLY
        _assert_never_claims_page_backed(b)

    def test_source_marked_page_backed_with_zero_surviving_chunks_is_bundle_failed(self):
        page_source = Source(id="S1", url="https://a.example/",
                             canonical_url="https://a.example/", title="A",
                             snippet="", provider_rank=1, retrieval_status="fetched",
                             grounding=GROUNDING_PAGE_BACKED,
                             final_url="https://a.example/")
        b = _bundle([page_source], [])
        assert b.grounding == GROUNDING_FAILED
        _assert_never_claims_page_backed(b)
