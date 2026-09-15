# SPDX-License-Identifier: AGPL-3.0-or-later
"""Chunking, query scoring and budgeted evidence selection
(localm.web_retrieval.chunking)."""

from __future__ import annotations

from localm.web_retrieval import (
    make_chunks,
    query_terms,
    rank_chunks,
    select_evidence,
)
from localm.web_retrieval.chunking import (
    CHUNK_MAX_CHARS,
    score_text,
    split_paragraphs,
)
from tests._web_retrieval_fixtures import (
    ANSWER,
    QUERY,
    boilerplate_text,
    filler_paragraph,
)


def _doc(paragraphs: list[str]) -> str:
    return "\n\n".join(paragraphs)


class TestSplitParagraphs:
    def test_offsets_point_at_the_paragraphs(self):
        text = "  first para\n\n\n\nsecond\npara  \n\nthird"
        out = split_paragraphs(text)
        assert [p for _, p in out] == ["first para", "second\npara", "third"]
        for offset, para in out:
            assert text[offset:offset + len(para)] == para

    def test_blank_only_input(self):
        assert split_paragraphs("\n\n  \n\n") == []


class TestMakeChunks:
    def test_paragraphs_merged_up_to_target_and_never_past_max(self):
        text = _doc([filler_paragraph(i, 300) for i in range(12)])
        chunks = make_chunks(text)
        assert 2 <= len(chunks) < 12
        assert all(len(c.text) <= CHUNK_MAX_CHARS for c in chunks)
        assert [c.index for c in chunks] == list(range(len(chunks)))
        for c in chunks:
            first_line = c.text.split("\n", 1)[0]
            assert text[c.offset:].startswith(first_line)

    def test_every_paragraph_survives(self):
        paras = [filler_paragraph(i, 250) for i in range(9)]
        joined = "\n".join(c.text for c in make_chunks(_doc(paras)))
        for p in paras:
            assert p in joined

    def test_long_paragraph_split_at_sentence_ends(self):
        para = " ".join(f"Sentence {i} ends here." for i in range(200))
        chunks = make_chunks(para)
        assert len(chunks) > 1
        assert all(len(c.text) <= CHUNK_MAX_CHARS for c in chunks)
        assert all(c.text.endswith(".") for c in chunks)
        for c in chunks:
            assert para[c.offset:].startswith(c.text)

    def test_unbreakable_run_hard_split(self):
        para = "word " * 800
        chunks = make_chunks(para.strip())
        assert len(chunks) > 1
        assert all(len(c.text) <= CHUNK_MAX_CHARS for c in chunks)
        assert " ".join(c.text for c in chunks).split() == para.split()

    def test_no_space_run_hard_split(self):
        para = "x" * 3000
        chunks = make_chunks(para)
        assert "".join(c.text for c in chunks) == para
        assert all(len(c.text) <= CHUNK_MAX_CHARS for c in chunks)

    def test_empty_text(self):
        assert make_chunks("") == []


class TestQueryTerms:
    def test_stopwords_removed_lowercased_deduped(self):
        assert query_terms("What is the Linz Museum, the LINZ one?") == \
            ["linz", "museum", "one"]

    def test_all_stopwords_fall_back_to_every_token(self):
        assert query_terms("what is the") == ["what", "is", "the"]

    def test_unicode_and_numbers(self):
        assert query_terms("Grüße 2026 x") == ["grüße", "2026"]

    def test_empty(self):
        assert query_terms("") == []


class TestScoreText:
    def test_more_terms_score_higher_and_no_terms_is_zero(self):
        terms = ["linz", "museum", "hours"]
        full = score_text("the linz museum hours are posted", terms)
        partial = score_text("a museum in another town", terms)
        none = score_text("committee budget minutes", terms)
        assert full > partial > none == 0.0

    def test_phrase_bonus(self):
        terms = ["linz", "museum"]
        with_phrase = score_text("visit the linz museum today", terms,
                                 phrase="linz museum")
        without = score_text("the museum in linz today", terms,
                             phrase="linz museum")
        assert with_phrase > without

    def test_prefix_match_both_directions(self):
        assert score_text("the linzer torte", ["linz"]) > 0
        assert score_text("open at nine", ["opening"]) > 0
        assert score_text("ab cd", ["abc"]) == 0.0

    def test_term_frequency_saturates(self):
        low = score_text("linz", ["linz"])
        high = score_text("linz " * 50, ["linz"])
        assert high > low
        assert high == score_text("linz " * 3, ["linz"])

    def test_empty_terms(self):
        assert score_text("anything", []) == 0.0


class TestRankChunks:
    def test_answer_behind_7000_chars_of_boilerplate_ranks_first(self):
        text = _doc([boilerplate_text(7000), filler_paragraph(1), ANSWER,
                     filler_paragraph(2)])
        assert text.find(ANSWER) > 6000
        ranked = rank_chunks(text, QUERY)
        assert ANSWER in ranked[0].chunk.text
        assert ranked[0].score > 0
        assert ranked[0].chunk.offset > 6000

    def test_ties_keep_document_order(self):
        text = _doc([filler_paragraph(i, 300) for i in range(6)])
        ranked = rank_chunks(text, QUERY)
        assert all(s.score == 0 for s in ranked)
        assert [s.chunk.index for s in ranked] == list(range(len(ranked)))


class TestSelectEvidence:
    def _pages(self, n: int, matching: bool = True) -> list[tuple[str, str, str]]:
        out = []
        for i in range(1, n + 1):
            paras = [filler_paragraph(j, 350) for j in range(20)]
            if matching:
                paras.insert(10, f"{ANSWER} Source number {i}.")
            out.append((f"S{i}", "page", _doc(paras)))
        return out

    def test_budget_and_per_source_cap_respected(self):
        pages = [(f"S{i}", "page",
                  _doc([f"{ANSWER} Variant {j} of source {i}."
                        for j in range(60)])) for i in range(1, 5)]
        chunks = select_evidence(pages, QUERY, budget=12_000, per_source=4_000)
        assert sum(len(c.text) for c in chunks) <= 12_000
        for sid in ("S1", "S2", "S3", "S4"):
            assert 0 < sum(len(c.text) for c in chunks
                           if c.source_id == sid) <= 4_000

    def test_every_page_source_gets_its_best_chunk_first(self):
        chunks = select_evidence(self._pages(3), QUERY)
        for sid in ("S1", "S2", "S3"):
            mine = [c for c in chunks if c.source_id == sid]
            assert mine and any(ANSWER in c.text for c in mine)

    def test_grouped_by_source_order_and_by_offset_within(self):
        chunks = select_evidence(self._pages(3), QUERY)
        order = [c.source_id for c in chunks]
        assert order == sorted(order, key=lambda s: int(s[1:]))
        for sid in ("S1", "S2", "S3"):
            offsets = [c.offset for c in chunks if c.source_id == sid]
            assert offsets == sorted(offsets)

    def test_zero_match_source_still_contributes_its_lead_chunk_only(self):
        chunks = select_evidence(self._pages(1, matching=False), QUERY)
        assert len(chunks) == 1
        assert chunks[0].offset == 0 and chunks[0].score == 0.0

    def test_snippets_included_once_first_and_capped(self):
        sources = [("S1", "snippet", "  a   snippet  "),
                   ("S2", "page", _doc([ANSWER])),
                   ("S3", "snippet", "x" * 5000)]
        chunks = select_evidence(sources, QUERY, per_source=4_000)
        kinds = [(c.source_id, c.kind) for c in chunks]
        assert kinds == [("S1", "snippet"), ("S2", "page"), ("S3", "snippet")]
        assert chunks[0].text == "a snippet"
        assert len(chunks[2].text) == 4_000

    def test_empty_and_blank_sources_yield_nothing(self):
        assert select_evidence([("S1", "page", ""), ("S2", "snippet", "  ")],
                               QUERY) == []

    def test_tiny_budget_selects_only_what_fits(self):
        sources = [("S1", "snippet", "short"), ("S2", "page", _doc([ANSWER]))]
        chunks = select_evidence(sources, QUERY, budget=10)
        assert [c.source_id for c in chunks] == ["S1"]

    def test_stage_two_prefers_higher_score_across_sources(self):
        weak = _doc([filler_paragraph(j, 350) for j in range(6)]
                    + ["A museum is mentioned once."])
        strong = _doc([f"{ANSWER} Detail {j}." for j in range(6)])
        chunks = select_evidence([("S1", "page", weak), ("S2", "page", strong)],
                                 QUERY, budget=2_600, per_source=2_000)
        by_source = {sid: sum(len(c.text) for c in chunks if c.source_id == sid)
                     for sid in ("S1", "S2")}
        assert by_source["S2"] > by_source["S1"] > 0
