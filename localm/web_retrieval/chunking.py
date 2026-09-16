# SPDX-License-Identifier: AGPL-3.0-or-later
"""Query-aware chunk selection under an evidence budget.

Extracted text is split into paragraph-aligned chunks (about 900 characters,
at most 1,400; a longer paragraph is split at sentence ends). Each chunk is
scored against the query: the share of distinct query terms it contains, a
saturating term-frequency bonus, and a bonus when the whole query phrase
occurs. Query terms are the query's word tokens minus a small English and
German stopword list; when every token is a stopword, all tokens count.

``select_evidence`` fills the budget in three stages: provider snippets first
(each is its source's only evidence), then every page source's best chunk in
source order, then remaining positive-scoring chunks from any source by score.
No source exceeds its per-source cap and the total never exceeds the budget.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Sequence

from .contracts import (
    CHUNK_PAGE,
    CHUNK_SNIPPET,
    EVIDENCE_BUDGET_CHARS,
    PER_SOURCE_CAP_CHARS,
    EvidenceChunk,
)

CHUNK_TARGET_CHARS = 900
CHUNK_MAX_CHARS = 1_400

_WORD_RE = re.compile(r"\w+")
_SENTENCE_BREAK_RE = re.compile(r"(?<=[.!?])\s+")
_MIN_TERM_LEN = 2
_PREFIX_MATCH_MIN_LEN = 4
_PHRASE_BONUS = 3.0
_PHRASE_MIN_LEN = 8
_TF_SATURATION = 3
_TF_WEIGHT = 0.1
_COVERAGE_WEIGHT = 10.0

_STOPWORDS = frozenset("""
a an and are as at be been but by can could did do does for from had has have
how i if in into is it its of on or should that the their there these they this
to was we were what when where which who why will with would you your
der die das den dem des ein eine einer eines einem einen und oder aber ist sind
war waren wird werden wurde wie was wer wo wann warum wen wem mit von zu zum
zur im am in auf fuer für den nicht auch noch nur bei nach aus ueber über
unter sich ich du er sie es wir ihr man
""".split())


@dataclass(frozen=True)
class Chunk:
    """A span of text starting at character ``offset`` of its document;
    ``index`` is the chunk's position in document order."""

    offset: int
    text: str
    index: int


@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    score: float


def split_paragraphs(text: str) -> list[tuple[int, str]]:
    """``(offset, paragraph)`` for every non-blank paragraph, paragraphs being
    separated by a blank line."""
    out: list[tuple[int, str]] = []
    offset = 0
    for para in text.split("\n\n"):
        stripped = para.strip()
        if stripped:
            lead = len(para) - len(para.lstrip())
            out.append((offset + lead, stripped))
        offset += len(para) + 2
    return out


def _split_long(offset: int, para: str, maximum: int) -> list[tuple[int, str]]:
    """Split a paragraph longer than *maximum* at sentence ends, then at the
    last space before *maximum*, then hard."""
    pieces: list[tuple[int, str]] = []
    sentences: list[tuple[int, str]] = []
    pos = 0
    for m in _SENTENCE_BREAK_RE.finditer(para):
        sentences.append((pos, para[pos:m.start()]))
        pos = m.end()
    sentences.append((pos, para[pos:]))

    cur_start = None
    cur: list[str] = []
    cur_len = 0
    for s_off, sentence in sentences:
        if not sentence:
            continue
        while len(sentence) > maximum:
            if cur:
                pieces.append((offset + cur_start, " ".join(cur)))
                cur, cur_len, cur_start = [], 0, None
            cut = sentence.rfind(" ", 0, maximum)
            if cut < maximum // 2:
                cut = maximum
            pieces.append((offset + s_off, sentence[:cut].rstrip()))
            skipped = len(sentence[:cut]) + (len(sentence[cut:])
                                             - len(sentence[cut:].lstrip()))
            sentence = sentence[cut:].lstrip()
            s_off += skipped
        if not sentence:
            continue
        if cur and cur_len + 1 + len(sentence) > maximum:
            pieces.append((offset + cur_start, " ".join(cur)))
            cur, cur_len, cur_start = [], 0, None
        if not cur:
            cur_start = s_off
        cur.append(sentence)
        cur_len += len(sentence) + (1 if len(cur) > 1 else 0)
    if cur:
        pieces.append((offset + cur_start, " ".join(cur)))
    return pieces


def make_chunks(text: str, *, target: int = CHUNK_TARGET_CHARS,
                maximum: int = CHUNK_MAX_CHARS) -> list[Chunk]:
    """Paragraph-aligned chunks of about *target* characters, none longer than
    *maximum*, in document order."""
    chunks: list[Chunk] = []
    cur: list[str] = []
    cur_len = 0
    cur_start = 0

    def flush() -> None:
        nonlocal cur, cur_len
        if cur:
            chunks.append(Chunk(offset=cur_start, text="\n".join(cur),
                                index=len(chunks)))
            cur, cur_len = [], 0

    for offset, para in split_paragraphs(text):
        pieces = ([(offset, para)] if len(para) <= maximum
                  else _split_long(offset, para, maximum))
        for p_off, piece in pieces:
            if cur and cur_len + 1 + len(piece) > target:
                flush()
            if not cur:
                cur_start = p_off
            cur.append(piece)
            cur_len += len(piece) + (1 if len(cur) > 1 else 0)
    flush()
    return chunks


def query_terms(query: str) -> list[str]:
    """Distinct lower-cased word tokens of *query* of at least two characters,
    stopwords removed; when nothing survives the stopword filter, every token
    of at least two characters."""
    tokens = [t for t in _WORD_RE.findall((query or "").lower())
              if len(t) >= _MIN_TERM_LEN]
    kept: list[str] = []
    for t in tokens:
        if t not in _STOPWORDS and t not in kept:
            kept.append(t)
    if kept:
        return kept
    for t in tokens:
        if t not in kept:
            kept.append(t)
    return kept


def _term_matches(term: str, token: str) -> bool:
    if token == term:
        return True
    if len(term) >= _PREFIX_MATCH_MIN_LEN and token.startswith(term):
        return True
    return len(token) >= _PREFIX_MATCH_MIN_LEN and term.startswith(token)


def score_text(text: str, terms: Sequence[str], phrase: str = "") -> float:
    """Relevance of *text* to *terms*: 10 x the share of terms present, plus
    0.1 per occurrence (saturating at 3 per term), plus 3 when *phrase* (the
    whitespace-normalised lower-cased query, at least 8 characters) occurs.
    0 when no term is present."""
    if not terms:
        return 0.0
    lowered = text.lower()
    counts = Counter(_WORD_RE.findall(lowered))
    matched = 0
    tf = 0.0
    for term in terms:
        occurrences = sum(c for tok, c in counts.items()
                          if _term_matches(term, tok))
        if occurrences:
            matched += 1
            tf += min(occurrences, _TF_SATURATION) * _TF_WEIGHT
    if not matched:
        return 0.0
    score = _COVERAGE_WEIGHT * matched / len(terms) + tf
    if phrase and len(phrase) >= _PHRASE_MIN_LEN \
            and phrase in " ".join(lowered.split()):
        score += _PHRASE_BONUS
    return score


def rank_chunks(text: str, query: str) -> list[ScoredChunk]:
    """Chunks of *text* ordered by descending score, ties in document order."""
    terms = query_terms(query)
    phrase = " ".join((query or "").lower().split())
    scored = [ScoredChunk(chunk=c, score=score_text(c.text, terms, phrase))
              for c in make_chunks(text)]
    scored.sort(key=lambda s: (-s.score, s.chunk.index))
    return scored


def select_evidence(
    sources: Sequence[tuple[str, str, str]],
    query: str,
    *,
    budget: int = EVIDENCE_BUDGET_CHARS,
    per_source: int = PER_SOURCE_CAP_CHARS,
) -> list[EvidenceChunk]:
    """Pick evidence for *query* from *sources*, each ``(source_id, kind,
    text)`` in source order with *kind* ``page`` (extracted page text) or
    ``snippet`` (the provider snippet, taken whole). Each source id may appear
    once (``ValueError`` otherwise). Returns chunks grouped by source in the
    given order and by offset within a source. A snippet longer than
    *per_source* is cut at *per_source*."""
    remaining = budget
    used: dict[str, int] = defaultdict(int)
    order = [sid for sid, _, _ in sources]
    if len(set(order)) != len(order):
        raise ValueError("select_evidence: a source id appears more than once")
    picked: dict[str, list[ScoredChunk]] = defaultdict(list)
    snippets: dict[str, str] = {}

    def take(sid: str, text_len: int) -> bool:
        nonlocal remaining
        if text_len <= 0 or text_len > remaining \
                or used[sid] + text_len > per_source:
            return False
        used[sid] += text_len
        remaining -= text_len
        return True

    for sid, kind, text in sources:
        if kind != CHUNK_SNIPPET:
            continue
        snippet = " ".join((text or "").split())[:per_source]
        if snippet and take(sid, len(snippet)):
            snippets[sid] = snippet

    ranked: dict[str, list[ScoredChunk]] = {}
    for sid, kind, text in sources:
        if kind == CHUNK_PAGE:
            ranked[sid] = rank_chunks(text or "", query)

    for sid in order:
        for sc in ranked.get(sid, ()):
            if take(sid, len(sc.chunk.text)):
                picked[sid].append(sc)
                break

    pool: list[tuple[ScoredChunk, int]] = []
    for position, sid in enumerate(order):
        chosen = picked.get(sid, [])
        for sc in ranked.get(sid, ()):
            if sc.score > 0 and sc not in chosen:
                pool.append((sc, position))
    pool.sort(key=lambda item: (-item[0].score, item[1],
                                item[0].chunk.index))
    for sc, position in pool:
        sid = order[position]
        if take(sid, len(sc.chunk.text)):
            picked[sid].append(sc)

    out: list[EvidenceChunk] = []
    for sid in order:
        if sid in snippets:
            out.append(EvidenceChunk(source_id=sid, text=snippets[sid],
                                     score=0.0, offset=0, kind=CHUNK_SNIPPET))
        for sc in sorted(picked.get(sid, []), key=lambda s: s.chunk.offset):
            out.append(EvidenceChunk(source_id=sid, text=sc.chunk.text,
                                     score=sc.score, offset=sc.chunk.offset,
                                     kind=CHUNK_PAGE))
    return out
