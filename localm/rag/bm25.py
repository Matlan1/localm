# SPDX-License-Identifier: AGPL-3.0-or-later
"""Minimal BM25 ranking - pure stdlib, fast enough for home-scale corpora.

A few thousand chunks score in single-digit milliseconds; this is the
always-available retrieval baseline (the ctypes GGUF binding has no
embedding support, so vectors can never be assumed).
"""

from __future__ import annotations

import math
import re
from collections import Counter

_K1 = 1.5
_B = 0.75

# Unicode-aware word runs: keeps accented latin, Cyrillic, Greek, Arabic,
# CJK, etc. (Python 3's \w is unicode by default; re.UNICODE is explicit).
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

# Scripts that are written without spaces between words (CJK ideographs,
# Hiragana, Katakana, Hangul). For these we have no reliable word boundary,
# so we fall back to per-character tokens: that lets a query character match
# a corpus character, which is the standard cheap approach for space-free
# scripts. Latin / Cyrillic / Arabic / etc. word runs are kept whole.
_CJK_RANGES = (
    (0x3040, 0x30FF),    # Hiragana + Katakana
    (0x3400, 0x4DBF),    # CJK Unified Ideographs Extension A
    (0x4E00, 0x9FFF),    # CJK Unified Ideographs
    (0xF900, 0xFAFF),    # CJK Compatibility Ideographs
    (0xAC00, 0xD7AF),    # Hangul syllables
    (0x20000, 0x2A6DF),  # CJK Unified Ideographs Extension B
)


def _is_cjk(ch: str) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _CJK_RANGES)


# Common English function words. OPT-IN only (see tokenize/BM25 stop_words): a
# caller filters these from the LEXICAL index+query so a stopword can never be
# the SOLE basis of a match. On a small or topically-narrow corpus a stopword
# can land in few enough docs to earn a high IDF and, once max-normalised into
# the hybrid blend, outrank the true semantic match. Kept to genuine function
# words; non-English tokens are absent from the set and pass through untouched.
#
# NOT the default: localm.memory.store imports tokenize() and depends on
# stopwords SURVIVING (its self-reference check reads "i"/"me"/"my" from the raw
# token stream), and memory/coder keep their own relevance-gate stopword
# copies.
ENGLISH_STOP_WORDS = frozenset(
    "a an and are as at be been but by can could did do does done for from had has "
    "have he her him his i if in into is it its me my no not of on only or our over "
    "own same she should so some such than that the their them then there these they "
    "this to too under up us very was we were what when where which who will with "
    "would you your".split()
)


def tokenize(text: str, stop_words: "frozenset[str] | None" = None) -> list[str]:
    """Lowercase unicode word tokens.

    Word runs are split on unicode word boundaries (so accented latin,
    Cyrillic, Arabic, etc. survive instead of collapsing to nothing under
    the old ASCII-only rule). Characters from space-free scripts (CJK,
    Hangul) are emitted one token per character so a query can still match.

    When *stop_words* is given, tokens in that set are dropped (case is already
    folded above). Default None leaves every token in place; callers that need
    stopwords (e.g. the memory self-reference check) rely on that.
    """
    tokens: list[str] = []
    for run in _TOKEN_RE.findall(text.lower()):
        buf: list[str] = []
        for ch in run:
            if _is_cjk(ch):
                if buf:
                    tokens.append("".join(buf))
                    buf = []
                tokens.append(ch)
            else:
                buf.append(ch)
        if buf:
            tokens.append("".join(buf))
    if stop_words:
        tokens = [t for t in tokens if t not in stop_words]
    return tokens


class BM25:
    """Index a list of texts once; score queries against all of them."""

    def __init__(self, texts: list[str],
                 stop_words: "frozenset[str] | None" = None) -> None:
        # Applied to BOTH sides: indexed here, queries in scores(). Filtering at
        # index time also keeps stopwords out of the IDF table and document
        # lengths, so a stopword contributes no lexical signal at all.
        self._stop_words = stop_words
        self._tfs: list[Counter] = []
        self._lengths: list[int] = []
        df: Counter = Counter()
        for text in texts:
            tokens = tokenize(text, stop_words)
            tf = Counter(tokens)
            self._tfs.append(tf)
            self._lengths.append(len(tokens))
            df.update(tf.keys())
        self._n = len(texts)
        self._avg_len = (sum(self._lengths) / self._n) if self._n else 0.0
        # Standard BM25 idf with the +1 inside the log (never negative)
        self._idf = {
            term: math.log(1 + (self._n - count + 0.5) / (count + 0.5))
            for term, count in df.items()
        }

    def scores(self, query: str) -> list[float]:
        """BM25 score of *query* against every indexed text, in index order."""
        out = [0.0] * self._n
        if not self._n or not self._avg_len:
            return out
        terms = tokenize(query, self._stop_words)
        for term in terms:
            idf = self._idf.get(term)
            if idf is None:
                continue
            for i, tf in enumerate(self._tfs):
                f = tf.get(term)
                if not f:
                    continue
                denom = f + _K1 * (1 - _B + _B * self._lengths[i] / self._avg_len)
                out[i] += idf * (f * (_K1 + 1)) / denom
        return out
