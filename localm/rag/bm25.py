"""Minimal BM25 ranking — pure stdlib, fast enough for home-scale corpora.

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

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens. Deliberately simple and deterministic."""
    return _TOKEN_RE.findall(text.lower())


class BM25:
    """Index a list of texts once; score queries against all of them."""

    def __init__(self, texts: list[str]) -> None:
        self._tfs: list[Counter] = []
        self._lengths: list[int] = []
        df: Counter = Counter()
        for text in texts:
            tokens = tokenize(text)
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
        terms = tokenize(query)
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
