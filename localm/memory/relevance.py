# SPDX-License-Identifier: AGPL-3.0-or-later
"""Lexical relevance gate shared by the chat memory store and the coder episode store.

A query is lexically relevant to a record when they share enough DISTINCTIVE
content words: at least ``LEX_MIN_OVERLAP`` of them, or every distinctive word of
a shorter query. In a corpus of at least ``GENERIC_MIN_CORPUS`` records, a word
present in more than ``GENERIC_DF`` of them is generic for that corpus and counts
for nothing, so a store full of local-model talk is not matched by "local" or
"model".
"""

from __future__ import annotations

from typing import Iterable

LEX_MIN_OVERLAP = 2
GENERIC_DF = 0.30
GENERIC_MIN_CORPUS = 8


def generic_tokens(corpus: list[set]) -> frozenset:
    """Tokens present in more than GENERIC_DF of *corpus* (a list of per-record
    token sets). Empty for a corpus smaller than GENERIC_MIN_CORPUS, where document
    frequency is too noisy to mean anything."""
    n = len(corpus)
    if n < GENERIC_MIN_CORPUS:
        return frozenset()
    counts: dict = {}
    for toks in corpus:
        for t in toks:
            counts[t] = counts.get(t, 0) + 1
    return frozenset(t for t, c in counts.items() if c / n > GENERIC_DF)


def lexical_match(query_tokens: Iterable[str], record_tokens: Iterable[str],
                  generic: frozenset = frozenset()) -> bool:
    """True when the record shares enough distinctive words with the query.

    A query made only of generic words, or sharing none, never matches."""
    q = set(query_tokens) - generic
    if not q:
        return False
    shared = q & set(record_tokens)
    return len(shared) >= max(1, min(LEX_MIN_OVERLAP, len(q)))
