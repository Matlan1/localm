# SPDX-License-Identifier: AGPL-3.0-or-later
"""Prompt-lookup (n-gram) drafting: no second model, no extra VRAM.

The draft for the token just sampled is what followed the most recent earlier
occurrence of the longest trailing n-gram (n from n_max down to n_min) of the
tokens in the cache plus that token.
"""
from __future__ import annotations

import weakref
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from ._drafting import SPEC_NGRAM, CountedSource

NGRAM_N_MIN = 3
NGRAM_N_MAX = 5
NGRAM_DRAFT_TOKENS_DEFAULT = 8
NGRAM_DRAFT_TOKENS_MAX = 16
# Draft tokens per step on a model with recurrent layers, where every draft
# token is a recurrent-state snapshot the context keeps in VRAM.
NGRAM_RECURRENT_DRAFT_TOKENS_MAX = 4


def ngram_draft_cap(draft_tokens: Optional[int], recurrent: bool,
                    default: int = NGRAM_DRAFT_TOKENS_DEFAULT) -> int:
    """Draft tokens one step of an ngram or draft source may propose:
    *draft_tokens* (None for *default*) clamped to 1..NGRAM_DRAFT_TOKENS_MAX,
    and to NGRAM_RECURRENT_DRAFT_TOKENS_MAX on a model with recurrent layers."""
    n = default if draft_tokens is None else int(draft_tokens)
    n = max(1, min(n, NGRAM_DRAFT_TOKENS_MAX))
    if recurrent:
        n = min(n, NGRAM_RECURRENT_DRAFT_TOKENS_MAX)
    return n


def ngram_rs_seq(default_n_rs_seq, draft_max: int) -> int:
    """The ``n_rs_seq`` a context drafting *draft_max* n-gram tokens is created
    with: the runtime's default, but at least 2 and at least *draft_max*."""
    return max(int(default_n_rs_seq or 0), 2, int(draft_max))


class NgramIndex:
    """The tokens of one sequence and, for every n-gram of them with n in
    n_min..n_max, the end positions of its occurrences in ascending order; the
    last one is the most recent.

    ``lookup`` treats ``tokens + [token]`` as the history; the n-grams ending at
    that trailing token are not in the index, so the trailing n-gram never
    matches itself.
    """

    def __init__(self, n_min: int = NGRAM_N_MIN, n_max: int = NGRAM_N_MAX) -> None:
        if not 1 <= n_min <= n_max:
            raise ValueError("need 1 <= n_min <= n_max, got %d, %d" % (n_min, n_max))
        self.n_min = n_min
        self.n_max = n_max
        self.tokens: List[int] = []
        self._ends: Dict[Tuple[int, ...], List[int]] = {}

    def __len__(self) -> int:
        return len(self.tokens)

    def extend(self, tokens: Sequence[int]) -> None:
        """Append *tokens* and index every n-gram ending at one of them."""
        toks = self.tokens
        ends = self._ends
        for t in tokens:
            toks.append(int(t))
            e = len(toks) - 1
            for n in range(self.n_min, min(self.n_max, e + 1) + 1):
                key = tuple(toks[e - n + 1:e + 1])
                at = ends.get(key)
                if at is None:
                    ends[key] = [e]
                else:
                    at.append(e)

    def truncate(self, length: int) -> None:
        """Keep the first *length* tokens and drop every n-gram occurrence
        ending at a removed position; the cost is proportional to the tokens
        removed."""
        length = max(0, length)
        toks = self.tokens
        ends = self._ends
        for e in range(len(toks) - 1, length - 1, -1):
            for n in range(self.n_min, min(self.n_max, e + 1) + 1):
                key = tuple(toks[e - n + 1:e + 1])
                at = ends.get(key)
                if at and at[-1] == e:
                    at.pop()
                    if not at:
                        del ends[key]
        del toks[length:]

    def sync(self, tokens: Sequence[int]) -> None:
        """Make the index hold exactly *tokens*, keeping the shared prefix."""
        mine = self.tokens
        n = min(len(mine), len(tokens))
        i = 0
        while i < n and mine[i] == tokens[i]:
            i += 1
        self.truncate(i)
        if len(self.tokens) < len(tokens):
            self.extend(tokens[len(self.tokens):])

    def lookup(self, token: int, n_draft: int) -> List[int]:
        """Up to *n_draft* tokens that followed the most recent earlier
        occurrence of the longest trailing n-gram of ``tokens + [token]``;
        [] when no n-gram of n_min or more tokens occurred before."""
        if n_draft <= 0:
            return []
        toks = self.tokens
        end = len(toks)                    # position of *token*
        for n in range(min(self.n_max, end + 1), self.n_min - 1, -1):
            key = tuple(toks[end - n + 1:end]) + (int(token),)
            at = self._ends.get(key)
            if at:
                q = at[-1]
                stop = min(q + 1 + n_draft, end + 1)
                out = toks[q + 1:min(stop, end)]
                if stop > end:
                    out.append(int(token))
                return out
        return []


class NgramSource(CountedSource):
    """Drafts by prompt lookup over the tokens in the main cache.

    The index follows ``llm._cached_tokens``: begin_call keeps the prefix it
    shares with them and indexes the rest, and each proposal first indexes the
    tokens appended since. Model-level status is ``status``; the reply's figures
    are the counters below, reset by begin_call.
    """

    name = SPEC_NGRAM
    label = "n-gram drafting"
    free_miss = True

    def __init__(self, llm, draft_max: int = NGRAM_DRAFT_TOKENS_DEFAULT,
                 is_eog: Optional[Callable[[int], bool]] = None) -> None:
        super().__init__(max(1, min(int(draft_max), NGRAM_DRAFT_TOKENS_MAX)))
        self._llm_ref = weakref.ref(llm)
        self.index = NgramIndex()
        self._is_eog = is_eog

    @property
    def _llm(self):
        return self._llm_ref()

    def begin_call(self) -> bool:
        self.reset_call()
        self._drafting = self.usable
        if self._drafting:
            self.index.sync(self._llm._cached_tokens)
        return self._drafting

    def ready(self, pos: int) -> bool:
        return True

    def budget(self, pos: int, tokens_left: Optional[int]) -> int:
        n = self.draft_max
        if tokens_left is not None:
            n = min(n, tokens_left)
        return max(0, min(n, self._llm._ctx_capacity - pos - 1))

    def propose(self, token: int, pos: int, n_max: int) -> List[int]:
        cached = self._llm._cached_tokens
        if len(self.index) != len(cached):
            if len(self.index) < len(cached):
                self.index.extend(cached[len(self.index):])
            else:
                self.index.sync(cached)
        if len(self.index) != pos:
            self.stop_this_call("draft-out-of-step")
            return []
        drafts = self.index.lookup(token, n_max)
        is_eog = self._is_eog or self._llm._tokenizer.is_eog
        for i, d in enumerate(drafts):
            if is_eog(d):
                return drafts[:i]
        return drafts
