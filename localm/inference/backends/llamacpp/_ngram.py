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
from ._stepcosts import DRAFT_GAIN_MARGIN

NGRAM_N_MIN = 3
NGRAM_N_MAX = 5
NGRAM_DRAFT_TOKENS_DEFAULT = 8
NGRAM_DRAFT_TOKENS_MAX = 16
# Draft tokens per step on a model with recurrent layers, where every draft
# token is a recurrent-state snapshot the context keeps in VRAM.
NGRAM_RECURRENT_DRAFT_TOKENS_MAX = 4
# Source positions a match may lie past the last candidate token checked right
# and still pick up the same copy ("resume").
NGRAM_RESUME_GAP = 32
# Prior chance that a candidate token is wrong, worth NGRAM_PRIOR_WEIGHT checks:
# for a match before the reply's first token, and for one in the reply.
NGRAM_PRIOR_MISS_CONTEXT = 0.1
NGRAM_PRIOR_MISS_REPLY = 0.4
NGRAM_PRIOR_WEIGHT = 2.0
# Weight a kind's counts keep at each new candidate of that kind: over the
# model's life, and within the reply.
NGRAM_LIFE_DECAY = 0.98
NGRAM_REPLY_DECAY = 0.8
# Weight of the model's-life estimate as the prior of the reply's.
NGRAM_REPLY_PRIOR_WEIGHT = 3.0

PLACE_CONTINUE = "continue"
PLACE_RESUME = "resume"
PLACE_START = "start"
SOURCE_CONTEXT = "context"
SOURCE_REPLY = "reply"


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
        return self.match(token, n_draft)[0]

    def match(self, token: int, n_draft: int) -> Tuple[List[int], int]:
        """``(lookup, end)``: the drafts and the position of the occurrence's
        last token, the drafts being what followed it at end + 1 onward;
        ``([], -1)`` when nothing matched or *n_draft* is 0 or less."""
        if n_draft <= 0:
            return [], -1
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
                return out, q
        return [], -1


class _Counts:
    """Decayed counts per draft position j (1..size) of candidate tokens
    checked there, and of those found wrong."""

    __slots__ = ("checked", "wrong")

    def __init__(self, size: int) -> None:
        self.checked = [0.0] * (size + 1)
        self.wrong = [0.0] * (size + 1)

    def scale(self, factor: float) -> None:
        self.checked = [x * factor for x in self.checked]
        self.wrong = [x * factor for x in self.wrong]

    def add(self, j: int, wrong: bool) -> None:
        self.checked[j] += 1.0
        if wrong:
            self.wrong[j] += 1.0


class CandidateRuns:
    """How far n-gram candidates of each kind run before a wrong token.

    A kind is ``(place, source)``. Per kind and draft position j (1..size)
    there are counts over the model's life and counts within the reply. The
    chance that token j is wrong, the tokens before it being right, is the
    reply's counts at j with the model's-life estimate as their prior, worth
    ``NGRAM_REPLY_PRIOR_WEIGHT`` checks; the model's-life estimate is that
    kind's counts with ``NGRAM_PRIOR_MISS_CONTEXT`` (source "context") or
    ``NGRAM_PRIOR_MISS_REPLY`` as their prior, worth ``NGRAM_PRIOR_WEIGHT``.
    Each new candidate of a kind keeps ``NGRAM_LIFE_DECAY`` and
    ``NGRAM_REPLY_DECAY`` of that kind's counts; ``new_reply`` clears the
    reply's counts.
    """

    def __init__(self, size: int) -> None:
        self.size = size
        self._kinds: Dict[Tuple[str, str], Tuple[_Counts, _Counts]] = {}

    def _counts(self, kind: Tuple[str, str]) -> Tuple[_Counts, _Counts]:
        counts = self._kinds.get(kind)
        if counts is None:
            counts = self._kinds[kind] = (_Counts(self.size), _Counts(self.size))
        return counts

    def miss(self, kind: Tuple[str, str], j: int) -> float:
        """Estimated chance that token *j* of a candidate of *kind* is wrong
        when the tokens before it are right."""
        life, reply = self._counts(kind)
        prior = NGRAM_PRIOR_MISS_CONTEXT if kind[1] == SOURCE_CONTEXT else NGRAM_PRIOR_MISS_REPLY
        life_miss = ((life.wrong[j] + prior * NGRAM_PRIOR_WEIGHT)
                     / (life.checked[j] + NGRAM_PRIOR_WEIGHT))
        return ((reply.wrong[j] + life_miss * NGRAM_REPLY_PRIOR_WEIGHT)
                / (reply.checked[j] + NGRAM_REPLY_PRIOR_WEIGHT))

    def expected_tokens(self, kind: Tuple[str, str], k: int) -> List[float]:
        """Element i for i in 0..*k*: the tokens a step drafting the first i
        tokens of a candidate of *kind* is expected to make available, the
        step's own token plus each draft's chance that it and every draft
        before it are right."""
        out = [1.0]
        right = 1.0
        for j in range(1, min(k, self.size) + 1):
            right *= 1.0 - self.miss(kind, j)
            out.append(out[-1] + right)
        return out

    def new_candidate(self, kind: Tuple[str, str]) -> None:
        """Age *kind*'s counts for a new candidate of that kind."""
        life, reply = self._counts(kind)
        life.scale(NGRAM_LIFE_DECAY)
        reply.scale(NGRAM_REPLY_DECAY)

    def check(self, kind: Tuple[str, str], j: int, wrong: bool) -> None:
        """Count token *j* (1..size) of a candidate of *kind* as checked,
        and as wrong when *wrong*."""
        for counts in self._counts(kind):
            counts.add(j, wrong)

    def new_reply(self) -> None:
        for _life, reply in self._kinds.values():
            reply.scale(0.0)

    def report(self) -> Dict[str, float]:
        """Per kind seen, keyed "place/source": the tokens a step drafting a
        full-length candidate of it is expected to make available."""
        return {"%s/%s" % kind: round(self.expected_tokens(kind, self.size)[-1], 2)
                for kind in sorted(self._kinds)}


class NgramSource(CountedSource):
    """Drafts by prompt lookup over the tokens in the main cache.

    The index follows ``llm._cached_tokens``: begin_call keeps the prefix it
    shares with them and indexes the rest, and each proposal first indexes the
    tokens appended since. Model-level status is ``status``; the reply's figures
    are the counters below, reset by begin_call.

    With measured ``costs`` the draft length is chosen in ``propose`` once the
    step's candidate (the lookup up to ``draft_max``) is known: the length with
    the most expected tokens per second under ``step_cost``, from ``runs``,
    which beats a plain step by ``DRAFT_GAIN_MARGIN``, else none. Every
    candidate a step looks up stays open, drafted or not, and each later
    proposal checks its tokens against the ones the reply went on to hold,
    counting each in ``runs`` under the candidate's kind, until one is wrong or
    all are checked. A candidate's place is "continue" when its match ends on
    the source position of the last candidate token checked right in this
    reply, "resume" when it ends 1 to ``NGRAM_RESUME_GAP`` positions past it,
    otherwise "start"; its source is "context" when the match ends before the
    reply's first token, otherwise "reply".
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
        self.runs = CandidateRuns(self.draft_max)
        # Open candidates: [position, tokens, tokens checked right, kind, match end].
        self._open: List[list] = []
        self._last_right: Optional[int] = None
        self._reply_start = 0

    @property
    def _llm(self):
        return self._llm_ref()

    def reset_call(self, skipped: str = "") -> None:
        super().reset_call(skipped)
        self._open = []
        self._last_right = None
        self.runs.new_reply()

    def begin_call(self) -> bool:
        self.reset_call()
        self._drafting = self.usable
        cached = self._llm._cached_tokens
        if self._drafting:
            self.index.sync(cached)
        self._reply_start = len(cached)
        return self._drafting

    def ready(self, pos: int) -> bool:
        return True

    def budget(self, pos: int, tokens_left: Optional[int]) -> int:
        """Drafts the step at *pos* may propose: ``cap_drafts``. With
        measured ``costs`` ``propose`` drafts fewer when fewer pay."""
        return self.cap_drafts(pos, tokens_left)

    def on_verify(self, drafted: int, accepted: int) -> None:
        """Count the verification; the drafted tokens are checked with the
        rest of their candidate."""
        self.count_verify(drafted, accepted)

    def length_report(self) -> dict:
        """``runs``: ``CandidateRuns.report``."""
        return {"runs": self.runs.report()}

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
        if self.costs is None:
            return self._candidate(token, n_max)[0]
        self._check_open(cached, token)
        drafts, end = self._candidate(token, self.draft_max)
        if not drafts:
            return []
        kind = self._kind(end)
        self.runs.new_candidate(kind)
        self._open.append([pos, drafts, 0, kind, end])
        k = self._length(kind, min(len(drafts), n_max))
        if k == 0:
            self.held_steps += 1
        return drafts[:k]

    def _candidate(self, token: int, n: int) -> Tuple[List[int], int]:
        """``NgramIndex.match`` cut before the first end-of-generation token."""
        drafts, end = self.index.match(token, n)
        is_eog = self._is_eog or self._llm._tokenizer.is_eog
        for i, d in enumerate(drafts):
            if is_eog(d):
                return drafts[:i], end
        return drafts, end

    def _check_open(self, cached: List[int], token: int) -> None:
        """Check the open candidates against ``cached`` followed by *token*,
        closing each at its first wrong token or when all are checked."""
        held = len(cached) + 1
        still = []
        for item in self._open:
            pos, drafts, done, kind, end = item
            right = True
            while done < len(drafts) and pos + 1 + done < held:
                at = pos + 1 + done
                right = (cached[at] if at < len(cached) else token) == drafts[done]
                self.runs.check(kind, done + 1, not right)
                if not right:
                    break
                done += 1
                self._last_right = end + done
            if right and done < len(drafts):
                item[2] = done
                still.append(item)
        self._open = still

    def _kind(self, end: int) -> Tuple[str, str]:
        """The kind of a candidate whose match ends at *end*."""
        gap = None if self._last_right is None else end - self._last_right
        if gap == 0:
            place = PLACE_CONTINUE
        elif gap is not None and 0 < gap <= NGRAM_RESUME_GAP:
            place = PLACE_RESUME
        else:
            place = PLACE_START
        return place, SOURCE_CONTEXT if end < self._reply_start else SOURCE_REPLY

    def _length(self, kind: Tuple[str, str], n: int) -> int:
        """The draft length in 0..*n* with the most expected tokens per second
        for a candidate of *kind*; 0 unless one beats a plain step by
        ``DRAFT_GAIN_MARGIN``."""
        expected = self.runs.expected_tokens(kind, n)
        best_k, best_rate = 0, (1.0 + DRAFT_GAIN_MARGIN) / self.step_cost(0)
        for k in range(1, len(expected)):
            rate = expected[k] / self.step_cost(k)
            if rate > best_rate:
                best_k, best_rate = k, rate
        return best_k
