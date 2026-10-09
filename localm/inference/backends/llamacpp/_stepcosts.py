# SPDX-License-Identifier: AGPL-3.0-or-later
"""Step costs of speculative decoding and the draft length they pick.

``StepCosts`` holds the decode times measured once at load. ``expected_tokens``
is what one verified step yields at a per-draft acceptance p, and
``best_length`` the draft length with the most expected tokens per second
under a cost function. ``CountedSource`` in ``_drafting`` corrects the measured
costs with the step times seen while generating and picks each step's length.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

# Target batch sizes the load-time measurement times; others are interpolated.
VERIFY_MEASURE_SIZES = (2, 3, 5, 9, 17)
# A one-token target decode at most this slow is measured with 3 warm-up and 5
# timed decodes per figure, at most MEASURE_MEDIUM_DECODE_S with 1 and 3.
MEASURE_FAST_DECODE_S = 0.02
MEASURE_MEDIUM_DECODE_S = 0.05
# Prior acceptance evidence: a draft accepted with probability 0.6, worth two steps.
ACCEPTANCE_PRIOR_ACCEPTED = 1.2
ACCEPTANCE_PRIOR_REJECTED = 0.8
# Weight each verification keeps of the acceptance evidence before it.
ACCEPTANCE_DECAY = 0.9
# A draft length is chosen only when it beats a plain step by this fraction.
DRAFT_GAIN_MARGIN = 0.05
# Steps decided not to draft before one drafts anyway to measure acceptance;
# doubled after each probe with a rejected draft, up to ACCEPTANCE_PROBE_MAX_EVERY.
ACCEPTANCE_PROBE_EVERY = 32
ACCEPTANCE_PROBE_MAX_EVERY = 256
# Acceptance a probing step is sized for.
ACCEPTANCE_PROBE_P = 0.9
# Weight a newly observed step time gets in the running figure for its length.
OBSERVED_COST_WEIGHT = 0.2
# A newly observed step time is clipped to within this factor of that figure.
OBSERVED_COST_CLIP = 3.0
# Acceptance at which a draft model must beat plain decoding to be kept at load.
DRAFT_GATE_ACCEPTANCE = 0.85
# Tokens a reply of unknown length is assumed to have left.
UNBOUNDED_REPLY_TOKENS = 256


def expected_tokens(p: float, k: int) -> float:
    """Tokens one verified step makes available when it drafts *k* tokens and
    each is accepted with probability *p* independently: the accepted drafts
    plus the target's own token, ``1 + p + ... + p**k``."""
    if p >= 1.0:
        return k + 1.0
    return (1.0 - p ** (k + 1)) / (1.0 - p)


def best_length(p: float, k_max: int, step_cost: Callable[[int], float]) -> int:
    """The draft length in 0..*k_max* with the most expected tokens per second
    at acceptance *p*, where ``step_cost(k)`` is the seconds of a step drafting
    k (k 0 a plain step); 0 unless one beats a plain step by
    ``DRAFT_GAIN_MARGIN``."""
    best_k, best_rate = 0, (1.0 + DRAFT_GAIN_MARGIN) / step_cost(0)
    for k in range(1, max(0, k_max) + 1):
        rate = expected_tokens(p, k) / step_cost(k)
        if rate > best_rate:
            best_k, best_rate = k, rate
    return best_k


def measure_plan(probe_s: float, top: int) -> Tuple[int, int, List[int]]:
    """``(warm, reps, sizes)`` for measuring a target whose one-token decode
    took *probe_s* seconds with verification batches up to *top* tokens: every
    ``VERIFY_MEASURE_SIZES`` size below *top* plus *top*, with 3 warm-up and 5
    timed decodes when *probe_s* is at most ``MEASURE_FAST_DECODE_S`` and 1
    and 3 when it is at most ``MEASURE_MEDIUM_DECODE_S``; otherwise 1 and 1 of
    sizes 2 and *top* only. No sizes when *top* is below 2."""
    if top < 2:
        return 1, 1, []
    sizes = sorted({n for n in VERIFY_MEASURE_SIZES if n < top} | {top})
    if probe_s <= MEASURE_FAST_DECODE_S:
        return 3, 5, sizes
    if probe_s <= MEASURE_MEDIUM_DECODE_S:
        return 1, 3, sizes
    return 1, 1, sorted({2, top})


@dataclass(frozen=True)
class StepCosts:
    """Seconds one decode takes on this load, measured once at load:
    ``target`` one target token, ``verify`` a target batch of n tokens keyed
    by n (n >= 2), ``draft`` one draft token and ``draft_prefill`` one token of
    a batched draft decode (both 0 for a source without a draft model)."""
    target: float
    verify: Dict[int, float]
    draft: float = 0.0
    draft_prefill: float = 0.0

    def verify_cost(self, n: int) -> float:
        """Seconds of a target batch of *n* tokens from the measured figures
        made non-decreasing in n (each at least the one for fewer tokens):
        the figure at a measured size, linear between measured sizes, and past
        the largest the last slope."""
        if n <= 1:
            return self.target
        points = []
        for size, seconds in sorted({1: self.target, **self.verify}.items()):
            points.append((size, max(seconds, points[-1][1]) if points else seconds))
        if len(points) == 1:
            return self.target * n
        for (n0, t0), (n1, t1) in zip(points, points[1:]):
            if n <= n1:
                return t0 + (t1 - t0) * (n - n0) / (n1 - n0)
        (n0, t0), (n1, t1) = points[-2], points[-1]
        return t1 + (t1 - t0) * (n - n1) / (n1 - n0)

    def step_cost(self, k: int) -> float:
        """Seconds of a step drafting *k* tokens: k draft decodes and a target
        batch of k + 1. k 0 is a plain step."""
        if k <= 0:
            return self.target
        return k * self.draft + self.verify_cost(k + 1)

    def best_length(self, p: float, k_max: int) -> int:
        """``best_length`` under these measured costs."""
        return best_length(p, k_max, self.step_cost)

    def can_pay(self, k_max: int, p: float = 1.0) -> bool:
        """Whether any draft length up to *k_max* beats a plain step at
        acceptance *p*."""
        return self.best_length(p, k_max) > 0

    def report(self) -> dict:
        """The figures in milliseconds."""
        return {"target_ms": round(self.target * 1000, 3),
                "verify_ms": {n: round(t * 1000, 3) for n, t in sorted(self.verify.items())},
                "draft_ms": round(self.draft * 1000, 3),
                "draft_prefill_ms": round(self.draft_prefill * 1000, 3)}
