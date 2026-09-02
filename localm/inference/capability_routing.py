# SPDX-License-Identifier: AGPL-3.0-or-later
"""Pick the model that answers a request, from what the request needs.

Pure planning: nothing here loads, evicts or mutates anything. It takes what a
request needs plus a registry snapshot and returns a ``RoutingDecision`` the
caller applies, so the decision is testable on its own and one function owns the
rule that a pinned model is never changed.

THE BINDING CONSTRAINT: a model the user named EXPLICITLY is never swapped. It
is enforced twice, deliberately, and the two are not redundant:

1. Structurally, at the only call site that can change which model loads.
   ``get_engine`` already computes pinned-ness to decide whether to resolve an
   unnamed request, and routing is applied INSIDE that unnamed branch, so a
   pinned request cannot reach it.
2. Here, via ``pinned``, so the gap can still be REPORTED for a pinned request
   (that is what produces the suggestion the user sees) without any path through
   this module being able to act on it.

A pinned request therefore still gets a decision describing what it lacks, and
``resolved`` on that decision is always the model the user asked for.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

from localm.model_manager import capabilities as caps

# Divisor for the tokenizer-free prompt estimate. Routing has to size a prompt
# BEFORE it knows which model will answer, and a tokenizer belongs to a model, so
# an exact count is not available at this point. Same chars/4 rule
# count_tokens_or_estimate falls back to.
_CHARS_PER_TOKEN_ESTIMATE = 4

# Headroom multiplier on the estimated prompt size when deciding whether a model
# is roomy enough. A reply needs room too, and the estimate above is rough in
# both directions, so a prompt is only treated as too big for a model when it
# exceeds the whole window including this margin.
_CONTEXT_HEADROOM = 1.25

# Below this estimated prompt size the context question is not asked at all.
# The smallest trained window among chat models in real use is 2048 tokens, so
# a prompt under it cannot overflow any of them, and asking anyway would cost
# a registry read plus a probe per candidate on every short request.
_CONTEXT_ROUTING_FLOOR_TOKENS = 2048


@dataclass(frozen=True)
class CapabilityNeeds:
    """What one request needs from whatever model answers it.

    *capabilities* are boolean capability names (see
    ``capabilities.BOOLEAN_CAPABILITIES``). *min_context* is an estimated token
    count the model's trained window must cover, or None when the request states
    no context requirement."""

    capabilities: Tuple[str, ...] = ()
    min_context: Optional[int] = None

    def is_empty(self) -> bool:
        return not self.capabilities and self.min_context is None


@dataclass(frozen=True)
class RoutingDecision:
    """What routing concluded, and enough of why to audit it afterwards.

    ``resolved`` is the model that will answer. ``routed`` says whether that
    differs from ``current``; when ``pinned`` it is always False, by
    construction.

    ``gaps`` maps each needed capability the CURRENT model does not confirm to
    its tri-state as measured (``False`` = confirmed absent, ``None`` = never
    inspected). Those are different facts and stay different here: a summary that
    flattened them would report a model nobody has looked at as one that cannot
    do the job.

    ``unmet`` names the needs no installed model could satisfy, which is the
    honest fallback the ADR asks for - routing that found nowhere better to go
    says so instead of silently doing nothing."""

    current: Optional[str]
    resolved: Optional[str]
    pinned: bool
    needs: CapabilityNeeds
    gaps: Dict[str, Optional[bool]] = field(default_factory=dict)
    unmet: Tuple[str, ...] = ()
    candidates: Tuple[str, ...] = ()

    @property
    def routed(self) -> bool:
        # Deliberately NOT gated on current being set. With no model resolved at
        # all, resolved names one and routed must agree, or the audit surface
        # reports a model that then does not answer.
        return self.resolved is not None and self.resolved != self.current

    @property
    def has_gap(self) -> bool:
        return bool(self.gaps)

    def describe(self) -> str:
        """One line naming what happened, for the audit log and the response
        header. Says which capability drove the choice, never just that a choice
        was made."""
        if not self.has_gap:
            return "no capability gap"
        parts = []
        for cap, state in sorted(self.gaps.items()):
            parts.append(f"{cap}=" + ("absent" if state is False else "unknown"))
        gap_text = ", ".join(parts)
        if self.routed:
            return f"routed {self.current} -> {self.resolved} ({gap_text})"
        if self.pinned:
            return f"kept pinned {self.current} ({gap_text})"
        if self.unmet:
            return (f"kept {self.current} ({gap_text}); "
                    f"no installed model provides {', '.join(self.unmet)}")
        return f"kept {self.current} ({gap_text})"


def context_need(messages: Sequence[dict]) -> Optional[int]:
    """The context window *messages* needs, or None when the prompt is too small
    for the question to matter.

    Returns the estimate plus ``_CONTEXT_HEADROOM`` so a model is only judged too
    small when the prompt exceeds its whole window with room for a reply.

    Below ``_CONTEXT_ROUTING_FLOOR_TOKENS`` this answers None rather than a small
    number: no chat model in real use has a window that small, so the comparison
    could never find a shortfall, and skipping it keeps a registry read and a
    per-candidate probe off every ordinary short request."""
    est = estimate_prompt_tokens(messages)
    if est < _CONTEXT_ROUTING_FLOOR_TOKENS:
        return None
    return int(est * _CONTEXT_HEADROOM)


def estimate_prompt_tokens(messages: Sequence[dict]) -> int:
    """Rough token size of *messages*, without a tokenizer.

    An ESTIMATE and treated as one: it decides only whether to prefer a roomier
    model, never whether to refuse a request, so being wrong costs a suboptimal
    model choice rather than a rejected prompt. Text parts of a structured
    content list are counted; an image part contributes nothing here, because its
    real cost depends on a projector this has not chosen yet."""
    total = 0
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            total += len(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    total += len(part["text"])
    return max(1, total // _CHARS_PER_TOKEN_ESTIMATE)


def _model_satisfies(name: str, needs: CapabilityNeeds, reg: dict,
                     dir_cache: dict) -> bool:
    """Whether *name* is CONFIRMED to meet every need.

    Positive membership only: an unknown capability does not qualify a model,
    because routing must not send a request somewhere nobody has inspected."""
    for cap in needs.capabilities:
        if caps.model_capability(name, cap, reg=reg, dir_cache=dir_cache) is not True:
            return False
    if needs.min_context is not None:
        ctx = caps.model_context_length(name, reg=reg)
        if ctx is None or ctx < needs.min_context:
            return False
    return True


def _current_gaps(name: Optional[str], needs: CapabilityNeeds, reg: dict,
                  dir_cache: dict) -> Dict[str, Optional[bool]]:
    """The needs *name* does not confirm, each with the tri-state as measured.

    A capability is a gap when it is not confirmed True, so an UNKNOWN counts.
    That is a preference for certainty, not a claim of absence, and the recorded
    ``None`` is what keeps the two distinguishable everywhere downstream: a
    caller must never render "this model cannot do X" from a None."""
    gaps: Dict[str, Optional[bool]] = {}
    if name is None:
        return {c: None for c in needs.capabilities}
    for cap in needs.capabilities:
        state = caps.model_capability(name, cap, reg=reg, dir_cache=dir_cache)
        if state is not True:
            gaps[cap] = state
    if needs.min_context is not None:
        # Context is the one need that gaps ONLY on a confirmed shortfall, never
        # on an unknown, and the asymmetry with the capabilities above is
        # deliberate. Those were REQUESTED, so an unconfirmed model does not
        # satisfy a stated requirement. This one is DERIVED from the prompt's
        # size, and an unknown window is the normal state of an entry nobody has
        # measured; treating it as a gap would manufacture a shortfall out of an
        # absence of evidence and re-route almost every request on a registry
        # that predates these fields.
        ctx = caps.model_context_length(name, reg=reg)
        if ctx is not None and ctx < needs.min_context:
            gaps[caps.CONTEXT_LENGTH] = False
    return gaps


def plan_route(current: Optional[str], needs: CapabilityNeeds, *,
               pinned: bool, resident: Sequence[str] = (),
               reg: Optional[dict] = None) -> RoutingDecision:
    """Decide which model should answer a request needing *needs*.

    *current* is the model that would answer if nothing changed. *pinned* says
    the user named it explicitly, which makes the choice fixed: the returned
    decision still describes the gap, and ``resolved`` still equals *current*.

    *resident* is the models already loaded, preferred among equally qualified
    candidates so routing does not evict a perfectly good model to load an
    equivalent one.

    Ranking among qualified candidates: already resident first, then the largest
    confirmed context window, then name, so the result is deterministic and a
    test can assert on it."""
    # Before the registry read, not after: a request that states no needs is the
    # common case and must not pay for a read it cannot use.
    if needs.is_empty():
        return RoutingDecision(current=current, resolved=current, pinned=pinned,
                               needs=needs)

    reg = caps._registry._mm.load_registry() if reg is None else reg
    if not isinstance(reg, dict):
        reg = {}
    dir_cache: dict = {}

    gaps = _current_gaps(current, needs, reg, dir_cache)
    if not gaps:
        return RoutingDecision(current=current, resolved=current, pinned=pinned,
                               needs=needs)

    if pinned:
        # The gap is reported so the caller can surface it. Nothing here may act
        # on it: resolved stays the model the user asked for.
        return RoutingDecision(current=current, resolved=current, pinned=True,
                               needs=needs, gaps=gaps)

    qualified = [n for n in reg
                 if n != current and _model_satisfies(n, needs, reg, dir_cache)]
    if not qualified:
        unmet = tuple(sorted(gaps))
        return RoutingDecision(current=current, resolved=current, pinned=False,
                               needs=needs, gaps=gaps, unmet=unmet)

    resident_set = set(resident)

    def rank(n: str):
        ctx = caps.model_context_length(n, reg=reg) or 0
        return (0 if n in resident_set else 1, -ctx, n)

    qualified.sort(key=rank)
    return RoutingDecision(current=current, resolved=qualified[0], pinned=False,
                           needs=needs, gaps=gaps,
                           candidates=tuple(qualified))
