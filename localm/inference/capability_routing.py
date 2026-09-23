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

import math
from dataclasses import dataclass, field, replace
from typing import Dict, Optional, Sequence, Tuple

from localm.model_manager import capabilities as caps
from localm.model_manager.registry import is_llm

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

# Capabilities a model must have to take the request at all: a model without
# vision refuses an image. A confirmed context shortfall is the other such need.
# Every other capability only changes how the request is answered.
_REQUIRED_TO_ANSWER = (caps.VISION,)


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

    ``unmet`` names the needs the model that answers still lacks: every gap
    when no installed model could take the request, or, when a model was chosen
    for what the request cannot be answered without (an image, the context
    window), the other needs it lacks because no installed model has them all."""

    current: Optional[str]
    resolved: Optional[str]
    pinned: bool
    needs: CapabilityNeeds
    gaps: Dict[str, Optional[bool]] = field(default_factory=dict)
    unmet: Tuple[str, ...] = ()
    candidates: Tuple[str, ...] = ()
    load_errors: Tuple[str, ...] = ()

    def without_route(self, load_errors: Sequence[str] = ()) -> "RoutingDecision":
        """This decision with the route withdrawn: *current* answers, every gap
        is unmet, and *load_errors* records why each candidate could not be
        used."""
        return replace(self, resolved=self.current, unmet=tuple(sorted(self.gaps)),
                       load_errors=tuple(load_errors))

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
            if self.unmet:
                return (f"routed {self.current} -> {self.resolved} ({gap_text}); "
                        f"no installed model also provides {', '.join(self.unmet)}")
            return f"routed {self.current} -> {self.resolved} ({gap_text})"
        if self.pinned:
            return f"kept pinned {self.current} ({gap_text})"
        if self.load_errors:
            return (f"kept {self.current} ({gap_text}); no capable model could "
                    f"be loaded: {'; '.join(self.load_errors)}")
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


def compaction_context_need(messages: Sequence[dict], trained: Optional[int],
                            ratio: float) -> Optional[int]:
    """The trained window a conversation needs so it is not compacted, for a
    client that compacts at *ratio* of the window: ``ceil(estimate / ratio)``
    once the conversation has reached *ratio* of *trained* (the answering
    model's trained window), else None. None too when *trained* is unknown."""
    if not trained or trained <= 0 or ratio <= 0:
        return None
    est = estimate_prompt_tokens(messages)
    if est < ratio * trained:
        return None
    return math.ceil(est / ratio)


def request_needs(messages: Sequence[dict], *, required: Sequence[str] = (),
                  min_context: Optional[int] = None) -> CapabilityNeeds:
    """What a chat request with *messages* needs: vision when a message carries
    an image, the context window its size implies, plus *required* capabilities
    and *min_context*. The same derivation the server applies to
    ``/v1/chat/completions``."""
    from localm.inference.backends.base import messages_contain_image
    wanted = list(required)
    if messages_contain_image(list(messages)) and caps.VISION not in wanted:
        wanted.append(caps.VISION)
    derived = context_need(messages) if messages else None
    ctx = [c for c in (derived, min_context) if isinstance(c, int) and c > 0]
    return CapabilityNeeds(capabilities=tuple(wanted),
                           min_context=max(ctx) if ctx else None)


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


def _is_routing_target(entry) -> bool:
    """Whether a registry *entry* is a chat model that can be loaded to answer:
    a text-generation LLM whose file is not recorded as missing."""
    return is_llm(entry) and not entry.get("missing")


def _model_satisfies(name: str, needs: CapabilityNeeds, reg: dict,
                     dir_cache: dict) -> bool:
    """Whether *name* is CONFIRMED to meet every need.

    Positive membership only: an unknown capability does not qualify a model,
    because routing must not send a request somewhere nobody has inspected."""
    if not _is_routing_target(reg.get(name)):
        return False
    for cap in needs.capabilities:
        if caps.model_capability(name, cap, reg=reg, dir_cache=dir_cache) is not True:
            return False
    if needs.min_context is not None:
        ctx = caps.model_context_length(name, reg=reg)
        if ctx is None or ctx < needs.min_context:
            return False
    return True


def _current_gaps(name: Optional[str], needs: CapabilityNeeds, reg: dict,
                  dir_cache: dict,
                  known: Optional[Dict[str, bool]] = None) -> Dict[str, Optional[bool]]:
    """The needs *name* does not confirm, each with the tri-state as measured.

    A capability is a gap when it is not confirmed True, so an UNKNOWN counts.
    That is a preference for certainty, not a claim of absence, and the recorded
    ``None`` is what keeps the two distinguishable everywhere downstream: a
    caller must never render "this model cannot do X" from a None."""
    gaps: Dict[str, Optional[bool]] = {}
    known = known or {}
    if name is None:
        return {c: None for c in needs.capabilities}
    for cap in needs.capabilities:
        if known.get(cap) is True:
            continue
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
               reg: Optional[dict] = None,
               current_known: Optional[Dict[str, bool]] = None) -> RoutingDecision:
    """Decide which model should answer a request needing *needs*.

    *current* is the model that would answer if nothing changed. *pinned* says
    the user named it explicitly, which makes the choice fixed: the returned
    decision still describes the gap, and ``resolved`` still equals *current*.

    *resident* is the models already loaded, preferred among equally qualified
    candidates so routing does not evict a perfectly good model to load an
    equivalent one.

    *current_known* maps a capability to True when the live engine behind
    *current* is confirmed to have it (for example a loaded model accepting
    images through a projector the registry does not record), so that need is
    not a gap.

    Only chat LLMs whose file is not recorded missing are candidates.

    When no model meets every need, a model is still chosen when the current
    one cannot take the request at all (an image it cannot read, a confirmed
    context shortfall): the candidates are the models that meet those needs, the
    ones meeting more of the rest first, and ``unmet`` names what the chosen one
    lacks.

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

    gaps = _current_gaps(current, needs, reg, dir_cache, current_known)
    if not gaps:
        return RoutingDecision(current=current, resolved=current, pinned=pinned,
                               needs=needs)

    if pinned:
        # The gap is reported so the caller can surface it. Nothing here may act
        # on it: resolved stays the model the user asked for.
        return RoutingDecision(current=current, resolved=current, pinned=True,
                               needs=needs, gaps=gaps)

    resident_set = set(resident)

    def rank(n: str):
        ctx = caps.model_context_length(n, reg=reg) or 0
        return (0 if n in resident_set else 1, -ctx, n)

    qualified = [n for n in reg
                 if n != current and _model_satisfies(n, needs, reg, dir_cache)]
    if qualified:
        qualified.sort(key=rank)
        return RoutingDecision(current=current, resolved=qualified[0], pinned=False,
                               needs=needs, gaps=gaps,
                               candidates=tuple(qualified))

    required = CapabilityNeeds(
        capabilities=tuple(c for c in needs.capabilities if c in _REQUIRED_TO_ANSWER),
        min_context=needs.min_context)
    optional = tuple(c for c in needs.capabilities if c not in _REQUIRED_TO_ANSWER)
    cannot_take = (any(c in gaps for c in required.capabilities)
                   or caps.CONTEXT_LENGTH in gaps)
    if cannot_take and optional:
        def has(n: str, cap: str) -> bool:
            return caps.model_capability(n, cap, reg=reg, dir_cache=dir_cache) is True

        partial = [n for n in reg
                   if n != current and _model_satisfies(n, required, reg, dir_cache)]
        if partial:
            partial.sort(key=lambda n: (-sum(has(n, c) for c in optional), *rank(n)))
            unmet = tuple(sorted(c for c in optional if not has(partial[0], c)))
            return RoutingDecision(current=current, resolved=partial[0], pinned=False,
                                   needs=needs, gaps=gaps, unmet=unmet,
                                   candidates=tuple(partial))

    return RoutingDecision(current=current, resolved=current, pinned=False,
                           needs=needs, gaps=gaps, unmet=tuple(sorted(gaps)))
