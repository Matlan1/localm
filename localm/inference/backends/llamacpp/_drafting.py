# SPDX-License-Identifier: AGPL-3.0-or-later
"""Draft sources for speculative decoding in LlamaCpp._generate.

A draft source proposes tokens to follow the token just sampled; the decode
loop verifies them with one batch on the main context and the request's own
sampler, keeps the longest matching prefix, and removes the rest from the main
cache. The loop drives a source through these calls, in this order per reply:

    begin_call()        once, after prefill; True when this reply may draft
    drafting()          before each step; False once drafting stopped for the reply
    ready(pos)          whether a draft can be proposed for the token at pos
    budget(pos, left)   how many drafts the step may propose
    propose(token, pos, n_max)   the drafts; caller holds LlamaCpp._gen_lock
    after_verify(accepted, pos)  a verification batch at pos kept *accepted*
    after_single_token(token, pos)   the main context decoded *token* alone at pos
    on_step_seconds(k, s)  the step before took *s* seconds and drafted *k*
    finish()            the reply ended normally; caller holds _gen_lock
    end_call()          always, last; frees what begin_call made

Counters for the reply are recorded through on_verify and on_paused_step, a
failure that stops drafting for the reply through stop_this_call, and a main
cache that cannot drop a rejected draft through rewind_unsupported. A reply
that cannot draft at all (a turn with an image) calls skip_call instead of
begin_call. ``report`` describes the model's state and the last reply.
"""
from __future__ import annotations

import weakref
from typing import Any, Dict, List, Optional

from ._stepcosts import (
    ACCEPTANCE_DECAY, ACCEPTANCE_PRIOR_ACCEPTED, ACCEPTANCE_PRIOR_REJECTED,
    ACCEPTANCE_PROBE_EVERY, ACCEPTANCE_PROBE_MAX_EVERY, ACCEPTANCE_PROBE_P,
    OBSERVED_COST_CLIP, OBSERVED_COST_WEIGHT, StepCosts, best_length)

SPEC_OFF = "off"
SPEC_MTP = "mtp"
SPEC_NGRAM = "ngram"
SPEC_DRAFT = "draft"
SPEC_SOURCES = (SPEC_OFF, SPEC_MTP, SPEC_NGRAM, SPEC_DRAFT)


def resolve_spec_source(spec_source: Optional[str], mtp_enabled: bool) -> str:
    """The draft source a model uses: *spec_source* when it names one of
    ``SPEC_SOURCES``, else ``mtp`` when *mtp_enabled* and ``off`` otherwise.
    None and "" count as unset; any other value raises ValueError."""
    if spec_source is None or spec_source == "":
        return SPEC_MTP if mtp_enabled else SPEC_OFF
    value = str(spec_source).strip().lower()
    if value not in SPEC_SOURCES:
        raise ValueError("spec_source must be one of %s, got %r"
                         % (", ".join(SPEC_SOURCES), spec_source))
    return value


class DraftSource:
    """The interface the decode loop drives. This base never drafts.

    ``free_miss`` is True for a source whose proposal costs next to nothing, so
    a step it proposes nothing for is timed as a plain step.
    """

    name = "off"
    needs_rewind = True
    free_miss = False
    # Whether the loop times the steps the pacer does not, for on_step_seconds.
    observes_steps = False

    def begin_call(self) -> bool:
        return False

    def end_call(self) -> None:
        pass

    def drafting(self) -> bool:
        return False

    def ready(self, pos: int) -> bool:
        return False

    def budget(self, pos: int, tokens_left: Optional[int]) -> int:
        return 0

    def propose(self, token: int, pos: int, n_max: int) -> List[int]:
        return []

    def after_verify(self, accepted: List[int], pos: int) -> None:
        pass

    def after_single_token(self, token: int, pos: int) -> None:
        pass

    def finish(self) -> None:
        pass

    def stop_this_call(self, status: str) -> None:
        pass

    def on_verify(self, drafted: int, accepted: int) -> None:
        pass

    def on_paused_step(self) -> None:
        pass

    def on_step_seconds(self, drafted: Optional[int], seconds: float) -> None:
        """Record that a step of this reply took *seconds*, from before its
        proposal until the next token was in hand and without the time spent
        in the consumer, and verified *drafted* drafts: 0 for a step that
        decoded one token without proposing any, None for a step whose time
        does not stand for its length (a proposal or verification that
        failed, a proposal that drafted nothing, a step that grew the
        context)."""

    def rewind_unsupported(self) -> None:
        pass

    def extra_vram_bytes(self) -> int:
        """Bytes of VRAM this source needs beyond the main model and context."""
        return 0

    def skip_call(self, reason: str) -> None:
        """Record that the reply about to run cannot draft, and why."""

    def close(self) -> None:
        """Free what the source holds natively. Called while the model's own
        context and weights are still allocated, before they are freed."""

    def report(self) -> dict:
        """The model's speculation state and the last reply's figures:
        ``status`` (model level), ``active`` (the reply speculated),
        ``call_status`` (why it stopped partway, "" when it did not),
        ``skipped`` (why it could not draft at all, "" when it could),
        ``drafted``, ``accepted``, ``steps``, ``paused_steps`` and
        ``draft_max``."""
        return {"status": "disabled", "active": False, "call_status": "",
                "skipped": "", "drafted": 0, "accepted": 0, "steps": 0,
                "paused_steps": 0, "draft_max": 0}


class CountedSource(DraftSource):
    """A source that keeps its own status and per-reply counters, and chooses
    each step's draft length from measured step costs.

    ``usable`` and ``status`` are model level: ``disable`` clears the first and
    names why in the second, for the rest of the model's life. The counters are
    reset by ``begin_call`` (through ``reset_call``) and by ``skip_call``;
    ``held_steps`` counts the reply's steps ``choose_length`` gave no drafts.
    ``label`` names the source in log lines.

    ``costs`` is the ``StepCosts`` measured at load, or None. With costs,
    ``step_cost`` corrects them with the step times the loop reports through
    ``on_step_seconds`` and ``choose_length`` picks the draft length. The
    acceptance is estimated separately for a step right after one that
    verified every draft it proposed ("after a full accept") and for any other
    step. The estimates, the probing schedule and the observed figures live
    for the model's life. A subclass needs ``_llm`` for ``cap_drafts``.
    """

    label = "drafting"
    observes_steps = True

    def __init__(self, draft_max: int) -> None:
        self.draft_max = draft_max
        self.usable = True
        self.status = "ok"
        self._drafting = False
        self.active_this_call = False
        self.call_status = ""
        self.skipped = ""
        self.drafted = 0
        self.accepted = 0
        self.steps = 0
        self.paused_steps = 0
        self.held_steps = 0
        self.costs: Optional[StepCosts] = None
        self._observed: Dict[int, float] = {}
        self._step_over_s = 0.0
        self._draft_over_s = 0.0
        # Decayed [accepted drafts, rejections], keyed by "after a full accept".
        self._evidence: Dict[bool, List[float]] = {False: [0.0, 0.0], True: [0.0, 0.0]}
        self._hot = False
        self._chosen_hot = False
        self._since_probe = ACCEPTANCE_PROBE_EVERY
        self._probe_every = ACCEPTANCE_PROBE_EVERY
        self._probing = False

    def reset_call(self, skipped: str = "") -> None:
        self.active_this_call = False
        self.call_status = ""
        self.skipped = skipped
        self.drafted = 0
        self.accepted = 0
        self.steps = 0
        self.paused_steps = 0
        self.held_steps = 0

    def skip_call(self, reason: str) -> None:
        self.reset_call(reason if self.usable else "")
        self._drafting = False

    def report(self) -> dict:
        """``DraftSource.report`` plus ``held_steps``, ``acceptance`` and
        ``acceptance_after_full_accept``, and with measured costs ``costs``
        (``StepCosts.report``) and ``observed_ms``, the corrected step
        milliseconds of each draft length seen so far."""
        out = {"status": self.status, "active": self.active_this_call,
               "call_status": self.call_status, "skipped": self.skipped,
               "drafted": self.drafted, "accepted": self.accepted,
               "steps": self.steps, "paused_steps": self.paused_steps,
               "held_steps": self.held_steps, "draft_max": self.draft_max,
               "acceptance": round(self.acceptance(), 3),
               "acceptance_after_full_accept": round(self.acceptance(True), 3)}
        if self.costs is not None:
            out["costs"] = self.costs.report()
            out["observed_ms"] = {k: round(s * 1000, 3)
                                  for k, s in sorted(self._observed.items())}
        return out

    def drafting(self) -> bool:
        return self._drafting

    def stop_this_call(self, status: str) -> None:
        self._drafting = False
        self.active_this_call = False
        self.call_status = status
        from localm.debuglog import logger
        logger.info("%s stopped for this reply - %s", self.label, status)

    def disable(self, status: str) -> None:
        """Stop drafting for the rest of the model's life and record why."""
        self.usable = False
        self._drafting = False
        self.status = status
        from localm.debuglog import logger
        logger.warning("%s disabled for this model - %s", self.label, status)

    def on_verify(self, drafted: int, accepted: int) -> None:
        """Count a verification of *drafted* drafts of which *accepted* were
        kept, and fold it into the acceptance evidence of the state its length
        was chosen in. A probe that had a rejection doubles the probe interval
        up to ``ACCEPTANCE_PROBE_MAX_EVERY``; one without resets it."""
        self.steps += 1
        self.drafted += drafted
        self.accepted += accepted
        self.active_this_call = True
        rejected = accepted < drafted
        evidence = self._evidence[self._chosen_hot]
        evidence[0] = evidence[0] * ACCEPTANCE_DECAY + accepted
        evidence[1] = evidence[1] * ACCEPTANCE_DECAY + (1.0 if rejected else 0.0)
        if self._probing:
            self._probe_every = (min(2 * self._probe_every, ACCEPTANCE_PROBE_MAX_EVERY)
                                 if rejected else ACCEPTANCE_PROBE_EVERY)
            self._probing = False
        self._hot = drafted > 0 and not rejected
        self._since_probe = 0

    def on_paused_step(self) -> None:
        self.paused_steps += 1

    def rewind_unsupported(self) -> None:
        self.disable("rewind-unsupported")

    def acceptance(self, after_full_accept: bool = False) -> float:
        """Estimated probability that one draft is accepted. For a step that
        is not after a full accept: accepted drafts over accepted drafts plus
        rejections, each verification weighing ``ACCEPTANCE_DECAY`` of the
        evidence before it, with the prior added. After a full accept
        (*after_full_accept*): the same over that state's own evidence, with
        the first estimate as its prior at the same weight."""
        weight = ACCEPTANCE_PRIOR_ACCEPTED + ACCEPTANCE_PRIOR_REJECTED
        accepted, rejected = self._evidence[False]
        p = (accepted + ACCEPTANCE_PRIOR_ACCEPTED) / (accepted + rejected + weight)
        if not after_full_accept:
            return p
        accepted, rejected = self._evidence[True]
        return (accepted + weight * p) / (accepted + rejected + weight)

    def modelled_step_cost(self, drafted: int) -> float:
        """Seconds of a step drafting *drafted*: the measured ``costs`` plus
        the estimated seconds every step takes beyond them and, per draft, the
        estimated seconds a draft adds beyond that (each never below 0).
        Raises RuntimeError without ``costs``."""
        costs = self.costs
        if costs is None:
            raise RuntimeError("step costs are not measured")
        return (costs.step_cost(drafted) + max(0.0, self._step_over_s)
                + max(0.0, self._draft_over_s) * max(0, drafted))

    def step_cost(self, drafted: int) -> float:
        """Seconds of a step drafting *drafted*: the running figure of the
        steps of that length seen so far, else ``modelled_step_cost``.
        Requires ``costs``."""
        seen = self._observed.get(drafted)
        return seen if seen is not None else self.modelled_step_cost(drafted)

    def on_step_seconds(self, drafted: Optional[int], seconds: float) -> None:
        """Fold a step's *seconds*, clipped to within ``OBSERVED_COST_CLIP``
        times the running figure for its length, into that figure, which
        starts at ``step_cost`` and moves ``OBSERVED_COST_WEIGHT`` of the way
        to each new time, and into the overhead estimates of
        ``modelled_step_cost``: a plain step's time beyond the measured
        one-token decode is the overhead of every step, and a drafting step's
        time beyond its measured decodes and that overhead, divided by its
        drafts, the overhead of a draft. Ignored without ``costs``, for
        *drafted* None or for a time of 0 or less."""
        if drafted is None or self.costs is None or seconds <= 0.0:
            return
        before = self.step_cost(drafted)
        seconds = min(max(seconds, before / OBSERVED_COST_CLIP), before * OBSERVED_COST_CLIP)
        extra = seconds - self.costs.step_cost(drafted)
        if drafted <= 0:
            self._step_over_s += OBSERVED_COST_WEIGHT * (extra - self._step_over_s)
        else:
            per = (extra - max(0.0, self._step_over_s)) / drafted
            self._draft_over_s += OBSERVED_COST_WEIGHT * (per - self._draft_over_s)
        self._observed[drafted] = before + OBSERVED_COST_WEIGHT * (seconds - before)

    @property
    def _llm(self) -> Any:
        """The LlamaCpp instance the source drafts for; a subclass provides it."""
        raise NotImplementedError

    def cap_drafts(self, pos: int, tokens_left: Optional[int]) -> int:
        """Most drafts the step at *pos* may propose: ``draft_max``, the
        tokens left and the room left in the main cache."""
        n = self.draft_max
        if tokens_left is not None:
            n = min(n, tokens_left)
        return max(0, min(n, self._llm._ctx_capacity - pos - 1))

    def length_pays(self, p: float, k: int, tokens_left: Optional[int]) -> bool:
        """Whether a step drafting *k* at acceptance *p* is worth what it costs
        beyond ``step_cost``. Always True here."""
        return True

    def choose_length(self, n: int, tokens_left: Optional[int]) -> int:
        """The draft length of a step allowed at most *n* drafts: the
        ``best_length`` under ``step_cost`` at the ``acceptance`` of the
        step's state (after a full accept when the step before verified every
        draft it proposed). When that is 0 and this method has given 0 for
        at least the probe interval (``ACCEPTANCE_PROBE_EVERY`` at first)
        since the last verification, the length sized for
        ``ACCEPTANCE_PROBE_P`` instead (0 when none pays even there), so the
        acceptance is measured. 0 when
        ``length_pays`` refuses the length; each 0 counts in ``held_steps``,
        and one after a full accept also decays that state's evidence by
        ``ACCEPTANCE_DECAY``. Requires ``costs``."""
        hot, self._hot = self._hot, False
        self._chosen_hot = hot
        self._probing = False
        if n <= 0:
            return 0
        p = self.acceptance(hot)
        k = best_length(p, n, self.step_cost)
        if k == 0 and self._since_probe >= self._probe_every:
            p = ACCEPTANCE_PROBE_P
            k = best_length(p, n, self.step_cost)
            self._probing = k > 0
        if k and not self.length_pays(p, k, tokens_left):
            k = 0
            self._probing = False
        if k == 0:
            self._since_probe += 1
            self.held_steps += 1
            if hot:
                evidence = self._evidence[True]
                evidence[0] *= ACCEPTANCE_DECAY
                evidence[1] *= ACCEPTANCE_DECAY
        return k


class MtpSource(DraftSource):
    """Drafts with the model's own multi-token-prediction head.

    Every call delegates to the MTP state and methods on the LlamaCpp instance,
    so the ``mtp_*`` attributes there stay the record of what the reply did.
    The draft sampler is a greedy chain made by begin_call and freed by end_call.
    The instance is held through a weak reference, so a model that keeps its
    source is still finalized when its last reference goes.
    """

    name = "mtp"

    def __init__(self, llm) -> None:
        self._llm_ref = weakref.ref(llm)
        self._sampler = None

    @property
    def _llm(self):
        return self._llm_ref()

    def begin_call(self) -> bool:
        from .llama import _greedy_chain
        llm = self._llm
        self._sampler = (_greedy_chain()
                         if llm._mtp_ctx_ptr is not None and llm._mtp_usable else None)
        llm._mtp_drafting = self._sampler is not None
        return self._sampler is not None

    def end_call(self) -> None:
        from .llama import api
        sampler, self._sampler = self._sampler, None
        if sampler is not None:
            api.llama_sampler_free(sampler)

    def drafting(self) -> bool:
        llm = self._llm
        return (llm._mtp_ctx_ptr is not None and self._sampler is not None
                and llm._mtp_usable and llm._mtp_drafting)

    def ready(self, pos: int) -> bool:
        return self._llm._pending_h_pos == pos - 1

    def budget(self, pos: int, tokens_left: Optional[int]) -> int:
        return self._llm._mtp_draft_budget(pos, tokens_left)

    def propose(self, token: int, pos: int, n_max: int) -> List[int]:
        return self._llm._propose_drafts(token, pos, n_max, self._sampler)

    def after_verify(self, accepted: List[int], pos: int) -> None:
        self._llm._after_verify(accepted, pos)

    def after_single_token(self, token: int, pos: int) -> None:
        self._llm._after_main_token(token, pos)

    def finish(self) -> None:
        self._llm._finish_draft_tracking()

    def stop_this_call(self, status: str) -> None:
        self._llm._stop_drafting_this_call(status)

    def on_verify(self, drafted: int, accepted: int) -> None:
        llm = self._llm
        llm.mtp_steps += 1
        llm.mtp_drafted += drafted
        llm.mtp_accepted += accepted
        llm.mtp_active_this_call = True

    def on_paused_step(self) -> None:
        self._llm.mtp_paused_steps += 1

    def report(self) -> dict:
        llm = self._llm
        return {"status": str(getattr(llm, "mtp_status", "") or ""),
                "active": bool(llm.mtp_active_this_call),
                "call_status": str(llm.mtp_call_status or ""),
                "skipped": str(llm.mtp_skipped or ""),
                "drafted": int(llm.mtp_drafted), "accepted": int(llm.mtp_accepted),
                "steps": int(llm.mtp_steps), "paused_steps": int(llm.mtp_paused_steps),
                "draft_max": int(llm._mtp_draft_max)}

    def rewind_unsupported(self) -> None:
        llm = self._llm
        llm._mtp_usable = False
        llm.supports_mtp = False
        llm.mtp_status = "rewind-unsupported"
        from localm.debuglog import logger
        logger.warning(
            "MTP: this model's KV cache cannot drop a rejected "
            "draft token; speculation disabled for this model")
