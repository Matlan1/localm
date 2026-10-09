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
from typing import List, Optional

SPEC_OFF = "off"
SPEC_MTP = "mtp"
SPEC_NGRAM = "ngram"
SPEC_SOURCES = (SPEC_OFF, SPEC_MTP, SPEC_NGRAM)


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

    def rewind_unsupported(self) -> None:
        pass

    def extra_vram_bytes(self) -> int:
        """Bytes of VRAM this source needs beyond the main model and context."""
        return 0

    def skip_call(self, reason: str) -> None:
        """Record that the reply about to run cannot draft, and why."""

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
