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
cache that cannot drop a rejected draft through rewind_unsupported.
"""
from __future__ import annotations

from typing import List, Optional


class DraftSource:
    """The interface the decode loop drives. This base never drafts."""

    name = "off"
    needs_rewind = True

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


class MtpSource(DraftSource):
    """Drafts with the model's own multi-token-prediction head.

    Every call delegates to the MTP state and methods on the LlamaCpp instance,
    so the ``mtp_*`` attributes there stay the record of what the reply did.
    The draft sampler is a greedy chain made by begin_call and freed by end_call.
    """

    name = "mtp"

    def __init__(self, llm) -> None:
        self._llm = llm
        self._sampler = None

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

    def rewind_unsupported(self) -> None:
        llm = self._llm
        llm._mtp_usable = False
        llm.supports_mtp = False
        llm.mtp_status = "rewind-unsupported"
        from localm.debuglog import logger
        logger.warning(
            "MTP: this model's KV cache cannot drop a rejected "
            "draft token; speculation disabled for this model")
