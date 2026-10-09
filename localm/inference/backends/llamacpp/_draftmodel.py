# SPDX-License-Identifier: AGPL-3.0-or-later
"""Draft-model speculative decoding: a second, smaller GGUF drafts tokens on
its own context for the loaded model to verify.

The draft model must share the target's vocabulary (``draft_vocab_mismatch``).
Its cache follows the main cache lazily: each proposal first makes the draft
cache hold the main cache's tokens, keeping what the two still share, then
decodes the sampled token and one draft at a time.
"""
from __future__ import annotations

import weakref
from dataclasses import dataclass
from typing import Callable, List, Optional

from ._drafting import SPEC_DRAFT, CountedSource

DRAFT_MODEL_DRAFT_TOKENS_DEFAULT = 2
DRAFT_MODEL_DRAFT_TOKENS_MAX = 16
# Draft and target vocabulary sizes may differ by at most this many entries.
DRAFT_VOCAB_SIZE_MAX_DIFFERENCE = 128
# Token text is compared from this id up.
DRAFT_VOCAB_CHECK_START_ID = 5
# Batch and micro-batch of the draft context.
DRAFT_CONTEXT_BATCH = 512
# VRAM charged for the draft context beyond its weights, KV cache and logits
# buffer. See test_the_draft_model_charge_covers_the_measured_buffers.
DRAFT_COMPUTE_MARGIN_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class VocabView:
    """What the compatibility rule reads from one model's vocabulary.
    ``vocab_type`` is only compared for equality."""
    vocab_type: object
    n_tokens: int
    add_bos: bool
    add_eos: bool
    bos: int
    eos: int
    text: Callable[[int], bytes]


def draft_vocab_mismatch(target: VocabView, draft: VocabView) -> Optional[str]:
    """Why *draft* cannot draft for *target*, or None when it can.

    The two must have the same vocabulary type; the same add-BOS and add-EOS
    flags, and the same BOS / EOS id where it is added; sizes at most
    DRAFT_VOCAB_SIZE_MAX_DIFFERENCE apart; and the same token text for every id
    from DRAFT_VOCAB_CHECK_START_ID up to the smaller size."""
    if target.vocab_type != draft.vocab_type:
        return "vocab-type %s != %s" % (draft.vocab_type, target.vocab_type)
    if target.add_bos != draft.add_bos or (target.add_bos and target.bos != draft.bos):
        return "bos differs"
    if target.add_eos != draft.add_eos or (target.add_eos and target.eos != draft.eos):
        return "eos differs"
    if abs(target.n_tokens - draft.n_tokens) > DRAFT_VOCAB_SIZE_MAX_DIFFERENCE:
        return "vocab size %d vs %d" % (draft.n_tokens, target.n_tokens)
    for i in range(DRAFT_VOCAB_CHECK_START_ID, min(target.n_tokens, draft.n_tokens)):
        if target.text(i) != draft.text(i):
            return "token %d differs" % i
    return None


def gguf_vocab_view(signature: dict) -> VocabView:
    """The VocabView of a GGUF's ``gguf_vocab_signature``. A flag the file
    does not carry reads as False and an id it does not carry as -1."""
    tokens = signature["tokens"]
    return VocabView(
        vocab_type=signature.get("model"),
        n_tokens=len(tokens),
        add_bos=bool(signature.get("add_bos")),
        add_eos=bool(signature.get("add_eos")),
        bos=-1 if signature.get("bos") is None else int(signature["bos"]),
        eos=-1 if signature.get("eos") is None else int(signature["eos"]),
        text=lambda i: tokens[i].encode("utf-8"))


def native_vocab_view(api, model_ptr) -> VocabView:
    """The VocabView of a loaded model, read through the native vocab API.
    Token text is read through one bound function, since the comparison reads
    every token."""
    import ctypes
    vocab = api.llama_model_get_vocab(model_ptr)
    get_text = api._bind("llama_vocab_get_text", ctypes.c_char_p,
                         api.LlamaVocab, api.llama_token)
    return VocabView(
        vocab_type=int(api.llama_vocab_type(vocab)),
        n_tokens=int(api.llama_vocab_n_tokens(vocab)),
        add_bos=bool(api.llama_vocab_get_add_bos(vocab)),
        add_eos=bool(api.llama_vocab_get_add_eos(vocab)),
        bos=int(api.llama_vocab_bos(vocab)),
        eos=int(api.llama_vocab_eos(vocab)),
        text=lambda i: get_text(vocab, i) or b"")


class DraftModelSource(CountedSource):
    """Drafts with a second model on its own context.

    Owns the draft model, its context and a greedy sampler chain; ``close``
    frees all three and must run before the target's context and model are
    freed. The draft context is created at the main context's size and
    recreated when the main one grows. ``_tokens`` is what the draft cache
    holds, position i at index i; its first ``_valid`` entries are known to
    equal the main cache's.
    """

    name = SPEC_DRAFT
    label = "draft-model drafting"
    free_miss = False

    def __init__(self, llm, model_ptr, draft_max: int,
                 n_threads: Optional[int] = None) -> None:
        super().__init__(draft_max)
        self._llm_ref = weakref.ref(llm)
        self._model = model_ptr
        self._n_threads = n_threads
        self._ctx = None
        self._ctx_capacity = 0
        self._ctx_batch = 0
        self._sampler = None
        self._tokens: List[int] = []
        self._valid = 0

    @property
    def _llm(self):
        return self._llm_ref()

    def create_context(self, n_ctx: int, offload_kqv: bool) -> str:
        """(Re)create the draft context at *n_ctx* tokens with an empty cache.
        Returns "" on success, else "draft-context-refused"."""
        from .llama import _greedy_chain, api
        self.free_context()
        cp = api.llama_context_default_params()
        cp.n_ctx = n_ctx
        cp.n_batch = min(n_ctx, DRAFT_CONTEXT_BATCH)
        cp.n_ubatch = cp.n_batch
        cp.offload_kqv = offload_kqv
        if self._n_threads is not None:
            cp.n_threads = self._n_threads
            cp.n_threads_batch = self._n_threads
        ctx = api.llama_init_from_model(self._model, cp)
        if not ctx:
            return "draft-context-refused"
        self._ctx = ctx
        self._ctx_capacity = cp.n_ctx
        self._ctx_batch = cp.n_batch
        if self._sampler is None:
            self._sampler = _greedy_chain()
        return ""

    def free_context(self) -> None:
        from .llama import api
        ctx, self._ctx = self._ctx, None
        self._ctx_capacity = 0
        self._tokens = []
        self._valid = 0
        if ctx is not None:
            api.llama_free(ctx)

    def close(self) -> None:
        """Free the draft context, sampler and model. Idempotent."""
        from .llama import api
        self.free_context()
        sampler, self._sampler = self._sampler, None
        if sampler is not None:
            api.llama_sampler_free(sampler)
        model, self._model = self._model, None
        if model is not None:
            api.llama_free_model(model)

    def begin_call(self) -> bool:
        self.reset_call()
        self._drafting = self.usable and self._ctx is not None
        self._valid = 0
        return self._drafting

    def ready(self, pos: int) -> bool:
        return True

    def budget(self, pos: int, tokens_left: Optional[int]) -> int:
        n = self.draft_max
        if tokens_left is not None:
            n = min(n, tokens_left)
        return max(0, min(n, self._llm._ctx_capacity - pos - 1))

    def _reset_cache(self) -> None:
        from .llama import api
        self._tokens = []
        self._valid = 0
        if self._ctx is not None:
            api.llama_memory_clear(api.llama_get_memory(self._ctx), True)

    def _decode(self, tokens: List[int], start: int) -> int:
        from .llama import api
        llm = self._llm
        step = max(1, self._ctx_batch)
        for i in range(0, len(tokens), step):
            batch = llm._create_batch(tokens[i:i + step], start + i, logits_at_last_only=True)
            try:
                ret = api.llama_decode(self._ctx, batch)
            finally:
                api.llama_batch_free(batch)
            if ret != 0:
                return ret
        return 0

    def propose(self, token: int, pos: int, n_max: int) -> List[int]:
        from .llama import api
        llm = self._llm
        cached = llm._cached_tokens
        if len(cached) != pos:
            self.stop_this_call("draft-out-of-step")
            return []
        if llm._ctx_capacity > self._ctx_capacity:
            failure = self.create_context(llm._ctx_capacity, llm._offload_kqv)
            if failure:
                self.disable(failure)
                return []
        p = self._valid
        lim = min(len(self._tokens), len(cached))
        while p < lim and self._tokens[p] == cached[p]:
            p += 1
        if p < len(self._tokens):
            if not api.llama_memory_seq_rm(api.llama_get_memory(self._ctx), 0, p, -1):
                self.disable("draft-rewind-unsupported")
                return []
            del self._tokens[p:]
        pending = list(cached[p:]) + [token]
        try:
            ret = self._decode(pending, p)
            if ret != 0:
                self._reset_cache()
                self.stop_this_call("draft-decode-failed:%d" % ret)
                return []
            self._tokens.extend(pending)
            self._valid = len(cached)
            is_eog = llm._tokenizer.is_eog
            drafts: List[int] = []
            while True:
                d = api.llama_sampler_sample(self._sampler, self._ctx, -1)
                if is_eog(d):
                    break
                drafts.append(d)
                if len(drafts) >= n_max:
                    break
                ret = self._decode([d], pos + len(drafts))
                if ret != 0:
                    self._reset_cache()
                    self.stop_this_call("draft-decode-failed:%d" % ret)
                    return []
                self._tokens.append(d)
            return drafts
        except Exception:
            self._reset_cache()
            raise
