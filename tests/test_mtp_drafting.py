# SPDX-License-Identifier: AGPL-3.0-or-later
"""MTP speculative drafting driven through a fake native runtime.

The fake keeps a real main cache and a real draft cache (token and hidden state
per position), refuses a batch whose first position does not follow the cache,
rolls the main cache back only as far as its recurrent snapshots allow, and
builds real ctypes batches, so what LlamaCpp._generate writes into each cache
can be checked on the data itself.

The target "model" continues a token t with next_token(t). The draft head
predicts the same continuation except at the positions a test marks wrong. A
hidden state is a row filled with HIDDEN_BASE + position for the main context
and DRAFT_HIDDEN_BASE + position for the draft head's own output.
"""

import ctypes
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from localm.inference.backends.llamacpp._structs import LlamaBatch
from tests._bare_llama import make_bare_llama

EOG = 999
N_EMBD = 4
HIDDEN_BASE = 1000.0
DRAFT_HIDDEN_BASE = 5000.0
PROMPT = [11, 12, 13, 14, 15, 16]


def next_token(token):
    """The target model's continuation of *token*."""
    return 100 + (token * 7 + 3) % 500


class FakeNative:
    """The native calls _generate makes, over a modelled main and draft cache."""

    def __init__(self, llm, *, wrong_draft_positions=(), n_rs_seq=None,
                 fail_draft_decode=None, main_cost=1.0, row_cost=0.1, draft_cost=0.2):
        self.llm = llm
        self.now = 0.0                # virtual seconds, advanced by each decode
        self.main_cost, self.row_cost, self.draft_cost = main_cost, row_cost, draft_cost
        self.main = llm._ctx_ptr
        self.draft = llm._mtp_ctx_ptr
        self.main_cache = {}          # pos -> token
        self.draft_cache = {}         # pos -> (token, first hidden-state value)
        self.wrong = set(wrong_draft_positions)
        self.n_rs_seq = n_rs_seq if n_rs_seq is not None else 1 << 30
        self.fail_draft_decode = fail_draft_decode
        self.draft_decodes = []       # (positions, tokens, h values, logits flags)
        self.main_decodes = []        # (positions, tokens)
        self.main_sampler = object()
        self.draft_sampler = object()
        self._keep = []
        self._last = {}               # ctx id -> (positions, tokens, logits, h rows buffer)
        self.draft_samples = 0

    # -- batches ----------------------------------------------------------
    def batch_init(self, n, embd, n_seq_max):
        b = LlamaBatch()
        tok = (ctypes.c_int32 * n)()
        emb = (ctypes.c_float * (n * embd))() if embd else None
        pos = (ctypes.c_int32 * n)()
        nsq = (ctypes.c_int32 * n)()
        seq_rows = [(ctypes.c_int32 * max(1, n_seq_max))() for _ in range(n)]
        seq = (ctypes.POINTER(ctypes.c_int32) * n)(
            *[ctypes.cast(r, ctypes.POINTER(ctypes.c_int32)) for r in seq_rows])
        lg = (ctypes.c_int8 * n)()
        self._keep.append((tok, emb, pos, nsq, seq_rows, seq, lg))
        b.n_tokens = 0
        b.token = None if embd else ctypes.cast(tok, ctypes.c_void_p).value
        b.embd = ctypes.cast(emb, ctypes.c_void_p).value if emb is not None else None
        b.pos = ctypes.cast(pos, ctypes.c_void_p).value
        b.n_seq_id = ctypes.cast(nsq, ctypes.c_void_p).value
        b.seq_id = ctypes.cast(seq, ctypes.c_void_p).value
        b.logits = ctypes.cast(lg, ctypes.c_void_p).value
        return b

    @staticmethod
    def _read(batch):
        n = batch.n_tokens
        tok = ctypes.cast(batch.token, ctypes.POINTER(ctypes.c_int32))
        pos = ctypes.cast(batch.pos, ctypes.POINTER(ctypes.c_int32))
        lg = ctypes.cast(batch.logits, ctypes.POINTER(ctypes.c_int8))
        h = None
        if batch.embd:
            e = ctypes.cast(batch.embd, ctypes.POINTER(ctypes.c_float))
            h = [e[i * N_EMBD] for i in range(n)]
        return ([pos[i] for i in range(n)], [tok[i] for i in range(n)],
                [lg[i] for i in range(n)], h)

    # -- decode -----------------------------------------------------------
    def clock(self):
        return self.now

    def decode(self, ctx, batch):
        positions, tokens, logits, h = self._read(batch)
        if ctx is self.main:
            self.now += self.main_cost + self.row_cost * (len(tokens) - 1)
        else:
            self.now += self.draft_cost
        if ctx is self.main:
            last = max(self.main_cache, default=-1)
            if positions[0] != last + 1:
                return -1
            for p, t in zip(positions, tokens):
                self.main_cache[p] = t
            rows = (ctypes.c_float * (len(tokens) * N_EMBD))()
            for i, p in enumerate(positions):
                for j in range(N_EMBD):
                    rows[i * N_EMBD + j] = HIDDEN_BASE + p
            self._last[id(ctx)] = (positions, tokens, logits, rows)
            self.main_decodes.append((positions, tokens))
            return 0
        assert ctx is self.draft
        if self.fail_draft_decode is not None and self.fail_draft_decode(len(self.draft_decodes), positions):
            self.draft_decodes.append((positions, tokens, h, logits))
            return 1
        last = max(self.draft_cache, default=-1)
        if positions[0] != last + 1:
            return -1
        for p, t, hv in zip(positions, tokens, h):
            self.draft_cache[p] = (t, hv)
        rows = (ctypes.c_float * (len(tokens) * N_EMBD))()
        for i, p in enumerate(positions):
            for j in range(N_EMBD):
                rows[i * N_EMBD + j] = DRAFT_HIDDEN_BASE + p
        self._last[id(ctx)] = (positions, tokens, logits, rows)
        self.draft_decodes.append((positions, tokens, h, logits))
        return 0

    def nextn_ith(self, ctx, i):
        positions, _, logits, rows = self._last[id(ctx)]
        if i < 0:
            i = len(positions) + i
        if ctx is self.draft and not logits[i]:
            return None
        return ctypes.cast(ctypes.addressof(rows) + i * N_EMBD * 4,
                           ctypes.POINTER(ctypes.c_float))

    def nextn_rows(self, ctx, first, count):
        out = []
        for i in range(first, first + count):
            p = self.nextn_ith(ctx, i)
            out.append(ctypes.cast(p, ctypes.c_void_p).value if p else None)
        return out

    # -- sampling ---------------------------------------------------------
    def sample(self, sampler, ctx, idx):
        positions, tokens, logits, _ = self._last[id(ctx)]
        if idx < 0:
            idx = max(i for i, flag in enumerate(logits) if flag)
        assert logits[idx], f"sampled row {idx} has no output"
        token, pos = tokens[idx], positions[idx]
        if ctx is self.draft:
            assert sampler is self.draft_sampler
            self.draft_samples += 1
            guess = next_token(token)
            return guess + 1 if (pos + 1) in self.wrong else guess
        assert sampler is self.main_sampler
        return next_token(token)

    # -- memory -----------------------------------------------------------
    def seq_rm(self, ctx, p0):
        cache = self.main_cache if ctx is self.main else self.draft_cache
        if ctx is self.main:
            last = max(cache, default=-1)
            if 0 < p0 <= last and last - (p0 - 1) > self.n_rs_seq:
                return False
        for p in [p for p in cache if p >= p0]:
            del cache[p]
        return True

    def clear(self, ctx):
        (self.main_cache if ctx is self.main else self.draft_cache).clear()

    def install(self, mock_api):
        self.llm._clock = self.clock
        mock_api.llama_batch_init.side_effect = self.batch_init
        mock_api.llama_batch_free.side_effect = lambda b: None
        mock_api.llama_decode.side_effect = self.decode
        mock_api.llama_get_embeddings_nextn_ith.side_effect = self.nextn_ith
        mock_api.llama_get_embeddings_nextn_rows.side_effect = self.nextn_rows
        mock_api.llama_sampler_sample.side_effect = self.sample
        mock_api.llama_sampler_chain_init.return_value = self.draft_sampler
        mock_api.llama_get_memory.side_effect = lambda ctx: ctx
        mock_api.llama_memory_clear.side_effect = lambda mem, data: self.clear(mem)
        mock_api.llama_memory_seq_rm.side_effect = lambda mem, s, p0, p1: self.seq_rm(mem, p0)
        mock_api.llama_kv_cache_seq_rm.side_effect = lambda ctx, s, p0, p1: self.seq_rm(ctx, p0)


def _llama(*, mtp=True, draft_tokens=1):
    llm = make_bare_llama(
        _model_ptr=ctypes.c_void_p(1),
        _ctx_ptr=ctypes.c_void_p(2),
        _mtp_ctx_ptr=ctypes.c_void_p(3) if mtp else None,
        supports_mtp=mtp,
        mtp_status="ok:qwen35" if mtp else "disabled",
    )
    llm._mtp_wants_h = mtp
    llm._n_embd = N_EMBD
    llm._mtp_draft_max = draft_tokens
    llm._mtp_ctx_capacity = 4096 if mtp else 0
    llm._tokenizer.is_eog.side_effect = lambda t: t == EOG
    llm._fit_generation_budget = lambda n_prompt, max_new: max_new
    llm._can_reuse_kv = lambda needed: True
    return llm


def _generate(llm, fake, *, max_new_tokens=24, prompt=PROMPT, **kwargs):
    with patch("localm.inference.backends.llamacpp.llama.api") as mock_api, \
         patch("localm.inference.backends.llamacpp.llama._build_sampler",
               return_value=fake.main_sampler):
        fake.install(mock_api)
        tokens = list(llm._generate(
            prompt_tokens=list(prompt), max_new_tokens=max_new_tokens,
            temperature=0.0, top_k=40, top_p=0.95, repeat_penalty=1.0, **kwargs))
    return tokens, mock_api


def _reference(prompt, n):
    """What the target model alone produces after *prompt*: *n* tokens."""
    out, t = [], prompt[-1]
    for _ in range(n):
        t = next_token(t)
        out.append(t)
    return out


@pytest.mark.parametrize("draft_tokens", [1, 2, 3, 4])
@pytest.mark.parametrize("wrong", [(), (8, 9, 14), tuple(range(7, 40, 3)), tuple(range(7, 40))])
def test_mtp_output_matches_the_target_alone_for_every_draft_count(draft_tokens, wrong):
    llm = _llama(draft_tokens=draft_tokens)
    fake = FakeNative(llm, wrong_draft_positions=wrong)

    tokens, _ = _generate(llm, fake, max_new_tokens=24)

    assert tokens == _reference(PROMPT, 24)
    # The main cache holds exactly the prompt and what was emitted.
    main = [fake.main_cache[p] for p in sorted(fake.main_cache)]
    assert main == (PROMPT + tokens)[:len(main)]
    assert len(main) >= len(PROMPT) + len(tokens) - 1
    assert llm.mtp_call_status == ""
    assert llm.mtp_drafted > 0


def test_mtp_off_reference_run_matches_the_same_reference():
    llm = _llama(mtp=False)
    fake = FakeNative(llm)

    tokens, _ = _generate(llm, fake, max_new_tokens=24)

    assert tokens == _reference(PROMPT, 24)
    assert fake.draft_decodes == []


@pytest.mark.parametrize("draft_tokens", [1, 3])
def test_every_draft_cache_position_pairs_its_token_with_the_hidden_state_before_it(draft_tokens):
    """Prefill, accepted drafts, single-token steps and the first draft of the
    reply all pair the token at p with the target's hidden state for p - 1."""
    llm = _llama(draft_tokens=draft_tokens)
    fake = FakeNative(llm, wrong_draft_positions=(8, 12, 13, 20))

    tokens, _ = _generate(llm, fake, max_new_tokens=20)

    assert tokens == _reference(PROMPT, 20)
    assert fake.draft_cache, "nothing reached the draft cache"
    main_tokens = PROMPT + tokens
    for p in sorted(fake.draft_cache):
        token, h = fake.draft_cache[p]
        assert token == main_tokens[p], (p, token)
        expected = 0.0 if p == 0 else HIDDEN_BASE + p - 1
        assert h == expected, f"position {p} carries hidden state {h}, expected {expected}"
    # The draft cache ends in step with the main cache.
    assert sorted(fake.draft_cache) == list(range(len(fake.draft_cache)))
    assert len(fake.draft_cache) == len(fake.main_cache)


def test_the_first_draft_reads_the_hidden_state_of_the_last_prompt_token():
    llm = _llama()
    fake = FakeNative(llm)

    _generate(llm, fake, max_new_tokens=4)

    first_draft = next(d for d in fake.draft_decodes if any(d[3]))
    positions, tokens, h, logits = first_draft
    assert positions[-1] == len(PROMPT)
    assert h[-1] == HIDDEN_BASE + len(PROMPT) - 1


def test_further_drafts_read_the_draft_heads_own_hidden_state():
    llm = _llama(draft_tokens=3)
    fake = FakeNative(llm)

    _generate(llm, fake, max_new_tokens=4)

    # The first step drafts at len(PROMPT) and then two more single rows.
    rows = [d for d in fake.draft_decodes if len(d[0]) == 1 and d[0][0] > len(PROMPT)]
    assert rows, fake.draft_decodes
    positions, tokens, h, logits = rows[0]
    assert h[0] == DRAFT_HIDDEN_BASE + positions[0] - 1


def test_an_accepted_step_costs_one_draft_decode():
    """Accepted tokens reach the draft cache in the next draft decode, not in
    a separate decode of their own."""
    llm = _llama(draft_tokens=1)
    fake = FakeNative(llm)

    tokens, _ = _generate(llm, fake, max_new_tokens=12)

    verifies = [d for d in fake.main_decodes if len(d[1]) > 1 and d[0][0] >= len(PROMPT)]
    reply_draft_decodes = [d for d in fake.draft_decodes if d[0][-1] >= len(PROMPT)]
    assert verifies
    # One draft decode per verification, plus at most one final flush.
    assert len(reply_draft_decodes) <= len(verifies) + 1
    # Queued rows never request an output row.
    for positions, _tokens, _h, logits in fake.draft_decodes:
        assert all(flag == 0 for flag in logits[:-1]), logits


@pytest.mark.parametrize("draft_tokens", [2, 3, 4])
def test_a_fully_rejected_step_rolls_back_within_the_snapshots_the_context_asked_for(draft_tokens):
    """The main context is created with _mtp_rollback_snapshots(...) recurrent
    snapshots; a step whose drafts are all rejected must roll back that far."""
    llm = _llama(draft_tokens=draft_tokens)
    snapshots = llm._mtp_rollback_snapshots(SimpleNamespace(n_rs_seq=0))
    fake = FakeNative(llm, wrong_draft_positions=tuple(range(7, 80)), n_rs_seq=snapshots)

    tokens, _ = _generate(llm, fake, max_new_tokens=16)

    assert tokens == _reference(PROMPT, 16)
    assert llm.mtp_status == "ok:qwen35"
    assert snapshots >= draft_tokens


def test_rollback_beyond_the_snapshots_disables_mtp_and_still_finishes_the_reply():
    llm = _llama(draft_tokens=3)
    fake = FakeNative(llm, wrong_draft_positions=tuple(range(7, 60)), n_rs_seq=1)

    tokens, _ = _generate(llm, fake, max_new_tokens=10)

    assert tokens == _reference(PROMPT, 10)
    assert llm.mtp_status == "rewind-unsupported"
    assert llm.supports_mtp is False


def test_drafted_and_accepted_counts_follow_the_verification_outcomes():
    """Two drafts per step, the draft for position 8 wrong, nine tokens:

    step at 6 drafts 7, 8: 7 accepted, 8 rejected, the target's 8 carried;
    step at 8 drafts 9, 10: both accepted, 11 sampled from the last row;
    step at 11 drafts 12, 13: both accepted; 14 is the ninth token.
    """
    from localm.inference.backends.llamacpp.llama import _DraftPacer
    llm = _llama(draft_tokens=2)
    llm._draft_pacer = _DraftPacer(probe_every=1 << 30, bootstrap_every=1 << 30)
    fake = FakeNative(llm, wrong_draft_positions=(8,))

    tokens, _ = _generate(llm, fake, max_new_tokens=9)

    assert tokens == _reference(PROMPT, 9)
    assert (llm.mtp_drafted, llm.mtp_accepted) == (6, 5)
    assert llm.mtp_active_this_call is True


def test_a_grammar_reply_never_drafts_but_keeps_the_draft_cache_in_step():
    llm = _llama(draft_tokens=2)
    fake = FakeNative(llm)

    tokens, mock_api = _generate(llm, fake, max_new_tokens=12, grammar='root ::= "a"')

    assert tokens == _reference(PROMPT, 12)
    assert fake.draft_samples == 0
    assert llm.mtp_drafted == 0 and llm.mtp_active_this_call is False
    assert len(fake.draft_cache) == len(fake.main_cache)
    for p in sorted(fake.draft_cache)[1:]:
        assert fake.draft_cache[p][1] == HIDDEN_BASE + p - 1


def test_a_failing_queued_row_decode_stops_drafting_and_says_why():
    """The draft decode carrying accepted tokens fails: this reply stops
    speculating, reports it, and still finishes on the target alone."""
    llm = _llama(draft_tokens=1)
    calls = {"n": 0}

    def fail(index, positions):
        # Fail the first draft decode that carries a queued row.
        if len(positions) > 1 and positions[0] > len(PROMPT) and not calls["n"]:
            calls["n"] += 1
            return True
        return False

    fake = FakeNative(llm, fail_draft_decode=fail)

    tokens, _ = _generate(llm, fake, max_new_tokens=12)

    assert tokens == _reference(PROMPT, 12)
    assert calls["n"] == 1
    assert llm.mtp_call_status == "draft-decode-failed:1"
    assert llm.mtp_active_this_call is False
    assert llm._mtp_draft_stale is True
    failed_at = next(i for i, d in enumerate(fake.draft_decodes) if len(d[0]) > 1 and d[0][0] > len(PROMPT))
    assert all(not any(d[3]) for d in fake.draft_decodes[failed_at + 1:]), (
        "drafting continued after the failure")
    # The model keeps its capability.
    assert (llm.supports_mtp, llm._mtp_usable) == (True, True)


def test_a_failing_end_of_reply_flush_is_reported():
    """A grammar reply never drafts; the rows it queued for the draft cache are
    decoded when it ends, and a failure there is reported, not swallowed."""
    llm = _llama(draft_tokens=1)
    fake = FakeNative(llm)

    def fail(index, positions):
        return fail.armed
    fail.armed = False
    fake.fail_draft_decode = fail

    with patch("localm.inference.backends.llamacpp.llama.api") as mock_api, \
         patch("localm.inference.backends.llamacpp.llama._build_sampler",
               return_value=fake.main_sampler):
        fake.install(mock_api)
        gen = llm._generate(prompt_tokens=list(PROMPT), max_new_tokens=6,
                            temperature=0.0, top_k=40, top_p=0.95,
                            repeat_penalty=1.0, grammar='root ::= "a"')
        tokens = [next(gen) for _ in range(6)]
        fail.armed = True
        assert list(gen) == []

    assert tokens == _reference(PROMPT, 6)
    assert llm.mtp_call_status == "draft-catchup-failed:1"


def test_draft_budget_respects_the_reply_budget_and_both_caches():
    llm = _llama(draft_tokens=4)
    llm._ctx_capacity = 100
    llm._mtp_ctx_capacity = 100
    assert llm._mtp_draft_budget(10, None) == 4
    assert llm._mtp_draft_budget(10, 2) == 2
    assert llm._mtp_draft_budget(97, None) == 2       # verification needs pos..pos+n < 100
    assert llm._mtp_draft_budget(99, None) == 0
    llm._mtp_ctx_capacity = 50
    assert llm._mtp_draft_budget(47, None) == 2


@pytest.mark.parametrize("k", range(1, 5))
def test_the_contexts_keep_a_snapshot_per_draft_token(k):
    llm = make_bare_llama()
    llm._mtp_draft_max = k
    assert llm._mtp_rollback_snapshots(SimpleNamespace(n_rs_seq=0)) >= max(2, k)


def test_drafts_are_picked_by_a_greedy_chain_and_the_draft_context_samples_on_the_backend():
    """A bare greedy sampler re-allocates a vocabulary-sized array per sample;
    the draft sampler is a chain, and a chain is attached to the draft context
    so the choice is made inside llama_decode."""
    from localm.inference.backends.llamacpp import llama as llama_mod

    llm = make_bare_llama(_model_ptr=ctypes.c_void_p(1), _ctx_ptr=ctypes.c_void_p(2))
    with patch.object(llama_mod, "api") as mock_api:
        chain = MagicMock(name="chain")
        mock_api.llama_sampler_chain_init.return_value = chain
        mock_api.llama_init_from_model.return_value = ctypes.c_void_p(9)
        mock_api.llama_model_mtp_support.return_value = (True, "ok")
        mock_api.llama_set_embeddings_nextn.return_value = True
        mock_api.llama_model_n_embd.return_value = 4
        mock_api.has_backend_sampling.return_value = True
        mock_api.llama_set_sampler.return_value = True
        assert llm._create_mtp_context(4096) == ""

        mock_api.llama_set_sampler.assert_called_once_with(llm._mtp_ctx_ptr, 0, chain)
        mock_api.llama_sampler_chain_add.assert_called_once_with(
            chain, mock_api.llama_sampler_init_greedy.return_value)
        assert llm._mtp_backend_chain is chain

        order = []
        mock_api.llama_free.side_effect = lambda ctx: order.append("ctx")
        mock_api.llama_sampler_free.side_effect = lambda s: order.append("chain")
        llm._rebuild_mtp_context(8192, True)
    assert order[:2] == ["ctx", "chain"], "the chain must outlive the context it is attached to"


def test_a_refused_backend_sampler_is_freed_and_drafting_samples_on_the_cpu():
    from localm.inference.backends.llamacpp import llama as llama_mod

    llm = make_bare_llama(_model_ptr=ctypes.c_void_p(1), _ctx_ptr=ctypes.c_void_p(2))
    with patch.object(llama_mod, "api") as mock_api:
        mock_api.llama_init_from_model.return_value = ctypes.c_void_p(9)
        mock_api.llama_model_mtp_support.return_value = (True, "ok")
        mock_api.llama_set_embeddings_nextn.return_value = True
        mock_api.llama_model_n_embd.return_value = 4
        mock_api.has_backend_sampling.return_value = True
        mock_api.llama_set_sampler.return_value = False
        assert llm._create_mtp_context(4096) == ""
        mock_api.llama_sampler_free.assert_called_once_with(
            mock_api.llama_sampler_chain_init.return_value)
    assert llm._mtp_backend_chain is None


def test_a_reused_prompt_mirrors_only_the_new_suffix_with_hidden_states():
    """A follow-up turn keeps the shared prefix in both caches; the new suffix
    is mirrored with the target's hidden states, its first token paired with
    the state the previous reply left for that position."""
    llm = _llama(draft_tokens=2)
    fake = FakeNative(llm)
    first, _ = _generate(llm, fake, max_new_tokens=8)
    before = len(fake.draft_decodes)

    follow = PROMPT + first + [21, 22, 23]
    second, _ = _generate(llm, fake, max_new_tokens=6, prompt=follow)

    assert second == _reference(follow, 6)
    mirrored = fake.draft_decodes[before]
    assert mirrored[0][0] == len(PROMPT) + len(first)
    main_tokens = follow + second
    for p in sorted(fake.draft_cache)[1:]:
        token, h = fake.draft_cache[p]
        assert token == main_tokens[p]
        assert h == HIDDEN_BASE + p - 1, (p, h)


# --------------------------------------------------------------------------- #
#  Pacing: drafting runs only while it costs less time per token than plain   #
# --------------------------------------------------------------------------- #

def _feed(pacer, spec_cost, plain_cost, steps):
    """Drive *pacer* for *steps* steps; a speculative step takes *spec_cost*
    seconds and makes two tokens available, a plain one *plain_cost* and one."""
    decisions = []
    for _ in range(steps):
        speculative = pacer.speculate()
        decisions.append(speculative)
        if speculative:
            pacer.record(True, spec_cost, 2)
        else:
            pacer.record(False, plain_cost, 1)
    return decisions


def test_a_paying_draft_keeps_speculating_apart_from_plain_probes():
    from localm.inference.backends.llamacpp.llama import _DraftPacer
    pacer = _DraftPacer()

    decisions = _feed(pacer, spec_cost=1.2, plain_cost=1.0, steps=200)

    assert pacer.pauses == 0
    plain = decisions.count(False)
    assert pacer.n_plain >= pacer.min_samples
    assert plain <= 200 // pacer.probe_every + pacer.min_samples * pacer.bootstrap_every


def test_a_losing_draft_is_paused_and_retried_with_growing_pauses():
    from localm.inference.backends.llamacpp.llama import _DraftPacer
    pacer = _DraftPacer(pause_steps=8, max_pause_steps=32)

    decisions = _feed(pacer, spec_cost=2.6, plain_cost=1.0, steps=300)

    assert pacer.pauses >= 3
    assert decisions.count(True) < 300 * 0.25
    # Pauses double up to the ceiling.
    assert pacer._pause_len == 32


def test_a_draft_that_starts_paying_again_resumes():
    from localm.inference.backends.llamacpp.llama import _DraftPacer
    pacer = _DraftPacer(pause_steps=8)
    _feed(pacer, spec_cost=2.6, plain_cost=1.0, steps=40)
    assert pacer.pauses >= 1

    later = _feed(pacer, spec_cost=1.2, plain_cost=1.0, steps=120)

    assert later[-60:].count(True) > 50
    assert pacer._pause_len == pacer.first_pause


def test_drafting_that_costs_more_per_token_is_paused_and_the_reply_is_unchanged():
    """A draft decode costing as much as a main decode makes speculation
    slower than plain decoding: the reply pauses drafting, says so, and is the
    same text."""
    llm = _llama(draft_tokens=1)
    fake = FakeNative(llm, wrong_draft_positions=tuple(range(7, 200, 2)),
                      draft_cost=1.0, row_cost=0.5)

    tokens, _ = _generate(llm, fake, max_new_tokens=120)

    assert tokens == _reference(PROMPT, 120)
    assert llm.mtp_paused_steps > 0
    assert llm.mtp_paused_steps >= llm.mtp_steps
    assert llm.mtp_call_status == ""


def test_cheap_drafting_is_never_paused():
    llm = _llama(draft_tokens=1)
    fake = FakeNative(llm, draft_cost=0.1, row_cost=0.05)

    tokens, _ = _generate(llm, fake, max_new_tokens=120)

    assert tokens == _reference(PROMPT, 120)
    assert llm.mtp_paused_steps == 0
    assert llm._draft_pacer.pauses == 0
    assert llm.mtp_steps > 40


def test_the_pacer_is_kept_across_replies():
    llm = _llama(draft_tokens=1)
    fake = FakeNative(llm, wrong_draft_positions=tuple(range(7, 400, 2)),
                      draft_cost=1.0, row_cost=0.5)
    _generate(llm, fake, max_new_tokens=60)
    first = llm._draft_pacer
    assert first.pauses >= 1

    llm._cached_tokens = []
    llm._draft_pos = 0
    fake2 = FakeNative(llm, wrong_draft_positions=tuple(range(7, 400, 2)),
                       draft_cost=1.0, row_cost=0.5)
    _generate(llm, fake2, max_new_tokens=60)

    assert llm._draft_pacer is first
    assert llm.mtp_paused_steps > 0
