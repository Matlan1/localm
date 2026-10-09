# SPDX-License-Identifier: AGPL-3.0-or-later
"""Prompt-lookup (n-gram) drafting: the index, and the source driven through
the FakeNative runtime of test_mtp_drafting."""

import ctypes
import random

import pytest

from localm.inference.backends.llamacpp._ngram import (
    NGRAM_LIFE_DECAY, NGRAM_PRIOR_MISS_CONTEXT, NGRAM_PRIOR_MISS_REPLY,
    NGRAM_PRIOR_WEIGHT, NGRAM_REPLY_DECAY, NGRAM_REPLY_PRIOR_WEIGHT, NGRAM_RESUME_GAP,
    CandidateRuns, NgramIndex, NgramSource)
from tests._bare_llama import make_bare_llama
from tests.test_mtp_drafting import EOG, PROMPT, FakeNative, _generate, _reference


def _index(tokens, n_min=3, n_max=5):
    idx = NgramIndex(n_min, n_max)
    idx.extend(tokens)
    return idx


# --------------------------------------------------------------------------- #
#  The index                                                                  #
# --------------------------------------------------------------------------- #

def test_the_longest_matching_ngram_wins():
    # "0 1 2 3" occurred once, before 7; the shorter "1 2 3" last occurred before 8.
    idx = _index([0, 1, 2, 3, 7, 5, 1, 2, 3, 8, 0, 1, 2])
    assert idx.lookup(3, 1) == [7]


def test_the_most_recent_occurrence_wins():
    idx = _index([1, 2, 3, 7, 5, 1, 2, 3, 8, 5, 6, 1, 2], n_min=3, n_max=3)
    assert idx.lookup(3, 2) == [8, 5]


def test_the_trailing_ngram_never_matches_itself():
    idx = _index([4, 5])
    assert idx.lookup(6, 4) == []
    idx.extend([6])
    assert idx.lookup(9, 4) == []


def test_history_shorter_than_n_min_drafts_nothing():
    assert _index([]).lookup(1, 4) == []
    assert _index([1]).lookup(2, 4) == []


def test_a_period_one_stream_drafts_one_token_per_step():
    idx = _index([7] * 3)
    assert idx.lookup(7, 8) == [7]


def test_the_draft_can_run_up_to_the_trailing_token():
    idx = _index([1, 2, 3, 1, 2])
    assert idx.lookup(3, 8) == [1, 2, 3]


def test_the_draft_length_is_capped():
    loop = list(range(10, 40))
    idx = _index(loop * 2 + [10, 11])
    assert idx.lookup(12, 5) == [13, 14, 15, 16, 17]
    assert idx.lookup(12, 0) == []


def test_truncate_then_extend_equals_a_rebuild():
    rnd = random.Random(3)
    tokens = [rnd.randrange(0, 6) for _ in range(400)]
    for cut in (0, 1, 2, 57, 399, 400):
        idx = _index(tokens)
        idx.truncate(cut)
        idx.extend(tokens[cut:])
        fresh = _index(tokens)
        assert idx.tokens == fresh.tokens
        assert idx._ends == fresh._ends


def test_sync_keeps_the_shared_prefix_and_follows_the_new_tokens():
    idx = _index([1, 2, 3, 4, 5, 6])
    idx.sync([1, 2, 3, 9, 9, 9, 9])
    assert idx.tokens == [1, 2, 3, 9, 9, 9, 9]
    assert idx._ends == _index([1, 2, 3, 9, 9, 9, 9])._ends


def test_invalid_window_is_refused():
    with pytest.raises(ValueError):
        NgramIndex(0, 3)
    with pytest.raises(ValueError):
        NgramIndex(4, 3)


@pytest.mark.parametrize("recurrent,asked,expected", [
    (False, None, 8), (False, 12, 12), (False, 99, 16), (False, 0, 1),
    (True, None, 4), (True, 2, 2), (True, 16, 4)])
def test_the_draft_cap(recurrent, asked, expected):
    from localm.inference.backends.llamacpp._ngram import ngram_draft_cap, ngram_rs_seq
    cap = ngram_draft_cap(asked, recurrent)
    assert cap == expected
    assert ngram_rs_seq(0, cap) >= max(2, cap)


# --------------------------------------------------------------------------- #
#  The source in the decode loop                                              #
# --------------------------------------------------------------------------- #

def _llama(draft_max=8):
    llm = make_bare_llama(_model_ptr=ctypes.c_void_p(1), _ctx_ptr=ctypes.c_void_p(2))
    llm._tokenizer.is_eog.side_effect = lambda t: t == EOG
    llm._fit_generation_budget = lambda n_prompt, max_new: max_new
    llm._can_reuse_kv = lambda needed: True
    llm._spec_source_name = "ngram"
    llm._mtp_enabled = False
    llm._spec_draft_max = draft_max
    llm._source = NgramSource(llm, draft_max=draft_max)
    return llm


REF = _reference(PROMPT, 40)
# The prompt already holds the reply the target will give: lookup hits.
REPEATING = PROMPT + REF + [77] + PROMPT
# The reply never occurred in the prompt: lookup misses on every step until
# the target's own reply repeats (its continuation has period 20).
FRESH = [21, 22, 23, 24, 25, 26]
FRESH_TOKENS = 20
# After PROMPT the prompt continued with 777.., so drafts past the first are wrong.
DIVERGING = PROMPT + [REF[0], REF[1], 777, 778, 779] + [50] + PROMPT


@pytest.mark.parametrize("draft_max", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("prompt", [REPEATING, FRESH, DIVERGING],
                         ids=["repeating", "fresh", "diverging"])
def test_ngram_output_matches_the_target_alone(draft_max, prompt):
    llm = _llama(draft_max)
    fake = FakeNative(llm)
    n = FRESH_TOKENS if prompt is FRESH else 30

    tokens, mock_api = _generate(llm, fake, max_new_tokens=n, prompt=prompt)

    assert tokens == _reference(prompt, n)
    assert fake.main_accepted == tokens
    main = [fake.main_cache[p] for p in sorted(fake.main_cache)]
    assert main == (prompt + tokens)[:len(main)]
    mock_api.llama_sampler_accept.assert_not_called()
    src = llm._source
    if prompt is REPEATING:
        assert src.accepted > 0
        reply_decodes = [d for d in fake.main_decodes if d[0][0] >= len(prompt)]
        assert len(reply_decodes) < len(tokens)
    elif prompt is FRESH:
        assert src.drafted == 0 and src.steps == 0
    elif draft_max > 1:
        assert src.drafted > src.accepted


def test_a_reply_that_repeats_itself_drafts_from_its_own_tokens():
    llm = _llama(8)
    fake = FakeNative(llm)

    tokens, _ = _generate(llm, fake, max_new_tokens=60, prompt=FRESH)

    assert tokens == _reference(FRESH, 60)
    assert llm._source.accepted > 20


def test_a_miss_is_timed_as_a_plain_step():
    llm = _llama()
    fake = FakeNative(llm)

    _generate(llm, fake, max_new_tokens=FRESH_TOKENS, prompt=FRESH)

    assert llm._draft_pacer.n_spec == 0
    assert llm._draft_pacer.n_plain > 0


@pytest.mark.parametrize("draft_max", [1, 3])
@pytest.mark.parametrize("nth,shift", [(4, 1), (3, 7)])
def test_a_grammar_reply_drafts_and_its_sampler_sees_only_emitted_tokens(draft_max, nth, shift):
    from tests.test_mtp_drafting import _grammar_every
    grammar = _grammar_every(nth, shift)
    off = make_bare_llama(_model_ptr=ctypes.c_void_p(1), _ctx_ptr=ctypes.c_void_p(2))
    off._tokenizer.is_eog.side_effect = lambda t: t == EOG
    off._fit_generation_budget = lambda n_prompt, max_new: max_new
    off._can_reuse_kv = lambda needed: True
    off_fake = FakeNative(off, grammar=grammar)
    reference, _ = _generate(off, off_fake, max_new_tokens=30, prompt=REPEATING,
                             grammar='root ::= "a"')
    assert off_fake.draft_decodes == []

    llm = _llama(draft_max)
    fake = FakeNative(llm, grammar=grammar)
    tokens, mock_api = _generate(llm, fake, max_new_tokens=30, prompt=REPEATING,
                                 grammar='root ::= "a"')

    assert tokens == reference
    assert fake.main_accepted == tokens
    assert llm._source.drafted > 0
    mock_api.llama_sampler_accept.assert_not_called()


def test_a_follow_up_turn_reuses_the_index_and_still_matches():
    llm = _llama()
    fake = FakeNative(llm)
    first, _ = _generate(llm, fake, max_new_tokens=10, prompt=REPEATING)
    assert llm._source.index.tokens[:len(REPEATING)] == REPEATING

    follow = REPEATING + first + [61, 62] + PROMPT
    second, _ = _generate(llm, fake, max_new_tokens=12, prompt=follow)

    assert second == _reference(follow, 12)
    assert llm._source.accepted > 0
    assert llm._source.index.tokens[:len(follow)] == follow


def test_a_new_conversation_rebuilds_the_index():
    llm = _llama()
    fake = FakeNative(llm)
    _generate(llm, fake, max_new_tokens=10, prompt=REPEATING)

    tokens, _ = _generate(llm, fake, max_new_tokens=10, prompt=FRESH)

    assert tokens == _reference(FRESH, 10)
    assert llm._source.drafted == 0
    assert llm._source.index.tokens[:len(FRESH)] == FRESH
    assert REPEATING[len(PROMPT)] not in llm._source.index.tokens


def test_an_end_of_generation_token_is_never_drafted():
    prompt = PROMPT + [REF[0], REF[1], EOG, 5, 6] + [50] + PROMPT
    llm = _llama(8)
    fake = FakeNative(llm)

    tokens, _ = _generate(llm, fake, max_new_tokens=10, prompt=prompt)

    assert tokens == _reference(prompt, 10)
    assert llm._source.drafted > 0
    for positions, toks in fake.main_decodes:
        if positions[0] >= len(prompt):
            assert EOG not in toks


def test_a_cache_that_cannot_drop_rejected_drafts_disables_ngram_and_finishes():
    llm = _llama(4)
    fake = FakeNative(llm, n_rs_seq=1)

    tokens, _ = _generate(llm, fake, max_new_tokens=16, prompt=DIVERGING)

    assert tokens == _reference(DIVERGING, 16)
    src = llm._source
    assert (src.usable, src.status) == (False, "rewind-unsupported")
    report = llm.speculation_report()
    assert report["source"] == "ngram" and report["status"] == "rewind-unsupported"

    again, _ = _generate(llm, FakeNative(llm), max_new_tokens=6, prompt=REPEATING)
    assert again == _reference(REPEATING, 6)
    assert src.drafted == 0


def test_an_image_turn_reports_it_skipped_drafting():
    llm = _llama()
    llm._source.skip_call("image")
    assert llm.speculation_report()["skipped"] == "image"
    llm._source.usable = False
    llm._source.skip_call("image")
    assert llm.speculation_report()["skipped"] == ""


def test_the_report_counts_this_reply_only():
    llm = _llama()
    _generate(llm, FakeNative(llm), max_new_tokens=20, prompt=REPEATING)
    first = llm.speculation_report()
    assert first["drafted"] > 0 and first["active"] is True

    llm._cached_tokens = []
    _generate(llm, FakeNative(llm), max_new_tokens=6, prompt=FRESH)
    second = llm.speculation_report()
    assert (second["drafted"], second["accepted"], second["active"]) == (0, 0, False)
    assert second["draft_max"] == 8


def test_the_source_holds_its_model_weakly():
    import gc
    import weakref
    from localm.inference.backends.llamacpp.llama import LlamaCpp
    llm = LlamaCpp.__new__(LlamaCpp)
    llm.close = lambda: None
    llm._source = NgramSource(llm)
    gone = weakref.ref(llm)
    was = gc.isenabled()
    gc.disable()
    try:
        del llm
        assert gone() is None
    finally:
        if was:
            gc.enable()


def test_a_truncated_tail_is_never_matched():
    idx = _index([1, 2, 3, 4, 5, 6, 7, 8])
    idx.truncate(3)
    idx.extend([9, 4, 5])
    # "4 5 6" ended at 5 in the dropped tail; after the cut it never occurred.
    assert idx.lookup(6, 4) == []
    assert idx._ends == _index([1, 2, 3, 9, 4, 5])._ends


def test_a_longer_new_conversation_rebuilds_the_index_from_its_first_token():
    llm = _llama()
    fake = FakeNative(llm)
    _generate(llm, fake, max_new_tokens=4, prompt=PROMPT)

    other = [31 + i for i in range(40)]
    tokens, _ = _generate(llm, fake, max_new_tokens=6, prompt=other)

    assert tokens == _reference(other, 6)
    assert llm._source.index.tokens[:len(other)] == other


# --------------------------------------------------------------------------- #
#  Review follow-ups: image turn, initial snapshots, growth, pacing           #
# --------------------------------------------------------------------------- #

def test_an_image_turn_after_a_drafting_reply_reports_off_image():
    from unittest.mock import MagicMock, patch

    from localm.inference.backends.llamacpp import llama as llama_mod
    from tests._fake_mtmd import fake_vision_prompt
    llm = _llama()
    _generate(llm, FakeNative(llm), max_new_tokens=20, prompt=REPEATING)
    assert llm.speculation_report()["drafted"] > 0

    llm._mtmd = MagicMock(marker="<image>", encode_count=0)
    llm._mtmd.tokenize.return_value = fake_vision_prompt(text_tokens=(1, 2, 3, 4, 5))
    llm._create_batch = MagicMock(return_value=MagicMock())
    messages = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:fake"}},
        {"type": "text", "text": "describe this"}]}]
    with patch.object(llama_mod, "api") as mock_api, \
         patch.object(llama_mod, "_apply_model_template", return_value=("prompt", None)), \
         patch.object(llama_mod, "_build_sampler", return_value=MagicMock()), \
         patch.object(llama_mod.LlamaCpp, "_messages_with_markers",
                      return_value=(messages, [])):
        mock_api.llama_sampler_sample.side_effect = [100, 101, EOG]
        mock_api.llama_decode.return_value = 0
        mock_api.llama_n_ctx.return_value = 4096
        tokens = list(llm._generate_image(messages, max_new_tokens=8, temperature=0.8,
                                          top_k=40, top_p=0.95, repeat_penalty=1.1))

    assert tokens == [100, 101]
    report = llm.speculation_report()
    assert (report["skipped"], report["drafted"], report["active"]) == ("image", 0, False)


@pytest.mark.parametrize("probe,expected_cap", [
    ("attention", 8), ("recurrent", 4), ("hybrid", 4), ("no-api", 4), ("raises", 4)])
def test_the_first_context_keeps_a_snapshot_per_draft_token(probe, expected_cap):
    from types import SimpleNamespace
    from unittest.mock import patch

    from localm.inference.backends.llamacpp import llama as llama_mod
    llm = make_bare_llama(_model_ptr=ctypes.c_void_p(1))
    llm._spec_source_name = "ngram"
    llm._mtp_enabled = False
    cp = SimpleNamespace(n_rs_seq=0)
    with patch.object(llama_mod, "api") as api:
        api.has_hybrid_api.return_value = probe != "no-api"
        api.llama_model_is_recurrent.return_value = probe == "recurrent"
        api.llama_model_is_hybrid.return_value = probe == "hybrid"
        if probe == "raises":
            api.llama_model_is_recurrent.side_effect = OSError("probe")
        llm._apply_initial_spec_params(cp, None)
    assert llm._spec_draft_max == expected_cap
    assert cp.n_rs_seq == expected_cap


class _GrowingFake(FakeNative):
    """A main context that refuses a decode past *capacity*; recreating the
    context gives a bigger, empty one."""

    def __init__(self, llm, capacity, **kw):
        super().__init__(llm, **kw)
        self.capacity = capacity
        self.grown = 0

    def decode(self, ctx, batch):
        if ctx is self.main:
            positions, _t, _l, _h = self._read(batch)
            if max(positions) >= self.capacity:
                return 1
        return super().decode(ctx, batch)

    def install(self, mock_api):
        from types import SimpleNamespace
        super().install(mock_api)
        mock_api.llama_context_default_params.side_effect = lambda: SimpleNamespace(n_rs_seq=0)

        def _init(model, cp):
            self.grown += 1
            self.capacity = cp.n_ctx
            self.main_cache.clear()
            return self.main
        mock_api.llama_init_from_model.side_effect = _init


@pytest.mark.parametrize("draft_max", [2, 8])
def test_ngram_output_survives_mid_generation_context_growth(draft_max):
    llm = _llama(draft_max)
    llm._ctx_capacity = len(REPEATING) + 12
    llm._n_ctx = llm._ctx_capacity
    llm._n_ctx_grow = 256
    fake = _GrowingFake(llm, capacity=llm._ctx_capacity)

    tokens, _ = _generate(llm, fake, max_new_tokens=40, prompt=REPEATING)

    assert fake.grown >= 1
    assert tokens == _reference(REPEATING, 40)
    assert llm._source.accepted > 0
    index = llm._source.index.tokens
    assert len(index) > llm._n_ctx
    assert index == llm._cached_tokens[:len(index)]


def test_a_step_that_grew_the_context_is_not_timed_for_the_source():
    llm = _llama(8)
    llm._source.costs = _row_costs(0.1)
    llm._ctx_capacity = len(REPEATING) + 12
    llm._n_ctx = llm._ctx_capacity
    llm._n_ctx_grow = 256
    fake = _GrowingFake(llm, capacity=llm._ctx_capacity)
    seen = []
    record = llm._source.on_step_seconds
    llm._source.on_step_seconds = lambda k, s: (seen.append((k, s)), record(k, s))

    tokens, _ = _generate(llm, fake, max_new_tokens=40, prompt=REPEATING)

    assert fake.grown >= 1
    assert tokens == _reference(REPEATING, 40)
    assert any(k is None and s > 5.0 for k, s in seen)
    assert all(s < 2.0 for k, s in seen if k == 0)
    assert llm._source.step_cost(0) < 1.5


def test_ngram_drafting_that_costs_more_than_it_saves_is_paused():
    llm = _llama(8)
    fake = FakeNative(llm, main_cost=1.0, row_cost=1.5)

    tokens, _ = _generate(llm, fake, max_new_tokens=120, prompt=REPEATING * 3)

    assert tokens == _reference(REPEATING * 3, 120)
    assert llm._draft_pacer.pauses >= 1
    assert llm._source.paused_steps > 0


def test_cheap_ngram_drafting_keeps_going():
    llm = _llama(8)
    fake = FakeNative(llm, main_cost=1.0, row_cost=0.05)

    tokens, _ = _generate(llm, fake, max_new_tokens=120, prompt=REPEATING * 3)

    assert tokens == _reference(REPEATING * 3, 120)
    assert llm._draft_pacer.pauses == 0
    assert llm._source.paused_steps == 0
    assert llm._source.steps > 5


def _row_costs(row):
    """Measured costs of FakeNative(main_cost=1.0, row_cost=row)."""
    from localm.inference.backends.llamacpp._stepcosts import StepCosts
    return StepCosts(target=1.0, verify={n: 1.0 + row * (n - 1) for n in (2, 3, 5, 9)})


def _timed_run(prompt, n, row, costs):
    from localm.inference.backends.llamacpp._drafting import DraftSource
    llm = _llama(8)
    if costs == "off":
        llm._source = DraftSource()
    else:
        llm._source.costs = costs
    fake = FakeNative(llm, main_cost=1.0, row_cost=row)
    tokens, _ = _generate(llm, fake, max_new_tokens=n, prompt=prompt)
    assert tokens == _reference(prompt, n)
    return llm, fake.now


@pytest.mark.parametrize("prompt", [REPEATING, DIVERGING, FRESH],
                         ids=["repeating", "diverging", "fresh"])
@pytest.mark.parametrize("row", [0.05, 0.6, 1.5])
def test_measured_ngram_output_matches_the_target_alone(prompt, row):
    llm, _ = _timed_run(prompt, 40, row, _row_costs(row))
    assert llm._source.costs is not None


@pytest.mark.parametrize("prompt", [REPEATING * 3, DIVERGING * 3, FRESH],
                         ids=["repeating", "diverging", "fresh"])
def test_measured_ngram_never_drafts_where_verifying_costs_more_than_it_yields(prompt):
    plain_llm, plain = _timed_run(prompt, 120, 1.5, "off")
    blind, blind_s = _timed_run(prompt, 120, 1.5, None)
    measured, measured_s = _timed_run(prompt, 120, 1.5, _row_costs(1.5))
    assert blind._source.drafted > 0 and blind_s > plain
    assert measured._source.drafted == 0
    assert measured_s == pytest.approx(plain)
    assert set(measured._source._observed) == {0}


def test_measured_ngram_keeps_drafting_where_verifying_is_cheap():
    _, plain = _timed_run(REPEATING * 3, 120, 0.05, "off")
    llm, measured = _timed_run(REPEATING * 3, 120, 0.05, _row_costs(0.05))
    src = llm._source
    assert src.accepted > 90
    assert measured < 0.4 * plain
    assert 0 in src._observed and max(src._observed) == 8


def test_match_reports_where_the_occurrence_ends():
    idx = _index([0, 1, 2, 3, 7, 5, 1, 2, 3, 8, 0, 1, 2])
    assert idx.match(3, 2) == ([7, 5], 3)
    assert idx.match(9, 2) == ([], -1)
    assert idx.match(3, 0) == ([], -1)


# --------------------------------------------------------------------------- #
#  The draft length from how far candidates run                               #
# --------------------------------------------------------------------------- #

def test_a_new_kind_starts_from_the_prior_of_its_source():
    runs = CandidateRuns(4)
    assert runs.miss(("start", "context"), 1) == pytest.approx(NGRAM_PRIOR_MISS_CONTEXT)
    assert runs.miss(("start", "reply"), 3) == pytest.approx(NGRAM_PRIOR_MISS_REPLY)
    hit = 1.0 - NGRAM_PRIOR_MISS_CONTEXT
    assert runs.expected_tokens(("start", "context"), 2) == pytest.approx(
        [1.0, 1.0 + hit, 1.0 + hit + hit * hit])


def test_the_reply_counts_lead_and_the_life_counts_are_their_prior():
    runs = CandidateRuns(4)
    kind = ("resume", "context")
    runs.new_candidate(kind)
    runs.check(kind, 1, False)
    runs.check(kind, 2, True)

    def life(wrong, checked):
        return ((wrong + NGRAM_PRIOR_MISS_CONTEXT * NGRAM_PRIOR_WEIGHT)
                / (checked + NGRAM_PRIOR_WEIGHT))

    def both(wrong, checked, life_miss):
        return ((wrong + life_miss * NGRAM_REPLY_PRIOR_WEIGHT)
                / (checked + NGRAM_REPLY_PRIOR_WEIGHT))

    assert runs.miss(kind, 1) == pytest.approx(both(0.0, 1.0, life(0.0, 1.0)))
    assert runs.miss(kind, 2) == pytest.approx(both(1.0, 1.0, life(1.0, 1.0)))
    assert runs.miss(kind, 3) == pytest.approx(NGRAM_PRIOR_MISS_CONTEXT)
    runs.new_candidate(kind)
    aged_life, aged_reply = NGRAM_LIFE_DECAY, NGRAM_REPLY_DECAY
    assert runs.miss(kind, 2) == pytest.approx(
        both(aged_reply, aged_reply, life(aged_life, aged_life)))
    runs.new_reply()
    assert runs.miss(kind, 2) == pytest.approx(life(aged_life, aged_life))
    assert runs.miss(("resume", "reply"), 2) == pytest.approx(NGRAM_PRIOR_MISS_REPLY)


@pytest.mark.parametrize("last_right,end,expected", [
    (None, 40, ("start", "context")),
    (40, 40, ("continue", "context")),
    (40, 41, ("resume", "context")),
    (40, 40 + NGRAM_RESUME_GAP, ("resume", "context")),
    (40, 41 + NGRAM_RESUME_GAP, ("start", "context")),
    (40, 39, ("start", "context")),
    (100, 100, ("continue", "reply")),
    (90, 105, ("resume", "reply")),
])
def test_a_candidate_is_kinded_by_where_its_match_ends(last_right, end, expected):
    src = _llama()._source
    src._reply_start = 100
    src._last_right = last_right
    assert src._kind(end) == expected


def test_an_open_candidate_is_checked_until_its_first_wrong_token():
    src = _llama()._source
    kind = ("start", "context")
    src._open = [[10, [1, 2, 3, 4], 0, kind, 3]]
    src._check_open(list(range(10)) + [50], 1)
    assert src._open == [[10, [1, 2, 3, 4], 1, kind, 3]]
    assert src._last_right == 4

    src._check_open(list(range(10)) + [50, 1, 2], 9)

    life, reply = src.runs._counts(kind)
    assert life.checked[1:] == [1.0, 1.0, 1.0] + [0.0] * 5
    assert life.wrong[1:] == [0.0, 0.0, 1.0] + [0.0] * 5
    assert (reply.checked, reply.wrong) == (life.checked, life.wrong)
    assert src._last_right == 5
    assert src._open == []


def test_a_new_reply_keeps_what_the_model_learned_and_clears_the_reply():
    src = _llama()._source
    kind = ("continue", "reply")
    src.runs.check(kind, 1, True)
    src._open = [[3, [1], 0, kind, 1]]
    src._last_right = 7
    src.reset_call()
    life, reply = src.runs._counts(kind)
    assert (life.checked[1], life.wrong[1]) == (1.0, 1.0)
    assert (reply.checked[1], reply.wrong[1]) == (0.0, 0.0)
    assert src._open == [] and src._last_right is None


def test_the_report_carries_the_runs_and_a_draft_model_keeps_its_acceptance():
    from localm.inference.backends.llamacpp._drafting import CountedSource
    src = _llama()._source
    src.runs.check(("start", "context"), 1, False)
    rep = src.report()
    assert rep["runs"] == src.runs.report()
    assert set(rep["runs"]) == {"start/context"}
    assert "acceptance" not in rep and "acceptance_after_full_accept" not in rep
    other = CountedSource(4).report()
    assert other["acceptance"] == pytest.approx(0.6)
    assert "runs" not in other


def test_the_length_with_the_most_expected_tokens_per_second_wins():
    src = _llama(8)._source
    src.costs = _row_costs(0.5)
    kind = ("continue", "context")
    src.runs.expected_tokens = lambda k, n: [1.0, 1.9, 2.7, 3.4, 4.0][:n + 1]
    assert src._length(kind, 4) == 3
    assert src._length(kind, 2) == 2
    src.runs.expected_tokens = lambda k, n: [1.0, 1.4, 1.5][:n + 1]
    assert src._length(kind, 2) == 0
    src.runs.expected_tokens = lambda k, n: [1.0, 1.55][:n + 1]
    assert src._length(kind, 1) == 0
    src.runs.expected_tokens = lambda k, n: [1.0, 1.6][:n + 1]
    assert src._length(kind, 1) == 1


def _renamed_copy(segments=6, run=9):
    """A prompt holding a file whose runs of *run* distinct tokens each end
    in identifier 40, and a reply that copies the file with 40 renamed to 41."""
    file, first = [], 200
    for _ in range(segments):
        file += list(range(first, first + run)) + [40]
        first += run
    return [1, 2, 3] + file + [4, 5], [41 if t == 40 else t for t in file]


def _scripted_run(prompt, reply, row, costs):
    """Generate *reply* after *prompt* with the target scripted to it: ``(the
    source, virtual seconds)``. *costs* "off" decodes one token at a time."""
    from localm.inference.backends.llamacpp._drafting import DraftSource
    llm = _llama(8)
    if costs == "off":
        llm._source = DraftSource()
    else:
        llm._source.costs = costs
    fake = FakeNative(llm, main_cost=1.0, row_cost=row,
                      grammar=lambda i, chosen: reply[i] if i < len(reply) else chosen)
    tokens, _ = _generate(llm, fake, max_new_tokens=len(reply), prompt=prompt)
    assert tokens == reply
    return llm._source, fake.now


@pytest.mark.parametrize("row", [0.15, 0.4])
def test_a_copy_with_renames_drafts_up_to_each_rename(row):
    prompt, reply = _renamed_copy()
    _, plain_s = _scripted_run(prompt, reply, row, "off")
    blind, blind_s = _scripted_run(prompt, reply, row, None)
    src, measured_s = _scripted_run(prompt, reply, row, _row_costs(row))
    assert measured_s <= 1.02 * blind_s
    assert plain_s - measured_s > 0.25 * len(reply)
    assert src.accepted >= 0.8 * blind.accepted
    assert src.runs.report()["resume/context"] > 5.0


def test_a_context_whose_matches_are_always_wrong_is_held_back():
    stems = [list(range(300 + 3 * i, 303 + 3 * i)) for i in range(20)]
    prompt = [1, 2] + [t for s in stems for t in s + [50]] + [4, 5]
    reply = [t for s in stems for t in s + [51]]
    _, plain_s = _scripted_run(prompt, reply, 0.4, "off")
    src, measured_s = _scripted_run(prompt, reply, 0.4, _row_costs(0.4))
    assert src.accepted == 0
    assert src.steps <= 3
    assert measured_s - plain_s <= 0.1 * len(reply)
    assert src.held_steps > 10


def test_a_diverging_follow_up_reindexes_only_what_changed():
    class _Counting(NgramIndex):
        indexed = 0

        def extend(self, tokens):
            self.indexed += len(tokens)
            super().extend(tokens)

    idx = _Counting()
    idx.extend(list(range(1000)))
    idx.indexed = 0
    idx.sync(list(range(995)) + [7, 7])

    assert idx.indexed == 2
    assert idx._ends == _index(list(range(995)) + [7, 7])._ends
