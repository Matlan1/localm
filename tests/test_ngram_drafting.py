# SPDX-License-Identifier: AGPL-3.0-or-later
"""Prompt-lookup (n-gram) drafting: the index, and the source driven through
the FakeNative runtime of test_mtp_drafting."""

import ctypes
import random

import pytest

from localm.inference.backends.llamacpp._ngram import NgramIndex, NgramSource
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
    llm._ngram_draft_max = draft_max
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
