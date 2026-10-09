# SPDX-License-Identifier: AGPL-3.0-or-later
"""Draft-model speculative drafting: the vocabulary rule, the draft cache kept
in step with the main cache through the FakeNative runtime of
test_mtp_drafting, the load and free paths, the VRAM charge and the settings."""

import ctypes
import struct
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from localm.inference.backends.llamacpp._draftmodel import (
    DRAFT_COMPUTE_MARGIN_BYTES, DRAFT_CONTEXT_BATCH, DRAFT_VOCAB_CHECK_START_ID,
    DRAFT_VOCAB_SIZE_MAX_DIFFERENCE, DraftModelSource, VocabView, draft_vocab_mismatch,
    gguf_vocab_view)
from tests._bare_llama import make_bare_llama
from tests.test_model_capabilities import _kv_string, _kv_string_array, _kv_uint32, write_gguf
from tests.test_mtp_drafting import EOG, PROMPT, FakeNative, _generate, _reference


# --------------------------------------------------------------------------- #
#  The vocabulary rule                                                        #
# --------------------------------------------------------------------------- #

def _view(n=300, *, vocab_type=2, add_bos=False, add_eos=False, bos=1, eos=2,
          changed=None):
    texts = {i: b"t%d" % i for i in range(n)}
    texts.update(changed or {})
    return VocabView(vocab_type, n, add_bos, add_eos, bos, eos, lambda i: texts[i])


@pytest.mark.parametrize("draft,ok", [
    (_view(), True),
    (_view(vocab_type=1), False),
    (_view(add_bos=True), False),
    (_view(add_eos=True), False),
    (_view(300 + DRAFT_VOCAB_SIZE_MAX_DIFFERENCE), True),
    (_view(300 + DRAFT_VOCAB_SIZE_MAX_DIFFERENCE + 1), False),
    (_view(changed={DRAFT_VOCAB_CHECK_START_ID: b"other"}), False),
    (_view(changed={299: b"other"}), False),
    (_view(changed={DRAFT_VOCAB_CHECK_START_ID - 1: b"other"}), True),
])
def test_the_draft_vocabulary_rule(draft, ok):
    assert (draft_vocab_mismatch(_view(), draft) is None) is ok


def test_bos_and_eos_ids_matter_only_when_added():
    assert draft_vocab_mismatch(_view(bos=1), _view(bos=9)) is None
    assert draft_vocab_mismatch(_view(add_bos=True, bos=1),
                                _view(add_bos=True, bos=9)) == "bos differs"
    assert draft_vocab_mismatch(_view(add_eos=True, eos=2),
                                _view(add_eos=True, eos=9)) == "eos differs"


def _kv_bool(value: bool) -> bytes:
    return struct.pack("<I", 7) + struct.pack("<?", value)


def _vocab_gguf(path, tokens, *, model="gpt2", add_bos=None, bos=None, arch="qwen2",
                extra=()):
    kv = [("general.architecture", _kv_string(arch)),
          ("tokenizer.ggml.model", _kv_string(model)),
          ("tokenizer.ggml.tokens", _kv_string_array(tokens)), *extra]
    if add_bos is not None:
        kv.append(("tokenizer.ggml.add_bos_token", _kv_bool(add_bos)))
    if bos is not None:
        kv.append(("tokenizer.ggml.bos_token_id", _kv_uint32(bos)))
    return write_gguf(path, kv)


def test_the_vocabulary_is_read_from_gguf_metadata(tmp_path):
    from localm.model_manager.gguf import gguf_vocab_signature
    tokens = ["<s>", "a", "b", "c", "d", "e", "f", "g"]
    sig = gguf_vocab_signature(_vocab_gguf(tmp_path / "a.gguf", tokens, add_bos=True, bos=0))
    assert sig == {"model": "gpt2", "tokens": tokens, "add_bos": True, "add_eos": None,
                   "bos": 0, "eos": None}
    view = gguf_vocab_view(sig)
    assert (view.n_tokens, view.add_bos, view.bos, view.eos, view.text(6)) == (
        8, True, 0, -1, b"f")


def test_two_gguf_files_compare_by_their_metadata(tmp_path):
    from localm.model_manager.gguf import gguf_vocab_signature
    tokens = ["x%d" % i for i in range(40)]
    a = gguf_vocab_view(gguf_vocab_signature(_vocab_gguf(tmp_path / "a.gguf", tokens)))
    b = gguf_vocab_view(gguf_vocab_signature(_vocab_gguf(tmp_path / "b.gguf", tokens)))
    c = gguf_vocab_view(gguf_vocab_signature(
        _vocab_gguf(tmp_path / "c.gguf", tokens, model="llama")))
    assert draft_vocab_mismatch(a, b) is None
    assert draft_vocab_mismatch(a, c) is not None


@pytest.mark.parametrize("arch,extra,refused", [
    ("qwen2", (), False), ("eagle3", (), True), ("dflash", (), True), ("llada", (), True),
    ("t5", (), True), ("bert", (), True), ("qwen2", (("qwen2.pooling_type", 1),), True)])
def test_only_a_causal_chat_model_can_be_a_draft_model(tmp_path, arch, extra, refused):
    from localm.inference.backends.llamacpp._draftmodel import draft_role_refusal
    kv = [(k, _kv_uint32(v)) for k, v in extra]
    path = _vocab_gguf(tmp_path / "d.gguf", ["x%d" % i for i in range(40)], arch=arch, extra=kv)
    assert (draft_role_refusal(path) is not None) is refused
    (tmp_path / "junk.gguf").write_bytes(b"nope")
    assert draft_role_refusal(tmp_path / "junk.gguf") is None


def test_a_file_without_a_vocabulary_has_no_signature(tmp_path):
    from localm.model_manager.gguf import gguf_vocab_signature
    path = write_gguf(tmp_path / "n.gguf", [("general.architecture", _kv_string("x"))])
    assert gguf_vocab_signature(path) is None
    (tmp_path / "junk.gguf").write_bytes(b"nope")
    assert gguf_vocab_signature(tmp_path / "junk.gguf") is None


# --------------------------------------------------------------------------- #
#  The source in the decode loop                                              #
# --------------------------------------------------------------------------- #

class DraftFake(FakeNative):
    """FakeNative whose draft context is an ordinary token decoder: what the
    draft model holds at each position, and a greedy draft that follows the
    target except at the positions a test marks wrong."""

    def __init__(self, llm, *, refuse_draft_rm=False, **kw):
        super().__init__(llm, **kw)
        self.draft = llm._source._ctx
        self.refuse_draft_rm = refuse_draft_rm
        self.draft_rm = []

    def decode(self, ctx, batch):
        if ctx is not self.draft:
            return super().decode(ctx, batch)
        positions, tokens, logits, _h = self._read(batch)
        self.now += self.draft_cost
        if (self.fail_draft_decode is not None
                and self.fail_draft_decode(len(self.draft_decodes), positions)):
            self.draft_decodes.append((positions, tokens, None, logits))
            return 1
        last = max(self.draft_cache, default=-1)
        if positions[0] != last + 1:
            return -1
        for p, t in zip(positions, tokens):
            self.draft_cache[p] = (t, None)
        self._last[id(ctx)] = (positions, tokens, logits, None)
        self.draft_decodes.append((positions, tokens, None, logits))
        return 0

    def seq_rm(self, ctx, p0):
        if ctx is self.draft:
            self.draft_rm.append(p0)
            if self.refuse_draft_rm:
                return False
        return super().seq_rm(ctx, p0)


def _flat_costs():
    """Step costs under which every draft length up to the cap is chosen."""
    from localm.inference.backends.llamacpp._stepcosts import StepCosts
    return StepCosts(target=1.0, verify={n: 1.0 for n in (2, 3, 5, 9, 17)}, draft=0.0,
                     draft_prefill=0.0)


def _llama(draft_max=4):
    llm = make_bare_llama(_model_ptr=ctypes.c_void_p(1), _ctx_ptr=ctypes.c_void_p(2))
    llm._tokenizer.is_eog.side_effect = lambda t: t == EOG
    llm._fit_generation_budget = lambda n_prompt, max_new: max_new
    llm._can_reuse_kv = lambda needed: True
    llm._spec_source_name = "draft"
    llm._mtp_enabled = False
    llm._spec_draft_max = draft_max
    src = DraftModelSource(llm, ctypes.c_void_p(5), draft_max)
    src._ctx = ctypes.c_void_p(6)
    src._ctx_capacity = 4096
    src._ctx_batch = 3
    src.costs = _flat_costs()
    llm._source = src
    return llm


def _arm(llm, fake):
    llm._source._sampler = fake.draft_sampler


def _run(llm, fake, **kw):
    _arm(llm, fake)
    return _generate(llm, fake, **kw)


def _draft_tokens(fake):
    return [fake.draft_cache[p][0] for p in sorted(fake.draft_cache)]


@pytest.mark.parametrize("draft_max", [1, 2, 3, 4])
@pytest.mark.parametrize("wrong", [(), (8, 9, 14), tuple(range(7, 40, 3)), tuple(range(7, 40))])
def test_draft_model_output_matches_the_target_alone(draft_max, wrong):
    llm = _llama(draft_max)
    fake = DraftFake(llm, wrong_draft_positions=wrong)

    tokens, mock_api = _run(llm, fake, max_new_tokens=24)

    assert tokens == _reference(PROMPT, 24)
    assert fake.main_accepted == tokens
    src = llm._source
    assert src.steps > 0 and src.drafted >= src.accepted
    if not wrong:
        assert src.accepted == src.drafted
    mock_api.llama_sampler_accept.assert_not_called()
    main = PROMPT + tokens
    held = _draft_tokens(fake)
    assert held[:len(src._tokens)] == src._tokens
    assert src._tokens[:src._valid] == main[:src._valid]


def test_a_rejected_draft_is_trimmed_from_the_draft_cache_before_the_next_step():
    llm = _llama(4)
    fake = DraftFake(llm, wrong_draft_positions=(9,))

    tokens, _ = _run(llm, fake, max_new_tokens=12)

    assert tokens == _reference(PROMPT, 12)
    assert fake.draft_rm, "the rejected draft tail was never removed"
    main = PROMPT + tokens
    held = _draft_tokens(fake)
    assert held[:len(held) - llm._source.draft_max] == main[:len(held) - llm._source.draft_max]


def test_a_follow_up_turn_feeds_the_draft_model_only_the_new_tokens():
    llm = _llama(2)
    fake = DraftFake(llm)
    first, _ = _run(llm, fake, max_new_tokens=8)
    before = len(fake.draft_decodes)

    follow = PROMPT + first + [21, 22, 23]
    second, _ = _run(llm, fake, max_new_tokens=6, prompt=follow)

    assert second == _reference(follow, 6)
    first_new = fake.draft_decodes[before][0][0]
    assert first_new >= len(PROMPT) + len(first) - (llm._source.draft_max + 1)
    assert 0 not in fake.draft_rm


def test_a_new_conversation_rebuilds_the_draft_cache_from_position_zero():
    llm = _llama(2)
    fake = DraftFake(llm)
    _run(llm, fake, max_new_tokens=8)

    other = [31, 32, 33, 34]
    tokens, _ = _run(llm, fake, max_new_tokens=6, prompt=other)

    assert tokens == _reference(other, 6)
    assert 0 in fake.draft_rm
    assert _draft_tokens(fake)[:len(other)] == other


def test_a_failing_draft_decode_stops_drafting_for_the_reply_and_resets_the_cache():
    llm = _llama(3)
    calls = {"n": 0}

    def fail(index, positions):
        if positions[0] > len(PROMPT) + 3 and not calls["n"]:
            calls["n"] += 1
            return True
        return False

    fake = DraftFake(llm, fail_draft_decode=fail)
    tokens, mock_api = _run(llm, fake, max_new_tokens=16)

    assert tokens == _reference(PROMPT, 16)
    src = llm._source
    assert src.call_status == "draft-decode-failed:1"
    assert src.usable is True
    assert src._tokens == []
    mock_api.llama_memory_clear.assert_any_call(src._ctx, True)

    again, _ = _run(llm, fake, max_new_tokens=6)
    assert again == _reference(PROMPT, 6)


def test_a_catch_up_decode_that_fails_part_way_clears_what_it_wrote():
    llm = _llama(2)
    assert len(PROMPT) + 1 > llm._source._ctx_batch
    fake = DraftFake(llm, fail_draft_decode=lambda index, positions: index == 1)

    tokens, _ = _run(llm, fake, max_new_tokens=8)

    src = llm._source
    assert tokens == _reference(PROMPT, 8)
    assert (src.call_status, src.accepted) == ("draft-decode-failed:1", 0)
    assert fake.draft_cache == {}

    again, _ = _run(llm, fake, max_new_tokens=8)
    assert again == _reference(PROMPT, 8)
    assert src.call_status == "" and src.accepted > 0


def test_a_draft_cache_that_cannot_drop_a_rejected_draft_disables_the_source():
    llm = _llama(4)
    fake = DraftFake(llm, wrong_draft_positions=(8,), refuse_draft_rm=True)

    tokens, _ = _run(llm, fake, max_new_tokens=16)

    assert tokens == _reference(PROMPT, 16)
    src = llm._source
    assert (src.usable, src.status) == (False, "draft-rewind-unsupported")


def test_an_end_of_generation_draft_is_never_proposed():
    llm = _llama(4)
    fake = DraftFake(llm)
    _arm(llm, fake)
    real = fake.sample

    def sample(sampler, ctx, idx):
        out = real(sampler, ctx, idx)
        return EOG if ctx is fake.draft and fake.draft_samples % 3 == 0 else out

    fake.sample = sample
    tokens, mock_api = _generate(llm, fake, max_new_tokens=12)

    assert tokens == _reference(PROMPT, 12)
    for positions, toks in fake.main_decodes:
        if positions[0] >= len(PROMPT):
            assert EOG not in toks


def test_a_main_context_that_grew_recreates_the_draft_context_and_refills_it():
    llm = _llama(2)
    fake = DraftFake(llm)
    _arm(llm, fake)
    src = llm._source
    src._ctx_capacity = 16
    made = []

    def create(n_ctx, offload_kqv):
        made.append(n_ctx)
        src._tokens = []
        src._valid = 0
        fake.draft_cache.clear()
        src._ctx_capacity = n_ctx
        return ""

    src.create_context = create
    tokens, _ = _generate(llm, fake, max_new_tokens=10)

    assert made == [llm._ctx_capacity]
    assert tokens == _reference(PROMPT, 10)
    assert src.accepted > 0


def test_an_image_turn_reports_it_skipped_drafting():
    llm = _llama()
    llm._source.skip_call("image")
    assert llm.speculation_report()["skipped"] == "image"


def test_close_frees_the_context_the_sampler_and_the_model_once():
    from localm.inference.backends.llamacpp import llama as llama_mod
    llm = _llama()
    src = llm._source
    src._sampler = object()
    with patch.object(llama_mod, "api") as api:
        order = []
        api.llama_free.side_effect = lambda c: order.append("ctx")
        api.llama_sampler_free.side_effect = lambda s: order.append("sampler")
        api.llama_free_model.side_effect = lambda m: order.append("model")
        src.close()
        src.close()
    assert order == ["ctx", "sampler", "model"]


def test_the_draft_model_is_freed_before_the_target_model():
    from localm.inference.backends.llamacpp import llama as llama_mod
    llm = _llama()
    llm._source._sampler = object()
    with patch.object(llama_mod, "api") as api:
        order = []
        api.llama_free.side_effect = lambda c: order.append(("ctx", c.value))
        api.llama_free_model.side_effect = lambda m: order.append(("model", m.value))
        api.llama_sampler_free.side_effect = lambda s: None
        llm._free_native()
    assert order == [("ctx", 6), ("model", 5), ("ctx", 2), ("model", 1)]


@pytest.mark.parametrize("configured,recurrent,cap", [
    (None, False, 8), (6, False, 6), (None, True, 4), (2, True, 2)])
def test_the_draft_source_takes_its_own_default_and_the_recurrent_cap(configured, recurrent, cap):
    from localm.inference.backends.llamacpp import llama as llama_mod
    from localm.inference.backends.llamacpp._draftmodel import DRAFT_MODEL_DRAFT_TOKENS_DEFAULT
    llm = make_bare_llama(_model_ptr=ctypes.c_void_p(1))
    llm._spec_source_name = "draft"
    llm._mtp_enabled = False
    cp = SimpleNamespace(n_rs_seq=0)
    with patch.object(llama_mod, "api") as api:
        api.has_hybrid_api.return_value = True
        api.llama_model_is_recurrent.return_value = recurrent
        api.llama_model_is_hybrid.return_value = False
        llm._apply_initial_spec_params(cp, configured)
    assert DRAFT_MODEL_DRAFT_TOKENS_DEFAULT == 8
    assert llm._spec_draft_max == cap
    assert cp.n_rs_seq == cap


# --------------------------------------------------------------------------- #
#  Measured step costs and the draft length                                   #
# --------------------------------------------------------------------------- #

def _costs(target=1.0, verify=None, draft=0.1, draft_prefill=0.001):
    from localm.inference.backends.llamacpp._stepcosts import StepCosts
    return StepCosts(target=target, verify=verify or {2: 1.2, 3: 1.4, 5: 1.8, 9: 2.6},
                     draft=draft, draft_prefill=draft_prefill)


def test_verify_cost_interpolates_and_extends_the_measured_sizes():
    c = _costs()
    assert c.verify_cost(1) == 1.0 and c.verify_cost(3) == pytest.approx(1.4)
    assert c.verify_cost(4) == pytest.approx(1.6)
    assert c.verify_cost(7) == pytest.approx(2.2)
    assert c.verify_cost(11) == pytest.approx(3.0)
    assert c.step_cost(0) == 1.0 and c.step_cost(3) == pytest.approx(0.3 + 1.6)


@pytest.mark.parametrize("p,best", [(0.0, 0), (0.3, 0), (0.5, 1), (0.8, 3), (1.0, 8)])
def test_the_best_length_follows_the_acceptance_and_the_costs(p, best):
    assert _costs().best_length(p, 8) == best


def test_expected_tokens_is_the_geometric_sum():
    from localm.inference.backends.llamacpp._stepcosts import expected_tokens
    assert expected_tokens(0.5, 2) == pytest.approx(1.75)
    assert expected_tokens(1.0, 4) == 5.0
    assert expected_tokens(0.0, 4) == 1.0


def test_a_draft_slower_than_the_target_cannot_pay():
    cpu_draft = _costs(target=0.008, verify={2: 0.0103, 3: 0.0127, 5: 0.0163},
                       draft=0.025)
    gpu_draft = _costs(target=0.008, verify={2: 0.0103, 3: 0.0127, 5: 0.0163},
                       draft=0.0032)
    assert cpu_draft.can_pay(4) is False and cpu_draft.best_length(1.0, 4) == 0
    assert gpu_draft.can_pay(4) is True


def test_paying_is_judged_at_the_given_acceptance():
    marginal = _costs(draft=0.65)
    assert marginal.can_pay(8) is True
    assert marginal.can_pay(8, 0.85) is False
    assert _costs(draft=0.3).can_pay(8, 0.85) is True


@pytest.mark.parametrize("probe,top,plan", [
    (0.01, 9, (3, 5, [2, 3, 5, 9])), (0.02, 5, (3, 5, [2, 3, 5])),
    (0.01, 17, (3, 5, [2, 3, 5, 9, 17])), (0.01, 12, (3, 5, [2, 3, 5, 9, 12])),
    (0.03, 9, (1, 3, [2, 3, 5, 9])), (0.05, 5, (1, 3, [2, 3, 5])),
    (0.06, 9, (1, 1, [2, 9])), (2.0, 2, (1, 1, [2])), (0.01, 1, (1, 1, []))])
def test_a_slow_target_is_measured_with_fewer_decodes(probe, top, plan):
    from localm.inference.backends.llamacpp._stepcosts import measure_plan
    assert measure_plan(probe, top) == plan


def _measured_llama(draft_max=8, costs=None):
    llm = _llama(draft_max)
    llm._source.costs = costs if costs is not None else _costs()
    return llm


def test_acceptance_starts_at_the_prior_and_follows_the_verify_results():
    llm = _measured_llama()
    src = llm._source
    assert src.acceptance() == pytest.approx(0.6)
    for _ in range(10):
        src.on_verify(4, 4)
    high = src.acceptance()
    for _ in range(30):
        src.on_verify(4, 0)
    assert high > 0.9 and src.acceptance() < 0.3


def _steps(src, n, outcome):
    """Run *n* budget steps at position 20, verifying each drafting step with
    ``outcome(i, k, after_full_accept)`` drafts accepted; the lengths."""
    lengths = []
    for i in range(n):
        k = src.budget(20, None)
        lengths.append(k)
        if k:
            src.on_verify(k, outcome(i, k, src._chosen_hot))
    return lengths


def test_the_draft_length_rises_falls_and_recovers_with_acceptance():
    llm = _measured_llama()
    src = llm._source
    llm._cached_tokens = list(range(20))
    src._tokens = list(range(20))
    lengths = _steps(src, 400, lambda i, k, hot: k if i < 40 or i >= 100 else 0)
    assert max(lengths[:40]) == 8
    assert 0 in lengths[40:100]
    assert all(k > 0 for k in lengths[-20:])


def _marginal_llama():
    marginal = _costs(target=5.78, verify={2: 6.29, 3: 7.0, 5: 8.5}, draft=3.48)
    llm = _measured_llama(draft_max=4, costs=marginal)
    llm._cached_tokens = list(range(20))
    llm._source._tokens = list(range(20))
    return llm


def test_a_draft_that_the_prior_rejects_still_probes_and_backs_off():
    from localm.inference.backends.llamacpp._stepcosts import (
        ACCEPTANCE_PROBE_EVERY, ACCEPTANCE_PROBE_MAX_EVERY, ACCEPTANCE_PROBE_P)
    assert (ACCEPTANCE_PROBE_EVERY, ACCEPTANCE_PROBE_MAX_EVERY) == (32, 256)
    llm = _marginal_llama()
    assert llm._source.costs.best_length(0.6, 4) == 0
    assert llm._source.costs.best_length(ACCEPTANCE_PROBE_P, 4) > 0
    lengths = _steps(llm._source, 720, lambda i, k, hot: 0)
    assert [i for i, k in enumerate(lengths) if k] == [0, 65, 194, 451, 708]


def test_a_probe_without_a_rejection_resets_the_back_off():
    llm = _marginal_llama()
    accept_at = {65}
    lengths = _steps(llm._source, 170, lambda i, k, hot: k if i in accept_at else 0)
    assert [i for i, k in enumerate(lengths) if k] == [0, 65, 98, 163]


def test_a_probe_after_which_drafting_pays_resets_the_back_off_and_a_reply_restarts_it():
    from localm.inference.backends.llamacpp._stepcosts import ACCEPTANCE_PROBE_EVERY
    costs = _costs(verify={2: 1.581, 3: 2.0, 5: 2.6, 9: 4.6}, draft=0.0)
    assert costs.best_length(0.6, 8) == 0 and costs.best_length(0.7, 8) > 0
    llm = _measured_llama(draft_max=8, costs=costs)
    src = llm._source
    llm._cached_tokens = list(range(20))
    src._tokens = list(range(20))
    k = src.budget(20, None)
    assert k == 4 and src._probing
    src.on_verify(k, k - 1)
    assert src._probe_every == ACCEPTANCE_PROBE_EVERY
    src._probe_every = 4 * ACCEPTANCE_PROBE_EVERY
    src.begin_call()
    assert src._probe_every == ACCEPTANCE_PROBE_EVERY


def test_a_full_accept_does_not_carry_into_the_next_reply_or_past_a_pause():
    llm = _measured_llama()
    src = llm._source
    src.on_verify(4, 4)
    assert src._hot is True
    src.begin_call()
    assert src._hot is False
    src.on_verify(4, 4)
    src.on_paused_step()
    assert src._hot is False


def test_the_steps_held_back_are_counted_per_reply():
    llm = _marginal_llama()
    src = llm._source
    src.begin_call()
    lengths = [src.budget(20, None) for _ in range(5)]
    src.on_verify(lengths[0], 0)
    lengths += [src.budget(20, None) for _ in range(5)]
    assert src.held_steps == lengths.count(0) == 5
    assert src.report()["held_steps"] == 5
    src.begin_call()
    assert src.held_steps == 0


def test_a_probe_that_drafts_nothing_is_tried_again_on_the_next_step():
    from localm.inference.backends.llamacpp._stepcosts import ACCEPTANCE_PROBE_EVERY
    llm = _marginal_llama()
    src = llm._source
    assert [src.budget(20, None) > 0 for _ in range(3)] == [True, True, True]
    src.on_verify(2, 0)
    assert [src.budget(20, None) > 0 for _ in range(ACCEPTANCE_PROBE_EVERY)] == (
        [False] * ACCEPTANCE_PROBE_EVERY)


def test_a_probe_that_finds_high_acceptance_keeps_drafting():
    llm = _marginal_llama()
    lengths = _steps(llm._source, 120, lambda i, k, hot: k)
    assert lengths[0] > 0
    assert all(k > 0 for k in lengths[-60:])


def test_the_estimate_after_a_full_accept_starts_at_the_other_one():
    llm = _measured_llama()
    src = llm._source
    assert src.acceptance(True) == pytest.approx(src.acceptance()) == pytest.approx(0.6)
    src.on_verify(4, 4)
    assert src.acceptance() == pytest.approx(5.2 / 6)
    assert src.acceptance(True) == pytest.approx(src.acceptance())


def test_runs_of_full_accepts_draft_longer_than_other_steps():
    llm = _measured_llama()
    src = llm._source
    llm._cached_tokens = list(range(20))
    src._tokens = list(range(20))
    run = {"hot": 0, "cold": 0}

    def outcome(i, k, hot):
        if hot:
            run["hot"] += 1
            return k if run["hot"] % 6 else 0
        run["cold"] += 1
        return k if run["cold"] % 3 == 0 else 0

    _steps(src, 600, outcome)
    assert src.acceptance(True) > src.acceptance() + 0.2
    src._since_probe = 0
    src._hot = True
    after_accept = src.budget(20, None)
    other = src.budget(20, None)
    assert after_accept > other


def test_a_held_step_after_a_full_accept_ages_that_estimate():
    llm = _measured_llama()
    src = llm._source
    llm._cached_tokens = list(range(20))
    src._tokens = list(range(20))
    for _ in range(10):
        src.on_verify(8, 8)
    src._evidence[True] = [0.0, 6.0]
    low = src.acceptance(True)
    assert low < 0.3 and src.acceptance() > 0.9
    assert src._hot is True
    assert src.budget(20, None) == 0
    assert low < src.acceptance(True) < src.acceptance()


def test_without_measured_costs_a_step_drafts_at_most_two():
    llm = _llama(8)
    llm._source.costs = None
    assert llm._source.budget(10, None) == 2
    assert llm._source.budget(10, 1) == 1


def test_a_long_catch_up_on_a_slow_draft_skips_drafting_for_the_reply():
    slow_prefill = _costs(draft_prefill=0.5)
    llm = _measured_llama(costs=slow_prefill)
    src = llm._source
    for _ in range(20):
        src.on_verify(8, 8)
    llm._cached_tokens = list(range(400))
    assert src.budget(400, 100) == 0
    src._tokens = list(range(396))
    assert src.budget(400, 100) > 0
    assert src._valid == 396


def test_observed_step_times_correct_the_measured_costs():
    llm = _measured_llama(costs=_costs(draft=0.1))
    src = llm._source
    assert src.step_cost(3) == pytest.approx(0.3 + 1.6)
    src.on_step_seconds(3, 1.9 + 0.35)
    assert src.step_cost(3) == pytest.approx(1.9 + 0.2 * 0.35)
    assert src.drafting_overhead() == pytest.approx((0.2 * 0.35, 0.0))
    assert src.step_cost(1) == pytest.approx(0.1 + 1.2 + 0.2 * 0.35)
    for _ in range(60):
        src.on_step_seconds(3, 1.9 + 0.35)
        src.on_step_seconds(0, 1.0 + 0.4)
    assert src.step_cost(3) == pytest.approx(1.9 + 0.35)
    assert src.step_cost(0) == pytest.approx(1.4)
    assert src._step_over_s == pytest.approx(0.4, rel=1e-4)
    assert src.drafting_overhead() == (0.0, 0.0)
    assert src.step_cost(8) == pytest.approx(0.8 + 2.6 + 0.4, rel=1e-4)


def test_the_drafting_overhead_is_fitted_as_fixed_plus_per_draft():
    llm = _measured_llama(costs=_costs(draft=0.1))
    src = llm._source
    for _ in range(80):
        src.on_step_seconds(0, 1.0 + 0.1)
        src.on_step_seconds(2, 0.2 + 1.4 + 0.1 + 0.6)
    assert src._step_over_s == pytest.approx(0.1)
    assert src.drafting_overhead() == pytest.approx((0.6, 0.0))
    assert src.step_cost(4) == pytest.approx(0.4 + 1.8 + 0.1 + 0.6)
    for _ in range(80):
        src.on_step_seconds(4, 0.4 + 1.8 + 0.1 + 1.2)
    assert src.drafting_overhead() == pytest.approx((0.0, 0.3), abs=1e-6)
    assert src.step_cost(6) == pytest.approx(0.6 + 2.2 + 0.1 + 6 * 0.3)


def test_a_plain_steps_overhead_does_not_price_a_moe_target_out_of_drafting():
    from localm.inference.backends.llamacpp._stepcosts import ACCEPTANCE_PROBE_P, best_length
    moe = _costs(target=0.0074, verify={2: 0.0096, 3: 0.0121, 5: 0.0206, 9: 0.0296},
                 draft=0.0033)
    llm = _measured_llama(costs=moe)
    src = llm._source
    for _ in range(40):
        src.on_step_seconds(0, 0.0085)
    assert src.step_cost(1) == pytest.approx(0.0033 + 0.0096 + 0.0011, rel=1e-3)
    assert best_length(ACCEPTANCE_PROBE_P, 8, src.step_cost) > 0


def test_a_step_faster_than_measured_never_makes_an_unseen_length_cheaper():
    llm = _measured_llama(costs=_costs())
    src = llm._source
    for _ in range(30):
        src.on_step_seconds(2, 0.5)
        src.on_step_seconds(0, 0.5)
    assert src._step_over_s < 0 and src.drafting_overhead() == (0.0, 0.0)
    assert src.step_cost(2) < 1.4
    assert src.step_cost(5) == pytest.approx(0.5 + 2.0)


def test_step_times_are_ignored_without_measured_costs_or_a_time():
    llm = _llama(4)
    src = llm._source
    src.costs = None
    src.on_step_seconds(2, 5.0)
    src.costs = _costs()
    src.on_step_seconds(2, 0.0)
    assert src._observed == {} and src._step_over_s == 0.0


def test_a_fixed_drafting_overhead_seen_at_short_lengths_does_not_price_out_long_drafts():
    from localm.inference.backends.llamacpp._stepcosts import best_length
    moe = _costs(target=0.0074, verify={2: 0.0097, 3: 0.0121, 5: 0.0207, 9: 0.0294}, draft=0.0)
    llm = _measured_llama(costs=moe)
    src = llm._source
    for _ in range(40):
        src.on_step_seconds(0, 0.0077)
        src.on_step_seconds(1, 0.0115)
        src.on_step_seconds(2, 0.0138)
    fixed, per_draft = src.drafting_overhead()
    assert per_draft == 0.0 and fixed == pytest.approx(0.00145, abs=1e-4)
    assert best_length(0.95, 8, src.step_cost) == 8


def test_a_step_with_a_long_catch_up_is_not_timed():
    llm = _measured_llama(costs=_costs())
    src = llm._source
    src._catch_up = 40
    src.on_step_seconds(2, 9.0)
    assert src._observed == {}
    src._catch_up = 2
    src.on_step_seconds(2, 1.5)
    assert 2 in src._observed and src._catch_up == 0


@pytest.mark.parametrize("clear", ["unrecorded step", "new reply"])
def test_a_catch_up_count_never_outlives_its_step(clear):
    llm = _measured_llama(costs=_costs())
    src = llm._source
    src._catch_up = 40
    if clear == "unrecorded step":
        src.on_step_seconds(None, 9.0)
    else:
        src.begin_call()
    src.on_step_seconds(0, 1.2)
    assert 0 in src._observed


def test_one_outlying_step_moves_a_figure_at_most_a_fifth_of_the_way_to_three_times_it():
    llm = _measured_llama(costs=_costs())
    src = llm._source
    src.on_step_seconds(0, 100.0)
    assert src.step_cost(0) == pytest.approx(1.0 + 0.2 * (3.0 - 1.0))
    src.on_step_seconds(0, 0.001)
    assert src.step_cost(0) == pytest.approx(1.4 + 0.2 * (1.4 / 3 - 1.4))
    assert src._step_over_s == pytest.approx(0.2 * 2.0 + 0.2 * (1.4 / 3 - 1.0 - 0.4))


def test_a_measured_verify_curve_never_falls_as_the_batch_grows():
    noisy = _costs(target=1.0, verify={2: 3.0, 3: 2.0, 5: 4.0})
    assert noisy.verify_cost(2) == 3.0
    assert noisy.verify_cost(3) == 3.0
    assert noisy.verify_cost(4) == pytest.approx(3.5)
    assert noisy.verify == {2: 3.0, 3: 2.0, 5: 4.0}


def test_a_draft_context_the_next_proposal_recreates_counts_as_empty():
    llm = _measured_llama(costs=_costs(draft_prefill=0.5))
    src = llm._source
    for _ in range(20):
        src.on_verify(8, 8)
    llm._cached_tokens = list(range(150))
    src._tokens = list(range(150))
    llm._ctx_capacity = src._ctx_capacity * 2
    assert src.budget(150, 100) == 0
    src._ctx_capacity = llm._ctx_capacity
    assert src.budget(150, 100) > 0


def test_the_loop_reports_each_steps_time_and_drafts_to_the_source():
    llm = _measured_llama(draft_max=3, costs=_flat_costs())
    fake = DraftFake(llm, main_cost=1.0, row_cost=0.1, draft_cost=0.2)
    seen = []
    src = llm._source
    record = src.on_step_seconds
    src.on_step_seconds = lambda k, s: (seen.append((k, s)), record(k, s))
    tokens, _ = _run(llm, fake, max_new_tokens=40)
    assert tokens == _reference(PROMPT, 40)
    first, rest = seen[0], seen[1:]
    assert first == (3, pytest.approx(5 * 0.2 + 1.0 + 0.1 * 3))
    plain = [s for k, s in rest if k == 0]
    drafting = [(k, s) for k, s in rest if k > 0]
    assert plain and drafting
    assert all(s == pytest.approx(1.0) for s in plain)
    for k, s in drafting:
        assert s == pytest.approx(k * 0.2 + 1.0 + 0.1 * k)
    assert sum(k for k, _ in seen if k) <= src.drafted
    replay = _measured_llama(draft_max=3, costs=_flat_costs())._source
    for k, s in rest:
        replay.on_step_seconds(k, s)
    assert src._observed == pytest.approx(replay._observed)
    assert src._step_over_s == pytest.approx(replay._step_over_s)
    assert src.drafting_overhead() == pytest.approx(replay.drafting_overhead())


@pytest.mark.parametrize("wrong", [(), (8, 9, 14), tuple(range(7, 40, 3))])
def test_adaptive_draft_lengths_keep_the_output_of_the_target_alone(wrong):
    llm = _measured_llama()
    fake = DraftFake(llm, wrong_draft_positions=wrong)
    tokens, _ = _run(llm, fake, max_new_tokens=24)
    assert tokens == _reference(PROMPT, 24)
    assert llm._source.steps > 0


def test_the_step_costs_are_measured_on_both_contexts():
    from localm.inference.backends.llamacpp import llama as llama_mod
    llm = _llama(4)
    fake = DraftFake(llm, main_cost=1.0, row_cost=0.1, draft_cost=0.2)
    with patch.object(llama_mod, "api") as api:
        fake.install(api)
        api.llama_vocab_n_tokens.return_value = 1000
        costs = llm._measure_step_costs(llm._source)
    assert costs.target == pytest.approx(1.0)
    assert sorted(costs.verify) == [2, 5]
    assert costs.verify[5] == pytest.approx(1.4)
    assert costs.draft == pytest.approx(0.2)
    assert costs.draft_prefill == pytest.approx(0.2 / 64)
    assert fake.main_cache == {} and fake.draft_cache == {}
    assert llm._source._tokens == []


def test_a_fast_target_is_measured_at_every_verify_size():
    from localm.inference.backends.llamacpp import llama as llama_mod
    llm = _llama(4)
    fake = DraftFake(llm, main_cost=0.01, row_cost=0.001, draft_cost=0.002)
    with patch.object(llama_mod, "api") as api:
        fake.install(api)
        api.llama_vocab_n_tokens.return_value = 1000
        costs = llm._measure_step_costs(llm._source)
    assert sorted(costs.verify) == [2, 3, 5]
    assert costs.verify[3] == pytest.approx(0.012)
    timed_main = [d for d in fake.main_decodes if d[0][0] == 32]
    assert len(timed_main) == 2 + 8 * 4


def test_a_target_measured_at_no_time_gives_no_costs():
    from localm.inference.backends.llamacpp import llama as llama_mod
    llm = _llama(4)
    fake = DraftFake(llm, main_cost=0.0, row_cost=0.0, draft_cost=0.002)
    with patch.object(llama_mod, "api") as api:
        fake.install(api)
        api.llama_vocab_n_tokens.return_value = 1000
        assert llm._measure_step_costs(llm._source) is None
    assert fake.main_cache == {} and fake.draft_cache == {}


def test_an_ngram_source_is_measured_on_the_target_alone():
    from localm.inference.backends.llamacpp import llama as llama_mod
    from localm.inference.backends.llamacpp._ngram import NgramSource
    llm = _llama(4)
    llm._spec_source_name = "ngram"
    llm._source = NgramSource(llm, draft_max=4)
    fake = FakeNative(llm, main_cost=0.01, row_cost=0.001)
    with patch.object(llama_mod, "api") as api:
        fake.install(api)
        api.llama_vocab_n_tokens.return_value = 1000
        costs = llm._measure_step_costs(llm._source)
    assert costs.target == pytest.approx(0.01)
    assert costs.verify[5] == pytest.approx(0.014)
    assert (costs.draft, costs.draft_prefill) == (0.0, 0.0)
    assert fake.main_cache == {}


def test_a_failed_measurement_leaves_no_costs():
    from localm.inference.backends.llamacpp import llama as llama_mod
    llm = _llama(4)
    fake = DraftFake(llm, fail_draft_decode=lambda index, positions: True)
    with patch.object(llama_mod, "api") as api:
        fake.install(api)
        api.llama_vocab_n_tokens.return_value = 1000
        assert llm._measure_step_costs(llm._source) is None


@pytest.mark.parametrize("costs,status", [
    ("pays", "ok"), ("cannot", "draft-cannot-pay"), ("marginal", "draft-cannot-pay"),
    (None, "ok")])
def test_a_draft_model_that_cannot_pay_is_freed_at_load(costs, status):
    llm = _llama(8)
    src = llm._source
    src.costs = None
    measured = {"pays": _costs(), "cannot": _costs(draft=5.0),
                "marginal": _costs(draft=0.65), None: None}[costs]
    llm._cache_can_drop_a_speculative_token = lambda: True
    llm._load_draft_model = lambda *a, **kw: None
    llm._measure_step_costs = lambda source: measured
    freed = []
    src.close = lambda: freed.append(True)
    llm._set_up_spec_source("d.gguf", None, True, True)
    refused = status == "draft-cannot-pay"
    assert (src.status, src.usable) == (status, not refused)
    assert src.costs is measured
    assert bool(freed) is refused


@pytest.mark.parametrize("costs,status", [
    ("pays", "ok"), ("high-only", "ok"), ("marginal", "ngram-cannot-pay"),
    ("cannot", "ngram-cannot-pay"), (None, "ok")])
def test_ngram_is_measured_at_load_and_turned_off_unless_it_pays_at_the_probe_acceptance(
        costs, status):
    from localm.inference.backends.llamacpp._ngram import NgramSource
    llm = make_bare_llama(_model_ptr=ctypes.c_void_p(1), _ctx_ptr=ctypes.c_void_p(2))
    llm._spec_source_name = "ngram"
    llm._spec_draft_max = 8
    measured = {"pays": _costs(draft=0.0), "high-only": _costs(verify={2: 1.6, 9: 6.0}, draft=0.0),
                "marginal": _costs(verify={2: 1.85, 9: 8.5}, draft=0.0),
                "cannot": _costs(verify={2: 2.0, 9: 9.0}, draft=0.0), None: None}[costs]
    llm._cache_can_drop_a_speculative_token = lambda: True
    llm._measure_step_costs = lambda source: measured
    llm._set_up_spec_source(None, None, True, True)
    src = llm._source
    assert isinstance(src, NgramSource)
    assert (src.status, src.usable) == (status, status == "ok")
    assert src.costs is measured


# --------------------------------------------------------------------------- #
#  Loading the draft model                                                    #
# --------------------------------------------------------------------------- #

def _loading_llama(tmp_path):
    from localm.inference.backends.llamacpp import llama as llama_mod
    llm = make_bare_llama(_model_ptr=ctypes.c_void_p(1), _ctx_ptr=ctypes.c_void_p(2))
    llm._spec_source_name = "draft"
    llm._spec_draft_max = 4
    llm._main_gpu_index = 0
    path = tmp_path / "draft.gguf"
    path.write_bytes(b"GGUF")
    return llm, str(path), llama_mod


def _view_api(api, target_type=2, draft_type=2):
    def vocab(model):
        return model
    api.llama_model_get_vocab.side_effect = vocab
    api.llama_vocab_type.side_effect = lambda v: target_type if v.value == 1 else draft_type
    api.llama_vocab_n_tokens.return_value = 10
    api.llama_vocab_get_add_bos.return_value = False
    api.llama_vocab_get_add_eos.return_value = False
    api.llama_vocab_bos.return_value = 1
    api.llama_vocab_eos.return_value = 2
    api._bind.return_value = lambda v, i: b"t%d" % i


@pytest.mark.parametrize("case,status", [
    ("missing", "draft-model-missing"), ("load-failed", "draft-load-failed"),
    ("vocab", "draft-vocab-mismatch"), ("recurrent", "draft-rewind-unsupported"),
    ("context", "draft-context-refused"), ("ok", "ok")])
def test_each_draft_load_outcome_is_reported_and_frees_what_it_loaded(tmp_path, case, status):
    llm, path, llama_mod = _loading_llama(tmp_path)
    with patch.object(llama_mod, "api") as api, \
         patch.object(llama_mod, "set_use_mmap"):
        _view_api(api, draft_type=3 if case == "vocab" else 2)
        api.llama_load_model_from_file.return_value = (
            None if case == "load-failed" else ctypes.c_void_p(5))
        api.has_hybrid_api.return_value = True
        api.llama_model_is_recurrent.return_value = case == "recurrent"
        api.llama_model_is_hybrid.return_value = False
        api.llama_context_default_params.side_effect = lambda: SimpleNamespace()
        api.llama_init_from_model.return_value = (
            None if case == "context" else ctypes.c_void_p(6))
        llm._load_draft_model(path + (".nope" if case == "missing" else ""), None, True)
        src = llm._source
        freed = [c.args[0].value for c in api.llama_free_model.call_args_list]
    assert src.status == status
    assert src.usable is (case == "ok")
    if case in ("vocab", "recurrent", "context"):
        assert freed == [5], "the draft model must be freed when it cannot be used"
        assert src._model is None
    if case == "ok":
        assert freed == [] and src._ctx.value == 6
        assert src._ctx_capacity == llm._ctx_capacity


@pytest.mark.parametrize("source,can_drop,loads,status", [
    ("draft", True, True, "ok"), ("draft", False, False, "rewind-unsupported"),
    ("ngram", True, False, "ok"), ("ngram", False, False, "rewind-unsupported")])
def test_the_main_cache_is_probed_before_a_draft_model_is_loaded(source, can_drop, loads, status):
    llm = make_bare_llama(_model_ptr=ctypes.c_void_p(1), _ctx_ptr=ctypes.c_void_p(2))
    llm._spec_source_name = source
    llm._spec_draft_max = 2
    order = []
    llm._cache_can_drop_a_speculative_token = lambda: order.append("probe") or can_drop
    llm._load_draft_model = lambda *a, **kw: order.append(("load", kw.get("on_gpu")))
    llm._measure_step_costs = lambda source: order.append("measure")

    llm._set_up_spec_source("d.gguf", None, False, False)

    expected = ["probe"] + ([("load", False)] if loads else [])
    if source == "ngram" and can_drop:
        expected.append("measure")
    assert order == expected
    src = llm._source
    assert src.name == source and src.status == status and src.usable is (status == "ok")
    if source == "draft" and not can_drop:
        assert src._model is None and src._ctx is None


@pytest.mark.parametrize("case", ["ok", "context", "vocab", "cancel"])
def test_native_draft_context_and_free_calls_run_with_stderr_quieted(tmp_path, case):
    import contextlib
    import threading

    from localm.inference.backends.base import ModelLoadCancelled
    llm, path, llama_mod = _loading_llama(tmp_path)
    quieted = []
    depth = {"n": 0}

    @contextlib.contextmanager
    def quiet():
        depth["n"] += 1
        try:
            yield
        finally:
            depth["n"] -= 1

    def record(name, result=None):
        def call(*a, **kw):
            quieted.append((name, depth["n"] > 0))
            return result
        return call

    if case == "cancel":
        llm._cancel_event = threading.Event()
        llm._cancel_event.set()
    with patch.object(llama_mod, "api") as api, \
         patch.object(llama_mod, "set_use_mmap"), \
         patch.object(llama_mod, "_quiet_stderr", quiet), \
         patch("localm.discover.apply_gpu_split", return_value=None):
        _view_api(api, draft_type=3 if case == "vocab" else 2)
        api.llama_load_model_from_file.return_value = (
            None if case == "cancel" else ctypes.c_void_p(5))
        api.has_hybrid_api.return_value = False
        api.llama_context_default_params.side_effect = lambda: SimpleNamespace()
        api.llama_init_from_model.side_effect = record(
            "init", None if case == "context" else ctypes.c_void_p(6))
        api.llama_free_model.side_effect = record("free_model")
        api.llama_free.side_effect = record("free_ctx")
        if case == "cancel":
            with pytest.raises(ModelLoadCancelled):
                llm._load_draft_model(path, None, False)
        else:
            llm._load_draft_model(path, None, False)
    assert quieted, "no native call was recorded"
    assert all(inside for _name, inside in quieted), quieted
    if case == "ok":
        assert [n for n, _ in quieted] == ["init"]


def test_a_cancelled_draft_load_frees_the_target_and_raises(tmp_path):
    import threading

    from localm.inference.backends.base import ModelLoadCancelled
    llm, path, llama_mod = _loading_llama(tmp_path)
    llm._cancel_event = threading.Event()
    llm._cancel_event.set()
    with patch.object(llama_mod, "api") as api, \
         patch.object(llama_mod, "set_use_mmap"):
        api.llama_load_model_from_file.return_value = None
        with pytest.raises(ModelLoadCancelled) as caught:
            llm._load_draft_model(path, None, True)
        freed_models = [c.args[0].value for c in api.llama_free_model.call_args_list]
    assert isinstance(caught.value, ModelLoadCancelled)
    assert freed_models == [1]
    assert (llm._model_ptr, llm._ctx_ptr) == (None, None)


def test_a_draft_file_that_is_not_a_chat_model_is_never_loaded(tmp_path):
    llm, _path, llama_mod = _loading_llama(tmp_path)
    head = _vocab_gguf(tmp_path / "head.gguf", ["x%d" % i for i in range(40)], arch="eagle3")
    with patch.object(llama_mod, "api") as api:
        llm._load_draft_model(str(head), None, True)
    assert (llm._source.status, llm._source.usable) == ("draft-unsupported-role", False)
    api.llama_load_model_from_file.assert_not_called()


@pytest.mark.parametrize("main_gpu,ratios", [(None, None), (1, [(0, 0.6), (1, 0.4)])])
def test_the_draft_model_is_split_over_the_targets_devices(tmp_path, main_gpu, ratios):
    llm, path, llama_mod = _loading_llama(tmp_path)
    llm._main_gpu_arg = main_gpu
    llm._gpu_split_ratios_arg = ratios
    params = SimpleNamespace(n_gpu_layers=0, split_mode=1, main_gpu=0)
    with patch.object(llama_mod, "api") as api, \
         patch.object(llama_mod, "set_use_mmap") as mmap, \
         patch("localm.discover.apply_main_gpu") as main, \
         patch("localm.discover.apply_gpu_split", return_value=None) as split:
        api.llama_model_default_params.return_value = params
        api.llama_load_model_from_file.return_value = None
        llm._load_draft_model(path, None, True)
    assert params.n_gpu_layers == 99
    assert not hasattr(params, "devices")
    if main_gpu is None:
        main.assert_called_once_with(params)
    else:
        main.assert_called_once_with(params, slot=main_gpu)
    split.assert_called_once_with(params, ratios_override=ratios)
    mmap.assert_called_once_with(params, False)


def test_a_cpu_draft_model_is_not_split_and_gets_no_gpu_device(tmp_path):
    llm, path, llama_mod = _loading_llama(tmp_path)
    llm._gpu_split_ratios_arg = [(0, 0.5), (1, 0.5)]
    params = SimpleNamespace(n_gpu_layers=-1, split_mode=1, main_gpu=0, devices=None)
    seen = {}

    def load(p, mp):
        device_list = ctypes.cast(mp.devices, ctypes.POINTER(ctypes.c_void_p))
        seen["first"] = device_list[0]
        return None
    with patch.object(llama_mod, "api") as api, \
         patch.object(llama_mod, "set_use_mmap"), \
         patch("localm.discover.apply_main_gpu") as main, \
         patch("localm.discover.apply_gpu_split") as split:
        api.llama_model_default_params.return_value = params
        api.llama_load_model_from_file.side_effect = load
        llm._load_draft_model(path, None, True, on_gpu=False)
    assert params.n_gpu_layers == 0
    assert params.devices and seen == {"first": None}
    main.assert_not_called()
    split.assert_not_called()


# --------------------------------------------------------------------------- #
#  VRAM charge                                                                #
# --------------------------------------------------------------------------- #

def test_the_draft_model_charge_is_weights_kv_logits_and_margin(tmp_path):
    from localm.inference.backends.gguf import GgufBackend
    from localm.inference.backends.llamacpp._split_fit import logits_buffer_bytes
    draft = tmp_path / "d.gguf"
    draft.write_bytes(b"x" * 4096)
    b = GgufBackend(str(tmp_path / "m.gguf"), spec_source="draft",
                    spec_draft_model=str(draft), n_ctx=2048)
    with patch("localm.model_manager.gguf.gguf_kv_bytes_per_token", return_value=1000), \
         patch("localm.model_manager.gguf._gguf_split_layout_meta", return_value=(24, 151936)):
        charge = b._draft_model_vram_bytes()
        per_token = b._spec_kv_per_token()
    assert charge == (4096 + 2048 * 1000
                      + logits_buffer_bytes(151936, 2048, max_batch=DRAFT_CONTEXT_BATCH)
                      + DRAFT_COMPUTE_MARGIN_BYTES)
    assert per_token == 1000
    assert b._spec_extra_vram_bytes() == charge


def test_no_draft_charge_without_the_draft_source_or_a_file(tmp_path):
    from localm.inference.backends.gguf import GgufBackend
    draft = tmp_path / "d.gguf"
    draft.write_bytes(b"x" * 4096)
    assert GgufBackend(str(tmp_path / "m.gguf"), spec_source="ngram",
                       spec_draft_model=str(draft))._draft_model_vram_bytes() == 0
    missing = GgufBackend(str(tmp_path / "m.gguf"), spec_source="draft",
                          spec_draft_model=str(tmp_path / "gone.gguf"))
    assert missing._draft_model_vram_bytes() == 0


GiB = 1024 ** 3
_TARGET, _KV, _OVERHEAD, _CHARGE = 8 * GiB, 1000, 512 * 1024 * 1024, 1 * GiB


def _placed(tmp_path, free, *, n_gpu_layers=99, layer_count=48, split=None):
    from localm.inference.backends.gguf import GgufBackend
    draft = tmp_path / "d.gguf"
    draft.write_bytes(b"x")
    b = GgufBackend(str(tmp_path / "m.gguf"), spec_source="draft",
                    spec_draft_model=str(draft), n_ctx=2048, n_gpu_layers=n_gpu_layers)
    with patch.object(GgufBackend, "_draft_model_charge_bytes", return_value=_CHARGE), \
         patch.object(GgufBackend, "_split_free_total_bytes", return_value=(free, None, 1)), \
         patch.object(GgufBackend, "_free_vram_bytes", return_value=None), \
         patch.object(GgufBackend, "_vram_model_bytes", return_value=_TARGET), \
         patch.object(GgufBackend, "_kv_bytes_per_token", return_value=_KV), \
         patch.object(GgufBackend, "_split_overhead_bytes", return_value=_OVERHEAD), \
         patch.object(GgufBackend, "_mtp_draft_context_vram_bytes", return_value=0), \
         patch.object(GgufBackend, "_recurrent_state_vram_bytes", return_value=0), \
         patch.object(GgufBackend, "_cached_layer_count", return_value=layer_count), \
         patch.object(GgufBackend, "_implicit_split_fit", side_effect=split or (lambda *a: None)):
        on_gpu = b._decide_draft_placement()
        extra = b._spec_extra_vram_bytes()
    return b, on_gpu, extra


def test_the_draft_model_goes_on_the_gpu_only_beside_the_whole_target(tmp_path):
    need = _TARGET + 2048 * _KV + _OVERHEAD + _CHARGE
    b, on_gpu, extra = _placed(tmp_path, need)
    assert (on_gpu, b.draft_model_on_gpu, extra) == (True, True, _CHARGE)
    b, on_gpu, extra = _placed(tmp_path, need - 1)
    assert (on_gpu, b.draft_model_on_gpu, extra) == (False, False, 0)


@pytest.mark.parametrize("on_gpu", [True, False])
def test_the_draft_kv_grows_the_gpu_charge_only_while_the_draft_is_on_the_gpu(tmp_path, on_gpu):
    from localm.inference.backends.gguf import GgufBackend
    b = GgufBackend(str(tmp_path / "m.gguf"), spec_source="draft",
                    spec_draft_model=str(tmp_path / "d.gguf"))
    b.draft_model_on_gpu = on_gpu
    b._draft_kv_per_token_cached = 1000
    with patch.object(GgufBackend, "_draft_model_charge_bytes", return_value=_CHARGE), \
         patch.object(GgufBackend, "_mtp_draft_kv_per_token", return_value=0):
        assert b._spec_kv_per_token() == (1000 if on_gpu else 0)


def test_a_partial_target_counts_only_its_gpu_layers(tmp_path):
    need = _TARGET // 2 + 2048 * _KV + _OVERHEAD + _CHARGE
    assert _placed(tmp_path, need, n_gpu_layers=24)[1] is True
    assert _placed(tmp_path, need - 1, n_gpu_layers=24)[1] is False
    assert _placed(tmp_path, 10 ** 15, n_gpu_layers=0)[1] is False


def _two_gpus(draft_charge):
    from localm.inference.backends.llamacpp._split_fit import plan_split
    devices = [{"index": 0, "free": 14 * GiB}, {"index": 1, "free": 14 * GiB}]
    return plan_split(devices, layer_bytes=[2 * GiB] * 10, layer_kv_bytes=[0] * 10,
                      output_bytes=0, n_gpu_layers=99, logits_bytes=0,
                      reserve_bytes=0, spread_bytes=draft_charge)


def test_the_draft_is_charged_across_split_devices_by_share():
    from localm.inference.backends.llamacpp._split_fit import charge_devices
    devices = [{"index": 0, "free": 30 * GiB}, {"index": 1, "free": 10 * GiB}]
    charges = charge_devices(devices, [3.0, 1.0], layer_bytes=[GiB] * 8,
                             layer_kv_bytes=[0] * 8, output_bytes=0, n_gpu_layers=99,
                             logits_bytes=0, reserve_bytes=7, spread_bytes=4 * GiB)
    assert [c.reserve for c in charges] == [7 + 3 * GiB, 7 + GiB]
    assert all(type(c.reserve) is int for c in charges)
    idle = charge_devices(devices, [1.0, 0.0], layer_bytes=[GiB] * 8,
                          layer_kv_bytes=[0] * 8, output_bytes=0, n_gpu_layers=99,
                          logits_bytes=0, reserve_bytes=7, spread_bytes=4 * GiB)
    assert [c.reserve for c in idle] == [7 + 4 * GiB, 0]


@pytest.mark.parametrize("charge,expected", [(1 * GiB, True), (5 * GiB, False)])
def test_on_an_implicit_split_the_draft_goes_on_the_gpus_only_if_the_split_holds(
        tmp_path, charge, expected):
    from localm.inference.backends.gguf import GgufBackend
    draft = tmp_path / "d.gguf"
    draft.write_bytes(b"x")
    b = GgufBackend(str(tmp_path / "m.gguf"), spec_source="draft",
                    spec_draft_model=str(draft), n_ctx=2048)
    with patch.object(GgufBackend, "_draft_model_charge_bytes", return_value=charge), \
         patch.object(GgufBackend, "_implicit_split_fit",
                      side_effect=lambda layers: _two_gpus(b._draft_model_vram_bytes())), \
         patch.object(GgufBackend, "_split_free_total_bytes") as combined:
        assert b._decide_draft_placement() is expected
        assert b._spec_extra_vram_bytes() == (charge if expected else 0)
    combined.assert_not_called()


def test_unmeasurable_vram_keeps_the_draft_model_on_the_gpu(tmp_path):
    b, on_gpu, extra = _placed(tmp_path, None)
    assert (on_gpu, extra) == (True, _CHARGE)


def test_the_draft_model_is_placed_before_the_target_is_sized(tmp_path):
    from localm.inference.backends.gguf import GgufBackend
    order = []
    b = GgufBackend(str(tmp_path / "m.gguf"), spec_source="draft",
                    spec_draft_model=str(tmp_path / "d.gguf"))

    def place(self):
        order.append("place")
        self.draft_model_on_gpu = False
        return False

    def size(self):
        order.append(("size", self._spec_extra_vram_bytes()))
        return 99

    with patch("localm.model_manager.missing_split_parts", return_value=[]), \
         patch("localm.model_manager.gguf_pretokenizer", return_value=None), \
         patch("localm.inference.pretokenizer_guard.load_refusal", return_value=None), \
         patch.object(GgufBackend, "_decide_draft_placement", place), \
         patch.object(GgufBackend, "_draft_model_charge_bytes", return_value=_CHARGE), \
         patch.object(GgufBackend, "_mtp_draft_context_vram_bytes", return_value=0), \
         patch.object(GgufBackend, "_effective_gpu_layers", size), \
         patch.object(GgufBackend, "_check_vram"), \
         patch.object(GgufBackend, "_load_native"):
        b.load()
    assert order == ["place", ("size", 0)]


def test_the_worker_hands_a_cpu_placement_to_the_draft_load():
    from localm.inference.backends.llamacpp import _worker
    seen = {}

    class _Llm:
        supports_images = False

        def __init__(self, **kw):
            seen.update(kw)

    for on_gpu, expected in ((False, False), (True, True)):
        seen.clear()
        w = _worker.GgufWorker("m.gguf", None, 2048, 99, None, 0, spec_source="draft",
                               spec_draft_model="d.gguf", spec_draft_gpu=on_gpu)
        with patch("localm.inference.backends.llamacpp._loader.load_lib"), \
             patch("localm.inference.backends.llamacpp.LlamaCpp", _Llm):
            w.load()
        assert seen.get("spec_draft_gpu") is expected
        assert w.draft_model_on_gpu is on_gpu


def test_a_cpu_placed_draft_model_loads_without_gpu_layers_and_says_so(tmp_path):
    llm, path, llama_mod = _loading_llama(tmp_path)
    with patch.object(llama_mod, "api") as api, \
         patch.object(llama_mod, "set_use_mmap"), \
         patch("localm.discover.apply_gpu_split", return_value=None):
        params = SimpleNamespace(n_gpu_layers=-1, split_mode=1, main_gpu=0)
        api.llama_model_default_params.return_value = params
        _view_api(api)
        api.llama_load_model_from_file.return_value = ctypes.c_void_p(5)
        api.has_hybrid_api.return_value = True
        api.llama_model_is_recurrent.return_value = False
        api.llama_model_is_hybrid.return_value = False
        made = []
        api.llama_context_default_params.side_effect = lambda: made.append(
            SimpleNamespace()) or made[-1]
        api.llama_init_from_model.return_value = ctypes.c_void_p(6)
        llm._offload_kqv = True
        llm._load_draft_model(path, None, True, on_gpu=False)
    src = llm._source
    assert params.n_gpu_layers == 0
    assert made[-1].offload_kqv is False
    assert (src.usable, src.status) == (True, "ok-cpu")


def test_a_cpu_placed_draft_model_is_reported_on_the_reply():
    from localm.inference.backends.gguf import GgufBackend
    b = GgufBackend("m.gguf", spec_source="draft")
    b._loaded = True
    b._record_mtp({"finish_reason": "stop", "speculation": {
        "source": "draft", "status": "ok-cpu", "active": True, "call_status": "",
        "skipped": "", "drafted": 8, "accepted": 5, "steps": 4, "paused_steps": 0,
        "draft_max": 2}})
    usage = b.last_speculation_usage
    assert (usage["source"], usage["state"], usage["reason"]) == ("draft", "on", "draft-on-cpu")


@pytest.mark.parametrize("source,configured,copies", [
    ("draft", None, 4), ("draft", 2, 3), ("ngram", None, 5), ("ngram", 2, 3)])
def test_recurrent_state_sizing_uses_the_sources_own_draft_cap(tmp_path, monkeypatch, source,
                                                               configured, copies):
    from localm.inference.backends.gguf import GgufBackend
    from localm.inference.backends.llamacpp import _draftmodel
    from localm.inference.backends.llamacpp import llama as llama_mod
    monkeypatch.setattr(_draftmodel, "DRAFT_MODEL_DRAFT_TOKENS_DEFAULT", 3)
    b = GgufBackend(str(tmp_path / "m.gguf"), spec_source=source, spec_draft_tokens=configured)
    with patch("localm.model_manager.gguf.gguf_recurrent_state_bytes", return_value=1000), \
         patch.object(GgufBackend, "_gguf_parsed_tensor_entries", return_value=None):
        charge = b._recurrent_state_vram_bytes()
    llm = make_bare_llama(_model_ptr=ctypes.c_void_p(1))
    llm._spec_source_name = source
    llm._mtp_enabled = False
    cp = SimpleNamespace(n_rs_seq=0)
    with patch.object(llama_mod, "api") as api:
        api.has_hybrid_api.return_value = True
        api.llama_model_is_recurrent.return_value = True
        api.llama_model_is_hybrid.return_value = False
        llm._apply_initial_spec_params(cp, configured)
    assert charge == 1000 * copies
    assert charge == 1000 * (1 + cp.n_rs_seq)


@pytest.mark.parametrize("placed,llm,expected", [
    (True, None, True), (False, None, False),
    (True, SimpleNamespace(_source=SimpleNamespace(loaded=True)), True),
    (True, SimpleNamespace(_source=SimpleNamespace(loaded=False)), False),
    (True, SimpleNamespace(_source=None), True)])
def test_the_worker_charges_draft_growth_only_while_the_draft_model_is_loaded(placed, llm,
                                                                              expected):
    from localm.inference.backends.llamacpp._worker import GgufWorker
    w = GgufWorker("m.gguf", None, 2048, 99, None, 0, spec_source="draft",
                   spec_draft_model="d.gguf", spec_draft_gpu=placed)
    w._llm = llm
    w._draft_kv_per_token_cached = 1000
    assert w.draft_model_on_gpu is expected
    with patch.object(GgufWorker, "_draft_model_charge_bytes", return_value=_CHARGE), \
         patch.object(GgufWorker, "_mtp_draft_kv_per_token", return_value=0):
        assert w._spec_kv_per_token() == (1000 if expected else 0)
        assert w._spec_extra_vram_bytes() == (_CHARGE if expected else 0)


def test_a_disabled_draft_source_frees_its_native_state():
    from localm.inference.backends.llamacpp import llama as llama_mod
    llm = _llama()
    src = llm._source
    src._sampler = object()
    with patch.object(llama_mod, "api") as api:
        src.disable("draft-rewind-unsupported")
        freed_models = [c.args[0].value for c in api.llama_free_model.call_args_list]
        freed_ctx = [c.args[0].value for c in api.llama_free.call_args_list]
    assert (src.usable, src.status, src.loaded) == (False, "draft-rewind-unsupported", False)
    assert (freed_models, freed_ctx) == ([5], [6])
    api.llama_sampler_free.assert_called_once()


def test_no_charge_for_a_draft_model_beside_an_encoder_decoder_model(tmp_path):
    from localm.inference.backends.gguf import GgufBackend
    tokens = ["x%d" % i for i in range(40)]
    draft = _vocab_gguf(tmp_path / "d.gguf", tokens)
    charges = {}
    for arch in ("qwen2", "t5"):
        target = _vocab_gguf(tmp_path / ("m-%s.gguf" % arch), tokens, arch=arch)
        b = GgufBackend(str(target), spec_source="draft", spec_draft_model=str(draft), n_ctx=2048)
        with patch("localm.model_manager.gguf.gguf_kv_bytes_per_token", return_value=1000), \
             patch("localm.model_manager.gguf._gguf_split_layout_meta", return_value=(2, 40)), \
             patch("localm.model_manager.gguf.gguf_recurrent_state_bytes", return_value=0):
            charges[arch] = b._draft_model_charge_bytes()
    assert charges["qwen2"] > 0 and charges["t5"] == 0


@pytest.mark.parametrize("draft_model,recurrent,arch,charged", [
    ("gpt2", 0, "qwen2", True), ("llama", 0, "qwen2", False), ("gpt2", 4096, "qwen2", False),
    ("gpt2", 0, "eagle3", False)])
def test_no_charge_for_a_draft_model_the_metadata_already_rejects(tmp_path, draft_model,
                                                                  recurrent, arch, charged):
    from localm.inference.backends.gguf import GgufBackend
    tokens = ["x%d" % i for i in range(40)]
    target = _vocab_gguf(tmp_path / "m.gguf", tokens)
    draft = _vocab_gguf(tmp_path / "d.gguf", tokens, model=draft_model, arch=arch)
    b = GgufBackend(str(target), spec_source="draft", spec_draft_model=str(draft), n_ctx=2048)
    with patch("localm.model_manager.gguf.gguf_kv_bytes_per_token", return_value=1000), \
         patch("localm.model_manager.gguf._gguf_split_layout_meta", return_value=(2, 40)), \
         patch("localm.model_manager.gguf.gguf_recurrent_state_bytes",
               side_effect=lambda p, **kw: recurrent if p == draft else 0):
        charge = b._draft_model_charge_bytes()
    assert (charge > 0) is charged


@pytest.mark.parametrize("gpu_layers,level", [(0, "info"), (99, "warning")])
def test_a_cpu_draft_is_logged_by_its_cause(tmp_path, gpu_layers, level):
    from localm.inference.backends.gguf import GgufBackend
    b = GgufBackend(str(tmp_path / "m.gguf"), spec_source="draft",
                    spec_draft_model=str(tmp_path / "d.gguf"), n_gpu_layers=gpu_layers)
    with patch("localm.model_manager.missing_split_parts", return_value=[]), \
         patch("localm.model_manager.gguf_pretokenizer", return_value=None), \
         patch("localm.inference.pretokenizer_guard.load_refusal", return_value=None), \
         patch.object(GgufBackend, "_decide_draft_placement", return_value=False), \
         patch.object(GgufBackend, "_effective_gpu_layers", return_value=gpu_layers), \
         patch.object(GgufBackend, "_check_vram"), \
         patch.object(GgufBackend, "_load_native"), \
         patch("localm.debuglog.logger") as log:
        b.load()
    other = "warning" if level == "info" else "info"
    getattr(log, level).assert_called_once()
    assert ("does not fit" in getattr(log, level).call_args.args[0]) is (level == "warning")
    getattr(log, other).assert_not_called()


def test_the_engine_reports_the_draft_placement():
    from localm.inference import engine as engine_mod
    eng = object.__new__(engine_mod.Engine)
    eng._backend = SimpleNamespace(draft_model_on_gpu=False)
    assert eng.draft_model_on_gpu() is False
    eng._backend = SimpleNamespace()
    assert eng.draft_model_on_gpu() is None


def test_a_split_draft_model_is_charged_for_every_part(tmp_path):
    from localm.inference.backends.gguf import GgufBackend
    from localm.model_manager.gguf import gguf_file_bytes
    first = tmp_path / "d-00001-of-00002.gguf"
    first.write_bytes(b"x" * 1000)
    (tmp_path / "d-00002-of-00002.gguf").write_bytes(b"x" * 3000)
    assert gguf_file_bytes(first) == 4000
    assert gguf_file_bytes(tmp_path / "gone.gguf") == 0
    b = GgufBackend(str(tmp_path / "m.gguf"), spec_source="draft",
                    spec_draft_model=str(first), n_ctx=16)
    with patch("localm.model_manager.gguf.gguf_kv_bytes_per_token", return_value=0), \
         patch("localm.model_manager.gguf._gguf_split_layout_meta", return_value=None), \
         patch.object(GgufBackend, "_draft_model_rejected_by_metadata", return_value=False):
        assert b._draft_model_charge_bytes() == 4000 + DRAFT_COMPUTE_MARGIN_BYTES


def test_the_draft_model_charge_covers_the_measured_buffers():
    """Qwen2.5-0.5B-Instruct Q8_0 as a draft at n_ctx 4096 on ROCm: llama.cpp
    reported 500.84 MiB of weights on the GPU, a 48.00 MiB KV buffer and a
    298.50 MiB compute buffer for a 512-token batch. The charge for the same
    load must cover each."""
    from localm.inference.backends.llamacpp._split_fit import logits_buffer_bytes
    mib = 1024 * 1024
    file_bytes, n_vocab, n_ctx, kv_per_token = 531068480, 151936, 4096, 12288
    assert file_bytes >= 500.84 * mib
    assert n_ctx * kv_per_token >= 48.00 * mib
    assert (logits_buffer_bytes(n_vocab, n_ctx, max_batch=DRAFT_CONTEXT_BATCH)
            + DRAFT_COMPUTE_MARGIN_BYTES) >= 298.50 * mib


# --------------------------------------------------------------------------- #
#  bench-spec --source draft and spec-drafts                                  #
# --------------------------------------------------------------------------- #

def test_bench_spec_drafts_with_the_named_draft_model(cli_runner, tmp_path):
    from localm.cli import models as models_mod
    from localm.inference import engine
    from tests.test_spec_source_settings import _spec_arm
    stub = _spec_arm([50.0], [70.0])
    seen = []

    def arm(*a, draft_tokens=None, draft_model=None):
        seen.append((a[2], draft_tokens, draft_model))
        return stub(*a, draft_tokens=draft_tokens)

    small = tmp_path / "small.gguf"
    with patch.object(models_mod, "get_operator_model_info",
                      return_value=("model.gguf", None)), \
         patch("localm.model_manager.registry.get_operator_model_info",
               side_effect=lambda n: (small, None) if n == "small" else None), \
         patch.object(models_mod, "_spec_probe_arm", arm):
        res = cli_runner.invoke(models_mod.main,
                                ["bench-spec", "model.gguf", "--source", "draft",
                                 "--draft-model", "small", "--rounds", "1", "-d", "3"])
    assert res.exit_code == 0, res.output
    assert "Draft model is 1.40x faster" in res.output
    assert seen == [("off", None, None), ("draft", 3, str(small))]
    assert engine.resolve_spec_draft_model({}) is None


@pytest.mark.parametrize("counts,shown", [((128, 8), True), ((0, 0), False)])
def test_bench_spec_names_the_mixture_of_experts_shape(cli_runner, counts, shown):
    from localm.cli import models as models_mod
    from tests.test_spec_source_settings import _spec_arm
    with patch.object(models_mod, "get_operator_model_info",
                      return_value=("model.gguf", None)), \
         patch("localm.model_manager.gguf.gguf_expert_counts", return_value=counts), \
         patch.object(models_mod, "_spec_probe_arm", _spec_arm([50.0], [70.0])):
        res = cli_runner.invoke(models_mod.main, ["bench-spec", "model.gguf", "--rounds", "1"])
    assert res.exit_code == 0, res.output
    assert ("8 of 128 experts per token" in res.output) is shown


def test_the_engine_hands_out_the_measured_step_costs():
    from localm.inference import engine as engine_mod
    eng = object.__new__(engine_mod.Engine)
    eng._backend = SimpleNamespace(last_speculation={"costs": {"target_ms": 8.0}})
    assert eng.draft_step_costs() == {"target_ms": 8.0, "observed_ms": {}}
    eng._backend = SimpleNamespace(last_speculation={"costs": {"target_ms": 8.0},
                                                     "observed_ms": {0: 9.0}})
    assert eng.draft_step_costs() == {"target_ms": 8.0, "observed_ms": {0: 9.0}}
    eng._backend = SimpleNamespace(last_speculation=None)
    assert eng.draft_step_costs() is None


def test_the_report_carries_the_measured_and_observed_costs():
    llm = _measured_llama(costs=_costs())
    src = llm._source
    assert "observed_ms" in src.report() and src.report()["observed_ms"] == {}
    src.on_step_seconds(0, 1.0)
    rep = src.report()
    assert rep["costs"]["target_ms"] == 1000.0
    assert rep["observed_ms"] == {0: 1000.0}
    assert rep["acceptance"] == pytest.approx(0.6)
    src.costs = None
    assert "costs" not in src.report()


def test_bench_spec_without_a_draft_model_says_how_to_find_one(cli_runner):
    from localm.cli import models as models_mod
    arm = MagicMock()
    with patch.object(models_mod, "get_operator_model_info",
                      return_value=("model.gguf", None)), \
         patch.object(models_mod, "_spec_probe_arm", arm):
        res = cli_runner.invoke(models_mod.main,
                                ["bench-spec", "model.gguf", "--source", "draft"])
    assert res.exit_code == 1
    assert "No draft model" in res.output and "spec-drafts" in res.output
    arm.assert_not_called()


def _registry_models(tmp_path):
    same = ["x%d" % i for i in range(40)]
    other = ["y%d" % i for i in range(40)]
    target = _vocab_gguf(tmp_path / "big.gguf", same)
    with open(target, "ab") as f:
        f.write(b"\0" * 8192)
    files = {
        "big": (target, "llm"),
        "small": (_vocab_gguf(tmp_path / "small.gguf", same), "llm"),
        "tiny": (_vocab_gguf(tmp_path / "tiny.gguf", same[:30]), "llm"),
        "foreign": (_vocab_gguf(tmp_path / "foreign.gguf", other), "llm"),
        "embedder": (_vocab_gguf(tmp_path / "embed.gguf", same), "embedding"),
        "head": (_vocab_gguf(tmp_path / "head.gguf", same, arch="eagle3"), "llm"),
        "pooler": (_vocab_gguf(tmp_path / "pool.gguf", same,
                               extra=[("qwen2.pooling_type", _kv_uint32(1))]), "llm"),
    }
    return target, {n: {"path": str(p), "model_type": t} for n, (p, t) in files.items()}


def test_spec_drafts_lists_only_models_that_share_the_vocabulary(cli_runner, tmp_path):
    from localm.cli import models as models_mod
    target, registry = _registry_models(tmp_path)
    with patch.object(models_mod, "get_operator_model_info", return_value=(str(target), None)), \
         patch("localm.model_manager.load_registry", return_value=registry):
        res = cli_runner.invoke(models_mod.main, ["spec-drafts", "big"], terminal_width=200)
    assert res.exit_code == 0, res.output
    rows = [ln for ln in res.output.splitlines() if " GB " in ln]
    assert [r.split()[1] for r in rows] == ["tiny", "small"]


def test_spec_drafts_says_so_when_nothing_fits(cli_runner, tmp_path):
    from localm.cli import models as models_mod
    target, registry = _registry_models(tmp_path)
    registry = {"foreign": registry["foreign"]}
    with patch.object(models_mod, "get_operator_model_info", return_value=(str(target), None)), \
         patch("localm.model_manager.load_registry", return_value=registry):
        res = cli_runner.invoke(models_mod.main, ["spec-drafts", "big"])
    assert res.exit_code == 0, res.output
    assert "No downloaded causal chat model shares" in res.output


# --------------------------------------------------------------------------- #
#  Settings                                                                   #
# --------------------------------------------------------------------------- #

def test_the_draft_source_and_model_settings(tmp_path):
    from localm import settings_schema as ss
    from localm.bugreport import _SAFE_CONFIG_KEYS
    from localm.config import DEFAULT_CONFIG
    from localm.inference.routing_latch import LOAD_CONFIG_KEYS
    assert ss.validate_update({"spec_source": "draft"}) == {"spec_source": "draft"}
    assert "spec_draft_model" in ss.admin_only_keys()
    assert DEFAULT_CONFIG["spec_draft_model"] == ""
    assert "spec_draft_model" in _SAFE_CONFIG_KEYS and "spec_draft_model" in LOAD_CONFIG_KEYS


def test_the_draft_model_name_resolves_through_the_registry(tmp_path):
    from localm.inference import engine
    path = tmp_path / "d.gguf"
    with patch("localm.model_manager.registry.get_operator_model_info",
               side_effect=lambda n: (path, None) if n == "small" else None):
        assert engine.resolve_spec_draft_model({"spec_draft_model": "small"}) == str(path)
        assert engine.resolve_spec_draft_model({"spec_draft_model": "gone"}) == "gone"
        assert engine.resolve_spec_draft_model({}, "small") == str(path)
    assert engine.resolve_spec_draft_model({"spec_draft_model": "  "}) is None


def test_create_backend_passes_the_draft_model_only_for_the_draft_source():
    from localm.config import DEFAULT_CONFIG
    from localm.inference import engine as engine_mod
    captured = {}

    class _FakeBackend:
        def __init__(self, *a, **kw):
            captured.update(kw)

    cfg = dict(DEFAULT_CONFIG, spec_draft_model="small")
    with patch.object(engine_mod, "load_config", return_value=cfg), \
         patch.object(engine_mod, "resolve_spec_draft_model", return_value="/m/small.gguf"), \
         patch("localm.inference.backends.gguf.GgufBackend", _FakeBackend):
        engine_mod.create_backend("model.gguf", spec_source="draft")
        assert captured["spec_draft_model"] == "/m/small.gguf"
        engine_mod.create_backend("model.gguf", spec_source="ngram")
        assert captured["spec_draft_model"] is None
