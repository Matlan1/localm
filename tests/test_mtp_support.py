# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for Multi-Token Prediction (MTP) model support."""

import ctypes
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from localm.config import DEFAULT_CONFIG
from localm.settings_schema import CORE_FIELDS
from localm.inference.backends.base import BaseBackend
from localm.inference.backends.gguf import GgufBackend
from localm.inference.backends.llamacpp._structs import (
    LLAMA_CONTEXT_TYPE_DEFAULT,
    LLAMA_CONTEXT_TYPE_MTP,
    LlamaModelParamsV2,
)
from localm.inference.backends.llamacpp import _api as api
from localm.inference.engine import Engine
from tests._bare_llama import make_bare_llama


def test_mtp_constants_and_structs():
    """Verify MTP context type constants and model param struct offsets."""
    assert LLAMA_CONTEXT_TYPE_DEFAULT == 0
    assert LLAMA_CONTEXT_TYPE_MTP == 1
    assert hasattr(LlamaModelParamsV2, "load_mtp")
    mp = LlamaModelParamsV2()
    assert hasattr(mp, "load_mtp")
    mp.load_mtp = True
    assert mp.load_mtp is True


def test_mtp_config_and_settings_schema():
    """Verify mtp_enabled setting is in default config and schema."""
    assert "mtp_enabled" in DEFAULT_CONFIG
    assert DEFAULT_CONFIG["mtp_enabled"] is True

    schema_field = next((f for f in CORE_FIELDS if f.key == "mtp_enabled"), None)
    assert schema_field is not None
    assert schema_field.group == "Engine"


def test_base_backend_and_engine_capability():
    """Verify supports_mtp capability defaults and exposure on Engine."""
    class DummyBackend(BaseBackend):
        @property
        def loaded(self) -> bool:
            return False
        def load(self): pass
        def unload(self): pass
        def chat_stream(self, *args, **kwargs):
            return iter([])
        def generate(self, *args, **kwargs): pass

    dummy = DummyBackend()
    assert dummy.supports_mtp is False

    with patch("localm.inference.engine.create_backend", return_value=dummy):
        engine = Engine("dummy-model")
        assert engine.supports_mtp is False


def test_llama_model_has_mtp_detection():
    """Verify GGUF metadata detection for MTP architectures."""
    mock_model = ctypes.c_void_p(1234)

    # 1. Direct native library check
    with patch.object(api, "load_lib") as mock_load_lib:
        mock_dll = MagicMock()
        mock_dll.llama_model_has_mtp.return_value = True
        mock_load_lib.return_value = mock_dll
        with patch.object(api, "_bind", return_value=lambda m: True):
            assert api.llama_model_has_mtp(mock_model) is True

    # 2. GGUF metadata check (DeepSeek nextn_predict_layers)
    with patch.object(api, "load_lib") as mock_load_lib, \
         patch.object(api, "has_model_meta_api", return_value=True):
        mock_dll = MagicMock(spec=[])  # no native llama_model_has_mtp
        mock_load_lib.return_value = mock_dll

        def fake_meta_val(model, key):
            if key == "general.architecture":
                return "deepseek2"
            if key == "deepseek2.nextn_predict_layers":
                return "1"
            return None

        with patch.object(api, "llama_model_meta_val_str", side_effect=fake_meta_val):
            assert api.llama_model_has_mtp(mock_model) is True

    # 3. GGUF metadata check, tolerated mtp_head_count spelling, on an arch
    #    that does build an MTP graph
    with patch.object(api, "load_lib") as mock_load_lib, \
         patch.object(api, "has_model_meta_api", return_value=True):
        mock_dll = MagicMock(spec=[])
        mock_load_lib.return_value = mock_dll

        def fake_meta_val_qwen(model, key):
            if key == "general.architecture":
                return "qwen35moe"
            if key == "qwen35moe.mtp_head_count":
                return "2"
            return None

        with patch.object(api, "llama_model_meta_val_str", side_effect=fake_meta_val_qwen):
            assert api.llama_model_has_mtp(mock_model) is True

    # 4. Standard non-MTP model
    with patch.object(api, "load_lib") as mock_load_lib, \
         patch.object(api, "has_model_meta_api", return_value=True):
        mock_dll = MagicMock(spec=[])
        mock_load_lib.return_value = mock_dll

        def fake_meta_val_none(model, key):
            if key == "general.architecture":
                return "llama"
            return None

        with patch.object(api, "llama_model_meta_val_str", side_effect=fake_meta_val_none):
            assert api.llama_model_has_mtp(mock_model) is False


def _detect(arch, extra):
    """Run the real detector against a synthetic GGUF metadata table."""
    def _val(model, key):
        if key == "general.architecture":
            return arch
        return extra.get(key)

    with patch.object(api, "load_lib") as mock_load_lib,          patch.object(api, "has_model_meta_api", return_value=True):
        mock_load_lib.return_value = MagicMock(spec=[])  # no native llama_model_has_mtp
        with patch.object(api, "llama_model_meta_val_str", side_effect=_val):
            return api.llama_model_mtp_support(ctypes.c_void_p(1234))


def test_metadata_key_alone_does_not_engage_mtp():
    """An architecture that ships nextn metadata but builds no MTP graph is refused.

    glm4moe (GLM-4.5 / 4.5-Air / 4.6) is the real case: published GGUFs carry
    glm4moe.nextn_predict_layers=1 and the NextN tensors, while upstream's
    build_arch_graph ignores the MTP graph type, so an MTP context there is a
    second full decoder rather than a draft head.
    """
    supported, reason = _detect("glm4moe", {"glm4moe.nextn_predict_layers": "1"})
    assert supported is False
    assert reason == "no-mtp-graph:glm4moe"


def test_mtp_engages_on_an_architecture_with_a_draft_graph():
    """The same metadata on an architecture that does build an MTP graph is accepted."""
    supported, reason = _detect("qwen35", {"qwen35.nextn_predict_layers": "1"})
    assert supported is True
    assert reason == "ok:qwen35"


def test_mtp_needs_metadata_as_well_as_a_capable_architecture():
    """A capable architecture with no nextn metadata is refused."""
    supported, reason = _detect("qwen35", {})
    assert supported is False
    assert reason == "no-mtp-metadata"


def test_every_allowlisted_architecture_is_accepted_with_metadata():
    """No allowlist entry is unreachable - a typo would strand one silently."""
    for arch in sorted(api.MTP_GRAPH_ARCHITECTURES):
        supported, reason = _detect(arch, {f"{arch}.nextn_predict_layers": "1"})
        assert supported is True, f"{arch} is allowlisted but was refused ({reason})"


def test_zero_or_unparsable_nextn_value_is_not_mtp():
    """nextn_predict_layers=0 declares no heads; a non-numeric value declares nothing."""
    assert _detect("qwen35", {"qwen35.nextn_predict_layers": "0"})[1] == "no-mtp-metadata"
    assert _detect("qwen35", {"qwen35.nextn_predict_layers": "x"})[1] == "no-mtp-metadata"


def test_absent_metadata_api_is_reported_as_its_own_reason():
    """"Could not look" is a distinct answer from "looked and found no MTP"."""
    with patch.object(api, "load_lib") as mock_load_lib,          patch.object(api, "has_model_meta_api", return_value=False):
        mock_load_lib.return_value = MagicMock(spec=[])
        supported, reason = api.llama_model_mtp_support(ctypes.c_void_p(1234))
    assert supported is False
    assert reason == "no-metadata-api"


def test_gguf_backend_supports_mtp():
    """Verify GgufBackend correctly reflects supports_mtp state."""
    backend = GgufBackend("test_model.gguf")
    assert backend.supports_mtp is False

    # Simulate load metadata with supports_mtp=True
    backend._loaded = True
    backend._supports_mtp = True
    assert backend.supports_mtp is True

    # Simulate load metadata with supports_mtp=False
    backend._supports_mtp = False
    assert backend.supports_mtp is False


#  Speculative MTP decoding: which sampler decides, and what the chain is told
#
#  llama_sampler_sample ACCEPTS the token it returns into every stateful sampler
#  in the chain, and llama.cpp offers no way to rewind that. So the sampler that
#  decides a speculation is the one whose state advances, and every token it is
#  told about has to be a token that is actually emitted. The harness below
#  records both sides so a test can assert that directly, rather than asserting
#  that a token happened to appear somewhere in the output.

class _SpecRecorder:
    """Drives LlamaCpp._generate's native sampling and records every call.

    Three call shapes are distinguished by (sampler identity, idx):

        HEAD    main sampler,  idx -1   the ordinary next-token sample
        DRAFT   draft sampler, idx -1   a speculative proposal off the MTP context
        VERIFY  main sampler,  idx 0    the target model's own continuation

    Each keyword supplies the tokens its shape returns, in order.
    """

    EOG = 999

    def __init__(self, head=(), draft=(), verify=()):
        self.main_sampler = MagicMock(name="main_sampler")
        self.draft_sampler = MagicMock(name="draft_sampler")
        self._queues = {"HEAD": list(head), "DRAFT": list(draft), "VERIFY": list(verify)}
        self.calls = []          # (shape, token) in call order
        self.told_main = []      # tokens the main chain accepted, in order

    def _shape(self, sampler, idx):
        if sampler is self.draft_sampler:
            return "DRAFT"
        return "VERIFY" if idx == 0 else "HEAD"

    def sample(self, sampler, ctx, idx):
        shape = self._shape(sampler, idx)
        queue = self._queues[shape]
        if not queue:
            # _generate wraps the DRAFT sample in `except Exception`, so raising
            # here would be swallowed and the run would continue on a silently
            # different path. An exhausted draft queue therefore proposes
            # end-of-generation, which sends that iteration down the ordinary
            # single-token path and is visible in the call log.
            assert shape == "DRAFT", f"_generate asked for an unscripted {shape} sample"
            self.calls.append(("DRAFT_EXHAUSTED", self.EOG))
            return self.EOG
        token = queue.pop(0)
        self.calls.append((shape, token))
        if shape != "DRAFT":
            self.told_main.append(token)
        return token

    def decode(self, ctx, batch):
        self.calls.append(("DECODE", None))
        return 0

    def accept(self, sampler, token):
        if sampler is self.main_sampler:
            self.told_main.append(token)

    def shapes(self):
        return [shape for shape, _ in self.calls]


def _run_generate(recorder, *, max_new_tokens, **kwargs):
    """Run _generate against *recorder*; return (yielded tokens, the mock api)."""
    llm = make_bare_llama(
        _model_ptr=ctypes.c_void_p(1),
        _ctx_ptr=ctypes.c_void_p(2),
        _mtp_ctx_ptr=ctypes.c_void_p(3),
        supports_mtp=True,
    )
    llm._tokenizer.is_eog.side_effect = lambda t: t == _SpecRecorder.EOG
    llm._fit_generation_budget = lambda n_prompt, max_new: max_new
    llm._can_reuse_kv = lambda needed: False
    llm._prefill_fresh_context = MagicMock()
    llm._create_batch = MagicMock(return_value=MagicMock())

    with patch("localm.inference.backends.llamacpp.llama.api") as mock_api, \
         patch("localm.inference.backends.llamacpp.llama._build_sampler",
               return_value=recorder.main_sampler):
        mock_api.llama_sampler_init_greedy.return_value = recorder.draft_sampler
        mock_api.llama_sampler_sample.side_effect = recorder.sample
        mock_api.llama_sampler_accept.side_effect = recorder.accept
        mock_api.llama_decode.side_effect = recorder.decode
        tokens = list(llm._generate(
            prompt_tokens=[1, 2],
            max_new_tokens=max_new_tokens,
            temperature=0.8,
            top_k=40,
            top_p=0.95,
            repeat_penalty=1.1,
            **kwargs,
        ))
    return tokens, mock_api


def _assert_chain_matches_output(recorder, tokens):
    """Every token the main sampler chain was told about is a token that was
    emitted, in the same order.

    A trailing end-of-generation token is the one permitted exception: it is
    sampled, ends the turn, is never emitted, and the sampler is freed straight
    after. Anything else means the repetition window advanced on a token the
    caller never received.
    """
    told = recorder.told_main
    assert told[:len(tokens)] == tokens, (
        f"main chain saw {told}, output was {tokens}")
    assert told[len(tokens):] in ([], [_SpecRecorder.EOG]), (
        f"main chain was told about {told[len(tokens):]} beyond the output")


def test_mtp_verification_uses_the_request_sampler_not_the_greedy_drafter():
    """The token that decides a speculation is drawn through the REQUEST's
    sampler. Verifying with the bare greedy drafter drops temperature, top_k,
    top_p and the repetition window for every accepted speculative token, which
    on an MTP model is a large share of the reply."""
    rec = _SpecRecorder(head=[100, _SpecRecorder.EOG], draft=[101], verify=[101])

    tokens, _ = _run_generate(rec, max_new_tokens=4)

    assert ("VERIFY", 101) in rec.calls, (
        f"no verification went through the request sampler: {rec.calls}")
    assert tokens == [100, 101]


def test_mtp_accepted_draft_enters_the_chain_exactly_once():
    """llama_sampler_sample already accepts what it returns, so an accepted
    draft needs no second accept. A duplicate advances the repetition window
    twice and, with a grammar in the chain, threw across the C ABI."""
    rec = _SpecRecorder(head=[100, _SpecRecorder.EOG], draft=[101], verify=[101])

    tokens, mock_api = _run_generate(rec, max_new_tokens=4)

    assert tokens == [100, 101]
    assert rec.told_main.count(101) == 1, (
        f"token 101 entered the main chain {rec.told_main.count(101)} times")
    mock_api.llama_sampler_accept.assert_not_called()
    _assert_chain_matches_output(rec, tokens)


def test_mtp_rejected_draft_emits_the_target_models_own_token():
    """On a mismatch the target model's own token is what the position emits.
    Discarding it leaves the sampler advanced on a token that was never
    produced, and llama.cpp offers no way to rewind that."""
    rec = _SpecRecorder(head=[200, _SpecRecorder.EOG], draft=[201], verify=[202])

    tokens, _ = _run_generate(rec, max_new_tokens=4)

    assert tokens == [200, 202], f"expected the target's own token, got {tokens}"
    assert 201 not in tokens
    _assert_chain_matches_output(rec, tokens)


def test_mtp_rejected_draft_does_not_resample_from_the_stale_logits_row():
    """The speculative batch is decoded with logits for BOTH rows, so idx -1 is
    the row produced after the DRAFT token. A reject removes that token from the
    KV cache, so no sample is taken at idx -1 before the next decode."""
    rec = _SpecRecorder(head=[200, _SpecRecorder.EOG], draft=[201], verify=[202])

    _run_generate(rec, max_new_tokens=4)

    shapes = rec.shapes()
    verify_at = shapes.index("VERIFY")
    after = shapes[verify_at + 1:]
    assert "HEAD" not in after or "DECODE" in after[:after.index("HEAD")], (
        f"a next-token sample followed the reject with no decode between: {shapes}")


def test_mtp_rejected_draft_rolls_the_speculative_slot_out_of_both_caches():
    """The draft token was decoded into the main context at pos + 1 and does not
    survive its rejection; the MTP context is trimmed to the same position."""
    rec = _SpecRecorder(head=[200, _SpecRecorder.EOG], draft=[201], verify=[202])

    _, mock_api = _run_generate(rec, max_new_tokens=4)

    trimmed = [call.args for call in mock_api.llama_kv_cache_seq_rm.call_args_list]
    assert trimmed, "the rejected speculation was left in the KV cache"
    assert all(args[2] == 3 for args in trimmed), trimmed   # prompt len 2, pos + 1


def test_mtp_end_of_generation_verification_ends_the_turn_without_emitting():
    """A rejected speculation whose replacement is end-of-generation ends the
    turn: the token is not emitted, and nothing is sampled after it."""
    rec = _SpecRecorder(head=[300], draft=[301], verify=[_SpecRecorder.EOG])

    tokens, _ = _run_generate(rec, max_new_tokens=8)

    assert tokens == [300]
    assert rec.shapes()[-1] == "VERIFY"


def test_mtp_drafting_is_disabled_while_a_grammar_is_active():
    """No speculation runs with a grammar sampler in the chain: the speculative
    path decodes and rolls back a token the grammar never saw, and a
    mis-sequenced accept on a grammar sampler throws across the C ABI."""
    rec = _SpecRecorder(head=[400, _SpecRecorder.EOG])

    tokens, mock_api = _run_generate(rec, max_new_tokens=4, grammar='root ::= "a"')

    assert tokens == [400]
    assert "DRAFT" not in rec.shapes(), rec.shapes()
    mock_api.llama_sampler_init_greedy.assert_not_called()


def test_a_stuck_draft_cell_disables_mtp_and_keeps_generating():
    """A rejected draft whose KV cell cannot be removed must not end the turn.

    llama_memory_seq_rm returns false on a memory module that cannot partially
    rewind. The rejected token then stays at pos + 1, and llama.cpp refuses any
    later batch whose start position it already holds, so generation dies a
    couple of tokens in and reports a token budget it never reached. Measured on
    unsloth/Qwen3.5-0.8B-MTP-GGUF: 28 characters and finish_reason "length"
    against 941 characters with MTP off.

    The cache here is modelled rather than mocked flat, because the defect's
    symptom IS the later decodes failing: against a decode that always returns 0
    the token stream comes out right whether the fix is present or not.
    """
    rec = _SpecRecorder(head=[600, 604, _SpecRecorder.EOG],
                        draft=[601], verify=[602])

    llm = make_bare_llama(
        _model_ptr=ctypes.c_void_p(1),
        _ctx_ptr=ctypes.c_void_p(2),
        _mtp_ctx_ptr=ctypes.c_void_p(3),
        supports_mtp=True,
    )
    llm._tokenizer.is_eog.side_effect = lambda t: t == _SpecRecorder.EOG
    llm._fit_generation_budget = lambda n_prompt, max_new: max_new
    llm._can_reuse_kv = lambda needed: False
    llm._prefill_fresh_context = MagicMock()
    llm._create_batch = lambda tokens, pos, **kw: SimpleNamespace(
        tokens=list(tokens), pos=pos)

    # Highest position the main cache holds. The prompt is 2 tokens, and
    # _prefill_fresh_context is mocked, so start where a real prefill would end.
    main = {"last": 1}

    def decode(ctx, batch):
        rec.calls.append(("DECODE", None))
        if ctx is not llm._ctx_ptr:
            return 0
        if batch.pos <= main["last"]:
            # llama.cpp: "the tokens for sequence 0 in the input batch have a
            # starting position of Y ... required that the position satisfies X < Y"
            return -1
        main["last"] = batch.pos + len(batch.tokens) - 1
        return 0

    def seq_rm(ctx, seq_id, p0, p1):
        return False        # this memory module cannot partially rewind

    def clear(mem, data):
        main["last"] = -1

    with patch("localm.inference.backends.llamacpp.llama.api") as mock_api,          patch("localm.inference.backends.llamacpp.llama._build_sampler",
               return_value=rec.main_sampler):
        mock_api.llama_sampler_init_greedy.return_value = rec.draft_sampler
        mock_api.llama_sampler_sample.side_effect = rec.sample
        mock_api.llama_sampler_accept.side_effect = rec.accept
        mock_api.llama_decode.side_effect = decode
        mock_api.llama_kv_cache_seq_rm.side_effect = seq_rm
        mock_api.llama_memory_clear.side_effect = clear
        tokens = list(llm._generate(
            prompt_tokens=[1, 2], max_new_tokens=8, temperature=0.8,
            top_k=40, top_p=0.95, repeat_penalty=1.1,
        ))

    # THE PROPERTY: the turn survives the stuck cell. Assert the token stream
    # before the status flags - a truncated reply is the user-visible loss, and
    # a status assertion alone reads as a detail worth adjusting.
    assert tokens == [600, 602, 604], tokens
    assert llm.last_finish_reason == "stop"

    # MTP is off for this model from here on, and says why.
    assert llm.supports_mtp is False
    assert llm.mtp_status == "rewind-unsupported"
    assert llm._mtp_usable is False

    # No speculation is attempted after the failure.
    assert rec.shapes().count("DRAFT") == 1, rec.shapes()


def _mtp_prefill_llama(capacity):
    llm = make_bare_llama(
        _model_ptr=ctypes.c_void_p(1),
        _ctx_ptr=ctypes.c_void_p(2),
        _mtp_ctx_ptr=ctypes.c_void_p(3),
        supports_mtp=True,
    )
    llm._mtp_ctx_capacity = capacity
    llm._create_batch = lambda tokens, pos, **kw: SimpleNamespace(
        tokens=list(tokens), pos=pos)
    return llm


def test_a_conversation_outgrowing_the_draft_context_stops_drafting():
    """The draft context is created once and never resized while the main one
    grows, so past its own n_ctx every draft decode fails. Stop instead of
    paying a doomed decode per token, and report why."""
    llm = _mtp_prefill_llama(2048)

    with patch("localm.inference.backends.llamacpp.llama.api") as mock_api:
        llm._prefill_mtp(list(range(10)), base_pos=2045)

    mock_api.llama_decode.assert_not_called()
    assert llm.supports_mtp is False
    assert llm.mtp_status == "draft-context-full"
    assert llm._mtp_usable is False


def test_a_failed_draft_prefill_decode_stops_drafting():
    """A draft decode that fails leaves the draft cache out of step with the
    main one, so every later draft would be conditioned on the wrong prefix.
    The return value used to be discarded inside a bare except."""
    llm = _mtp_prefill_llama(2048)

    with patch("localm.inference.backends.llamacpp.llama.api") as mock_api:
        mock_api.llama_decode.return_value = -1
        llm._prefill_mtp([1, 2, 3], base_pos=0)

    assert llm.supports_mtp is False
    assert llm.mtp_status == "draft-prefill-failed:-1"
    assert llm._mtp_usable is False


def test_a_healthy_draft_prefill_leaves_mtp_alone():
    """The false-positive direction: a normal prefill must not disable anything."""
    llm = _mtp_prefill_llama(2048)

    with patch("localm.inference.backends.llamacpp.llama.api") as mock_api:
        mock_api.llama_decode.return_value = 0
        llm._prefill_mtp([1, 2, 3], base_pos=0)

    assert mock_api.llama_decode.called
    assert llm.supports_mtp is True
    assert llm._mtp_usable is True


def test_mtp_two_consecutive_rejections_each_emit_their_own_token():
    """A carried replacement token opens a fresh speculation of its own, so the
    carry cannot be a one-shot that silently drops the second rejection."""
    rec = _SpecRecorder(
        head=[500, _SpecRecorder.EOG],
        draft=[501, 503],
        verify=[502, 504],
    )

    tokens, _ = _run_generate(rec, max_new_tokens=8)

    assert tokens == [500, 502, 504]
    _assert_chain_matches_output(rec, tokens)


def test_mtp_sampler_state_never_advances_past_an_emitted_token():
    """The state-consistency property on its own, over a run that both accepts
    and rejects a speculation, so it cannot be shadowed by an earlier assertion
    about the token stream.

    llama.cpp offers no way to rewind a sampler, so a token the chain accepts
    and the caller never receives leaves the repetition window permanently out
    of step with the reply that was actually produced.
    """
    rec = _SpecRecorder(
        head=[600, 700, _SpecRecorder.EOG],
        draft=[601, 603],
        verify=[601, 604],
    )

    tokens, _ = _run_generate(rec, max_new_tokens=8)

    _assert_chain_matches_output(rec, tokens)


def test_mtp_carried_token_is_not_dropped_at_the_token_budget_boundary():
    """The carry survives the last budgeted token. The in-loop budget check runs
    before the speculative block, so a speculation only starts with budget left
    and its replacement token always has an iteration to be emitted in."""
    rec = _SpecRecorder(head=[200], draft=[201], verify=[202])

    tokens, _ = _run_generate(rec, max_new_tokens=2)

    assert tokens == [200, 202]
    _assert_chain_matches_output(rec, tokens)

