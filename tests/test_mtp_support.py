# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for Multi-Token Prediction (MTP) model support."""

import ctypes
import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

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
from localm.inference.backends.llamacpp import llama as LlamaCppModule
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
    assert DEFAULT_CONFIG["mtp_enabled"] is False

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


def test_mtp_default_is_off_until_speculation_is_measured_to_pay():
    """MTP ships OFF, and every layer's default agrees with the config.

    A rejected draft still costs a two-token verification, and on a small model
    that costs meaningfully more than verifying one token, so speculation does
    not pay; it turns positive on a model large enough that the two cost about
    the same.
    """
    import inspect

    from localm.config import DEFAULT_CONFIG
    from localm.inference.backends.gguf import GgufBackend
    from localm.inference.backends.llamacpp._worker import GgufWorker
    from localm.inference.backends.llamacpp.llama import LlamaCpp

    assert DEFAULT_CONFIG["mtp_enabled"] is False

    for owner in (GgufBackend.__init__, LlamaCpp.__init__, GgufWorker.__init__):
        param = inspect.signature(owner).parameters["mtp_enabled"]
        assert param.default is False, (
            f"{owner.__qualname__} still defaults mtp_enabled to {param.default!r}, "
            "so a caller that omits it re-enables MTP")


def test_engine_does_not_re_enable_mtp_when_the_key_is_absent():
    """A config with no mtp_enabled key falls back to False, never True."""
    import inspect

    from localm.inference import engine as engine_mod

    src = inspect.getsource(engine_mod.Engine._create_backend
                            if hasattr(engine_mod.Engine, "_create_backend")
                            else engine_mod)
    assert 'cfg.get("mtp_enabled", True)' not in src
    assert 'cfg.get("mtp_enabled", False)' in src


def test_recurrent_rollback_is_requested_when_mtp_is_enabled():
    """A recurrent cache can only be rewound if it kept per-token snapshots.

    Speculation writes a draft token into the cache and takes it back out when
    the target rejects it. llama.cpp keeps no recurrent-state snapshots by
    default, so on a hybrid model that removal fails, the rejected token stays,
    and every later batch is refused for inconsistent positions. Measured on a
    real hybrid MTP model: the same one-position rollback returns False with no
    snapshots and True with them, which is the difference between MTP declining
    at load and running.

    One snapshot covers a one-token draft; the request is for two so a longer
    draft has room. It costs nothing on a model with no recurrent layers.
    """
    from localm.inference.backends.llamacpp import _structs

    for params in (_structs.LlamaContextParamsV1, _structs.LlamaContextParamsV2):
        assert hasattr(params(), "n_rs_seq"), (
            f"{params.__name__} has no n_rs_seq, so rollback cannot be requested "
            "and MTP silently declines on every hybrid model")

    from localm.inference.backends.llamacpp.llama import LlamaCpp
    src = inspect.getsource(LlamaCpp)
    assert src.count("n_rs_seq") >= 2, (
        "both the initial context and the grown one must request rollback, or "
        "speculation stops the moment a conversation outgrows its first context")


def test_an_image_turn_clears_the_draft_cache_too():
    """An image turn must not leave the draft cache describing the old context.

    mtmd evaluates an image prompt from position 0, so the main cache is emptied
    first. The draft cache has to go with it: the next text turn rebuilds both
    from scratch, but only because it finds no cached tokens to reuse. Clearing
    one and not the other would leave drafts after an image conditioned on a
    conversation that is no longer there, and the reply would still look fine,
    because a bad draft is rejected rather than emitted.

    Verified live on a vision MTP model: 14 drafts on a text turn, 0 on the image
    turn, 15 on the text turn after it.
    """
    llm = make_bare_llama(
        _model_ptr=ctypes.c_void_p(1),
        _ctx_ptr=ctypes.c_void_p(2),
        _mtp_ctx_ptr=ctypes.c_void_p(3),
        supports_mtp=True,
    )
    llm._cached_tokens = [1, 2, 3]
    llm._kv_supported = True

    with patch("localm.inference.backends.llamacpp.llama.api") as mock_api:
        mock_api.llama_get_memory.side_effect = lambda ctx: ("mem", int(ctx.value))
        llm._reset_kv_for_image()

    cleared = {call.args[0] for call in mock_api.llama_memory_clear.call_args_list}
    assert ("mem", 2) in cleared, "the main cache was not cleared for the image eval"
    assert ("mem", 3) in cleared, (
        "the draft cache survived an image turn, so drafts after an image would be "
        "conditioned on a conversation that is no longer in the main cache")
    assert llm._cached_tokens == []


def test_a_stopped_session_stops_reporting_mtp_support():
    """supports_mtp is read from the load response, and the child can turn
    speculation off after that, so later calls have to correct it.

    Without this the flag describes a session that stopped speculating hours
    ago, which is the same stale observable this whole area exists to remove.
    """
    from localm.inference.backends import gguf as gguf_mod

    for status in sorted(gguf_mod._MTP_STOPPED):
        backend = GgufBackend("test_model.gguf")
        backend._loaded = True
        backend._supports_mtp = True
        backend._record_mtp({"mtp_status": status, "mtp_active": False})
        assert backend.supports_mtp is False, (
            f"{status} stops speculation for this model, so supports_mtp must "
            "stop saying otherwise")
        assert backend.last_mtp_status == status


def test_a_call_that_merely_did_not_speculate_leaves_the_capability_alone():
    """An image turn does not speculate and must not be read as the model having
    lost the ability - the next text turn speculates normally."""
    backend = GgufBackend("test_model.gguf")
    backend._loaded = True
    backend._supports_mtp = True

    backend._record_mtp({"mtp_status": "ok:qwen35", "mtp_active": False})

    assert backend.supports_mtp is True
    assert backend.last_mtp_active is False    # this call did not speculate
    assert backend.last_mtp_status == "ok:qwen35"


def test_an_envelope_without_the_field_changes_nothing():
    """A child that does not report it must not be read as a stop."""
    backend = GgufBackend("test_model.gguf")
    backend._loaded = True
    backend._supports_mtp = True

    backend._record_mtp({"finish_reason": "stop"})

    assert backend.supports_mtp is True
    assert backend.last_mtp_status is None


def test_the_stopped_statuses_are_ones_the_child_can_actually_report():
    """A status in the stop set that the child never emits would be dead, and one
    the child emits that is missing from the set leaves the flag stale. Both are
    silent, so pin the set against llama.py's own vocabulary."""
    from pathlib import Path

    from localm.inference.backends import gguf as gguf_mod

    src = Path(inspect.getfile(LlamaCppModule)).read_text(encoding="utf-8")
    for status in sorted(gguf_mod._MTP_STOPPED):
        assert f'"{status}"' in src, (
            f"{status!r} is treated as a permanent stop but llama.py never "
            "reports it, so the entry is dead")


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
#  in the chain, and llama.cpp offers no way to rewind that: the sampler that
#  decides a speculation is the one whose state advances, and every token it is
#  told about has to be a token that is actually emitted. The harness below
#  records both sides.

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


def _arm_drafting(llm):
    """Give a bare LlamaCpp what the draft path requires.

    Drafting feeds the head the target's hidden state, so it is skipped entirely
    when there is none, which is the fail-closed behaviour these fixtures would
    otherwise exercise instead of the speculative path they are about. The batch
    helpers do real ctypes work that cannot run against a mock api, so they are
    stubbed here.
    """
    llm._mtp_wants_h = True
    llm._pending_h = object()
    llm._n_embd = 4
    llm._capture_h = lambda row=-1: True
    llm._create_draft_batch = lambda token, pos: (
        llm._create_batch([token], pos, logits_at_last_only=True), None, None)
    llm._free_draft_batch = staticmethod(lambda batch, original: None)
    return llm


def _run_generate(recorder, *, max_new_tokens, **kwargs):
    """Run _generate against *recorder*; return (yielded tokens, the mock api)."""
    llm = make_bare_llama(
        _model_ptr=ctypes.c_void_p(1),
        _ctx_ptr=ctypes.c_void_p(2),
        _mtp_ctx_ptr=ctypes.c_void_p(3),
        supports_mtp=True,
    )
    _arm_drafting(llm)
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
    top_p and the repetition window for every accepted speculative token."""
    rec = _SpecRecorder(head=[100, _SpecRecorder.EOG], draft=[101], verify=[101])

    tokens, _ = _run_generate(rec, max_new_tokens=4)

    assert ("VERIFY", 101) in rec.calls, (
        f"no verification went through the request sampler: {rec.calls}")
    assert tokens == [100, 101]


def test_mtp_accepted_draft_enters_the_chain_exactly_once():
    """llama_sampler_sample already accepts what it returns, so an accepted
    draft needs no second accept. A duplicate advances the repetition window
    twice and, with a grammar in the chain, throws across the C ABI."""
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
    couple of tokens in and reports a token budget it never reached.

    The cache here is modelled rather than mocked flat, so that the later
    decodes fail the way they do against a real memory module.
    """
    rec = _SpecRecorder(head=[600, 604, _SpecRecorder.EOG],
                        draft=[601], verify=[602])

    llm = make_bare_llama(
        _model_ptr=ctypes.c_void_p(1),
        _ctx_ptr=ctypes.c_void_p(2),
        _mtp_ctx_ptr=ctypes.c_void_p(3),
        supports_mtp=True,
    )
    _arm_drafting(llm)
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

    # The turn survives the stuck cell: the token stream is asserted before the
    # status flags.
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
    _arm_drafting(llm)
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
    main one, so every later draft would be conditioned on the wrong prefix."""
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
    and rejects a speculation.

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


# --- Real end-to-end proof, against a real MTP-head GGUF ---------------------
#
# Everything above drives _generate with a scripted api.* mock. temperature=0.0
# there makes _build_sampler's chain greedy too, so those tests cannot tell
# sampler from draft_sampler - the fixture's value space never intersects the
# defect's trigger space. This drives a real model with a real, non-greedy
# sampling config through real native decode instead.

_MTP_REPO = "unsloth/Qwen3.5-0.8B-MTP-GGUF"
_MTP_FILE = "Qwen3.5-0.8B-Q4_K_M.gguf"


@pytest.fixture(scope="module")
def real_mtp_model_path():
    from huggingface_hub import hf_hub_download
    try:
        return hf_hub_download(repo_id=_MTP_REPO, filename=_MTP_FILE)
    except Exception as e:
        pytest.skip(f"could not fetch {_MTP_REPO}/{_MTP_FILE}: {e}")


@pytest.mark.integration
@pytest.mark.real_gguf
def test_real_mtp_model_verification_is_distribution_exact(real_mtp_model_path):
    """The headline fidelity property, against a real MTP-head model: with the
    request's own sampler deciding verification (llama.py's speculative MTP
    block), MTP-enabled generation must produce byte-identical output to the
    same run with MTP disabled, given the same seed and the same non-greedy
    sampling config. Upstream's own speculative decoding is distribution-exact
    by construction (common_sampler_sample_and_accept_n); this is what breaks
    first if verification ever samples from draft_sampler again instead of the
    request's own chain.
    """
    from localm.inference.backends.llamacpp.llama import LlamaCpp
    from localm.inference.backends.llamacpp._loader import load_lib
    from localm.inference.backends.llamacpp import _api as api

    try:
        load_lib()
    except Exception as e:
        pytest.skip(f"native llama runtime not provisioned (run 'localm "
                    f"setup-llama'): {e}")

    seed = 20260901
    sampling = dict(temperature=0.8, top_p=0.95, top_k=40, repeat_penalty=1.1)
    messages = [{"role": "user",
                 "content": "Write a short paragraph about a cat exploring a garden."}]

    def _run(mtp_enabled):
        try:
            llm = LlamaCpp(real_mtp_model_path, n_ctx=2048, n_gpu_layers=99,
                            seed=seed, mtp_enabled=mtp_enabled)
        except Exception as e:
            pytest.skip(f"model failed to load on this machine: {e}")
        accepted = 0
        rejected = 0
        try:
            if mtp_enabled:
                last_draft = []
                original = api.llama_sampler_sample

                def _spy(sampler, ctx, idx):
                    nonlocal accepted, rejected
                    token = original(sampler, ctx, idx)
                    if llm._mtp_ctx_ptr is not None and ctx == llm._mtp_ctx_ptr:
                        last_draft.append(token)
                    elif ctx == llm._ctx_ptr and idx == 0:
                        if last_draft and token == last_draft[-1]:
                            accepted += 1
                        else:
                            rejected += 1
                        last_draft.clear()
                    return token

                api.llama_sampler_sample = _spy
                try:
                    out = llm.create_chat_completion(
                        messages, max_tokens=150, stream=False, seed=seed, **sampling)
                finally:
                    api.llama_sampler_sample = original
            else:
                out = llm.create_chat_completion(
                    messages, max_tokens=150, stream=False, seed=seed, **sampling)
            return out["choices"][0]["message"]["content"], accepted, rejected
        finally:
            llm.close()

    on_text, accepted, rejected = _run(mtp_enabled=True)
    off_text, _, _ = _run(mtp_enabled=False)

    if accepted == 0 and rejected == 0:
        pytest.skip("no draft/verify cycle observed on this run - MTP did not "
                    "engage, nothing to verify")

    assert len(on_text) >= 10, f"suspiciously short output: {on_text!r}"
    assert accepted > 0, "the accept path never ran - fixture did not exercise it"
    assert rejected > 0, "the reject path never ran - fixture did not exercise it"
    assert on_text == off_text, (
        "MTP-enabled output diverged from the MTP-disabled control with the "
        "same seed and sampling config - verification is no longer sampling "
        "through the request's own sampler")


# --------------------------------------------------------------------------- #
#  _generate_image: an image-bearing turn must never touch the draft context. #
#  test_an_image_turn_clears_the_draft_cache_too (above) pins the KV-reset    #
#  half in isolation; this drives the real generator end to end.              #
# --------------------------------------------------------------------------- #

def test_generate_image_never_samples_or_decodes_the_draft_context():
    """The vision decode loop has no draft context: it must never sample from
    or decode into _mtp_ctx_ptr, and mtp_active_this_call must read False
    afterwards even though supports_mtp (a MODEL capability) stays True."""
    llm = make_bare_llama(
        _model_ptr=ctypes.c_void_p(1),
        _ctx_ptr=ctypes.c_void_p(2),
        _mtp_ctx_ptr=ctypes.c_void_p(3),
        supports_mtp=True,
        mtp_active_this_call=True,   # as if a PRIOR text turn had speculated
    )
    llm._mtmd = MagicMock(marker="<image>")
    llm._mtmd.eval_into.return_value = 5   # position after the mtmd prefill
    llm._create_batch = MagicMock(return_value=MagicMock())
    llm._tokenizer.is_eog.side_effect = lambda t: t == _SpecRecorder.EOG

    messages = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:fake"}},
        {"type": "text", "text": "describe this"},
    ]}]

    with patch("localm.inference.backends.llamacpp.llama.api") as mock_api, \
         patch("localm.inference.backends.llamacpp.llama._apply_model_template",
               return_value=("prompt", None)), \
         patch("localm.inference.backends.llamacpp.llama._build_sampler",
               return_value=MagicMock()), \
         patch.object(LlamaCppModule.LlamaCpp, "_messages_with_markers",
                      return_value=(messages, [])):
        mock_api.llama_sampler_sample.side_effect = [100, 101, _SpecRecorder.EOG]
        mock_api.llama_decode.return_value = 0
        tokens = list(llm._generate_image(
            messages, max_new_tokens=8, temperature=0.8, top_k=40, top_p=0.95,
            repeat_penalty=1.1,
        ))

    assert tokens == [100, 101]
    assert llm.mtp_active_this_call is False, (
        "an image turn read as having speculated - supports_mtp staying True "
        "is a model capability, not a statement about this call")

    decode_ctxs = [call.args[0] for call in mock_api.llama_decode.call_args_list]
    assert llm._mtp_ctx_ptr not in decode_ctxs, (
        f"the draft context was decoded into during an image turn: {decode_ctxs}")
    sample_ctxs = [call.args[1] for call in mock_api.llama_sampler_sample.call_args_list]
    assert llm._mtp_ctx_ptr not in sample_ctxs, (
        f"the draft context was sampled during an image turn: {sample_ctxs}")
    # supports_mtp is unaffected - it describes the MODEL, not this request.
    assert llm.supports_mtp is True


# --------------------------------------------------------------------------- #
#  VRAM preflight: the MTP draft context must be charged, not silently free.  #
# --------------------------------------------------------------------------- #

def _mtp_sizing_backend(*, mtp_enabled=True, n_ctx=4096, ctx_auto=False):
    return GgufBackend("fake-model.gguf", n_gpu_layers=99, mtp_enabled=mtp_enabled,
                       n_ctx=n_ctx, ctx_auto=ctx_auto)


def test_mtp_draft_vram_is_zero_when_mtp_is_disabled():
    b = _mtp_sizing_backend(mtp_enabled=False)
    with patch("localm.model_manager.gguf.gguf_nextn_predict_layers",
               return_value=("qwen35", 1)) as mocked:
        assert b._mtp_draft_context_vram_bytes() == 0
    mocked.assert_not_called(), "mtp_enabled=False must short-circuit before any file read"


def test_mtp_draft_vram_is_zero_when_the_architecture_has_no_real_mtp_graph():
    # glm4moe declares the nextn metadata key but build_arch_graph ignores
    # gtype for it - the SAME false-positive _api.py's MTP_GRAPH_ARCHITECTURES
    # gate already refuses at load time. Sizing must agree, or it charges VRAM
    # for a context that will never actually be created.
    b = _mtp_sizing_backend()
    with patch("localm.model_manager.gguf.gguf_nextn_predict_layers",
               return_value=("glm4moe", 1)):
        assert b._mtp_draft_context_vram_bytes() == 0


def test_mtp_draft_vram_is_zero_when_no_nextn_layers_are_declared():
    b = _mtp_sizing_backend()
    with patch("localm.model_manager.gguf.gguf_nextn_predict_layers",
               return_value=("qwen35", 0)):
        assert b._mtp_draft_context_vram_bytes() == 0


def test_mtp_draft_vram_charges_kv_plus_the_flat_overhead_when_eligible():
    # n_ctx below the 2048 cap, so this test is about the arithmetic alone -
    # see test_mtp_draft_vram_caps_the_draft_context_at_2048_tokens for the cap.
    b = _mtp_sizing_backend(n_ctx=1024)
    with patch("localm.model_manager.gguf.gguf_nextn_predict_layers",
               return_value=("qwen35", 1)), \
         patch("localm.model_manager.gguf.gguf_mtp_draft_kv_bytes_per_token",
               return_value=1000):
        charge = b._mtp_draft_context_vram_bytes()
    assert charge == 1024 * 1000 + GgufBackend._VRAM_OVERHEAD_BYTES


def test_mtp_draft_vram_caps_the_draft_context_at_2048_tokens():
    # The main n_ctx is far above the 2048 cap llama.py's own
    # cp_mtp.n_ctx = min(n_ctx, 2048) actually allocates.
    b = _mtp_sizing_backend(n_ctx=65536)
    with patch("localm.model_manager.gguf.gguf_nextn_predict_layers",
               return_value=("qwen35", 1)), \
         patch("localm.model_manager.gguf.gguf_mtp_draft_kv_bytes_per_token",
               return_value=1000):
        charge = b._mtp_draft_context_vram_bytes()
    assert charge == 2048 * 1000 + GgufBackend._VRAM_OVERHEAD_BYTES


def test_mtp_draft_vram_is_memoised_per_instance():
    b = _mtp_sizing_backend()
    with patch("localm.model_manager.gguf.gguf_nextn_predict_layers",
               return_value=("qwen35", 1)) as mocked, \
         patch("localm.model_manager.gguf.gguf_mtp_draft_kv_bytes_per_token",
               return_value=1000):
        b._mtp_draft_context_vram_bytes()
        b._mtp_draft_context_vram_bytes()
    assert mocked.call_count == 1, "the GGUF header should be probed once per load"


def test_mtp_draft_vram_never_raises_on_a_probe_failure():
    b = _mtp_sizing_backend()
    with patch("localm.model_manager.gguf.gguf_nextn_predict_layers",
               side_effect=RuntimeError("boom")):
        assert b._mtp_draft_context_vram_bytes() == 0


def _mtp_vram_levels(free, total):
    """Patch every VRAM-reading path GgufBackend._check_vram/_auto_ctx_max/
    _auto_gpu_layers can fall through to, so a bare 'free, total' fully
    determines what they see."""
    from contextlib import ExitStack
    from localm.inference.backends.llamacpp import _loader
    stack = ExitStack()
    stack.enter_context(patch.object(
        GgufBackend, "_free_total_vram_bytes", return_value=(free, total)))
    stack.enter_context(patch.object(
        _loader, "gpu_memory_isolated", return_value=(free, total)))
    stack.enter_context(patch.object(
        GgufBackend, "_device_global_free_bytes", return_value=None))
    return stack


def test_check_vram_raises_when_the_mtp_draft_context_pushes_over_the_ceiling():
    """_check_vram's hard 'can never fit' refusal must account for the MTP
    draft context's own VRAM, not just weights + the main KV cache - the same
    total that fits without the draft charge must refuse with it."""
    import pytest

    def backend():
        b = _mtp_sizing_backend(n_ctx=4096)
        b._model_bytes = lambda: 3 * 1024 ** 3
        b._kv_bytes_per_token = lambda: 0
        return b

    total = 3 * 1024 ** 3 + GgufBackend._VRAM_OVERHEAD_BYTES + 100_000

    b0 = backend()
    b0._mtp_draft_context_vram_bytes = lambda: 0
    with _mtp_vram_levels(total, total):
        b0._check_vram()          # fits without the MTP charge - no raise

    b1 = backend()
    b1._mtp_draft_context_vram_bytes = lambda: 2 * 1024 ** 3
    with _mtp_vram_levels(total, total):
        with pytest.raises(RuntimeError, match="Context too large"):
            b1._check_vram()


def test_auto_ctx_max_shrinks_the_budget_by_the_mtp_draft_context():
    def backend():
        b = _mtp_sizing_backend(n_ctx=4096, ctx_auto=True)
        b._model_bytes = lambda: 1 * 1024 ** 3
        b._kv_bytes_per_token = lambda: 50_000
        return b

    # free is picked so the UNCAPPED budget (2 GB worth of tokens at 50000
    # bytes/token) sits well below _AUTO_CTX_MAX=65536 in both arms - a
    # difference the cap would otherwise hide.
    free = 1 * 1024 ** 3 + GgufBackend._VRAM_OVERHEAD_BYTES + 2_000_000_000
    total = free

    b_off = backend()
    b_off._mtp_draft_context_vram_bytes = lambda: 0
    with _mtp_vram_levels(free, total):
        ctx_off = b_off._auto_ctx_max()

    b_on = backend()
    b_on._mtp_draft_context_vram_bytes = lambda: 500 * 1024 ** 2
    with _mtp_vram_levels(free, total):
        ctx_on = b_on._auto_ctx_max()

    assert ctx_on < ctx_off, (
        f"an MTP-reserving load auto-sized the SAME context ceiling "
        f"({ctx_on} vs {ctx_off}) as one that reserves nothing for it")


def test_auto_gpu_layers_offloads_fewer_when_mtp_will_allocate_a_draft_context():
    """End to end, through the real gate: an MTP-enabled, MTP-eligible load
    reserves the draft context's VRAM before deciding how many layers fit, so
    it offloads fewer of them than the identical load with MTP off."""
    def backend(mtp_enabled):
        b = _mtp_sizing_backend(mtp_enabled=mtp_enabled, n_ctx=4096)
        b.n_gpu_layers_auto = True
        b._model_bytes = lambda: 4 * 1024 ** 3
        b._cached_layer_count = lambda: 32
        b._kv_bytes_per_token = lambda: 1000
        return b

    free, total = 3 * 1024 ** 3, 8 * 1024 ** 3

    with patch("localm.model_manager.gguf.gguf_nextn_predict_layers",
               return_value=("qwen35", 1)), \
         patch("localm.model_manager.gguf.gguf_mtp_draft_kv_bytes_per_token",
               return_value=2000):
        with _mtp_vram_levels(free, total):
            n_with_mtp = backend(True)._auto_gpu_layers()
        with _mtp_vram_levels(free, total):
            n_without_mtp = backend(False)._auto_gpu_layers()

    assert 0 < n_with_mtp < n_without_mtp <= 99, (
        f"n_with_mtp={n_with_mtp} n_without_mtp={n_without_mtp}: MTP-enabled "
        f"sizing must offload strictly fewer layers once it reserves VRAM "
        f"for its own draft context")


def test_the_mtp_draft_charge_does_not_scale_with_the_split_device_count():
    """On an N-device split the charged overhead is N main-context compute
    buffers plus exactly ONE for the MTP draft context, never N of them.

    Arm 1 accepts a combined total sized for 3 main buffers plus one draft
    buffer, carrying less than one buffer of slack. Arm 2 is the control: the
    same total at 4 devices is refused.
    """
    # llama.cpp b10375: every src/models/ graph_mtp builds ONE block in the
    # nextn tail range at or above hparams.n_layer(); src/llama-model.cpp
    # :1360-1366 assigns that block and the output head to the last device.
    ov = GgufBackend._VRAM_OVERHEAD_BYTES
    weights = 3 * 1024 ** 3
    draft_kv_per_token = 1000
    draft_ctx = 1024
    slack = 100_000
    total = weights + 3 * ov + (draft_ctx * draft_kv_per_token + ov) + slack

    def check(devices):
        b = _mtp_sizing_backend(n_ctx=draft_ctx)
        b._model_bytes = lambda: weights
        b._kv_bytes_per_token = lambda: 0
        with patch("localm.model_manager.gguf.gguf_nextn_predict_layers",
                   return_value=("qwen35", 1)), \
             patch("localm.model_manager.gguf.gguf_mtp_draft_kv_bytes_per_token",
                   return_value=draft_kv_per_token), \
             patch.object(GgufBackend, "_split_free_total_bytes",
                          return_value=(total, total, devices)):
            b._check_vram()

    check(3)

    with pytest.raises(RuntimeError, match="Context too large"):
        check(4)

# --- mtp_enabled override plumbing + the bench-mtp comparison ----------------


@pytest.mark.parametrize("cfg_value,override,expected", [
    (False, True, True),
    (True, False, False),
    (False, None, False),
    (True, None, True),
])
def test_create_backend_mtp_override_beats_the_config_key(
        cfg_value, override, expected):
    """An explicit mtp_enabled= wins over the stored setting; None reads it.

    bench-mtp measures both arms in one process against one config, so without
    this the MTP-on arm would silently re-read mtp_enabled and both arms would
    run identically.
    """
    from localm.inference import engine as engine_mod

    captured = {}

    class _FakeBackend:
        def __init__(self, *a, **kw):
            captured.update(kw)

    cfg = dict(DEFAULT_CONFIG)
    cfg["mtp_enabled"] = cfg_value
    with patch.object(engine_mod, "load_config", return_value=cfg), \
         patch("localm.inference.backends.gguf.GgufBackend", _FakeBackend):
        engine_mod.create_backend("model.gguf", mtp_enabled=override)

    assert captured["mtp_enabled"] is expected


def test_engine_forwards_the_mtp_override_to_create_backend():
    """Engine(mtp_enabled=...) reaches create_backend rather than being dropped."""
    from localm.inference import engine as engine_mod

    seen = {}

    def _fake_create_backend(model_path, **kw):
        seen.update(kw)
        return MagicMock()

    with patch.object(engine_mod, "create_backend", _fake_create_backend):
        engine_mod.Engine("model.gguf", mtp_enabled=True)
    assert seen["mtp_enabled"] is True


def _bench_mtp_result(rates_off, rates_on, supports=True, status=None,
                      placement=None):
    """Build a _mtp_probe_arm double returning fixed rates per arm."""
    def _arm(model_path, display, mtp_enabled, gen_tokens, ctx, gpu_layers):
        return ((rates_on if mtp_enabled else rates_off), supports, status,
                placement)
    return _arm


def test_bench_mtp_stops_when_the_model_has_no_draft_head(cli_runner):
    """A model without a usable MTP head gets a plain answer, not a comparison.

    Reporting a ratio here would attribute ordinary run-to-run noise to a
    setting that is doing nothing for this model.
    """
    from localm.cli import models as models_mod

    # Rates that WOULD read as a 1.40x win, so dropping the early return prints
    # a verdict instead of nothing and this test fails rather than passing on a
    # tie it arranged for itself.
    with patch.object(models_mod, "get_model_info",
                      return_value=("model.gguf", None)), \
         patch.object(models_mod, "_mtp_probe_arm",
                      _bench_mtp_result([50.0], [70.0], supports=False,
                                        status="unsupported_arch")):
        res = cli_runner.invoke(models_mod.bench_mtp, ["model.gguf"])

    assert res.exit_code == 0, res.output
    assert "no usable MTP draft head" in res.output
    assert "decode tok/s" not in res.output, (
        "the comparison table was printed for a model with no draft head")
    assert "faster" not in res.output


@pytest.mark.parametrize("off,on,phrase", [
    ([50.0], [70.0], "MTP is 1.40x faster"),
    ([100.0], [60.0], "MTP is slower"),
    ([100.0], [101.0], "No meaningful difference"),
])
def test_bench_mtp_reports_the_measured_verdict(cli_runner, off, on, phrase):
    from localm.cli import models as models_mod

    with patch.object(models_mod, "get_model_info",
                      return_value=("model.gguf", None)), \
         patch.object(models_mod, "_mtp_probe_arm",
                      _bench_mtp_result(off, on)):
        res = cli_runner.invoke(models_mod.bench_mtp,
                                ["model.gguf", "--rounds", "1"])

    assert res.exit_code == 0, res.output
    assert phrase in res.output


def test_bench_mtp_names_cpu_offload_when_mtp_loses(cli_runner):
    """A partially offloaded model is told why, since that is fixable."""
    from localm.cli import models as models_mod

    placement = {"gpu_layers_offloaded": 12, "gpu_layers_total": 28,
                 "degraded": True}
    with patch.object(models_mod, "get_model_info",
                      return_value=("model.gguf", None)), \
         patch.object(models_mod, "_mtp_probe_arm",
                      _bench_mtp_result([100.0], [60.0], placement=placement)):
        res = cli_runner.invoke(models_mod.bench_mtp,
                                ["model.gguf", "--rounds", "1"])

    assert "12 of 28 layers are on the GPU" in res.output


def test_bench_mtp_never_writes_the_setting(cli_runner):
    """The command measures and reports; applying the result stays the user's
    call, so a run must leave mtp_enabled exactly as it found it."""
    from localm.cli import models as models_mod
    from localm.config import load_config, save_config

    cfg = load_config()
    cfg["mtp_enabled"] = False
    save_config(cfg)

    with patch.object(models_mod, "get_model_info",
                      return_value=("model.gguf", None)), \
         patch.object(models_mod, "_mtp_probe_arm",
                      _bench_mtp_result([50.0], [70.0])):
        res = cli_runner.invoke(models_mod.bench_mtp,
                                ["model.gguf", "--rounds", "1"])

    assert res.exit_code == 0, res.output
    assert load_config()["mtp_enabled"] is False
