# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for Multi-Token Prediction (MTP) model support."""

import contextlib
import ctypes
import inspect
import threading
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
    LlamaModelParamsV3,
)
from localm.inference.backends.llamacpp import _api as api
from localm.inference.engine import Engine
from localm.inference.backends.llamacpp import llama as LlamaCppModule
from tests._bare_llama import make_bare_llama, prime_after_prefill, stub_mtp_native
from tests._fake_mtmd import fake_vision_prompt
from tests._real_gguf import fetch_gguf, require_native_runtime


def test_mtp_constants_and_structs():
    """Verify MTP context type constants and model param struct offsets."""
    assert LLAMA_CONTEXT_TYPE_DEFAULT == 0
    assert LLAMA_CONTEXT_TYPE_MTP == 1
    for cls in (LlamaModelParamsV2, LlamaModelParamsV3):
        assert hasattr(cls, "load_mtp")
        mp = cls()
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

    llama.cpp reads nextn_predict_layers for every architecture, so the key can
    appear on one whose build_arch_graph ignores the MTP graph type; an MTP
    context there is a second full decoder rather than a draft head. glm4
    (dense GLM-4) is outside MTP_GRAPH_ARCHITECTURES at the pinned build.
    (glm4moe was the real published case until upstream gave it an MTP graph;
    it is now in the allowlist, which is why the example moved.)
    """
    supported, reason = _detect("glm4", {"glm4.nextn_predict_layers": "1"})
    assert supported is False
    assert reason == "no-mtp-graph:glm4"


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

    for params in (_structs.LlamaContextParamsV1, _structs.LlamaContextParamsV2,
                   _structs.LlamaContextParamsV3):
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
        assert f'"{status}' in src, (
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
    stubbed here, and a mocked prefill leaves the state a real one does.
    Pacing is held open so every step drafts, as the scripted samples assume.
    """
    stub_mtp_native(llm)
    llm._prefill_fresh_context = MagicMock(
        side_effect=lambda tokens, needed: prime_after_prefill(llm, tokens))
    # Every step drafts: no plain probe step, so no pause decision either.
    llm._draft_pacer = LlamaCppModule._DraftPacer(probe_every=1 << 30, bootstrap_every=1 << 30)
    return llm


def _run_generate(recorder, *, max_new_tokens, decode=None, llm_holder=None, **kwargs):
    """Run _generate against *recorder*; return (yielded tokens, the mock api).

    *decode* replaces the recorder's decode; *llm_holder*, a list, receives the
    LlamaCpp the run used.
    """
    llm = make_bare_llama(
        _model_ptr=ctypes.c_void_p(1),
        _ctx_ptr=ctypes.c_void_p(2),
        _mtp_ctx_ptr=ctypes.c_void_p(3),
        supports_mtp=True,
        mtp_status="ok:qwen35",
    )
    if llm_holder is not None:
        llm_holder.append(llm)
    _arm_drafting(llm)
    llm._tokenizer.is_eog.side_effect = lambda t: t == _SpecRecorder.EOG
    llm._fit_generation_budget = lambda n_prompt, max_new: max_new
    llm._can_reuse_kv = lambda needed: False
    llm._create_batch = MagicMock(return_value=MagicMock())

    with patch("localm.inference.backends.llamacpp.llama.api") as mock_api, \
         patch("localm.inference.backends.llamacpp.llama._build_sampler",
               return_value=recorder.main_sampler):
        mock_api.llama_sampler_chain_init.return_value = recorder.draft_sampler
        mock_api.llama_sampler_sample.side_effect = recorder.sample
        mock_api.llama_sampler_accept.side_effect = recorder.accept
        mock_api.llama_decode.side_effect = decode or recorder.decode
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


def test_a_grammar_request_drafts_and_never_accepts_a_draft_into_its_sampler():
    """A grammar in the request's sampler does not stop drafting, and no token
    is ever accepted into that sampler except by sampling it."""
    rec = _SpecRecorder(head=[400, _SpecRecorder.EOG], draft=[401], verify=[401])

    tokens, mock_api = _run_generate(rec, max_new_tokens=4, grammar='root ::= "a"')

    assert tokens == [400, 401]
    assert ["DRAFT", "VERIFY"] == [s for s in rec.shapes() if s in ("DRAFT", "VERIFY")]
    assert rec.told_main == [400, 401, _SpecRecorder.EOG]
    mock_api.llama_sampler_accept.assert_not_called()


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
        mock_api.llama_sampler_chain_init.return_value = rec.draft_sampler
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


# --- _gen_lock is never held across a yield ---------------------------------

_LOCK_WAIT_S = 2.0


@contextlib.contextmanager
def _suspended_on_an_accepted_draft():
    """Yield (llm, gen, mock_api) with *gen* suspended at the yield of an
    accepted draft token, on the calling thread."""
    rec = _SpecRecorder(head=[100, _SpecRecorder.EOG], draft=[101], verify=[101])
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
    llm._create_batch = MagicMock(return_value=MagicMock())

    with patch("localm.inference.backends.llamacpp.llama.api") as mock_api, \
         patch("localm.inference.backends.llamacpp.llama._build_sampler",
               return_value=rec.main_sampler):
        mock_api.llama_sampler_chain_init.return_value = rec.draft_sampler
        mock_api.llama_sampler_sample.side_effect = rec.sample
        mock_api.llama_sampler_accept.side_effect = rec.accept
        mock_api.llama_decode.side_effect = rec.decode
        gen = llm._generate(
            prompt_tokens=[1, 2], max_new_tokens=4, temperature=0.8,
            top_k=40, top_p=0.95, repeat_penalty=1.1)
        try:
            assert next(gen) == 100
            assert next(gen) == 101
            yield llm, gen, mock_api
        finally:
            gen.close()


def _on_thread(fn):
    """Run *fn* on a daemon thread; return (thread, outcome dict)."""
    outcome = {}

    def _run():
        try:
            outcome["value"] = fn()
        except BaseException as exc:
            outcome["error"] = exc

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return thread, outcome


def test_gen_lock_is_free_while_a_draft_token_is_suspended_at_the_yield():
    """A consumer holding the generator at an accepted draft token must not
    hold _gen_lock, or every other thread's native call waits on the consumer."""
    with _suspended_on_an_accepted_draft() as (llm, gen, _api):
        def _try_lock():
            if llm._gen_lock.acquire(timeout=_LOCK_WAIT_S):
                llm._gen_lock.release()
                return True
            return False

        thread, outcome = _on_thread(_try_lock)
        thread.join(_LOCK_WAIT_S + 3)

        assert outcome.get("value") is True, (
            "another thread could not take _gen_lock while the generator was "
            f"suspended at a yielded draft token: {outcome}")


def test_close_from_another_thread_frees_native_state_while_suspended_at_a_draft_token():
    """close() on a second thread returns promptly and frees the native
    context even though the generator is parked at an accepted draft token."""
    with _suspended_on_an_accepted_draft() as (llm, gen, mock_api):
        thread, outcome = _on_thread(llm.close)
        thread.join(_LOCK_WAIT_S + 3)

        assert llm._ctx_ptr is None and llm._model_ptr is None, (
            "close() did not free the native state while the generator was "
            "suspended at a yielded draft token")
        mock_api.llama_free.assert_called()
        assert not thread.is_alive()
        assert "error" not in outcome, outcome


def test_a_draft_token_generator_can_be_resumed_on_another_thread():
    """Resuming a generator suspended at an accepted draft token on a thread
    other than the one that suspended it does not fail on lock ownership."""
    with _suspended_on_an_accepted_draft() as (llm, gen, _api):
        def _resume():
            try:
                return ("token", next(gen))
            except StopIteration:
                return ("done", None)

        thread, outcome = _on_thread(_resume)
        thread.join(_LOCK_WAIT_S + 3)

        assert not thread.is_alive()
        assert "error" not in outcome, (
            f"resuming on another thread raised: {outcome.get('error')!r}")
        assert outcome["value"] == ("done", None)


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
    return fetch_gguf(_MTP_REPO, _MTP_FILE)


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
    from localm.inference.backends.llamacpp import _api as api

    require_native_runtime()

    seed = 20260901
    sampling = dict(temperature=0.8, top_p=0.95, top_k=40, repeat_penalty=1.1)
    messages = [{"role": "user",
                 "content": "Write a short paragraph about a cat exploring a garden."}]

    def _run(mtp_enabled):
        llm = LlamaCpp(real_mtp_model_path, n_ctx=2048, n_gpu_layers=99,
                       seed=seed, mtp_enabled=mtp_enabled)
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


_TOOL_SYSTEM = (
    "You can search the web. To search, reply with exactly one block of the form "
    '<tool_call>{"name": "web_search", "args": {"query": "..."}}</tool_call> and nothing else.')


@pytest.mark.integration
@pytest.mark.real_gguf
@pytest.mark.parametrize("case", ["json-object", "lazy-tool-call-plain-reply",
                                  "lazy-tool-call-called", "forced-tool-call"])
def test_real_mtp_model_with_a_grammar_drafts_and_matches_mtp_off(real_mtp_model_path, case):
    """With a grammar in the request's sampler a real MTP model drafts, and its
    reply is byte-identical to the same request without MTP: the grammar only
    ever sees the emitted tokens, in order."""
    from localm.inference import gbnf
    from localm.inference.backends.llamacpp.llama import LlamaCpp

    require_native_runtime()

    seed = 20261007
    if case == "json-object":
        messages = [{"role": "user", "content": "Describe a cat as a JSON object with "
                                                "the keys name, color, age and hobbies."}]
        kwargs = dict(grammar=gbnf.JSON_OBJECT, temperature=0.8, top_p=0.95, top_k=40)
    elif case == "lazy-tool-call-plain-reply":
        messages = [{"role": "system", "content": _TOOL_SYSTEM},
                    {"role": "user", "content": "Explain in two paragraphs how a fridge works."}]
        kwargs = dict(grammar=gbnf.TOOL_CALL_SINGLE, grammar_lazy=True,
                      grammar_triggers=[gbnf.TOOL_CALL_TRIGGER], temperature=0.0)
    elif case == "lazy-tool-call-called":
        messages = [{"role": "system", "content": _TOOL_SYSTEM},
                    {"role": "user", "content": "Search the web for today's weather in Vienna."}]
        kwargs = dict(grammar=gbnf.TOOL_CALL_SINGLE, grammar_lazy=True,
                      grammar_triggers=[gbnf.TOOL_CALL_TRIGGER], temperature=0.0)
    else:
        messages = [{"role": "system", "content": _TOOL_SYSTEM},
                    {"role": "user", "content": "Search the web for the population of Graz."}]
        kwargs = dict(grammar=gbnf.TOOL_CALLS_ONLY, temperature=0.0)

    def _run(mtp_enabled):
        llm = LlamaCpp(real_mtp_model_path, n_ctx=2048, n_gpu_layers=99,
                       seed=seed, mtp_enabled=mtp_enabled)
        try:
            if mtp_enabled:
                # Every step drafts: the pacer must not pause this short reply.
                llm._draft_pacer = LlamaCppModule._DraftPacer(
                    probe_every=1 << 30, bootstrap_every=1 << 30)
            out = llm.create_chat_completion(messages, max_tokens=160, stream=False,
                                             seed=seed, repeat_penalty=1.0, **kwargs)
            return (out["choices"][0]["message"]["content"], llm.mtp_drafted,
                    llm.mtp_accepted, llm.mtp_call_status)
        finally:
            llm.close()

    on_text, drafted, accepted, status = _run(True)
    off_text, _, _, _ = _run(False)

    assert len(on_text) >= 10, f"suspiciously short output: {on_text!r}"
    assert status == "", status
    assert drafted > 0, "MTP never drafted on a grammar request"
    assert on_text == off_text, (case, on_text, off_text)
    if case in ("lazy-tool-call-called", "forced-tool-call"):
        assert "<tool_call>" in on_text, on_text


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
    llm._mtmd = MagicMock(marker="<image>", encode_count=0)
    llm._mtmd.tokenize.return_value = fake_vision_prompt(text_tokens=(1, 2, 3, 4, 5))
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
        mock_api.llama_n_ctx.return_value = 4096
        tokens = list(llm._generate_image(
            messages, max_new_tokens=8, temperature=0.8, top_k=40, top_p=0.95,
            repeat_penalty=1.1,
        ))

    assert tokens == [100, 101]
    assert llm.mtp_active_this_call is False, (
        "an image turn read as having speculated - supports_mtp staying True "
        "is a model capability, not a statement about this call")
    assert llm.mtp_skipped == "image"

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
    # An architecture outside MTP_GRAPH_ARCHITECTURES declaring the nextn
    # metadata key - the SAME false-positive _api.py's gate already refuses at
    # load time. Sizing must agree, or it charges VRAM for a context that will
    # never actually be created.
    b = _mtp_sizing_backend()
    with patch("localm.model_manager.gguf.gguf_nextn_predict_layers",
               return_value=("glm4", 1)):
        assert b._mtp_draft_context_vram_bytes() == 0


def test_mtp_draft_vram_is_zero_when_no_nextn_layers_are_declared():
    b = _mtp_sizing_backend()
    with patch("localm.model_manager.gguf.gguf_nextn_predict_layers",
               return_value=("qwen35", 0)):
        assert b._mtp_draft_context_vram_bytes() == 0


def test_mtp_draft_vram_charges_kv_plus_the_flat_overhead_when_eligible():
    b = _mtp_sizing_backend(n_ctx=1024)
    with patch("localm.model_manager.gguf.gguf_nextn_predict_layers",
               return_value=("qwen35", 1)), \
         patch("localm.model_manager.gguf.gguf_mtp_draft_kv_bytes_per_token",
               return_value=1000):
        charge = b._mtp_draft_context_vram_bytes()
    assert charge == 1024 * 1000 + GgufBackend._VRAM_OVERHEAD_BYTES


def test_mtp_draft_vram_scales_with_the_main_context_past_2048_tokens():
    # The draft context is created at the main context's size, so its KV charge
    # is not capped.
    b = _mtp_sizing_backend(n_ctx=65536)
    with patch("localm.model_manager.gguf.gguf_nextn_predict_layers",
               return_value=("qwen35", 1)), \
         patch("localm.model_manager.gguf.gguf_mtp_draft_kv_bytes_per_token",
               return_value=1000):
        charge = b._mtp_draft_context_vram_bytes()
    assert charge == 65536 * 1000 + GgufBackend._VRAM_OVERHEAD_BYTES


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
                      placement=None, counts=(10, 8), texts_off=("a", "b", "c"),
                      texts_on=None, seen=None):
    """Build a _mtp_probe_arm double returning fixed rates per arm; *seen*
    collects the draft_tokens each MTP-on arm was asked for."""
    def _arm(model_path, display, mtp_enabled, gen_tokens, ctx, gpu_layers,
             draft_tokens=None):
        if seen is not None and mtp_enabled:
            seen.append(draft_tokens)
        if mtp_enabled:
            return (rates_on, supports, status, placement, counts,
                    list(texts_on if texts_on is not None else texts_off))
        return (rates_off, supports, status, placement, (0, 0), list(texts_off))
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
    with patch.object(models_mod, "get_operator_model_info",
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

    with patch.object(models_mod, "get_operator_model_info",
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
    with patch.object(models_mod, "get_operator_model_info",
                      return_value=("model.gguf", None)), \
         patch.object(models_mod, "_mtp_probe_arm",
                      _bench_mtp_result([100.0], [60.0], placement=placement)):
        res = cli_runner.invoke(models_mod.bench_mtp,
                                ["model.gguf", "--rounds", "1"])

    assert "12/28 layers on the GPU, the rest on the CPU" in res.output


def test_bench_mtp_never_writes_the_setting(cli_runner):
    """The command measures and reports; applying the result stays the user's
    call, so a run must leave mtp_enabled exactly as it found it."""
    from localm.cli import models as models_mod
    from localm.config import load_config, save_config

    cfg = load_config()
    cfg["mtp_enabled"] = False
    save_config(cfg)

    with patch.object(models_mod, "get_operator_model_info",
                      return_value=("model.gguf", None)), \
         patch.object(models_mod, "_mtp_probe_arm",
                      _bench_mtp_result([50.0], [70.0])):
        res = cli_runner.invoke(models_mod.bench_mtp,
                                ["model.gguf", "--rounds", "1"])

    assert res.exit_code == 0, res.output
    assert load_config()["mtp_enabled"] is False


# --------------------------------------------------------------------------- #
#  Draft context sized to the main context; per-call stop; status kinds       #
# --------------------------------------------------------------------------- #

def _context_params():
    return SimpleNamespace(ctx_type=0, n_ctx=0, n_batch=0, n_ubatch=0,
                           offload_kqv=True, n_threads=0, n_threads_batch=0,
                           n_rs_seq=0)


def _context_factory(mock_api, refuse_draft=False):
    """Make llama_init_from_model record each context's params; the context
    asked for with ctx_type MTP is the draft one."""
    made = []

    def init(model, cp):
        is_draft = cp.ctx_type == LLAMA_CONTEXT_TYPE_MTP
        ptr = None if (is_draft and refuse_draft) else ctypes.c_void_p(100 + len(made))
        made.append(SimpleNamespace(draft=is_draft, n_ctx=cp.n_ctx, n_batch=cp.n_batch,
                                    offload_kqv=cp.offload_kqv, ptr=ptr))
        return ptr

    mock_api.llama_context_default_params.side_effect = _context_params
    mock_api.llama_init_from_model.side_effect = init
    mock_api.llama_set_embeddings_nextn.return_value = True
    mock_api.llama_model_n_embd.return_value = 4
    mock_api.llama_model_mtp_support.return_value = (True, "ok:qwen35")
    mock_api.llama_decode.return_value = 0
    return made


def _growing_llama(target):
    llm = make_bare_llama(
        _model_ptr=ctypes.c_void_p(1),
        _ctx_ptr=ctypes.c_void_p(2),
        _mtp_ctx_ptr=ctypes.c_void_p(3),
        supports_mtp=True,
        mtp_status="ok:qwen35",
    )
    stub_mtp_native(llm)
    llm._mtp_ctx_capacity = 2048
    llm._target_ctx = lambda needed: target
    return llm


def test_a_grown_main_context_gets_a_draft_context_of_the_same_size():
    """The draft context is recreated at the main context's size, so a prompt
    longer than 2048 tokens still speculates on the next request."""
    llm = _growing_llama(target=8192)
    batches = []
    llm._create_batch = lambda tokens, pos, **kw: (
        batches.append((len(tokens), pos)) or SimpleNamespace())

    with patch("localm.inference.backends.llamacpp.llama.api") as mock_api:
        made = _context_factory(mock_api)
        llm._prefill_fresh_context(list(range(2335)), 2400)

    main, draft = made
    assert (main.draft, main.n_ctx) == (False, 8192)
    assert (draft.draft, draft.n_ctx) == (True, 8192), made
    assert draft.n_batch == 2048
    assert llm._mtp_ctx_capacity == 8192
    assert llm._mtp_ctx_ptr is draft.ptr
    # The 2335-token prompt was mirrored into the draft cache in two chunks.
    draft_decodes = [c for c in mock_api.llama_decode.call_args_list
                     if c.args[0] is draft.ptr]
    assert len(draft_decodes) == 2
    assert (llm.supports_mtp, llm._mtp_usable, llm.mtp_status) == (True, True, "ok:qwen35")


def test_the_main_context_exposes_its_hidden_state_before_its_first_decode():
    """A recreated main context has to be told to expose the next-n state again,
    or no draft after the growth has a hidden state to read."""
    llm = _growing_llama(target=8192)
    llm._create_batch = lambda tokens, pos, **kw: SimpleNamespace()
    order = []

    with patch("localm.inference.backends.llamacpp.llama.api") as mock_api:
        _context_factory(mock_api)
        mock_api.llama_set_embeddings_nextn.side_effect = (
            lambda ctx, *a: order.append(("expose", ctx.value)) or True)
        mock_api.llama_decode.side_effect = (
            lambda ctx, batch: order.append(("decode", ctx.value)) or 0)
        llm._prefill_fresh_context([1, 2, 3], 10)

    main_ptr = llm._ctx_ptr.value
    assert order.index(("expose", main_ptr)) < order.index(("decode", main_ptr)), order


def test_a_draft_context_that_cannot_be_recreated_stops_speculation_and_says_why():
    llm = _growing_llama(target=8192)
    llm._create_batch = lambda tokens, pos, **kw: SimpleNamespace()

    with patch("localm.inference.backends.llamacpp.llama.api") as mock_api:
        _context_factory(mock_api, refuse_draft=True)
        llm._prefill_fresh_context([1, 2, 3], 10)

    assert llm._mtp_ctx_ptr is None
    assert llm.mtp_status == "context-refused"
    assert (llm.supports_mtp, llm._mtp_usable) == (False, False)


def test_the_draft_context_follows_the_main_contexts_kv_placement():
    llm = _growing_llama(target=8192)
    llm._create_batch = lambda tokens, pos, **kw: SimpleNamespace()
    llm._vram_check = lambda target, current: False

    with patch("localm.inference.backends.llamacpp.llama.api") as mock_api:
        made = _context_factory(mock_api)
        llm._prefill_fresh_context([1, 2, 3], 10)

    assert [m.offload_kqv for m in made] == [False, False]


def test_a_draft_cache_left_stale_by_a_failed_reply_is_refilled_from_the_whole_prompt():
    def run(stale):
        llm = _growing_llama(target=4096)
        llm._mtp_ctx_capacity = 4096
        llm._mtp_draft_stale = stale
        llm._cached_tokens = [1, 2, 3]
        llm._draft_pos = 3
        llm._can_reuse_kv = lambda needed: True
        decoded = {2: [], 3: []}
        llm._create_batch = lambda tokens, pos, **kw: SimpleNamespace(tokens=list(tokens), pos=pos)
        with patch("localm.inference.backends.llamacpp.llama.api") as mock_api:
            mock_api.llama_memory_seq_rm.return_value = True
            mock_api.llama_decode.side_effect = (
                lambda ctx, batch: decoded[ctx.value].append((batch.tokens, batch.pos)) or 0)
            llm._prefill_with_reuse([1, 2, 3, 4])
        return llm, decoded

    llm, decoded = run(stale=True)
    # The draft cache is refilled from position 0: the kept prefix, then the
    # suffix the main context decoded.
    assert decoded[3] == [([1, 2, 3], 0), ([4], 3)]
    assert decoded[2] == [([4], 3)]              # main cache keeps its prefix
    assert llm._mtp_draft_stale is False
    assert llm._draft_pos == 4

    _, decoded = run(stale=False)
    assert decoded[3] == [([4], 3)]              # control: a synced draft cache gets the suffix


def _decode_that_fails_the_draft(recorder, fail_from, how):
    """A decode that serves the recorder, except the draft context's decodes from
    number *fail_from* on, which return 1 or raise. A draft step is one
    draft-context decode, carrying the previous step's accepted tokens."""
    seen = {"draft": 0}

    def decode(ctx, batch):
        if ctx.value == 3:
            seen["draft"] += 1
            if seen["draft"] >= fail_from:
                if how == "raise":
                    raise RuntimeError("draft decode blew up")
                return 1
        return recorder.decode(ctx, batch)

    return decode, seen


@pytest.mark.parametrize("how, status", [
    ("return", "draft-decode-failed:1"),
    ("raise", "draft-decode-error:RuntimeError"),
])
def test_a_draft_decode_failing_mid_reply_stops_drafting_and_reports_it(how, status):
    rec = _SpecRecorder(head=[500, 502, 504, _SpecRecorder.EOG],
                        draft=[501, 503], verify=[501])
    decode, seen = _decode_that_fails_the_draft(rec, fail_from=2, how=how)
    llm_holder = []

    tokens, _ = _run_generate(rec, max_new_tokens=8, decode=decode, llm_holder=llm_holder)

    llm = llm_holder[0]
    assert tokens == [500, 501, 502, 504], tokens          # data before the flags
    assert seen["draft"] == 2, "drafting continued after the draft decode failed"
    assert llm.mtp_active_this_call is False
    assert llm.mtp_call_status == status
    # The model keeps its capability: only this reply stopped speculating.
    assert (llm.supports_mtp, llm._mtp_usable, llm.mtp_status) == (True, True, "ok:qwen35")
    assert llm._mtp_draft_stale is True


def test_a_reply_that_never_fails_a_draft_reports_no_stop():
    rec = _SpecRecorder(head=[500, _SpecRecorder.EOG], draft=[501], verify=[501])
    llm_holder = []

    _run_generate(rec, max_new_tokens=8, llm_holder=llm_holder)

    assert llm_holder[0].mtp_call_status == ""
    assert llm_holder[0].mtp_active_this_call is True


def test_a_new_reply_drafts_again_after_one_that_stopped():
    rec = _SpecRecorder(head=[500, 502, _SpecRecorder.EOG], draft=[501], verify=[501])
    decode, _ = _decode_that_fails_the_draft(rec, fail_from=2, how="return")
    llm_holder = []
    _run_generate(rec, max_new_tokens=8, decode=decode, llm_holder=llm_holder)
    llm = llm_holder[0]
    assert llm.mtp_call_status != ""

    rec2 = _SpecRecorder(head=[600, _SpecRecorder.EOG], draft=[601], verify=[601])
    with patch("localm.inference.backends.llamacpp.llama.api") as mock_api, \
         patch("localm.inference.backends.llamacpp.llama._build_sampler",
               return_value=rec2.main_sampler):
        mock_api.llama_sampler_chain_init.return_value = rec2.draft_sampler
        mock_api.llama_sampler_sample.side_effect = rec2.sample
        mock_api.llama_sampler_accept.side_effect = rec2.accept
        mock_api.llama_decode.side_effect = rec2.decode
        tokens = list(llm._generate(prompt_tokens=[1, 2], max_new_tokens=8,
                                    temperature=0.8, top_k=40, top_p=0.95,
                                    repeat_penalty=1.1))

    assert tokens == [600, 601], tokens
    assert "DRAFT" in rec2.shapes()
    assert llm.mtp_active_this_call is True
    assert llm.mtp_call_status == ""


@pytest.mark.parametrize("status", [
    "draft-prefill-failed:-1", "draft-prefill-failed:2",
    "draft-prefill-error:OSError", "draft-trim-error:RuntimeError",
])
def test_a_suffixed_permanent_status_latches_supports_mtp_off(status):
    backend = GgufBackend("test_model.gguf")
    backend._loaded = True
    backend._supports_mtp = True

    backend._record_mtp({"mtp_status": status, "mtp_active": False})

    assert backend.supports_mtp is False
    assert backend.last_mtp_status == status


def test_every_status_the_child_disables_mtp_with_is_in_the_stop_set():
    """The reverse of the vocabulary pin above: a status _disable_mtp can set
    that the parent does not latch leaves supports_mtp True forever."""
    import ast
    from pathlib import Path

    from localm.inference.backends import gguf as gguf_mod

    tree = ast.parse(Path(inspect.getfile(LlamaCppModule)).read_text(encoding="utf-8"))
    kinds = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "_disable_mtp" and node.args):
            arg = node.args[0]
            if isinstance(arg, ast.BinOp):
                arg = arg.left
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                kinds.add(arg.value.split(":", 1)[0])
    assert kinds, "found no _disable_mtp call sites to check"
    assert kinds <= gguf_mod._MTP_STOPPED, sorted(kinds - gguf_mod._MTP_STOPPED)


def test_a_per_call_stop_is_recorded_without_latching_the_model_off():
    backend = GgufBackend("test_model.gguf")
    backend._loaded = True
    backend._supports_mtp = True

    backend._record_mtp({"mtp_status": "ok:qwen35", "mtp_active": False,
                         "mtp_call_status": "draft-decode-failed:1"})

    assert backend.last_mtp_call_status == "draft-decode-failed:1"
    assert backend.last_mtp_active is False
    assert backend.supports_mtp is True

    backend._record_mtp({"mtp_status": "ok:qwen35", "mtp_active": True})
    assert backend.last_mtp_call_status == ""


def test_the_done_envelope_carries_the_per_call_stop():
    import inspect as _inspect

    from localm.inference.backends.llamacpp import _runner, _worker

    assert '"mtp_call_status": worker.mtp_call_status' in _inspect.getsource(_runner)
    w = _worker.GgufWorker.__new__(_worker.GgufWorker)
    w._llm = SimpleNamespace(mtp_call_status="draft-decode-failed:1")
    assert w.mtp_call_status == "draft-decode-failed:1"
    w._llm = None
    assert w.mtp_call_status == ""


def test_the_draft_context_is_created_at_the_size_it_is_asked_for():
    llm = make_bare_llama(_model_ptr=ctypes.c_void_p(1), _ctx_ptr=ctypes.c_void_p(2))
    with patch("localm.inference.backends.llamacpp.llama.api") as mock_api:
        made = _context_factory(mock_api)
        assert llm._create_mtp_context(16384) == ""

    assert [(m.draft, m.n_ctx, m.n_batch) for m in made] == [(True, 16384, 2048)]
    assert llm._mtp_ctx_capacity == 16384


def test_a_small_context_gets_a_batch_no_larger_than_itself():
    llm = make_bare_llama(_model_ptr=ctypes.c_void_p(1), _ctx_ptr=ctypes.c_void_p(2))
    with patch("localm.inference.backends.llamacpp.llama.api") as mock_api:
        made = _context_factory(mock_api)
        llm._create_mtp_context(1024)

    assert (made[0].n_ctx, made[0].n_batch) == (1024, 1024)


def test_a_context_growth_charges_the_draft_contexts_kv_too():
    def decision(draft_per_token):
        b = _mtp_sizing_backend(n_ctx=4096)
        b.effective_gpu_layers = 99
        b._kv_bytes_per_token = lambda: 1000
        b._mtp_draft_kv_per_token = lambda: draft_per_token
        # Room for the main KV growth (4096 * 1000) but not for the draft's too.
        with patch.object(GgufBackend, "_free_vram_bytes", return_value=6_000_000):
            return b._check_context_fit(8192, current_ctx=4096)

    assert decision(0) is True
    assert decision(1000) is False


def test_the_draft_kv_per_token_is_the_probed_value_when_eligible_and_zero_otherwise():
    b = _mtp_sizing_backend(n_ctx=4096)
    with patch("localm.model_manager.gguf.gguf_nextn_predict_layers",
               return_value=("qwen35", 1)), \
         patch("localm.model_manager.gguf.gguf_mtp_draft_kv_bytes_per_token",
               return_value=1234):
        assert b._mtp_draft_kv_per_token() == 1234
    assert _mtp_sizing_backend(mtp_enabled=False)._mtp_draft_kv_per_token() == 0


def test_a_draft_context_is_never_created_for_a_model_without_an_mtp_graph():
    llm = make_bare_llama(_model_ptr=ctypes.c_void_p(1), _ctx_ptr=ctypes.c_void_p(2))
    with patch("localm.inference.backends.llamacpp.llama.api") as mock_api:
        made = _context_factory(mock_api)
        mock_api.llama_model_mtp_support.return_value = (False, "no-mtp-graph")
        assert llm._create_mtp_context(4096) == "no-mtp-graph"

    assert made == []


# --------------------------------------------------------------------------- #
#  Draft-count setting and per-reply MTP figures                              #
# --------------------------------------------------------------------------- #

def test_the_draft_count_setting_defaults_to_the_native_default():
    from localm.inference.backends.llamacpp.llama import (
        MTP_DRAFT_TOKENS_DEFAULT,
        MTP_DRAFT_TOKENS_MAX,
    )
    assert DEFAULT_CONFIG["mtp_draft_tokens"] == MTP_DRAFT_TOKENS_DEFAULT
    field = next(f for f in CORE_FIELDS if f.key == "mtp_draft_tokens")
    assert (field.group, field.min, field.max) == ("Engine", 1, MTP_DRAFT_TOKENS_MAX)


@pytest.mark.parametrize("cfg_value, override, expected", [
    (3, None, 3), (3, 1, 1), (0, None, 1), (99, None, 3), ("x", None, 1), (None, None, 1),
])
def test_the_draft_count_is_read_from_config_clamped_and_overridable(cfg_value, override, expected):
    from localm.inference.engine import _resolve_mtp_draft_tokens
    cfg = {} if cfg_value is None else {"mtp_draft_tokens": cfg_value}
    assert _resolve_mtp_draft_tokens(cfg, override) == expected


def test_the_draft_count_reaches_the_native_instance(tmp_path):
    from localm.inference.engine import create_backend
    model = tmp_path / "m.gguf"
    model.write_bytes(b"GGUF")
    with patch("localm.inference.engine.load_config",
               return_value={**DEFAULT_CONFIG, "mtp_draft_tokens": 3}):
        backend = create_backend(str(model))
    assert backend.mtp_draft_tokens == 3

    from localm.inference.backends.llamacpp import _worker
    w = _worker.GgufWorker.__new__(_worker.GgufWorker)
    w._llm = SimpleNamespace(mtp_drafted=7, mtp_accepted=5, mtp_skipped="image")
    assert (w.mtp_drafted, w.mtp_accepted, w.mtp_skipped) == (7, 5, "image")
    w._llm = None
    assert (w.mtp_drafted, w.mtp_accepted, w.mtp_skipped) == (0, 0, "")


def test_the_done_envelope_carries_the_draft_counts():
    import inspect as _inspect

    from localm.inference.backends.llamacpp import _runner

    src = _inspect.getsource(_runner)
    assert '"mtp_drafted": worker.mtp_drafted' in src
    assert '"mtp_accepted": worker.mtp_accepted' in src
    assert '"mtp_steps": worker.mtp_steps' in src
    assert '"mtp_paused_steps": worker.mtp_paused_steps' in src
    assert '"mtp_skipped": worker.mtp_skipped' in src


@pytest.mark.parametrize("done, supports, expected", [
    ({"mtp_status": "ok:qwen35", "mtp_active": True, "mtp_drafted": 40, "mtp_accepted": 31},
     True, {"state": "on", "drafted": 40, "accepted": 31, "paused_steps": 0, "reason": None}),
    ({"mtp_status": "ok:qwen35", "mtp_active": False, "mtp_call_status": "draft-decode-failed:1",
      "mtp_drafted": 4, "mtp_accepted": 3},
     True, {"state": "stopped", "drafted": 4, "accepted": 3, "paused_steps": 0,
            "reason": "draft-decode-failed:1"}),
    ({"mtp_status": "ok:qwen35", "mtp_active": False}, True,
     {"state": "idle", "drafted": 0, "accepted": 0, "paused_steps": 0, "reason": None}),
    ({"mtp_status": "ok:qwen35", "mtp_active": True, "mtp_drafted": 6, "mtp_accepted": 3,
      "mtp_steps": 6, "mtp_paused_steps": 40}, True,
     {"state": "paused", "drafted": 6, "accepted": 3, "paused_steps": 40,
      "reason": "slower-than-plain"}),
    ({"mtp_status": "ok:qwen35", "mtp_active": True, "mtp_drafted": 60, "mtp_accepted": 50,
      "mtp_steps": 60, "mtp_paused_steps": 10}, True,
     {"state": "on", "drafted": 60, "accepted": 50, "paused_steps": 10, "reason": None}),
    ({"mtp_status": "no-mtp-graph:llama", "mtp_active": False}, False,
     {"state": "unavailable", "drafted": 0, "accepted": 0, "paused_steps": 0,
      "reason": "no-mtp-graph:llama"}),
    ({"mtp_status": "ok:qwen35", "mtp_active": False, "mtp_skipped": "image"}, True,
     {"state": "off", "drafted": 0, "accepted": 0, "paused_steps": 0, "reason": "image"}),
])
def test_the_backend_summarises_the_last_reply_for_the_api(done, supports, expected):
    backend = GgufBackend("test_model.gguf", mtp_enabled=True)
    backend._loaded = True
    backend._supports_mtp = supports

    backend._record_mtp(done)

    assert backend.last_mtp_usage == expected


def test_no_mtp_figures_when_mtp_is_off():
    backend = GgufBackend("test_model.gguf", mtp_enabled=False)
    backend._loaded = True
    backend._record_mtp({"mtp_status": "disabled", "mtp_active": False})
    assert backend.last_mtp_usage is None


def test_a_malformed_count_in_the_envelope_reads_as_zero():
    backend = GgufBackend("test_model.gguf", mtp_enabled=True)
    backend._loaded = True
    backend._supports_mtp = True
    backend._record_mtp({"mtp_status": "ok", "mtp_active": True,
                         "mtp_drafted": "lots", "mtp_accepted": -3})
    assert (backend.last_mtp_drafted, backend.last_mtp_accepted) == (0, 0)


def test_the_chat_usage_carries_the_mtp_figures():
    from localm.inference import http_server
    from localm.inference.protocol import UsageInfo

    engine = SimpleNamespace(mtp_usage=lambda: {
        "state": "on", "drafted": 10, "accepted": 8, "paused_steps": 2, "reason": None})
    usage = UsageInfo(total_tokens=5, mtp=http_server._mtp_usage(engine))
    assert usage.model_dump()["mtp"] == {
        "state": "on", "drafted": 10, "accepted": 8, "paused_steps": 2, "reason": None}

    # Engines without figures (mocks, HF, MTP off) leave the field out.
    assert http_server._mtp_usage(MagicMock()) is None
    assert http_server._mtp_usage(SimpleNamespace(mtp_usage=lambda: None)) is None
    assert http_server._mtp_usage(SimpleNamespace()) is None
    assert http_server._mtp_usage(SimpleNamespace(mtp_usage=lambda: {"drafted": 1})) is None


def test_engine_mtp_usage_passes_the_backend_summary_through():
    eng = Engine.__new__(Engine)
    eng._backend = SimpleNamespace(last_mtp_usage={"state": "idle", "drafted": 0,
                                                   "accepted": 0, "reason": None})
    assert eng.mtp_usage()["state"] == "idle"
    eng._backend = SimpleNamespace()
    assert eng.mtp_usage() is None


def test_bench_mtp_reports_acceptance_and_identical_output(cli_runner):
    from localm.cli import models as models_mod

    seen = []
    with patch.object(models_mod, "get_operator_model_info",
                      return_value=("model.gguf", None)), \
         patch.object(models_mod, "_mtp_probe_arm",
                      _bench_mtp_result([50.0], [70.0], counts=(40, 30), seen=seen)):
        res = cli_runner.invoke(models_mod.bench_mtp,
                                ["model.gguf", "--rounds", "1", "--draft-tokens", "3"])

    assert res.exit_code == 0, res.output
    assert "Drafts accepted: 30 of 40 (75%)" in res.output
    assert "Output identical to MTP off: 3 of 3 replies" in res.output
    assert seen == [3]


def test_bench_mtp_flags_output_that_differs_from_mtp_off(cli_runner):
    from localm.cli import models as models_mod

    with patch.object(models_mod, "get_operator_model_info",
                      return_value=("model.gguf", None)), \
         patch.object(models_mod, "_mtp_probe_arm",
                      _bench_mtp_result([50.0], [70.0], texts_on=("a", "X", "c"))):
        res = cli_runner.invoke(models_mod.bench_mtp, ["model.gguf", "--rounds", "1"])

    assert res.exit_code == 0, res.output
    assert "Output differs from MTP off in 1 of 3 replies" in res.output



@pytest.mark.parametrize("status", ["rewind-unsupported", "context-refused"])
def test_a_reply_that_turned_mtp_off_for_the_model_reports_it_stopped(status):
    """A reply that speculated and then lost MTP for the model (a stuck rollback,
    a draft context that could not be recreated) reports that it stopped and
    why; the next reply reports MTP unavailable."""
    backend = GgufBackend("test_model.gguf", mtp_enabled=True)
    backend._loaded = True
    backend._supports_mtp = True

    backend._record_mtp({"mtp_status": status, "mtp_active": True,
                         "mtp_drafted": 5, "mtp_accepted": 4, "mtp_steps": 5})

    usage = backend.last_mtp_usage
    assert (usage["state"], usage["reason"], usage["drafted"]) == ("stopped", status, 5)

    backend._reset_mtp_call()
    backend._record_mtp({"mtp_status": status, "mtp_active": False})
    assert backend.last_mtp_usage["state"] == "unavailable"


def test_a_reply_that_ends_without_a_report_shows_no_figures_from_the_last_one():
    """A cancelled reply never sends its done envelope; it must not show the
    previous reply's MTP figures."""
    backend = GgufBackend("test_model.gguf", mtp_enabled=True)
    backend._loaded = True
    backend._supports_mtp = True
    backend._record_mtp({"mtp_status": "ok", "mtp_active": True,
                         "mtp_drafted": 40, "mtp_accepted": 30, "mtp_steps": 40})

    class _Runner:
        last_done = None

        def chat_stream(self, **kwargs):
            yield "a"
            yield "b"

    backend._runner = _Runner()
    gen = backend.chat_stream([{"role": "user", "content": "hi"}])
    assert next(gen) == "a"
    gen.close()

    usage = backend.last_mtp_usage
    assert (usage["state"], usage["drafted"], usage["accepted"]) == ("idle", 0, 0)
