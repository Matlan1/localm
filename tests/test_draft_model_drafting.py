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


def _vocab_gguf(path, tokens, *, model="gpt2", add_bos=None, bos=None):
    kv = [("general.architecture", _kv_string("qwen2")),
          ("tokenizer.ggml.model", _kv_string(model)),
          ("tokenizer.ggml.tokens", _kv_string_array(tokens))]
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


def test_the_draft_model_loads_on_the_main_gpu_without_splitting(tmp_path):
    llm, path, llama_mod = _loading_llama(tmp_path)
    llm._main_gpu_index = 1
    params = SimpleNamespace(n_gpu_layers=0, split_mode=1, main_gpu=0)
    with patch.object(llama_mod, "api") as api, \
         patch.object(llama_mod, "set_use_mmap") as mmap:
        api.llama_model_default_params.return_value = params
        api.llama_load_model_from_file.return_value = None
        llm._load_draft_model(path, None, True)
    assert (params.n_gpu_layers, params.split_mode, params.main_gpu) == (99, 0, 1)
    mmap.assert_called_once_with(params, False)


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
