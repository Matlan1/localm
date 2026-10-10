# SPDX-License-Identifier: AGPL-3.0-or-later
"""``min_p``, ``presence_penalty`` and ``frequency_penalty`` from the engine
down to the sampler: the GGUF sampler chain applies them, the HF worker applies
them through transformers and a logits processor, a backend that cannot apply
them refuses them, and nothing is sent when the caller set nothing."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from localm.inference.backends.base import UnsupportedInputError
from localm.inference.backends.llamacpp import llama as L
from tests._bare_llama import make_bare_llama

_CFG = {"max_tokens": 512, "temperature": 0.7, "top_p": 0.9, "top_k": 40,
        "repeat_penalty": 1.1}


def _api(has_penalties=True, needs_vocab=False, n_vocab=32000):
    api = MagicMock()
    api.llama_sampler_chain_init.return_value = 500
    api.has_penalties_sampler.return_value = has_penalties
    api.penalties_needs_n_vocab.return_value = needs_vocab
    api.llama_vocab_n_tokens.return_value = n_vocab
    return api


# ------------------------------------------------------------------ native chain


def test_openai_penalties_add_the_penalties_stage_with_no_repeat_penalty():
    api = _api()
    with patch.object(L, "api", api):
        L._build_sampler(vocab=1, temperature=0.7, repeat_penalty=1.0,
                         penalty_freq=0.4, penalty_present=0.6)
    api.llama_sampler_init_penalties.assert_called_once_with(64, 1.0, 0.4, 0.6, n_vocab=32000)


def test_all_three_penalties_share_one_stage():
    api = _api()
    with patch.object(L, "api", api):
        L._build_sampler(vocab=1, temperature=0.7, repeat_penalty=1.2,
                         penalty_freq=0.1, penalty_present=0.2)
    api.llama_sampler_init_penalties.assert_called_once_with(64, 1.2, 0.1, 0.2, n_vocab=32000)


def test_no_penalty_adds_no_penalties_stage():
    api = _api()
    with patch.object(L, "api", api):
        L._build_sampler(vocab=1, temperature=0.7, repeat_penalty=1.0)
    api.llama_sampler_init_penalties.assert_not_called()


def test_min_p_reaches_the_min_p_stage():
    api = _api()
    with patch.object(L, "api", api):
        L._build_sampler(vocab=1, temperature=0.7, min_p=0.2)
    api.llama_sampler_init_min_p.assert_called_once_with(0.2, 1)


@pytest.mark.parametrize("has_penalties, needs_vocab, n_vocab",
                         [(False, False, 32000), (True, True, 0)])
def test_openai_penalties_are_refused_when_the_runtime_cannot_apply_them(
        has_penalties, needs_vocab, n_vocab):
    api = _api(has_penalties, needs_vocab, n_vocab)
    with patch.object(L, "api", api):
        with pytest.raises(UnsupportedInputError, match="presence_penalty"):
            L._build_sampler(vocab=1, temperature=0.7, penalty_present=0.5)
    api.llama_sampler_free.assert_called_once_with(500)


def test_a_repeat_penalty_alone_is_still_skipped_quietly_without_the_stage():
    api = _api(has_penalties=False)
    with patch.object(L, "api", api):
        L._build_sampler(vocab=1, temperature=0.7, repeat_penalty=1.3)
    api.llama_sampler_free.assert_not_called()
    api.llama_sampler_init_penalties.assert_not_called()


# ------------------------------------------------------------------ create_chat_completion


def _llm(monkeypatch, **overrides):
    llm = make_bare_llama(_model_ptr=1, _ctx_ptr=1, **overrides)
    monkeypatch.setattr(L, "_apply_model_template", lambda model, messages: ("hi", None))
    monkeypatch.setattr(L, "_untrusted_prompt_ranges", lambda *a: ())
    llm._tokenizer.encode = lambda text, add_bos=True, untrusted_ranges=(): [1, 2]
    return llm


def _capture(llm, monkeypatch, name):
    seen: dict = {}

    def fake(*args, **kw):
        seen.update(kw)
        yield from ()
    monkeypatch.setattr(llm, name, fake)
    return seen


def test_the_text_path_gets_the_sampling_options(monkeypatch):
    llm = _llm(monkeypatch)
    seen = _capture(llm, monkeypatch, "_generate")
    llm.create_chat_completion([{"role": "user", "content": "hi"}], min_p=0.1,
                               presence_penalty=0.2, frequency_penalty=0.3)
    assert seen["sampling"] == {"min_p": 0.1, "penalty_present": 0.2, "penalty_freq": 0.3}


def test_no_option_means_an_empty_sampling_dict(monkeypatch):
    llm = _llm(monkeypatch)
    seen = _capture(llm, monkeypatch, "_generate")
    llm.create_chat_completion([{"role": "user", "content": "hi"}])
    assert seen["sampling"] == {}


def test_the_image_path_gets_the_grammar_and_the_options(monkeypatch):
    llm = _llm(monkeypatch)
    llm._mtmd = object()
    seen = _capture(llm, monkeypatch, "_generate_image")
    image = {"role": "user", "content": [
        {"type": "text", "text": "what"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}
    llm.create_chat_completion([image], grammar='root ::= "x"', grammar_lazy=True,
                               grammar_triggers=["t"], min_p=0.3)
    assert seen["grammar"] == 'root ::= "x"'
    assert seen["grammar_lazy"] is True and seen["grammar_triggers"] == ["t"]
    assert seen["sampling"] == {"min_p": 0.3}


def test_the_encoder_decoder_path_gets_the_options(monkeypatch):
    llm = _llm(monkeypatch, is_encoder_decoder=True)
    seen = _capture(llm, monkeypatch, "_generate_encoder_decoder")
    llm.create_chat_completion([{"role": "user", "content": "hi"}], frequency_penalty=1.0)
    assert seen["sampling"] == {"penalty_freq": 1.0}


def test_a_diffusion_model_refuses_the_options(monkeypatch):
    llm = _llm(monkeypatch, is_diffusion=True)
    seen = _capture(llm, monkeypatch, "_generate_diffusion")
    with pytest.raises(UnsupportedInputError, match="diffusion"):
        llm.create_chat_completion([{"role": "user", "content": "hi"}], min_p=0.1)
    assert seen == {}
    llm.create_chat_completion([{"role": "user", "content": "hi"}], min_p=0.0,
                               presence_penalty=0.0)
    assert "sampling" not in seen


def test_the_image_sampler_is_built_with_the_grammar(monkeypatch):
    class _Built(Exception):
        pass
    built: dict = {}

    def fake_build(**kw):
        built.update(kw)
        raise _Built()
    monkeypatch.setattr(L, "_build_sampler", fake_build)
    monkeypatch.setattr(L.pretokenizer_guard, "check_text", lambda *a: None)
    llm = _llm(monkeypatch)
    llm._mtmd = MagicMock(marker="<image>", encode_count=0)
    llm._mtmd.tokenize.return_value = MagicMock(n_tokens=8)
    monkeypatch.setattr(llm, "_fit_generation_budget", lambda n, m: m)
    monkeypatch.setattr(llm, "_prefill_vision", lambda *a: (8, 0))
    with pytest.raises(_Built):
        list(llm._generate_image([{"role": "user", "content": "x"}], max_new_tokens=4,
                                 temperature=0.7, top_k=40, top_p=0.9, repeat_penalty=1.0,
                                 grammar='root ::= "x"', grammar_lazy=True,
                                 grammar_triggers=["t"], sampling={"min_p": 0.2}))
    assert built["grammar"] == 'root ::= "x"' and built["min_p"] == 0.2
    assert built["grammar_lazy"] is True and built["grammar_triggers"] == ["t"]


# ------------------------------------------------------------------ backends and engine


def test_gguf_backend_refuses_options_only_for_a_diffusion_model():
    from localm.inference.backends.gguf import GgufBackend
    backend = GgufBackend.__new__(GgufBackend)
    with patch.object(GgufBackend, "is_diffusion", property(lambda self: False)):
        assert backend.unsupported_sampling({"min_p": 0.1, "presence_penalty": 1.0}) == []
    with patch.object(GgufBackend, "is_diffusion", property(lambda self: True)):
        assert backend.unsupported_sampling({"min_p": 0.1}) == ["min_p"]
        assert backend.unsupported_sampling({"min_p": 0.0, "presence_penalty": 0,
                                             "frequency_penalty": 0.5}) == ["frequency_penalty"]


def test_a_backend_that_declares_nothing_refuses_everything():
    from localm.inference.backends.base import BaseBackend
    assert BaseBackend.unsupported_sampling(
        MagicMock(), {"min_p": 0.1, "frequency_penalty": 0.2}) == ["min_p", "frequency_penalty"]


def test_hf_backend_applies_every_option():
    from localm.inference.backends.hf import HFBackend
    assert HFBackend.unsupported_sampling(MagicMock(), {"min_p": 0.1, "presence_penalty": 1}) == []


def test_the_gguf_worker_passes_set_options_to_the_model():
    from localm.inference.backends.llamacpp._worker import GgufWorker
    w = GgufWorker("m.gguf", None, 512, 0, None, 512)
    captured: dict = {}

    class _Llm:
        def create_chat_completion(self, **kw):
            captured.update(kw)
            return iter(())
    w._llm = _Llm()
    list(w.chat_stream([{"role": "user", "content": "hi"}], presence_penalty=0.7))
    assert captured["presence_penalty"] == 0.7
    assert "min_p" not in captured and "frequency_penalty" not in captured


def _engine(backend):
    from localm.inference.engine import Engine
    engine = object.__new__(Engine)
    engine.model_path = "m.gguf"
    engine.display_name = "m"
    engine._backend = backend
    return engine


def _backend():
    backend = MagicMock()
    backend.loaded = True
    backend.chat_stream.side_effect = lambda messages, **kw: iter(["ok"])
    return backend


@patch("localm.inference.engine.load_config", return_value=_CFG)
def test_engine_passes_only_the_options_that_were_set(_cfg):
    backend = _backend()
    engine = _engine(backend)
    list(engine.chat_stream([{"role": "user", "content": "hi"}]))
    assert not {"min_p", "presence_penalty", "frequency_penalty"} & set(
        backend.chat_stream.call_args.kwargs)
    list(engine.chat_stream([{"role": "user", "content": "hi"}], min_p=0.0,
                            frequency_penalty=0.5))
    kw = backend.chat_stream.call_args.kwargs
    assert kw["min_p"] == 0.0 and kw["frequency_penalty"] == 0.5
    assert "presence_penalty" not in kw


def test_engine_reports_the_backends_refusals():
    backend = _backend()
    backend.unsupported_sampling.return_value = ["min_p"]
    assert _engine(backend).unsupported_sampling({"min_p": 0.2}) == ["min_p"]
    backend.unsupported_sampling.assert_called_once_with({"min_p": 0.2})


# ------------------------------------------------------------------ HF penalty processor


def test_the_hf_penalty_processor_subtracts_frequency_and_presence():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from localm.inference.backends._hf_worker import _penalty_processor
    proc = _penalty_processor(2, presence=0.5, frequency=1.0)
    input_ids = torch.tensor([[7, 7, 3, 3, 1]])   # prompt 7 7, generated 3 3 1
    scores = torch.zeros(1, 8)
    out = proc(input_ids, scores)
    expected = torch.zeros(1, 8)
    expected[0, 3] = -(2 * 1.0 + 0.5)
    expected[0, 1] = -(1 * 1.0 + 0.5)
    assert torch.allclose(out, expected)


def test_the_hf_penalty_processor_ignores_the_prompt():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from localm.inference.backends._hf_worker import _penalty_processor
    proc = _penalty_processor(3, presence=1.0, frequency=1.0)
    scores = torch.zeros(1, 4)
    assert torch.equal(proc(torch.tensor([[1, 2, 3]]), scores), scores)


def test_no_hf_penalty_means_no_processor():
    from localm.inference.backends._hf_worker import _penalty_processor
    assert _penalty_processor(5, None, None) is None
    assert _penalty_processor(5, 0.0, 0.0) is None


def test_the_hf_penalty_window_starts_after_the_decoder_start_token():
    from types import SimpleNamespace

    from localm.inference.backends._hf_worker import _penalty_offset
    seq2seq = SimpleNamespace(config=SimpleNamespace(is_encoder_decoder=True))
    causal = SimpleNamespace(config=SimpleNamespace(is_encoder_decoder=False))
    assert _penalty_offset(seq2seq, 57) == 1
    assert _penalty_offset(causal, 57) == 57
    assert _penalty_offset(SimpleNamespace(), 9) == 9


def test_a_gguf_model_whose_grammar_faulted_refuses_grammar():
    from localm.inference.backends.base import GRAMMAR_FAULTED_MESSAGE, GrammarUnsupportedError
    from localm.inference.backends.gguf import GgufBackend
    backend = GgufBackend.__new__(GgufBackend)
    backend._runner = None
    backend._loaded = False
    with patch.object(GgufBackend, "is_diffusion", property(lambda self: False)):
        backend._grammar_unsupported = False
        assert backend.supports_grammar is True
        backend.validate_grammar('root ::= "x"')
        backend._grammar_unsupported = True
        assert backend.supports_grammar is False
        with pytest.raises(GrammarUnsupportedError) as exc:
            backend.validate_grammar('root ::= "x"')
        assert str(exc.value) == GRAMMAR_FAULTED_MESSAGE
        backend.validate_grammar(None)
