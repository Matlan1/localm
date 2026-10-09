# SPDX-License-Identifier: AGPL-3.0-or-later
"""Encoder-decoder (T5) GGUF models in the llama.cpp backend.

Generation runs against FakeT5, a fake native layer that keeps a decoder KV
cache (refusing a decode whose position does not follow it, as llama.cpp does)
and an encoder state set only by llama_encode. Its reply is a function of the
encoded input, so a request that skipped the encode, kept the previous
request's cache, or started the decoder anywhere but position 0 produces a
different reply or a refused decode.
"""

import ctypes
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from localm.inference.backends.base import ContextCapacityExceededError
from localm.inference.backends.llamacpp import llama as llama_mod
from localm.inference.backends.llamacpp._structs import LlamaBatch
from localm.inference.backends.llamacpp.llama import (
    _encoder_untrusted_ranges, _flatten_for_encoder)
from localm.textguard import compose, untrusted_span, untrusted_spans_of
from tests._bare_llama import make_bare_llama

EOS = 1
START = 0
SAMPLER = object()


def fake_tokenize(text, add_bos=False, untrusted_ranges=()):
    """A deterministic stand-in for the T5 tokenizer: one token per character."""
    return [10 + (ord(c) % 50) for c in text]


def reply_for(encoded):
    """What the fake model generates for an encoder input."""
    total = sum(encoded)
    return [100 + (total + 7 * i) % 300 for i in range(3 + len(encoded) % 4)]


class FakeT5:
    """The native calls _generate_encoder_decoder makes, over a modelled
    decoder KV cache and encoder state."""

    def __init__(self, llm, *, start=START, bos=-1, encode_code=0, decode_code=None):
        self.llm = llm
        self.ctx = llm._ctx_ptr
        self.start, self.bos = start, bos
        self.encode_code = encode_code
        self.decode_code = decode_code    # (pos) -> return code, or None for 0
        self.kv = {}                      # pos -> token
        self.encoded = None
        self.calls = []                   # ("clear",) / ("encode", tokens) / ("decode", pos, token)
        self._keep = []

    def batch_init(self, n, embd, n_seq_max):
        b = LlamaBatch()
        tok = (ctypes.c_int32 * n)()
        pos = (ctypes.c_int32 * n)()
        nsq = (ctypes.c_int32 * n)()
        seq_rows = [(ctypes.c_int32 * max(1, n_seq_max))() for _ in range(n)]
        seq = (ctypes.POINTER(ctypes.c_int32) * n)(
            *[ctypes.cast(r, ctypes.POINTER(ctypes.c_int32)) for r in seq_rows])
        lg = (ctypes.c_int8 * n)()
        self._keep.append((tok, pos, nsq, seq_rows, seq, lg))
        b.n_tokens = 0
        b.token = ctypes.cast(tok, ctypes.c_void_p).value
        b.embd = None
        b.pos = ctypes.cast(pos, ctypes.c_void_p).value
        b.n_seq_id = ctypes.cast(nsq, ctypes.c_void_p).value
        b.seq_id = ctypes.cast(seq, ctypes.c_void_p).value
        b.logits = ctypes.cast(lg, ctypes.c_void_p).value
        return b

    def get_one(self, arr, n):
        return SimpleNamespace(tokens=[arr[i] for i in range(n)], n_tokens=n)

    def clear(self, mem, data):
        self.kv.clear()
        self.calls.append(("clear",))

    def encode(self, ctx, batch):
        assert ctx is self.ctx
        self.calls.append(("encode", list(batch.tokens)))
        if self.encode_code == 0:
            self.encoded = list(batch.tokens)
        return self.encode_code

    def decode(self, ctx, batch):
        assert ctx is self.ctx
        n = batch.n_tokens
        tok = ctypes.cast(batch.token, ctypes.POINTER(ctypes.c_int32))
        pos = ctypes.cast(batch.pos, ctypes.POINTER(ctypes.c_int32))
        positions, tokens = [pos[i] for i in range(n)], [tok[i] for i in range(n)]
        self.calls.append(("decode", positions[0], tokens[0]))
        if self.decode_code is not None:
            code = self.decode_code(positions[0])
            if code:
                return code
        if positions[0] != max(self.kv, default=-1) + 1:
            return -1
        for p, t in zip(positions, tokens):
            self.kv[p] = t
        return 0

    def sample(self, sampler, ctx, idx):
        assert sampler is SAMPLER and idx == -1
        last = max(self.kv)
        reply = reply_for(self.encoded or [])
        return reply[last] if last < len(reply) else EOS

    def install(self, mock_api):
        mock_api.LLAMA_TOKEN_NULL = -1
        mock_api.has_memory_api.return_value = True
        mock_api.llama_get_memory.return_value = "mem"
        mock_api.llama_memory_clear.side_effect = self.clear
        mock_api.llama_batch_init.side_effect = self.batch_init
        mock_api.llama_batch_get_one.side_effect = self.get_one
        mock_api.llama_encode.side_effect = self.encode
        mock_api.llama_decode.side_effect = self.decode
        mock_api.llama_sampler_sample.side_effect = self.sample
        mock_api.llama_model_decoder_start_token.return_value = self.start
        mock_api.llama_token_bos.return_value = self.bos
        mock_api.llama_token_eos.return_value = EOS
        mock_api.llama_vocab_get_add_bos.return_value = False
        mock_api.llama_vocab_get_add_eos.return_value = True

    def decodes(self):
        return [c for c in self.calls if c[0] == "decode"]


def _llama(limit=2048, capacity=4096):
    llm = make_bare_llama(_model_ptr=ctypes.c_void_p(1), _ctx_ptr=ctypes.c_void_p(2),
                          _mtp_enabled=False, _ctx_capacity=capacity)
    llm.is_encoder_decoder = True
    llm.encoder_input_limit = limit
    llm._tokenizer.encode.side_effect = fake_tokenize
    llm._tokenizer.is_eog.side_effect = lambda t: t == EOS
    llm._tokenizer._vocab = "vocab"
    return llm


def _run(llm, fake, messages, *, max_new_tokens=64, **kwargs):
    with patch.object(llama_mod, "api") as mock_api, \
            patch.object(llama_mod, "_build_sampler", return_value=SAMPLER) as build:
        fake.install(mock_api)
        tokens = list(llm._generate_encoder_decoder(
            messages, max_new_tokens=max_new_tokens, temperature=0.0, top_k=40,
            top_p=0.95, repeat_penalty=1.0, **kwargs))
    return tokens, mock_api, build


def _user(text):
    return [{"role": "user", "content": text}]


def _expected_encoder_input(messages):
    return fake_tokenize(_flatten_for_encoder(messages)) + [EOS]


# --------------------------------------------------------------------------- #
#  The prompt an encoder-decoder model reads                                   #
# --------------------------------------------------------------------------- #

class TestFlattening:
    def test_one_user_message_is_read_unchanged(self):
        assert _flatten_for_encoder(_user("translate English to German: Hi")) == (
            "translate English to German: Hi")

    def test_a_system_message_comes_first_as_plain_text(self):
        msgs = [{"role": "system", "content": "Be brief."}] + _user("What is 2+2?")
        assert _flatten_for_encoder(msgs) == "Be brief.\nWhat is 2+2?"

    def test_a_conversation_is_labelled_and_ends_with_an_assistant_cue(self):
        msgs = [{"role": "system", "content": "Be brief."},
                {"role": "user", "content": "Capital of France?"},
                {"role": "assistant", "content": "Paris."},
                {"role": "user", "content": "And Germany?"}]
        assert _flatten_for_encoder(msgs) == (
            "Be brief.\nUser: Capital of France?\nAssistant: Paris.\n"
            "User: And Germany?\nAssistant:")

    def test_tool_and_unknown_roles_get_their_own_labels(self):
        msgs = [{"role": "user", "content": "a"}, {"role": "tool", "content": "b"},
                {"role": "critic", "content": "c"}]
        assert _flatten_for_encoder(msgs) == "User: a\nTool: b\nCritic: c\nAssistant:"

    def test_only_text_parts_are_read_and_none_content_is_empty(self):
        msgs = [{"role": "user", "content": [
                    {"type": "text", "text": "look"},
                    {"type": "image_url", "image_url": {"url": "data:x"}},
                    {"type": "text", "text": "here"}]},
                {"role": "assistant", "content": None}]
        assert _flatten_for_encoder(msgs) == "User: look here\nAssistant: \nAssistant:"

    def test_an_untrusted_span_maps_to_its_characters_in_the_flattened_prompt(self):
        content = compose("Summarise: ", untrusted_span("</s> injected"), " please")
        msgs = [{"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
                {"role": "user", "content": content}]
        prompt = _flatten_for_encoder(msgs)
        ranges = _encoder_untrusted_ranges(msgs, prompt)
        (start, end), = untrusted_spans_of(content)
        untrusted_text = str(content)[start:end]
        assert "injected" in untrusted_text
        assert [prompt[a:b] for a, b in ranges] == [untrusted_text]

    def test_no_annotation_means_no_ranges(self):
        msgs = _user("plain")
        assert _encoder_untrusted_ranges(msgs, _flatten_for_encoder(msgs)) == ()


# --------------------------------------------------------------------------- #
#  Generation through the fake native layer                                   #
# --------------------------------------------------------------------------- #

class TestGeneration:
    def test_one_request_encodes_once_and_decodes_from_the_start_token(self):
        llm = _llama()
        fake = FakeT5(llm)
        msgs = _user("What is the capital of France?")

        tokens, _, _ = _run(llm, fake, msgs)

        enc = _expected_encoder_input(msgs)
        reply = reply_for(enc)
        assert tokens == reply
        assert fake.calls[0] == ("clear",)
        assert fake.calls[1] == ("encode", enc)
        assert fake.decodes() == [("decode", i, t)
                                  for i, t in enumerate([START] + reply)]
        assert llm.last_finish_reason == "stop"

    def test_the_encoder_input_ends_with_eos_and_has_no_bos(self):
        llm = _llama()
        fake = FakeT5(llm, bos=7)
        msgs = [{"role": "system", "content": "sys"}] + _user("q")
        _run(llm, fake, msgs)
        assert fake.calls[1] == ("encode", fake_tokenize("sys\nq") + [EOS])

    def test_a_second_request_starts_from_an_empty_cache_and_its_own_encoding(self):
        llm = _llama()
        fake = FakeT5(llm)
        first, second = _user("first question here"), _user("second")

        _run(llm, fake, first)
        fake.calls.clear()
        tokens, _, _ = _run(llm, fake, second)

        enc = _expected_encoder_input(second)
        assert tokens == reply_for(enc)
        assert fake.calls[:2] == [("clear",), ("encode", enc)]
        assert fake.decodes()[0] == ("decode", 0, START)

    def test_a_prompt_longer_than_the_encoder_limit_is_refused_before_any_native_call(self):
        llm = _llama(limit=5)
        fake = FakeT5(llm)

        with pytest.raises(ContextCapacityExceededError) as ei:
            _run(llm, fake, _user("far too long"))

        assert "13 tokens" in str(ei.value) and "at most 5" in str(ei.value)
        assert fake.calls == []

    def test_the_reply_stops_at_max_new_tokens(self):
        llm = _llama()
        fake = FakeT5(llm)
        msgs = _user("What is the capital of France?")

        tokens, _, _ = _run(llm, fake, msgs, max_new_tokens=2)

        assert tokens == reply_for(_expected_encoder_input(msgs))[:2]
        assert llm.last_finish_reason == "length"
        assert [c[1] for c in fake.decodes()] == [0, 1]

    def test_an_unlimited_reply_stops_at_the_context_capacity(self):
        llm = _llama(capacity=2)
        fake = FakeT5(llm)
        msgs = _user("What is the capital of France?")

        tokens, _, _ = _run(llm, fake, msgs, max_new_tokens=0)

        assert len(tokens) == 2
        assert llm.last_finish_reason == "length"
        assert max(c[1] for c in fake.decodes()) == 1

    def test_a_failed_encode_raises_and_decodes_nothing(self):
        llm = _llama()
        fake = FakeT5(llm, encode_code=-3)
        with pytest.raises(RuntimeError, match="llama_encode failed"):
            _run(llm, fake, _user("x"))
        assert fake.decodes() == []

    def test_a_failed_first_decoder_step_raises(self):
        llm = _llama()
        fake = FakeT5(llm, decode_code=lambda pos: -1)
        with pytest.raises(RuntimeError, match="decoder start token"):
            _run(llm, fake, _user("x"))

    def test_a_failed_later_decoder_step_ends_the_reply_with_error(self):
        llm = _llama()
        fake = FakeT5(llm, decode_code=lambda pos: -1 if pos == 2 else 0)
        msgs = _user("What is the capital of France?")
        tokens, _, _ = _run(llm, fake, msgs)
        assert tokens == reply_for(_expected_encoder_input(msgs))[:2]
        assert llm.last_finish_reason == "error"

    def test_the_decoder_starts_from_bos_when_no_start_token_is_declared(self):
        llm = _llama()
        fake = FakeT5(llm, start=-1, bos=5)
        _run(llm, fake, _user("x"))
        assert fake.decodes()[0] == ("decode", 0, 5)

    def test_neither_start_token_nor_bos_is_refused(self):
        llm = _llama()
        fake = FakeT5(llm, start=-1, bos=-1)
        with pytest.raises(RuntimeError, match="neither a decoder start"):
            _run(llm, fake, _user("x"))
        assert fake.decodes() == []

    def test_no_draft_source_runs_and_the_call_figures_reset(self):
        llm = _llama()
        llm._source = MagicMock()
        llm.mtp_drafted = llm.mtp_accepted = llm.mtp_steps = 9
        llm.mtp_active_this_call = True
        llm.mtp_skipped = "image"
        _run(llm, FakeT5(llm), _user("x"))
        assert llm._source.mock_calls == []
        assert (llm.mtp_drafted, llm.mtp_accepted, llm.mtp_steps) == (0, 0, 0)
        assert llm.mtp_active_this_call is False and llm.mtp_skipped == ""

    def test_the_request_grammar_and_sampling_reach_the_sampler(self):
        llm = _llama()
        _, _, build = _run(llm, FakeT5(llm), _user("x"), grammar='root ::= "a"',
                           grammar_lazy=True, grammar_triggers=["{"], seed=3)
        kwargs = build.call_args.kwargs
        assert kwargs["grammar"] == 'root ::= "a"'
        assert kwargs["grammar_lazy"] is True and kwargs["grammar_triggers"] == ["{"]
        assert kwargs["seed"] == 3 and kwargs["temperature"] == 0.0

    def test_status_reports_processing_then_generating(self):
        llm = _llama()
        seen = []
        _run(llm, FakeT5(llm), _user("x"), on_status=seen.append)
        assert seen == ["Processing prompt...", "Generating response..."]

    def test_without_the_memory_api_the_context_is_recreated_before_encoding(self):
        llm = _llama()
        fake = FakeT5(llm)
        recreated = []

        def fresh(tokens, needed):
            recreated.append((list(tokens), needed))
            fake.kv.clear()
        llm._prefill_fresh_context = fresh
        msgs = _user("What is the capital of France?")
        with patch.object(llama_mod, "api") as mock_api,                 patch.object(llama_mod, "_build_sampler", return_value=SAMPLER):
            fake.install(mock_api)
            mock_api.has_memory_api.return_value = False
            tokens = list(llm._generate_encoder_decoder(
                msgs, max_new_tokens=64, temperature=0.0, top_k=40, top_p=0.95,
                repeat_penalty=1.0))
        assert recreated == [([], 4096)]
        mock_api.llama_memory_clear.assert_not_called()
        assert fake.calls[0] == ("encode", _expected_encoder_input(msgs))
        assert tokens == reply_for(_expected_encoder_input(msgs))

    def test_a_closed_model_ends_the_reply_without_native_calls(self):
        llm = _llama()
        llm._stop.set()
        fake = FakeT5(llm)
        tokens, _, _ = _run(llm, fake, _user("x"))
        assert tokens == [] and fake.calls == []
        assert llm.last_finish_reason == "error"


class TestChatCompletionRouting:
    def _complete(self, llm, fake, stream):
        with patch.object(llama_mod, "api") as mock_api, \
                patch.object(llama_mod, "_build_sampler", return_value=SAMPLER), \
                patch.object(llama_mod, "_apply_model_template") as template:
            fake.install(mock_api)
            llm._tokenizer.token_to_piece_bytes.side_effect = lambda t: f"w{t}".encode()
            result = llm.create_chat_completion(
                _user("hi"), max_tokens=64, temperature=0.0, stream=stream)
            if stream:
                result = list(result)
        return result, template

    def test_non_streaming(self):
        llm = _llama()
        result, template = self._complete(llm, FakeT5(llm), stream=False)
        expected = "".join(f"w{t}" for t in reply_for(_expected_encoder_input(_user("hi"))))
        assert result["choices"][0]["message"]["content"] == expected
        assert result["choices"][0]["finish_reason"] == "stop"
        template.assert_not_called()

    def test_streaming(self):
        llm = _llama()
        chunks, template = self._complete(llm, FakeT5(llm), stream=True)
        text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
        expected = "".join(f"w{t}" for t in reply_for(_expected_encoder_input(_user("hi"))))
        assert text == expected
        assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
        template.assert_not_called()


# --------------------------------------------------------------------------- #
#  Load: detection, the encoder limit, speculation, vision                    #
# --------------------------------------------------------------------------- #

class TestLoad:
    META = {"general.architecture": "t5", "t5.attention.key_length": "64",
            "t5.attention.value_length": "64"}

    def _api(self, *, encoder_api=True, has_encoder=True, has_decoder=True,
             n_ubatch=512, meta=None):
        mock_api = MagicMock()
        mock_api.llama_model_default_params.return_value = SimpleNamespace(
            main_gpu=0, n_gpu_layers=0, use_mmap=True)
        mock_api.llama_context_default_params.return_value = SimpleNamespace(
            n_ctx=0, n_batch=0, n_ubatch=0, offload_kqv=True, flash_attn_type=0,
            n_rs_seq=0)
        mock_api.llama_load_model_from_file.return_value = 0xB00
        mock_api.llama_init_from_model.return_value = 0xC00
        mock_api.has_encoder_api.return_value = encoder_api
        mock_api.llama_model_has_encoder.return_value = has_encoder
        mock_api.llama_model_has_decoder.return_value = has_decoder
        mock_api.llama_n_ubatch.return_value = n_ubatch
        mock_api.has_model_meta_api.return_value = True
        table = dict(self.META if meta is None else meta)
        mock_api.llama_model_meta_val_str.side_effect = lambda model, key: table.get(key)
        mock_api.llama_model_n_layer.return_value = 8
        mock_api.has_kv_head_api.return_value = True
        mock_api.has_hybrid_api.return_value = True
        mock_api.llama_model_is_hybrid.return_value = False
        mock_api.llama_model_is_recurrent.return_value = False
        mock_api.llama_model_n_embd.return_value = 512
        mock_api.llama_model_n_head.return_value = 6
        mock_api.llama_model_n_head_kv.return_value = 6
        return mock_api

    def _build(self, mock_api, monkeypatch, **kwargs):
        monkeypatch.setattr("localm.config.load_config", lambda: {"main_gpu_index": None})
        from localm.inference.backends.llamacpp.llama import LlamaCpp
        with patch.object(llama_mod, "api", mock_api), \
                patch.object(LlamaCpp, "_load_mmproj") as load_mmproj:
            llm = LlamaCpp("m.gguf", n_ctx=4096, n_gpu_layers=0, verbose=True, **kwargs)
            llm.close()
        return llm, load_mmproj

    def test_an_encoder_decoder_model_is_detected_with_its_encoder_limit(self, monkeypatch):
        llm, _ = self._build(self._api(n_ubatch=512), monkeypatch)
        assert llm.is_encoder_decoder is True
        assert llm.encoder_input_limit == 512

    def test_the_limit_falls_back_to_the_clamped_request_without_the_accessor(self, monkeypatch):
        llm, _ = self._build(self._api(n_ubatch=None), monkeypatch)
        assert llm.encoder_input_limit == 2048

    def test_an_encoder_only_model_is_not_an_encoder_decoder(self, monkeypatch):
        llm, _ = self._build(self._api(has_decoder=False), monkeypatch)
        assert llm.is_encoder_decoder is False and llm.encoder_input_limit == 0

    def test_a_decoder_only_model_is_not_an_encoder_decoder(self, monkeypatch):
        llm, _ = self._build(self._api(has_encoder=False), monkeypatch)
        assert llm.is_encoder_decoder is False

    def test_mtp_is_off_with_an_encoder_decoder_status(self, monkeypatch):
        mock_api = self._api()
        llm, _ = self._build(mock_api, monkeypatch, mtp_enabled=True)
        assert llm.mtp_status == "encoder-decoder" and llm.supports_mtp is False
        mock_api.llama_model_mtp_support.assert_not_called()

    @pytest.mark.parametrize("spec_source", ["ngram", "mtp"])
    def test_the_draft_source_is_off_with_an_encoder_decoder_status(self, monkeypatch,
                                                                     spec_source):
        mock_api = self._api()
        llm, _ = self._build(mock_api, monkeypatch, spec_source=spec_source)
        report = llm.speculation_report()
        assert report["source"] == "off" and report["status"] == "encoder-decoder"
        assert llm._mtp_enabled is False
        mock_api.llama_decode.assert_not_called()
        mock_api.llama_model_mtp_support.assert_not_called()

    def test_a_decoder_only_model_keeps_its_draft_source(self, monkeypatch):
        llm, _ = self._build(self._api(has_encoder=False), monkeypatch, spec_source="ngram")
        assert llm.speculation_report()["source"] == "ngram"

    def test_no_recurrent_rollback_snapshots_are_requested(self, monkeypatch):
        mock_api = self._api()
        self._build(mock_api, monkeypatch, spec_source="ngram")
        cp = mock_api.llama_init_from_model.call_args.args[1]
        assert cp.n_rs_seq == 0

    def test_a_vision_projector_is_not_loaded(self, monkeypatch):
        llm, load_mmproj = self._build(self._api(), monkeypatch, mmproj_path="p.gguf")
        load_mmproj.assert_not_called()
        assert llm.supports_images is False

    def test_the_kv_cost_uses_the_declared_head_width(self, monkeypatch):
        llm, _ = self._build(self._api(), monkeypatch)
        assert llm.kv_bytes_per_token == 8 * 6 * (64 + 64) * 2 == 12288

    def test_the_kv_cost_counts_the_declared_decoder_layers(self, monkeypatch):
        meta = dict(self.META, **{"t5.decoder_block_count": "4"})
        llm, _ = self._build(self._api(meta=meta), monkeypatch)
        assert llm.kv_bytes_per_token == 4 * 6 * (64 + 64) * 2

    def test_a_runtime_without_the_encoder_api_refuses_a_t5_model(self, monkeypatch):
        mock_api = self._api(encoder_api=False)
        monkeypatch.setattr("localm.config.load_config", lambda: {"main_gpu_index": None})
        from localm.inference.backends.llamacpp.llama import LlamaCpp
        with patch.object(llama_mod, "api", mock_api):
            with pytest.raises(RuntimeError) as ei:
                LlamaCpp("m.gguf", n_ctx=4096, n_gpu_layers=0, verbose=True)
            message = str(ei.value)
            del ei
        assert "'t5'" in message and "llama_encode" in message and "setup-llama" in message
        mock_api.llama_free_model.assert_called_once_with(0xB00)
        mock_api.llama_init_from_model.assert_not_called()
        mock_api.llama_model_has_encoder.assert_not_called()

    def test_a_runtime_without_the_encoder_api_still_loads_other_models(self, monkeypatch):
        mock_api = self._api(encoder_api=False, meta={"general.architecture": "llama"})
        llm, _ = self._build(mock_api, monkeypatch)
        assert llm.is_encoder_decoder is False
        mock_api.llama_init_from_model.assert_called()


# --------------------------------------------------------------------------- #
#  The worker and the parent                                                   #
# --------------------------------------------------------------------------- #

class TestWorkerAndParent:
    def test_the_worker_counts_the_encoder_input(self):
        from localm.inference.backends.llamacpp._worker import GgufWorker
        worker = GgufWorker.__new__(GgufWorker)
        worker._llm = SimpleNamespace(is_encoder_decoder=True,
                                      encoder_tokens=lambda msgs: [1] * (len(msgs) + 4))
        with patch.object(llama_mod, "_apply_model_template") as template:
            assert worker.count_messages_tokens(_user("a") + _user("b")) == 6
        template.assert_not_called()

    def test_the_worker_load_reports_the_encoder_fields(self):
        from localm.inference.backends.llamacpp import _worker
        fake_llm = SimpleNamespace(supports_images=False, is_encoder_decoder=True,
                                   encoder_input_limit=512)
        worker = _worker.GgufWorker(model_path="m.gguf", mmproj_path=None, n_ctx=4096,
                                    n_gpu_layers=0, n_ctx_max=None, n_ctx_grow=4096)
        with patch("localm.inference.backends.llamacpp._loader.load_lib"), \
                patch("localm.inference.backends.llamacpp.LlamaCpp",
                      return_value=fake_llm):
            meta = worker.load()
        assert meta["encoder_decoder"] is True and meta["encoder_input_limit"] == 512

    def _backend(self, tmp_path):
        from localm.inference.backends.gguf import GgufBackend
        f = tmp_path / "model.gguf"
        f.write_bytes(b"\0" * 4096)
        return GgufBackend(str(f), n_gpu_layers=0, n_ctx=4096)

    def _load(self, backend, meta):
        backend.effective_gpu_layers = 0
        with patch("localm.discover.list_gpus", return_value=([], "ok")), \
                patch("localm.inference.backends.llamacpp._runner.ModelRunner."
                      "spawn_and_load", return_value=meta):
            backend._load_native()

    def test_the_parent_takes_the_encoder_limit_as_the_context_capacity(self, tmp_path):
        from localm.inference.engine import Engine
        b = self._backend(tmp_path)
        self._load(b, {"n_layers": 8, "kv_bytes_per_token": 0, "supports_images": False,
                       "encoder_decoder": True, "encoder_input_limit": 512})
        assert b.encoder_decoder is True and b.effective_ctx_max == 512
        engine = Engine.__new__(Engine)
        engine._backend = b
        assert engine.encoder_decoder is True and engine.context_capacity() == 512

    def test_a_decoder_only_load_keeps_its_ceiling(self, tmp_path):
        b = self._backend(tmp_path)
        self._load(b, {"n_layers": 8, "kv_bytes_per_token": 0, "supports_images": False,
                       "encoder_decoder": False, "encoder_input_limit": 512})
        assert b.encoder_decoder is False and b.effective_ctx_max != 512

    def test_an_encoder_decoder_chat_is_compacted_only_when_its_prompt_overflows(self):
        from localm.inference.http_server import _needs_compaction
        five = _user("a") * 5
        assert _needs_compaction(2048, 60, five) is True
        assert _needs_compaction(2048, 60, five, True) is False
        assert _needs_compaction(2048, 2048, five, True) is False
        assert _needs_compaction(2048, 2049, five, True) is True
        assert _needs_compaction(2048, 5000, _user("a") * 3, True) is False

    def test_only_a_real_true_marks_an_engine_encoder_decoder(self):
        from localm.inference.http_server import _engine_is_encoder_decoder
        assert _engine_is_encoder_decoder(SimpleNamespace(encoder_decoder=True)) is True
        assert _engine_is_encoder_decoder(SimpleNamespace(encoder_decoder=False)) is False
        assert _engine_is_encoder_decoder(MagicMock()) is False
        assert _engine_is_encoder_decoder(object()) is False

    def test_the_overflow_refusal_names_the_one_pass_limit(self):
        from localm.inference.http_server import context_overflow_detail
        text = context_overflow_detail(3000, 2048, True)
        assert "3000" in text and "2048" in text and "encoder-decoder" in text
        assert "n_ctx_max" not in text
        assert "n_ctx_max" in context_overflow_detail(3000, 2048)
