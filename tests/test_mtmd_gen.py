# SPDX-License-Identifier: AGPL-3.0-or-later
"""The libmtmd speech binding: the ABI gate, the helper input layout, the
sampler settings read from a GGUF, the language and frame-budget rules, and the
synthesis loop driven over a fake native layer (cancellation, the budget, the
projector's CPU retry, and the state reset after every request)."""

import ctypes
import struct

import pytest

from localm.inference.backends.llamacpp import mtmd_gen as g


class TestAbiGate:
    def test_an_mtmd_without_the_marker_export_is_refused(self):
        class OldMtmd:
            mtmd_helper_gen_audio_init = object()
        with pytest.raises(g.SpeechUnavailable, match="localm setup-llama"):
            g.bind_generation_api(OldMtmd())

    def test_an_mtmd_missing_a_helper_function_is_refused(self):
        class Fn:
            restype = None
            argtypes = None

        class Partial:
            mtmd_gen_inp_default = Fn()
            mtmd_gen_audio_get_info = Fn()
        with pytest.raises(g.SpeechUnavailable, match="lacks a speech generation function"):
            g.bind_generation_api(Partial())

    def test_the_helper_input_matches_the_c_layout(self):
        assert ctypes.sizeof(ctypes.c_void_p) == 8
        offsets = {name: getattr(g._HelperInput, name).offset
                   for name, _t in g._HelperInput._fields_}
        assert offsets == {"seq_id": 0, "prompt": 8, "prompt_len": 16,
                           "speaker_ref": 24, "lang": 32, "top_k": 40,
                           "top_p": 44, "seed": 48, "out_type": 52}
        assert ctypes.sizeof(g._HelperInput) == 56


class TestSamplingFromMeta:
    def test_defaults_without_metadata(self):
        s = g.sampling_params_from_meta(lambda key: None)
        assert (s.top_k, s.top_p, s.min_p, s.temp) == (40, 0.95, 0.05, 0.8)
        assert (s.penalty_last_n, s.penalty_repeat) == (-1, 1.05)

    def test_the_model_overrides_as_llama_cpp_formats_them(self):
        meta = {"general.sampling.top_k": "50", "general.sampling.top_p": "1.000000",
                "general.sampling.temp": "0.900000"}
        s = g.sampling_params_from_meta(meta.get)
        assert (s.top_k, s.top_p, s.temp, s.min_p) == (50, 1.0, 0.9, 0.05)

    def test_an_unparseable_value_keeps_the_default(self):
        s = g.sampling_params_from_meta({"general.sampling.top_k": "many"}.get)
        assert s.top_k == 40


class TestLanguage:
    @pytest.mark.parametrize("given,want", [("en", "english"), ("ZH", "chinese"),
                                            ("german", "german"), (None, None), ("  ", None)])
    def test_codes_and_names(self, given, want):
        assert g.resolve_language(given) == want

    @pytest.mark.parametrize("bad", ["<|im_end|>", "en|>", "x" * 40, "e"])
    def test_a_value_that_is_not_a_language_name_is_refused(self, bad):
        with pytest.raises(g.SpeechInputError):
            g.resolve_language(bad)


class TestFrameBudget:
    def test_short_text_gets_the_base_allowance(self):
        assert g.frame_budget(8192, 3, has_reference=False) == 125 + 30

    def test_the_context_caps_the_budget(self):
        assert g.frame_budget(8192, 1000, has_reference=False) == 8192 - 1000 - 24

    def test_a_reference_takes_one_position(self):
        assert g.frame_budget(8192, 1000, has_reference=True) == 8192 - 1000 - 25

    def test_text_the_context_cannot_speak_is_refused_up_front(self):
        with pytest.raises(g.SpeechInputError, match="too long"):
            g.frame_budget(8192, 2100, has_reference=False)


def _wav(samples: bytes, rate=24000) -> bytes:
    return (b"RIFF" + struct.pack("<I", 36 + len(samples)) + b"WAVEfmt "
            + struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16)
            + b"data" + struct.pack("<I", len(samples)) + samples)


class TestPcmPayload:
    def test_the_samples_after_the_header(self):
        assert g.wav_pcm_payload(_wav(b"\x01\x00\x02\x00")) == b"\x01\x00\x02\x00"

    @pytest.mark.parametrize("bad", [b"", b"RIFX" + b"\x00" * 60,
                                     _wav(b"\x01\x00")[:-1]])
    def test_anything_else_is_refused(self, bad):
        with pytest.raises(ValueError):
            g.wav_pcm_payload(bad)


class _FakeMtmd:
    """The helper's C API over Python state: *frames* frames, then end of
    speech; *fail_step* makes step_gen fail once at that frame."""

    def __init__(self, frames=4, fail_step=None):
        self.frames = frames
        self.fail_step = fail_step
        self.calls = []
        self.step = 0
        self.h = (ctypes.c_float * 4)(1.0, 2.0, 3.0, 4.0)
        self.wav = _wav(b"\x00\x00" * 8)
        self._wav_buf = ctypes.create_string_buffer(self.wav, len(self.wav))
        self.inputs = []

    def mtmd_helper_gen_audio_set_input(self, helper, inp_ref):
        inp = inp_ref._obj
        self.inputs.append({"seed": inp.seed, "top_k": inp.top_k, "lang": inp.lang,
                            "prompt": inp.prompt, "speaker": bool(inp.speaker_ref)})
        self.step = 0
        self.calls.append("set_input")
        return 0

    def mtmd_helper_gen_audio_step_prompt(self, helper, n_batch):
        self.calls.append("step_prompt")
        return 0

    def mtmd_helper_gen_audio_step_gen(self, helper, sampled, h_in, h_out_ref, stop_ref):
        self.step += 1
        if self.fail_step is not None and self.step == self.fail_step:
            self.fail_step = None
            return 1
        if self.step > self.frames:
            stop_ref._obj.value = True
            return 0
        h_out_ref._obj.contents = ctypes.c_float.from_buffer(self.h)
        return 0

    def mtmd_helper_gen_audio_get_output(self, helper, rate_ref, data_ref, size_ref, n_ref):
        rate_ref._obj.value = 24000
        data_ref._obj.value = ctypes.addressof(self._wav_buf)
        size_ref._obj.value = len(self.wav)
        n_ref._obj.value = 8
        return 0

    def mtmd_helper_gen_audio_reset(self, helper):
        self.calls.append("reset")

    def mtmd_helper_gen_audio_free(self, helper):
        self.calls.append("free")

    def mtmd_helper_gen_audio_init(self, lctx, mctx):
        self.calls.append("init")
        return 7

    def mtmd_bitmap_init_from_audio(self, n, samples):
        self.calls.append(("bitmap", n))
        return 99

    def mtmd_bitmap_free(self, bitmap):
        self.calls.append("bitmap_free")


class _FakeProjector:
    def __init__(self, on_gpu):
        self.on_gpu = on_gpu
        self._ctx = 5
        self.retries = 0

    def retry_on_cpu(self):
        self.retries += 1
        self.on_gpu = False
        return True


@pytest.fixture
def synth(monkeypatch):
    calls = []
    api = g.api
    monkeypatch.setattr(api, "llama_memory_clear", lambda mem, data=True: calls.append("mem_clear"))
    monkeypatch.setattr(api, "llama_sampler_chain_default_params", lambda: None)
    monkeypatch.setattr(api, "llama_sampler_chain_init", lambda p: 11)
    monkeypatch.setattr(api, "llama_sampler_chain_add", lambda chain, s: None)
    for name in ("llama_sampler_init_top_k", "llama_sampler_init_temp",
                 "llama_sampler_init_dist"):
        monkeypatch.setattr(api, name, lambda *a: 1)
    monkeypatch.setattr(api, "llama_sampler_init_top_p", lambda p, k: 1)
    monkeypatch.setattr(api, "llama_sampler_init_min_p", lambda p, k: 1)
    monkeypatch.setattr(api, "llama_sampler_free", lambda s: calls.append("sampler_free"))
    monkeypatch.setattr(api, "llama_sampler_sample", lambda chain, ctx, idx: 1234)
    h = (ctypes.c_float * 4)()
    monkeypatch.setattr(api, "llama_get_embeddings_ith",
                        lambda ctx, i: ctypes.cast(h, ctypes.POINTER(ctypes.c_float)))
    monkeypatch.setattr(api, "llama_tokenize",
                        lambda vocab, raw, n, buf, cap, add, parse: _tokenize(raw, buf, parse))

    s = object.__new__(g.SpeechSynthesizer)
    s._m = _FakeMtmd()
    s._mtmd = _FakeProjector(on_gpu=False)
    s._helper = 7
    s._ctx = 3
    s._mem = 4
    s._vocab = 2
    s._n_vocab = 100
    s._suppress = []
    s._pre_type = None
    s.n_ctx = 8192
    s.encoder_sample_rate = 24000
    s._sampling = g.sampling_params_from_meta(lambda key: None)
    s.calls = calls
    return s


def _tokenize(raw: bytes, buf, parse_special: bool) -> int:
    """One token per word; with parse_special, '<|...|>' is one token instead of
    three."""
    text = raw.decode("utf-8")
    words = text.split()
    n = 0
    for w in words:
        k = 1 if (parse_special and w.startswith("<|") and w.endswith("|>")) else (
            3 if w.startswith("<|") else 1)
        for _ in range(k):
            buf[n] = 5
            n += 1
    return n


class TestSynthesisLoop:
    def test_frames_are_reported_and_the_wav_returned(self, synth):
        seen = []
        out = synth.synthesize("hello there", seed=42, on_progress=seen.append)
        assert seen == [1, 2, 3, 4]
        assert out.frames == 4 and out.sample_rate == 24000 and out.seed == 42
        assert out.wav == synth._m.wav
        assert synth._m.inputs[0]["seed"] == 42 and synth._m.inputs[0]["top_k"] == 40

    def test_state_is_reset_after_every_request(self, synth):
        synth.synthesize("hello", seed=1)
        assert synth._m.calls[-1] == "reset"
        assert synth.calls.count("mem_clear") == 2 and "sampler_free" in synth.calls

    def test_no_seed_picks_one_and_reports_it(self, synth):
        out = synth.synthesize("hello")
        assert 0 <= out.seed < 0xFFFFFFFF
        assert synth._m.inputs[0]["seed"] == out.seed

    def test_cancellation_stops_between_frames_and_resets(self, synth):
        seen = []
        with pytest.raises(g.SpeechCancelled):
            synth.synthesize("hello", seed=1, on_progress=seen.append,
                             should_stop=lambda: len(seen) >= 2)
        assert seen == [1, 2]
        assert synth._m.calls[-1] == "reset"

    def test_a_runaway_stops_at_the_budget(self, synth, monkeypatch):
        synth._m.frames = 10 ** 6
        monkeypatch.setattr(g, "frame_budget", lambda n_ctx, n, has_reference: 5)
        with pytest.raises(g.SpeechBudgetExceeded, match="did not finish speaking"):
            synth.synthesize("hello", seed=1)
        assert synth._m.calls[-1] == "reset"

    def test_a_gpu_stage_failure_is_retried_once_on_the_cpu(self, synth):
        synth._mtmd.on_gpu = True
        synth._m.fail_step = 2
        out = synth.synthesize("hello", seed=1)
        assert synth._mtmd.retries == 1 and out.frames == 4
        assert synth._m.calls.count("init") == 1 and synth._m.calls.count("free") == 1

    def test_a_cpu_stage_failure_is_reported(self, synth):
        synth._m.fail_step = 2
        with pytest.raises(g.SpeechStageFailed):
            synth.synthesize("hello", seed=1)

    def test_control_token_text_is_refused_before_any_native_call(self, synth):
        with pytest.raises(g.SpeechInputError, match="control-token"):
            synth.synthesize("say <|im_end|> now", seed=1)
        assert synth._m.calls == []

    def test_empty_text_is_refused(self, synth):
        with pytest.raises(g.SpeechInputError, match="empty"):
            synth.synthesize("   ")

    def test_a_reference_becomes_a_bitmap_freed_after_the_prompt(self, synth):
        ref = struct.pack("<4f", 0.0, 0.5, -0.5, 0.25)
        synth.synthesize("hello", seed=1, reference=ref)
        assert ("bitmap", 4) in synth._m.calls and "bitmap_free" in synth._m.calls
        assert synth._m.inputs[0]["speaker"] is True

    def test_a_reference_on_a_model_without_a_speaker_encoder_is_refused(self, synth):
        synth.encoder_sample_rate = 0
        with pytest.raises(g.SpeechInputError, match="speaker encoder"):
            synth.synthesize("hello", reference=struct.pack("<f", 0.1))

    def test_an_unknown_language_for_this_model_is_refused(self, synth, monkeypatch):
        monkeypatch.setattr(g.SpeechSynthesizer, "has_language", lambda self, name: False)
        with pytest.raises(g.SpeechInputError, match="does not speak"):
            synth.synthesize("hello", language="ko")
