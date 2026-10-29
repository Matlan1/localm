# SPDX-License-Identifier: AGPL-3.0-or-later
"""The libmtmd speech binding: the ABI gate, the helper input layout, the
sampler settings read from a GGUF, the language and frame-budget rules, and the
synthesis loop driven over a fake native layer (cancellation, the budget, the
projector's CPU retry, and the state reset after every request)."""

import ctypes
import struct

import pytest

from localm.inference.backends.llamacpp import mtmd_gen as g


class _Fn:
    def __init__(self, argtypes=None):
        self.restype = None
        self.argtypes = argtypes


class _FakeLib:
    """A CDLL stand-in: ``lib[name]`` returns a fresh function object for every
    exported *name*, as ctypes does; attributes are the shared, cached ones."""

    def __init__(self, exports):
        self._exports = set(exports)

    def __getitem__(self, name):
        if name not in self._exports:
            raise AttributeError(f"function '{name}' not found")
        return _Fn()


def _exports(interface):
    return {*g._SIGNATURES, *g._INTERFACE_SIGNATURES[interface]}


def _offsets(cls):
    return {n: getattr(cls, n).offset for n, _t in cls._fields_}


class TestAbiGate:
    def test_an_mtmd_without_either_marker_export_is_refused(self):
        lib = _FakeLib(_exports(g.INTERFACE_LEGACY) - {"mtmd_gen_inp_default"})
        with pytest.raises(g.SpeechUnavailable, match="localm setup-llama"):
            g.bind_generation_api(lib)

    @pytest.mark.parametrize("interface", [g.INTERFACE_LEGACY, g.INTERFACE_STATE,
                                           g.INTERFACE_STATE_V2])
    def test_an_mtmd_missing_a_helper_function_is_refused(self, interface):
        lib = _FakeLib(_exports(interface) - {"mtmd_helper_gen_audio_step_gen"})
        with pytest.raises(g.SpeechUnavailable, match="lacks a speech generation function"):
            g.bind_generation_api(lib)

    def test_a_state_marker_without_the_state_functions_is_refused(self):
        lib = _FakeLib(_exports(g.INTERFACE_LEGACY) | {"mtmd_gen_state_init"})
        with pytest.raises(g.SpeechUnavailable, match="lacks a speech generation function"):
            g.bind_generation_api(lib)

    def test_a_state_v2_marker_without_the_state_functions_is_refused(self):
        lib = _FakeLib(_exports(g.INTERFACE_LEGACY) | {g._STATE_V2_MARKER})
        with pytest.raises(g.SpeechUnavailable, match="lacks a speech generation function"):
            g.bind_generation_api(lib)

    def test_an_mtmd_without_the_state_marker_binds_the_legacy_interface(self):
        ns = g.bind_generation_api(_FakeLib(_exports(g.INTERFACE_LEGACY)))
        assert ns.interface == g.INTERFACE_LEGACY
        assert ns.helper_input is g._HelperInputLegacy
        assert ns.mtmd_gen_audio_process.argtypes == [
            ctypes.c_void_p, ctypes.POINTER(g._GenInpLegacy),
            ctypes.POINTER(g._GenOutLegacy)]
        assert ns.mtmd_gen_inp_default.restype is g._GenInpLegacy
        assert ns.mtmd_helper_gen_audio_set_input.argtypes[1] is ctypes.POINTER(
            g._HelperInputLegacy)
        assert not hasattr(ns, "mtmd_gen_state_init")

    def test_an_mtmd_with_the_state_marker_binds_the_state_interface(self):
        ns = g.bind_generation_api(_FakeLib(_exports(g.INTERFACE_STATE)))
        assert ns.interface == g.INTERFACE_STATE
        assert ns.helper_input is g._HelperInput
        assert ns.mtmd_gen_audio_process.argtypes == [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(g._GenInp),
            ctypes.POINTER(g._GenOut)]
        assert ns.mtmd_gen_inp_default.restype is g._GenInp
        assert ns.mtmd_helper_gen_audio_set_input.argtypes[1] is ctypes.POINTER(
            g._HelperInput)
        assert ns.mtmd_gen_state_init.argtypes == [ctypes.c_void_p, g._GenStateParams]
        assert ns.mtmd_gen_state_free.argtypes == [ctypes.c_void_p]

    def test_an_mtmd_with_the_state_v2_marker_binds_the_state_v2_interface(self):
        ns = g.bind_generation_api(_FakeLib(_exports(g.INTERFACE_STATE_V2)))
        assert ns.interface == g.INTERFACE_STATE_V2
        assert ns.helper_input is g._HelperInputV2
        assert ns.mtmd_gen_audio_process.argtypes == [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(g._GenInpV2),
            ctypes.POINTER(g._GenOut)]
        assert ns.mtmd_gen_inp_default.restype is g._GenInpV2
        assert ns.mtmd_helper_gen_audio_set_input.argtypes[1] is ctypes.POINTER(
            g._HelperInputV2)
        assert ns.mtmd_gen_state_init.argtypes == [ctypes.c_void_p, g._GenStateParams]
        assert ns.mtmd_helper_gen_audio_next_is_text.restype is ctypes.c_bool

    def test_the_state_v2_marker_wins_over_the_state_marker(self):
        exports = _exports(g.INTERFACE_STATE) | _exports(g.INTERFACE_STATE_V2)
        assert g.bind_generation_api(_FakeLib(exports)).interface == g.INTERFACE_STATE_V2

    def test_a_state_mtmd_without_the_v2_marker_stays_on_the_state_interface(self):
        exports = _exports(g.INTERFACE_STATE_V2) - {g._STATE_V2_MARKER}
        ns = g.bind_generation_api(_FakeLib(exports))
        assert ns.interface == g.INTERFACE_STATE
        assert ns.helper_input is g._HelperInput

    def test_binding_leaves_the_shared_attributes_of_the_library_alone(self):
        lib = _FakeLib(_exports(g.INTERFACE_STATE))
        shared = _Fn(argtypes=[ctypes.c_char_p])
        lib.mtmd_bitmap_init_from_audio = shared
        ns = g.bind_generation_api(lib)
        assert lib.mtmd_bitmap_init_from_audio is shared
        assert shared.argtypes == [ctypes.c_char_p]
        assert ns.mtmd_bitmap_init_from_audio.argtypes == [
            ctypes.c_size_t, ctypes.POINTER(ctypes.c_float)]
        assert g.bind_generation_api(lib) is ns

    def test_the_helper_inputs_match_the_c_layouts(self):
        assert ctypes.sizeof(ctypes.c_void_p) == 8
        assert _offsets(g._HelperInputLegacy) == {
            "seq_id": 0, "prompt": 8, "prompt_len": 16, "speaker_ref": 24, "lang": 32,
            "top_k": 40, "top_p": 44, "seed": 48, "out_type": 52}
        assert ctypes.sizeof(g._HelperInputLegacy) == 56
        assert _offsets(g._HelperInput) == {
            "stream": 0, "seq_id": 4, "prompt": 8, "prompt_len": 16, "speaker_ref": 24,
            "lang": 32, "top_k": 40, "top_p": 44, "seed": 48, "out_type": 52}
        assert ctypes.sizeof(g._HelperInput) == 56
        assert _offsets(g._HelperInputV2) == {
            "stream": 0, "seq_id": 4, "prompt": 8, "prompt_len": 16, "n_past": 24,
            "out_mode": 28, "speaker_ref": 32, "lang": 40, "voice": 48, "top_k": 56,
            "top_p": 60, "seed": 64, "out_type": 68}
        assert ctypes.sizeof(g._HelperInputV2) == 72

    def test_the_gen_structs_match_the_c_layouts(self):
        assert _offsets(g._GenInpLegacy) == {
            "type": 0, "code0": 4, "embd": 8, "top_k": 16, "top_p": 20, "seed": 24,
            "temp": 28, "codes": 32, "n_codes": 40, "feats": 48, "n_feats": 56,
            "state_data": 64, "state_size": 72}
        assert ctypes.sizeof(g._GenInpLegacy) == 80
        assert _offsets(g._GenOutLegacy) == {
            "codes": 0, "n_codes": 8, "feats": 16, "n_feats": 24, "embd": 32,
            "is_eos": 40, "audio": 48, "n_samples": 56, "state_data": 64,
            "state_size": 72}
        assert ctypes.sizeof(g._GenOutLegacy) == 80
        assert _offsets(g._GenInp) == {
            "type": 0, "code0": 4, "embd": 8, "top_k": 16, "top_p": 20, "temp": 24,
            "codes": 32, "n_codes": 40, "feats": 48, "n_feats": 56}
        assert ctypes.sizeof(g._GenInp) == 64
        assert _offsets(g._GenInpV2) == {
            "type": 0, "code0": 4, "embd": 8, "top_k": 16, "top_p": 20, "temp": 24,
            "penalty_repeat": 28, "penalty_last_n": 32, "codes": 40, "n_codes": 48,
            "feats": 56, "n_feats": 64}
        assert ctypes.sizeof(g._GenInpV2) == 72
        assert _offsets(g._GenOut) == {
            "codes": 0, "n_codes": 8, "feats": 16, "n_feats": 24, "embd": 32,
            "is_eos": 40, "audio": 48, "n_samples": 56}
        assert ctypes.sizeof(g._GenOut) == 64
        assert ctypes.sizeof(g._GenStateParams) == 4


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

    def __init__(self, frames=4, fail_step=None, interface=g.INTERFACE_STATE):
        self.interface = interface
        self.helper_input = g._INTERFACE_STRUCTS[interface]["mtmd_helper_gen_audio_inp"]
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
        if type(inp) is not self.helper_input:
            return 1
        self.inputs.append({"layout": type(inp), "stream": getattr(inp, "stream", None),
                            "seed": inp.seed, "top_k": inp.top_k, "top_p": inp.top_p,
                            "lang": inp.lang, "n_past": getattr(inp, "n_past", None),
                            "out_mode": getattr(inp, "out_mode", None),
                            "voice": getattr(inp, "voice", None),
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

    def mtmd_gen_inp_default(self, ctx):
        if self.interface == g.INTERFACE_STATE_V2:
            return g._GenInpV2()
        if self.interface == g.INTERFACE_STATE:
            return g._GenInp()
        return g._GenInpLegacy(seed=0xFFFFFFFF)

    def mtmd_gen_audio_process(self, ctx, *args):
        if self.interface != g.INTERFACE_LEGACY:
            if len(args) != 3:
                self.calls.append("invalid gen state")
                return 1
            self.calls.append(("process", args[1]._obj.type))
            return 0
        inp = args[0]._obj
        self.calls.append(("reseed", inp.type, inp.seed, bool(inp.embd)))
        return 0


class _FakeProjector:
    def __init__(self, on_gpu):
        self.on_gpu = on_gpu
        self._ctx = 5
        self.retries = 0
        self.moves = []

    def switch_device(self, *, gpu):
        self.moves.append("gpu" if gpu else "cpu")
        self.on_gpu = gpu
        return True

    def retry_on_cpu(self):
        self.retries += 1
        self.on_gpu = False
        return True


@pytest.fixture(params=[g.INTERFACE_STATE, g.INTERFACE_STATE_V2, g.INTERFACE_LEGACY])
def synth(monkeypatch, request):
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
    s._m = _FakeMtmd(interface=request.param)
    s._mtmd = _FakeProjector(on_gpu=False)
    s._helper = 7
    s._ctx = 3
    s._mem = 4
    s._vocab = 2
    s._n_vocab = 100
    s._n_embd = 8
    s._last_gen_seed = None
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


class TestProjectorTopP:
    def test_the_projector_top_p_makes_it_sort_its_candidates(self):
        sent = ctypes.c_float(g.projector_top_p(1.0)).value
        assert sent < 1.0
        assert sent == 1.0 - 2.0 ** -24
        assert ctypes.c_float(sent + 2.0 ** -25).value == 1.0

    @pytest.mark.parametrize("value", [1.0, 0.0, -0.5, 2.0, float("inf"), float("nan"),
                                       1e-50, 1.0 - 1e-12])
    def test_a_value_outside_zero_and_one_is_capped(self, value):
        assert g.projector_top_p(value) == 1.0 - 2.0 ** -24

    @pytest.mark.parametrize("value", [0.95, 0.5, 1e-6])
    def test_a_value_between_zero_and_one_is_kept(self, value):
        assert g.projector_top_p(value) == value


class TestSynthesisLoop:
    def test_frames_are_reported_and_the_wav_returned(self, synth):
        seen = []
        out = synth.synthesize("hello there", seed=42, on_progress=seen.append)
        assert seen == [1, 2, 3, 4]
        assert out.frames == 4 and out.sample_rate == 24000 and out.seed == 42
        assert out.wav == synth._m.wav
        assert synth._m.inputs[0]["seed"] == 42 and synth._m.inputs[0]["top_k"] == 40

    def test_the_helper_gets_the_input_layout_of_the_interface(self, synth):
        synth.synthesize("hello", seed=3)
        sent = synth._m.inputs[0]
        assert sent["layout"] is synth._m.helper_input
        if synth._m.interface != g.INTERFACE_LEGACY:
            assert sent["stream"] is False
        assert sent["seed"] == 3

    def test_the_state_v2_input_asks_for_a_new_sequence_with_the_default_output(
            self, synth):
        if synth._m.interface != g.INTERFACE_STATE_V2:
            pytest.skip("the state_v2 interface only")
        synth.synthesize("hello", language="en", seed=3)
        sent = synth._m.inputs[0]
        assert (sent["n_past"], sent["out_mode"], sent["voice"]) == (0, 0, None)
        assert sent["lang"] == b"english" and sent["top_k"] == 40

    def test_a_repeated_seed_resets_the_legacy_generator_before_the_request(self, synth):
        if synth._m.interface != g.INTERFACE_LEGACY:
            pytest.skip("the legacy interface only")
        synth.synthesize("hello", seed=42)
        assert not any(isinstance(c, tuple) and c[0] == "reseed" for c in synth._m.calls)
        synth._m.calls.clear()
        synth.synthesize("hello", seed=42)
        reseeds = [c for c in synth._m.calls if isinstance(c, tuple) and c[0] == "reseed"]
        assert reseeds == [("reseed", 0, 43, True)]
        assert synth._m.calls.index(reseeds[0]) < synth._m.calls.index("set_input")
        synth._m.calls.clear()
        synth.synthesize("hello", seed=7)
        assert not any(isinstance(c, tuple) and c[0] == "reseed" for c in synth._m.calls)

    @pytest.mark.parametrize("on_gpu", [False, True])
    def test_a_repeated_seed_on_a_state_interface_makes_no_generator_call(
            self, synth, on_gpu):
        if synth._m.interface == g.INTERFACE_LEGACY:
            pytest.skip("the state interfaces only")
        synth._mtmd.on_gpu = on_gpu
        outs = [synth.synthesize("hello", seed=42) for _ in range(3)]
        assert [o.seed for o in outs] == [42, 42, 42]
        assert [i["seed"] for i in synth._m.inputs] == [42, 42, 42]
        assert "invalid gen state" not in synth._m.calls
        assert not any(isinstance(c, tuple) and c[0] == "process" for c in synth._m.calls)
        assert synth._mtmd.retries == 0 and synth._mtmd.on_gpu is on_gpu

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
        out = synth.synthesize("hello")
        assert synth._mtmd.retries == 1 and out.frames == 4
        assert synth._m.calls.count("init") == 1 and synth._m.calls.count("free") == 1

    def test_a_projector_that_cannot_reopen_on_the_cpu_makes_it_unavailable(self, synth):
        synth._mtmd.on_gpu = True
        synth._mtmd.retry_on_cpu = lambda: False
        synth._m.fail_step = 2
        with pytest.raises(g.SpeechUnavailable, match="could not be reopened"):
            synth.synthesize("hello")
        with pytest.raises(g.SpeechUnavailable, match="closed"):
            synth.synthesize("hello", seed=1)

    def test_a_seeded_request_keeps_the_projector_on_the_gpu(self, synth):
        synth._mtmd.on_gpu = True
        device_at_set_input = []
        real = synth._m.mtmd_helper_gen_audio_set_input

        def spy(helper, inp_ref):
            device_at_set_input.append("gpu" if synth._mtmd.on_gpu else "cpu")
            return real(helper, inp_ref)

        synth._m.mtmd_helper_gen_audio_set_input = spy
        out = synth.synthesize("hello", seed=5)
        assert out.seed == 5 and out.frames == 4
        assert device_at_set_input == ["gpu"]
        assert synth._mtmd.moves == [] and synth._mtmd.on_gpu is True
        assert synth._m.calls.count("free") == 0 and synth._m.calls.count("init") == 0

    @pytest.mark.parametrize("model_top_p", [1.0, 0.0, -1.0, 1.5])
    def test_the_projector_gets_a_top_p_below_one_when_the_model_keeps_everything(
            self, synth, model_top_p):
        synth._sampling = g.sampling_params_from_meta(
            lambda key: {g._META_TOP_K: "50", g._META_TOP_P: str(model_top_p)}.get(key))
        synth.synthesize("hello", seed=5)
        sent = synth._m.inputs[0]["top_p"]
        assert sent < 1.0 and sent == 1.0 - 2.0 ** -24
        assert synth._m.inputs[0]["top_k"] == 50

    @pytest.mark.parametrize("model_top_p", [0.95, 0.5])
    def test_the_projector_gets_the_model_top_p_when_it_is_below_one(self, synth, model_top_p):
        synth._sampling = g.sampling_params_from_meta(
            lambda key: {g._META_TOP_P: str(model_top_p)}.get(key))
        synth.synthesize("hello")
        assert synth._m.inputs[0]["top_p"] == ctypes.c_float(model_top_p).value

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
