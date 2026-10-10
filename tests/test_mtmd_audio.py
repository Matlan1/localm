# SPDX-License-Identifier: AGPL-3.0-or-later
"""Audio input through mtmd: MtmdContext and LlamaCpp's media prefill, driven
through the real code against the fakes in tests/_fake_mtmd.py."""
import base64
import struct
from array import array
from unittest.mock import MagicMock, patch

import pytest

from localm.inference.backends.base import (
    AUDIO_CPU_FALLBACK_STATUS,
    AUDIO_UNSUPPORTED_MESSAGE,
    AudioInputError,
    UnsupportedInputError,
    VISION_CPU_FALLBACK_STATUS,
    VisionInputError,
)
from localm.inference.backends.llamacpp import mtmd as lmtmd
from localm.inference.backends.llamacpp.llama import LlamaCpp

from tests._bare_llama import make_bare_llama
from tests._fake_mtmd import (
    MARKER,
    FakeKV,
    FakeLlamaApi,
    FakeMtmdLib,
    fake_create_batch,
    make_mtmd_context,
    solid_image,
)

RATE = 16000


def _clip(value: float, n: int = 3200) -> lmtmd.AudioClip:
    samples = array("f", [value] * n)
    return lmtmd.AudioClip(samples.tobytes(), n)


def _wav_b64(values, rate=RATE) -> str:
    frames = struct.pack("<%dh" % len(values), *values)
    fmt = struct.pack("<HHIIHH", 1, 1, rate, rate * 2, 2, 16)
    body = (b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt
            + b"data" + struct.pack("<I", len(frames)) + frames)
    return base64.b64encode(b"RIFF" + struct.pack("<I", len(body)) + body).decode()


def _audio_part(values=(1000,) * 3200, fmt="wav"):
    return {"type": "input_audio", "input_audio": {"data": _wav_b64(list(values)),
                                                   "format": fmt}}


class _Rig:
    """One LlamaCpp with a fake KV cache, fake llama api and fake mtmd."""

    def __init__(self, *, vision=True, audio_rate=RATE, n_ctx=4096):
        self.kv = FakeKV()
        self.api = FakeLlamaApi(self.kv, n_ctx=n_ctx)
        self.lib = FakeMtmdLib(self.kv)
        self.llm = make_bare_llama(_model_ptr=0x11, _ctx_ptr=0x22, _ctx_capacity=n_ctx,
                                   _n_ctx=n_ctx, _n_ctx_grow=256)
        self.llm._mtmd = make_mtmd_context(self.lib, vision=vision, audio_rate=audio_rate)
        self.llm._create_batch = fake_create_batch

    def prefill(self, prompt, media, *, on_status=None):
        vprompt = self.llm._mtmd.tokenize(prompt, media, add_special=True)
        try:
            with patch("localm.inference.backends.llamacpp.llama.api", self.api):
                return self.llm._prefill_vision(vprompt, vprompt.n_tokens + 8, on_status)
        finally:
            vprompt.free()


class TestTokenize:
    def test_an_audio_clip_becomes_an_audio_bitmap_and_an_audio_chunk(self):
        rig = _Rig()
        clip = _clip(0.25)
        vprompt = rig.llm._mtmd.tokenize(f"user {MARKER} transcribe", [clip],
                                         add_special=True)
        try:
            assert rig.lib.audio_bitmaps == [(clip.n_samples, clip.samples)]
            media = [c for c in vprompt.chunks if c.tokens is None]
            assert [c.kind for c in media] == ["audio"]
            assert media[0].key[0] == lmtmd._audio_content_id(clip)
            assert vprompt.n_audio == 1
            assert [c.kind for c in vprompt.chunks if c.tokens is not None] == ["text", "text"]
        finally:
            vprompt.free()

    def test_identical_clips_share_a_content_id_and_different_ones_do_not(self):
        assert (lmtmd._audio_content_id(_clip(0.25))
                == lmtmd._audio_content_id(_clip(0.25)))
        assert (lmtmd._audio_content_id(_clip(0.25))
                != lmtmd._audio_content_id(_clip(0.5)))
        assert (lmtmd._audio_content_id(_clip(0.0, 3200))
                != lmtmd._audio_content_id(_clip(0.0, 3201)))

    def test_audio_to_a_vision_only_projector_is_refused_before_any_native_call(self):
        rig = _Rig(audio_rate=0)
        with pytest.raises(VisionInputError, match="no audio encoder"):
            rig.llm._mtmd.tokenize(f"{MARKER}", [_clip(0.1)], add_special=True)
        assert rig.lib.audio_bitmaps == []

    def test_an_image_to_an_audio_only_projector_is_refused(self):
        rig = _Rig(vision=False)
        with pytest.raises(VisionInputError, match="no vision encoder"):
            rig.llm._mtmd.tokenize(f"{MARKER}", [solid_image(5)], add_special=True)
        assert rig.lib.bitmaps == {}

    @pytest.mark.parametrize("clip", [
        lmtmd.AudioClip(array("f", [0.1]).tobytes(), 1),
        lmtmd.AudioClip(b"", 0),
        lmtmd.AudioClip(array("f", [0.1] * 10).tobytes(), 11),
    ])
    def test_a_clip_the_runtime_would_abort_on_is_refused_first(self, clip):
        rig = _Rig()
        with pytest.raises(VisionInputError, match="malformed"):
            rig.llm._mtmd.tokenize(f"{MARKER}", [clip], add_special=True)
        assert rig.lib.audio_bitmaps == []

    def test_earlier_bitmaps_are_freed_when_a_later_item_is_refused(self):
        rig = _Rig()
        with pytest.raises(VisionInputError):
            rig.llm._mtmd.tokenize(f"{MARKER} {MARKER}",
                                   [_clip(0.1), lmtmd.AudioClip(b"", 0)],
                                   add_special=True)
        assert rig.lib.bitmaps_freed == 1

    def test_a_tokenize_failure_on_audio_says_audio(self):
        rig = _Rig()
        rig.lib.rc_tokenize = 2
        with pytest.raises(VisionInputError, match="process this audio"):
            rig.llm._mtmd.tokenize(f"{MARKER}", [_clip(0.1)], add_special=True)


class TestEvalAndCache:
    def test_a_follow_up_turn_does_not_encode_the_same_clip_again(self):
        rig = _Rig()
        clip = _clip(0.3)
        rig.prefill(f"user {MARKER} transcribe assistant", [clip])
        assert rig.lib.encode_calls == 1
        rig.prefill(f"user {MARKER} transcribe assistant hello user again assistant", [clip])
        assert rig.lib.encode_calls == 1

    def test_a_different_clip_is_encoded(self):
        rig = _Rig()
        rig.prefill(f"user {MARKER} transcribe", [_clip(0.3)])
        rig.prefill(f"user {MARKER} transcribe", [_clip(0.4)])
        assert rig.lib.encode_calls == 2

    @pytest.mark.parametrize("on_gpu,label", [(True, "GPU"), (False, "CPU")])
    def test_status_names_audio_when_only_audio_is_encoded(self, on_gpu, label):
        rig = _Rig()
        rig.llm._mtmd.on_gpu = on_gpu
        statuses = []
        rig.prefill(f"user {MARKER} hi", [_clip(0.2)], on_status=statuses.append)
        assert statuses == [f"Encoding audio ({label})..."]

    def test_status_names_image_when_an_image_is_encoded_too(self):
        rig = _Rig()
        statuses = []
        rig.prefill(f"user {MARKER} {MARKER} hi", [_clip(0.2), solid_image(9)],
                    on_status=statuses.append)
        assert statuses == ["Encoding image (GPU)..."]

    def test_a_failed_gpu_audio_encode_is_retryable_and_says_audio(self):
        rig = _Rig()
        rig.lib.fail_encode = True
        with pytest.raises(lmtmd.MtmdGpuEncodeFailed, match="encode this audio"):
            rig.prefill(f"user {MARKER} hi", [_clip(0.2)])

    def test_a_failed_cpu_audio_decode_is_a_plain_input_error(self):
        rig = _Rig()
        rig.llm._mtmd.on_gpu = False
        rig.lib.fail_decode = True
        with pytest.raises(VisionInputError, match="evaluate this audio") as info:
            rig.prefill(f"user {MARKER} hi", [_clip(0.2)])
        assert not isinstance(info.value, lmtmd.MtmdGpuEncodeFailed)


class TestMessagesWithMarkers:
    def test_audio_parts_become_markers_and_clips_at_the_projector_rate(self):
        messages = [{"role": "user", "content": [
            {"type": "text", "text": "transcribe"}, _audio_part([8192] * 3200)]}]
        out, media = LlamaCpp._messages_with_markers(messages, MARKER, RATE)
        assert out[0]["content"] == f"transcribe\n{MARKER}"
        assert len(media) == 1 and isinstance(media[0], lmtmd.AudioClip)
        assert media[0].n_samples == 3200
        assert array("f", media[0].samples)[0] == 0.25

    def test_audio_is_resampled_to_the_projector_rate(self, monkeypatch):
        from localm.inference import media as media_mod
        monkeypatch.setattr(media_mod, "_resample_with_av", lambda *a: None)
        part = {"type": "input_audio",
                "input_audio": {"data": _wav_b64([0] * 8000, rate=8000), "format": "wav"}}
        _out, media = LlamaCpp._messages_with_markers(
            [{"role": "user", "content": [part]}], MARKER, RATE)
        assert media[0].n_samples == 16000

    def test_audio_without_an_audio_projector_is_refused(self):
        messages = [{"role": "user", "content": [_audio_part()]}]
        with pytest.raises(UnsupportedInputError) as info:
            LlamaCpp._messages_with_markers(messages, MARKER, 0)
        assert str(info.value) == AUDIO_UNSUPPORTED_MESSAGE

    def test_undecodable_audio_is_an_audio_input_error(self):
        part = {"type": "input_audio", "input_audio": {"data": "!!", "format": "wav"}}
        with pytest.raises(AudioInputError):
            LlamaCpp._messages_with_markers([{"role": "user", "content": [part]}],
                                            MARKER, RATE)

    @pytest.mark.parametrize("url", [
        "data:image/png;base64",
        "data:image/png;base64,bm90IGFuIGltYWdl",
    ])
    def test_an_unreadable_image_is_a_per_request_error(self, url):
        part = {"type": "image_url", "image_url": {"url": url}}
        with pytest.raises(VisionInputError, match="could not be read") as info:
            LlamaCpp._messages_with_markers([{"role": "user", "content": [part]}],
                                            MARKER, RATE)
        assert "bm90IGFuIGltYWdl" not in str(info.value)

    def test_a_refused_image_fetch_keeps_the_policy_reason(self, monkeypatch):
        from localm.inference import media as media_mod
        from localm.netpolicy import NetworkPolicyError

        def refuse(url):
            raise NetworkPolicyError("private addresses are blocked")
        monkeypatch.setattr(media_mod, "decode_image_url", refuse)
        part = {"type": "image_url", "image_url": {"url": "http://10.0.0.1/a.png"}}
        with pytest.raises(VisionInputError, match="private addresses are blocked"):
            LlamaCpp._messages_with_markers([{"role": "user", "content": [part]}],
                                            MARKER, RATE)


class TestGenerateDispatch:
    def _llm(self, mtmd):
        llm = make_bare_llama(_model_ptr=111, _ctx_ptr=222)
        llm._mtmd = mtmd
        llm.is_encoder_decoder = False
        llm.is_diffusion = False
        return llm

    def test_audio_to_a_model_without_a_projector_is_refused_not_dropped(self):
        llm = self._llm(None)
        mock_api = MagicMock()
        mock_api.llama_model_chat_template.return_value = None
        with patch("localm.inference.backends.llamacpp.llama.api", mock_api), \
                patch("localm.inference.backends.llamacpp.llama._apply_model_template",
                      return_value=("prompt", None)):
            with pytest.raises(UnsupportedInputError) as info:
                llm.create_chat_completion([{"role": "user", "content": [_audio_part()]}])
        assert str(info.value) == AUDIO_UNSUPPORTED_MESSAGE
        llm._tokenizer.encode.assert_not_called()

    def test_capability_properties_follow_the_projector(self):
        lib = FakeMtmdLib(FakeKV())
        assert self._llm(make_mtmd_context(lib, vision=False, audio_rate=RATE)).supports_images is False
        assert self._llm(make_mtmd_context(lib, vision=False, audio_rate=RATE)).supports_audio is True
        assert self._llm(make_mtmd_context(lib, vision=True, audio_rate=0)).supports_audio is False
        assert self._llm(None).supports_audio is False

    @pytest.mark.parametrize("media_parts,expected", [
        ([_audio_part()], AUDIO_CPU_FALLBACK_STATUS),
        ([_audio_part(), {"type": "image_url", "image_url": {"url": "x"}}],
         VISION_CPU_FALLBACK_STATUS),
    ])
    def test_cpu_retry_status_names_the_media_that_failed(self, media_parts, expected):
        class _StopAfterRetry(Exception):
            pass

        from tests._fake_mtmd import fake_vision_prompt
        llm = self._llm(MagicMock())
        llm._mtmd.marker = MARKER
        llm._mtmd.on_gpu = True
        llm._mtmd.audio_sample_rate = RATE
        llm._mtmd.retry_on_cpu.return_value = True
        llm._mtmd.encode_count = 0
        llm._mtmd.has_embedding.return_value = False
        llm._mtmd.eval_media_chunk.side_effect = [lmtmd.MtmdGpuEncodeFailed(),
                                                  _StopAfterRetry()]

        def _tokenize(prompt, media, add_special):
            vp = fake_vision_prompt(image_tokens=4)
            kinds = ["audio" if isinstance(m, lmtmd.AudioClip) else "image" for m in media]
            vp.chunks[0] = lmtmd.MtmdChunk(0x5001, None, ("k", 0, 4, 4), 4, 4, kinds[0])
            for i, kind in enumerate(kinds[1:], start=1):
                vp.chunks.insert(i, lmtmd.MtmdChunk(0x5100 + i, None, (f"k{i}", 0, 4, 4),
                                                    4, 4, kind))
            return vp
        llm._mtmd.tokenize.side_effect = _tokenize

        mock_api = MagicMock()
        mock_api.llama_model_chat_template.return_value = None
        mock_api.llama_n_ctx.return_value = 4096
        statuses = []
        with patch("localm.inference.backends.llamacpp.llama.api", mock_api), \
                patch("localm.inference.media.decode_image_url",
                      return_value=MagicMock(convert=lambda _m: MagicMock(
                          width=1, height=1, tobytes=lambda: b"\x00\x00\x00"))):
            gen = llm._generate_image(
                [{"role": "user", "content": media_parts}], max_new_tokens=8,
                temperature=0.8, top_k=40, top_p=0.95, repeat_penalty=1.1,
                on_status=statuses.append)
            with pytest.raises(_StopAfterRetry):
                next(gen)
        assert statuses[-1] == expected


class _InitLib:
    """Just enough of the mtmd CDLL for the real MtmdContext.__init__."""

    def __init__(self, *, vision, audio, rate, has_api=True):
        self.localm_has_audio_api = has_api
        self._vision, self._audio, self._rate = vision, audio, rate

    def mtmd_context_params_default(self):
        return lmtmd._MtmdParams()

    def mtmd_init_from_file(self, path, model_ptr, params):
        return 0x1234

    def mtmd_support_vision(self, ctx):
        return self._vision

    def mtmd_support_audio(self, ctx):
        return self._audio

    def mtmd_get_audio_sample_rate(self, ctx):
        return self._rate

    def mtmd_default_marker(self):
        return MARKER.encode()

    def mtmd_free(self, ctx):
        pass


class TestContextInit:
    @pytest.fixture(autouse=True)
    def _no_probe(self, monkeypatch):
        monkeypatch.setattr(lmtmd, "_input_text_class", lmtmd._MtmdInputTextV2)
        monkeypatch.setenv("LOCALM_MTMD_CPU", "1")

    def _ctx(self, monkeypatch, **kw):
        monkeypatch.setattr(lmtmd, "_load_lib", lambda: _InitLib(**kw))
        return lmtmd.MtmdContext("/fake/mmproj.gguf", 0xBEEF)

    def test_an_audio_projector_reports_its_rate(self, monkeypatch):
        ctx = self._ctx(monkeypatch, vision=False, audio=True, rate=16000)
        assert (ctx.supports_vision, ctx.supports_audio, ctx.audio_sample_rate) == \
            (False, True, 16000)

    def test_a_vision_projector_takes_no_audio(self, monkeypatch):
        ctx = self._ctx(monkeypatch, vision=True, audio=False, rate=-1)
        assert (ctx.supports_audio, ctx.audio_sample_rate) == (False, 0)

    def test_an_audio_encoder_without_a_rate_is_disabled_loudly(self, monkeypatch, caplog):
        with caplog.at_level("WARNING", logger="localm"):
            ctx = self._ctx(monkeypatch, vision=False, audio=True, rate=0)
        assert ctx.supports_audio is False
        assert any("no sample rate" in r.getMessage() for r in caplog.records)

    def test_a_runtime_without_the_audio_calls_takes_no_audio(self, monkeypatch):
        ctx = self._ctx(monkeypatch, vision=True, audio=True, rate=16000, has_api=False)
        assert ctx.supports_audio is False


class TestLoadKeepsAnAudioOnlyProjector:
    def _load(self, monkeypatch, *, vision, audio):
        llm = make_bare_llama()
        fake = MagicMock(supports_vision=vision, supports_audio=audio)
        monkeypatch.setattr(lmtmd, "MtmdContext", lambda *a, **k: fake)
        monkeypatch.setattr(lmtmd, "compatible_mmproj_path", lambda p: p)
        llm._load_mmproj("/fake/mmproj.gguf", verbose=True)
        return llm, fake

    def test_audio_only_is_kept(self, monkeypatch):
        llm, fake = self._load(monkeypatch, vision=False, audio=True)
        assert llm._mtmd is fake
        assert llm.supports_audio is True and llm.supports_images is False

    def test_neither_is_freed_and_logged(self, monkeypatch, caplog):
        with caplog.at_level("WARNING", logger="localm"):
            llm, fake = self._load(monkeypatch, vision=False, audio=False)
        assert llm._mtmd is None
        fake.free.assert_called_once()
        assert any("neither a vision nor an audio encoder" in r.getMessage()
                   for r in caplog.records)
