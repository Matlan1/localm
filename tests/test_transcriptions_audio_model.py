# SPDX-License-Identifier: AGPL-3.0-or-later
"""POST /v1/audio/transcriptions answered by an installed GGUF model that hears
audio, when faster-whisper is not installed or the request names that model.

The real route, multipart parsing, engine selection, the pin, the inference gate,
generation and the real WAV decoder (through ``LlamaCpp._messages_with_markers``)
run; only the model engine and the Whisper worker are replaced.
"""

import base64
import struct
import threading
import time
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import localm.inference.http_server as hs
from localm.plugins.builtin.voice import audio_model
from localm.plugins.builtin.voice import transcriptions as tr  # noqa: F401 - identity check in the plugin fixture

URL = "/v1/audio/transcriptions"
MODEL = "asr-model"
REPLY = "the quick brown fox"


def _wav(seconds=1.0, rate=16000):
    n = int(seconds * rate)
    frames = struct.pack("<%dh" % n, *([1000] * n))
    fmt = struct.pack("<HHIIHH", 1, 1, rate, rate * 2, 2, 16)
    body = (b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt
            + b"data" + struct.pack("<I", len(frames)) + frames)
    return b"RIFF" + struct.pack("<I", len(body)) + body


class _Engine:
    """A model engine that hears audio. chat_stream decodes the clip with the
    real GGUF-backend decoder before yielding the reply."""

    display_name = MODEL
    loaded = True
    can_be_multimodal = True
    supports_images = False

    def __init__(self, reply=REPLY, *, audio=True, finish="stop", tokens=10,
                 capacity=4096, fail=None, delay=0.0):
        self.supports_audio = audio
        self.last_finish_reason = finish
        self.active_requests = 0
        self._backend = SimpleNamespace(mmproj_path="p.gguf", model_path=None)
        self.reply = reply
        self.tokens = tokens
        self.capacity = capacity
        self.fail = fail
        self.delay = delay
        self.running = 0
        self.peak = 0
        self.lock = threading.Lock()
        self.calls = []
        self.media = []
        self.pinned_during_call = None

    def context_capacity(self):
        return self.capacity

    def count_messages_tokens(self, messages):
        return self.tokens

    def chat_stream(self, messages, **kwargs):
        from localm.inference.backends.llamacpp.llama import LlamaCpp
        self.calls.append((messages, kwargs))
        self.pinned_during_call = self.active_requests
        _, media = LlamaCpp._messages_with_markers(messages, "<__media__>", audio_rate=16000)
        self.media.extend(media)
        with self.lock:
            self.running += 1
            self.peak = max(self.peak, self.running)
        try:
            time.sleep(self.delay)
            if self.fail is not None:
                raise self.fail
            yield self.reply
        finally:
            with self.lock:
                self.running -= 1


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path))
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.delenv("LOCALM_REQUIRE_AUTH", raising=False)
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", tmp_path)
    monkeypatch.setattr(cfg, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", tmp_path / "registry.json")
    return tmp_path


@pytest.fixture
def plugin(home, monkeypatch):
    """The voice plugin, installed. PluginManager imports it under a synthetic
    package name, so the module objects the running route uses are these."""
    import sys

    import localm.voice as voice
    monkeypatch.setattr(voice, "prefetch_stt_model",
                        lambda allow_download=None: (False, "stubbed in tests"))
    from localm.plugins.engine import PluginManager
    app = FastAPI()
    PluginManager(app, external_root=home / "noplugins").install("voice")
    for t in threading.enumerate():
        if t.name == voice.PREFETCH_THREAD_NAME:
            t.join(timeout=10)
    loaded = SimpleNamespace(
        app=app, tr=sys.modules["_localm_plugin_voice.transcriptions"],
        audio=sys.modules["_localm_plugin_voice.audio_model"])
    assert loaded.tr is not tr and loaded.audio is not audio_model
    return loaded


@pytest.fixture
def state(plugin, monkeypatch):
    """The knobs a test turns: whether Whisper is installed, which registered
    models hear audio, which are registered at all, and the engine served."""
    import localm.voice as voice
    s = SimpleNamespace(whisper=False, audio=[MODEL], registered={MODEL, "chat-model"},
                        engine=_Engine(), asked=[], whisper_calls=[], scans=[])
    monkeypatch.setattr(plugin.tr, "_whisper_installed", lambda: s.whisper)
    monkeypatch.setattr(plugin.audio, "audio_model_names",
                        lambda: s.scans.append(1) or list(s.audio))
    monkeypatch.setattr(plugin.audio, "is_registered", lambda n: n in s.registered)

    async def get_engine(name, **kw):
        s.asked.append((name, kw))
        return s.engine

    monkeypatch.setattr(hs, "get_engine", get_engine)
    monkeypatch.setattr(hs, "_engines", {})

    def worker(data, name, language, timeout, *, opts=None, local_files_only=True,
               blocked_reason=None):
        s.whisper_calls.append(data)
        return "whisper words", {"language": "en", "duration": 1.0, "segments": []}

    monkeypatch.setattr(voice, "_stt_plan", lambda language: {
        "name": "base", "language": language, "timeout": 5.0,
        "local_files_only": True, "blocked_reason": None})
    monkeypatch.setattr(voice, "_run_in_worker_detailed", worker)
    return s


@pytest.fixture
def client(state, plugin):
    with TestClient(plugin.app) as c:
        yield c


def _post(client, *, data=None, wav=None, name="a.wav", ctype="audio/wav"):
    wav = _wav() if wav is None else wav
    return client.post(URL, data=data or {}, files={"file": (name, wav, ctype)})


class TestEngineChoice:
    def test_whisper_stays_the_default_when_installed(self, client, state):
        state.whisper = True
        r = _post(client, data={"model": "whisper-1"})
        assert r.status_code == 200 and r.json() == {"text": "whisper words"}
        assert state.engine.calls == [] and state.asked == []
        assert state.scans == []

    def test_whisper_installed_and_no_model_named(self, client, state):
        state.whisper = True
        assert _post(client).json() == {"text": "whisper words"}
        assert state.asked == []

    @pytest.mark.parametrize("model", [None, "whisper-1", "WHISPER-1", "base"])
    def test_an_audio_model_answers_when_whisper_is_missing(self, client, state, model):
        r = _post(client, data={"model": model} if model else {})
        assert r.status_code == 200, r.text
        assert r.json() == {"text": REPLY}
        assert r.headers["X-Localm-Transcription-Model"] == MODEL
        assert state.whisper_calls == []
        assert state.asked == [(MODEL, {"activate": False})]

    def test_naming_the_audio_model_wins_over_installed_whisper(self, client, state):
        state.whisper = True
        r = _post(client, data={"model": MODEL})
        assert r.status_code == 200 and r.json() == {"text": REPLY}
        assert state.whisper_calls == []

    def test_a_loaded_audio_model_is_preferred(self, client, state, monkeypatch):
        state.audio = ["aaa-first", MODEL]
        monkeypatch.setattr(hs, "_engines", {MODEL: _Engine()})
        _post(client)
        assert state.asked[0][0] == MODEL

    def test_without_one_loaded_the_first_by_name_answers(self, client, state):
        state.audio = ["aaa-first", MODEL]
        _post(client)
        assert state.asked[0][0] == "aaa-first"

    def test_no_whisper_and_no_audio_model_is_501_with_the_way_out(
            self, client, state, monkeypatch):
        import localm.voice as voice
        state.audio = []

        def no_package(language):
            raise voice.VoiceError("Speech-to-text needs the faster-whisper package.",
                                   code="needs-faster-whisper")

        monkeypatch.setattr(voice, "_stt_plan", no_package)
        r = _post(client)
        assert r.status_code == 501
        assert state.whisper_calls == []
        assert "faster-whisper" in r.json()["detail"]
        assert "localm pull ggml-org/Qwen3-ASR-0.6B-GGUF" in r.json()["detail"]

    def test_an_unknown_model_is_404_and_names_the_choices(self, client, state):
        r = _post(client, data={"model": "gpt-4o-transcribe"})
        assert r.status_code == 404
        assert "whisper-1" in r.json()["detail"] and MODEL in r.json()["detail"]
        assert state.asked == []

    def test_a_registered_model_that_cannot_hear_is_400(self, client, state):
        r = _post(client, data={"model": "chat-model"})
        assert r.status_code == 400
        assert "cannot take audio" in r.json()["detail"] and MODEL in r.json()["detail"]
        assert state.asked == []

    def test_a_registered_model_that_cannot_hear_with_no_audio_model_says_how_to_get_one(
            self, client, state):
        state.audio = []
        r = _post(client, data={"model": "chat-model"})
        assert r.status_code == 400 and "localm pull" in r.json()["detail"]


class TestWhatTheModelReceives:
    def test_the_clip_and_the_instruction_reach_the_model_greedy(self, client, state):
        wav = _wav(0.5)
        _post(client, wav=wav)
        (messages, kwargs), = state.engine.calls
        audio, text = messages[0]["content"]
        assert audio == {"type": "input_audio", "input_audio": {
            "data": base64.b64encode(wav).decode(), "format": "wav"}}
        assert text == {"type": "text", "text": "Transcribe this."}
        assert kwargs["temperature"] == 0.0 and kwargs["repeat_penalty"] == 1.0
        assert kwargs["max_tokens"] == 4096 - 10

    def test_the_reply_budget_never_exceeds_the_ceiling(self, client, state):
        state.engine = _Engine(capacity=1_000_000)
        _post(client)
        assert state.engine.calls[0][1]["max_tokens"] == audio_model.REPLY_TOKEN_CEILING

    def test_the_decoder_hands_the_model_16khz_mono_samples(self, client, state):
        _post(client, wav=_wav(1.0, rate=22050))
        clip, = state.engine.media
        assert clip.n_samples == 16000

    def test_temperature_language_and_prompt_are_passed_on(self, client, state):
        _post(client, data={"temperature": "0.4", "language": "DE",
                            "prompt": "Kubernetes, Tilde"})
        (messages, kwargs), = state.engine.calls
        assert kwargs["temperature"] == 0.4
        text = messages[0]["content"][1]["text"]
        assert "'de'" in text and "Kubernetes, Tilde" in text

    def test_the_reply_budget_fits_what_the_prompt_leaves(self, client, state):
        state.engine = _Engine(tokens=4000, capacity=4096)
        _post(client)
        assert state.engine.calls[0][1]["max_tokens"] == 96

    def test_the_engine_is_pinned_during_the_call_and_released_after(self, client, state):
        r = _post(client)
        assert r.status_code == 200
        assert state.engine.pinned_during_call == 1
        assert state.engine.active_requests == 0

    @pytest.mark.parametrize("name,ctype,expected", [
        ("clip.mp3", "audio/mpeg", "mp3"),
        ("clip", "audio/x-m4a", "m4a"),
        ("clip", "audio/ogg; codecs=opus", "ogg"),
        ("clip.weird_ext!", "application/octet-stream", "audio"),
        ("", "", "audio"),
        ("clip.wav", "audio/wav", "audio"),
        ("clip.mp3", "audio/wav", "mp3"),
        ("clip", "audio/x-wav", "audio"),
    ])
    def test_format_label_of_a_non_wav_upload(self, name, ctype, expected):
        assert audio_model.format_label(b"ID3....", name, ctype) == expected

    def test_a_wav_payload_is_labelled_wav_whatever_the_name(self):
        assert audio_model.format_label(_wav(0.1), "x.mp3", "audio/mpeg") == "wav"


class TestTranscriptCleaning:
    @pytest.mark.parametrize("reply,expected", [
        ("language English<asr_text>The quick brown fox.", "The quick brown fox."),
        ("language None<asr_text>", ""),
        ("language Chinese<asr_text>你好。", "你好。"),
        ("  plain words from another model " + chr(10), "plain words from another model"),
        ("language English<asr_text>One.language English<asr_text> Two.", "One. Two."),
        ("the language English is spoken here", "the language English is spoken here"),
        ("<think>\nhmm, audio\n</think>\nHello there.", "Hello there."),
    ])
    def test_clean_transcript(self, reply, expected):
        assert audio_model.clean_transcript(reply) == expected

    def test_the_route_returns_the_cleaned_text(self, client, state):
        state.engine = _Engine(reply="language English<asr_text>the quick brown fox")
        assert _post(client).json() == {"text": "the quick brown fox"}

    def test_a_clip_with_no_speech_is_a_200_with_empty_text(self, client, state):
        state.engine = _Engine(reply="language None<asr_text>")
        r = _post(client)
        assert r.status_code == 200 and r.json() == {"text": ""}


class TestResponseFormats:
    def test_text_format(self, client, state):
        r = _post(client, data={"response_format": "text"})
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/plain")
        assert r.text == REPLY + "\n"

    @pytest.mark.parametrize("fmt", ["verbose_json", "srt", "vtt"])
    def test_formats_that_need_timestamps_are_refused_before_the_model_runs(
            self, client, state, fmt):
        r = _post(client, data={"response_format": fmt})
        assert r.status_code == 400
        assert "timestamps" in r.json()["detail"] and MODEL in r.json()["detail"]
        assert state.engine.calls == [] and state.asked == []

    def test_the_same_formats_still_work_through_whisper(self, client, state):
        state.whisper = True
        r = _post(client, data={"response_format": "srt"})
        assert r.status_code == 200


class TestFailures:
    def test_a_model_without_an_audio_encoder_is_400_with_guidance(self, client, state):
        state.engine = _Engine(audio=False)
        r = _post(client)
        assert r.status_code == 400
        assert "audio" in r.json()["detail"].lower()
        assert state.engine.calls == [] and state.engine.active_requests == 0

    def test_a_malformed_wav_is_400(self, client, state):
        r = _post(client, wav=b"RIFF....WAVEjunk")
        assert r.status_code == 400 and "WAV" in r.json()["detail"]
        assert state.engine.active_requests == 0

    def test_a_clip_shorter_than_the_minimum_is_400(self, client, state):
        r = _post(client, wav=_wav(0.01))
        assert r.status_code == 400 and "too short" in r.json()["detail"]

    def test_other_audio_without_the_voice_extra_is_501(self, client, state, monkeypatch):
        from localm.inference.backends.base import AudioDecodeUnavailable

        def no_av(raw, target, max_seconds):
            raise AudioDecodeUnavailable("reading other formats needs the voice extra")

        monkeypatch.setattr("localm.inference.media._decode_with_av", no_av)
        r = _post(client, wav=b"ID3\x03not-a-wav", name="a.mp3", ctype="audio/mpeg")
        assert r.status_code == 501 and "voice extra" in r.json()["detail"]
        assert state.engine.calls[0][0][0]["content"][0]["input_audio"]["format"] == "mp3"

    def test_a_transcript_cut_off_at_the_reply_limit_is_an_error_not_a_short_text(
            self, client, state):
        state.engine = _Engine(finish="length")
        r = _post(client)
        assert r.status_code == 502 and "cut off" in r.json()["detail"]

    def test_a_generation_interrupted_midway_is_an_error_not_a_short_text(
            self, client, state):
        state.engine = _Engine(finish="error")
        r = _post(client)
        assert r.status_code == 502 and "interrupted" in r.json()["detail"]
        assert state.engine.active_requests == 0

    def test_the_backend_refusing_a_clip_that_overflows_the_context_is_413(
            self, client, state):
        from localm.inference.backends.base import ContextCapacityExceededError
        state.engine = _Engine(fail=ContextCapacityExceededError("clip is too long"))
        r = _post(client)
        assert r.status_code == 413 and "too long" in r.json()["detail"]
        assert state.engine.active_requests == 0

    def test_a_runtime_failure_does_not_disclose_the_model_path(self, client, state):
        state.engine = _Engine(fail=RuntimeError(
            "Failed to load model: C:\\Users\\bob\\models\\asr.gguf (out of memory)"))
        r = _post(client)
        assert r.status_code == 502
        assert "bob" not in r.json()["detail"] and "out of memory" in r.json()["detail"]

    def test_a_model_name_outside_latin1_still_gets_a_header_and_a_200(
            self, client, state):
        state.audio = ["\u00e4rzte-asr-\u6a21\u578b"]
        r = _post(client)
        assert r.status_code == 200 and r.json() == {"text": REPLY}
        assert r.headers["X-Localm-Transcription-Model"].isascii()

    def test_a_clip_that_does_not_fit_the_context_is_413(self, client, state):
        state.engine = _Engine(tokens=5000, capacity=4096)
        r = _post(client)
        assert r.status_code == 413
        assert state.engine.calls == [] and state.engine.active_requests == 0

    def test_a_native_runtime_failure_is_reported_and_releases_the_pin(
            self, client, state):
        state.engine = _Engine(fail=RuntimeError("decode exploded"))
        r = _post(client)
        assert r.status_code == 502 and "decode exploded" in r.json()["detail"]
        assert state.engine.active_requests == 0

    def test_an_empty_upload_is_400_before_any_model_is_touched(self, client, state):
        r = client.post(URL, files={"file": ("a.wav", b"", "audio/wav")})
        assert r.status_code == 400 and state.asked == []

    def test_a_missing_file_is_400(self, client, state):
        r = client.post(URL, data={"model": MODEL},
                        files={"note": ("n.txt", b"hello", "text/plain")})
        assert r.status_code == 400 and "'file' field is required" in r.json()["detail"]
        assert state.asked == []


class TestConcurrency:
    def test_two_requests_at_once_are_both_answered_one_at_a_time(self, client, state):
        state.engine = _Engine(delay=0.15)
        results = []

        def go():
            results.append(_post(client).json())

        threads = [threading.Thread(target=go) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        assert results == [{"text": REPLY}, {"text": REPLY}]
        assert state.engine.peak == 1
        assert state.engine.active_requests == 0
