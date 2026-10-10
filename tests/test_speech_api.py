# SPDX-License-Identifier: AGPL-3.0-or-later
"""POST /v1/audio/speech: the OpenAI-compatible speech route.

The real route, request parsing, validation, model and voice resolution, job
handling and error mapping run; only the synthesis itself is replaced (it needs
a model and the isolated worker), at ``localm.inference.speech.synthesize``.
"""

import asyncio
import json
import struct
import threading

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from localm.inference import speech
from localm.inference._speech_runner import SpeechWorkerHung
from localm.inference.backends.base import PretokenizerUnsafeInputError
from localm.inference.backends.llamacpp import mtmd_gen as g

URL = "/v1/audio/speech"


def _wav(samples: bytes, rate=24000) -> bytes:
    return (b"RIFF" + struct.pack("<I", 36 + len(samples)) + b"WAVEfmt "
            + struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16)
            + b"data" + struct.pack("<I", len(samples)) + samples)


SAMPLES = struct.pack("<4h", 1, -1, 2, -2)
WAV = _wav(SAMPLES)


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


def _register(home, name, *, model_type="tts", mmproj=True):
    models = home / "models"
    models.mkdir(exist_ok=True)
    path = models / f"{name}.gguf"
    path.write_bytes(b"GGUF")
    entry = {"path": str(path), "source": "local", "model_type": model_type}
    if mmproj:
        mm = models / f"mmproj-{name}.gguf"
        mm.write_bytes(b"GGUF")
        entry["mmproj"] = str(mm)
    reg_file = home / "registry.json"
    reg = json.loads(reg_file.read_text(encoding="utf-8")) if reg_file.exists() else {}
    reg[name] = entry
    reg_file.write_text(json.dumps(reg), encoding="utf-8")
    return entry


class _Synth:
    """Stands in for speech.synthesize and records every call."""

    def __init__(self):
        self.calls = []
        self.error = None
        self.rate = 24000
        self.frames = 3

    def __call__(self, model, text, *, language=None, reference_wav=None, seed=None,
                 on_progress=None, should_cancel=None):
        self.calls.append({"model": model, "text": text, "language": language,
                           "reference": reference_wav, "seed": seed})
        if on_progress is not None:
            on_progress({"stage": "loading"})
            for n in range(1, self.frames + 1):
                on_progress({"stage": "speaking", "frames": n, "seconds": n / 12.5})
        if self.error is not None:
            raise self.error
        return speech.SpeechOutput(wav=_wav(SAMPLES, self.rate), sample_rate=self.rate,
                                   n_samples=4, frames=self.frames,
                                   seed=7 if seed is None else seed)


@pytest.fixture
def synth(monkeypatch):
    s = _Synth()
    monkeypatch.setattr(speech, "synthesize", s)
    return s


def _tts_app(home, kernel=False):
    from localm.plugins.engine import PluginManager
    if kernel:
        from localm.inference.http_server import create_app
        app = create_app(None)
    else:
        app = FastAPI()
    PluginManager(app, external_root=home / "noplugins").install("tts")
    if getattr(app.state, "jobs", None) is None:
        from localm.plugins.gui.jobs import JobManager
        app.state.jobs = JobManager()
    return app


@pytest.fixture
def client(home, synth):
    _register(home, "voice-1")
    with TestClient(_tts_app(home)) as c:
        yield c


def _post(client, body=None, **kw):
    return client.post(URL, json=body if body is not None else {"input": "Hello there."}, **kw)


class TestFormats:
    def test_default_is_wav_with_the_seed(self, client, synth):
        r = _post(client)
        assert r.status_code == 200, r.text
        assert r.headers["content-type"] == "audio/wav"
        assert r.content == WAV and r.headers["x-localm-seed"] == "7"
        assert synth.calls[0]["text"] == "Hello there." and synth.calls[0]["reference"] is None

    def test_pcm_is_the_samples_without_the_header(self, client):
        r = _post(client, {"input": "hi", "response_format": "pcm"})
        assert r.status_code == 200 and r.headers["content-type"] == "audio/pcm"
        assert r.content == SAMPLES

    def test_pcm_from_a_model_that_is_not_24_khz_is_refused(self, client, synth):
        synth.rate = 16000
        r = _post(client, {"input": "hi", "response_format": "pcm"})
        assert r.status_code == 400 and "24 kHz" in r.json()["detail"]

    @pytest.mark.parametrize("fmt", ["mp3", "opus", "aac", "flac"])
    def test_compressed_formats_are_refused_not_ignored(self, client, synth, fmt):
        r = _post(client, {"input": "hi", "response_format": fmt})
        assert r.status_code == 400 and "'wav' or 'pcm'" in r.json()["detail"]
        assert synth.calls == []

    def test_an_unknown_format_is_refused(self, client):
        r = _post(client, {"input": "hi", "response_format": "ogg"})
        assert r.status_code == 400 and "must be one of" in r.json()["detail"]


class TestValidation:
    @pytest.mark.parametrize("body,needle", [
        ({}, "'input' field is required"),
        ({"input": "   "}, "empty"),
        ({"input": "x" * 4097}, "too long"),
        ({"input": 5}, "must be a string"),
        ({"input": "hi", "speed": 1.5}, "speed other than 1.0"),
        ({"input": "hi", "speed": "fast"}, "speed must be a number"),
        ({"input": "hi", "instructions": "whisper it"}, "instructions are not supported"),
        ({"input": "hi", "stream_format": "sse"}, "stream_format 'audio'"),
        ({"input": "hi", "seed": -1}, "seed must be between"),
        ({"input": "hi", "seed": 2 ** 32}, "seed must be between"),
        ({"input": "hi", "seed": "abc"}, "seed must be an integer"),
        ({"input": "hi", "seed": True}, "seed must be an integer"),
        ({"input": "hi", "seed": 1.5}, "seed must be an integer"),
        ({"input": "hi", "language": "<|x|>"}, "language must be"),
    ])
    def test_bad_requests_are_400_before_any_synthesis(self, client, synth, body, needle):
        r = _post(client, body)
        assert r.status_code == 400, r.text
        assert needle in r.json()["detail"]
        assert synth.calls == []

    @pytest.mark.parametrize("extra", [{"speed": 1.0}, {"speed": 1}, {"instructions": ""},
                                       {"stream_format": "audio"}, {"user": "someone"}])
    def test_accepted_no_op_values_and_unknown_fields_pass(self, client, extra):
        assert _post(client, {"input": "hi", **extra}).status_code == 200

    def test_seed_and_language_reach_the_synthesis(self, client, synth):
        assert _post(client, {"input": "hi", "seed": 5, "language": "de"}).status_code == 200
        assert synth.calls[0]["seed"] == 5 and synth.calls[0]["language"] == "de"

    def test_invalid_json_is_400(self, client):
        r = client.post(URL, content=b"{not json", headers={"content-type": "application/json"})
        assert r.status_code == 400 and "valid JSON" in r.json()["detail"]

    def test_a_json_array_is_400(self, client):
        assert client.post(URL, json=["hi"]).status_code == 400

    def test_an_oversized_body_is_413(self, client, synth):
        r = client.post(URL, content=b"{" + b" " * (300 * 1024) + b"}",
                        headers={"content-type": "application/json"})
        assert r.status_code == 413 and synth.calls == []


class TestModelResolution:
    @pytest.mark.parametrize("model", [None, "", "tts-1", "tts-1-hd", "gpt-4o-mini-tts",
                                       "localm", "voice-1"])
    def test_aliases_and_the_name_resolve_to_the_registered_model(self, client, synth, model):
        body = {"input": "hi"} if model is None else {"input": "hi", "model": model}
        assert _post(client, body).status_code == 200
        assert synth.calls[0]["model"].name == "voice-1"

    def test_an_unknown_model_is_404(self, client):
        r = _post(client, {"input": "hi", "model": "nope"})
        assert r.status_code == 404 and "not registered" in r.json()["detail"]

    def test_a_chat_model_is_422(self, client, home):
        _register(home, "chatty", model_type="llm")
        r = _post(client, {"input": "hi", "model": "chatty"})
        assert r.status_code == 422 and "not a text-to-speech model" in r.json()["detail"]

    def test_an_alias_with_two_speech_models_is_400(self, client, home):
        _register(home, "voice-2")
        r = _post(client, {"input": "hi", "model": "tts-1"})
        assert r.status_code == 400 and "voice-1, voice-2" in r.json()["detail"]

    def test_a_model_without_its_mmproj_is_422(self, client, home):
        _register(home, "bare", mmproj=False)
        r = _post(client, {"input": "hi", "model": "bare"})
        assert r.status_code == 422 and "mmproj" in r.json()["detail"]

    def test_no_speech_model_registered_is_404(self, home, synth):
        with TestClient(_tts_app(home)) as c:
            r = _post(c)
        assert r.status_code == 404 and "No text-to-speech model" in r.json()["detail"]


class TestVoices:
    def test_a_named_voice_sends_its_recording(self, client, synth, home):
        (home / "voices").mkdir()
        (home / "voices" / "narrator.wav").write_bytes(WAV)
        assert _post(client, {"input": "hi", "voice": "narrator"}).status_code == 200
        assert synth.calls[0]["reference"] == WAV

    @pytest.mark.parametrize("voice", ["alloy", "../registry", "a/b", "x" * 80])
    def test_an_unknown_voice_is_400_and_lists_the_voices(self, client, synth, voice):
        r = _post(client, {"input": "hi", "voice": voice})
        assert r.status_code == 400 and "Available voices: default" in r.json()["detail"]
        assert synth.calls == []

    def test_an_uploaded_voice_file_is_sent_from_memory(self, client, synth):
        r = client.post(URL, data={"input": "hi", "seed": "3"},
                        files={"voice_file": ("ref.wav", WAV, "audio/wav")})
        assert r.status_code == 200, r.text
        assert synth.calls[0]["reference"] == WAV and synth.calls[0]["seed"] == 3

    def test_a_voice_name_and_a_voice_file_together_are_400(self, client, synth):
        r = client.post(URL, data={"input": "hi", "voice": "narrator"},
                        files={"voice_file": ("ref.wav", WAV, "audio/wav")})
        assert r.status_code == 400 and "not both" in r.json()["detail"]

    def test_another_file_field_is_400(self, client):
        r = client.post(URL, data={"input": "hi"}, files={"file": ("a.wav", WAV, "audio/wav")})
        assert r.status_code == 400 and "voice_file" in r.json()["detail"]

    def test_an_empty_voice_file_is_400(self, client):
        r = client.post(URL, data={"input": "hi"}, files={"voice_file": ("a.wav", b"", "audio/wav")})
        assert r.status_code == 400 and "empty" in r.json()["detail"]


class TestErrorMapping:
    @pytest.mark.parametrize("error,status", [
        (g.SpeechInputError("bad reference"), 400),
        (PretokenizerUnsafeInputError("too long a run"), 400),
        (g.SpeechUnavailable("runtime predates"), 501),
        (speech.SpeechUnavailableError("load failed"), 503),
        (SpeechWorkerHung("no progress"), 504),
        (g.SpeechBudgetExceeded("did not finish"), 502),
        (RuntimeError("worker crashed"), 502),
    ])
    def test_each_failure_has_its_status(self, client, synth, error, status):
        synth.error = error
        r = _post(client)
        assert r.status_code == status
        assert str(error) in r.json()["detail"]


class TestJobAndPrivacy:
    def test_the_synthesis_runs_as_a_speak_job_with_frame_progress(self, home, synth):
        _register(home, "voice-1")
        app = _tts_app(home)
        with TestClient(app) as c:
            assert _post(c).status_code == 200
        (row,) = app.state.jobs.snapshot()
        assert row["kind"] == "speak" and row["status"] == "done"
        job = app.state.jobs.get(row["id"])
        progress = [e for e in job._history if e.get("type") == "progress"]
        assert progress[0]["phase"] == "loading the speech model"
        assert progress[-1]["done"] == 3 and progress[-1]["unit"] == "frames"

    def test_nothing_is_written_to_disk(self, client, home):
        before = sorted(p for p in home.rglob("*"))
        assert _post(client).status_code == 200
        assert sorted(p for p in home.rglob("*")) == before

    def test_a_client_that_goes_away_cancels_the_synthesis(self, home, monkeypatch):
        from localm.plugins.builtin.tts import speech_route
        from localm.plugins.gui.jobs import JobManager
        started, seen_cancel = threading.Event(), threading.Event()

        def slow(model, text, *, on_progress=None, should_cancel=None, **kw):
            started.set()
            for _ in range(500):
                if should_cancel():
                    seen_cancel.set()
                    raise g.SpeechCancelled()
                threading.Event().wait(0.01)
            raise AssertionError("never cancelled")

        monkeypatch.setattr(speech, "synthesize", slow)

        async def gone():
            return True
        monkeypatch.setattr(speech_route, "_resolve_disconnect_poll", lambda request: gone)
        monkeypatch.setattr(speech_route, "_JOB_POLL_SECONDS", 0.05)
        params = speech_route.SpeechParams(None, "hi", "default", "wav", None, None, None)
        model = speech.SpeechModel("m", "m.gguf", "p.gguf")

        async def run():
            jobs = JobManager()
            job = jobs.start_fn("speak", speech_route._make_speak(model, params, None, {}))
            started.wait(5)
            await speech_route._await_job(job, request=None)

        with pytest.raises(asyncio.CancelledError):
            asyncio.run(run())
        assert seen_cancel.is_set()


class TestKernelOriginGate:
    """On the real kernel app in open mode the OpenAI SDK (no Origin, any bearer)
    and a local app are served; a non-local page is refused before any work."""

    @pytest.fixture
    def kernel_client(self, home, synth):
        _register(home, "voice-1")
        with TestClient(_tts_app(home, kernel=True)) as c:
            yield c

    def test_sdk_style_request_is_served(self, kernel_client):
        r = _post(kernel_client, headers={"Authorization": "Bearer sk-anything"})
        assert r.status_code == 200, r.text

    def test_a_cross_origin_local_app_is_served(self, kernel_client):
        r = _post(kernel_client, headers={"Origin": "http://localhost:9999"})
        assert r.status_code == 200, r.text

    def test_a_non_local_origin_is_refused_before_any_synthesis(self, kernel_client, synth):
        r = kernel_client.post(URL, data={"input": "hi"},
                               headers={"Origin": "https://evil.example"})
        assert r.status_code == 403 and synth.calls == []
