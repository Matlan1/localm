# SPDX-License-Identifier: AGPL-3.0-or-later
"""POST /v1/audio/transcriptions: the OpenAI-compatible transcription upload.

The real route, multipart parsing, parameter validation, formatting and error
mapping run; only the isolated Whisper worker is replaced (it needs a model
download), at ``localm.voice._run_in_worker_detailed``.
"""

import base64
import json
import sys
import threading
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from localm import voice_formats
from localm.voice import VoiceError

URL = "/v1/audio/transcriptions"
GOLDEN = Path(__file__).parent / "fixtures" / "openai_sdk"
GOLDEN_AUDIO = b"RIFF\x00\x01\r\n\xffWAVEdata"
AUDIO = b"RIFF\x24\x00\x00\x00WAVEfmt fake-audio-bytes"


def _seg(i, start, end, text, words=()):
    return {"id": i, "seek": 0, "start": start, "end": end, "text": text,
            "tokens": [1, 2], "temperature": 0.0, "avg_logprob": -0.25,
            "compression_ratio": 1.5, "no_speech_prob": 0.01,
            "words": list(words)}


DETAIL = {
    "language": "en", "duration": 3.5,
    "segments": [
        _seg(0, 0.0, 1.5, "Hello world.",
             [{"word": " Hello", "start": 0.0, "end": 0.5, "probability": 0.9},
              {"word": " world.", "start": 0.6, "end": 1.4, "probability": 0.8}]),
        _seg(1, 2.0, 3.25, "Second line."),
    ],
}
TEXT = "Hello world. Second line."


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
def voice_app(home, monkeypatch):
    import localm.voice as voice
    monkeypatch.setattr(voice, "prefetch_stt_model",
                        lambda allow_download=None: (False, "stubbed in tests"))
    from localm.plugins.engine import PluginManager
    app = FastAPI()
    PluginManager(app, external_root=home / "noplugins").install("voice")
    for t in threading.enumerate():
        if t.name == voice.PREFETCH_THREAD_NAME:
            t.join(timeout=10)
    return app


class _Worker:
    """Stands in for the isolated speech worker and records every dispatch."""

    def __init__(self):
        self.calls = []
        self.text = TEXT
        self.detail = DETAIL
        self.error = None

    def __call__(self, data, name, language, timeout, *, opts=None,
                 local_files_only=True, blocked_reason=None):
        self.calls.append({"data": data, "name": name, "language": language,
                           "opts": opts})
        if self.error is not None:
            raise self.error
        return self.text, dict(self.detail)


@pytest.fixture
def worker(monkeypatch):
    import localm.voice as voice
    w = _Worker()
    monkeypatch.setattr(voice, "_stt_plan", lambda language: {
        "name": "base", "language": language, "timeout": 5.0,
        "local_files_only": True, "blocked_reason": None})
    monkeypatch.setattr(voice, "_run_in_worker_detailed", w)
    return w


@pytest.fixture
def client(voice_app, worker):
    with TestClient(voice_app) as c:
        yield c


def _post(client, data=None, files=None, headers=None):
    files = files if files is not None else {"file": ("a.wav", AUDIO, "audio/wav")}
    return client.post(URL, data=data or {"model": "whisper-1"}, files=files,
                       headers=headers)


class TestResponseFormats:
    def test_default_is_json_text_only(self, client, worker):
        r = _post(client)
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("application/json")
        assert r.json() == {"text": TEXT}
        assert worker.calls[0]["data"] == AUDIO

    def test_text(self, client):
        r = _post(client, {"model": "whisper-1", "response_format": "text"})
        assert r.headers["content-type"].startswith("text/plain")
        assert r.text == TEXT + "\n"

    def test_srt(self, client):
        r = _post(client, {"model": "whisper-1", "response_format": "srt"})
        assert r.headers["content-type"].startswith("text/plain")
        assert r.text == voice_formats.to_srt(DETAIL["segments"])
        assert r.text.startswith("1\n00:00:00,000 --> 00:00:01,500\nHello world.")

    def test_vtt(self, client):
        r = _post(client, {"model": "whisper-1", "response_format": "vtt"})
        assert r.headers["content-type"].startswith("text/plain")
        assert r.text == voice_formats.to_vtt(DETAIL["segments"])
        assert r.text.startswith("WEBVTT\n")

    def test_verbose_json_defaults_to_segments(self, client):
        r = _post(client, {"model": "whisper-1", "response_format": "verbose_json"})
        body = r.json()
        assert body["task"] == "transcribe" and body["language"] == "english"
        assert body["duration"] == 3.5 and body["text"] == TEXT
        assert len(body["segments"]) == 2 and "words" not in body

    def test_verbose_json_word_granularity_adds_words_and_drops_segments(
            self, client, worker):
        r = _post(client, {"model": "whisper-1", "response_format": "verbose_json",
                           "timestamp_granularities[]": ["word"]})
        body = r.json()
        assert [w["word"] for w in body["words"]] == [" Hello", " world."]
        assert "segments" not in body
        assert worker.calls[0]["opts"]["word_timestamps"] is True

    def test_verbose_json_both_granularities(self, client, worker):
        r = _post(client, {"model": "whisper-1", "response_format": "verbose_json",
                           "timestamp_granularities[]": ["word", "segment"]})
        body = r.json()
        assert len(body["words"]) == 2 and len(body["segments"]) == 2

    def test_silence_is_empty_text_not_an_error(self, client, worker):
        worker.text, worker.detail = "", {**DETAIL, "segments": []}
        assert _post(client).json() == {"text": ""}
        assert _post(client, {"model": "whisper-1",
                              "response_format": "srt"}).text == ""


class TestParametersReachTheWorker:
    def test_language_prompt_temperature_are_forwarded(self, client, worker):
        r = _post(client, {"model": "whisper-1", "language": "FR",
                           "prompt": " glossary: localm ", "temperature": "0.2"})
        assert r.status_code == 200
        call = worker.calls[0]
        assert call["language"] == "fr"
        assert call["opts"] == {"prompt": "glossary: localm", "temperature": 0.2,
                                "word_timestamps": False}

    def test_unset_parameters_are_none(self, client, worker):
        _post(client)
        call = worker.calls[0]
        assert call["language"] is None
        assert call["opts"] == {"prompt": None, "temperature": None,
                                "word_timestamps": False}

    def test_model_is_optional(self, client, worker):
        assert _post(client, {}).status_code == 200


class TestModelParameter:
    def test_openai_alias_and_configured_name_are_served(self, client, home):
        (home / "config.json").write_text('{"voice_stt_model": "small"}',
                                          encoding="utf-8")
        for name in ("whisper-1", "small", "SMALL", "Systran/faster-whisper-small"):
            assert _post(client, {"model": name}).status_code == 200, name

    def test_another_model_name_is_a_404_that_names_what_is_served(
            self, client, worker):
        r = _post(client, {"model": "whisper-large-v3"})
        assert r.status_code == 404
        assert "whisper-large-v3" in r.json()["detail"]
        assert "'base'" in r.json()["detail"]
        assert worker.calls == []


class TestRefusals:
    def test_not_multipart_is_415(self, client, worker):
        r = client.post(URL, json={"model": "whisper-1"})
        assert r.status_code == 415
        assert worker.calls == []

    def test_missing_file_is_400(self, client, worker):
        r = client.post(URL, data={"model": "whisper-1"},
                        files={"unrelated": ("x.txt", b"x", "text/plain")})
        assert r.status_code == 400 and "'file'" in r.json()["detail"]
        assert worker.calls == []

    def test_two_files_are_400(self, client, worker):
        r = client.post(URL, data={"model": "whisper-1"}, files=[
            ("file", ("a.wav", AUDIO, "audio/wav")),
            ("file", ("b.wav", AUDIO, "audio/wav"))])
        assert r.status_code == 400
        assert worker.calls == []

    def test_empty_file_is_400(self, client, worker):
        r = _post(client, files={"file": ("a.wav", b"", "audio/wav")})
        assert r.status_code == 400
        assert worker.calls == []

    def test_file_over_the_cap_is_413_and_never_reaches_the_worker(
            self, client, worker, monkeypatch):
        mod = sys.modules["_localm_plugin_voice.transcriptions"]
        monkeypatch.setattr(mod, "MAX_AUDIO_BYTES", 1000)
        r = _post(client, files={"file": ("a.wav", b"x" * 1001, "audio/wav")})
        assert r.status_code == 413
        assert "Audio file too large" in r.json()["detail"]
        assert worker.calls == []
        ok = _post(client, files={"file": ("a.wav", b"x" * 1000, "audio/wav")})
        assert ok.status_code == 200

    def test_body_over_the_cap_is_refused_before_it_is_buffered(
            self, client, worker, monkeypatch):
        mod = sys.modules["_localm_plugin_voice.transcriptions"]
        monkeypatch.setattr(mod, "MAX_AUDIO_BYTES", 1000)
        monkeypatch.setattr(mod, "FORM_OVERHEAD_BYTES", 100)
        r = _post(client, files={"file": ("a.wav", b"x" * 5000, "audio/wav")})
        assert r.status_code == 413
        assert "Audio file too large" in r.json()["detail"]
        assert worker.calls == []

    def test_the_real_cap_is_25_mb_and_the_refusal_says_so(self, client, worker):
        over = 25 * 1024 * 1024 + 1
        r = _post(client, files={"file": ("a.wav", b"\0" * over, "audio/wav")})
        assert r.status_code == 413
        assert r.json()["detail"] == "Audio file too large (max 25 MB)."
        assert worker.calls == []

    @pytest.mark.parametrize("fields", [
        {"response_format": "mp3"},
        {"response_format": "verbose"},
        {"language": "english"},
        {"language": "e"},
        {"temperature": "abc"},
        {"temperature": "nan"},
        {"temperature": "inf"},
        {"temperature": "1.5"},
        {"temperature": "-0.1"},
        {"timestamp_granularities[]": ["word"]},
        {"response_format": "verbose_json", "timestamp_granularities[]": ["char"]},
        {"stream": "true"},
        {"include[]": ["logprobs"]},
        {"chunking_strategy": "auto"},
        {"known_speaker_names[]": ["a"]},
        {"prompt": "p" * 9000},
    ])
    def test_bad_or_unsupported_field_is_400(self, client, worker, fields):
        r = _post(client, {"model": "whisper-1", **fields})
        assert r.status_code == 400, r.text
        assert worker.calls == []

    def test_stream_false_is_accepted(self, client):
        assert _post(client, {"model": "whisper-1", "stream": "false"}).status_code == 200

    @pytest.mark.parametrize("value", ["0", "1", "0.0", "1.0", "0.7"])
    def test_temperature_range_edges_are_accepted(self, client, value):
        assert _post(client, {"model": "whisper-1",
                              "temperature": value}).status_code == 200


class TestVoiceErrorMapping:
    @pytest.mark.parametrize("code, status", [
        ("bad-request", 400), ("decode", 400), ("empty", 400),
        ("needs-faster-whisper", 501), ("download-blocked", 409),
        ("spawn", 503), ("load", 503), ("timeout", 504),
        ("decoder-fault", 500), ("crash", 502), ("transcribe", 502),
        ("something-new", 502)])
    def test_status_by_class(self, client, worker, code, status):
        worker.error = VoiceError("boom detail", code=code)
        r = _post(client)
        assert r.status_code == status
        assert "boom detail" in r.json()["detail"]

    def test_unexpected_exception_is_a_502(self, client, worker):
        worker.error = RuntimeError("engine blew up")
        r = _post(client)
        assert r.status_code == 502 and "engine blew up" in r.json()["detail"]


class TestUploadPathSafety:
    HOSTILE = ["../../../evil.wav", "..\\..\\evil.wav", "C:\\Windows\\evil.wav",
               "/etc/passwd", "nul\x00.wav", "con", "a" * 400 + ".wav",
               "x.wav:stream"]

    @pytest.mark.parametrize("name", HOSTILE)
    def test_file_name_never_touches_the_filesystem(
            self, client, worker, home, tmp_path_factory, monkeypatch, name):
        sandbox = tmp_path_factory.mktemp("cwd")
        monkeypatch.chdir(sandbox)
        before = sorted(p for p in home.rglob("*"))
        r = client.post(URL, data={"model": "whisper-1"}, files={
            "file": (name, AUDIO, "audio/wav")})
        assert r.status_code == 200
        assert worker.calls[-1]["data"] == AUDIO
        assert sorted(p for p in home.rglob("*")) == before
        assert list(sandbox.iterdir()) == []
        assert not Path("evil.wav").exists()

    def test_audio_is_not_persisted_in_privacy_mode(
            self, client, worker, home, monkeypatch):
        from localm import audit
        monkeypatch.setattr(audit, "effective_mode",
                            lambda *a, **k: audit.SessionMode.PRIVACY)
        before = sorted(p for p in home.rglob("*"))
        assert _post(client).status_code == 200
        assert sorted(p for p in home.rglob("*")) == before


class TestAuth:
    @pytest.fixture
    def keys(self, voice_app):
        from localm import auth
        return {
            "voice": auth.create_key("v", ["voice"])["key"],
            "chat": auth.create_key("c", ["chat"])["key"],
            "image": auth.create_key("i", ["image"])["key"],
            "admin": auth.create_key("a", ["admin"], allow_privileged=True)["key"],
        }

    @staticmethod
    def _bearer(key):
        return {"Authorization": f"Bearer {key}"}

    def test_no_key_is_401_and_the_upload_is_not_read(
            self, client, worker, keys):
        r = _post(client)
        assert r.status_code == 401
        assert worker.calls == []

    def test_garbage_key_is_401(self, client, worker, keys):
        assert _post(client, headers=self._bearer("not-a-key")).status_code == 401

    @pytest.mark.parametrize("who", ["chat", "image"])
    def test_a_key_without_the_voice_scope_is_403(self, client, worker, keys, who):
        r = _post(client, headers=self._bearer(keys[who]))
        assert r.status_code == 403
        assert "voice" in r.json()["detail"]
        assert worker.calls == []

    @pytest.mark.parametrize("who", ["voice", "admin"])
    def test_voice_and_admin_keys_are_served(self, client, worker, keys, who):
        r = _post(client, headers=self._bearer(keys[who]))
        assert r.status_code == 200 and r.json() == {"text": TEXT}
        assert len(worker.calls) == 1

    def test_open_mode_serves_without_a_key(self, client, worker):
        assert _post(client).status_code == 200


class TestRealSdkWireFormat:
    """The bytes the official openai SDK itself sent (captured with a mock
    transport; see tests/fixtures/openai_sdk), replayed against the route."""

    def _golden(self):
        rec = json.loads((GOLDEN / "transcription_all_params.json").read_text(
            encoding="utf-8"))
        return "/v1" + rec["path"], rec["content_type"], base64.b64decode(rec["body_b64"])

    def test_every_field_the_sdk_sent_is_understood(self, client, worker):
        path, ctype, body = self._golden()
        assert path == URL and ctype.startswith("multipart/form-data; boundary=")
        r = client.post(path, content=body, headers={"content-type": ctype})
        assert r.status_code == 200, r.text
        out = r.json()
        assert out["task"] == "transcribe" and len(out["segments"]) == 2
        assert [w["word"] for w in out["words"]] == [" Hello", " world."]
        call = worker.calls[0]
        assert call["data"] == GOLDEN_AUDIO
        assert call["language"] == "en"
        assert call["opts"] == {"prompt": "glossary", "temperature": 0.2,
                                "word_timestamps": True}


def test_openapi_describes_the_multipart_request_body(voice_app):
    schema = voice_app.openapi()
    body = schema["paths"][URL]["post"]["requestBody"]["content"]["multipart/form-data"]["schema"]
    assert body["required"] == ["file"]
    assert body["properties"]["response_format"]["enum"] == list(voice_formats.RESPONSE_FORMATS)


class TestKernelOriginGate:
    """On the real kernel app, in open mode, the OpenAI SDK (no Origin header,
    an arbitrary bearer) and a cross-origin local app must both get past the
    origin/shell-token gate; a sibling plugin route must still be refused."""

    @pytest.fixture
    def kernel_client(self, home, monkeypatch, worker):
        import localm.voice as voice
        monkeypatch.setattr(voice, "prefetch_stt_model",
                            lambda allow_download=None: (False, "stubbed in tests"))
        from localm.inference.http_server import create_app
        from localm.plugins.engine import PluginManager
        app = create_app(None)
        PluginManager(app, external_root=home / "noplugins").install("voice")
        for t in threading.enumerate():
            if t.name == voice.PREFETCH_THREAD_NAME:
                t.join(timeout=10)
        with TestClient(app) as c:
            yield c

    def test_sdk_style_request_without_origin_is_served(self, kernel_client, worker):
        r = _post(kernel_client, headers={"Authorization": "Bearer sk-anything"})
        assert r.status_code == 200, r.text

    def test_cross_origin_local_app_is_served(self, kernel_client, worker):
        r = _post(kernel_client, headers={"Origin": "http://localhost:9999"})
        assert r.status_code == 200, r.text

    def test_sibling_plugin_route_is_still_refused_cross_origin(self, kernel_client):
        r = kernel_client.post("/api/voice/transcribe", json={"audio_b64": "AAAA"},
                               headers={"Origin": "http://localhost:9999"})
        assert r.status_code == 403 and "cross-origin" in r.json()["detail"].lower()

    def test_sibling_plugin_route_still_needs_the_shell_token_in_open_mode(
            self, kernel_client):
        r = kernel_client.post("/api/voice/transcribe", json={"audio_b64": "AAAA"})
        assert r.status_code == 403 and "Open-mode management" in r.json()["detail"]

    def test_a_non_local_origin_is_refused_before_the_upload_is_read(
            self, kernel_client, worker):
        r = _post(kernel_client, headers={"Origin": "https://evil.example"})
        assert r.status_code == 403
        assert "Cross-origin" in r.json()["detail"]
        assert worker.calls == []

    @pytest.mark.parametrize("origin", [
        "http://localhost.evil.example", "http://127.0.0.1.evil.example:8642",
        "https://localhost:9999.evil.example", "null"])
    def test_look_alike_origins_are_refused(self, kernel_client, worker, origin):
        assert _post(kernel_client, headers={"Origin": origin}).status_code == 403
        assert worker.calls == []

    @pytest.mark.parametrize("origin", [
        "http://localhost", "https://127.0.0.1:3000", "http://testserver"])
    def test_local_and_same_origin_pages_are_served(self, kernel_client, origin):
        assert _post(kernel_client, headers={"Origin": origin}).status_code == 200

    def test_a_configured_cors_origin_is_served_and_others_are_not(
            self, kernel_client, home):
        (home / "config.json").write_text(
            '{"cors_origins": ["https://app.example"]}', encoding="utf-8")
        assert _post(kernel_client,
                     headers={"Origin": "https://app.example"}).status_code == 200
        assert _post(kernel_client,
                     headers={"Origin": "https://other.example"}).status_code == 403

    def test_a_wildcard_cors_setting_serves_any_origin(self, kernel_client, home):
        (home / "config.json").write_text('{"cors_origins": "*"}', encoding="utf-8")
        assert _post(kernel_client,
                     headers={"Origin": "https://anywhere.example"}).status_code == 200
