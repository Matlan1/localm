# SPDX-License-Identifier: AGPL-3.0-or-later
"""REAL end-to-end test of POST /v1/audio/transcriptions answered by a GGUF model
that hears audio: Qwen3-ASR-0.6B (Q8_0) and its audio projector through the real
Engine and its isolated worker, CPU only, behind the real voice plugin route.

@integration so the default `pytest -m "not integration"` skips it. Model, projector
and spoken clip come from the same sources as tests/test_audio_input_real_gguf.py.
"""

from __future__ import annotations

import os
import sys
import threading

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import localm.inference.http_server as hs
from tests import test_audio_input_real_gguf as _real

pytestmark = [pytest.mark.integration, pytest.mark.real_gguf]

files = _real.files
clip = _real.clip
_words = _real._words

URL = "/v1/audio/transcriptions"
NAME = "qwen3-asr-real"


@pytest.fixture(scope="module")
def engine(files):
    from localm.inference.engine import Engine
    model, mmproj = files
    saved = {k: os.environ.get(k) for k in ("LOCALM_MTMD_CPU", "HIP_VISIBLE_DEVICES",
                                            "CUDA_VISIBLE_DEVICES")}
    os.environ.update(LOCALM_MTMD_CPU="1", HIP_VISIBLE_DEVICES="-1",
                      CUDA_VISIBLE_DEVICES="-1")
    eng = Engine(model, mmproj_path=mmproj, n_ctx=4096, n_gpu_layers=0,
                 display_name=NAME)
    try:
        eng.load()
        yield eng
    finally:
        eng.unload()
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@pytest.fixture
def client(engine, tmp_path, monkeypatch):
    import localm.config as cfg
    import localm.voice as voice
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path))
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.delenv("LOCALM_REQUIRE_AUTH", raising=False)
    monkeypatch.setattr(cfg, "HOME_DIR", tmp_path)
    monkeypatch.setattr(cfg, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", tmp_path / "registry.json")
    monkeypatch.setattr(voice, "prefetch_stt_model",
                        lambda allow_download=None: (False, "stubbed in tests"))
    from localm.plugins.engine import PluginManager
    app = FastAPI()
    PluginManager(app, external_root=tmp_path / "noplugins").install("voice")
    for t in threading.enumerate():
        if t.name == voice.PREFETCH_THREAD_NAME:
            t.join(timeout=10)
    plugin = sys.modules["_localm_plugin_voice.transcriptions"]
    audio = sys.modules["_localm_plugin_voice.audio_model"]
    monkeypatch.setattr(plugin, "_whisper_installed", lambda: False)
    monkeypatch.setattr(audio, "audio_model_names", lambda: [NAME])
    asked = []

    async def get_engine(name, **kw):
        asked.append(name)
        return engine

    monkeypatch.setattr(hs, "get_engine", get_engine)
    with TestClient(app) as c:
        c.asked = asked
        yield c


def test_the_clip_is_transcribed_as_json(client, clip):
    wav, phrase = clip
    r = client.post(URL, files={"file": ("fox.wav", wav, "audio/wav")})
    assert r.status_code == 200, r.text
    assert client.asked == [NAME]
    assert r.headers["X-Localm-Transcription-Model"] == NAME
    assert "<asr_text>" not in r.json()["text"] and "language" not in r.json()["text"]
    assert _words(r.json()["text"]) == _words(phrase), r.json()


def test_the_clip_is_transcribed_as_text(client, clip):
    wav, phrase = clip
    r = client.post(URL, data={"response_format": "text", "model": "whisper-1"},
                    files={"file": ("fox.wav", wav, "audio/wav")})
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("text/plain")
    assert "<asr_text>" not in r.text
    assert _words(r.text) == _words(phrase), r.text


def test_a_bad_clip_is_400_and_the_model_keeps_serving(client, clip):
    r = client.post(URL, files={"file": ("x.wav", b"RIFF....WAVEjunk", "audio/wav")})
    assert r.status_code == 400 and "WAV" in r.json()["detail"], r.text
    wav, phrase = clip
    again = client.post(URL, files={"file": ("fox.wav", wav, "audio/wav")})
    assert again.status_code == 200
    assert _words(phrase) in _words(again.json()["text"])


def _flac_from_wav(wav: bytes) -> bytes:
    import io

    av = pytest.importorskip("av")
    out = io.BytesIO()
    with av.open(io.BytesIO(wav)) as src, av.open(out, "w", format="flac") as dst:
        stream_in = src.streams.audio[0]
        stream_out = dst.add_stream("flac", rate=stream_in.rate)
        for frame in src.decode(stream_in):
            for packet in stream_out.encode(frame):
                dst.mux(packet)
        for packet in stream_out.encode(None):
            dst.mux(packet)
    return out.getvalue()


def _repeated(wav: bytes, times: int) -> bytes:
    import io
    import wave

    with wave.open(io.BytesIO(wav)) as src:
        params, frames = src.getparams(), src.readframes(src.getnframes())
    out = io.BytesIO()
    with wave.open(out, "wb") as dst:
        dst.setparams(params)
        dst.writeframes(frames * times)
    return out.getvalue()


def test_a_flac_clip_is_transcribed(client, clip):
    wav, phrase = clip
    flac = _flac_from_wav(wav)
    assert flac[:4] == b"fLaC"
    r = client.post(URL, files={"file": ("fox.flac", flac, "audio/flac")})
    assert r.status_code == 200, r.text
    assert _words(r.json()["text"]) == _words(phrase), r.json()


def test_a_long_clip_is_transcribed_whole(client, clip):
    wav, phrase = clip
    r = client.post(URL, files={"file": ("fox.wav", _repeated(wav, 6), "audio/wav")})
    assert r.status_code == 200, r.text
    assert _words(r.json()["text"]).count(_words(phrase)) >= 3, r.json()
