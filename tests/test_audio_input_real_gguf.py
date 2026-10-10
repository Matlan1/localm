# SPDX-License-Identifier: AGPL-3.0-or-later
"""REAL end-to-end test of audio input: Qwen3-ASR-0.6B (Q8_0, about 805 MB, and
its audio projector, about 214 MB) through GgufBackend and its isolated worker,
CPU only.

@integration so the default `pytest -m "not integration"` skips it. The model
and projector come from LOCALM_TEST_AUDIO_MODEL and LOCALM_TEST_AUDIO_MMPROJ
when set, else from the Hub. The spoken clip comes from LOCALM_TEST_AUDIO_WAV
(with its words in LOCALM_TEST_AUDIO_PHRASE), else it is synthesised with the
Windows speech synthesizer; elsewhere the test skips without a clip.
"""

from __future__ import annotations

import base64
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tests._real_gguf import fetch_gguf, require_native_runtime

pytestmark = [pytest.mark.integration, pytest.mark.real_gguf]

_REPO = "ggml-org/Qwen3-ASR-0.6B-GGUF"
_MODEL = "Qwen3-ASR-0.6B-Q8_0.gguf"
_MMPROJ = "mmproj-Qwen3-ASR-0.6B-Q8_0.gguf"
_PHRASE = "the quick brown fox jumps over the lazy dog"
_GREEDY = dict(max_tokens=64, temperature=0.0, repeat_penalty=1.0, seed=1)


@pytest.fixture(scope="module")
def files():
    require_native_runtime()
    model = os.environ.get("LOCALM_TEST_AUDIO_MODEL") or fetch_gguf(_REPO, _MODEL)
    mmproj = os.environ.get("LOCALM_TEST_AUDIO_MMPROJ") or fetch_gguf(_REPO, _MMPROJ)
    return model, mmproj


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    given = os.environ.get("LOCALM_TEST_AUDIO_WAV")
    if given:
        return Path(given).read_bytes(), os.environ.get("LOCALM_TEST_AUDIO_PHRASE", _PHRASE)
    if sys.platform != "win32" or not shutil.which("powershell"):
        pytest.skip("no LOCALM_TEST_AUDIO_WAV and no Windows speech synthesizer")
    out = tmp_path_factory.mktemp("speech") / "fox.wav"
    script = (
        "Add-Type -AssemblyName System.Speech;"
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer;"
        f"$s.SetOutputToWaveFile('{out}');"
        f"$s.Speak('{_PHRASE}.');$s.Dispose()")
    subprocess.run(["powershell", "-NoProfile", "-Command", script], check=True,
                   timeout=120)
    return out.read_bytes(), _PHRASE


@pytest.fixture(scope="module")
def backend(files):
    from localm.inference.backends.gguf import GgufBackend
    model, mmproj = files
    saved = {k: os.environ.get(k) for k in ("LOCALM_MTMD_CPU", "HIP_VISIBLE_DEVICES",
                                            "CUDA_VISIBLE_DEVICES")}
    os.environ.update(LOCALM_MTMD_CPU="1", HIP_VISIBLE_DEVICES="-1",
                      CUDA_VISIBLE_DEVICES="-1")
    be = GgufBackend(model, mmproj_path=mmproj, n_ctx=4096, n_gpu_layers=0)
    try:
        be.load()
        yield be
    finally:
        be.unload()
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _audio_message(wav: bytes, text: str = "Transcribe this audio."):
    return [{"role": "user", "content": [
        {"type": "input_audio",
         "input_audio": {"data": base64.b64encode(wav).decode(), "format": "wav"}},
        {"type": "text", "text": text}]}]


def _words(text: str) -> str:
    return " ".join("".join(c for c in text.lower() if c.isalnum() or c == " ").split())


def test_the_projector_is_audio_only(files):
    from localm.model_manager.gguf import gguf_mmproj_modalities
    assert gguf_mmproj_modalities(Path(files[1])) == {"vision": False, "audio": True}


def test_the_loaded_model_takes_audio_and_not_images(backend):
    assert backend.supports_audio is True
    assert backend.supports_images is False


def test_the_clip_is_transcribed(backend, clip):
    wav, phrase = clip
    reply = "".join(backend.chat_stream(_audio_message(wav), **_GREEDY))
    assert _words(phrase) in _words(reply), reply


def test_a_bad_clip_is_refused_and_the_worker_keeps_serving(backend, clip):
    from localm.inference.backends.base import UnsupportedInputError
    with pytest.raises(UnsupportedInputError, match="WAV"):
        "".join(backend.chat_stream(_audio_message(b"RIFF....WAVEjunk"), **_GREEDY))
    wav, phrase = clip
    reply = "".join(backend.chat_stream(_audio_message(wav), **_GREEDY))
    assert _words(phrase) in _words(reply), reply


def test_an_image_is_refused(backend):
    from localm.inference.backends.base import UnsupportedInputError
    msg = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}}]}]
    with pytest.raises(UnsupportedInputError):
        next(backend.chat_stream(msg))
