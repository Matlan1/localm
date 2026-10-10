# SPDX-License-Identifier: AGPL-3.0-or-later
"""A real Qwen3-TTS GGUF and its mmproj through the real isolated speech worker.

Set LOCALM_TEST_TTS_MODEL and LOCALM_TEST_TTS_MMPROJ (for example
Qwen3-TTS-12Hz-1.7B-Base-Q4_K_M.gguf and mmproj-Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf
from ggml-org/Qwen3-TTS-12Hz-1.7B-Base-GGUF) to run it; it is skipped otherwise
and never downloads. LOCALM_TEST_TTS_REFERENCE_WAV, when set, names the WAV
llama.cpp's own ``llama-tts -p "Hello world" -s 1234`` wrote for the same files,
and the output must match it byte for byte. The worker runs on the CPU.
"""

import hashlib
import os
import time
import wave
from io import BytesIO

import pytest

from localm.inference import speech
from localm.inference.backends.llamacpp.mtmd_gen import SpeechCancelled

MODEL = os.environ.get("LOCALM_TEST_TTS_MODEL")
MMPROJ = os.environ.get("LOCALM_TEST_TTS_MMPROJ")
REFERENCE = os.environ.get("LOCALM_TEST_TTS_REFERENCE_WAV")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.real_gguf,
    pytest.mark.skipif(not (MODEL and MMPROJ),
                       reason="set LOCALM_TEST_TTS_MODEL and LOCALM_TEST_TTS_MMPROJ"),
]


@pytest.fixture
def model(monkeypatch):
    for var in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        monkeypatch.setenv(var, "-1")
    monkeypatch.setattr(speech, "_choose_gpu_layers", lambda m, n_ctx: (0, None))
    speech.reset_speech()
    yield speech.SpeechModel("tts", MODEL, MMPROJ)
    speech.reset_speech()


def _info(wav_bytes):
    with wave.open(BytesIO(wav_bytes)) as w:
        return w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()


def test_hello_world_is_a_24_khz_mono_wav_with_audio(model):
    out = speech.synthesize(model, "Hello world", seed=1234)
    channels, width, rate, frames = _info(out.wav)
    assert (channels, width, rate) == (1, 2, 24000)
    assert frames == out.n_samples == out.frames * 1920 and out.frames > 5


@pytest.mark.skipif(not REFERENCE, reason="set LOCALM_TEST_TTS_REFERENCE_WAV")
def test_the_output_matches_llama_tts_byte_for_byte(model):
    out = speech.synthesize(model, "Hello world", seed=1234)
    with open(REFERENCE, "rb") as f:
        want = f.read()
    assert hashlib.sha256(out.wav).hexdigest() == hashlib.sha256(want).hexdigest()


def test_the_same_seed_gives_the_same_audio(model):
    a = speech.synthesize(model, "Good morning.", seed=99)
    b = speech.synthesize(model, "Good morning.", seed=99)
    assert a.wav == b.wav


def test_a_cancel_stops_it_and_the_worker_serves_the_next_request(model):
    seen = []
    t0 = time.monotonic()
    with pytest.raises(SpeechCancelled):
        speech.synthesize(model, "This sentence is long enough to need many frames of audio "
                          "before it is finished, so a cancel lands in the middle of it.",
                          seed=5, on_progress=seen.append,
                          should_cancel=lambda: any(e.get("frames", 0) >= 3 for e in seen))
    assert time.monotonic() - t0 < 120
    out = speech.synthesize(model, "Hello world", seed=1234)
    assert out.frames > 5
