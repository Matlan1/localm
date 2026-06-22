# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Speech-to-text for the GUI chat: Whisper via faster-whisper (CPU int8).

faster-whisper was chosen over openai-whisper/torch because it runs well on
CPU with int8 quantization and ships as plain wheels - no torch required, so
it works on the GGUF-only base install. It is still optional:

    pip install "localm[voice]"

The model (config ``voice_stt_model``, default "base") is downloaded from
HuggingFace into faster-whisper's cache on FIRST use - that one download is
the only network access; transcription itself is fully local and offline.

Text-to-speech needs no backend at all: the GUI uses the browser's built-in
speechSynthesis (Windows SAPI voices), which is offline by construction.
"""

from __future__ import annotations

import io
import threading
from typing import Optional

_lock = threading.Lock()
_model = None
_model_name: Optional[str] = None


class VoiceError(Exception):
    """Transcription failed; the message says why and what to install."""


def stt_available() -> tuple[bool, str]:
    """(available, reason) - lets the GUI grey out the mic button up front
    instead of letting the user record and only then failing."""
    try:
        import faster_whisper  # noqa: F401
        return True, ""
    except ImportError:
        return False, (
            "Speech-to-text needs the faster-whisper package. Install it "
            "with: pip install \"localm[voice]\"  (then restart the server)")


def stt_model_cached() -> tuple[bool, str]:
    """(cached, model_name) - is the configured Whisper model already in the
    local HuggingFace cache? First use otherwise downloads it; the GUI asks
    for consent before triggering that one network access."""
    from pathlib import Path

    from localm.config import load_config
    name = str(load_config().get("voice_stt_model", "base"))
    if Path(name).expanduser().is_dir():
        return True, name                       # local model directory
    repo = name if "/" in name else f"Systran/faster-whisper-{name}"
    try:
        from faster_whisper.utils import _MODELS   # name → repo mapping
        repo = _MODELS.get(name, repo)
    except Exception:
        pass
    try:
        from huggingface_hub import try_to_load_from_cache
        hit = try_to_load_from_cache(repo, "model.bin")
        return isinstance(hit, str), name
    except Exception:
        return False, name


def transcribe_bytes(data: bytes, language: Optional[str] = None) -> str:
    """Transcribe an audio blob (webm/ogg/wav/mp3 - anything PyAV decodes).
    Loads and caches the Whisper model on first call."""
    # SRV-1: reject an empty recording up front, before importing or touching the
    # native stack, so a 0-byte blob (a mic that captured nothing) is a clean
    # client error rather than feeding nothing into the decoder.
    if not data:
        raise VoiceError("Empty recording (no audio was captured).")

    try:
        from faster_whisper import WhisperModel
        from faster_whisper.audio import decode_audio
    except ImportError:
        raise VoiceError(
            "Speech-to-text needs the faster-whisper package. Install it "
            "with: pip install \"localm[voice]\"")

    # SRV-1: decode the container to PCM in Python FIRST, before the native
    # transcription path. PyAV raises an ordinary Python exception on an empty,
    # truncated or non-audio blob, whereas handing raw bytes straight to the
    # model can fault at the C level - a crash no Python except can catch, which
    # takes the whole server down (the reported "server dead" on a bad/silent
    # recording). Decoding here converts those into a clean VoiceError and lets
    # us reject a recording that holds no audio before it reaches the model.
    try:
        audio = decode_audio(io.BytesIO(data))
    except Exception as e:
        raise VoiceError(
            f"Could not decode the recording (corrupt or unsupported audio): {e}")
    if audio is None or len(audio) == 0:
        raise VoiceError("No audio in the recording (it was empty or zero-length).")

    from localm.config import load_config
    cfg = load_config()
    name = str(cfg.get("voice_stt_model", "base"))
    lang = language or cfg.get("voice_stt_language") or None

    global _model, _model_name
    with _lock:
        if _model is None or _model_name != name:
            try:
                _model = WhisperModel(name, device="cpu", compute_type="int8")
                _model_name = name
            except Exception as e:
                _model = None
                raise VoiceError(
                    f"Could not load Whisper model '{name}': {e}. The first "
                    "use downloads it from HuggingFace - check the network, "
                    "or set a different model: localm config voice_stt_model tiny")
        model = _model

    # Pass the already-decoded PCM array (not the raw bytes) so the native path
    # never re-decodes an untrusted container.
    try:
        segments, _info = model.transcribe(audio, language=lang)
        text = " ".join(s.text.strip() for s in segments).strip()
    except Exception as e:
        raise VoiceError(f"Transcription failed: {e}")
    if not text:
        raise VoiceError("No speech detected in the recording")
    return text
