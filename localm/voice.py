"""
Speech-to-text for the GUI chat: Whisper via faster-whisper (CPU int8).

faster-whisper was chosen over openai-whisper/torch because it runs well on
CPU with int8 quantization and ships as plain wheels — no torch required, so
it works on the GGUF-only base install. It is still optional:

    pip install "localm[voice]"

The model (config ``voice_stt_model``, default "base") is downloaded from
HuggingFace into faster-whisper's cache on FIRST use — that one download is
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
    """(available, reason) — lets the GUI grey out the mic button up front
    instead of letting the user record and only then failing."""
    try:
        import faster_whisper  # noqa: F401
        return True, ""
    except ImportError:
        return False, (
            "Speech-to-text needs the faster-whisper package. Install it "
            "with: pip install \"localm[voice]\"  (then restart the server)")


def transcribe_bytes(data: bytes, language: Optional[str] = None) -> str:
    """Transcribe an audio blob (webm/ogg/wav/mp3 — anything PyAV decodes).
    Loads and caches the Whisper model on first call."""
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise VoiceError(
            "Speech-to-text needs the faster-whisper package. Install it "
            "with: pip install \"localm[voice]\"")

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
                    "use downloads it from HuggingFace — check the network, "
                    "or set a different model: localm config voice_stt_model tiny")
        model = _model

    try:
        segments, _info = model.transcribe(io.BytesIO(data), language=lang)
        text = " ".join(s.text.strip() for s in segments).strip()
    except Exception as e:
        raise VoiceError(f"Transcription failed: {e}")
    if not text:
        raise VoiceError("No speech detected in the recording")
    return text
