"""Voice plugin: Whisper speech-to-text for the chat mic button.

Routes (mounted by the engine, auto-scoped to the ``voice`` capability):
  GET  /api/voice/status      - is STT usable / is the model cached
  POST /api/voice/transcribe  - transcribe a base64 audio blob

Audio is decoded in memory and never written to disk, so privacy mode stays
trace-free.
"""

from __future__ import annotations

import asyncio
import base64

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

_router = APIRouter()


class TranscribeRequest(BaseModel):
    audio_b64: str
    language: str | None = None


@_router.get("/api/voice/status")
async def voice_status():
    from localm.voice import stt_available, stt_model_cached
    ok, reason = stt_available()
    cached, model_name = stt_model_cached() if ok else (False, "")
    return {"available": ok, "reason": reason,
            "model_cached": cached, "model": model_name}


@_router.post("/api/voice/transcribe")
async def voice_transcribe(req: TranscribeRequest):
    from localm.voice import VoiceError, transcribe_bytes
    try:
        data = base64.b64decode(req.audio_b64, validate=True)
    except Exception:
        raise HTTPException(400, "audio_b64 is not valid base64")
    if len(data) > 25_000_000:
        raise HTTPException(413, "Recording too large (max 25 MB)")
    loop = asyncio.get_running_loop()
    try:
        text = await loop.run_in_executor(
            None, lambda: transcribe_bytes(data, language=req.language))
    except VoiceError as e:
        status = 501 if "faster-whisper" in str(e) else 422
        raise HTTPException(status, str(e))
    return {"text": text}


def register(host) -> None:
    host.mount_router(_router)


def unregister() -> None:
    pass
