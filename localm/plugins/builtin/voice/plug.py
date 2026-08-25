# SPDX-License-Identifier: AGPL-3.0-or-later
"""Voice plugin: Whisper speech-to-text for the chat mic button."""

from __future__ import annotations

import asyncio
import base64

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from localm.inference.errors import route_errors
from localm.executor import get_plugin_executor
from localm.voice import VoiceError

_router = APIRouter()


class TranscribeRequest(BaseModel):
    audio_b64: str
    language: str | None = None


@_router.get("/api/voice/status")
async def voice_status(request: Request):
    """STT availability for the mic button, with the honest reason when it is not usable - including a model download blocked by the network policy (net_mode), never just 'unavailable'."""
    import importlib.util
    from localm.voice import stt_available, stt_model_cached
    ok, reason = stt_available()
    have_pkg = importlib.util.find_spec("faster_whisper") is not None
    cached, model_name = stt_model_cached() if have_pkg else (False, "")
    can_download = False
    if have_pkg and not cached:
        from localm.netpolicy import network_mode
        if network_mode() != "off":
            import localm.inference.http_server as _hs
            from localm import scopes
            held = _hs.caller_scopes(request)
            can_download = held is None or scopes.grants(held, scopes.CONFIG_WRITE)
    return {"available": ok, "reason": reason,
            "model_cached": cached, "model": model_name,
            "can_download": can_download}


@_router.post("/api/voice/model/download")
async def voice_model_download(request: Request):
    """One-time download of the CONFIGURED Whisper model, for when the network policy does not download it automatically (net_mode=ask)."""
    import localm.inference.http_server as _hs
    from localm import scopes
    held = _hs.caller_scopes(request)
    if held is not None and not scopes.grants(held, scopes.CONFIG_WRITE):
        raise HTTPException(
            403, "Downloading the speech model needs the config:write scope "
                 "(the same permission that governs the network policy).")
    from localm.voice import prefetch_stt_model, stt_model_cached
    cached, name = stt_model_cached()
    if cached:
        return {"status": "already_cached", "model": name}
    from localm.netpolicy import network_mode
    if network_mode() == "off":
        raise HTTPException(
            409, "Network access is disabled (net_mode=off), which blocks even "
                 "an explicitly requested model download. Set net_mode to ask "
                 "or allow first.")
    jobs = getattr(request.app.state, "jobs", None)
    if jobs is None:
        raise HTTPException(503, "This server has no background job registry, "
                                 "so the download cannot be started.")

    def _run(job):
        job.push({"type": "line",
                  "text": f"Downloading Whisper model '{name}' (one-time)..."})
        ok, reason = prefetch_stt_model(allow_download=True)
        if not ok:
            job.push({"type": "line", "text": f"error: {reason}"})
            return False
        job.push({"type": "line",
                  "text": f"Ready: Whisper '{name}' is downloaded - speech-to-"
                          "text now runs fully offline. No settings were "
                          "changed."})
        return True

    from localm.inference.http_server import principal_id
    job = jobs.start_fn("voice-model-download", _run, owner=principal_id(request))
    return {"job_id": job.id, "model": name}


def _voice_error_status(e: VoiceError) -> tuple[int, str]:
    """Branch on the structured failure class, not the human message: rewording a message must never flip a status code. 501 = the capability is not installed; 409 = the model download is blocked by the network policy (a state conflict another request - the download action, a config change - can resolve, n..."""
    code = getattr(e, "code", "")
    if code == "needs-faster-whisper":
        return 501, str(e)
    if code == "download-blocked":
        return 409, str(e)
    return 422, str(e)


@_router.post("/api/voice/transcribe")
@route_errors({
    VoiceError: _voice_error_status,
    Exception: lambda e: (502, f"Transcription failed: {e}"),
})
async def voice_transcribe(req: TranscribeRequest):
    from localm.voice import transcribe_bytes
    try:
        data = base64.b64decode(req.audio_b64, validate=True)
    except Exception:
        raise HTTPException(400, "audio_b64 is not valid base64")
    if not data:
        raise HTTPException(422, "Empty recording (no audio was captured)")
    if len(data) > 25_000_000:
        raise HTTPException(413, "Recording too large (max 25 MB)")
    loop = asyncio.get_running_loop()
    text = await loop.run_in_executor(
        get_plugin_executor(),
        lambda: transcribe_bytes(data, language=req.language))
    return {"text": text}


def register(host) -> None:
    host.mount_router(_router)


def on_install() -> None:
    """Prefetch the configured Whisper model when the plugin is installed, so it does not install fine and then stall on a surprise download the first time someone clicks the mic."""
    import threading

    def _run() -> None:
        try:
            from localm.voice import prefetch_stt_model
            prefetch_stt_model(allow_download=True)   # logs its own outcome
        except Exception as e:
            # The engine's hook wrapper cannot see this thread; never let the
            # outcome vanish (rule 5).
            from localm.debuglog import logger
            logger.warning("voice: Whisper model prefetch failed: %s", e)

    from localm.voice import PREFETCH_THREAD_NAME
    threading.Thread(target=_run, name=PREFETCH_THREAD_NAME,
                     daemon=True).start()


def unregister() -> None:
    pass
