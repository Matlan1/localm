# SPDX-License-Identifier: AGPL-3.0-or-later
"""Voice plugin: Whisper speech-to-text for the chat mic button.

Routes (mounted by the engine, auto-scoped to the ``voice`` capability):
  GET  /api/voice/status          - is STT usable / is the model cached / why not
  POST /api/voice/transcribe      - transcribe a base64 audio blob
  POST /api/voice/model/download  - one-time, non-persistent model download

Audio is decoded in memory and never written to disk, so privacy mode stays
trace-free.
"""

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
    """STT availability for the mic button, with the honest reason when it is
    not usable - including a model download blocked by the network policy
    (net_mode), never just "unavailable".

    ``can_download`` tells the GUI whether to offer the one-time "download the
    model now" action: the model is missing, the faster-whisper package is
    there (without it the download would produce a model nothing can load yet,
    and the reason says to install the extra first), net_mode is not "off" (off
    has no bypass), and the caller could authorize it - open mode, or a key
    granting config:write, the same scope that governs net_mode itself. The
    flag only drives UI; POST /api/voice/model/download re-checks all of it
    server-side."""
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
    """One-time download of the CONFIGURED Whisper model, for when the network
    policy does not download it automatically (net_mode=ask).

    This is an explicit consent for exactly ONE download: the authorization is
    a call argument consumed by the job (``prefetch_stt_model(allow_download=
    True)``), nothing is written to config, and net_mode stays exactly as
    configured for every other network path. Gated on config:write - the same
    scope that could change net_mode itself - so a key that could not lift the
    policy cannot bypass it here either; open mode is the trusted local owner.
    net_mode=off always refuses: off is the kill switch, and only a real
    config change lifts it."""
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
    """Map the structured failure class to a status code. 501 = the capability
    is not installed; 409 = the model download is blocked by the network policy
    (a state conflict another request - the download action, a config change -
    can resolve, not a defect in this recording); everything else is a 422 on
    this input."""
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
    """Prefetch the configured Whisper model when the plugin is installed, so a
    first click on the mic does not stall on a surprise download.

    Installing the plugin is itself an explicit user action, so the prefetch
    passes ``allow_download=True`` for this one download and nothing is
    persisted. net_mode=off still refuses, and the install succeeds either way,
    with /api/voice/status carrying the honest reason until the model is
    fetched.

    Runs on a background thread: the install route invokes this hook on the
    server's event loop, and the pip extras may still be installing. The
    prefetch needs only huggingface_hub (a core dependency), not
    faster-whisper."""
    import threading

    def _run() -> None:
        try:
            from localm.voice import prefetch_stt_model
            prefetch_stt_model(allow_download=True)   # logs its own outcome
        except Exception as e:
            # The engine's hook wrapper cannot see this thread, so the failure is
            # logged here.
            from localm.debuglog import logger
            logger.warning("voice: Whisper model prefetch failed: %s", e)

    from localm.voice import PREFETCH_THREAD_NAME
    threading.Thread(target=_run, name=PREFETCH_THREAD_NAME,
                     daemon=True).start()


def unregister() -> None:
    pass
