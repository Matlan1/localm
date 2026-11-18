# SPDX-License-Identifier: AGPL-3.0-or-later
"""Voice plugin: speech-to-text for the chat mic button (Whisper, or a registered
audio model when Whisper is not available).

Routes (mounted by the engine, auto-scoped to the ``voice`` capability):
  GET  /api/voice/status          - is STT usable / is the model cached / why not
  POST /api/voice/transcribe      - transcribe a base64 audio blob
  POST /api/voice/model/download  - one-time, non-persistent model download
  POST /v1/audio/transcriptions   - OpenAI-compatible transcription upload
                                    (see transcriptions.py)

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
from . import audio_model as _audio_model
from . import transcriptions as _transcriptions

_WHISPER_UNAVAILABLE = ("needs-faster-whisper", "download-blocked")

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
    and the reason says to install the extra first), the network policy allows
    the download (``netpolicy.download_refusal``), and the caller could
    authorize it - open mode, or a key granting config:write, the same scope
    that governs net_mode itself. ``download_blocked`` says why not when the
    model is missing and the package is there, else it is "". When the mic is
    unavailable for that cause, ``reason`` is the same text, except under
    net_mode=off, which keeps its own. When Whisper is not usable but a
    registered model can hear audio, ``available`` is true and ``audio_model``
    names the model the mic route falls back to. The flag only drives UI; POST
    /api/voice/model/download re-checks all of it server-side."""
    import importlib.util
    from localm.voice import stt_available, stt_model_cached
    ok, reason = stt_available()
    have_pkg = importlib.util.find_spec("faster_whisper") is not None
    cached, model_name = stt_model_cached() if have_pkg else (False, "")
    can_download = False
    download_blocked = ""
    if have_pkg and not cached:
        import localm.inference.http_server as _hs
        from localm import scopes
        from localm.netpolicy import download_refusal, network_mode
        from localm.voice import _STT_HUB_ENDPOINT
        refusal = await asyncio.get_running_loop().run_in_executor(
            get_plugin_executor(), download_refusal, _STT_HUB_ENDPOINT)
        if refusal:
            download_blocked = refusal
            # stt_available's net_mode=off text offers no download, so it stays.
            if ok or network_mode() != "off":
                ok, reason = False, refusal
        elif _hs.caller_grants(request, scopes.CONFIG_WRITE):
            can_download = True
        else:
            download_blocked = ("Downloading the speech model needs the config:write "
                                "scope (the same permission that governs the network "
                                "policy).")
            if not ok:
                reason = download_blocked
    audio_model = ""
    if not ok:
        names = await asyncio.get_running_loop().run_in_executor(
            get_plugin_executor(), _audio_model.audio_model_names)
        if names:
            ok, reason, audio_model = True, "", names[0]
    return {"available": ok, "reason": reason,
            "model_cached": cached, "model": model_name,
            "can_download": can_download, "download_blocked": download_blocked,
            "audio_model": audio_model}


@_router.post("/api/voice/model/download")
async def voice_model_download(request: Request):
    """One-time download of the Whisper model configured when this request
    arrives, for when the network policy does not download it automatically
    (net_mode=ask). The job fetches and verifies that model even if
    voice_stt_model is changed while it runs.

    This is an explicit consent for exactly ONE download: the authorization is
    a call argument consumed by the job (``prefetch_stt_model(allow_download=
    True)``), nothing is written to config, and net_mode stays exactly as
    configured for every other network path. Gated on config:write - the same
    scope that could change net_mode itself - so a key that could not lift the
    policy cannot bypass it here either; open mode is the trusted local owner.
    net_mode=off refuses by default, and only a real config change lifts it -
    either net_mode itself, or net_allow_model_downloads exempting explicit
    downloads specifically."""
    import localm.inference.http_server as _hs
    from localm import scopes
    if not _hs.caller_grants(request, scopes.CONFIG_WRITE):
        raise HTTPException(
            403, "Downloading the speech model needs the config:write scope "
                 "(the same permission that governs the network policy).")
    from localm.voice import prefetch_stt_model, stt_model_cached
    cached, name = stt_model_cached()
    if cached:
        return {"status": "already_cached", "model": name}
    from localm.netpolicy import downloads_allowed_when_off, network_mode
    if network_mode() == "off" and not downloads_allowed_when_off():
        from localm.config import net_off_download_refusal
        raise HTTPException(409, net_off_download_refusal("the voice model was not downloaded"))
    jobs = getattr(request.app.state, "jobs", None)
    if jobs is None:
        raise HTTPException(503, "This server has no background job registry, "
                                 "so the download cannot be started.")

    def _run(job):
        job.push({"type": "line",
                  "text": f"Downloading Whisper model '{name}' (one-time)..."})
        ok, reason = prefetch_stt_model(allow_download=True, name=name)
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
    can resolve, not a defect in this recording); 500 = the decoder library
    itself failed (not the recording's fault); everything else is a 422 on
    this input."""
    code = getattr(e, "code", "")
    if code == "needs-faster-whisper":
        return 501, str(e)
    if code == "download-blocked":
        return 409, str(e)
    if code == "decoder-fault":
        return 500, str(e)
    return 422, str(e)


@_router.post("/api/voice/transcribe")
@route_errors({
    VoiceError: _voice_error_status,
    Exception: lambda e: (502, f"Transcription failed: {e}"),
})
async def voice_transcribe(req: TranscribeRequest, request: Request):
    from localm.voice import transcribe_bytes
    try:
        data = base64.b64decode(req.audio_b64, validate=True)
    except Exception as exc:
        raise HTTPException(400, "audio_b64 is not valid base64") from exc
    if not data:
        raise HTTPException(422, "Empty recording (no audio was captured)")
    if len(data) > 25_000_000:
        raise HTTPException(413, "Recording too large (max 25 MB)")
    loop = asyncio.get_running_loop()
    executor = get_plugin_executor()
    audio_name = await loop.run_in_executor(executor, _transcriptions.resolve_engine, None)
    if audio_name is None:
        try:
            text = await loop.run_in_executor(
                executor, lambda: transcribe_bytes(data, language=req.language))
            return {"text": text}
        except VoiceError as e:
            if getattr(e, "code", "") not in _WHISPER_UNAVAILABLE:
                raise
            names = await loop.run_in_executor(executor, _audio_model.audio_model_names)
            if not names:
                raise
            audio_name = await loop.run_in_executor(
                executor, _audio_model.preferred, names)
    text = await _audio_model.transcribe(
        request, audio_name, data, _audio_model.format_label(data, "", ""),
        language=req.language, prompt=None, temperature=None)
    return {"text": text}


def register(host) -> None:
    host.mount_router(_router)
    host.mount_router(_transcriptions.router)


def on_install() -> str:
    """Prefetch the configured Whisper model when the plugin is installed, so a
    first click on the mic does not stall on a surprise download. Every install
    path runs this hook: the Plugins page, ``localm plugin install`` and
    ``plugin setup`` (setup's plugin step), and the MCP install_plugin tool.

    Installing the plugin is itself an explicit user action, so the prefetch
    passes ``allow_download=True`` for this one download and nothing is
    persisted. net_mode=off still refuses unless net_allow_model_downloads
    exempts explicit downloads: nothing is downloaded and the returned note
    says why. The install succeeds either way, with /api/voice/status carrying
    the honest reason until the model is fetched.

    The model name is read once here and passed on, so the check and the
    download are about the same model. On the server's event loop (a live
    install) the download runs on a background thread, and the pip extras may
    still be installing; the prefetch needs only huggingface_hub (a core
    dependency), not faster-whisper. With no running event loop (the CLI,
    setup, the MCP tool) it runs inline, because that process may exit as soon
    as this hook returns, which ends a daemon thread mid-download.

    Returns a one-line note for the person installing."""
    import asyncio
    import threading

    from localm.debuglog import logger
    from localm.netpolicy import downloads_allowed_when_off, network_mode
    from localm.voice import (PREFETCH_THREAD_NAME, prefetch_stt_model,
                              stt_model_cached)
    cached, name = stt_model_cached()
    if cached:
        return f"The Whisper '{name}' speech model is already downloaded."
    if network_mode() == "off" and not downloads_allowed_when_off():
        from localm.config import net_off_download_refusal
        note = net_off_download_refusal(
            f"the Whisper '{name}' speech model was not downloaded")
        logger.info("voice: %s", note)
        return note

    def _prefetch() -> str:
        try:
            ok, reason = prefetch_stt_model(allow_download=True, name=name)
        except Exception as e:
            reason = f"Whisper model '{name}' prefetch failed: {e}"
            logger.warning("voice: %s", reason)
            return reason
        if ok:
            return (f"Downloaded the Whisper '{name}' speech model; speech-to-text "
                    "now runs fully offline.")
        return reason                              # prefetch logged it

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return _prefetch()
    threading.Thread(target=_prefetch, name=PREFETCH_THREAD_NAME,
                     daemon=True).start()
    return (f"Downloading the Whisper '{name}' speech model in the background "
            "(one time).")


def unregister() -> None:
    pass
