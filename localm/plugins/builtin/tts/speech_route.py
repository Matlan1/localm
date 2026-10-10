# SPDX-License-Identifier: AGPL-3.0-or-later
"""OpenAI-compatible ``POST /v1/audio/speech`` over a text-to-speech GGUF.

The request is JSON (``model``, ``input``, ``voice``, ``response_format``,
``speed``, ``instructions``, ``stream_format``, plus localm's ``language`` and
``seed``) or multipart/form-data with the same fields and an optional
``voice_file`` part: a WAV recording whose voice the speech imitates. The audio
is generated in the isolated speech worker and returned from memory; nothing is
written to disk, so privacy mode needs no extra gating. Each synthesis runs as a
background job of kind ``speak``, so it shows on the activity surfaces with its
progress, and a client that disconnects cancels it.

``response_format`` is ``wav`` (the default) or ``pcm`` (raw 16-bit
little-endian mono samples at the model's 24 kHz). Parameters that would change
the output and are not implemented (another format, ``speed`` other than 1,
``instructions``, ``stream_format`` ``sse``) are a 400 rather than being
ignored. Unknown fields are ignored. The response carries the seed used in
``X-Localm-Seed``, so a result can be reproduced.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response

from localm.inference.http_server import _resolve_disconnect_poll, principal_id
from localm.multipartform import MultipartError, read_form

router = APIRouter()

MAX_JSON_BYTES = 256 * 1024
FORM_OVERHEAD_BYTES = 1024 * 1024
RESPONSE_FORMATS = {"wav": "audio/wav", "pcm": "audio/pcm"}
PCM_SAMPLE_RATE = 24000
_UNSUPPORTED_FORMATS = ("mp3", "opus", "aac", "flac")
_SEED_MAX = 0xFFFFFFFE


@dataclass
class SpeechParams:
    model: Optional[str]
    text: str
    voice: str
    response_format: str
    language: Optional[str]
    seed: Optional[int]
    voice_file: Optional[bytes]


def _bad(message: str) -> HTTPException:
    return HTTPException(400, message)


def _as_str(value, name: str) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        raise _bad(f"'{name}' must be a string.")
    return value


def _parse_speed(value) -> None:
    if value is None or value == "":
        return
    try:
        speed = float(value)
    except (TypeError, ValueError):
        speed = math.nan
    if isinstance(value, bool) or not math.isfinite(speed):
        raise _bad(f"speed must be a number (got {str(value)[:40]!r}).")
    if speed != 1.0:
        raise _bad("speed other than 1.0 is not supported by this server's speech "
                   "models (send 1.0 or omit it).")


def _parse_seed(value) -> Optional[int]:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise _bad("seed must be an integer.")
    try:
        seed = int(value)
    except (TypeError, ValueError, OverflowError):
        raise _bad(f"seed must be an integer (got {str(value)[:40]!r}).") from None
    if isinstance(value, float) and seed != value:
        raise _bad("seed must be an integer.")
    if not 0 <= seed <= _SEED_MAX:
        raise _bad(f"seed must be between 0 and {_SEED_MAX}.")
    return seed


def parse_params(fields: dict, voice_file: Optional[bytes]) -> SpeechParams:
    """Validate the request fields (already decoded from JSON or form data).
    Raises ``HTTPException`` (400)."""
    from localm.inference import speech
    from localm.inference.backends.llamacpp.mtmd_gen import (
        SpeechInputError, resolve_language)
    text = _as_str(fields.get("input"), "input")
    if text is None:
        raise _bad("The 'input' field is required: the text to speak.")
    if not text.strip():
        raise _bad("The 'input' text is empty.")
    if len(text) > speech.MAX_INPUT_CHARS:
        raise _bad(f"input is too long (max {speech.MAX_INPUT_CHARS} characters).")

    fmt = (_as_str(fields.get("response_format"), "response_format") or "wav").strip().lower()
    if fmt in _UNSUPPORTED_FORMATS:
        raise _bad(f"response_format '{fmt}' is not supported: this server returns "
                   "'wav' or 'pcm'.")
    if fmt not in RESPONSE_FORMATS:
        raise _bad(f"response_format must be one of {', '.join(RESPONSE_FORMATS)} "
                   f"(got {fmt[:40]!r}).")

    _parse_speed(fields.get("speed"))
    instructions = _as_str(fields.get("instructions"), "instructions")
    if instructions is not None and instructions.strip():
        raise _bad("instructions are not supported by this server's speech models "
                   "(omit the field).")
    stream_format = (_as_str(fields.get("stream_format"), "stream_format") or "audio").strip().lower()
    if stream_format != "audio":
        raise _bad("Only stream_format 'audio' is supported (the whole file in one "
                   "response).")

    voice = (_as_str(fields.get("voice"), "voice") or speech.DEFAULT_VOICE).strip() \
        or speech.DEFAULT_VOICE
    if voice_file is not None and voice != speech.DEFAULT_VOICE:
        raise _bad("Send either a named 'voice' or a 'voice_file' recording, not both.")

    language = _as_str(fields.get("language"), "language")
    try:
        resolve_language(language)
    except SpeechInputError as e:
        raise _bad(str(e)) from e

    return SpeechParams(
        model=_as_str(fields.get("model"), "model"),
        text=text, voice=voice, response_format=fmt,
        language=(language or "").strip() or None,
        seed=_parse_seed(fields.get("seed")),
        voice_file=voice_file)


async def _read_json(request: Request) -> dict:
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > MAX_JSON_BYTES:
            raise HTTPException(413, f"Request body too large (max "
                                     f"{MAX_JSON_BYTES // 1024} KB).")
    try:
        data = json.loads(bytes(body).decode("utf-8"))
    except (UnicodeDecodeError, ValueError, RecursionError) as e:
        raise _bad("The request body is not valid JSON.") from e
    if not isinstance(data, dict):
        raise _bad("The request body must be a JSON object.")
    return data


async def _read_request(request: Request) -> tuple[dict, Optional[bytes]]:
    from localm.inference import speech
    ctype = (request.headers.get("content-type") or "").lower()
    if not ctype.startswith("multipart/form-data"):
        return await _read_json(request), None
    try:
        form = await read_form(
            request, max_bytes=speech.MAX_REFERENCE_BYTES + FORM_OVERHEAD_BYTES,
            too_large=f"Request too large (the voice recording may be at most "
                      f"{speech.MAX_REFERENCE_BYTES // (1024 * 1024)} MB).")
    except MultipartError as e:
        raise HTTPException(e.status, e.message) from e
    unexpected = [n for n in form.files if n != "voice_file"]
    if unexpected:
        raise _bad(f"Unexpected file field '{unexpected[0][:40]}': the only file this "
                   "route takes is 'voice_file'.")
    files = form.files.get("voice_file") or []
    if len(files) > 1:
        raise _bad("Send at most one 'voice_file'.")
    voice_file = None
    if files:
        voice_file = files[0].data
        if not voice_file:
            raise _bad("The uploaded 'voice_file' is empty.")
    fields = {name: form.first(name) for name in form.fields}
    return fields, voice_file


def _error_status(e: BaseException) -> tuple[int, str]:
    """HTTP status for a synthesis failure: a bad input is a 400, a runtime or
    model that cannot synthesize a 501, a model that failed to load a 503, a
    hung worker (stopped) a 504, a runaway or failed generation or a worker
    crash a 502."""
    from localm.inference import speech
    from localm.inference._speech_runner import SpeechWorkerHung
    from localm.inference.backends.base import PretokenizerUnsafeInputError
    from localm.inference.backends.llamacpp import mtmd_gen as g
    if isinstance(e, speech.SpeechModelError):
        return e.status, str(e)
    if isinstance(e, (g.SpeechInputError, PretokenizerUnsafeInputError)):
        return 400, str(e)
    if isinstance(e, g.SpeechUnavailable):
        return 501, str(e)
    if isinstance(e, speech.SpeechUnavailableError):
        return 503, f"The speech model could not be loaded: {e}"
    if isinstance(e, SpeechWorkerHung):
        return 504, str(e)
    if isinstance(e, g.SpeechBudgetExceeded):
        return 502, str(e)
    return 502, f"Speech synthesis failed: {e}"


def _make_speak(model, params: SpeechParams, reference: Optional[bytes], holder: dict):
    from localm.inference import speech
    from localm.inference.backends.llamacpp.mtmd_gen import SpeechCancelled

    def fn(job) -> bool:
        def on_progress(event: dict) -> None:
            stage = event.get("stage")
            if stage == "loading":
                job.progress(phase="loading the speech model")
            elif stage == "waiting":
                job.progress(phase="waiting for another speech request")
            elif stage == "speaking":
                frames = int(event.get("frames") or 0)
                job.progress(phase="speaking", done=frames, unit="frames",
                             audio_seconds=round(float(event.get("seconds") or 0.0), 2))

        try:
            holder["out"] = speech.synthesize(
                model, params.text, language=params.language, reference_wav=reference,
                seed=params.seed, on_progress=on_progress,
                should_cancel=lambda: job.cancel_requested)
        except SpeechCancelled:
            return False
        except BaseException as e:
            holder["error"] = e
            job.push({"type": "line", "text": _error_status(e)[1]})
            return False
        out = holder["out"]
        job.push({"type": "line", "text": f"{out.seconds:.2f} s of audio"})
        return True

    return fn


_OPENAPI = {"requestBody": {"required": True, "content": {
    "application/json": {"schema": {
        "type": "object", "required": ["input"],
        "properties": {
            "model": {"type": "string",
                      "description": "A registered text-to-speech model; tts-1, "
                                     "tts-1-hd, gpt-4o-mini-tts or omitted name the "
                                     "only one."},
            "input": {"type": "string", "maxLength": 4096},
            "voice": {"type": "string", "default": "default",
                      "description": "'default' or the name of a WAV in the voices "
                                     "folder."},
            "response_format": {"type": "string", "enum": list(RESPONSE_FORMATS),
                                "default": "wav"},
            "speed": {"type": "number", "enum": [1.0]},
            "language": {"type": "string",
                         "description": "Code such as 'en' or name such as 'english'."},
            "seed": {"type": "integer", "minimum": 0, "maximum": _SEED_MAX},
        }}},
    "multipart/form-data": {"schema": {
        "type": "object", "required": ["input"],
        "properties": {
            "input": {"type": "string"},
            "voice_file": {"type": "string", "format": "binary",
                           "description": "A WAV recording (at most 30 s) whose voice "
                                          "to imitate."},
        }}}}},
    "responses": {"200": {"content": {"audio/wav": {}, "audio/pcm": {}}}}}


@router.post("/v1/audio/speech", openapi_extra=_OPENAPI)
async def create_speech(request: Request):
    from localm.inference import speech
    from localm.inference.backends.llamacpp.mtmd_gen import wav_pcm_payload
    from localm.plugins.builtin.voice.transcriptions import _origin_allowed
    if not _origin_allowed(request):
        raise HTTPException(
            403, "Cross-origin request refused (only this server's own pages, local "
                 "apps, or a configured 'cors_origins' entry may request speech).")
    fields, voice_file = await _read_request(request)
    params = parse_params(fields, voice_file)
    import asyncio
    loop = asyncio.get_running_loop()
    try:
        model = await loop.run_in_executor(None, speech.resolve_speech_model, params.model)
        reference = voice_file
        if reference is None:
            reference = await loop.run_in_executor(None, speech.voice_reference, params.voice)
    except speech.SpeechModelError as e:
        raise HTTPException(e.status, str(e)) from e

    jobs = getattr(request.app.state, "jobs", None)
    if jobs is None:
        raise HTTPException(503, "Speech synthesis needs this server's background "
                                 "job registry, which is unavailable.")
    from localm.debuglog import logger
    logger.info("v1 speech: %d-char input, model %s, format %s", len(params.text),
                model.name, params.response_format)
    holder: dict = {}
    job = jobs.start_fn("speak", _make_speak(model, params, reference, holder),
                        owner=principal_id(request), label=f"Speak with {model.name}")
    status = await _await_job(job, request)
    if status == "cancelled":
        raise HTTPException(409, "The speech synthesis was cancelled.")
    if "error" in holder:
        code, message = _error_status(holder["error"])
        raise HTTPException(code, message)
    out = holder.get("out")
    if status != "done" or out is None:
        raise HTTPException(502, "Speech synthesis failed: no audio was produced.")
    body = out.wav
    if params.response_format == "pcm":
        if out.sample_rate != PCM_SAMPLE_RATE:
            raise HTTPException(
                400, f"response_format 'pcm' is 24 kHz audio, and this model speaks "
                     f"at {out.sample_rate} Hz; request 'wav' instead.")
        body = wav_pcm_payload(out.wav)
    return Response(content=body, media_type=RESPONSE_FORMATS[params.response_format],
                    headers={"X-Localm-Seed": str(out.seed)})


_JOB_POLL_SECONDS = 1.0
_CANCEL_SETTLE_SECONDS = 30.0


async def _await_job(job, request: Request) -> str:
    """Wait for *job* to end and return its status. A client that goes away
    cancels the job; the wait then ends once the job stops (at most
    ``_CANCEL_SETTLE_SECONDS``) by raising ``asyncio.CancelledError``."""
    import asyncio
    import time
    queue = job.subscribe()
    poll = _resolve_disconnect_poll(request)
    deadline = None
    try:
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=_JOB_POLL_SECONDS)
            except TimeoutError:
                if deadline is None:
                    if poll is not None and await poll():
                        job.cancel()
                        deadline = time.monotonic() + _CANCEL_SETTLE_SECONDS
                elif time.monotonic() >= deadline:
                    raise asyncio.CancelledError() from None
                continue
            if event.get("type") == "end":
                if deadline is not None:
                    raise asyncio.CancelledError()
                return str(event.get("status", "failed"))
    finally:
        job.unsubscribe(queue)
