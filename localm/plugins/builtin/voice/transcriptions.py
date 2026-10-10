# SPDX-License-Identifier: AGPL-3.0-or-later
"""OpenAI-compatible ``POST /v1/audio/transcriptions`` over the Whisper worker.

The request is a multipart/form-data upload (``file`` plus form fields), parsed
in memory by ``localm.multipartform``; the audio is decoded and transcribed in
the isolated speech worker and is never written to disk, so privacy mode needs
no extra gating. The client-supplied file name is never used as a path.

Honoured form fields: ``file``, ``model``, ``language``, ``prompt``,
``response_format`` (json, text, srt, vtt, verbose_json), ``temperature`` and
``timestamp_granularities[]``. Fields that change the output and are not
implemented (``stream``, ``include[]``, ``chunking_strategy``,
``known_speaker_*``) are a 400 rather than being ignored.

Two engines answer. Whisper is the default and is used whenever faster-whisper
is installed and ``model`` is blank, ``whisper-1`` or the configured
``voice_stt_model``. An installed GGUF model that hears audio (see
``audio_model.py``) answers when ``model`` names it, or when faster-whisper is
not installed and ``model`` is blank or a Whisper name; it produces ``json``
and ``text`` only. A ``model`` that is neither a Whisper name nor an
audio-capable registered model is a 404, or a 400 when it is registered but
cannot hear.
"""

from __future__ import annotations

import asyncio
import importlib.util
import math
import re
from dataclasses import dataclass
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse

from localm import voice_formats
from localm.executor import get_plugin_executor
from localm.inference.errors import route_errors
from localm.multipartform import Form, MultipartError, read_form
from localm.pathscrub import scrub_paths
from localm.voice import VoiceError, transcribe_detailed

from . import audio_model

router = APIRouter()

MAX_AUDIO_BYTES = 25 * 1024 * 1024
FORM_OVERHEAD_BYTES = 1024 * 1024
MAX_PROMPT_CHARS = 8192
OPENAI_MODEL_ALIAS = "whisper-1"
GRANULARITIES = ("word", "segment")

_UNSUPPORTED_FIELDS = ("include[]", "include", "chunking_strategy",
                       "known_speaker_names[]", "known_speaker_names",
                       "known_speaker_references[]", "known_speaker_references")
_LANGUAGE_RE = re.compile(r"^[A-Za-z]{2,3}$")
_LOCAL_ORIGIN_RE = re.compile(r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$")


@dataclass
class TranscriptionParams:
    model: Optional[str]
    language: Optional[str]
    prompt: Optional[str]
    response_format: str
    temperature: Optional[float]
    word_timestamps: bool
    include_segments: bool


def _origin_allowed(request: Request) -> bool:
    """False for a browser request whose Origin is not this server itself, a
    local page (localhost or 127.0.0.1, any port) or a configured
    ``cors_origins`` entry. A request with no Origin (an SDK, curl) passes.

    A multipart upload is a CORS "simple request", so a browser sends it from any
    website without a preflight; this keeps the route at the same reach as the
    JSON inference routes, whose preflight only local origins pass."""
    origin = request.headers.get("origin")
    if not origin:
        return True
    if origin.split("://", 1)[-1] == request.headers.get("host", ""):
        return True
    if _LOCAL_ORIGIN_RE.match(origin):
        return True
    from localm.config import load_config
    configured = load_config().get("cors_origins")
    return configured == "*" or (isinstance(configured, list) and origin in configured)


def _audio_too_large() -> str:
    return f"Audio file too large (max {MAX_AUDIO_BYTES // (1024 * 1024)} MB)."


def _bad_request(message: str) -> HTTPException:
    return HTTPException(400, message)


def _served_model_names() -> set[str]:
    from localm.config import load_config
    from localm.voice import _stt_repo_for
    name = str(load_config().get("voice_stt_model", "base"))
    return {OPENAI_MODEL_ALIAS, name.lower(), _stt_repo_for(name).lower()}


def _whisper_installed() -> bool:
    return importlib.util.find_spec("faster_whisper") is not None


def resolve_engine(model: Optional[str]) -> Optional[str]:
    """The engine that answers a request naming *model*: ``None`` for Whisper,
    else the registered audio-capable model's name. Raises ``HTTPException``
    (404 for an unknown model, 400 for a registered one that cannot hear).

    Blocking (it reads model files); callers on the event loop run it in an
    executor."""
    name = (model or "").strip()
    whisper_name = not name or name.lower() in _served_model_names()
    if whisper_name and _whisper_installed():
        return None
    audio = audio_model.audio_model_names()
    if whisper_name:
        return audio_model.preferred(audio) if audio else None
    if name in audio:
        return name
    from localm.config import load_config
    configured = str(load_config().get("voice_stt_model", "base"))
    if audio_model.is_registered(name):
        raise HTTPException(
            400, f"The model '{name[:80]}' cannot take audio input, so it cannot "
                 "transcribe. " + _audio_choices(audio))
    raise HTTPException(
        404, f"The model '{name[:80]}' does not exist on this server. It "
             f"transcribes with the Whisper model '{configured}' (send model "
             f"'{OPENAI_MODEL_ALIAS}' or '{configured}')"
             + (f" or with an installed audio model: {', '.join(audio)}."
                if audio else "."))


def _audio_choices(audio: list[str]) -> str:
    if audio:
        return "Models here that can: " + ", ".join(audio) + "."
    return ("No installed model can. Pull one, for example "
            f"'localm pull {audio_model.SUGGESTED_PULL}'.")


def parse_params(form: Form) -> TranscriptionParams:
    """Validate the non-file form fields. Raises ``HTTPException`` (400)."""
    for name in _UNSUPPORTED_FIELDS:
        if name in form.fields:
            raise _bad_request(f"The '{name.rstrip('[]')}' parameter is not supported "
                               "by this server.")
    stream = (form.first("stream") or "").strip().lower()
    if stream in ("true", "1"):
        raise _bad_request("Streaming transcription is not supported "
                           "(send stream=false or omit it).")

    response_format = (form.first("response_format") or "json").strip().lower()
    if response_format not in voice_formats.RESPONSE_FORMATS:
        raise _bad_request(
            f"response_format must be one of {', '.join(voice_formats.RESPONSE_FORMATS)} "
            f"(got {response_format!r}).")

    language = (form.first("language") or "").strip()
    if language and not _LANGUAGE_RE.match(language):
        raise _bad_request(
            f"language must be an ISO 639-1 code such as 'en' (got {language!r}).")

    prompt = form.first("prompt")
    if prompt is not None and len(prompt) > MAX_PROMPT_CHARS:
        raise _bad_request(f"prompt is too long (max {MAX_PROMPT_CHARS} characters).")

    temperature: Optional[float] = None
    raw_temp = form.first("temperature")
    if raw_temp is not None and raw_temp.strip():
        try:
            temperature = float(raw_temp)
        except ValueError:
            temperature = math.nan
        if not math.isfinite(temperature) or not 0.0 <= temperature <= 1.0:
            raise _bad_request(
                f"temperature must be a number between 0 and 1 (got {raw_temp!r}).")

    granularities = (form.fields.get("timestamp_granularities[]")
                     or form.fields.get("timestamp_granularities") or [])
    granularities = [g.strip().lower() for g in granularities if g.strip()]
    for g in granularities:
        if g not in GRANULARITIES:
            raise _bad_request(
                f"timestamp_granularities must be 'word' or 'segment' (got {g!r}).")
    if granularities and response_format != "verbose_json":
        raise _bad_request(
            "timestamp_granularities requires response_format 'verbose_json'.")

    return TranscriptionParams(
        model=form.first("model"),
        language=language.lower() or None,
        prompt=prompt.strip() or None if prompt is not None else None,
        response_format=response_format,
        temperature=temperature,
        word_timestamps="word" in granularities,
        include_segments=("segment" in granularities) or not granularities,
    )


def _error_status(e: VoiceError) -> tuple[int, str]:
    """Map a ``VoiceError`` class to an HTTP status: a bad input is a 400, a
    missing package a 501, a blocked model download a 409, an unavailable engine
    a 503, a hang a 504, and a decoder or engine fault a 500 or 502."""
    code = getattr(e, "code", "")
    if code == "needs-faster-whisper":
        return 501, (f"{e}. Or pull a model that hears audio, for example "
                     f"'localm pull {audio_model.SUGGESTED_PULL}', and this route "
                     "will use it.")
    status = {
        "bad-request": 400, "decode": 400, "empty": 400,
        "needs-faster-whisper": 501, "download-blocked": 409,
        "spawn": 503, "load": 503, "timeout": 504,
        "decoder-fault": 500,
    }.get(code, 502)
    return status, str(e)


def render(detail: dict, params: TranscriptionParams):
    """Build the HTTP response for a transcription in the requested format."""
    fmt = params.response_format
    if fmt == "json":
        return JSONResponse({"text": detail["text"]})
    if fmt == "text":
        return PlainTextResponse(detail["text"] + "\n")
    if fmt == "srt":
        return PlainTextResponse(voice_formats.to_srt(detail["segments"]))
    if fmt == "vtt":
        return PlainTextResponse(voice_formats.to_vtt(detail["segments"]))
    return JSONResponse(voice_formats.to_verbose_json(
        detail, include_segments=params.include_segments,
        include_words=params.word_timestamps))


_OPENAPI_REQUEST_BODY = {"requestBody": {"required": True, "content": {
    "multipart/form-data": {"schema": {
        "type": "object", "required": ["file"],
        "properties": {
            "file": {"type": "string", "format": "binary",
                     "description": "The audio file, at most 25 MB."},
            "model": {"type": "string",
                      "description": "whisper-1, the configured voice_stt_model, "
                                     "or an installed model that hears audio."},
            "language": {"type": "string",
                         "description": "ISO 639-1 code; detected when omitted."},
            "prompt": {"type": "string"},
            "response_format": {"type": "string",
                                "enum": list(voice_formats.RESPONSE_FORMATS),
                                "default": "json"},
            "temperature": {"type": "number", "minimum": 0, "maximum": 1},
            "timestamp_granularities[]": {
                "type": "array",
                "items": {"type": "string", "enum": list(GRANULARITIES)}},
        }}}}}}


@router.post("/v1/audio/transcriptions", openapi_extra=_OPENAPI_REQUEST_BODY)
@route_errors({
    VoiceError: _error_status,
    Exception: lambda e: (502, f"Transcription failed: {scrub_paths(str(e))}"),
})
async def create_transcription(request: Request):
    if not _origin_allowed(request):
        raise HTTPException(
            403, "Cross-origin request refused (only this server's own pages, local "
                 "apps, or a configured 'cors_origins' entry may upload audio).")
    try:
        form = await read_form(
            request, max_bytes=MAX_AUDIO_BYTES + FORM_OVERHEAD_BYTES,
            too_large=_audio_too_large())
    except MultipartError as e:
        raise HTTPException(e.status, e.message) from e
    params = parse_params(form)
    loop = asyncio.get_running_loop()
    audio_name = await loop.run_in_executor(
        get_plugin_executor(), resolve_engine, params.model)
    if audio_name is not None and params.response_format not in audio_model.SUPPORTED_FORMATS:
        raise _bad_request(
            f"response_format '{params.response_format}' needs timestamps, which "
            f"the audio model '{audio_name}' does not produce. Use 'json' or "
            "'text', or install faster-whisper (pip install \"localm[voice]\") "
            "and send model 'whisper-1'.")

    uploads = form.files.get("file") or []
    if not uploads:
        raise _bad_request("The 'file' field is required: upload the audio as a "
                           "multipart file part named 'file'.")
    if len(uploads) > 1 or len(form.files) > 1:
        raise _bad_request("Send exactly one audio file in the 'file' field.")
    data = uploads[0].data
    if not data:
        raise _bad_request("The uploaded audio file is empty.")
    if len(data) > MAX_AUDIO_BYTES:
        raise HTTPException(413, _audio_too_large())

    if audio_name is not None:
        text = await audio_model.transcribe(
            request, audio_name, data,
            audio_model.format_label(data, uploads[0].filename, uploads[0].content_type),
            language=params.language, prompt=params.prompt,
            temperature=params.temperature)
        response = render({"text": text}, params)
        response.headers["X-Localm-Transcription-Model"] = (
            audio_name.encode("ascii", "replace").decode("ascii"))
        return response
    detail = await loop.run_in_executor(
        get_plugin_executor(),
        lambda: transcribe_detailed(
            data, language=params.language, prompt=params.prompt,
            temperature=params.temperature,
            word_timestamps=params.word_timestamps))
    return render(detail, params)
