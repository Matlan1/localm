# SPDX-License-Identifier: AGPL-3.0-or-later
"""Transcription through an installed GGUF model that hears audio.

``POST /v1/audio/transcriptions`` answers with one of these models when
faster-whisper is not installed, or when the request's ``model`` names one. The
clip is sent to the model as an ``input_audio`` chat part with a transcription
instruction, in memory, the same way ``/v1/chat/completions`` would carry it.
Nothing is recorded in the audit log or the transcript. Only ``json`` and
``text`` can be produced: the model returns plain text without timestamps.
"""

from __future__ import annotations

import asyncio
import base64
import re
from typing import Optional

from fastapi import HTTPException, Request

INSTRUCTION = "Transcribe this."
SUGGESTED_PULL = "ggml-org/Qwen3-ASR-0.6B-GGUF:Qwen3-ASR-0.6B-Q8_0.gguf"
SUPPORTED_FORMATS = ("json", "text")
REPLY_TOKEN_CEILING = 8192

_LABEL_RE = re.compile(r"^[a-z0-9]{1,8}$")
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_ASR_TAG = "<asr_text>"
_ASR_LANGUAGE_RE = re.compile(r"language [^<\n]{0,40}" + re.escape(_ASR_TAG))


def audio_model_names() -> list[str]:
    """Registered models that can take audio input, sorted. Reads model files,
    so callers on the event loop run it in an executor."""
    from localm.model_manager import audio_capable_models
    return audio_capable_models()


def is_registered(name: str) -> bool:
    from localm.config import load_registry
    reg = load_registry()
    return isinstance(reg, dict) and name in reg


def preferred(names: list[str]) -> str:
    """The model to answer with: one already loaded when there is one (no model
    swap), else the first by name."""
    import localm.inference.http_server as _hs
    engines = dict(list(_hs._engines.items()))
    for name in names:
        if getattr(engines.get(name), "loaded", False):
            return name
    return names[0]


def format_label(data: bytes, filename: str, content_type: str) -> str:
    """The ``input_audio.format`` for an upload: ``wav`` for a RIFF/WAVE
    payload only, else the file extension or the content type's subtype, else
    ``audio``. A non-WAV payload is never labelled ``wav``."""
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return "wav"
    suffix = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    subtype = (content_type or "").split(";")[0].rpartition("/")[2].lower()
    subtype = subtype.removeprefix("x-")
    for label in (suffix, subtype):
        if _LABEL_RE.match(label) and label != "wav":
            return label
    return "audio"


def instruction(language: Optional[str], prompt: Optional[str]) -> str:
    """The text part sent with the clip."""
    text = INSTRUCTION
    if language:
        text += f" The speech is in the language with ISO 639-1 code '{language}'."
    if prompt:
        text += f" Context that may help: {prompt}"
    return text


def clean_transcript(reply: str) -> str:
    """The spoken words in a model's *reply*. A Qwen3-ASR model opens each
    transcript with ``language <name>`` and an ``<asr_text>`` marker
    (``language None`` for a clip with no speech); both are removed, as is any
    ``<think>`` block."""
    reply = _THINK_RE.sub("", reply)
    return _ASR_LANGUAGE_RE.sub("", reply).replace(_ASR_TAG, "").strip()


def build_messages(data: bytes, label: str, language: Optional[str],
                   prompt: Optional[str]) -> list[dict]:
    return [{"role": "user", "content": [
        {"type": "input_audio",
         "input_audio": {"data": base64.b64encode(data).decode("ascii"),
                         "format": label}},
        {"type": "text", "text": instruction(language, prompt)},
    ]}]


def _generated_tokens(engine, text: str, budget: int) -> int:
    """The token count of the *text* a model generated, or *budget* when the
    engine cannot count them."""
    try:
        count = engine.count_tokens(text)
    except Exception:
        return budget
    return count if isinstance(count, int) and not isinstance(count, bool) and count > 0 else budget


async def transcribe(request: Request, model: str, data: bytes, label: str, *,
                     language: Optional[str], prompt: Optional[str],
                     temperature: Optional[float]) -> str:
    """The transcript of *data* from the registered model *model*. Raises
    ``HTTPException``: 400 for a model that cannot take audio or a clip it
    rejects, 413 for a clip that does not fit the model's context, 501 when
    non-WAV audio needs the voice extra, 502 for a transcript cut off at the
    reply limit."""
    import localm.inference.http_server as _hs
    from localm.inference.inference_gate import InferenceGate

    engine = await _hs.get_engine(model, activate=False)
    _hs._pin(engine)
    try:
        _hs._touch_activity(engine.display_name)
        if getattr(engine, "supports_audio", False) is not True:
            from localm.model_manager import audio_input_guidance
            backend = getattr(engine, "_backend", None)
            projector_failed = (bool(getattr(backend, "mmproj_path", None))
                                and engine.supports_images is not True)
            raise HTTPException(400, audio_input_guidance(projector_failed=projector_failed))

        messages = build_messages(data, label, language, prompt)
        loop = asyncio.get_running_loop()
        try:
            prompt_tokens = await loop.run_in_executor(
                None, engine.count_messages_tokens, messages)
        except _hs._BACKEND_ERROR_TYPES as e:
            raise HTTPException(_hs.backend_error_status(e) or 500, str(e)) from e
        capacity = engine.context_capacity()
        budget = REPLY_TOKEN_CEILING
        if isinstance(capacity, int) and capacity > 0 and isinstance(prompt_tokens, int):
            if prompt_tokens >= capacity:
                raise HTTPException(413, _hs.context_overflow_detail(prompt_tokens, capacity))
            budget = min(REPLY_TOKEN_CEILING, capacity - prompt_tokens)

        sem = _hs._inference_sems.setdefault(engine.display_name, InferenceGate())
        _hs._admit_generation(sem, engine)
        timing: dict = {}
        try:
            async with sem:
                text = await _hs._generate_full(
                    engine, messages, request, timing=timing, max_tokens=budget,
                    temperature=0.0 if temperature is None else temperature,
                    repeat_penalty=1.0)
        except _hs._BACKEND_ERROR_TYPES as e:
            raise HTTPException(_hs.backend_error_status(e) or 500, str(e)) from e
        except RuntimeError as e:
            from localm.pathscrub import scrub_paths
            raise HTTPException(502, f"Transcription failed: {scrub_paths(str(e))}") from e
        finish = (timing.get("outcome") or {}).get("finish_reason")
        if finish == "length":
            generated = await loop.run_in_executor(None, _generated_tokens, engine, text, budget)
            raise HTTPException(
                502, f"The transcript from '{engine.display_name}' was cut off at "
                     f"{generated} tokens. Send a shorter clip.")
        if finish == "error":
            raise HTTPException(
                502, f"The transcription by '{engine.display_name}' was interrupted "
                     "before it finished, so the text is incomplete. Try again.")
        return clean_transcript(text)
    finally:
        _hs._unpin(engine)
        _hs._touch_activity(engine.display_name)
