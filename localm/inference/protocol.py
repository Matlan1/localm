# SPDX-License-Identifier: AGPL-3.0-or-later
"""OpenAI-compatible request / response types (multimodal-aware)."""

from __future__ import annotations

import uuid
from typing import Annotated, List, Literal, Optional, Union

from pydantic import BaseModel, Field


# ------------------------------------------------------------------ #
#  Request content parts                                               #
# ------------------------------------------------------------------ #

class TextPart(BaseModel):
    type: Literal["text"]
    text: str


class ImageUrl(BaseModel):
    url: str          # "data:image/jpeg;base64,..." or http URL
    detail: str = "auto"


class ImagePart(BaseModel):
    type: Literal["image_url"]
    image_url: ImageUrl


class InputAudioData(BaseModel):
    data: str         # base64-encoded audio
    format: str = "wav"


class AudioPart(BaseModel):
    type: Literal["input_audio"]
    input_audio: InputAudioData


ContentPart = Annotated[
    Union[TextPart, ImagePart, AudioPart],
    Field(discriminator="type"),
]

MessageContent = Union[str, List[ContentPart]]


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: MessageContent = ""
    # The model's reasoning, separated from the visible answer (H4). Present only
    # on assistant responses when the model emitted a <think> block; ignored on
    # input. OpenAI reasoning-model convention (clients that don't know it ignore
    # the extra field).
    reasoning_content: Optional[str] = None

    def text_only(self) -> str:
        """Flatten content to plain text (discards media)."""
        if isinstance(self.content, str):
            return self.content
        return " ".join(p.text for p in self.content if isinstance(p, TextPart))

    def images(self) -> list:
        """Return PIL Images decoded from all image_url parts."""
        from localm.inference.media import decode_image_url
        if isinstance(self.content, str):
            return []
        return [
            decode_image_url(p.image_url.url)
            for p in self.content
            if isinstance(p, ImagePart)
        ]

    def audios(self) -> list:
        """Return (audio_array, sample_rate) tuples decoded from audio parts."""
        from localm.inference.media import decode_audio
        if isinstance(self.content, str):
            return []
        return [
            decode_audio(p.input_audio.data, p.input_audio.format)
            for p in self.content
            if isinstance(p, AudioPart)
        ]


# ------------------------------------------------------------------ #
#  Chat completion request                                             #
# ------------------------------------------------------------------ #

class EmbeddingRequest(BaseModel):
    """OpenAI /v1/embeddings request."""
    model: str = "localm"
    input: Union[str, List[str]]      # single text or batch
    encoding_format: str = "float"    # "float" (JSON array) or "base64"
                                      # (base64 little-endian float32 buffer)


class CompletionRequest(BaseModel):
    """OpenAI /v1/completions (raw text completion) request."""
    model: str = "localm"
    prompt: str
    stream: bool = False
    # A request-level cap must be >= 1. The engine uses max_tokens <= 0 internally
    # as an "unlimited" sentinel, so an unvalidated 0/negative from a client would
    # silently become an unbounded generation (a soft DoS) instead of the short/no
    # reply the client asked for. Reject it (matching OpenAI) rather than misread it.
    max_tokens: Optional[int] = Field(None, ge=1)
    # allow_inf_nan=False: stdlib json parses the bare NaN/Infinity tokens, and a
    # non-finite temperature/top_p/penalty flows straight into the native sampler.
    # Reject it with a clean 422 instead of feeding NaN into logit scaling.
    temperature: Optional[float] = Field(None, allow_inf_nan=False)
    top_p: Optional[float] = Field(None, allow_inf_nan=False)
    top_k: Optional[int] = None
    repeat_penalty: Optional[float] = Field(None, allow_inf_nan=False)
    grammar: Optional[str] = None
    # Lazy grammar: unconstrained until the output matches a trigger pattern,
    # then the grammar enforces (text-or-tool). Requires grammar_triggers.
    grammar_lazy: bool = False
    grammar_triggers: Optional[List[str]] = None
    seed: Optional[int] = None


class ChatRequest(BaseModel):
    model: str = "localm"
    messages: List[Message]
    stream: bool = False
    # See CompletionRequest for the rationale: a request cap must be >= 1 (0/negative
    # collides with the internal "unlimited" sentinel), and temperature/top_p/penalty
    # must be finite (a non-finite value would poison the native sampler).
    max_tokens: Optional[int] = Field(None, ge=1)
    temperature: Optional[float] = Field(None, allow_inf_nan=False)
    top_p: Optional[float] = Field(None, allow_inf_nan=False)
    top_k: Optional[int] = None
    repeat_penalty: Optional[float] = Field(None, allow_inf_nan=False)
    grammar: Optional[str] = None  # GBNF grammar string for constrained sampling
    # Lazy grammar: unconstrained until the output matches a trigger pattern,
    # then the grammar enforces (text-or-tool). Requires grammar_triggers.
    grammar_lazy: bool = False
    grammar_triggers: Optional[List[str]] = None
    seed: Optional[int] = None     # RNG seed for reproducible generation


# ------------------------------------------------------------------ #
#  Responses                                                           #
# ------------------------------------------------------------------ #

class ChoiceDelta(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None
    # Streamed reasoning tokens (H4), routed out of `content`. A delta carries
    # one or the other; clients that don't know the field ignore it.
    reasoning_content: Optional[str] = None


class StreamChoice(BaseModel):
    index: int = 0
    delta: ChoiceDelta
    finish_reason: Optional[str] = None


class UsageInfo(BaseModel):
    prompt_tokens:     int = 0
    completion_tokens: int = 0
    total_tokens:      int = 0
    # localm extensions - OpenAI clients ignore unknown fields
    ttft_ms:        Optional[float] = None   # time to first generated token
    tokens_per_sec: Optional[float] = None   # completion tokens / generation time
    context_capacity: Optional[int] = None   # total tokens allowed in context


class ChatChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[StreamChoice]
    usage: Optional[UsageInfo] = None

    @classmethod
    def token(cls, token: str, model: str, chunk_id: str, ts: int) -> "ChatChunk":
        return cls(
            id=chunk_id,
            created=ts,
            model=model,
            choices=[StreamChoice(delta=ChoiceDelta(content=token))],
        )

    @classmethod
    def done(
        cls,
        model: str,
        chunk_id: str,
        ts: int,
        usage: Optional["UsageInfo"] = None,
        finish_reason: str = "stop",
    ) -> "ChatChunk":
        return cls(
            id=chunk_id,
            created=ts,
            model=model,
            choices=[StreamChoice(delta=ChoiceDelta(), finish_reason=finish_reason)],
            usage=usage,
        )


class FullChoice(BaseModel):
    index: int = 0
    message: Message
    finish_reason: str = "stop"


class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[FullChoice]
    usage: Optional[UsageInfo] = None


def make_chunk_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:12]}"
