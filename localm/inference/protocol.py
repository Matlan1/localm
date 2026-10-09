# SPDX-License-Identifier: AGPL-3.0-or-later
"""OpenAI-compatible request / response types (multimodal-aware)."""

from __future__ import annotations

import uuid
from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator

from localm.inference.backends.base import LOADING_MODEL_STATUS, VISION_CPU_FALLBACK_STATUS
from localm.inference.stop_sequences import normalize_stop


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
    # The model's reasoning, separated from the visible answer. Present only on
    # assistant responses when the model emitted a <think> block; ignored on
    # input. Clients that do not know the field ignore it.
    reasoning_content: Optional[str] = None
    # Character ranges of ``content`` that came from an untrusted source, as
    # ``[[start, end], ...]``. The backend tokenises those ranges with
    # special-token parsing off. Optional and additive: a client that omits it
    # gets exactly the previous behaviour.
    #
    # A range can only DISABLE special-token parsing over part of the sender's
    # own prompt, never enable it, so a wrong or hostile value degrades that
    # request's own output and cannot weaken anyone else's. Values are clamped
    # to the content length when applied.
    untrusted_spans: Optional[List[List[int]]] = None
    # Set when the client, not the user, wrote this message. "tool": a web tool
    # event (a search result, a page read, a control note) the GUI sends as
    # user-role text only so strict chat templates keep user/assistant
    # alternation. "client": text the client generated itself, such as the
    # GUI's compaction "Summarise the following conversation ..." prompt. The
    # server leaves a marked row out of what it treats as the user's own words:
    # the memory recall query and the audit user line.
    # Optional and additive: a client that omits it gets exactly the previous
    # behaviour. Request-only, so a response message never carries it.
    origin: Optional[Literal["tool", "client"]] = Field(None, exclude=True)

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
    # None, not "localm": "localm" is truthy, so it must not also be the
    # field's own default, or an omitted field is indistinguishable from an
    # explicit "localm" request.
    model: Optional[str] = None
    input: Union[str, List[str]]      # single text or batch
    encoding_format: str = "float"    # "float" (JSON array) or "base64"
                                      # (base64 little-endian float32 buffer)


class CompletionRequest(BaseModel):
    """OpenAI /v1/completions (raw text completion) request."""
    # None, not "localm": "localm" is truthy, so a request that OMITS this
    # field would be indistinguishable from one explicitly asking for the
    # "localm" sentinel, and `req.model or engine.display_name` below would
    # never fall through to the model that actually answered.
    model: Optional[str] = None
    prompt: str
    stream: bool = False
    # A request-level cap must be >= 1: the engine uses max_tokens <= 0
    # internally as an "unlimited" sentinel, so a 0/negative from a client is
    # rejected rather than turned into an unbounded generation.
    max_tokens: Optional[int] = Field(None, ge=1)
    # allow_inf_nan=False: stdlib json parses the bare NaN/Infinity tokens, and
    # a non-finite temperature/top_p/penalty would flow straight into the native
    # sampler; it is rejected with a 422 instead.
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
    # Text that ends the reply when generated: one string or a list. The reply
    # is cut before the first match and finish_reason is "stop".
    stop: Optional[List[str]] = None

    @field_validator("stop", mode="before")
    @classmethod
    def _stop_sequences(cls, v):
        """Accept one string or a list of strings; reject anything else."""
        return normalize_stop(v)


class ChatRequest(BaseModel):
    # None, not "localm": "localm" is truthy, so it must not also be the
    # field's own default, or an omitted field is indistinguishable from an
    # explicit "localm" request.
    model: Optional[str] = None
    messages: List[Message]
    stream: bool = False
    # A request cap must be >= 1 (0/negative collides with the internal
    # "unlimited" sentinel), and temperature/top_p/penalty must be finite (a
    # non-finite value reaches the native sampler).
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
    # Text that ends the reply when generated: one string or a list. The reply
    # is cut before the first match and finish_reason is "stop".
    stop: Optional[List[str]] = None
    # Capabilities the answering model must have, e.g. ["tool_use"]. Consulted
    # ONLY when no model is pinned: with an explicit `model`, a gap is reported
    # and the pinned model still answers.
    #
    # Deliberately not named `tools` and deliberately not the OpenAI
    # tools/tool_choice schema. Accepting that shape would advertise
    # tool-calling protocol support this server does not implement; this field
    # claims only what it does, which is to steer model selection. Vision and
    # context length need no entry here - both are derived from the request
    # itself (an image part, the prompt's size).
    required_capabilities: Optional[List[str]] = None
    # Whether `model` is a pin. Unset: a named model is pinned and an absent,
    # empty or "localm" one is not. False: `model` names the preferred model,
    # which routing may replace when it lacks something this request needs.
    # True: whatever answers is never replaced, named or not.
    pin_model: Optional[bool] = None
    # Tokens the answering model's trained context window must hold. Combined
    # with the window the prompt's own size implies; the larger one applies.
    min_context: Optional[int] = Field(None, ge=1)
    # Chat-template switches. Only ``enable_thinking`` is applied: false asks a
    # reasoning model to answer without its reasoning channel. Other keys are
    # accepted and ignored.
    chat_template_kwargs: Optional[Dict[str, Any]] = None

    @field_validator("stop", mode="before")
    @classmethod
    def _stop_sequences(cls, v):
        """Accept one string or a list of strings; reject anything else."""
        return normalize_stop(v)

    @field_validator("chat_template_kwargs")
    @classmethod
    def _enable_thinking_is_boolean(cls, v):
        """Reject an ``enable_thinking`` value that is neither a boolean nor
        null."""
        flag = None if v is None else v.get("enable_thinking")
        if flag is not None and not isinstance(flag, bool):
            raise ValueError("chat_template_kwargs.enable_thinking must be a boolean")
        return v

    @field_validator("required_capabilities")
    @classmethod
    def _known_capabilities(cls, v):
        """Reject an unrecognised capability name instead of ignoring it.

        A typo that silently routed nowhere would look identical to "no
        installed model qualifies", so the caller would never learn the name was
        wrong. Imported lazily: the capability module pulls in the registry, and
        this module is imported at server start."""
        if v is None:
            return v
        from localm.model_manager.capabilities import BOOLEAN_CAPABILITIES
        unknown = [c for c in v if c not in BOOLEAN_CAPABILITIES]
        if unknown:
            raise ValueError(
                f"unknown capability {unknown}; "
                f"expected any of {list(BOOLEAN_CAPABILITIES)}")
        return v


# ------------------------------------------------------------------ #
#  Responses                                                           #
# ------------------------------------------------------------------ #

class ChoiceDelta(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None
    # Streamed reasoning tokens, routed out of `content`. A delta carries one
    # or the other; clients that do not know the field ignore it.
    reasoning_content: Optional[str] = None
    status: Optional[str] = None
    # Stable id for `status` (see STATUS_CODE_BY_TEXT), for a client that
    # localizes the status text instead of displaying it verbatim. None when
    # `status` is not one of the known strings.
    status_code: Optional[str] = None


# Emitted while the prompt is tokenized and evaluated, before the first token.
PROCESSING_PROMPT_STATUS = "Processing prompt..."

# Emitted by the SSE route (not a backend's on_status) while a request sits
# behind the per-model semaphore, before the model has started on it.
WAITING_FOR_MODEL_STATUS = "Waiting for another request to finish..."

# Emitted by the SSE route while it summarises older messages to fit the
# context window, before the reply starts.
COMPACTING_STATUS = "Compacting conversation..."

# Emitted by the SSE route while the memory plugin looks up memories for the
# turn, before the reply starts.
RECALLING_MEMORY_STATUS = "Recalling memories..."

# Emitted by the SSE route while the embedding model memory recall needs is
# downloaded for the first time, before the reply starts.
DOWNLOADING_EMBEDDER_STATUS = "Downloading the embedding model..."

# Emitted by the SSE route while chat-pipeline inlet hooks run, before the
# reply starts.
RUNNING_CHAT_HOOKS_STATUS = "Running chat plugins..."

# Emitted by the SSE route while a requested grammar is checked against the
# model, before the reply starts.
CHECKING_GRAMMAR_STATUS = "Checking grammar..."


# Stable ids for the status strings backends pass to on_status(), keyed by the
# exact English text. `status` always carries the English text for CLI, MCP,
# and any other client that does not know the code.
STATUS_CODE_BY_TEXT: dict[str, str] = {
    PROCESSING_PROMPT_STATUS: "processing",
    "Generating response...": "generating",
    "Encoding image...": "encoding_image",
    "Encoding image (GPU)...": "encoding_image_gpu",
    "Encoding image (CPU)...": "encoding_image_cpu",
    VISION_CPU_FALLBACK_STATUS: "vision_cpu_retry",
    WAITING_FOR_MODEL_STATUS: "waiting",
    COMPACTING_STATUS: "compacting",
    LOADING_MODEL_STATUS: "loading_model",
    RECALLING_MEMORY_STATUS: "recalling_memory",
    DOWNLOADING_EMBEDDER_STATUS: "downloading_embedder",
    RUNNING_CHAT_HOOKS_STATUS: "chat_hooks",
    CHECKING_GRAMMAR_STATUS: "checking_grammar",
}


class StreamChoice(BaseModel):
    index: int = 0
    delta: ChoiceDelta
    finish_reason: Optional[str] = None


class MtpUsage(BaseModel):
    """Multi-Token Prediction for one reply.

    state is "on" (the reply speculated), "paused" (drafting was measured
    slower than one-token decoding and was paused for at least as many steps
    as it ran), "stopped" (it stopped partway, see reason), "off" (this reply
    could not draft at all: reason "image" for a turn with an image), "idle" (MTP is
    available but this reply drafted nothing) or "unavailable" (this model
    cannot speculate, see reason).
    """
    state: str
    drafted: int = 0                 # draft tokens sent to verification
    accepted: int = 0                # how many of them the model accepted
    paused_steps: int = 0            # steps run without drafting because it was slower
    reason: Optional[str] = None


class SpeculationUsage(BaseModel):
    """Speculative drafting for one reply, for any draft source.

    source is the draft source ("mtp" or "ngram"). state and the counts mean
    what they mean in MtpUsage; for ngram, "unavailable" carries the model
    status as reason (e.g. "rewind-unsupported") and "idle" means nothing in
    the reply matched earlier text.
    """
    source: str
    state: str
    drafted: int = 0
    accepted: int = 0
    paused_steps: int = 0
    reason: Optional[str] = None


class UsageInfo(BaseModel):
    prompt_tokens:     int = 0
    completion_tokens: int = 0
    total_tokens:      int = 0
    # localm extensions - OpenAI clients ignore unknown fields
    ttft_ms:        Optional[float] = None   # time to first generated token
    tokens_per_sec: Optional[float] = None   # completion tokens / generation time
    context_capacity: Optional[int] = None   # total tokens allowed in context
    mtp: Optional[MtpUsage] = None           # set when MTP is enabled for the model
    speculation: Optional[SpeculationUsage] = None  # set when a draft source is on


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
    def status_chunk(cls, text: str, model: str, chunk_id: str, ts: int) -> "ChatChunk":
        return cls(
            id=chunk_id,
            created=ts,
            model=model,
            choices=[StreamChoice(delta=ChoiceDelta(
                status=text, status_code=STATUS_CODE_BY_TEXT.get(text)))],
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
