# SPDX-License-Identifier: AGPL-3.0-or-later
"""The Ollama wire format on top of localm's OpenAI-shaped chat path.

Pure translation, no I/O: request models, option and message mapping, the
NDJSON / JSON response builders, the SSE-to-NDJSON stream translator and the
stop-sequence filter. ``routes/ollama.py`` owns the routes, auth and the calls
into the engine."""

from __future__ import annotations

import base64
import codecs
import json
import re
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, AsyncIterator, Iterable, Optional, TypeGuard, Union

from pydantic import BaseModel, ConfigDict

INFERENCE_POST_PATHS = frozenset(
    {"/api/chat", "/api/generate", "/api/embed", "/api/embeddings"})
SHOW_PATH = "/api/show"
READ_GET_PATHS = frozenset({"/api/tags", "/api/ps", "/api/version"})
COPY_PATH = "/api/copy"
BLOBS_PREFIX = "/api/blobs/"

# Full route paths, matched with ``in``: a prefix would also exempt the GUI's
# /api/embedding/warmup from the origin guard.
CROSS_ORIGIN_OK_PATHS = INFERENCE_POST_PATHS | {SHOW_PATH}
OPEN_MODE_GET_PATHS = READ_GET_PATHS

TOOLS_UNSUPPORTED = (
    "tools and tool_calls are not supported on the Ollama API yet; "
    "send the request without them")
SCHEMA_FORMAT_UNSUPPORTED = (
    "format as a JSON schema is not supported yet; use format \"json\" or "
    "omit format")


class OllamaError(Exception):
    """A request the Ollama layer refuses; rendered as ``{"error": message}``."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


# ------------------------------------------------------------------ #
#  Requests                                                            #
# ------------------------------------------------------------------ #

class OllamaMessage(BaseModel):
    model_config = ConfigDict(extra="allow")
    role: str
    content: Optional[str] = ""
    images: Optional[list[str]] = None
    thinking: Optional[str] = None
    tool_calls: Optional[list[Any]] = None
    tool_name: Optional[str] = None


class _Base(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: Optional[str] = None
    options: Optional[dict[str, Any]] = None
    stream: Optional[bool] = None
    keep_alive: Optional[Union[float, int, str]] = None
    format: Optional[Union[str, dict[str, Any]]] = None
    think: Optional[Union[bool, str]] = None


class OllamaChatRequest(_Base):
    messages: Optional[list[OllamaMessage]] = None
    tools: Optional[list[Any]] = None


class OllamaGenerateRequest(_Base):
    prompt: Optional[str] = None
    suffix: Optional[str] = None
    system: Optional[str] = None
    template: Optional[str] = None
    context: Optional[list[int]] = None
    raw: Optional[bool] = None
    images: Optional[list[str]] = None


class OllamaEmbedRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: Optional[str] = None
    input: Optional[Union[str, list[str]]] = None
    prompt: Optional[str] = None
    truncate: Optional[bool] = None
    dimensions: Optional[int] = None
    options: Optional[dict[str, Any]] = None
    keep_alive: Optional[Union[float, int, str]] = None


class OllamaShowRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: Optional[str] = None
    name: Optional[str] = None
    verbose: Optional[bool] = None


class OllamaCopyRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    source: Optional[str] = None
    destination: Optional[str] = None


def request_model_name(req: Any) -> str:
    """The model a request names (``model``, or the legacy ``name``); raises a
    400 when it names none."""
    name = (getattr(req, "model", None) or getattr(req, "name", None) or "").strip()
    if not name:
        raise OllamaError(400, "model is required")
    return name


def unknown_fields(req: BaseModel) -> list[str]:
    """Top-level request keys the models above do not define."""
    return sorted((req.model_extra or {}).keys())


_ZERO_DURATION = re.compile(r"^\s*0+(\.0+)?\s*(ns|us|µs|ms|s|m|h)?\s*$")


def keep_alive_is_zero(value: Any) -> bool:
    """Whether a ``keep_alive`` value asks for an immediate unload."""
    if isinstance(value, bool) or value is None:
        return False
    if isinstance(value, (int, float)):
        return value == 0
    return isinstance(value, str) and bool(_ZERO_DURATION.match(value))


def _finite_number(value: Any) -> TypeGuard[float]:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and value == value and value not in (float("inf"), float("-inf")))


_OPTION_PASSTHROUGH = ("temperature", "top_p", "top_k", "repeat_penalty", "seed")


def options_to_fields(options: Optional[dict[str, Any]]
                      ) -> tuple[dict[str, Any], list[str], list[str]]:
    """Map Ollama ``options`` onto ChatRequest fields.

    Returns ``(fields, stop, ignored)``: *fields* are the OpenAI-shaped request
    fields, *stop* the stop sequences (enforced by :class:`StopFilter`), and
    *ignored* the option names that have no localm equivalent. ``num_predict``
    maps to ``max_tokens`` only when it is 1 or more (-1 and -2 mean "no cap" and
    "fill the context" in Ollama, which an omitted cap already is)."""
    fields: dict[str, Any] = {}
    stop: list[str] = []
    ignored: list[str] = []
    for key, value in (options or {}).items():
        if value is None:
            continue
        if key in _OPTION_PASSTHROUGH:
            fields[key] = value
        elif key == "num_predict":
            if _finite_number(value) and value >= 1:
                fields["max_tokens"] = int(value)
        elif key == "stop":
            if isinstance(value, str):
                stop = [value]
            elif isinstance(value, list) and all(isinstance(s, str) for s in value):
                stop = [s for s in value if s]
            else:
                raise OllamaError(400, "options.stop must be a string or a list of strings")
        else:
            ignored.append(key)
    return fields, stop, sorted(ignored)


JSON_GRAMMAR = r'''root   ::= object
value  ::= object | array | string | number | ("true" | "false" | "null") ws

object ::=
  "{" ws (
            string ":" ws value
    ("," ws string ":" ws value)*
  )? "}" ws

array  ::=
  "[" ws (
            value
    ("," ws value)*
  )? "]" ws

string ::=
  "\"" (
    [^"\\\x7F\x00-\x1F] |
    "\\" (["\\bfnrt] | "u" [0-9a-fA-F]{4})
  )* "\"" ws

number ::= ("-"? ([0-9] | [1-9] [0-9]{0,15})) ("." [0-9]+)? ([eE] [-+]? [0-9] [1-9]{0,15})? ws

ws ::= | " " | "\n" [ \t]{0,20}
'''


def format_to_grammar(fmt: Any) -> Optional[str]:
    """The GBNF grammar a ``format`` value asks for, or ``None`` for no format.

    ``"json"`` constrains the reply to a JSON object. A JSON-schema ``format``
    is refused until schema-constrained output exists in the chat path."""
    if fmt is None or fmt == "":
        return None
    if fmt == "json":
        return JSON_GRAMMAR
    if isinstance(fmt, dict):
        raise OllamaError(400, SCHEMA_FORMAT_UNSUPPORTED)
    raise OllamaError(400, f"unsupported format {fmt!r}: expected \"json\"")


def thinking_requested(think: Any) -> Optional[bool]:
    """``think`` as ``enable_thinking``: ``None`` when unset, a bool when set
    (a level string such as "high" counts as on)."""
    if think is None:
        return None
    if isinstance(think, bool):
        return think
    if isinstance(think, str):
        if think.strip().lower() in ("", "false", "off", "none"):
            return False
        return True
    raise OllamaError(400, "think must be a boolean or a string")


def _sniff_mime(raw_head: bytes) -> str:
    if raw_head.startswith(b"\x89PNG"):
        return "image/png"
    if raw_head.startswith(b"\xff\xd8"):
        return "image/jpeg"
    if raw_head.startswith(b"GIF8"):
        return "image/gif"
    if raw_head[:4] == b"RIFF" and raw_head[8:12] == b"WEBP":
        return "image/webp"
    return "image/png"


def image_data_url(image: str) -> str:
    """An Ollama ``images`` entry (raw base64, or already a data URL) as a data URL."""
    image = image.strip()
    if image.startswith("data:"):
        return image
    b64 = "".join(image.split())
    try:
        head = base64.b64decode(b64[:48] + "=" * (-len(b64[:48]) % 4), validate=True)
    except ValueError:
        raise OllamaError(400, "images must be base64-encoded image data") from None
    return f"data:{_sniff_mime(head)};base64,{b64}"


def message_to_openai(msg: OllamaMessage) -> dict[str, Any]:
    """One Ollama chat message as an OpenAI-shaped message dict."""
    if msg.role not in ("system", "user", "assistant", "tool"):
        raise OllamaError(400, f"unsupported message role {msg.role!r}")
    if msg.tool_calls:
        raise OllamaError(400, TOOLS_UNSUPPORTED)
    text = msg.content or ""
    if not msg.images:
        return {"role": msg.role, "content": text}
    parts: list[dict[str, Any]] = []
    if text:
        parts.append({"type": "text", "text": text})
    for image in msg.images:
        parts.append({"type": "image_url",
                      "image_url": {"url": image_data_url(image)}})
    return {"role": msg.role, "content": parts}


@dataclass
class Plan:
    """A translated request: the OpenAI-shaped *body* plus what the response
    translation needs."""
    body: dict[str, Any]
    stream: bool
    stop: list[str] = field(default_factory=list)
    want_thinking: bool = False
    ignored_options: list[str] = field(default_factory=list)


def _common_fields(req: _Base, resolved_model: str) -> tuple[dict[str, Any], Plan]:
    fields, stop, ignored = options_to_fields(req.options)
    grammar = format_to_grammar(req.format)
    if grammar is not None:
        fields["grammar"] = grammar
    enable_thinking = thinking_requested(req.think)
    if enable_thinking is not None:
        fields["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
    stream = True if req.stream is None else bool(req.stream)
    body = {"model": resolved_model, "stream": stream, **fields}
    return body, Plan(body=body, stream=stream, stop=stop,
                      want_thinking=bool(enable_thinking),
                      ignored_options=ignored)


def plan_chat(req: OllamaChatRequest, resolved_model: str) -> Plan:
    """Translate an ``/api/chat`` request. The caller handles the load/unload
    idiom (no messages) before calling this."""
    if req.tools:
        raise OllamaError(400, TOOLS_UNSUPPORTED)
    body, plan = _common_fields(req, resolved_model)
    body["messages"] = [message_to_openai(m) for m in (req.messages or [])]
    return plan


def plan_generate(req: OllamaGenerateRequest, resolved_model: str) -> Plan:
    """Translate an ``/api/generate`` request. The caller handles the
    load/unload idiom (no prompt) before calling this."""
    if req.raw:
        raise OllamaError(
            400, "raw mode (a prompt sent without the chat template) is not "
                 "supported; omit raw")
    if req.suffix:
        raise OllamaError(400, "suffix (fill-in-the-middle) is not supported")
    if req.template:
        raise OllamaError(400, "a request-level template is not supported; the "
                               "model's own chat template is always used")
    if req.context:
        raise OllamaError(400, "context (a token array from an earlier "
                               "response) is not supported; send the conversation "
                               "with /api/chat")
    body, plan = _common_fields(req, resolved_model)
    messages: list[dict[str, Any]] = []
    if req.system:
        messages.append({"role": "system", "content": req.system})
    messages.append(message_to_openai(OllamaMessage(
        role="user", content=req.prompt or "", images=req.images)))
    body["messages"] = messages
    return plan


# ------------------------------------------------------------------ #
#  Stop sequences                                                      #
# ------------------------------------------------------------------ #

class StopFilter:
    """Cuts a text stream at the first stop sequence.

    ``feed`` returns the text that is safe to emit: everything before a stop
    sequence, and everything except a tail that could still become one.
    ``hit`` turns true once a stop sequence has been seen; nothing is emitted
    after that. ``flush`` releases the held tail when the stream ends without
    a hit."""

    def __init__(self, stops: Iterable[str]) -> None:
        self._stops = [s for s in stops if s]
        self._buf = ""
        self.hit = False

    def feed(self, text: str) -> str:
        if self.hit:
            return ""
        if not self._stops:
            return text
        self._buf += text
        cut = min((i for i in (self._buf.find(s) for s in self._stops) if i >= 0),
                  default=-1)
        if cut >= 0:
            out, self._buf, self.hit = self._buf[:cut], "", True
            return out
        hold = 0
        for s in self._stops:
            for k in range(min(len(s) - 1, len(self._buf)), 0, -1):
                if self._buf.endswith(s[:k]):
                    hold = max(hold, k)
                    break
        out = self._buf[:len(self._buf) - hold]
        self._buf = self._buf[len(self._buf) - hold:]
        return out

    def flush(self) -> str:
        out, self._buf = ("", "") if self.hit else (self._buf, "")
        return out


def apply_stop(text: str, stops: Iterable[str]) -> tuple[str, bool]:
    """*text* cut at the first stop sequence, and whether one was found."""
    flt = StopFilter(stops)
    out = flt.feed(text)
    if flt.hit:
        return out, True
    return out + flt.flush(), False


# ------------------------------------------------------------------ #
#  Responses                                                           #
# ------------------------------------------------------------------ #

def now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def iso_from_timestamp(ts: float) -> str:
    return datetime.fromtimestamp(ts, UTC).isoformat().replace("+00:00", "Z")


NEVER_EXPIRES = "2318-01-01T00:00:00Z"


def expiry_iso(idle_unload_seconds: int, idle_seconds: Optional[float],
               now: Optional[float] = None) -> str:
    """When a loaded model is due to be unloaded: *idle_unload_seconds* after
    its last request, or ``NEVER_EXPIRES`` when no idle timeout is set."""
    if idle_unload_seconds <= 0:
        return NEVER_EXPIRES
    remaining = max(idle_unload_seconds - max(idle_seconds or 0.0, 0.0), 0.0)
    return iso_from_timestamp((time.time() if now is None else now) + remaining)


def _int_or_none(value: Any) -> Optional[int]:
    return int(value) if _finite_number(value) else None


def usage_stats(usage: Optional[dict[str, Any]], total_ns: int) -> dict[str, int]:
    """Ollama's duration and count fields for a finished reply, from a localm
    ``usage`` block. ``total_duration`` is the wall time of the request;
    ``prompt_eval_duration`` is the time to the first token and
    ``eval_duration`` the decode time implied by the measured tokens per second.
    A field with no measurement behind it is left out."""
    out: dict[str, int] = {"total_duration": max(int(total_ns), 0)}
    if not isinstance(usage, dict):
        return out
    prompt = _int_or_none(usage.get("prompt_tokens"))
    completion = _int_or_none(usage.get("completion_tokens"))
    if prompt is not None:
        out["prompt_eval_count"] = prompt
    if completion is not None:
        out["eval_count"] = completion
    ttft = usage.get("ttft_ms")
    if _finite_number(ttft) and ttft >= 0:
        out["prompt_eval_duration"] = int(ttft * 1_000_000)
    tps = usage.get("tokens_per_sec")
    if completion is not None and _finite_number(tps) and tps > 0:
        out["eval_duration"] = int(completion / tps * 1_000_000_000)
    return out


def done_reason(finish_reason: Optional[str], stop_hit: bool = False) -> str:
    if stop_hit:
        return "stop"
    return "length" if finish_reason == "length" else "stop"


def _body_fields(kind: str, content: str, thinking: str, done: bool) -> dict[str, Any]:
    if kind == "chat":
        message: dict[str, Any] = {"role": "assistant", "content": content}
        if thinking:
            message["thinking"] = thinking
        return {"message": message}
    out: dict[str, Any] = {"response": content}
    if thinking:
        out["thinking"] = thinking
    return out


def reply_object(kind: str, model: str, *, content: str = "", thinking: str = "",
                 done: bool = False, reason: Optional[str] = None,
                 stats: Optional[dict[str, int]] = None) -> dict[str, Any]:
    """One response object (a stream line, or the whole non-streaming reply)."""
    out: dict[str, Any] = {"model": model, "created_at": now_iso()}
    out.update(_body_fields(kind, content, thinking, done))
    out["done"] = done
    if done:
        out["done_reason"] = reason or "stop"
        out.update(stats or {})
    return out


_LINE_BREAKERS = {"\x85": "\\u0085", "\u2028": "\\u2028", "\u2029": "\\u2029"}


def encode_line(obj: dict[str, Any]) -> bytes:
    """*obj* as one compact JSON line. The three non-ASCII characters that
    ``str.splitlines`` treats as line breaks are escaped, so a client that
    splits the stream with it still sees one JSON document per line."""
    text = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    for char, escape in _LINE_BREAKERS.items():
        text = text.replace(char, escape)
    return (text + "\n").encode("utf-8")


def completion_to_reply(kind: str, data: dict[str, Any], model: str, *,
                        want_thinking: bool, stop: list[str],
                        total_ns: int) -> dict[str, Any]:
    """A non-streaming OpenAI chat completion as an Ollama reply.

    Raises a 500 :class:`OllamaError` when the generation failed (the chat path
    reports that as ``finish_reason: "error"`` with the error text as content)."""
    choices = data.get("choices") or [{}]
    choice = choices[0] if isinstance(choices[0], dict) else {}
    message = choice.get("message") or {}
    text = message.get("content") or ""
    finish = choice.get("finish_reason")
    if finish == "error":
        raise OllamaError(500, text.strip() or "generation failed")
    text, stopped = apply_stop(text, stop)
    thinking = (message.get("reasoning_content") or "") if want_thinking else ""
    return reply_object(
        kind, model, content=text, thinking=thinking, done=True,
        reason=done_reason(finish, stopped),
        stats=usage_stats(data.get("usage"), total_ns))


# ------------------------------------------------------------------ #
#  Streaming                                                           #
# ------------------------------------------------------------------ #

async def iter_sse_json(chunks: AsyncIterator[Union[str, bytes]]
                        ) -> AsyncIterator[dict[str, Any]]:
    """The JSON objects in an SSE byte or text stream, in order. Comment lines,
    ``[DONE]`` and lines that are not JSON objects are skipped. Closes *chunks*
    when it is closed."""
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    pending = ""
    try:
        async for chunk in chunks:
            pending += chunk if isinstance(chunk, str) else decoder.decode(chunk)
            *lines, pending = pending.split("\n")
            for line in lines:
                obj = _sse_line_json(line)
                if obj is not None:
                    yield obj
        obj = _sse_line_json(pending + decoder.decode(b"", final=True))
        if obj is not None:
            yield obj
    finally:
        close = getattr(chunks, "aclose", None)
        if close is not None:
            await close()


def _sse_line_json(line: str) -> Optional[dict[str, Any]]:
    line = line.strip()
    if not line.startswith("data:"):
        return None
    payload = line[5:].strip()
    if not payload or payload == "[DONE]":
        return None
    try:
        obj = json.loads(payload)
    except ValueError:
        return None
    return obj if isinstance(obj, dict) else None


_ERROR_TEXT_PREFIX = "[inference error"


async def ndjson_stream(events: AsyncIterator[dict[str, Any]], *, kind: str,
                        model: str, want_thinking: bool, stop: list[str],
                        started: float) -> AsyncIterator[bytes]:
    """Translate OpenAI chat-completion chunks into Ollama NDJSON lines.

    Status chunks and the role chunk produce nothing. Reasoning deltas become
    ``thinking`` lines when *want_thinking*. A stop-sequence hit ends the
    reply (``done_reason: "stop"``) and closes *events*, which cancels the
    generation. A failed generation ends with one ``{"error": ...}`` line."""
    flt = StopFilter(stop)
    finish: Optional[str] = None
    usage: Optional[dict[str, Any]] = None
    failure: Optional[str] = None
    held: Optional[str] = None
    try:
        async for ev in events:
            refusal = ev.get("localm_error")
            if isinstance(refusal, dict):
                failure = str(refusal.get("detail") or "request failed")
                break
            if isinstance(ev.get("usage"), dict):
                usage = ev["usage"]
            choices = ev.get("choices") or []
            choice = choices[0] if choices and isinstance(choices[0], dict) else {}
            delta = choice.get("delta") or {}
            reasoning = delta.get("reasoning_content")
            if reasoning and want_thinking:
                yield encode_line(reply_object(kind, model, thinking=reasoning))
            text = delta.get("content")
            if text:
                if held is not None:
                    out = flt.feed(held)
                    held = None
                    if out:
                        yield encode_line(reply_object(kind, model, content=out))
                if text.lstrip().startswith(_ERROR_TEXT_PREFIX):
                    held = text
                else:
                    out = flt.feed(text)
                    if out:
                        yield encode_line(reply_object(kind, model, content=out))
                    if flt.hit:
                        break
            reason = choice.get("finish_reason")
            if reason:
                finish = reason
                break
    finally:
        close = getattr(events, "aclose", None)
        if close is not None:
            await close()
    if failure is not None or finish == "error":
        yield encode_line({"error": failure or (held or "").strip() or "generation failed"})
        return
    if held is not None:
        released = flt.feed(held)
        if released:
            yield encode_line(reply_object(kind, model, content=released))
    tail = flt.flush()
    if tail:
        yield encode_line(reply_object(kind, model, content=tail))
    total_ns = int((time.perf_counter() - started) * 1_000_000_000)
    yield encode_line(reply_object(
        kind, model, done=True, reason=done_reason(finish, flt.hit),
        stats=usage_stats(None if flt.hit else usage, total_ns)))


async def collect_reply(lines: AsyncIterator[bytes], kind: str) -> dict[str, Any]:
    """Merge the NDJSON lines of one reply into the single object a
    non-streaming request answers with. Raises a 500 :class:`OllamaError` when
    the stream carries an error line, and a 502 when it ends without ``done``."""
    content: list[str] = []
    thinking: list[str] = []
    final: Optional[dict[str, Any]] = None
    async for raw in lines:
        obj = json.loads(raw)
        if "error" in obj:
            raise OllamaError(500, str(obj["error"]))
        part = obj.get("message", {}) if kind == "chat" else obj
        content.append(part.get("content" if kind == "chat" else "response") or "")
        thinking.append(part.get("thinking") or "")
        if obj.get("done"):
            final = obj
    if final is None:
        raise OllamaError(502, "the model route ended without a final reply")
    text, reasoning = "".join(content), "".join(thinking)
    if kind == "chat":
        final["message"]["content"] = text
        if reasoning:
            final["message"]["thinking"] = reasoning
    else:
        final["response"] = text
        if reasoning:
            final["thinking"] = reasoning
    return final


# ------------------------------------------------------------------ #
#  Model listings                                                      #
# ------------------------------------------------------------------ #

def model_details(*, fmt: str, family: str = "", parameter_size: str = "",
                  quantization_level: str = "") -> dict[str, Any]:
    return {
        "parent_model": "",
        "format": fmt,
        "family": family,
        "families": [family] if family else None,
        "parameter_size": parameter_size,
        "quantization_level": quantization_level,
    }


def digest_of(entry: dict[str, Any]) -> str:
    """The registry entry's SHA-256 as bare hex, or "" when none is recorded."""
    sha = entry.get("sha256")
    if not isinstance(sha, str):
        return ""
    return sha[len("sha256:"):] if sha.startswith("sha256:") else sha


def resolve_model_name(name: str, known: Iterable[str]) -> str:
    """*name* as a registered model name: an exact match, else the name without
    a trailing ``:latest`` (the tag Ollama clients append to a bare name).
    Anything else comes back unchanged for the caller to refuse."""
    names = set(known)
    if name in names:
        return name
    if name.endswith(":latest") and name[:-len(":latest")] in names:
        return name[:-len(":latest")]
    return name
