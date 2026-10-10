# SPDX-License-Identifier: AGPL-3.0-or-later
"""The Anthropic Messages wire format on top of localm's OpenAI-shaped chat path.

Pure translation, no I/O: the request models, the request to an OpenAI chat
body, a chat completion to a Messages reply, and the chat SSE stream to the
Messages event stream. ``routes/anthropic.py`` owns the routes, auth and the
call into the chat route."""

from __future__ import annotations

import json
import uuid
from typing import Any, AsyncIterator, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

from localm.inference.stop_sequences import normalize_stop

_ERROR_TYPES = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    413: "request_too_large",
    422: "invalid_request_error",
    429: "rate_limit_error",
    503: "overloaded_error",
    529: "overloaded_error",
}

_ERROR_TEXT_PREFIX = "[inference error"

# Routes a cross-origin page may call (matched exactly, like the inference API).
CROSS_ORIGIN_OK_PATHS = frozenset({"/v1/messages", "/v1/messages/count_tokens"})


class AnthropicError(Exception):
    """A request the Messages layer refuses, rendered as an Anthropic error body."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


def error_type(status: int) -> str:
    """The Anthropic ``error.type`` for an HTTP *status*."""
    if status in _ERROR_TYPES:
        return _ERROR_TYPES[status]
    return "invalid_request_error" if 400 <= status < 500 else "api_error"


def error_body(status: int, message: str) -> dict[str, Any]:
    """``{"type": "error", "error": {"type", "message"}}`` for *status*."""
    return {"type": "error", "error": {"type": error_type(status), "message": message}}


# ------------------------------------------------------------------ #
#  Requests                                                            #
# ------------------------------------------------------------------ #

class _Base(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: Optional[str] = None
    messages: list[dict[str, Any]]
    system: Optional[Union[str, list[dict[str, Any]]]] = None
    tools: Optional[list[dict[str, Any]]] = None
    tool_choice: Optional[dict[str, Any]] = None
    thinking: Optional[dict[str, Any]] = None
    output_config: Optional[dict[str, Any]] = None
    mcp_servers: Optional[Any] = None


class MessagesRequest(_Base):
    """``POST /v1/messages``."""
    max_tokens: int = Field(ge=1)
    stop_sequences: Optional[list[str]] = None
    stream: bool = False
    temperature: Optional[float] = Field(None, allow_inf_nan=False)
    top_p: Optional[float] = Field(None, allow_inf_nan=False)
    top_k: Optional[int] = None
    metadata: Optional[dict[str, Any]] = None
    service_tier: Optional[str] = None
    container: Optional[Any] = None


class CountTokensRequest(_Base):
    """``POST /v1/messages/count_tokens``."""


def thinking_enabled(thinking: Optional[dict[str, Any]]) -> bool:
    """Whether a ``thinking`` value asks for reasoning (``enabled``,
    ``adaptive`` or ``between_tools``); absent and ``disabled`` do not. Raises a
    400 for anything else."""
    if thinking is None:
        return False
    kind = thinking.get("type") if isinstance(thinking, dict) else None
    if kind in ("enabled", "adaptive", "between_tools"):
        return True
    if kind == "disabled":
        return False
    raise AnthropicError(
        400, "thinking.type must be 'enabled', 'adaptive', 'between_tools' or 'disabled'")


def output_format(output_config: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """``output_config.format`` as an OpenAI ``response_format`` (a strict JSON
    schema), or ``None`` when no format is set. Raises a 400 for a malformed
    one."""
    if output_config is None:
        return None
    if not isinstance(output_config, dict):
        raise AnthropicError(400, "output_config must be an object")
    fmt = output_config.get("format")
    if fmt is None:
        return None
    if (not isinstance(fmt, dict) or fmt.get("type") != "json_schema"
            or not isinstance(fmt.get("schema"), dict)):
        raise AnthropicError(
            400, "output_config.format must be {\"type\": \"json_schema\", \"schema\": {...}}")
    return {"type": "json_schema",
            "json_schema": {"name": "output", "schema": fmt["schema"], "strict": True}}


def _system_text(system: Any) -> str:
    if system is None:
        return ""
    if isinstance(system, str):
        return system
    parts = []
    for i, block in enumerate(system):
        if not isinstance(block, dict) or block.get("type") != "text":
            raise AnthropicError(400, f"system[{i}] must be a text block")
        parts.append(str(block.get("text") or ""))
    return "\n\n".join(p for p in parts if p)


def _image_part(block: dict[str, Any], where: str) -> dict[str, Any]:
    source = block.get("source")
    if not isinstance(source, dict):
        raise AnthropicError(400, f"{where}.source must be an object")
    kind = source.get("type")
    if kind == "base64":
        media = source.get("media_type")
        data = source.get("data")
        if not isinstance(media, str) or not isinstance(data, str) or not data:
            raise AnthropicError(400, f"{where}.source needs media_type and data")
        return {"type": "image_url", "image_url": {"url": f"data:{media};base64,{data}"}}
    if kind == "url" and isinstance(source.get("url"), str):
        return {"type": "image_url", "image_url": {"url": source["url"]}}
    raise AnthropicError(
        400, f"{where}.source.type {kind!r} is not supported: send base64 or url")


def _tool_result_text(block: dict[str, Any], where: str) -> tuple[str, list[dict[str, Any]]]:
    content = block.get("content")
    images: list[dict[str, Any]] = []
    if content is None:
        text = ""
    elif isinstance(content, str):
        text = content
    elif isinstance(content, list):
        texts = []
        for j, part in enumerate(content):
            kind = part.get("type") if isinstance(part, dict) else None
            if kind == "text":
                texts.append(str(part.get("text") or ""))
            elif kind == "image":
                images.append(_image_part(part, f"{where}.content[{j}]"))
            else:
                raise AnthropicError(
                    400, f"{where}.content[{j}] type {kind!r} is not supported in a "
                         "tool_result: send text or image")
        text = "\n".join(texts)
    else:
        raise AnthropicError(400, f"{where}.content must be a string or a list of blocks")
    if block.get("is_error"):
        text = f"Error: {text}"
    return text, images


def _blocks(content: Any, where: str) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, list) and all(isinstance(b, dict) for b in content):
        return content
    raise AnthropicError(400, f"{where}.content must be a string or a list of blocks")


def messages_to_openai(system: Any, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The Messages conversation as OpenAI chat messages: ``tool_result`` blocks
    become ``tool`` messages (before the rest of their user turn), ``tool_use``
    blocks become the assistant's ``tool_calls``, and thinking blocks from
    earlier turns are dropped."""
    out: list[dict[str, Any]] = []
    sys_text = _system_text(system)
    if sys_text:
        out.append({"role": "system", "content": sys_text})
    for i, msg in enumerate(messages):
        where = f"messages[{i}]"
        role = msg.get("role")
        if role not in ("user", "assistant"):
            raise AnthropicError(400, f"{where}.role must be 'user' or 'assistant'")
        parts: list[dict[str, Any]] = []
        calls: list[dict[str, Any]] = []
        for j, block in enumerate(_blocks(msg.get("content"), where)):
            at = f"{where}.content[{j}]"
            kind = block.get("type")
            if kind == "text":
                parts.append({"type": "text", "text": str(block.get("text") or "")})
            elif kind == "image" and role == "user":
                parts.append(_image_part(block, at))
            elif kind == "tool_result" and role == "user":
                text, images = _tool_result_text(block, at)
                call_id = block.get("tool_use_id")
                if not isinstance(call_id, str) or not call_id:
                    raise AnthropicError(400, f"{at}.tool_use_id is required")
                out.append({"role": "tool", "tool_call_id": call_id, "content": text})
                parts.extend(images)
            elif kind == "tool_use" and role == "assistant":
                name = block.get("name")
                if not isinstance(name, str) or not name:
                    raise AnthropicError(400, f"{at}.name is required")
                args = block.get("input")
                calls.append({
                    "id": str(block.get("id") or "call_" + uuid.uuid4().hex[:24]),
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(
                        {} if args is None else args, ensure_ascii=False)}})
            elif kind in ("thinking", "redacted_thinking") and role == "assistant":
                continue
            else:
                raise AnthropicError(
                    400, f"{at}: a {kind!r} block is not supported in a {role} message")
        if role == "assistant":
            text = "".join(p["text"] for p in parts)
            entry: dict[str, Any] = {"role": "assistant", "content": text}
            if calls:
                entry["tool_calls"] = calls
            if text or calls:
                out.append(entry)
            continue
        if not parts:
            continue
        if len(parts) == 1 and parts[0]["type"] == "text":
            out.append({"role": "user", "content": parts[0]["text"]})
        else:
            out.append({"role": "user", "content": parts})
    return _merge_turns(out)


def _as_parts(content: Any) -> list[dict[str, Any]]:
    return [{"type": "text", "text": content}] if isinstance(content, str) else list(content)


def _merge_turns(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """*messages* with consecutive user turns, and consecutive assistant turns,
    joined into one, as the Messages API treats them."""
    out: list[dict[str, Any]] = []
    for msg in messages:
        prev = out[-1] if out else None
        if prev is None or prev["role"] != msg["role"] or msg["role"] not in ("user", "assistant"):
            out.append(dict(msg))
            continue
        if msg["role"] == "user":
            prev["content"] = _as_parts(prev["content"]) + _as_parts(msg["content"])
        else:
            prev["content"] = (prev.get("content") or "") + (msg.get("content") or "")
            calls = (prev.get("tool_calls") or []) + (msg.get("tool_calls") or [])
            if calls:
                prev["tool_calls"] = calls
    return out


def tools_to_openai(tools: Optional[list[dict[str, Any]]]) -> Optional[list[dict[str, Any]]]:
    """Messages ``tools`` as OpenAI function tools. A server tool (any ``type``
    other than ``custom``) is a 400 naming it."""
    if tools is None:
        return None
    out = []
    for i, tool in enumerate(tools):
        kind = tool.get("type", "custom")
        if kind not in (None, "custom"):
            raise AnthropicError(
                400, f"tools[{i}]: the {kind!r} tool is not supported; only custom "
                     "tools with an input_schema are served")
        fn: dict[str, Any] = {"name": tool.get("name"),
                              "parameters": tool.get("input_schema") or {"type": "object"}}
        if tool.get("description"):
            fn["description"] = tool["description"]
        out.append({"type": "function", "function": fn})
    return out


def tool_choice_to_openai(choice: Optional[dict[str, Any]]) -> tuple[Any, Optional[bool]]:
    """``(tool_choice, parallel_tool_calls)`` for a Messages ``tool_choice``."""
    if choice is None:
        return None, None
    kind = choice.get("type")
    parallel = False if choice.get("disable_parallel_tool_use") else None
    if kind == "auto":
        return "auto", parallel
    if kind == "any":
        return "required", parallel
    if kind == "none":
        return "none", None
    if kind == "tool":
        name = choice.get("name")
        if not isinstance(name, str) or not name:
            raise AnthropicError(400, "tool_choice.name is required when type is 'tool'")
        return {"type": "function", "function": {"name": name}}, parallel
    raise AnthropicError(
        400, "tool_choice.type must be 'auto', 'any', 'tool' or 'none'")


def _common(req: _Base) -> dict[str, Any]:
    if req.mcp_servers:
        raise AnthropicError(400, "mcp_servers is not supported")
    body: dict[str, Any] = {"messages": messages_to_openai(req.system, req.messages)}
    tools = tools_to_openai(req.tools)
    if tools:
        body["tools"] = tools
    choice, parallel = tool_choice_to_openai(req.tool_choice)
    if choice is not None and tools:
        body["tool_choice"] = choice
    if parallel is not None and tools:
        body["parallel_tool_calls"] = parallel
    body["chat_template_kwargs"] = {"enable_thinking": thinking_enabled(req.thinking)}
    fmt = output_format(req.output_config)
    if fmt is not None:
        body["response_format"] = fmt
    return body


def plan_messages(req: MessagesRequest, model: Optional[str]) -> dict[str, Any]:
    """The OpenAI chat body for a ``/v1/messages`` request."""
    if req.container:
        raise AnthropicError(400, "container is not supported")
    body = _common(req)
    body.update({"model": model, "max_tokens": req.max_tokens, "stream": req.stream})
    for key in ("temperature", "top_p", "top_k"):
        value = getattr(req, key)
        if value is not None:
            body[key] = value
    try:
        stop = normalize_stop(req.stop_sequences)
    except ValueError as exc:
        raise AnthropicError(400, f"stop_sequences: {exc}") from None
    if stop:
        body["stop"] = stop
    return body


def plan_count(req: CountTokensRequest, model: Optional[str]) -> dict[str, Any]:
    """The OpenAI chat body whose prompt a ``count_tokens`` request measures."""
    body = _common(req)
    body["model"] = model
    return body


# ------------------------------------------------------------------ #
#  Replies                                                             #
# ------------------------------------------------------------------ #

def new_message_id() -> str:
    return "msg_" + uuid.uuid4().hex[:24]


def stop_reason(finish: Optional[str], stop_sequence: Optional[str]) -> str:
    """The Messages ``stop_reason`` for an OpenAI ``finish_reason``."""
    if finish == "length":
        return "max_tokens"
    if finish == "tool_calls":
        return "tool_use"
    if stop_sequence:
        return "stop_sequence"
    return "end_turn"


def _usage(usage: Any) -> dict[str, int]:
    usage = usage if isinstance(usage, dict) else {}
    return {"input_tokens": int(usage.get("prompt_tokens") or 0),
            "output_tokens": int(usage.get("completion_tokens") or 0),
            "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}


def _tool_use_block(call: dict[str, Any]) -> dict[str, Any]:
    fn = call.get("function") or {}
    args = fn.get("arguments")
    try:
        parsed = json.loads(args) if isinstance(args, str) and args.strip() else {}
    except ValueError:
        parsed = {}
    return {"type": "tool_use", "id": call.get("id") or "toolu_" + uuid.uuid4().hex[:24],
            "name": fn.get("name") or "", "input": parsed if isinstance(parsed, dict) else {}}


def message_from_completion(data: dict[str, Any], model: str,
                            want_thinking: bool) -> dict[str, Any]:
    """A non-streaming OpenAI chat completion as a Messages reply. Raises a 500
    :class:`AnthropicError` when the generation failed."""
    choices = data.get("choices") or [{}]
    choice = choices[0] if isinstance(choices[0], dict) else {}
    message = choice.get("message") or {}
    finish = choice.get("finish_reason")
    text = message.get("content") or ""
    if finish == "error":
        raise AnthropicError(500, text.strip() or "generation failed")
    content: list[dict[str, Any]] = []
    reasoning = message.get("reasoning_content")
    if want_thinking and reasoning:
        content.append({"type": "thinking", "thinking": reasoning, "signature": ""})
    if text:
        content.append({"type": "text", "text": text})
    for call in message.get("tool_calls") or []:
        if isinstance(call, dict):
            content.append(_tool_use_block(call))
    stop_sequence = choice.get("stop_sequence") or None
    reason = stop_reason(finish, stop_sequence)
    return {
        "id": new_message_id(), "type": "message", "role": "assistant", "model": model,
        "content": content, "stop_reason": reason,
        "stop_sequence": stop_sequence if reason == "stop_sequence" else None,
        "usage": _usage(data.get("usage")),
    }


def sse_event(kind: str, data: dict[str, Any]) -> bytes:
    """One server-sent event: ``event: <kind>`` and its JSON *data*."""
    return f"event: {kind}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode()


class _Blocks:
    """The content blocks of a streamed reply: which one is open and its index."""

    def __init__(self) -> None:
        self.index = -1
        self.open: Optional[str] = None

    def start(self, kind: str, block: dict[str, Any]) -> list[bytes]:
        out = self.close()
        self.index += 1
        self.open = kind
        out.append(sse_event("content_block_start", {
            "type": "content_block_start", "index": self.index, "content_block": block}))
        return out

    def ensure(self, kind: str) -> list[bytes]:
        if self.open == kind:
            return []
        block: dict[str, Any] = ({"type": "text", "text": ""} if kind == "text"
                                 else {"type": "thinking", "thinking": "", "signature": ""})
        return self.start(kind, block)

    def delta(self, delta: dict[str, Any]) -> bytes:
        return sse_event("content_block_delta", {
            "type": "content_block_delta", "index": self.index, "delta": delta})

    def close(self) -> list[bytes]:
        if self.open is None:
            return []
        out = []
        if self.open == "thinking":
            out.append(self.delta({"type": "signature_delta", "signature": ""}))
        out.append(sse_event("content_block_stop",
                             {"type": "content_block_stop", "index": self.index}))
        self.open = None
        return out


async def message_stream(events: AsyncIterator[dict[str, Any]], *, model: str,
                         want_thinking: bool) -> AsyncIterator[bytes]:
    """Translate OpenAI chat-completion chunks into the Messages event stream:
    ``message_start``, ``ping``, the content blocks (text, thinking, tool_use with
    its input as one ``input_json_delta``), ``message_delta`` with the stop reason
    and usage, ``message_stop``. A failed generation ends with an ``error``
    event."""
    def opening(name: str) -> list[bytes]:
        return [sse_event("message_start", {"type": "message_start", "message": {
                    "id": new_message_id(), "type": "message", "role": "assistant",
                    "model": name, "content": [], "stop_reason": None, "stop_sequence": None,
                    "usage": {"input_tokens": 0, "output_tokens": 0,
                              "cache_creation_input_tokens": 0,
                              "cache_read_input_tokens": 0}}}),
                sse_event("ping", {"type": "ping"})]

    started = False
    blocks = _Blocks()
    finish: Optional[str] = None
    stop_sequence: Optional[str] = None
    usage: Optional[dict[str, Any]] = None
    failure: Optional[str] = None
    held: Optional[str] = None
    try:
        async for ev in events:
            if not started:
                started = True
                for line in opening(model or str(ev.get("model") or "")):
                    yield line
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
                for line in blocks.ensure("thinking"):
                    yield line
                yield blocks.delta({"type": "thinking_delta", "thinking": reasoning})
            text = delta.get("content")
            if text:
                if held is not None:
                    for line in blocks.ensure("text"):
                        yield line
                    yield blocks.delta({"type": "text_delta", "text": held})
                    held = None
                if text.lstrip().startswith(_ERROR_TEXT_PREFIX):
                    held = text
                else:
                    for line in blocks.ensure("text"):
                        yield line
                    yield blocks.delta({"type": "text_delta", "text": text})
            for call in delta.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                block = _tool_use_block(call)
                for line in blocks.start("tool_use", {**block, "input": {}}):
                    yield line
                yield blocks.delta({"type": "input_json_delta",
                                    "partial_json": json.dumps(block["input"],
                                                               ensure_ascii=False)})
                for line in blocks.close():
                    yield line
            if choice.get("finish_reason"):
                finish = choice["finish_reason"]
                stop_sequence = choice.get("stop_sequence") or None
                break
    finally:
        close = getattr(events, "aclose", None)
        if close is not None:
            await close()
    if not started:
        for line in opening(model):
            yield line
    if failure is not None or finish == "error":
        for line in blocks.close():
            yield line
        message = failure or (held or "").strip() or "generation failed"
        yield sse_event("error", error_body(500, message))
        return
    if held is not None:
        for line in blocks.ensure("text"):
            yield line
        yield blocks.delta({"type": "text_delta", "text": held})
    for line in blocks.close():
        yield line
    counts = _usage(usage)
    reason = stop_reason(finish, stop_sequence)
    yield sse_event("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": reason,
                  "stop_sequence": stop_sequence if reason == "stop_sequence" else None},
        "usage": {"input_tokens": counts["input_tokens"],
                  "output_tokens": counts["output_tokens"]}})
    yield sse_event("message_stop", {"type": "message_stop"})
