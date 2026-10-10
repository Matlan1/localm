# SPDX-License-Identifier: AGPL-3.0-or-later
"""The OpenAI Responses wire format on top of localm's chat-completions path.

Pure translation and an in-memory response store, no I/O: the request model,
the request to an OpenAI chat body, a chat completion to a Response object, and
the chat SSE stream to the Responses event stream. ``routes/responses.py`` owns
the routes, auth and the call into the chat route."""

from __future__ import annotations

import json
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

STORE_MAX_RESPONSES = 256
STORE_MAX_BYTES = 64 * 1024 * 1024
STORE_TTL_SECONDS = 3600

_ERROR_TEXT_PREFIX = "[inference error"

# Routes a cross-origin page may call (matched exactly, like the inference API).
CROSS_ORIGIN_OK_PATHS = frozenset({"/v1/responses"})

# ``include`` values accepted with no effect.
INCLUDE_ACCEPTED = frozenset({
    "reasoning.encrypted_content", "file_search_call.results", "web_search_call.results",
    "web_search_call.action.sources", "message.input_image.image_url",
    "computer_call_output.output.image_url", "code_interpreter_call.outputs",
})

# localm chat fields a Responses request may carry (for example through an SDK's
# ``extra_body``); they are passed to the chat route unchanged.
CHAT_EXTRAS = (
    "top_k", "min_p", "repeat_penalty", "presence_penalty", "frequency_penalty", "seed",
    "stop", "grammar", "grammar_lazy", "grammar_triggers", "required_capabilities",
    "pin_model", "min_context", "chat_template_kwargs",
)


class ResponsesError(Exception):
    """A request the Responses layer refuses, rendered as an OpenAI error body."""

    def __init__(self, status: int, message: str, param: Optional[str] = None) -> None:
        super().__init__(message)
        self.status = status
        self.message = message
        self.param = param


def error_body(status: int, message: str, param: Optional[str] = None) -> dict[str, Any]:
    """OpenAI's ``{"error": {"message", "type", "param", "code"}}``."""
    kind = ("invalid_request_error" if 400 <= status < 500 and status not in (401, 403, 404)
            else "authentication_error" if status == 401
            else "permission_error" if status == 403
            else "not_found_error" if status == 404 else "server_error")
    return {"error": {"message": message, "type": kind, "param": param, "code": None}}


class ResponsesRequest(BaseModel):
    """``POST /v1/responses``."""
    model_config = ConfigDict(extra="allow")
    model: Optional[str] = None
    input: Union[str, list[dict[str, Any]], None] = None
    instructions: Optional[str] = None
    tools: Optional[list[dict[str, Any]]] = None
    tool_choice: Optional[Union[str, dict[str, Any]]] = None
    parallel_tool_calls: Optional[bool] = None
    max_output_tokens: Optional[int] = Field(None, ge=1)
    temperature: Optional[float] = Field(None, allow_inf_nan=False)
    top_p: Optional[float] = Field(None, allow_inf_nan=False)
    stream: bool = False
    text: Optional[dict[str, Any]] = None
    reasoning: Optional[dict[str, Any]] = None
    previous_response_id: Optional[str] = None
    store: Optional[bool] = None
    metadata: Optional[dict[str, Any]] = None
    truncation: Optional[str] = None
    include: Optional[list[str]] = None
    user: Optional[str] = None
    background: Optional[bool] = None
    conversation: Optional[Any] = None
    prompt: Optional[Any] = None
    top_logprobs: Optional[int] = None
    max_tool_calls: Optional[int] = None


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


# ------------------------------------------------------------------ #
#  Requests                                                            #
# ------------------------------------------------------------------ #

def _content_parts(content: Any, role: str, where: str) -> Union[str, list[dict[str, Any]]]:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise ResponsesError(400, f"{where}.content must be a string or a list", f"{where}.content")
    parts: list[dict[str, Any]] = []
    for j, part in enumerate(content):
        kind = part.get("type") if isinstance(part, dict) else None
        at = f"{where}.content[{j}]"
        if kind in ("input_text", "output_text", "text"):
            parts.append({"type": "text", "text": str(part.get("text") or "")})
        elif kind == "refusal":
            parts.append({"type": "text", "text": str(part.get("refusal") or "")})
        elif kind == "input_audio" and role == "user":
            audio = part.get("input_audio")
            if not isinstance(audio, dict) or not isinstance(audio.get("data"), str):
                raise ResponsesError(400, f"{at}: an input_audio needs input_audio.data", at)
            parts.append({"type": "input_audio", "input_audio": {
                "data": audio["data"], "format": str(audio.get("format") or "wav")}})
        elif kind == "input_image" and role == "user":
            url = part.get("image_url")
            if not isinstance(url, str) or not url:
                raise ResponsesError(
                    400, f"{at}: an input_image needs image_url (a data URL or http URL); "
                         "file_id is not supported", at)
            parts.append({"type": "image_url", "image_url": {"url": url}})
        else:
            raise ResponsesError(400, f"{at}: a {kind!r} part is not supported", at)
    if len(parts) == 1 and parts[0]["type"] == "text":
        return parts[0]["text"]
    return parts


def input_to_messages(items: Union[str, list[dict[str, Any]], None]) -> list[dict[str, Any]]:
    """Responses ``input`` as OpenAI chat messages: function calls become an
    assistant message's ``tool_calls``, their outputs ``tool`` messages (images
    in an output go into one user message after the run of tool messages), and
    reasoning items are dropped."""
    if items is None:
        return []
    if isinstance(items, str):
        return [{"role": "user", "content": items}]
    out: list[dict[str, Any]] = []
    pending_images: list[dict[str, Any]] = []
    for i, item in enumerate(items):
        where = f"input[{i}]"
        if not isinstance(item, dict):
            raise ResponsesError(400, f"{where} must be an object", where)
        kind = item.get("type", "message")
        if kind == "message":
            role = item.get("role")
            if role == "developer":
                role = "system"
            if role not in ("user", "assistant", "system"):
                raise ResponsesError(
                    400, f"{where}.role must be user, assistant, system or developer",
                    f"{where}.role")
            out.append({"role": role, "content": _content_parts(item.get("content"), role, where)})
        elif kind == "function_call":
            name = item.get("name")
            call_id = item.get("call_id")
            if not isinstance(name, str) or not name or not isinstance(call_id, str) or not call_id:
                raise ResponsesError(400, f"{where}: a function_call needs name and call_id", where)
            call = {"id": call_id, "type": "function",
                    "function": {"name": name, "arguments": str(item.get("arguments") or "{}")}}
            last = out[-1] if out else None
            if last is not None and last["role"] == "assistant" and isinstance(
                    last.get("content"), str):
                last.setdefault("tool_calls", []).append(call)
            else:
                out.append({"role": "assistant", "content": "", "tool_calls": [call]})
        elif kind == "function_call_output":
            call_id = item.get("call_id")
            if not isinstance(call_id, str) or not call_id:
                raise ResponsesError(400, f"{where}: a function_call_output needs call_id", where)
            text, images = _call_output(item.get("output"), where)
            out.append({"role": "tool", "tool_call_id": call_id, "content": text})
            pending_images.extend(images)
            continue
        elif kind == "reasoning":
            continue
        else:
            raise ResponsesError(400, f"{where}: a {kind!r} item is not supported", where)
        if pending_images:
            out.insert(len(out) - 1, {"role": "user", "content": pending_images})
            pending_images = []
    if pending_images:
        out.append({"role": "user", "content": pending_images})
    return out


def _call_output(output: Any, where: str) -> tuple[str, list[dict[str, Any]]]:
    """A ``function_call_output``'s ``output`` as the tool message text and the
    image parts it carries. A string is the text; a list joins its text parts
    and returns its ``input_image`` parts; any other part is a 400."""
    if output is None or isinstance(output, str):
        return output or "", []
    if not isinstance(output, list):
        raise ResponsesError(400, f"{where}.output must be a string or a list", f"{where}.output")
    texts: list[str] = []
    images: list[dict[str, Any]] = []
    for j, part in enumerate(output):
        kind = part.get("type") if isinstance(part, dict) else None
        at = f"{where}.output[{j}]"
        if kind in ("input_text", "output_text", "text"):
            texts.append(str(part.get("text") or ""))
        elif kind == "input_image":
            url = part.get("image_url")
            if not isinstance(url, str) or not url:
                raise ResponsesError(
                    400, f"{at}: an input_image needs image_url (a data URL or http URL); "
                         "file_id is not supported", at)
            images.append({"type": "image_url", "image_url": {"url": url}})
        else:
            raise ResponsesError(400, f"{at}: a {kind!r} part is not supported", at)
    return "".join(texts), images


def tools_to_chat(tools: Optional[list[dict[str, Any]]]) -> Optional[list[dict[str, Any]]]:
    """Responses ``tools`` as chat function tools; a built-in tool is a 400."""
    if tools is None:
        return None
    out = []
    for i, tool in enumerate(tools):
        kind = tool.get("type")
        if kind != "function":
            raise ResponsesError(
                400, f"tools[{i}]: the {kind!r} tool is not supported; only function tools "
                     "are served", f"tools[{i}].type")
        fn = {"name": tool.get("name"), "parameters": tool.get("parameters") or {"type": "object"}}
        if tool.get("description"):
            fn["description"] = tool["description"]
        out.append({"type": "function", "function": fn})
    return out


def tool_choice_to_chat(choice: Any) -> Any:
    if choice is None or choice in ("auto", "none", "required"):
        return choice
    if isinstance(choice, dict) and choice.get("type") == "function":
        name = choice.get("name")
        if not isinstance(name, str) or not name:
            raise ResponsesError(400, "tool_choice.name is required", "tool_choice.name")
        return {"type": "function", "function": {"name": name}}
    raise ResponsesError(400, f"tool_choice {choice!r} is not supported", "tool_choice")


def text_format_to_chat(text: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """``text.format`` as a chat ``response_format``."""
    fmt = (text or {}).get("format")
    if fmt is None:
        return None
    if not isinstance(fmt, dict):
        raise ResponsesError(400, "text.format must be an object", "text.format")
    kind = fmt.get("type")
    if kind in ("text", "json_object"):
        return {"type": kind}
    if kind == "json_schema":
        spec = {k: fmt[k] for k in ("name", "schema", "strict", "description") if k in fmt}
        return {"type": "json_schema", "json_schema": spec}
    raise ResponsesError(400, f"text.format.type {kind!r} is not supported", "text.format.type")


def refuse_unserved(req: ResponsesRequest) -> None:
    """Raise :class:`ResponsesError` (400) for a field whose meaning localm cannot
    honour: ``background``, ``conversation``, ``prompt``, ``top_logprobs``, an
    ``include`` value outside :data:`INCLUDE_ACCEPTED`, ``text.verbosity`` other
    than ``medium``, and a ``truncation`` other than ``auto`` or ``disabled``."""
    for name, value in (("background", req.background), ("conversation", req.conversation),
                        ("prompt", req.prompt)):
        if value:
            raise ResponsesError(400, f"{name} is not supported", name)
    if req.top_logprobs:
        raise ResponsesError(400, "top_logprobs is not supported", "top_logprobs")
    for value in req.include or []:
        if value not in INCLUDE_ACCEPTED:
            raise ResponsesError(400, f"include {value!r} is not supported", "include")
    if req.truncation not in (None, "disabled", "auto"):
        raise ResponsesError(400, "truncation must be 'auto' or 'disabled'", "truncation")
    text = req.text if isinstance(req.text, dict) else {}
    if text.get("verbosity") not in (None, "medium"):
        raise ResponsesError(400, f"text.verbosity {text.get('verbosity')!r} is not supported",
                             "text.verbosity")


def ignored_fields(req: ResponsesRequest) -> list[str]:
    """The names of the request's fields that are neither Responses fields nor
    :data:`CHAT_EXTRAS`; they have no effect."""
    return sorted(k for k in (req.model_extra or {}) if k not in CHAT_EXTRAS)


def plan_chat(req: ResponsesRequest, history: list[dict[str, Any]]
              ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """``(body, conversation)``: the chat body for *req* after *history* (the
    stored conversation a ``previous_response_id`` continues, as chat messages),
    and that conversation with this request's input added (``instructions``
    left out: they do not carry over to a continuation)."""
    refuse_unserved(req)
    conversation = history + input_to_messages(req.input)
    if not any(m["role"] != "system" for m in conversation):
        raise ResponsesError(400, "input is required", "input")
    messages: list[dict[str, Any]] = []
    if req.instructions:
        messages.append({"role": "system", "content": req.instructions})
    messages += conversation
    body: dict[str, Any] = {"model": req.model, "messages": messages, "stream": req.stream}
    for src, dst in (("max_output_tokens", "max_tokens"), ("temperature", "temperature"),
                     ("top_p", "top_p"), ("parallel_tool_calls", "parallel_tool_calls")):
        value = getattr(req, src)
        if value is not None:
            body[dst] = value
    extra = req.model_extra or {}
    for key in CHAT_EXTRAS:
        if extra.get(key) is not None:
            body[key] = extra[key]
    tools = tools_to_chat(req.tools)
    if tools:
        body["tools"] = tools
    choice = tool_choice_to_chat(req.tool_choice)
    if choice is not None:
        body["tool_choice"] = choice
    fmt = text_format_to_chat(req.text)
    if fmt is not None:
        body["response_format"] = fmt
    effort = (req.reasoning or {}).get("effort")
    if isinstance(effort, str):
        body["reasoning_effort"] = effort
    return body, conversation


# ------------------------------------------------------------------ #
#  Replies                                                             #
# ------------------------------------------------------------------ #

def _usage(usage: Any) -> dict[str, Any]:
    usage = usage if isinstance(usage, dict) else {}
    prompt = int(usage.get("prompt_tokens") or 0)
    completion = int(usage.get("completion_tokens") or 0)
    return {"input_tokens": prompt, "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": completion, "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": prompt + completion}


def message_item(text: str, item_id: Optional[str] = None, status: str = "completed") -> dict:
    return {"type": "message", "id": item_id or _new_id("msg"), "status": status,
            "role": "assistant",
            "content": [{"type": "output_text", "text": text, "annotations": [], "logprobs": []}]}


def reasoning_item(text: str, item_id: Optional[str] = None) -> dict:
    return {"type": "reasoning", "id": item_id or _new_id("rs"), "summary": [],
            "content": [{"type": "reasoning_text", "text": text}]}


def call_item(call: dict[str, Any], item_id: Optional[str] = None,
              status: str = "completed") -> dict:
    fn = call.get("function") or {}
    return {"type": "function_call", "id": item_id or _new_id("fc"),
            "call_id": call.get("id") or _new_id("call"), "name": fn.get("name") or "",
            "arguments": fn.get("arguments") or "{}", "status": status}


@dataclass
class Shell:
    """The request echo every Response object carries."""
    req: ResponsesRequest
    response_id: str = field(default_factory=lambda: _new_id("resp"))
    created_at: int = field(default_factory=lambda: int(time.time()))

    def response(self, *, model: str, status: str, output: list, usage: Any = None,
                 error: Optional[dict] = None, incomplete: Optional[str] = None) -> dict:
        r = self.req
        return {
            "id": self.response_id, "object": "response", "created_at": self.created_at,
            "status": status, "error": error,
            "incomplete_details": {"reason": incomplete} if incomplete else None,
            "instructions": r.instructions, "max_output_tokens": r.max_output_tokens,
            "model": model, "output": output,
            "parallel_tool_calls": r.parallel_tool_calls is not False,
            "previous_response_id": r.previous_response_id,
            "reasoning": {"effort": (r.reasoning or {}).get("effort"), "summary": None},
            "store": r.store is not False, "temperature": r.temperature,
            "text": r.text or {"format": {"type": "text"}},
            "tool_choice": r.tool_choice or "auto", "tools": r.tools or [],
            "top_p": r.top_p, "truncation": r.truncation or "disabled",
            "usage": _usage(usage) if usage is not None else None,
            "user": r.user, "metadata": r.metadata or {},
        }


def response_from_completion(shell: Shell, data: dict[str, Any]) -> dict[str, Any]:
    """A non-streaming chat completion as a Response object."""
    choices = data.get("choices") or [{}]
    choice = choices[0] if isinstance(choices[0], dict) else {}
    message = choice.get("message") or {}
    finish = choice.get("finish_reason")
    model = str(data.get("model") or shell.req.model or "")
    text = message.get("content") or ""
    if finish == "error":
        return shell.response(model=model, status="failed", output=[],
                              usage=data.get("usage"),
                              error={"code": "server_error",
                                     "message": text.strip() or "generation failed"})
    output: list[dict[str, Any]] = []
    if message.get("reasoning_content"):
        output.append(reasoning_item(message["reasoning_content"]))
    if text:
        output.append(message_item(text, status="incomplete" if finish == "length" else "completed"))
    for call in message.get("tool_calls") or []:
        if isinstance(call, dict):
            output.append(call_item(call))
    return shell.response(model=model, status="incomplete" if finish == "length" else "completed",
                          output=output, usage=data.get("usage"),
                          incomplete="max_output_tokens" if finish == "length" else None)


class _Events:
    def __init__(self) -> None:
        self.seq = 0

    def __call__(self, kind: str, **data: Any) -> bytes:
        payload = {"type": kind, "sequence_number": self.seq, **data}
        self.seq += 1
        return f"event: {kind}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n".encode()


async def response_stream(shell: Shell, events: AsyncIterator[dict[str, Any]],
                          on_done=None) -> AsyncIterator[bytes]:
    """Translate chat-completion chunks into the Responses event stream.
    *on_done*, when given, is called with the final Response object."""
    ev = _Events()
    model = str(shell.req.model or "")
    output: list[dict[str, Any]] = []
    started = False
    cur: Optional[dict[str, Any]] = None
    finish: Optional[str] = None
    usage: Any = None
    failure: Optional[str] = None
    held: Optional[str] = None

    def start() -> list[bytes]:
        created = shell.response(model=model, status="in_progress", output=[])
        return [ev("response.created", response=created),
                ev("response.in_progress", response=created)]

    def close_current(status: str = "completed") -> list[bytes]:
        nonlocal cur
        if cur is None:
            return []
        out: list[bytes] = []
        item, idx = cur["item"], cur["index"]
        if item["type"] == "message":
            text = cur["text"]
            item = message_item(text, item["id"], status)
            out.append(ev("response.output_text.done", item_id=item["id"], output_index=idx,
                          content_index=0, text=text, logprobs=[]))
            out.append(ev("response.content_part.done", item_id=item["id"], output_index=idx,
                          content_index=0, part=item["content"][0]))
        elif item["type"] == "reasoning":
            text = cur["text"]
            item = reasoning_item(text, item["id"])
            out.append(ev("response.reasoning_text.done", item_id=item["id"], output_index=idx,
                          content_index=0, text=text))
            out.append(ev("response.content_part.done", item_id=item["id"], output_index=idx,
                          content_index=0, part=item["content"][0]))
        output.append(item)
        out.append(ev("response.output_item.done", output_index=idx, item=item))
        cur = None
        return out

    def open_item(kind: str) -> list[bytes]:
        nonlocal cur
        out = close_current()
        idx = len(output)
        if kind == "message":
            item = message_item("", status="in_progress")
            item["content"] = []
            part = {"type": "output_text", "text": "", "annotations": [], "logprobs": []}
        else:
            item = reasoning_item("")
            item["content"] = []
            part = {"type": "reasoning_text", "text": ""}
        cur = {"item": item, "index": idx, "text": ""}
        out.append(ev("response.output_item.added", output_index=idx, item=item))
        out.append(ev("response.content_part.added", item_id=item["id"], output_index=idx,
                      content_index=0, part=part))
        return out

    def text_delta(kind: str, text: str) -> list[bytes]:
        out = [] if cur is not None and cur["item"]["type"] == kind else open_item(kind)
        assert cur is not None
        cur["text"] += text
        name = "response.output_text.delta" if kind == "message" else "response.reasoning_text.delta"
        extra = {"logprobs": []} if kind == "message" else {}
        out.append(ev(name, item_id=cur["item"]["id"], output_index=cur["index"],
                      content_index=0, delta=text, **extra))
        return out

    try:
        async for chunk in events:
            if not started:
                model = str(chunk.get("model") or model)
                for line in start():
                    yield line
                started = True
            refusal = chunk.get("localm_error")
            if isinstance(refusal, dict):
                failure = str(refusal.get("detail") or "request failed")
                break
            if isinstance(chunk.get("usage"), dict):
                usage = chunk["usage"]
            choices = chunk.get("choices") or []
            choice = choices[0] if choices and isinstance(choices[0], dict) else {}
            delta = choice.get("delta") or {}
            if delta.get("reasoning_content"):
                for line in text_delta("reasoning", delta["reasoning_content"]):
                    yield line
            text = delta.get("content")
            if text:
                if held is not None:
                    for line in text_delta("message", held):
                        yield line
                    held = None
                if text.lstrip().startswith(_ERROR_TEXT_PREFIX):
                    held = text
                else:
                    for line in text_delta("message", text):
                        yield line
            for call in delta.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                for line in close_current():
                    yield line
                idx = len(output)
                item = call_item(call, status="in_progress")
                args = item["arguments"]
                yield ev("response.output_item.added", output_index=idx,
                         item={**item, "arguments": ""})
                yield ev("response.function_call_arguments.delta", item_id=item["id"],
                         output_index=idx, delta=args)
                yield ev("response.function_call_arguments.done", item_id=item["id"],
                         output_index=idx, arguments=args, name=item["name"])
                item["status"] = "completed"
                output.append(item)
                yield ev("response.output_item.done", output_index=idx, item=item)
            if choice.get("finish_reason"):
                finish = choice["finish_reason"]
                break
    finally:
        close = getattr(events, "aclose", None)
        if close is not None:
            await close()
    if not started:
        for line in start():
            yield line
    if failure is not None or finish == "error":
        message = failure or (held or "").strip() or "generation failed"
        for line in close_current("incomplete"):
            yield line
        final = shell.response(model=model, status="failed", output=output, usage=usage,
                               error={"code": "server_error", "message": message})
        yield ev("response.failed", response=final)
        return
    if held is not None:
        for line in text_delta("message", held):
            yield line
    incomplete = finish == "length"
    for line in close_current("incomplete" if incomplete else "completed"):
        yield line
    final = shell.response(model=model, status="incomplete" if incomplete else "completed",
                           output=output, usage=usage,
                           incomplete="max_output_tokens" if incomplete else None)
    if on_done is not None:
        on_done(final)
    yield ev("response.incomplete" if incomplete else "response.completed", response=final)


# ------------------------------------------------------------------ #
#  Store                                                               #
# ------------------------------------------------------------------ #

def output_messages(output: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """A Response's output items as the chat messages a continuation sees: its
    text and its function calls as one assistant message (reasoning left out)."""
    text = ""
    calls: list[dict[str, Any]] = []
    for item in output:
        if item.get("type") == "message":
            text += "".join(p.get("text") or "" for p in item.get("content") or [])
        elif item.get("type") == "function_call":
            calls.append({"id": item["call_id"], "type": "function",
                          "function": {"name": item["name"], "arguments": item["arguments"]}})
    if not text and not calls:
        return []
    message: dict[str, Any] = {"role": "assistant", "content": text}
    if calls:
        message["tool_calls"] = calls
    return [message]


@dataclass
class _Stored:
    principal: Optional[str]
    data: bytes
    expires: float


class ResponseStore:
    """The conversations of stored responses, kept in this process's memory as
    UTF-8 JSON for ``previous_response_id``: at most ``STORE_MAX_RESPONSES``
    entries and ``STORE_MAX_BYTES`` of JSON, each until ``STORE_TTL_SECONDS``
    after it was stored (an expired entry is removed at the next :meth:`put` or
    :meth:`get`), and visible only to the principal that stored it. Past a bound,
    the oldest entry of the principal holding the most entries (count bound) or
    the most bytes (byte bound) is dropped. Never written to disk. Thread-safe."""

    def __init__(self, max_items: int = STORE_MAX_RESPONSES,
                 max_bytes: int = STORE_MAX_BYTES, ttl: float = STORE_TTL_SECONDS) -> None:
        self._items: OrderedDict[str, _Stored] = OrderedDict()
        self._usage: dict[Optional[str], list[int]] = {}
        self._bytes = 0
        self._lock = threading.Lock()
        self._max_items = max_items
        self._max_bytes = max_bytes
        self._ttl = ttl

    def _drop(self, key: str) -> None:
        entry = self._items.pop(key)
        usage = self._usage[entry.principal]
        usage[0] -= 1
        usage[1] -= len(entry.data)
        if usage[0] == 0:
            del self._usage[entry.principal]
        self._bytes -= len(entry.data)

    def _expire(self, now: float) -> None:
        for key in [k for k, v in self._items.items() if v.expires <= now]:
            self._drop(key)

    def _evict(self) -> None:
        while len(self._items) > self._max_items or self._bytes > self._max_bytes:
            slot = 0 if len(self._items) > self._max_items else 1
            victim = max(self._usage, key=lambda p: self._usage[p][slot])
            self._drop(next(k for k, v in self._items.items() if v.principal == victim))

    def put(self, principal: Optional[str], response_id: str,
            conversation: list[dict[str, Any]]) -> bool:
        """Store *conversation* under *response_id* for *principal*. Returns
        whether it is kept: False when its JSON is larger than the byte bound, or
        when making room dropped it."""
        data = json.dumps(conversation, ensure_ascii=False).encode("utf-8")
        if len(data) > self._max_bytes:
            return False
        now = time.monotonic()
        with self._lock:
            self._expire(now)
            if response_id in self._items:
                self._drop(response_id)
            self._items[response_id] = _Stored(principal, data, now + self._ttl)
            usage = self._usage.setdefault(principal, [0, 0])
            usage[0] += 1
            usage[1] += len(data)
            self._bytes += len(data)
            self._evict()
            return response_id in self._items

    def get(self, principal: Optional[str], response_id: str) -> Optional[list[dict[str, Any]]]:
        """The conversation stored under *response_id* for *principal*, or None
        when there is none (unknown, expired, dropped, or another principal's)."""
        with self._lock:
            self._expire(time.monotonic())
            entry = self._items.get(response_id)
        if entry is None or entry.principal != principal:
            return None
        return json.loads(entry.data)
