# SPDX-License-Identifier: AGPL-3.0-or-later
"""OpenAI-style tool calling for any chat model.

The model is told about the tools in its system prompt and asked to answer a
call as ``<tool_call>{"name": ..., "arguments": {...}}</tool_call>``; the
reply is parsed back into calls. A grammar built from the tools' JSON schemas
guarantees the call is well formed: a lazy one for ``tool_choice: "auto"`` (free
text until the model opens a ``<tool_call>``), a forced one for ``"required"``
or a named function. Earlier calls and their results in the conversation are
rendered in the same format.

Pure Python: no engine, no web framework."""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from localm.inference.gbnf import TOOL_CALL_TRIGGER, check_grammar_structure
from localm.inference.json_schema_grammar import (
    SchemaGrammarError, literal, loosen_schema, schema_to_grammar,
)
from localm.textguard import compose, untrusted_span

MAX_TOOLS = 64
MAX_TOOLS_BYTES = 65536
OPEN_TAG = "<tool_call>"
CLOSE_TAG = "</tool_call>"
_NAME = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
_RESPONSE_TAG = re.compile(r"<((?:\s*/)?\s*tool_response)", re.IGNORECASE)


class ToolsError(ValueError):
    """The tools or tool_choice of a request are malformed."""


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]


@dataclass(frozen=True)
class ToolChoice:
    kind: str                      # "auto", "none", "required" or "function"
    name: Optional[str] = None


@dataclass
class ParsedCall:
    name: str
    arguments: dict[str, Any]
    id: str = field(default_factory=lambda: "call_" + uuid.uuid4().hex[:24])
    index: int = 0

    def arguments_json(self) -> str:
        return json.dumps(self.arguments, ensure_ascii=False, separators=(",", ":"))

    def as_openai(self, with_index: bool = False) -> dict[str, Any]:
        """The call as an OpenAI ``tool_calls`` entry (with ``index`` in a stream delta)."""
        out: dict[str, Any] = {"id": self.id, "type": "function", "function": {
            "name": self.name, "arguments": self.arguments_json()}}
        if with_index:
            out = {"index": self.index, **out}
        return out

    def block(self) -> str:
        """The call written the way the model writes it."""
        return _call_text({"name": self.name, "arguments": self.arguments})


# ------------------------------------------------------------------ request


def validate_tools(tools: Any) -> list[Tool]:
    """The request's ``tools`` as :class:`Tool` objects; raises :class:`ToolsError`."""
    if tools is None:
        return []
    if not isinstance(tools, list):
        raise ToolsError("tools must be a list")
    if len(tools) > MAX_TOOLS:
        raise ToolsError(f"at most {MAX_TOOLS} tools are supported")
    out: list[Tool] = []
    for i, item in enumerate(tools):
        if not isinstance(item, dict) or item.get("type", "function") != "function":
            raise ToolsError(f"tools[{i}] must be a function tool")
        fn = item.get("function")
        if not isinstance(fn, dict):
            raise ToolsError(f"tools[{i}].function must be an object")
        name = fn.get("name")
        if not isinstance(name, str) or not _NAME.match(name):
            raise ToolsError(
                f"tools[{i}].function.name must be 1 to 64 letters, digits, '_', '-' or '.'")
        if any(t.name == name for t in out):
            raise ToolsError(f"the function name '{name}' is used twice")
        description = fn.get("description") or ""
        if not isinstance(description, str):
            raise ToolsError(f"tools[{i}].function.description must be a string")
        params = fn.get("parameters")
        if params is None:
            params = {"type": "object", "properties": {}}
        if not isinstance(params, dict):
            raise ToolsError(f"tools[{i}].function.parameters must be a JSON schema object")
        out.append(Tool(name, description, params))
    try:
        size = len(json.dumps([t.parameters for t in out]))
    except (TypeError, ValueError):
        raise ToolsError("tool parameters must be JSON") from None
    if size > MAX_TOOLS_BYTES:
        raise ToolsError(f"the tools' parameter schemas are over {MAX_TOOLS_BYTES} bytes")
    return out


def parse_tool_choice(choice: Any, tools: list[Tool]) -> ToolChoice:
    """``tool_choice`` as a :class:`ToolChoice`; raises :class:`ToolsError`."""
    if choice is None:
        return ToolChoice("auto" if tools else "none")
    if isinstance(choice, str):
        if choice not in ("auto", "none", "required"):
            raise ToolsError("tool_choice must be 'auto', 'none', 'required' or a function")
        if choice == "required" and not tools:
            raise ToolsError("tool_choice 'required' needs tools")
        return ToolChoice(choice if tools or choice == "none" else "none")
    if isinstance(choice, dict):
        fn = choice.get("function")
        name = fn.get("name") if isinstance(fn, dict) else None
        if choice.get("type", "function") != "function" or not isinstance(name, str):
            raise ToolsError("tool_choice must name a function")
        if not any(t.name == name for t in tools):
            raise ToolsError(f"tool_choice names '{name}', which is not in tools")
        return ToolChoice("function", name)
    raise ToolsError("tool_choice must be a string or an object")


# ------------------------------------------------------------------ prompt


def tools_prompt(tools: list[Tool], choice: ToolChoice) -> str:
    """The system-prompt section that describes *tools* and the call format."""
    lines = [
        "# Tools", "",
        "You may call one or more functions to assist with the user query.", "",
        "You are provided with function signatures within <tools></tools> XML tags:",
        "<tools>",
    ]
    for t in tools:
        fn: dict[str, Any] = {"name": t.name}
        if t.description:
            fn["description"] = t.description
        fn["parameters"] = t.parameters
        lines.append(json.dumps({"type": "function", "function": fn}, ensure_ascii=False))
    lines += [
        "</tools>", "",
        "For each function call, return a json object with function name and arguments "
        "within <tool_call></tool_call> XML tags:",
        "<tool_call>",
        '{"name": <function-name>, "arguments": <args-json-object>}',
        "</tool_call>",
    ]
    if choice.kind == "required":
        lines += ["", "You must call at least one of these functions; do not answer in plain text."]
    elif choice.kind == "function":
        lines += ["", f"You must call the function {choice.name}; do not answer in plain text."]
    return "\n".join(lines)


def _call_text(call: dict[str, Any]) -> str:
    inner = call.get("function")
    fn: dict[str, Any] = inner if isinstance(inner, dict) else call
    name = fn.get("name", "")
    args = fn.get("arguments", {})
    if isinstance(args, str):
        try:
            args = json.loads(args) if args.strip() else {}
        except ValueError:
            args = {"input": args}
    return OPEN_TAG + "\n" + json.dumps(
        {"name": name, "arguments": args}, ensure_ascii=False) + "\n" + CLOSE_TAG


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content
                        if isinstance(p, dict) and p.get("type") == "text")
    return "" if content is None else str(content)


def has_tool_history(messages: list[dict]) -> bool:
    return any(m.get("tool_calls") or m.get("role") == "tool" for m in messages)


def render_messages(messages: list[dict], tools: list[Tool], choice: ToolChoice) -> list[dict]:
    """*messages* with the tool description added to the system prompt and every
    earlier call and tool result written as text. Messages without tool content
    are passed through unchanged."""
    out: list[dict] = []
    for m in messages:
        role = m.get("role")
        if role == "assistant" and m.get("tool_calls"):
            text = _text_of(m.get("content"))
            blocks = "\n".join(_call_text(c) for c in m["tool_calls"])
            rendered = {k: v for k, v in m.items() if k not in ("tool_calls", "content")}
            rendered["content"] = compose(text, "\n", blocks) if text else blocks
            out.append(rendered)
        elif role == "tool":
            body = compose("<tool_response>\n",
                           untrusted_span(_RESPONSE_TAG.sub(r"&lt;\1", _text_of(m.get("content")))),
                           "\n</tool_response>")
            if out and out[-1].get("_tool_results"):
                out[-1]["content"] = compose(out[-1]["content"], "\n", body)
            else:
                out.append({"role": "user", "content": body, "origin": "tool",
                            "_tool_results": True})
        else:
            out.append(dict(m))
    for m in out:
        m.pop("_tool_results", None)
    if tools and choice.kind != "none":
        prompt = tools_prompt(tools, choice)
        if out and out[0].get("role") == "system":
            first = dict(out[0])
            existing = _text_of(first.get("content"))
            first["content"] = compose(existing, "\n\n", prompt) if existing else prompt
            out[0] = first
        else:
            out.insert(0, {"role": "system", "content": prompt})
    return out


# ------------------------------------------------------------------ grammar


def tool_grammar(tools: list[Tool], choice: ToolChoice,
                 parallel: bool = True) -> tuple[str, bool, Optional[list[str]]]:
    """``(grammar, lazy, triggers)`` that makes the model's calls well formed.

    ``auto``: a lazy grammar, unconstrained until the model writes ``<tool_call>``.
    ``required`` or a named function: the grammar applies from the first token.
    With *parallel* false the grammar admits one call.

    A parameter schema keyword the compiler cannot enforce (``pattern``, number
    bounds, ...) is left out of that tool's schema and everything else in it is
    still enforced. A schema that cannot be compiled at all, or tools whose
    grammar is over the size limit, take any JSON object as their arguments.
    Each such step is logged."""
    from localm.inference.backends.base import InvalidGrammarError

    chosen = [t for t in tools if choice.kind != "function" or t.name == choice.name]
    grammar: Optional[str] = None
    failure: Exception = ToolsError("no tools")
    for generic in (False, True):
        notes: list[str] = []
        try:
            candidate = _wrap_calls(
                schema_to_grammar(_calls_schema(chosen, generic, notes)), parallel)
            check_grammar_structure(candidate)
        except (SchemaGrammarError, InvalidGrammarError) as exc:
            failure = exc
            if not generic:
                _log_info("tools: the argument schemas gave an unusable grammar (%s); "
                          "arguments are not constrained to the schemas", exc)
            continue
        grammar = candidate
        for note in notes:
            _log_info("tools: %s", note)
        break
    if grammar is None:
        raise ToolsError(f"the tools cannot be turned into a grammar: {failure}")
    if choice.kind == "auto":
        return grammar, True, [TOOL_CALL_TRIGGER]
    return grammar, False, None


def _log_info(message: str, *args: Any) -> None:
    from localm.debuglog import logger
    logger.info(message, *args)


def _rebase_refs(node: Any, base: str) -> Any:
    """Copy of *node* with each local ``$ref`` pointing at where the schema now sits."""
    if isinstance(node, list):
        return [_rebase_refs(v, base) for v in node]
    if not isinstance(node, dict):
        return node
    out: dict[str, Any] = {}
    for key, value in node.items():
        if key == "$ref" and isinstance(value, str) and value.startswith("#"):
            out[key] = base + value[1:]
        elif key in ("enum", "const", "default", "examples"):
            out[key] = value
        else:
            out[key] = _rebase_refs(value, base)
    return out


def _calls_schema(tools: list[Tool], generic: bool, notes: list[str]) -> dict[str, Any]:
    alts = []
    for i, t in enumerate(tools):
        base = f"#/anyOf/{i}/properties/arguments" if len(tools) > 1 else "#/properties/arguments"
        params: dict[str, Any] = {"type": "object"}
        if generic:
            notes.append(f"{t.name}: arguments are not constrained to its schema")
        else:
            loosened, dropped = loosen_schema(t.parameters)
            if loosened is None:
                notes.append(f"{t.name}: its schema cannot be compiled; arguments are not "
                             f"constrained to it")
            else:
                params = _rebase_refs(loosened, base)
                if dropped:
                    notes.append(f"{t.name}: not enforced: {', '.join(dropped)}")
        alts.append({"type": "object",
                     "properties": {"name": {"const": t.name}, "arguments": params},
                     "required": ["name", "arguments"]})
    return alts[0] if len(alts) == 1 else {"anyOf": alts}


def _wrap_calls(grammar: str, parallel: bool = True) -> str:
    """*grammar* (entry rule ``root`` for one call object) as one or more
    ``<tool_call>`` blocks, or exactly one when *parallel* is false."""
    first, _, rest = grammar.partition("\n")
    body = first.removeprefix("root ::= ")
    return (
        f"root ::= tc-block{'+' if parallel else ''}\n"
        f"tc-block ::= {literal(OPEN_TAG)} tc-ws tc-call {literal(CLOSE_TAG)} tc-ws\n"
        f"tc-call ::= {body}\n"
        "tc-ws ::= [ \\t\\n\\r]? [ \\t\\n\\r]? [ \\t\\n\\r]?\n"
        + rest)


# ------------------------------------------------------------------ parsing


def _as_call(obj: Any, names: Optional[set[str]]) -> Optional[ParsedCall]:
    if not isinstance(obj, dict):
        return None
    name = obj.get("name")
    args = obj.get("arguments", obj.get("parameters", obj.get("args", {})))
    if isinstance(args, str):
        try:
            args = json.loads(args) if args.strip() else {}
        except ValueError:
            return None
    if not isinstance(name, str) or not isinstance(args, dict):
        return None
    if names is not None and name not in names:
        return None
    return ParsedCall(name, args)


def _held_prefix(text: str, tag: str) -> int:
    """Length of the longest suffix of *text* that is a proper prefix of *tag*."""
    for k in range(min(len(tag) - 1, len(text)), 0, -1):
        if text.endswith(tag[:k]):
            return k
    return 0


_MAX_BARE_VALUE = 1 << 20


class _ValueEnd:
    """Finds, a chunk at a time, where the first JSON object or array of a text
    ends; ``scan`` returns the index just past it, or ``None`` while it is open."""

    def __init__(self) -> None:
        self._pos = 0
        self._depth = 0
        self._started = False
        self._in_string = False
        self._escaped = False

    def scan(self, text: str) -> Optional[int]:
        i, n = self._pos, len(text)
        while i < n:
            ch = text[i]
            i += 1
            if self._in_string:
                if self._escaped:
                    self._escaped = False
                elif ch == "\\":
                    self._escaped = True
                elif ch == '"':
                    self._in_string = False
            elif ch == '"':
                self._in_string = True
            elif ch in "{[":
                self._started = True
                self._depth += 1
            elif ch in "}]":
                self._depth -= 1
                if self._started and self._depth <= 0:
                    self._pos = i
                    return i
        self._pos = n
        return None


class _CloseFinder:
    """Finds, a chunk at a time, the ``</tool_call>`` that ends a call whose body
    is a JSON object: a tag inside a JSON string is part of the argument, not
    the end. A body that does not start with ``{`` is searched as plain text."""

    def __init__(self) -> None:
        self._pos = 0
        self._in_string = False
        self._escaped = False

    def find(self, text: str) -> int:
        lead = text.lstrip()
        if not lead:
            return -1
        if lead[0] != "{":
            return text.find(CLOSE_TAG)
        i, n = self._pos, len(text)
        while i < n:
            ch = text[i]
            if self._in_string:
                if self._escaped:
                    self._escaped = False
                elif ch == "\\":
                    self._escaped = True
                elif ch == '"':
                    self._in_string = False
            elif ch == '"':
                self._in_string = True
            elif ch == "<":
                if text.startswith(CLOSE_TAG, i):
                    return i
                if CLOSE_TAG.startswith(text[i:]):
                    self._pos = i
                    return -1
            i += 1
        self._pos = n
        return -1


class ToolCallStream:
    """Splits a reply stream into text and tool calls.

    ``feed`` and ``finish`` return lists of events: ``("text", str)`` and
    ``("call", ParsedCall)``. Text before a ``<tool_call>`` is released at once
    (minus a possible partial tag); a block is released when its closing tag
    arrives; one left open at the end is a call if its body is valid JSON, and
    text otherwise. A JSON object or list of objects that opens the reply or
    follows a call, the way some models answer without the tags, is a call when
    each object is one; otherwise it is text, released when its closing bracket
    arrives. Calls to a function not in *names* stay text."""

    def __init__(self, names: Optional[set[str]] = None) -> None:
        self.names = names
        self._buf = ""
        self._in_call = False
        self._seen_text = False
        self._skip_ws = False
        self._bare_checked = False
        self._value_end = _ValueEnd()
        self._close = _CloseFinder()

    def _new_segment(self) -> None:
        self._bare_checked = False
        self._value_end = _ValueEnd()

    def feed(self, text: str) -> list[tuple[str, Any]]:
        events: list[tuple[str, Any]] = []
        self._buf += text
        while True:
            if self._in_call:
                end = self._close.find(self._buf)
                if end < 0:
                    return events
                body, self._buf = self._buf[:end], self._buf[end + len(CLOSE_TAG):]
                self._in_call = False
                call = self._parse_body(body)
                if call is None:
                    events.append(("text", OPEN_TAG + body + CLOSE_TAG))
                    self._seen_text = True
                else:
                    events.append(("call", call))
                    self._skip_ws = True
                self._new_segment()
                continue
            if self._skip_ws:
                self._buf = self._buf.lstrip()
                if not self._buf:
                    return events
                self._skip_ws = False
            if not self._bare_checked and not self._seen_text:
                lead = self._buf.lstrip()
                if not lead:
                    return events
                if lead[0] == "[" and not lead[1:].lstrip():
                    return events
                if lead[0] == "{" or (lead[0] == "[" and lead[1:].lstrip()[:1] == "{"):
                    end = self._value_end.scan(self._buf)
                    if end is None:
                        if len(self._buf) <= _MAX_BARE_VALUE:
                            return events
                        events.append(("text", self._buf))
                        self._buf = ""
                        self._seen_text = True
                        self._bare_checked = True
                        return events
                    raw, self._buf = self._buf[:end], self._buf[end:]
                    calls = self._parse_bare(raw)
                    if calls:
                        events.extend(("call", c) for c in calls)
                        self._skip_ws = True
                        self._new_segment()
                    else:
                        events.append(("text", raw))
                        self._seen_text = True
                        self._bare_checked = True
                    continue
                self._bare_checked = True
            start = self._buf.find(OPEN_TAG)
            if start >= 0:
                before, self._buf = self._buf[:start], self._buf[start + len(OPEN_TAG):]
                if before:
                    events.append(("text", before))
                    self._seen_text = True
                self._in_call = True
                self._close = _CloseFinder()
                continue
            hold = _held_prefix(self._buf, OPEN_TAG)
            release = self._buf[:len(self._buf) - hold]
            if release:
                events.append(("text", release))
                self._seen_text = True
            self._buf = self._buf[len(self._buf) - hold:]
            return events

    def finish(self) -> list[tuple[str, Any]]:
        events: list[tuple[str, Any]] = []
        rest, self._buf = self._buf, ""
        if self._in_call:
            call = self._parse_body(rest)
            if call is not None:
                return [("call", call)]
            return [("text", OPEN_TAG + rest)]
        if rest and not (self._skip_ws and not rest.strip()):
            events.append(("text", rest))
        return events

    def _parse_body(self, body: str) -> Optional[ParsedCall]:
        try:
            return _as_call(json.loads(body.strip()), self.names)
        except ValueError:
            return None

    def _parse_bare(self, text: str) -> list[ParsedCall]:
        try:
            data = json.loads(text.strip())
        except ValueError:
            return []
        items = data if isinstance(data, list) else [data]
        calls = [_as_call(item, self.names) for item in items]
        return [c for c in calls if c is not None] if all(calls) else []


def extract_calls(text: str, names: Optional[set[str]] = None) -> tuple[str, list[ParsedCall]]:
    """``(remaining text, calls)`` for a whole reply."""
    stream = ToolCallStream(names)
    events = stream.feed(text) + stream.finish()
    content = "".join(v for kind, v in events if kind == "text")
    return content.strip() if any(k == "call" for k, _ in events) else content, \
        [v for kind, v in events if kind == "call"]
