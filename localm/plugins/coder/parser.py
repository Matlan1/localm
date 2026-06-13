"""
Parse tool calls from model response text.

Supported formats (in priority order):

1. XML wrapper with JSON body - primary:
   <tool_call>
   {"name": "read_file", "args": {"path": "src/main.py"}}
   </tool_call>

2. XML wrapper with name attr + JSON body:
   <tool_call name="read_file">
   {"path": "src/main.py"}
   </tool_call>

3. Gemma4 native format:
   <|tool_call>call:read_file{"path": "utils.py"}<tool_call|>
   Also handles <|"|> special quote tokens in args.

4. Fenced code block:
   ```tool_call        (also ```tool_code)
   {"name": "read_file", "args": {"path": "src/main.py"}}
   ```

5. Name-gated lenient forms (only when the caller passes the set of real tool
   names, and only when the parsed name is one of them - so a JSON example in
   prose is never mistaken for a call):
     - a ```json fence, or a bare ``` fence, wrapping the JSON above
     - a bare top-level JSON object with no wrapper at all:
       {"name": "read_file", "args": {"path": "src/main.py"}}

Returns a list of (tool_name, args_dict, raw_match) tuples so the caller
can reconstruct the text with results inserted in-place.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class ToolCall:
    name:  str
    args:  dict
    raw:   str      # the original matched text (for replacement)
    start: int      # char offset in the full response
    end:   int


# ---------------------------------------------------------------------------
#  Patterns
# ---------------------------------------------------------------------------

# Primary: <tool_call>…</tool_call>
_RE_XML = re.compile(
    r"<tool_call(?:\s+name=['\"](?P<name_attr>[^'\"]+)['\"])?>\s*"
    r"(?P<body>.+?)"
    r"\s*</tool_call>",
    re.DOTALL | re.IGNORECASE,
)

# Marker-variant wrapper. Finetunes mangle the canonical <tool_call> tags in
# the wild: <|tool_call>, <|tool_call|>, closing as <tool_call|> or
# <|/tool_call>, an optional "call:NAME" prefix (sometimes the literal
# "call:tool_call"), and whitespace before the JSON. The JSON body itself is
# usually valid - only the wrapper is broken - so accept any delimiter
# variant and recover the call from the body.
_RE_VARIANT = re.compile(
    r"<\|?/?tool_call\|?>\s*"
    r"(?:call:(?P<name>\w+)\s*)?"
    r"(?P<body>\{.*?\})"
    r"\s*<\|?/?tool_call\|?>",
    re.DOTALL,
)

# Fenced code block. The optional language tag tells an explicit tool fence
# (```tool_call / ```tool_code) from an ambiguous one (```json or a bare ```),
# which is only treated as a call when its name matches a real tool (the
# tool_names gate in parse_tool_calls).
_RE_FENCE = re.compile(
    r"```[ \t]*(?P<lang>[A-Za-z_][\w+.-]*)?[ \t]*\r?\n"
    r"(?P<body>.+?)"
    r"\r?\n[ \t]*```",
    re.DOTALL,
)

# Fence languages that explicitly signal a tool call (no name gate needed).
_EXPLICIT_FENCE_LANGS = frozenset({"tool_call", "tool_code", "tool"})

# Signals that the model TRIED to call a tool even when nothing parsed - used
# to fire a one-shot "reformat your tool call" repair turn instead of printing
# the broken call as the final answer.
_RE_TOOL_MARKER = re.compile(r"<\|?/?tool_call\|?>", re.IGNORECASE)
_RE_TOOL_FENCE = re.compile(r"```[ \t]*(?:tool_call|tool_code)\b", re.IGNORECASE)
_RE_NAME_KEY = re.compile(r"""["']name["']\s*:""")
_RE_ARGS_KEY = re.compile(r"""["'](?:args|arguments)["']\s*:""")


def looks_like_tool_attempt(text: str) -> bool:
    """True when *text* looks like a botched tool call - a tool-call marker or
    fence, or a JSON object carrying both a name and an args field - even
    though :func:`parse_tool_calls` recovered nothing. Lets the caller
    re-prompt for the correct format rather than show the broken call."""
    if _RE_TOOL_MARKER.search(text) or _RE_TOOL_FENCE.search(text):
        return True
    return bool(_RE_NAME_KEY.search(text) and _RE_ARGS_KEY.search(text))


def _lenient_json(body: str) -> Optional[dict]:
    """
    JSON parse tolerating the mangles local finetunes actually produce:

    - literal newlines/tabs inside string values (strict=False) - models
      write multi-line file content without escaping it
    - a doubled outer brace:  call:write_file{{"path": "x"}}  (seen from
      Gemma finetunes in the wild; it silently broke tool calling)
    - single-quoted keys:  {'path': "x"}
    """
    candidates = [body]
    if body.startswith("{{") and body.endswith("}}"):
        candidates.append(body[1:-1])
    for cand in candidates:
        for fix in (lambda s: s,
                    lambda s: re.sub(r"'([^']+)':", r'"\1":', s)):
            try:
                obj = json.loads(fix(cand), strict=False)
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                continue
    return None


def _parse_gemma_args(body: str) -> Optional[dict]:
    """
    Parse Gemma4's native tool-call argument format.

    Gemma4 may produce either standard JSON  {"key": "val"}
    or its special quote-token form  {key:<|"|>val<|"|>}.
    Both forms are handled here.
    """
    # Normalise <|"|> quote tokens → regular double-quotes
    body = body.replace('<|"|>', '"')
    body = body.strip()
    obj = _lenient_json(body)
    if obj is not None:
        return obj
    # Try bare-key form:  {key: "value", key2: 123}
    # Convert bare keys to quoted keys
    repaired = re.sub(r'(\{|,)\s*([A-Za-z_]\w*)\s*:', r'\1"\2":', body)
    return _lenient_json(repaired)


def _try_parse_body(body: str, name_attr: Optional[str]) -> Optional[tuple[str, dict]]:
    """
    Try to extract (tool_name, args) from the body text.

    Handles both:
    - Full JSON: {"name": "...", "args": {...}}
    - Args-only JSON (when name_attr is provided): {"path": "..."}
    """
    body = body.strip()
    obj = _lenient_json(body)
    if obj is None:
        return None

    # Full format: {"name": "...", "args": {...}}
    # "arguments" is accepted as an alias for "args" - models trained on the
    # OpenAI function-call schema emit that key.
    if "name" in obj:
        name = obj["name"]
        args = obj.get("args")
        if args is None:
            args = obj.get("arguments", {})
        if not isinstance(args, dict):
            return None
        return name, args

    # Args-only format: {"path": "..."} - requires name_attr
    if name_attr:
        return name_attr, obj

    return None


def _iter_top_level_json_objects(text: str):
    """Yield ``(start, end, substring)`` for each brace-balanced top-level
    ``{...}`` region. String literals are tracked so braces inside strings do
    not confuse the depth count. Used to recover a bare JSON tool call written
    with no wrapper at all."""
    i, n = 0, len(text)
    while i < n:
        if text[i] != "{":
            i += 1
            continue
        depth = 0
        in_str = False
        esc = False
        j = i
        while j < n:
            c = text[j]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            elif c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    yield i, j + 1, text[i:j + 1]
                    break
            j += 1
        i = j + 1


def parse_tool_calls(text: str, tool_names: Optional[set] = None) -> list[ToolCall]:
    """Extract all tool calls from a model response string.

    Recognised unconditionally (the wrapper itself signals intent):
      - ``<tool_call>...</tool_call>`` (with or without a ``name=`` attribute)
      - mangled ``<|tool_call>`` marker dialects from finetunes
      - ```` ```tool_call ```` / ```` ```tool_code ```` fenced blocks

    Recognised only when *tool_names* is supplied, and only when the parsed
    name is one of those tools (so a JSON example in prose is not mistaken for
    a call):
      - ```` ```json ```` (or a bare ```` ``` ````) fenced blocks
      - a bare top-level JSON object: ``{"name": "...", "args": {...}}``

    Passing *tool_names* is how the agent opts into the lenient, name-gated
    formats; callers that omit it get only the explicit wrappers (preserving
    the original behaviour).
    """
    calls: list[ToolCall] = []
    seen_spans: list[tuple[int, int]] = []

    def _overlaps(start: int, end: int) -> bool:
        return any(s < end and e > start for s, e in seen_spans)

    def _accept(name: str, args: dict, raw: str, start: int, end: int) -> None:
        calls.append(ToolCall(name=name, args=args, raw=raw, start=start, end=end))
        seen_spans.append((start, end))

    # 1. Canonical XML wrapper
    for m in _RE_XML.finditer(text):
        start, end = m.span()
        if _overlaps(start, end):
            continue
        parsed = _try_parse_body(m.group("body"), m.groupdict().get("name_attr"))
        if parsed is not None:
            _accept(parsed[0], parsed[1], m.group(0), start, end)

    # 2. Fenced blocks: ```tool_call/```tool_code are explicit; ```json and a
    #    bare ``` are accepted only when the name matches a real tool.
    for m in _RE_FENCE.finditer(text):
        start, end = m.span()
        if _overlaps(start, end):
            continue
        parsed = _try_parse_body(m.group("body"), None)
        if parsed is None:
            continue
        explicit = (m.group("lang") or "").lower() in _EXPLICIT_FENCE_LANGS
        if explicit or (tool_names is not None and parsed[0] in tool_names):
            _accept(parsed[0], parsed[1], m.group(0), start, end)

    # 3. Marker-variant wrappers (mangled <|tool_call> dialects)
    for m in _RE_VARIANT.finditer(text):
        start, end = m.span()
        if _overlaps(start, end):
            continue
        prefix_name = m.group("name")
        # "call:tool_call" is wrapper noise, not a tool name
        if prefix_name and prefix_name.lower() == "tool_call":
            prefix_name = None
        body = m.group("body").replace('<|"|>', '"')   # Gemma quote tokens
        parsed = _try_parse_body(body, prefix_name)
        if parsed is None and prefix_name:
            # Bare-key args form: {path: "x"} with the name in the prefix
            args = _parse_gemma_args(body)
            parsed = (prefix_name, args) if args is not None else None
        if parsed is not None:
            _accept(parsed[0], parsed[1], m.group(0), start, end)

    # 4. Bare top-level JSON object - name-gated, opt-in via tool_names
    if tool_names is not None:
        for start, end, chunk in _iter_top_level_json_objects(text):
            if _overlaps(start, end):
                continue
            parsed = _try_parse_body(chunk, None)
            if parsed is not None and parsed[0] in tool_names:
                _accept(parsed[0], parsed[1], chunk, start, end)

    # Sort by position in the response
    calls.sort(key=lambda c: c.start)
    return calls


def split_response(text: str, calls: list[ToolCall]) -> list[str | ToolCall]:
    """
    Split the response into alternating text and ToolCall segments.

    Returns a list where each element is either a str (plain text) or a
    ToolCall.  Useful for rendering the response in the terminal.
    """
    parts: list[str | ToolCall] = []
    pos = 0
    for call in calls:
        if call.start > pos:
            parts.append(text[pos:call.start])
        parts.append(call)
        pos = call.end
    if pos < len(text):
        parts.append(text[pos:])
    return parts
