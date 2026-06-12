"""
Parse tool calls from model response text.

Supported formats (in priority order):

1. XML wrapper with JSON body — primary:
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

4. Fenced code block fallback:
   ```tool_call
   {"name": "read_file", "args": {"path": "src/main.py"}}
   ```

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
# usually valid — only the wrapper is broken — so accept any delimiter
# variant and recover the call from the body.
_RE_VARIANT = re.compile(
    r"<\|?/?tool_call\|?>\s*"
    r"(?:call:(?P<name>\w+)\s*)?"
    r"(?P<body>\{.*?\})"
    r"\s*<\|?/?tool_call\|?>",
    re.DOTALL,
)

# Fallback: fenced ```tool_call … ```
_RE_FENCE = re.compile(
    r"```tool_call\s*\n(?P<body>.+?)\n```",
    re.DOTALL | re.IGNORECASE,
)


def _lenient_json(body: str) -> Optional[dict]:
    """
    JSON parse tolerating the mangles local finetunes actually produce:

    - literal newlines/tabs inside string values (strict=False) — models
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
    if "name" in obj:
        name = obj["name"]
        args = obj.get("args", {})
        if not isinstance(args, dict):
            return None
        return name, args

    # Args-only format: {"path": "..."} — requires name_attr
    if name_attr:
        return name_attr, obj

    return None


def parse_tool_calls(text: str) -> list[ToolCall]:
    """Extract all tool calls from a model response string."""
    calls: list[ToolCall] = []
    seen_spans: list[tuple[int, int]] = []

    def _overlaps(start: int, end: int) -> bool:
        return any(s < end and e > start for s, e in seen_spans)

    for pattern in (_RE_XML, _RE_FENCE):
        for m in pattern.finditer(text):
            start, end = m.span()
            if _overlaps(start, end):
                continue

            body      = m.group("body")
            name_attr = m.groupdict().get("name_attr")
            parsed    = _try_parse_body(body, name_attr)
            if parsed is None:
                continue

            name, args = parsed
            calls.append(ToolCall(name=name, args=args, raw=m.group(0), start=start, end=end))
            seen_spans.append((start, end))

    # Marker-variant wrappers (mangled <|tool_call> dialects)
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
        if parsed is None:
            continue

        name, args = parsed
        calls.append(ToolCall(name=name, args=args, raw=m.group(0), start=start, end=end))
        seen_spans.append((start, end))

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
