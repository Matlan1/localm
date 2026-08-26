# SPDX-License-Identifier: AGPL-3.0-or-later
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
    # True when this call was recovered ONLY via a name-gated fallback path (a
    # bare top-level JSON object, or a ```json/bare ``` fence) carrying no marker
    # that the model meant to call a tool, and accepted purely because its shape
    # matches a real tool name. False for every path that does carry such a
    # marker, however mangled: the canonical <tool_call> XML wrapper, an explicit
    # ```tool_call/```tool_code fence, and marker-variant dialects like
    # <|tool_call>. execution.py's confirmation gate keys on this flag.
    lenient: bool = False


# ---------------------------------------------------------------------------
#  Patterns
# ---------------------------------------------------------------------------

# Primary: <tool_call>...</tool_call>, matched as an OPENER paired with the next
# CLOSER rather than as one opener-body-closer regex, which keeps the scan linear
# (see _iter_xml_tool_calls).
_RE_XML_OPEN = re.compile(
    r"<tool_call(?:\s+name=['\"](?P<name_attr>[^'\"]+)['\"])?>",
    re.IGNORECASE,
)
_RE_XML_CLOSE = re.compile(r"</tool_call>", re.IGNORECASE)

# Marker-variant wrapper. Finetunes mangle the canonical <tool_call> tags in the
# wild: <|tool_call>, <|tool_call|>, closing as <tool_call|> or <|/tool_call>, an
# optional "call:NAME" prefix (sometimes the literal "call:tool_call"), and
# whitespace before the JSON. The JSON body itself is usually valid, so any
# delimiter variant is accepted and the call recovered from the body. There is no
# regex for it: the body is brace-matched instead, in _iter_marker_variant_calls.
#
# Fenced code block, matched as an OPENER paired with the next CLOSER rather than
# as one opener-body-closer regex. The optional language tag tells an explicit
# tool fence (```tool_call / ```tool_code) from an ambiguous one (```json or a
# bare ```), which is treated as a call only when its name matches a real tool
# (the tool_names gate in parse_tool_calls).
_RE_FENCE_OPEN = re.compile(r"```[ \t]*(?:(?P<lang>[A-Za-z_][\w+.-]*)[ \t]*)?\r?\n")
_RE_FENCE_CLOSE = re.compile(r"\r?\n[ \t]*```")

# Fence languages that explicitly signal a tool call (no name gate needed).
_EXPLICIT_FENCE_LANGS = frozenset({"tool_call", "tool_code", "tool"})

# Signals that the model TRIED to call a tool even when nothing parsed. Fires a
# one-shot repair turn instead of printing the broken call as the final answer.
_RE_TOOL_MARKER = re.compile(r"<\|?/?tool_call\|?>", re.IGNORECASE)
# The optional "call:NAME" prefix some finetunes put before the JSON body.
_RE_CALL_PREFIX = re.compile(r"call:(\w+)")
_RE_TOOL_FENCE = re.compile(r"```[ \t]*(?:tool_call|tool_code)\b", re.IGNORECASE)
_RE_NAME_KEY = re.compile(r"""["']name["']\s*:""")
_RE_ARGS_KEY = re.compile(r"""["'](?:args|arguments)["']\s*:""")


def looks_like_tool_attempt(text: str, tool_names: Optional[set] = None) -> bool:
    """True when *text* looks like a botched tool call - a tool-call marker or
    fence, or a JSON object carrying both a name and an args field - even
    though :func:`parse_tool_calls` recovered nothing. Lets the caller
    re-prompt for the correct format rather than show the broken call.

    When *tool_names* is supplied, ALSO recognises an XML-ish open tag naming
    one of those tools - ``<read_file``, ``<edit_file ...>`` - even with none
    of the markers above, which is the shape some finetunes emit instead of
    this project's ``<tool_call>{"name": ...}`` wrapper. A false hit costs one
    repair re-prompt, which tells the model to give its plain-text final answer
    if it did not mean to call a tool."""
    if _RE_TOOL_MARKER.search(text) or _RE_TOOL_FENCE.search(text):
        return True
    if _RE_NAME_KEY.search(text) and _RE_ARGS_KEY.search(text):
        return True
    if tool_names:
        pattern = re.compile(
            r"<\s*(?:" + "|".join(re.escape(n) for n in tool_names) + r")\b",
            re.IGNORECASE)
        if pattern.search(text):
            return True
    return False


def _detriple_quoted(s: str) -> str:
    """Convert Python triple-quoted string VALUES into valid JSON strings.

    Local models frequently emit multi-line ``content`` as a Python triple-quoted
    string (``"content": \"\"\"...\"\"\"``), which is NOT valid JSON. Each
    ``\"\"\"...\"\"\"`` run is replaced with a properly escaped JSON string.
    Best-effort: a wrong guess just fails json.loads and the caller moves on.
    """
    return re.sub(r'"""(.*?)"""',
                  lambda m: json.dumps(m.group(1)), s, flags=re.DOTALL)


def _strip_trailing_commas(s: str) -> str:
    """Remove trailing commas before } or ] (common LLM JSON mistake)."""
    return re.sub(r',(\s*[}\]])', r'\1', s)


def _quote_single_keys(s: str) -> str:
    return re.sub(r"'([^']+)':", r'"\1":', s)


def _fix_unescaped_backslashes(s: str) -> str:
    """Escapes backslashes in JSON strings that are not part of valid escape sequences.
    This helps recover Windows paths (e.g. drive-letter style paths) that models fail to escape."""
    return re.sub(r'(?<!\\)\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})', r'\\\\', s)


def _lenient_json(body: str) -> Optional[dict]:
    """
    JSON parse tolerating the mangles local finetunes actually produce:

    - literal newlines/tabs inside string values (strict=False) - models
      write multi-line file content without escaping it
    - a doubled outer brace:  call:write_file{{"path": "x"}}
    - single-quoted keys:  {'path': "x"}
    - Python triple-quoted string VALUES:  {"content": \"\"\"...\"\"\"}
    - trailing commas before } or ]
    - unescaped backslashes in Windows file paths
    """
    candidates = [body]
    if body.startswith("{{") and body.endswith("}}"):
        candidates.append(body[1:-1])
    # Each transform is a recovery layer; the last applies all of them, so a body
    # with several mangles at once still parses.
    transforms = (
        lambda s: s,
        _quote_single_keys,
        _detriple_quoted,
        _strip_trailing_commas,
        _fix_unescaped_backslashes,
        lambda s: _fix_unescaped_backslashes(_strip_trailing_commas(_detriple_quoted(_quote_single_keys(s)))),
    )
    for cand in candidates:
        for fix in transforms:
            try:
                obj = json.loads(fix(cand), strict=False)
                if isinstance(obj, dict):
                    return obj
            except (json.JSONDecodeError, ValueError, re.error):
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

    SECOND CONSUMER, outside this module: agent/context.py's
    ``_stream_hiding_tool_calls`` (the live streaming hider shared by the CLI
    and the GUI) imports and calls this directly, so it decides whether a
    name-gated fence is a real call by the same rule ``parse_tool_calls``
    uses. A change to this function's contract (return shape, what counts as
    a match) reaches that consumer too.
    """
    body = body.strip()
    obj = _lenient_json(body)
    if obj is None:
        return None

    # Full format: {"name": "...", "args": {...}}, with "arguments" accepted as an
    # alias for "args".
    if "name" in obj:
        name = obj["name"]
        args = obj.get("args")
        if args is None:
            args = obj.get("arguments", {})
        # A non-string name is a malformed call, treated like malformed JSON. An
        # unhashable one would raise TypeError at the set/dict lookups downstream.
        if not isinstance(name, str) or not isinstance(args, dict):
            return None
        return name, args

    # Args-only format: {"path": "..."} - requires name_attr
    if name_attr:
        return name_attr, obj

    return None


def _pair_delimited(text: str, opener_re, closer_re, min_body: int = 1):
    """Yield ``(opener_match, closer_match)`` for each opener paired with the next
    closer after it, left to right and non-overlapping - exactly the spans a
    single ``OPEN (?P<body>.+?) CLOSE`` regex would have matched, at linear cost.

    The second ``return`` is what keeps the scan linear: a closer search that
    fails from one opener can never succeed from a LATER one (a later opener ends
    further right, so it would search a suffix of the range that just came up
    empty), so the scan stops instead of re-walking the tail once per opener.

    ``pos = closer.end()`` reproduces finditer's non-overlapping advance.

    ``min_body`` is the minimum body length to accept, and callers do NOT agree on
    it: the two patterns behind parse_tool_calls need 1 (at least one character),
    the transcript splitter needs 0. With the wrong value a zero-length
    ``<tool_call></tool_call>`` is skipped, the opener pairs with a LATER closer,
    and everything between - prose and any real tool call - is swallowed into one
    unparseable body.
    """
    pos = 0
    while True:
        opener = opener_re.search(text, pos)
        if opener is None:
            return
        closer = closer_re.search(text, opener.end() + min_body)
        if closer is None:
            return
        yield opener, closer
        pos = closer.end()


def _iter_xml_tool_calls(text: str):
    """Yield ``(start, end, name_attr, body)`` for each ``<tool_call>`` block."""
    for opener, closer in _pair_delimited(text, _RE_XML_OPEN, _RE_XML_CLOSE):
        yield (opener.start(), closer.end(), opener.group("name_attr"),
               text[opener.end():closer.start()])


def strip_xml_tool_calls(text: str) -> tuple[list[tuple[Optional[str], str]], str]:
    """Split *text* into its ``<tool_call>`` calls and the text around them.

    Returns ``([(name_attr, body), ...], text_without_the_blocks)``. The coder's
    session transcript and the resume recap both consume those halves.

    ``min_body=0``, so an empty ``<tool_call></tool_call>`` is a match.

    ``name_attr`` is returned rather than discarded, so a caller rendering the
    stripped transcript can still name the tool.
    """
    calls: list[tuple[Optional[str], str]] = []
    kept: list[str] = []
    pos = 0
    for opener, closer in _pair_delimited(text, _RE_XML_OPEN, _RE_XML_CLOSE,
                                          min_body=0):
        calls.append((opener.group("name_attr"),
                      text[opener.end():closer.start()].strip()))
        kept.append(text[pos:opener.start()])
        pos = closer.end()
    kept.append(text[pos:])
    return calls, "".join(kept)


def _iter_fenced_blocks(text: str):
    """Yield ``(start, end, lang, body)`` for each fenced block."""
    for opener, closer in _pair_delimited(text, _RE_FENCE_OPEN, _RE_FENCE_CLOSE):
        yield (opener.start(), closer.end(), opener.group("lang"),
               text[opener.end():closer.start()])


def _object_end_from(text: str, i: int, last_close: int) -> int:
    r"""Index just past the brace-balanced ``{...}`` starting at *i*, or -1.

    Scanned LOCALLY from *i*, with string state local to this object, so an
    unmatched ``{`` or stray quote in the surrounding prose cannot void the
    braces of a later object.

    *last_close* is the index of the final ``}`` in the whole text, computed once
    by the caller. It bounds the scan, so a text containing no closing brace
    after *i* costs nothing.
    """
    if i >= len(text) or text[i] != "{" or last_close < i:
        return -1
    depth = 0
    in_str = False
    esc = False
    for j in range(i, last_close + 1):
        c = text[j]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            elif c == chr(10):
                # JSON forbids a raw newline inside a string, so this was never a
                # string: recover instead of consuming the rest of the text.
                in_str = False
        elif c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return j + 1
    return -1


def _iter_top_level_json_objects(text: str, last_close: int = None):
    """Yield ``(start, end, substring)`` for each brace-balanced top-level
    ``{...}`` region. Used to recover a bare JSON tool call written with no
    wrapper at all."""
    if last_close is None:
        last_close = text.rfind("}")
    i, n = 0, len(text)
    while i < n:
        if text[i] != "{":
            i += 1
            continue
        end = _object_end_from(text, i, last_close)
        if end < 0:
            # Unbalanced from here, so stop.
            return
        yield i, end, text[i:end]
        i = end


# Cap on how many EXPENSIVE retries one _iter_marker_variant_calls call gets. A
# failed brace-balance scan is retried from the very next marker, which is the
# only correct recovery: a scan started at a LATER marker resets its own depth to
# 0, so it can balance at a position an earlier scan never reached. Once the
# budget is spent the scan falls back to skipping past the last closing brace,
# which is fast but lossy, and bounds the cost of many never-balancing markers.
_MAX_EXPENSIVE_MARKER_RESCANS = 32


def _iter_marker_variant_calls(text: str, last_close: int = None):
    r"""Yield ``(start, end, name, body)`` for the mangled ``<|tool_call>`` dialects.

    Finetunes emit ``<|tool_call>``, ``<tool_call|>``, ``<|/tool_call>`` and an
    optional ``call:NAME`` prefix. Structured as marker -> balanced body ->
    marker rather than one regex: each marker is visited once, the body scan is
    bounded by the object itself, and because the scan is brace-balanced and
    string-aware a marker INSIDE the body (a write_file whose content contains
    ``<tool_call>``) does not terminate it.

    Also the safety net for a well-formed canonical ``<tool_call>...</tool_call>``
    that pass 1's strict opener/closer PAIRING (``_iter_xml_tool_calls``) mis-reads
    because an EARLIER, unclosed ``<tool_call>`` stole its closing tag.
    ``_RE_TOOL_MARKER`` matches plain ``<tool_call>``/``</tool_call>`` too (not
    just the ``<|...|>`` finetune dialects), so this pass gets a second,
    independent try at the later call by brace-matching FROM ITS OWN MARKER.
    """
    if last_close is None:
        last_close = text.rfind("}")
    pos = 0
    expensive_retries_left = _MAX_EXPENSIVE_MARKER_RESCANS
    while True:
        opener = _RE_TOOL_MARKER.search(text, pos)
        if opener is None:
            return
        i = opener.end()
        while i < len(text) and text[i].isspace():
            i += 1
        name = None
        prefix = _RE_CALL_PREFIX.match(text, i)
        if prefix is not None:
            name = prefix.group(1)
            i = prefix.end()
            while i < len(text) and text[i].isspace():
                i += 1
        body_end = _object_end_from(text, i, last_close)
        if body_end < 0:
            if i < len(text) and text[i] == "{" and i <= last_close:
                # A real (expensive) failed scan. Retry from the very next marker
                # until the retry budget for this call runs out, then fall back to
                # the fast skip.
                if expensive_retries_left > 0:
                    expensive_retries_left -= 1
                    pos = opener.end()
                else:
                    pos = last_close + 1
            else:
                # No object could ever start here (text[i] is not a brace, or
                # there is no closing brace left to reach), so keep looking for
                # the next marker.
                pos = opener.end()
            continue
        j = body_end
        while j < len(text) and text[j].isspace():
            j += 1
        closer = _RE_TOOL_MARKER.match(text, j)
        if closer is None:
            pos = opener.end()
            continue
        yield opener.start(), closer.end(), name, text[i:body_end]
        pos = closer.end()


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
    formats; callers that omit it get only the explicit wrappers. Every call
    recovered via one of those two name-gated formats has ``ToolCall.lenient``
    set True.
    """
    calls: list[ToolCall] = []
    seen_spans: list[tuple[int, int]] = []

    def _overlaps(start: int, end: int) -> bool:
        return any(s < end and e > start for s, e in seen_spans)

    def _accept(name: str, args: dict, raw: str, start: int, end: int,
                lenient: bool = False) -> None:
        calls.append(ToolCall(name=name, args=args, raw=raw, start=start,
                               end=end, lenient=lenient))
        seen_spans.append((start, end))

    # 1. Canonical XML wrapper
    for start, end, name_attr, body in _iter_xml_tool_calls(text):
        if _overlaps(start, end):
            continue
        parsed = _try_parse_body(body, name_attr)
        if parsed is not None:
            _accept(parsed[0], parsed[1], text[start:end], start, end)

    # 2. Fenced blocks: ```tool_call/```tool_code are explicit; ```json and a
    #    bare ``` are accepted only when the name matches a real tool.
    for start, end, lang, body in _iter_fenced_blocks(text):
        if _overlaps(start, end):
            continue
        parsed = _try_parse_body(body, None)
        if parsed is None:
            continue
        explicit = (lang or "").lower() in _EXPLICIT_FENCE_LANGS
        if explicit or (tool_names is not None and parsed[0] in tool_names):
            _accept(parsed[0], parsed[1], text[start:end], start, end,
                    lenient=not explicit)

    # The final "}" in the response, computed once and shared by passes 3 and 4:
    # it bounds every body scan, so text with no closing brace costs nothing.
    _last_close = text.rfind("}")

    # 3. Marker-variant wrappers (mangled <|tool_call> dialects)
    for start, end, prefix_name, raw_body in _iter_marker_variant_calls(text, _last_close):
        if _overlaps(start, end):
            continue
        # "call:tool_call" is wrapper noise, not a tool name
        if prefix_name and prefix_name.lower() == "tool_call":
            prefix_name = None
        body = raw_body.replace('<|"|>', '"')          # Gemma quote tokens
        parsed = _try_parse_body(body, prefix_name)
        if parsed is None and prefix_name:
            # Bare-key args form: {path: "x"} with the name in the prefix
            args = _parse_gemma_args(body)
            parsed = (prefix_name, args) if args is not None else None
        if parsed is not None:
            _accept(parsed[0], parsed[1], text[start:end], start, end)

    # 4. Bare top-level JSON object - name-gated, opt-in via tool_names
    if tool_names is not None:
        for start, end, chunk in _iter_top_level_json_objects(text, _last_close):
            if _overlaps(start, end):
                continue
            parsed = _try_parse_body(chunk, None)
            if parsed is not None and parsed[0] in tool_names:
                _accept(parsed[0], parsed[1], chunk, start, end, lenient=True)

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


def strip_tool_calls(text: str, tool_names: Optional[set] = None) -> tuple[list[ToolCall], str, int]:
    """Split *text* into every tool call parse_tool_calls recognises (all 5
    shapes) and the prose around them, plus a count of malformed leftovers.

    Returns ``(calls, clean_text, malformed_count)``. ``calls`` is exactly what
    :func:`parse_tool_calls` returns, so passing *tool_names* opts into the same
    name-gated fenced/bare-JSON forms it does; ``clean_text`` is *text* with every
    one of those spans removed via :func:`split_response`.

    ``malformed_count`` counts ``<tool_call>...</tool_call>`` blocks whose JSON
    body never parsed, so parse_tool_calls returned no call for them. After every
    parseable call is removed, a second pass with strip_xml_tool_calls (a pure
    delimiter match, no JSON validation) clears those out of the remainder and
    reports how many there were, so a caller can render a generic placeholder
    instead of raw tag soup.

    That second pass only understands the XML wrapper: a malformed marker-variant
    (``<|tool_call>...``) or a malformed explicit fence is NOT cleaned up here and
    still surfaces raw.
    """
    calls = parse_tool_calls(text, tool_names=tool_names)
    clean = "".join(s for s in split_response(text, calls) if isinstance(s, str))
    leftover, clean = strip_xml_tool_calls(clean)
    return calls, clean, len(leftover)
