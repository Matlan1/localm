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


# ---------------------------------------------------------------------------
#  Patterns
# ---------------------------------------------------------------------------

# Primary: <tool_call>…</tool_call>, matched as an OPENER paired with the next
# CLOSER rather than as one opener-body-closer regex (same shape as the fence
# below; see _iter_xml_tool_calls for why the pairing is linear).
#
# The single-regex form ``>\s*(?P<body>.+?)\s*</tool_call>`` was superlinear two
# independent ways, and BOTH had to go. All figures below are min-of-3 on this
# project's venv, quiet box.
# (1) The ``\s*`` either side of a lazy body are two ambiguous quantifiers around
#     it: CUBIC, 0.08 / 0.62 / 7.03s on 500 / 1000 / 2000 trailing spaces after a
#     bare ``<tool_call>``, i.e. a few hundred BPE tokens of hostile model output.
# (2) Independently of that, a lazy ``.+?`` scans to end-of-text whenever no
#     closer follows, and it does so from EVERY opener position. Folding the
#     whitespace into the body group does NOT fix this: that intermediate form
#     still cost 2.33s on ``'<tool_call>{{' * 5000`` and 2.21s on
#     ``'<tool_call>' * 5000``. Only the pairing fixes (2), and it takes both to
#     0.0000s.
# CodeQL alert 189 reports two witnesses for this pattern and they are wildly
# different in cost: at n=2000, ``'<tool_call>'`` + spaces is 7.03s while
# ``'<tool_call>a'`` + spaces is 0.0095s. Both are genuinely superlinear; the
# second just needs n=24,000 to become painful. Fixing only the loud one would
# have left a real alert open.
_RE_XML_OPEN = re.compile(
    r"<tool_call(?:\s+name=['\"](?P<name_attr>[^'\"]+)['\"])?>",
    re.IGNORECASE,
)
_RE_XML_CLOSE = re.compile(r"</tool_call>", re.IGNORECASE)

# Marker-variant wrapper. Finetunes mangle the canonical <tool_call> tags in
# the wild: <|tool_call>, <|tool_call|>, closing as <tool_call|> or
# <|/tool_call>, an optional "call:NAME" prefix (sometimes the literal
# "call:tool_call"), and whitespace before the JSON. The JSON body itself is
# usually valid - only the wrapper is broken - so accept any delimiter variant
# and recover the call from the body.
#
# There is deliberately NO regex for this any more. Every regex form of it was
# either quadratic (a lazy `\{.*?\}` body scans to end-of-text from every marker
# when no closing brace follows: measured 2.08s on 5,000 repetitions of
# `'<tool_call>{{'`) or, once tempered to stop at the next marker, unable to
# accept a body that legitimately CONTAINS one. The two cannot be reconciled in
# a pattern that treats the body as opaque text, so the body is brace-matched
# instead - see _iter_marker_variant_calls.
# Fenced code block, matched as an OPENER paired with the next CLOSER rather than
# as one opener-body-closer regex. The optional language tag tells an explicit
# tool fence (```tool_call / ```tool_code) from an ambiguous one (```json or a
# bare ```), which is only treated as a call when its name matches a real tool
# (the tool_names gate in parse_tool_calls).
#
# The single-regex form was superlinear two independent ways, and CodeQL alert 190
# reports one witness for each. All figures min-of-1 to min-of-3, quiet box.
# (1) Its opener was ``[ \t]*(?P<lang>...)?[ \t]*``, two adjacent ambiguous
#     quantifiers: 1.36 / 7.22 / 26.74s at 12.5k / 25k / 50k tabs after ``` and
#     95.9s at 100,000. Wrapping the trailing ``[ \t]*`` inside the lang group
#     fixes that, and is kept below.
# (2) A lazy ``.+?`` body scans to end-of-text whenever no closer follows, and it
#     does that from EVERY opener position, which is quadratic INDEPENDENTLY of
#     (1): 0.08 / 0.28 / 1.10 / 5.03s at 1k / 2k / 4k / 8k repetitions of
#     ``'```\na'``. De-ambiguating the lang group does not touch it. Only the
#     pairing below fixes it.
# Fixed, those same two inputs cost 0.0051s and 0.0009s. Tempering the body
# against ``` instead would have been a regression: a write_file call whose JSON
# content contains an inline code fence is ordinary coder traffic.
_RE_FENCE_OPEN = re.compile(r"```[ \t]*(?:(?P<lang>[A-Za-z_][\w+.-]*)[ \t]*)?\r?\n")
_RE_FENCE_CLOSE = re.compile(r"\r?\n[ \t]*```")

# Fence languages that explicitly signal a tool call (no name gate needed).
_EXPLICIT_FENCE_LANGS = frozenset({"tool_call", "tool_code", "tool"})

# Signals that the model TRIED to call a tool even when nothing parsed - used
# to fire a one-shot "reformat your tool call" repair turn instead of printing
# the broken call as the final answer.
_RE_TOOL_MARKER = re.compile(r"<\|?/?tool_call\|?>", re.IGNORECASE)
# The optional "call:NAME" prefix some finetunes put before the JSON body.
_RE_CALL_PREFIX = re.compile(r"call:(\w+)")
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


def _detriple_quoted(s: str) -> str:
    """Convert Python triple-quoted string VALUES into valid JSON strings.

    Local models frequently emit multi-line ``content`` as a Python triple-quoted
    string (``"content": \"\"\"...\"\"\"``), which is NOT valid JSON, so the call
    silently failed to parse (an empty assistant bubble, no file written). Replace
    each ``\"\"\"...\"\"\"`` run with a properly escaped JSON string. Best-effort:
    a fallback recovery, so a wrong guess just fails json.loads and we move on.
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
    - a doubled outer brace:  call:write_file{{"path": "x"}}  (seen from
      Gemma finetunes in the wild; it silently broke tool calling)
    - single-quoted keys:  {'path': "x"}
    - Python triple-quoted string VALUES:  {"content": \"\"\"...\"\"\"} (a very
      common local-model mistake that used to silently break write_file)
    - trailing commas before } or ]
    - unescaped backslashes in Windows file paths
    """
    candidates = [body]
    if body.startswith("{{") and body.endswith("}}"):
        candidates.append(body[1:-1])
    # Each transform is a recovery layer; the last applies all of them so a body
    # with several mangles at once (triple-quotes + a trailing comma) still parses.
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
        # A non-string name is a malformed call, treated like malformed JSON.
        # An unhashable one (dict/list) would even raise TypeError at the set/
        # dict lookups downstream ("in tool_names", "in disabled_tools").
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
    single ``OPEN (?P<body>.+?) CLOSE`` regex would have matched, at linear cost
    instead of quadratic.

    The cost fix is the second ``return``: a closer search that fails from one
    opener can never succeed from a LATER one (a later opener ends further right,
    so it would search a suffix of the range that just came up empty), so the
    scan stops instead of re-walking the tail once per opener. That is the whole
    defect - with a lazy ``.+?`` body, an unterminated opener costs a scan to
    end-of-text, and hostile text can plant thousands of them.

    ``pos = closer.end()`` reproduces finditer's non-overlapping advance.

    ``min_body`` is the minimum body length the replaced regex allowed, and it is
    a parameter because the callers do NOT agree on it. The two patterns behind
    parse_tool_calls used ``.+?`` (at least one character, so min_body=1), but the
    transcript splitter stands in for a regex that used ``.*?`` (min_body=0). With
    the wrong value a zero-length ``<tool_call></tool_call>`` is skipped, the
    opener pairs with a LATER closer, and everything between - prose and any real
    tool call - is swallowed into one unparseable body. Equivalence to the regexes
    these replaced is pinned by differential fuzzes in tests/test_redos_bounds.py,
    one per replaced pattern, not by inspection.
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
    session transcript and the resume recap both need those halves, and each used
    to carry its OWN copy of the regex - agent/session.py had
    ``<tool_call>\\s*(.*?)\\s*</tool_call>`` and sessions.py an inline
    ``<tool_call>.*?</tool_call>``. Both had a superlinear shape, and the
    transcript one re-ran over EVERY stored assistant message at teardown, so one
    poisoned response re-hung the Markdown export on every later close, forever.
    One implementation, one place to fix.

    ``min_body=0`` because both replaced regexes used ``.*?``: an empty
    ``<tool_call></tool_call>`` is a real match for them and must stay one here.

    ``name_attr`` is returned, not discarded: the replaced regexes did not match
    the ``name=`` attribute form at all, so those blocks used to survive into the
    transcript as raw XML that at least NAMED the tool. Now they are stripped, so
    the name has to come back out this way or the transcript renders a bare "?".
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

    Scanned LOCALLY from *i*, with string state local to this object. A global
    brace map was tried first and is wrong: the text around a tool call is model
    PROSE, and an unmatched ``{`` in prose (``use { to open``) leaves the map's
    depth non-zero, so a later stray quote enters string mode and silently voids
    every brace after it. A differential fuzz caught that - it lost calls the old
    pattern found, on exactly that shape.

    *last_close* is the index of the final ``}`` in the whole text, computed once
    by the caller. Without it, a body that never balances costs a scan to
    end-of-text FROM EVERY MARKER, which is the quadratic this whole change
    exists to remove (measured 19.3s on 5,000 repetitions of ``'<tool_call>{{'``,
    worse than the 2.08s regex it replaced). With it, a text containing no
    closing brace after *i* costs nothing at all.
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
                # string - recover rather than swallowing the rest of the text.
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
            # Unbalanced from here: the old scan-based form advanced past the
            # end of text at this point, so stopping matches it exactly.
            return
        yield i, end, text[i:end]
        i = end


def _iter_marker_variant_calls(text: str, last_close: int = None):
    r"""Yield ``(start, end, name, body)`` for the mangled ``<|tool_call>`` dialects.

    Finetunes emit ``<|tool_call>``, ``<tool_call|>``, ``<|/tool_call>`` and an
    optional ``call:NAME`` prefix. Structured as marker -> balanced body ->
    marker rather than one regex, for two reasons that pull the same way:

    COST. The regex form ``MARKER \s* (call:\w+)? \{.*?\} \s* MARKER`` scans to
    end-of-text from EVERY marker when no closing brace follows - quadratic, and
    measured at 2.08s on 5,000 repetitions of ``'<tool_call>{{'``. Here each
    marker is visited once and the body scan is bounded by the object itself.

    CORRECTNESS. The regex could not allow a marker inside the body without
    reintroducing that scan, so the shipped fix forbade it - which silently
    dropped a real case: the coder editing its own parser docs or test fixtures
    emits a write_file whose CONTENT contains ``<tool_call>``. A brace-balanced,
    string-aware scan has no such trade: it knows the marker is inside a string.
    """
    if last_close is None:
        last_close = text.rfind("}")
    pos = 0
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
            _accept(parsed[0], parsed[1], text[start:end], start, end)

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
