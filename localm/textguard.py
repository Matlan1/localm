# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Text-guard: defang untrusted text so it cannot forge a prompt boundary.

``neutralise()`` escapes the LEADING delimiter of two dangerous token classes so
the literal token no longer exists while the text stays human-readable:

  1. localm frame markers (``<tool_result>``, ``<untrusted_content>``) - stops a
     body from ending / forging the fence it is wrapped in.
  2. chat-template CONTROL TOKENS (``<|im_start|>``, ``</s>``, ``[INST]``,
     ``<start_of_turn>`` ...) - stops ROLE forgery via the model's own
     delimiters, which both backends parse as special tokens.

This was originally the coder's indirect-prompt-injection defense
(``localm/plugins/coder/provenance.py``). It is HOISTED here because more than
one KERNEL consumer needs it now: the agent-memory layer (``localm/memory``)
neutralises every recalled memory before injecting it as trusted context, and a
kernel library importing from a *plugin* (coder) would be backwards - a plugin
may be disabled, and coder will later depend on memory, not the reverse. Coder
re-exports ``neutralise`` from here so its existing call sites and tests are
unchanged; the escaping is byte-for-byte identical to the original.

It BLOCKS nothing and adds no policy - it only hardens a text boundary. Apply it
ONLY to untrusted / laundering-path content (fetched pages, tool output, stored
memory), never to trusted file reads that legitimately contain these strings.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

# Frame markers localm owns. The body of untrusted content must not be able to
# contain a literal one (or it could end / forge the frame). Match an opening or
# closing tag, tolerant of case and stray whitespace (``</ tool_result >``,
# ``<TOOL_RESULT>`` ...). Only the leading ``<`` is rewritten, so the rest of the
# text stays legible to the model.
# The whitespace tolerance is DE-AMBIGUATED, not bounded. ``\s*/?\s*`` is two
# adjacent unbounded quantifiers whenever ``/?`` matches empty, and both can claim
# the same whitespace, so a hostile ``'<' + ' ' * n`` costs O(n^2): measured 0.49s
# / 6.19s / 46.6s at 5,000 / 20,000 / 60,000 spaces (a repeat of the 60,000 case
# under the box-wide lock, on a box loaded to 100% CPU by other work, read 66.3s -
# both are upper bounds and the conclusion is the same), and 60,000 is exactly what
# POST /api/web/fetch accepts via max_chars, on remote-fetched page text.
# Moving the slash INSIDE the optional group removes the ambiguity: when the group
# does not participate there is only ONE ``\s*``, so there is only one way to match
# a whitespace run. Same language as the original (0 divergences over 200,000
# adversarial strings), and linear - the same 60,000-space input costs 0.0026s.
#
# Deliberately NOT bounded (an earlier revision used ``\s{0,8}``). This is an
# anti-evasion control: nothing in the codebase parses the closing fence, so its
# only consumer is the MODEL, a fuzzy reader. Any finite bound hands an attacker a
# trivial bypass by typing one more space, and here the bound bought nothing -
# the unbounded form is equally linear. ``_SPECIAL_RE`` below is bounded
# ({0,200}?) because it matches a token whose length is genuinely bounded, which
# is a different situation.
_FRAME_RE = re.compile(
    r"<((?:\s*/)?\s*(?:tool_result|untrusted_content))",
    re.IGNORECASE,
)

# Chat-template CONTROL TOKENS for the model families localm serves. Both backends
# tokenise the templated prompt with special-token parsing ON (GGUF llama_tokenize
# parse_special=True; HF tokenizer without split_special_tokens), so a literal
# control token in an untrusted body is parsed as a REAL role delimiter and can
# forge a system/assistant turn. We defang the leading delimiter so the byte
# sequence no longer matches the tokenizer's special-token trie, keeping the text
# legible. Best-effort and family-aware, covering ChatML, Llama-2/3, Mistral,
# Gemma, Qwen, Phi, GPT-style, EXAONE and GLM markers; the general fix is
# tokenising untrusted spans with special parsing off (a backend-level change).
# The pipe delimiter is matched as a CLASS of the ASCII bar (U+007C) and the
# FULLWIDTH bar (U+FF5C) that DeepSeek-R1 uses (the fullwidth <|Assistant|>).
# Requiring a pipe right after "<" and right before ">" precisely targets the
# <|...|> family (ChatML, Llama-3, Qwen, GPT, Phi, Cohere, DeepSeek) WITHOUT
# matching generics like Map<string, A|B>. An exotic pipe confusable or a
# non-pipe special token of a future family would need adding here.
_PIPE = r"[|｜]"   # ASCII bar U+007C and fullwidth bar U+FF5C (DeepSeek)
_SPECIAL_RE = re.compile(
    r"<" + _PIPE + r"[^<>\n]{0,200}?" + _PIPE + r">"  # <|...|> incl fullwidth pipe
    r"|<\|?/?tool_call\|?>"                  # Gemma native tool-call markers
    r"|<</?SYS>>"                            # <<SYS>>  <</SYS>>  (Llama-2 / Mistral)
    r"|</?s>"                                # <s>  </s>          (Llama-2 / Mistral BOS/EOS)
    r"|<(?:start|end)_of_turn>"              # Gemma turn markers
    r"|<(?:sop|eop)>"                        # GLM / ChatGLM turn-prefix markers
    r"|<(?:bos|eos|pad|unk|mask|cls|sep)>"   # sentinel tokens
    # Bracket control tokens. Each is an allowlist of literal role/tool names, so
    # ordinary [INFO]-style log lines and OCaml [|array|] literals are left alone.
    # Mistral: a forged [TOOL_CALLS] / [AVAILABLE_TOOLS] can fake a tool call.
    r"|\[/?(?:INST|SYSTEM_PROMPT|AVAILABLE_TOOLS|TOOL_CALLS|TOOL_RESULTS?)\]"
    # EXAONE role delimiters, the bracket-pipe counterpart of the <|...|> family.
    r"|\[\|(?:system|user|assistant|tool|endofturn)\|\]"
    # GLM / ChatGLM mask tokens, which open a templated conversation.
    r"|\[[gs]?MASK\]",
    re.IGNORECASE,
)


def _defang_special(m: "re.Match") -> str:
    """Escape the leading delimiter of a matched control token so it is inert."""
    s = m.group(0)
    if s.startswith("["):
        return "&#91;" + s[1:]
    return "&lt;" + s[1:]


def neutralise(text: str) -> str:
    """Defang frame markers AND chat-template control tokens in untrusted content.

    Two passes, both escaping only the leading delimiter so the literal token no
    longer exists while the text stays readable:
      1. tool_result / untrusted_content frame tags  (``</tool_result>`` ->
         ``&lt;/tool_result>``) - stops textual fence forgery / escape.
      2. model control tokens (``<|im_start|>``, ``</s>``, ``[INST]``,
         ``<start_of_turn>`` ...) - stops ROLE forgery via the model's real
         delimiters, which both backends would otherwise parse as special tokens.
    Ordinary ``<`` in fetched code (``a < b``, ``vector<int>``) is left alone, and
    existing ``&lt;`` is untouched. Apply only to untrusted / laundering-path
    content, never to trusted file reads.
    """
    if not text:
        return text
    text = _FRAME_RE.sub(r"&lt;\1", text)
    text = _SPECIAL_RE.sub(_defang_special, text)
    return text


def _normalise_spans(spans, length: int) -> Tuple[Tuple[int, int], ...]:
    """Clamp *spans* to ``[0, length]``, drop empties, sort and merge overlaps."""
    cleaned = []
    for item in spans or ():
        start, end = int(item[0]), int(item[1])
        start = max(0, min(start, length))
        end = max(0, min(end, length))
        if end > start:
            cleaned.append((start, end))
    cleaned.sort()
    merged: List[Tuple[int, int]] = []
    for start, end in cleaned:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return tuple(merged)


class GuardedText(str):
    """A ``str`` that records which of its character ranges came from untrusted input.

    It IS a ``str``: every existing consumer keeps working unchanged.
    ``untrusted_spans`` holds non-overlapping ``(start, end)`` character offsets
    into this string, ascending, marking text that a backend must tokenise with
    special-token parsing OFF.

    The annotation is dropped by anything that produces a plain ``str`` (an
    f-string, ``+``, ``.strip()``, a JSON round trip). A consumer that finds no
    annotation must read that as "no ranges are known", which degrades to the
    text-level ``neutralise()`` defence, never as "this text is safe to parse".
    Build annotated text with :func:`compose`, which is why an f-string is not
    used at a converted call site.
    """

    __slots__ = ("untrusted_spans",)

    def __new__(cls, text: str = "", untrusted_spans=()) -> "GuardedText":
        obj = super().__new__(cls, text)
        obj.untrusted_spans = _normalise_spans(untrusted_spans, len(obj))
        return obj


class _Untrusted:
    """A ``compose()`` part whose text is untrusted. Built by :func:`untrusted_span`."""

    __slots__ = ("text",)

    def __init__(self, text: str) -> None:
        self.text = text


def untrusted_span(text) -> _Untrusted:
    """Mark *text* as untrusted for :func:`compose`, defanging it via ``neutralise()``.

    ``neutralise()`` is applied here so that marking a span untrusted and
    defanging it cannot drift apart. It is idempotent, so a call site that
    already neutralised its text may pass the result in unchanged.
    """
    return _Untrusted(neutralise("" if text is None else str(text)))


def compose(*parts) -> GuardedText:
    """Concatenate *parts* into one :class:`GuardedText`, recording untrusted ranges.

    A plain ``str`` part is trusted. A part from :func:`untrusted_span` is recorded as
    an untrusted range. A :class:`GuardedText` part contributes its own ranges,
    shifted into the result, so composed blocks nest.
    """
    chunks: List[str] = []
    spans: List[Tuple[int, int]] = []
    pos = 0
    for part in parts:
        if isinstance(part, _Untrusted):
            piece = part.text
            if piece:
                spans.append((pos, pos + len(piece)))
        elif isinstance(part, GuardedText):
            piece = str(part)
            spans.extend((pos + a, pos + b) for a, b in part.untrusted_spans)
        elif part is None:
            piece = ""
        else:
            piece = str(part)
        chunks.append(piece)
        pos += len(piece)
    return GuardedText("".join(chunks), spans)


def compose_join(separator: str, parts) -> GuardedText:
    """``separator.join(parts)`` as a :func:`compose`, preserving untrusted ranges."""
    interleaved: List = []
    for i, part in enumerate(parts):
        if i:
            interleaved.append(separator)
        interleaved.append(part)
    return compose(*interleaved)


def untrusted_spans_of(value) -> Tuple[Tuple[int, int], ...]:
    """The untrusted ranges recorded on *value*, or ``()`` when it carries none."""
    spans = getattr(value, "untrusted_spans", ())
    if not isinstance(spans, tuple):
        return ()
    return spans


# Private-use bracketing keeps a probe sentinel out of any real chat template's
# own vocabulary; the uuid4 nonce keeps it unguessable by message content.
_SENTINEL_OPEN = "\ue000"
_SENTINEL_CLOSE = "\ue001"


def content_spans_via_sentinels(contents, render, rendered) -> "Optional[List[Tuple[int, int]]]":
    """Locate each of *contents* inside *rendered*, or return ``None``.

    *render* takes a list of replacement contents and returns the template's
    output for them. This calls it once with a unique sentinel per content,
    substitutes the real contents back into that skeleton, and requires the
    result to EQUAL *rendered*. Offsets are only returned when that holds, so a
    template that trims, escapes, reorders, drops or duplicates content yields
    ``None`` instead of a wrong offset, and nothing is ever searched for inside
    the rendered output.
    """
    import uuid

    nonce = uuid.uuid4().hex
    sentinels = [
        _SENTINEL_OPEN + str(i) + "-" + nonce + _SENTINEL_CLOSE
        for i in range(len(contents))
    ]
    try:
        skeleton = render(sentinels)
    except Exception:
        return None
    if not isinstance(skeleton, str):
        return None

    rebuilt: List[str] = []
    spans: List[Tuple[int, int]] = []
    out_len = 0
    pos = 0
    for sentinel, content in zip(sentinels, contents):
        found = skeleton.find(sentinel, pos)
        if found < 0:
            return None
        wrapper = skeleton[pos:found]
        rebuilt.append(wrapper)
        out_len += len(wrapper)
        spans.append((out_len, out_len + len(content)))
        rebuilt.append(content)
        out_len += len(content)
        pos = found + len(sentinel)
    rebuilt.append(skeleton[pos:])

    if "".join(rebuilt) != rendered:
        return None
    return spans


def map_untrusted_ranges(content_spans, per_content_spans) -> Tuple[Tuple[int, int], ...]:
    """Shift each content's own untrusted ranges into rendered-text coordinates."""
    ranges: List[Tuple[int, int]] = []
    for (start, _end), local in zip(content_spans, per_content_spans):
        ranges.extend((start + a, start + b) for a, b in local)
    return tuple(ranges)


def slice_guarded(value, start: int, end: int) -> GuardedText:
    """``value[start:end]`` as a :class:`GuardedText`, with its ranges remapped.

    Ranges that only partly overlap the slice are clipped to it, so a call site
    that truncates annotated text keeps the annotation over whatever survives
    instead of dropping it the way plain slicing does.
    """
    text = str(value)
    length = len(text)
    start = max(0, min(start, length))
    end = max(start, min(end, length))
    kept = [
        (max(a, start) - start, min(b, end) - start)
        for a, b in untrusted_spans_of(value)
        if min(b, end) > max(a, start)
    ]
    return GuardedText(text[start:end], kept)


def split_by_trust(text: str, spans) -> List[Tuple[str, bool]]:
    """Split *text* into ``(segment, is_untrusted)`` pairs that concatenate back to it."""
    out: List[Tuple[str, bool]] = []
    pos = 0
    for start, end in _normalise_spans(spans, len(text)):
        if start > pos:
            out.append((text[pos:start], False))
        out.append((text[start:end], True))
        pos = end
    if pos < len(text) or not out:
        out.append((text[pos:], False))
    return out
