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

The agent-memory layer (``localm/memory``) neutralises every recalled memory
before injecting it as trusted context. Coder re-exports ``neutralise`` from
here.

It BLOCKS nothing and adds no policy - it only hardens a text boundary. Apply it
ONLY to untrusted / laundering-path content (fetched pages, tool output, stored
memory), never to trusted file reads that legitimately contain these strings.
"""

from __future__ import annotations

import re

# Frame markers localm owns. Matches an opening or closing tag, tolerant of case
# and stray whitespace. Only the leading ``<`` is rewritten.
_FRAME_RE = re.compile(
    r"<((?:\s*/)?\s*(?:tool_result|untrusted_content))",
    re.IGNORECASE,
)

# Chat-template control tokens for the model families localm serves. The pipe
# delimiter matches both the ASCII bar (U+007C) and the fullwidth bar (U+FF5C).
# A pipe is required directly after "<" and before ">", so generics such as
# ``Map<string, A|B>`` do not match.
_PIPE = r"[|｜]"   # ASCII bar U+007C and fullwidth bar U+FF5C (DeepSeek)
_SPECIAL_RE = re.compile(
    r"<" + _PIPE + r"[^<>\n]{0,200}?" + _PIPE + r">"  # <|...|> incl fullwidth pipe
    r"|<\|?/?tool_call\|?>"                  # Gemma native tool-call markers
    r"|<</?SYS>>"                            # <<SYS>>  <</SYS>>  (Llama-2 / Mistral)
    r"|</?s>"                                # <s>  </s>          (Llama-2 / Mistral BOS/EOS)
    r"|<(?:start|end)_of_turn>"              # Gemma turn markers
    r"|<(?:bos|eos|pad|unk|mask|cls|sep)>"   # sentinel tokens
    # Mistral bracket control tokens, as a specific allowlist so ordinary
    # [INFO]-style log lines are left alone.
    r"|\[/?(?:INST|SYSTEM_PROMPT|AVAILABLE_TOOLS|TOOL_CALLS|TOOL_RESULTS?)\]",
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
