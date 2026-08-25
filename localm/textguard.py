# SPDX-License-Identifier: AGPL-3.0-or-later
"""Text-guard: defang untrusted text so it cannot forge a prompt boundary."""

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
    """Defang frame markers AND chat-template control tokens in untrusted content."""
    if not text:
        return text
    text = _FRAME_RE.sub(r"&lt;\1", text)
    text = _SPECIAL_RE.sub(_defang_special, text)
    return text
