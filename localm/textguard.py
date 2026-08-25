# SPDX-License-Identifier: AGPL-3.0-or-later
"""Text-guard: defang untrusted text so it cannot forge a prompt boundary."""

from __future__ import annotations

import re

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
# Gemma, Qwen, Phi, and GPT-style markers; the general fix is tokenising untrusted
# spans with special parsing off (a backend-level change).
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
    r"|<(?:bos|eos|pad|unk|mask|cls|sep)>"   # sentinel tokens
    # Mistral bracket control tokens (a forged [TOOL_CALLS] / [AVAILABLE_TOOLS]
    # can fake a tool call); kept to a specific allowlist so ordinary [INFO]-style
    # log lines are left alone.
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
