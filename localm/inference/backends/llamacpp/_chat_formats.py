# SPDX-License-Identifier: AGPL-3.0-or-later
"""Chat prompt formats localm renders itself for template families that
llama.cpp's built-in ``llama_chat_apply_template`` matcher does not recognise.

Each renderer is a plain string builder over ``(role, text)`` pairs: the
embedded Jinja template is only inspected for its family signature and variant
markers, never executed.
"""
from __future__ import annotations

from typing import Optional, Sequence

# Characters trimmed from both ends of a message, the same set llama.cpp's
# built-in formatters trim (C isspace()).
_TRIM_CHARS = " \t\n\r\v\f"

_G4_TURN_OPEN = "<|turn>"
_G4_TURN_CLOSE = "<turn|>\n"
_G4_CHANNEL_OPEN = "<|channel>"
_G4_CHANNEL_CLOSE = "<channel|>"
_G4_EMPTY_THOUGHT = "<|channel>thought\n<channel|>"


def is_gemma4_template(template: Optional[str]) -> bool:
    """True when *template* (a chat template's Jinja source) is a Gemma 4
    template: it uses the ``<|turn>`` / ``<turn|>`` turn markers and the
    ``<|tool_call>call:`` tool-call opener."""
    if not template:
        return False
    return ("<|turn>" in template and "<turn|>" in template
            and "<|tool_call>call:" in template)


def _gemma4_strip_thinking(text: str) -> str:
    """*text* without its ``<|channel>...<channel|>`` blocks, trimmed. Text
    after an unclosed ``<|channel>`` is dropped."""
    kept = []
    for part in text.split(_G4_CHANNEL_CLOSE):
        if _G4_CHANNEL_OPEN in part:
            kept.append(part.split(_G4_CHANNEL_OPEN)[0])
        else:
            kept.append(part)
    return "".join(kept).strip(_TRIM_CHARS)


def format_gemma4(messages: Sequence[tuple[str, str]], template: str) -> str:
    """Render *messages*, ``(role, text)`` pairs, as a Gemma 4 prompt with the
    generation prompt appended and thinking off, as the official Gemma 4 chat
    template renders a conversation without tools.

    - The prompt starts with ``<bos>``.
    - A first message with role ``system`` or ``developer`` becomes the
      ``<|turn>system`` turn.
    - Every other message is a ``<|turn>{role}`` turn closed by ``<turn|>``,
      ``assistant`` written as ``model``. Its text is trimmed; the text of a
      ``model`` turn also loses its ``<|channel>...<channel|>`` blocks.
    - Consecutive assistant messages follow *template*: when it defines
      ``continue_same_model_turn``, a ``model`` message directly after an
      assistant message opens no new turn; when it also defines
      ``continues_into_next``, a ``model`` turn directly followed by an
      assistant message is not closed. Otherwise each is a turn of its own.
    - A ``tool`` message is a ``<|turn>tool`` turn of its own.
    - The prompt ends ``<|turn>model\\n``, followed by an empty thought block
      ``<|channel>thought\\n<channel|>`` when *template* emits one for a reply
      with thinking off.

    Trimming removes ASCII whitespace only (``_TRIM_CHARS``).
    """
    continue_open = "continue_same_model_turn" in template
    leave_open = "continues_into_next" in template
    empty_thought = ("<|channel>thought\\n<channel|>" in template
                     or _G4_EMPTY_THOUGHT in template)

    out = ["<bos>"]
    loop = list(messages)
    if loop and loop[0][0] in ("system", "developer"):
        out.append(_G4_TURN_OPEN + "system\n")
        out.append(loop[0][1].strip(_TRIM_CHARS))
        out.append(_G4_TURN_CLOSE)
        loop = loop[1:]

    prev_role = None
    for i, (role, text) in enumerate(loop):
        turn_role = "model" if role == "assistant" else role
        next_role = loop[i + 1][0] if i + 1 < len(loop) else None
        is_model = turn_role == "model"
        if not (continue_open and is_model and prev_role == "assistant"):
            out.append(_G4_TURN_OPEN + turn_role + "\n")
        if is_model:
            out.append(_gemma4_strip_thinking(text))
        else:
            out.append(text.strip(_TRIM_CHARS))
        if not (leave_open and is_model and next_role == "assistant"):
            out.append(_G4_TURN_CLOSE)
        prev_role = role

    out.append(_G4_TURN_OPEN + "model\n")
    if empty_thought:
        out.append(_G4_EMPTY_THOUGHT)
    return "".join(out)


def render_known_family(template: Optional[str],
                        messages: Sequence[tuple[str, str]]) -> Optional[str]:
    """The prompt for *messages* when *template* belongs to a family rendered
    here, else ``None``."""
    if is_gemma4_template(template):
        return format_gemma4(messages, template or "")
    return None
