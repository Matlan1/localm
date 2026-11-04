# SPDX-License-Identifier: AGPL-3.0-or-later
"""Model-family detection shared across plugins.

The single source of truth for "is this a thinking/reasoning model", used by the
coder's per-family prompt tuning
(localm.plugins.coder.prompts.detect_model_family) and by regular chat's
<think> instruction. Detection keys on the model NAME it is handed, so an
opaque registry alias ("m8") resolves to not-thinking. A renamed model is detected
by the first name it had, so a rename does not change how it is treated.
"""

from __future__ import annotations

# Reasoning / chain-of-thought fine-tune families plus the common descriptive
# substrings, so a model named e.g. "Llama-3.3-...-Thinking-...-High-Reasoning"
# is recognised, not just the canonical repos. Lower-cased substring match.
THINKING_MARKERS = (
    "deepseek-r1", "deepseek_r1", "qwq", "qwen3",
    "thinking", "reasoning", "-r1", "_r1", "cot", "magistral", "gemma-4",
)


def family_text(model_id: str) -> str:
    """The text to detect a family from: *model_id*, with its first word replaced
    by the first name that registered model had when it was renamed. *model_id* may
    be a bare name or a name followed by a space and its source."""
    from localm.config import model_index
    head, sep, rest = (model_id or "").partition(" ")
    first = model_index().first_name.get(head)
    return f"{first}{sep}{rest}" if first else (model_id or "")


def is_thinking_model(model_id: str) -> bool:
    """True when *model_id* names a thinking / reasoning model family."""
    n = family_text(model_id).lower()
    return any(m in n for m in THINKING_MARKERS)
