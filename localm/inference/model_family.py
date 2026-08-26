# SPDX-License-Identifier: AGPL-3.0-or-later
"""Model-family detection shared across plugins.

The single source of truth for "is this a thinking/reasoning model", used by the
coder's per-family prompt tuning
(localm.plugins.coder.prompts.detect_model_family) and by regular chat's
<think> instruction. Detection keys on the model NAME it is handed, so an
opaque registry alias ("m8") resolves to not-thinking.
"""

from __future__ import annotations

# Reasoning / chain-of-thought fine-tune families plus the common descriptive
# substrings, so a model named e.g. "Llama-3.3-...-Thinking-...-High-Reasoning"
# is recognised, not just the canonical repos. Lower-cased substring match.
THINKING_MARKERS = (
    "deepseek-r1", "deepseek_r1", "qwq", "qwen3",
    "thinking", "reasoning", "-r1", "_r1", "cot", "magistral", "gemma-4",
)


def is_thinking_model(model_id: str) -> bool:
    """True when *model_id* names a thinking / reasoning model family."""
    n = (model_id or "").lower()
    return any(m in n for m in THINKING_MARKERS)
