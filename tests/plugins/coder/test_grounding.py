# SPDX-License-Identifier: AGPL-3.0-or-later
"""The system prompt carries an explicit grounding rule: read or search the
relevant files before answering a question about the project, and base every
claim on a tool result. This holds for every model family, including the
compressed small-model prompt.
"""

from pathlib import Path

from localm.plugins.coder.prompts import build_system_prompt

CWD = Path("/tmp/proj")


def _grounded(text: str) -> bool:
    low = text.lower()
    # The prompt tells the model to read or search BEFORE answering a project
    # question, and not to answer from assumption or memory.
    return ("before answering a question about" in low
            and ("read or search" in low)
            and ("assumption" in low or "memory" in low))


def test_default_prompt_requires_grounding():
    p = build_system_prompt(CWD, model_name="llama3.1-8b")
    assert _grounded(p), "default prompt lacks a grounding-before-answering rule"


def test_thinking_prompt_requires_grounding():
    p = build_system_prompt(CWD, model_name="qwen3-8b")
    assert _grounded(p)


def test_small_prompt_requires_grounding():
    # The compressed small-model prompt still grounds answers.
    p = build_system_prompt(CWD, model_name="phi-4-mini")
    assert _grounded(p), "small prompt lacks a grounding-before-answering rule"
