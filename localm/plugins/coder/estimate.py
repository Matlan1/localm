# SPDX-License-Identifier: AGPL-3.0-or-later
"""The estimate turn: one planning call, zero execution."""

from __future__ import annotations

# The instruction that makes this a plan rather than a run. A module constant so
# a test can assert the no-tools clause is actually what gets sent.
ESTIMATE_PREAMBLE = (
    "ESTIMATE ONLY - do not call any tools and do not make changes.\n"
    "For the following task, reply with:\n"
    "1. A short step-by-step plan (which files you would read and change)\n"
    "2. Roughly how many agent turns you expect it to take\n"
    "3. Risks or open questions that could change the estimate\n\n"
)


def estimate_prompt(task: str) -> str:
    """The single user message an estimate turn sends."""
    return f"{ESTIMATE_PREAMBLE}Task: {task}"


def estimate_task(agent, task: str) -> dict:
    """One planning turn, zero execution."""
    messages = [
        {"role": "system", "content": agent._system_prompt},
        {"role": "user", "content": estimate_prompt(task)},
    ]
    response = agent.backend.chat(messages, **agent.gen_kwargs)
    usage = getattr(agent.backend, "last_usage", {}) or {}
    return {
        "estimate": response,
        "prompt_tokens": usage.get("prompt_tokens"),
        "total_tokens": usage.get("total_tokens"),
    }
