# SPDX-License-Identifier: AGPL-3.0-or-later
"""The estimate turn: one planning call, zero execution.

``cli/estimate.py`` keeps the console presentation and imports the core from
here; the GUI's ``POST /api/coder/sessions/{id}/estimate`` calls the same core,
so the prompt that defines what "estimate" MEANS has one home.

Importing this module must NOT drag in click or rich, so nothing here reaches
into the ``cli`` package.
"""

from __future__ import annotations

# The instruction that makes this a plan rather than a run.
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
    """One planning turn, zero execution. Returns the CLI's own JSON payload:
    ``{"estimate", "prompt_tokens", "total_tokens"}``.

    Builds its OWN message list from the agent's system prompt instead of
    appending to ``agent._messages``, so the conversation, the turn counter,
    the token counter and the checkpoint are all left exactly as they were and
    the call is safe to make mid-session.
    """
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
