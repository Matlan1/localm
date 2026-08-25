# SPDX-License-Identifier: AGPL-3.0-or-later
"""The estimate turn: one planning call, zero execution.

This lives here rather than in ``cli/estimate.py`` for the same reason
``verify.py`` was lifted out of ``cli/goal.py`` - estimating is no longer
CLI-only. ``cli/estimate.py`` keeps the console presentation and imports the
core from here; the GUI's ``POST /api/coder/sessions/{id}/estimate`` calls the
same core. One home for the prompt that defines what "estimate" MEANS, so the
two surfaces cannot drift into estimating different things.

Importing this module must NOT drag in click or rich, so nothing here reaches
into the ``cli`` package.
"""

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
    """One planning turn, zero execution. Returns the CLI's own JSON payload:
    ``{"estimate", "prompt_tokens", "total_tokens"}``.

    Deliberately builds its OWN message list from the agent's system prompt
    instead of appending to ``agent._messages``: an estimate is a question
    ABOUT a task, not a turn OF one, so it leaves the conversation, the turn
    counter, the token counter and the checkpoint exactly as it found them.
    That property is what makes it safe to offer mid-session in the GUI, where
    - unlike the CLI, which exits straight afterwards - the session goes on
    living and a polluted history would be carried into every later turn.
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
