# SPDX-License-Identifier: AGPL-3.0-or-later
"""Estimate mode: one planning turn, zero execution."""

from __future__ import annotations

import sys

from ..agent import Agent
from ..display import console, print_info

def _run_estimate(agent: Agent, task: str, output_format: str) -> None:
    """
    One planning turn, zero execution.

    Sends the task with an instruction to produce a plan and effort estimate
    instead of tool calls, prints the result, and reports the prompt-side
    token cost so the user knows what a real run starts from.
    """
    prompt = (
        "ESTIMATE ONLY - do not call any tools and do not make changes.\n"
        "For the following task, reply with:\n"
        "1. A short step-by-step plan (which files you would read and change)\n"
        "2. Roughly how many agent turns you expect it to take\n"
        "3. Risks or open questions that could change the estimate\n\n"
        f"Task: {task}"
    )
    messages = [
        {"role": "system", "content": agent._system_prompt},
        {"role": "user", "content": prompt},
    ]
    response = agent.backend.chat(messages, **agent.gen_kwargs)
    usage = getattr(agent.backend, "last_usage", {}) or {}

    if output_format == "json":
        import json as _json
        sys.stdout.write(_json.dumps({
            "estimate": response,
            "prompt_tokens": usage.get("prompt_tokens"),
            "total_tokens": usage.get("total_tokens"),
        }, indent=2) + "\n")
        return

    console.print(response)
    if usage.get("total_tokens"):
        print_info(
            f"Planning turn used {usage['total_tokens']} tokens "
            f"({usage.get('prompt_tokens', '?')} prompt). A real run pays "
            "roughly the prompt cost on every turn."
        )
