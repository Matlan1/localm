# SPDX-License-Identifier: AGPL-3.0-or-later
"""Estimate mode: the console presentation of one planning turn."""

from __future__ import annotations

import sys

from ..agent import Agent
from ..display import console, print_info
from ..estimate import estimate_task


def _run_estimate(agent: Agent, task: str, output_format: str) -> None:
    """One planning turn, zero execution."""
    result = estimate_task(agent, task)

    if output_format == "json":
        import json as _json
        sys.stdout.write(_json.dumps(result, indent=2) + "\n")
        return

    console.print(result["estimate"])
    if result.get("total_tokens"):
        print_info(
            f"Planning turn used {result['total_tokens']} tokens "
            f"({result.get('prompt_tokens') or '?'} prompt). A real run pays "
            "roughly the prompt cost on every turn."
        )
