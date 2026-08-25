# SPDX-License-Identifier: AGPL-3.0-or-later
"""The confirmation-handler protocol: how a tool call reaches a human for approval."""

from __future__ import annotations

import inspect
from typing import Any, Optional

# The optional keyword this module negotiates. One name, checked in one place, so
# adding a second extension later means adding it here and nowhere else.
_AGENT_KW = "agent"


def handler_accepts_agent(handler: Any) -> bool:
    """True when *handler* can be passed the ``agent=`` keyword."""
    try:
        sig = inspect.signature(handler)
    except (TypeError, ValueError):
        return False
    for param in sig.parameters.values():
        if param.kind is inspect.Parameter.VAR_KEYWORD:
            return True
        if param.name == _AGENT_KW and param.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY):
            return True
    return False


def invoke_confirm(handler: Any, call: Any, agent: Optional[str] = None) -> bool:
    """Ask *handler* to approve *call*, naming the sub-agent when it can be told."""
    if handler_accepts_agent(handler):
        return handler(call, **{_AGENT_KW: agent})
    return handler(call)
