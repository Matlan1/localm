# SPDX-License-Identifier: AGPL-3.0-or-later
"""The confirmation-handler protocol: how a tool call reaches a human for approval.

A ``confirm_handler`` is ``Callable[[ToolCall], bool]``. The terminal REPL, the GUI
session (``sessions.CoderSession._confirm``), and anything a third party wires into
``Agent(confirm_handler=...)`` all implement it, so it is a PUBLIC surface.

The protocol grows only by OPTIONAL keywords, negotiated per handler:

    def handler(call) -> bool:                     # the original, still valid
    def handler(call, agent=None) -> bool:         # opts in to the extension
    def handler(call, **kw) -> bool:               # also opts in

``agent`` is the label of the sub-agent whose tool call this is, or None when the
top-level agent is asking for itself.

Negotiation is by signature inspection.
"""

from __future__ import annotations

import inspect
from typing import Any, Optional

# The optional keyword this module negotiates. One name, checked in one place.
_AGENT_KW = "agent"


def handler_accepts_agent(handler: Any) -> bool:
    """True when *handler* can be passed the ``agent=`` keyword.

    A builtin or C callable has no introspectable signature and is treated as
    the original one-argument protocol.
    """
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
    """Ask *handler* to approve *call*, naming the sub-agent when it can be told.

    Returns the handler's answer unchanged; no result is invented here and no
    default is substituted. A handler that raises propagates - callers treat an
    exception as a failed tool, not as an approval.

    *agent* is the asking sub-agent's label, or None for the top-level agent. A
    handler that opts in is passed the keyword ALWAYS, including when the value is
    None, so the call shape depends only on the handler's signature and
    ``def handler(call, agent)`` (no default) works on every confirmation.

    Inspection runs per call and is never cached.
    """
    if handler_accepts_agent(handler):
        return handler(call, **{_AGENT_KW: agent})
    return handler(call)
