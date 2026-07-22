# SPDX-License-Identifier: AGPL-3.0-or-later
"""The confirmation-handler protocol: how a tool call reaches a human for approval.

A ``confirm_handler`` is ``Callable[[ToolCall], bool]``. The terminal REPL, the GUI
session (``sessions.CoderSession._confirm``), and anything a third party wires into
``Agent(confirm_handler=...)`` all implement it. It is therefore a PUBLIC surface:
the coder's ``Agent`` is importable, and a plugin or an embedder can pass its own
callable. Changing its signature outright would break every such caller silently -
a handler that raises ``TypeError`` mid-run is a denied tool at best and, if a
caller retried, a doubly-prompted human at worst.

So the protocol grows only by OPTIONAL keywords, negotiated per handler:

    def handler(call) -> bool:                     # the original, still valid
    def handler(call, agent=None) -> bool:         # opts in to the extension
    def handler(call, **kw) -> bool:               # also opts in

``agent`` is the label of the sub-agent whose tool call this is, or None when the
top-level agent is asking for itself. Serialised parallel dispatch (tools/parallel.py)
already printed that attribution to the terminal; a GUI user saw a serialised but
ANONYMOUS approval card and could not tell which child was asking, because the label
never crossed into the event the browser renders. This is the seam that carries it.

Negotiation is by signature inspection, deliberately NOT by
``try: handler(call, agent=x) except TypeError: handler(call)``. A ``TypeError``
raised INSIDE a handler (a GUI session touching a bad attribute, say) is
indistinguishable from a signature mismatch at the call site, so the fallback would
re-invoke a handler that had already prompted - asking a human the same question
twice and taking the second answer. Inspection cannot confuse the two.
"""

from __future__ import annotations

import inspect
from typing import Any, Optional

# The optional keyword this module negotiates. One name, checked in one place, so
# adding a second extension later means adding it here and nowhere else.
_AGENT_KW = "agent"


def handler_accepts_agent(handler: Any) -> bool:
    """True when *handler* can be passed the ``agent=`` keyword.

    A builtin or C callable has no introspectable signature; treat it as the
    original one-argument protocol rather than risking a TypeError on a real
    confirmation.
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

    Returns the handler's answer unchanged. No result is invented here: this
    module routes a question to a human and returns their answer, so it must never
    substitute a default for either. A handler that raises propagates - callers
    treat an exception as a failed tool, not as an approval.

    *agent* is the asking sub-agent's label, or None for the top-level agent.
    Inspection runs per call rather than being cached: a confirmation is gated on a
    human answering, so microseconds are irrelevant, and a cache keyed on a bound
    method (a fresh object on every attribute access) would miss anyway.
    """
    if agent and handler_accepts_agent(handler):
        return handler(call, **{_AGENT_KW: agent})
    return handler(call)
