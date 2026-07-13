# SPDX-License-Identifier: AGPL-3.0-or-later
"""The ``spawn_agent`` tool: launch a focused child Agent for a sub-task.

The ``Agent`` class is imported lazily inside the call so a test that
monkeypatches ``localm.plugins.coder.agent.Agent`` is honoured, and so the
tools package has no import-time dependency on the agent module."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from .base import ToolResult
from .files import tool_read_file

def tool_spawn_agent(
    cwd: Path,
    task: str,
    name: str = "subagent",
    files: Optional[list] = None,
    model: Optional[str] = None,
    max_turns: int = 10,
    _parent_agent: Optional[Any] = None,
) -> ToolResult:
    """
    Spawn a child Agent with a focused task.

    The child inherits the parent's backend (or uses ``model`` if given),
    gets ``files`` pre-loaded into its first user message, and runs until
    it produces a final answer or ``max_turns`` is reached.

    Returns the child agent's final response as a string.
    """
    if _parent_agent is None:
        return ToolResult.error("spawn_agent requires a running parent agent")

    from ..agent import Agent

    backend = _parent_agent.backend
    if model and model != backend.model_id:
        from ..backends.http import make_localm_backend
        raw_url = getattr(backend, "_base_url", "http://127.0.0.1:8642/v1")
        try:
            port = int(raw_url.split(":")[-1].split("/")[0])
        except Exception:
            port = 8642
        try:
            backend = make_localm_backend(model, port=port)
        except Exception:
            backend = _parent_agent.backend

    preload_text = ""
    if files:
        failed: list[str] = []
        for fp in files:
            r = tool_read_file(cwd, fp)
            if not r.ok:
                failed.append(f"{fp}: {r.output}")
                continue
            preload_text += f"\n{r.output}\n"
        if failed:
            # Fail BEFORE spawning: silently feeding the read error to the
            # child as "file content" poisons its context, and the parent is
            # the one who can fix the path and retry.
            return ToolResult.error(
                "spawn_agent: could not pre-load file(s):\n  "
                + "\n  ".join(failed)
            )

    full_task = task
    if preload_text:
        full_task = f"Context files:\n{preload_text}\n\nTask:\n{task}"

    from ..audit import SessionMode as _SessionMode
    inherited_mode = getattr(_parent_agent, "mode", _SessionMode.PRIVACY)

    child = Agent(
        backend=backend,
        cwd=cwd,
        name=name,
        max_turns=max_turns,
        verbose=False,
        # A child must be no LESS confirmed than its parent: inherit the parent's
        # confirmation posture instead of hardcoding auto_approve=True, or a
        # parent that requires confirmation (auto_approve=False), is running
        # --dry-run, or has a GUI confirm_handler wired up would still spawn a
        # child that freely executes write_file/run_shell/git_push/etc. with zero
        # confirmation. confirm_handler is a synchronous callback, so passing it
        # through works even though the child runs non-interactively (run_task ->
        # _loop(interactive=False)): the child calls it in the same call stack the
        # parent's spawn_agent tool call is already on.
        auto_approve=getattr(_parent_agent, "auto_approve", True),
        dry_run=getattr(_parent_agent, "dry_run", False),
        always_confirm=getattr(_parent_agent, "always_confirm", None),
        confirm_handler=getattr(_parent_agent, "confirm_handler", None),
        parent=_parent_agent,
        mode=inherited_mode,
        # A child must be no MORE capable than its parent: inherit the restriction
        # and disabled tools so a restricted session cannot spawn a child that
        # re-enables run_shell etc. - that would be an RCE escape from a shareable
        # key. (spawn_agent is itself disabled for a restricted session, so this is
        # belt-and-suspenders.)
        restricted=getattr(_parent_agent, "restricted", False),
        disabled_tools=getattr(_parent_agent, "disabled_tools", frozenset()),
    )
    result_text = child.run_task(full_task)
    turns_used  = child.turns

    # Fold the child's changed-files + failure trace into the parent so a
    # delegation-heavy session still reflects at close (audit cluster 11): the
    # child is never close()d and shares this cwd, so without this the parent's
    # episode omits all delegated work and its failures. Best-effort - never let
    # episodic bookkeeping break the tool.
    try:
        _parent_agent._absorb_child_state(child)
    except Exception:
        pass

    # A sub-agent may have fetched untrusted web / MCP content and quoted it
    # verbatim in its summary; that text re-enters the PARENT loop as a (trusted)
    # spawn_agent result. Defang frame markers + chat-template control tokens in
    # it so the child cannot - knowingly or not - forge a role/frame boundary in
    # the parent. This is the structural-forgery half of provenance hardening; we
    # do not wrap it in the untrusted fence, so the parent can still act on a
    # legitimate delegated result (the child runs its own fence internally).
    from ..provenance import neutralise
    return ToolResult.success(
        neutralise(result_text),
        summary=f"sub-agent '{name}' finished in {turns_used} turn(s)",
    )
