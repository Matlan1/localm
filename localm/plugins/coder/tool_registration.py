# SPDX-License-Identifier: AGPL-3.0-or-later
"""Shared TOOL_REGISTRY insertion helper for the two foreign-tool adapters:
mcp.py's ``register_mcp_tools`` (subprocess JSON-RPC servers) and
plugin_tools.py's ``register_plugin_tools`` (in-process plugin-exported
functions). The two adapters need different transports and stay separate; only
the namespacing, collision handling, description neutralisation and insertion
steps are shared here.
"""

from __future__ import annotations

from typing import Callable, Optional

from .provenance import neutralise
from .tools import TOOL_REGISTRY, ToolDef


def register_foreign_tool(
    reg_name: str,
    *,
    fn,
    description: str,
    params: dict,
    destructive: bool,
    source_label: str,
    registered: list,
    warnings: list,
    reuse_if_already_ours: Optional[Callable[[ToolDef], bool]] = None,
) -> None:
    """
    Insert one foreign (MCP or plugin) tool into TOOL_REGISTRY under
    *reg_name*, appending to *registered*/*warnings* in place.

    A name clash with an unrelated entry warns and skips. When
    *reuse_if_already_ours* is given and returns True for the existing entry,
    the clash is a harmless re-registration (e.g. a sub-agent re-running the
    same discovery) and *reg_name* is silently reused instead of warned about
    - the plugin adapter's own idempotent-reregistration behaviour, not shared
    by the MCP adapter (which has no such re-init case today).

    *description* is neutralised (defangs any chat-template control token /
    frame marker a foreign name/description could carry into the system
    prompt - the model's highest-trust context) before insertion.
    """
    if reg_name in TOOL_REGISTRY:
        if reuse_if_already_ours is not None and reuse_if_already_ours(TOOL_REGISTRY[reg_name]):
            registered.append(reg_name)
        else:
            warnings.append(f"{source_label} tool name clash, skipped: {reg_name}")
        return

    TOOL_REGISTRY[reg_name] = ToolDef(
        name=reg_name,
        fn=fn,
        description=neutralise(description.strip()),
        params=params,
        destructive=destructive,
    )
    registered.append(reg_name)
