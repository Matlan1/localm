# SPDX-License-Identifier: AGPL-3.0-or-later
"""The tool families the MCP stdio server composes.

Each family module exposes ``build(...)`` returning
``{tool_name: {"description", "inputSchema", "annotations"?, "handler"}}`` for
every tool it owns, whether or not the server ends up advertising it.
``server.build_tools()`` merges the families with :func:`merge_tool_groups` and
then applies the feature gates.
"""

from __future__ import annotations

from typing import Dict, Iterable, Tuple


class ToolNameCollision(RuntimeError):
    """Two tool families define a tool of the same name."""


def merge_tool_groups(groups: Iterable[Tuple[str, Dict[str, dict]]]) -> Dict[str, dict]:
    """Merge ``(family label, tools)`` pairs into one tool table, in order.

    Raises :class:`ToolNameCollision` naming the tool and both families when a
    name appears twice; a later family never overwrites an earlier one.
    """
    merged: Dict[str, dict] = {}
    owner: Dict[str, str] = {}
    for label, tools in groups:
        for name, spec in tools.items():
            if name in merged:
                raise ToolNameCollision(
                    f"MCP tool {name!r} is defined by both the {owner[name]!r} "
                    f"and the {label!r} tool families")
            merged[name] = spec
            owner[name] = label
    return merged
