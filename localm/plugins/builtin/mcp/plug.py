# SPDX-License-Identifier: AGPL-3.0-or-later
"""MCP server plugin (registry entry).

Unlike the route-based plugins, MCP contributes a STDIO command, not HTTP routes:
external MCP clients launch ``localm mcp`` on demand (the actual server lives in
``localm.plugins.mcpserver``). There is nothing to mount on the FastAPI app, so
``register`` is a no-op - this module exists only so the engine can discover the
plugin, show it in the Plugins list, and track its enabled state.

The ``localm mcp`` command checks this plugin's enabled state at launch and
refuses to serve when it is disabled, so toggling the plugin actually gates the
server. Ships DISABLED by default; enable with ``localm plugin enable mcp``.
"""

from __future__ import annotations


def register(host) -> None:
    pass


def unregister() -> None:
    pass
