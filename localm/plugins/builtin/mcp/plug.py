# SPDX-License-Identifier: AGPL-3.0-or-later
"""MCP server plugin (registry entry).

MCP contributes a STDIO command, not HTTP routes: external MCP clients launch
``localm mcp`` on demand (the server itself lives in
``localm.plugins.mcpserver``). Nothing is mounted on the FastAPI app, so
``register`` is a no-op; the engine discovers the plugin here, shows it in the
Plugins list, and tracks its enabled state.

The ``localm mcp`` command checks this plugin's enabled state at launch and
refuses to serve when it is disabled. Ships DISABLED by default; enable with
``localm plugin enable mcp``.
"""

from __future__ import annotations


def register(host) -> None:
    pass


def unregister() -> None:
    pass
